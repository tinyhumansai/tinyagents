//! REPL language (`.ragsh`) — capability-bound interactive orchestration.
//!
//! The REPL language is the interactive, session-oriented counterpart to the
//! declarative `.rag` expressive language.  An operator (human or parent
//! orchestrator) drives a harness/graph session by issuing typed commands that
//! are policy-checked before they reach the runtime.
//!
//! This module is currently at **milestone R1** (Documentation and Types).  It
//! establishes the command grammar, a line-oriented parser, a session/capability
//! boundary, and structured outcomes.  Wiring to the live harness/graph runtime
//! is deferred to milestones R2–R6.
//!
//! # Grammar
//!
//! ```text
//! line      = verb ( ws+ arg )* ws*
//! verb      = [a-zA-Z][a-zA-Z0-9_-]*
//! arg       = quoted | bare
//! quoted    = '"' ( <any> | '\\' <any> )* '"'
//! bare      = ( <non-whitespace> )+
//! ```
//!
//! The first token is the command verb (matched case-insensitively).  Subsequent
//! tokens are positional arguments.  For the `call` verb the *remainder* of the
//! line after the capability name is parsed as a single JSON value, so
//! multi-token JSON objects and arrays are accepted verbatim.
//!
//! ## Verb table
//!
//! | Verb       | Signature                        | Notes                          |
//! |------------|----------------------------------|--------------------------------|
//! | `help`     | `help`                           | Also: `?`                      |
//! | `quit`     | `quit`                           | Also: `exit`, `q`              |
//! | `load`     | `load <path>`                    | Requires `"load"` capability   |
//! | `compile`  | `compile <name>`                 | Requires `"compile"` capability|
//! | `run`      | `run <graph> <input>`            | Requires `"run"` capability    |
//! | `set`      | `set <key> <value>`              | `<value>` may be quoted        |
//! | `get`      | `get <key>`                      |                                |
//! | `show`     | `show vars\|graphs\|status`      |                                |
//! | `call`     | `call <capability> <json>`       | JSON may span multiple tokens  |

pub mod types;

pub use types::*;

#[cfg(test)]
mod test;

// ── Public parser ─────────────────────────────────────────────────────────────

/// Parse a single `.ragsh` REPL command line into a [`ReplCommand`].
///
/// Leading and trailing whitespace is ignored.  The first token is matched
/// case-insensitively against the verb table.  For `call`, the remainder of
/// the line after the capability name is parsed as a JSON value.
///
/// # Errors
///
/// Returns [`crate::error::TinyAgentsError::Parse`] for:
///
/// * empty input
/// * unknown verb
/// * missing required argument(s)
/// * unterminated quoted string
/// * invalid JSON argument to `call`
pub fn parse_command(line: &str) -> crate::error::Result<ReplCommand> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(parse_err("empty input", 0, 0));
    }

    let (verb, rest) = split_token(trimmed)?;

    match verb.to_lowercase().as_str() {
        "help" | "?" => Ok(ReplCommand::Help),

        "quit" | "exit" | "q" => Ok(ReplCommand::Quit),

        "load" => {
            let (path, _) = require_token(rest, "load <path>")?;
            Ok(ReplCommand::Load { path })
        }

        "compile" => {
            let (name, _) = require_token(rest, "compile <name>")?;
            Ok(ReplCommand::Compile { name })
        }

        "run" => {
            let (graph, rest) = require_token(rest, "run <graph> <input>")?;
            let (input, _) = require_token(rest, "run <graph> <input>")?;
            Ok(ReplCommand::Run { graph, input })
        }

        "set" => {
            let (key, rest) = require_token(rest, "set <key> <value>")?;
            let (value, _) = require_token(rest, "set <key> <value>")?;
            Ok(ReplCommand::Set { key, value })
        }

        "get" => {
            let (key, _) = require_token(rest, "get <key>")?;
            Ok(ReplCommand::Get { key })
        }

        "show" => {
            let (what, _) = require_token(rest, "show <vars|graphs|status>")?;
            Ok(ReplCommand::Show { what })
        }

        "call" => {
            let (capability, json_rest) = require_token(rest, "call <capability> <json>")?;
            let json_str = json_rest.trim();
            if json_str.is_empty() {
                return Err(parse_err(
                    "call requires a JSON argument: call <capability> <json>",
                    0,
                    0,
                ));
            }
            let args: serde_json::Value = serde_json::from_str(json_str)
                .map_err(|e| parse_err(&format!("invalid JSON argument for `call`: {e}"), 0, 0))?;
            Ok(ReplCommand::Call { capability, args })
        }

        other => Err(parse_err(&format!("unknown command `{other}`"), 0, 0)),
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Build a [`crate::error::TinyAgentsError::Parse`] with the given message and
/// optional source position.
fn parse_err(message: &str, line: usize, column: usize) -> crate::error::TinyAgentsError {
    crate::error::TinyAgentsError::Parse {
        message: message.to_string(),
        line,
        column,
    }
}

/// Split the next token from `s`, returning `(token, remainder)`.
///
/// Handles quoted strings (`"..."`) with `\\`, `\"`, `\n`, `\t` escapes.
/// Bare tokens end at the first whitespace character.
///
/// Returns `None` if `s` (after trimming leading whitespace) is empty.
fn split_token(s: &str) -> crate::error::Result<(String, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return Err(parse_err("unexpected end of input", 0, 0));
    }

    if s.starts_with('"') {
        // Quoted string: scan from after the opening quote.
        // The labeled block returns the byte offset of the closing `"` within
        // `inner` so we can compute the remainder slice without a mutable
        // Option accumulator (which would trigger an unused-assignment warning).
        let inner = s
            .strip_prefix('"')
            .expect("already checked starts_with('\"')");
        let mut token = String::new();

        let inner_offset: usize = 'scan: {
            let mut chars = inner.char_indices();
            loop {
                match chars.next() {
                    None => {
                        return Err(parse_err("unterminated quoted string", 0, 0));
                    }
                    Some((i, '"')) => break 'scan i,
                    Some((_, '\\')) => match chars.next() {
                        Some((_, '"')) => token.push('"'),
                        Some((_, '\\')) => token.push('\\'),
                        Some((_, 'n')) => token.push('\n'),
                        Some((_, 't')) => token.push('\t'),
                        Some((_, c)) => {
                            token.push('\\');
                            token.push(c);
                        }
                        None => {
                            return Err(parse_err("unterminated escape sequence", 0, 0));
                        }
                    },
                    Some((_, c)) => token.push(c),
                }
            }
        };

        // Skip the opening `"` (1 byte) + bytes up to closing `"` + the closing `"` itself.
        let remainder = &s[1 + inner_offset + 1..];
        Ok((token, remainder))
    } else {
        // Bare word: ends at the first whitespace.
        let end = s
            .char_indices()
            .find(|(_, c)| c.is_whitespace())
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let token = s[..end].to_string();
        let remainder = &s[end..];
        Ok((token, remainder))
    }
}

/// Like [`split_token`] but returns a [`crate::error::TinyAgentsError::Parse`]
/// mentioning the expected usage if the remaining input is empty.
fn require_token<'a>(s: &'a str, usage: &str) -> crate::error::Result<(String, &'a str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return Err(parse_err(
            &format!("missing argument — usage: {usage}"),
            0,
            0,
        ));
    }
    split_token(s)
}
