//! REPL language (`.ragsh`) — capability-bound interactive orchestration; the
//! RLM/CodeAct surface of the runtime.
//!
//! `.ragsh` is TinyAgents' answer to the Recursive Language Model execution
//! model: instead of stuffing everything into one context window, an
//! orchestrator (a human, or a model acting as one) drives a session by issuing
//! small typed commands — set/get session *values*, load and compile a `.rag`
//! blueprint, run a graph, or `call` a registered capability — inspecting each
//! [`ReplOutcome`] and iterating. Because every capability-bearing command is
//! checked against a [`CapabilityPolicy`] allowlist before it can touch the
//! runtime, the same surface is safe to expose to a model that is recursively
//! orchestrating sub-models, sub-agents, and sub-graphs from inside a run.
//!
//! The REPL language is the interactive, session-oriented counterpart to the
//! declarative `.rag` expressive language.  An operator (human or parent
//! orchestrator) drives a harness/graph session by issuing typed commands that
//! are policy-checked before they reach the runtime.
//!
//! This module is currently at **milestone R1** (Documentation and Types).  It
//! establishes the command grammar, a line-oriented parser, a session/capability
//! boundary, and structured outcomes.  Commands that need live harness/graph
//! integration are policy-checked and returned as [`ReplOutcome::Planned`]
//! rather than executed; wiring to the live harness/graph runtime is deferred to
//! milestones R2–R6.
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

/// Rhai-backed `.ragsh` session runtime (the imperative RLM/CodeAct surface).
///
/// This is the evolution of the line-oriented command REPL above into a full
/// scripting session with a persistent namespace, policy-bounded capability
/// calls, and typed cell results — the [`session::ReplSession`] described in the
/// module design document. It is gated behind the `repl` cargo feature so the
/// default build stays free of the embedded Rhai engine.
///
/// The command-driven [`ReplSession`](crate::repl::ReplSession) above remains
/// available for the line-oriented REPL; the scripting engine is exposed as
/// [`session::ReplSession`] (and re-exported at the crate root as
/// [`crate::ReplSession`] when the feature is enabled) to keep both surfaces
/// compiling side by side.
#[cfg(feature = "repl")]
pub mod session;

// Re-export the non-colliding session types at the `repl` root for convenience.
// `session::ReplSession` is intentionally *not* re-exported here because the
// line-oriented `ReplSession` (above) already occupies that name in the default
// build; reach the scripting session via `repl::session::ReplSession` or the
// crate-root `crate::ReplSession` re-export.
#[cfg(feature = "repl")]
pub use session::{
    LanguageCompiler, ReplCallKind, ReplCallRecord, ReplCapabilities, ReplPolicy, ReplResult,
    ReplValue, ReplVariables,
};

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
        return Err(parse_err_at(trimmed, trimmed, "empty input"));
    }

    let (verb, rest) = split_token(trimmed, trimmed)?;

    match verb.to_lowercase().as_str() {
        "help" | "?" => Ok(ReplCommand::Help),

        "quit" | "exit" | "q" => Ok(ReplCommand::Quit),

        "load" => {
            let (path, _) = require_token(trimmed, rest, "load <path>")?;
            Ok(ReplCommand::Load { path })
        }

        "compile" => {
            let (name, _) = require_token(trimmed, rest, "compile <name>")?;
            Ok(ReplCommand::Compile { name })
        }

        "run" => {
            let (graph, rest) = require_token(trimmed, rest, "run <graph> <input>")?;
            let (input, _) = require_token(trimmed, rest, "run <graph> <input>")?;
            Ok(ReplCommand::Run { graph, input })
        }

        "set" => {
            let (key, rest) = require_token(trimmed, rest, "set <key> <value>")?;
            let (value, _) = require_token(trimmed, rest, "set <key> <value>")?;
            Ok(ReplCommand::Set { key, value })
        }

        "get" => {
            let (key, _) = require_token(trimmed, rest, "get <key>")?;
            Ok(ReplCommand::Get { key })
        }

        "show" => {
            let (what, _) = require_token(trimmed, rest, "show <vars|graphs|status>")?;
            Ok(ReplCommand::Show { what })
        }

        "call" => {
            let (capability, json_rest) = require_token(trimmed, rest, "call <capability> <json>")?;
            let json_str = json_rest.trim();
            if json_str.is_empty() {
                return Err(parse_err_at(
                    trimmed,
                    json_rest,
                    "call requires a JSON argument: call <capability> <json>",
                ));
            }
            let args: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
                parse_err_at(
                    trimmed,
                    json_str,
                    &format!("invalid JSON argument for `call`: {e}"),
                )
            })?;
            Ok(ReplCommand::Call { capability, args })
        }

        other => Err(parse_err_at(
            trimmed,
            trimmed,
            &format!("unknown command `{other}`"),
        )),
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

/// Builds a [`crate::error::TinyAgentsError::Parse`] pointing at `at` — a
/// substring slice of `origin` — reporting a real 1-based line/column instead
/// of the placeholder `(0, 0)`.
///
/// `parse_command` always parses a single command line, so the line is always
/// `1`; the column is the 1-based character offset of `at` within `origin`.
/// Falls back to the end of `origin` if `at` is not actually a subslice of it
/// (defensive; should not happen given how callers use this).
fn parse_err_at(origin: &str, at: &str, message: &str) -> crate::error::TinyAgentsError {
    let column = char_column(origin, at);
    parse_err(message, 1, column)
}

/// Computes the 1-based character column of `at` within `origin`, where `at`
/// is a substring slice of `origin` obtained by slicing (not copying).
fn char_column(origin: &str, at: &str) -> usize {
    let origin_start = origin.as_ptr() as usize;
    let origin_end = origin_start + origin.len();
    let at_start = at.as_ptr() as usize;
    let offset = if at_start >= origin_start && at_start <= origin_end {
        at_start - origin_start
    } else {
        // `at` isn't a subslice of `origin` (shouldn't happen); point at the end.
        origin.len()
    };
    origin[..offset].chars().count() + 1
}

/// Split the next token from `s`, returning `(token, remainder)`.
///
/// Handles quoted strings (`"..."`) with `\\`, `\"`, `\n`, `\t` escapes.
/// Bare tokens end at the first whitespace character.
///
/// Returns `None` if `s` (after trimming leading whitespace) is empty.
///
/// `origin` is the full trimmed command line `s` was sliced from; it is used
/// only to compute a real 1-based column for any parse error, via
/// [`parse_err_at`].
fn split_token<'a>(origin: &str, s: &'a str) -> crate::error::Result<(String, &'a str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return Err(parse_err_at(origin, s, "unexpected end of input"));
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
                        return Err(parse_err_at(origin, s, "unterminated quoted string"));
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
                            return Err(parse_err_at(origin, s, "unterminated escape sequence"));
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
fn require_token<'a>(
    origin: &str,
    s: &'a str,
    usage: &str,
) -> crate::error::Result<(String, &'a str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return Err(parse_err_at(
            origin,
            s,
            &format!("missing argument — usage: {usage}"),
        ));
    }
    split_token(origin, s)
}
