//! Best-effort repair of the relaxed / malformed JSON small local models emit
//! for tool-call arguments, turning it back into strict JSON.
//!
//! ## Why this exists
//!
//! Some OpenAI-compatible gateways fail to detokenize a model's native
//! tool-call template cleanly, so the argument blob placed in
//! `function.arguments` is frequently *not strict JSON*:
//!
//!   - **unquoted object keys** — `{tool:"X",arguments:{guild_id:"Y"}}`
//!   - **redundant wrapping braces** — `{{tool:"X",arguments:{…}}}`, which the
//!     model piles on (`{{{…}}}`, `{{{{…}}}}`) each time the previous attempt
//!     bounced back as an error.
//!   - **leaked chat-template quote tokens** — the gateway emits the model's
//!     string-delimiter token as literal text instead of a `"`, so a value
//!     arrives as `[<|">discord<|">]` rather than `["discord"]` (observed with
//!     Kimi-family models served via GMI).
//!
//! Strict `serde_json::from_str` rejects all of these, so the call is marked
//! [`crate::ToolCall::invalid`] and fed back to the model, which
//! "repairs" it by adding *another* brace — an infinite retry that burns the
//! step budget without ever executing the tool. A zero-argument call
//! (`NAME{}`) is the only shape that survives, because `{}` is valid strict
//! JSON.
//!
//! ## What it does
//!
//! Conservative, **meaning-preserving** repairs, composed and retried at each
//! brace depth:
//!
//!   0. substitute any leaked chat-template quote token (see
//!      [`LEAKED_QUOTE_TOKENS`]) back to a literal `"`, once up front,
//!   1. peel a redundant outer brace layer that wraps exactly one object
//!      (`{{…}}` → `{…}`), and
//!   2. quote bare identifier keys in object position (`{tool:…}` →
//!      `{"tool":…}`), string- and array-aware so string contents and
//!      array/value positions are never rewritten.
//!
//! The result is accepted **only** when it parses strictly *and* is a JSON
//! object, so a scalar scraped out of noise can never masquerade as arguments.
//! This is called only *after* strict parsing has already failed on the input
//! ([`super::convert::recover_tool_arguments`]), so a well-formed argument
//! object can never reach — or be rewritten by — this path.

use serde_json::Value;

/// Maximum redundant outer brace layers to peel. Bounds work on adversarial
/// `{{{{…}}}}` blobs while comfortably covering every depth seen in the wild
/// (≤5 layers before the model gives up).
const MAX_BRACE_PEEL: usize = 16;

/// Chat-template string-delimiter tokens some gateways emit as literal text in
/// place of a `"` when they fail to detokenize a model's tool-call template
/// (seen with Kimi-family models via GMI: `[<|">discord<|">]`). Both the
/// asymmetric (`<|">`) and symmetric (`<|"|>`) renderings are covered; longer
/// forms are listed first so a substitution never leaves a partial token behind.
/// Substituted to `"`, not deleted — unlike the structural markers stripped in
/// `convert::TOOL_CALL_TEMPLATE_MARKERS`.
const LEAKED_QUOTE_TOKENS: &[&str] = &["<|\"|>", "<|\">"];

/// Attempts to recover a strict-JSON **object** from a relaxed/malformed
/// tool-call argument string, or `None` when no conservative repair yields a
/// strictly-parseable object.
///
/// See the module docs for the repair strategy and the safety invariant (only
/// invoked after strict parsing has already failed).
pub(crate) fn recover_relaxed_object(raw: &str) -> Option<Value> {
    let normalized = normalize_leaked_quote_tokens(raw);
    let mut layer = normalized.trim().to_string();
    for _ in 0..=MAX_BRACE_PEEL {
        // Try the current brace layer verbatim, then with bare keys quoted.
        if let Some(obj) = parse_object(&layer) {
            return Some(obj);
        }
        let quoted = quote_bare_keys(&layer);
        if quoted != layer
            && let Some(obj) = parse_object(&quoted)
        {
            return Some(obj);
        }

        match peel_redundant_brace(&layer) {
            Some(inner) => layer = inner,
            None => break,
        }
    }
    None
}

/// Replaces any leaked chat-template quote token (see [`LEAKED_QUOTE_TOKENS`])
/// with a literal `"`. Returns the input unchanged when no token is present, so
/// well-formed input is untouched.
fn normalize_leaked_quote_tokens(raw: &str) -> String {
    let mut out = raw.to_string();
    for &token in LEAKED_QUOTE_TOKENS {
        if out.contains(token) {
            out = out.replace(token, "\"");
        }
    }
    out
}

/// Strictly parses `s`, returning it only when it is a JSON object.
fn parse_object(s: &str) -> Option<Value> {
    match serde_json::from_str::<Value>(s) {
        Ok(value @ Value::Object(_)) => Some(value),
        _ => None,
    }
}

/// If `s` is `{ X }` where `X` is itself exactly one complete `{…}` object
/// (ignoring surrounding whitespace), returns `X` — removing one redundant
/// wrapping brace layer.
///
/// Returns `None` when the outer braces are *not* redundant, so a legitimate
/// single-object argument is never unwrapped. This is safe because a bare
/// object nested directly inside another object with no key (`{{…}}`) is never
/// valid JSON, so peeling it can only ever move toward a valid parse.
fn peel_redundant_brace(s: &str) -> Option<String> {
    let trimmed = s.trim();
    let inner = trimmed.strip_prefix('{')?.strip_suffix('}')?.trim();
    // The inner content must itself be a single complete object; otherwise the
    // outer braces are structural (real arguments), not redundant wrapping.
    if inner.starts_with('{') && object_spans_all(inner) {
        Some(inner.to_string())
    } else {
        None
    }
}

/// True when `s` begins with `{` and the brace it opens closes exactly at the
/// end of `s` (string-aware) — i.e. `s` is a single `{…}` object with no
/// trailing content. Used to decide whether an outer brace layer is redundant.
fn object_spans_all(s: &str) -> bool {
    if !s.starts_with('{') {
        return false;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                // Guard against an unbalanced stray `}` underflowing.
                depth = match depth.checked_sub(1) {
                    Some(d) => d,
                    None => return false,
                };
                if depth == 0 {
                    // Matched the opening brace: redundant only if it is the last char.
                    return idx + ch.len_utf8() == s.len();
                }
            }
            _ => {}
        }
    }
    false
}

/// Whether `s` is inside a JSON object or array — governs when a `,` introduces
/// a new key (object) versus a new element (array).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

/// Quotes bare identifier keys that appear in object-key position, e.g.
/// `{tool:1,a:{b:2}}` → `{"tool":1,"a":{"b":2}}`.
///
/// String-literal and array aware: content inside `"…"` is never touched, and
/// identifiers in array or value position are left alone (so `["discord"]`,
/// `true`, numbers, and already-quoted keys pass through unchanged). Returns the
/// input verbatim when there is nothing to quote.
/// Reads a quote-delimited object key whose delimiters may be single quotes or
/// mismatched, returning the key text and the bytes consumed (including both
/// delimiters).
///
/// `rest` begins at the opening quote. Models that lose track of their own
/// string delimiters produce `'city'`, `"city'`, and `'city"` interchangeably —
/// all three mean the same key, and strict JSON accepts none of them.
///
/// Returns `None` for a well-formed `"key"` so the caller keeps using the
/// normal in-string path, and `None` for anything that does not look like a
/// key: the token must be terminated by `'` or `"` followed (after optional
/// whitespace) by a `:`, and must not span a line break or contain structural
/// JSON characters. That keeps a legitimate double-quoted key containing an
/// apostrophe (`{"it's fine": 1}`) from being truncated at the apostrophe,
/// because there the next character after `'` is not a colon.
fn take_quoted_key(rest: &str) -> Option<(String, usize)> {
    let mut chars = rest.char_indices();
    let (_, open) = chars.next()?;
    debug_assert!(open == '"' || open == '\'');

    let mut key = String::new();
    for (idx, ch) in chars {
        match ch {
            '"' | '\'' => {
                let after = &rest[idx + ch.len_utf8()..];
                if after.trim_start().starts_with(':') {
                    // A perfectly well-formed key needs no rewriting; let the
                    // ordinary scanner handle it so behaviour is unchanged.
                    if open == '"' && ch == '"' {
                        return None;
                    }
                    return Some((key, idx + ch.len_utf8()));
                }
                // Not the end of a key — record it and keep looking.
                key.push(ch);
            }
            // A key never spans a newline or contains structure; bail out and
            // let the ordinary scanner deal with whatever this really is.
            '\n' | '\r' | '{' | '}' | '[' | ']' | ':' => return None,
            _ => key.push(ch),
        }
    }
    None
}

fn quote_bare_keys(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut stack: Vec<Container> = Vec::new();
    let mut expect_key = false;
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = s.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            // A quote in key position may open a *mismatched* key delimiter
            // (`"city'`) or a single-quoted one (`'city'`), neither of which the
            // in-string scanner below can terminate correctly. Try that first;
            // a well-formed `"key"` falls through to the normal path.
            '"' | '\'' if expect_key && matches!(stack.last(), Some(Container::Object)) => {
                match take_quoted_key(&s[idx..]) {
                    Some((key, consumed)) => {
                        out.push('"');
                        out.push_str(&key.replace('\\', r"\\").replace('"', "\\\""));
                        out.push('"');
                        // Advance the iterator past the bytes just consumed.
                        while chars.peek().is_some_and(|&(next, _)| next < idx + consumed) {
                            chars.next();
                        }
                        expect_key = false;
                    }
                    None => {
                        in_string = true;
                        expect_key = false;
                        out.push(ch);
                    }
                }
            }
            '"' => {
                in_string = true;
                expect_key = false;
                out.push(ch);
            }
            '{' => {
                stack.push(Container::Object);
                expect_key = true;
                out.push(ch);
            }
            '}' => {
                stack.pop();
                expect_key = false;
                out.push(ch);
            }
            '[' => {
                stack.push(Container::Array);
                expect_key = false;
                out.push(ch);
            }
            ']' => {
                stack.pop();
                expect_key = false;
                out.push(ch);
            }
            ',' => {
                // A comma re-opens key position only inside an object.
                expect_key = matches!(stack.last(), Some(Container::Object));
                out.push(ch);
            }
            ':' => {
                expect_key = false;
                out.push(ch);
            }
            c if c.is_whitespace() => out.push(ch),
            c if expect_key
                && matches!(stack.last(), Some(Container::Object))
                && (c.is_ascii_alphabetic() || c == '_') =>
            {
                // Bare identifier key: consume it and wrap it in quotes.
                let start = idx;
                let mut end = idx + c.len_utf8();
                while let Some(&(next_idx, next_ch)) = chars.peek() {
                    if next_ch.is_ascii_alphanumeric()
                        || next_ch == '_'
                        || next_ch == '-'
                        || next_ch == '.'
                    {
                        end = next_idx + next_ch.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push('"');
                out.push_str(&s[start..end]);
                out.push('"');
                expect_key = false;
            }
            _ => {
                expect_key = false;
                out.push(ch);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn repairs_single_quoted_and_mismatched_keys() {
        // Captured from `llama3.2:3b` via Ollama: the model loses track of its
        // own string delimiters mid-object.
        assert_eq!(
            recover_relaxed_object(r#"{"name":"get_weather","parameters':{'city':"Paris"}}"#),
            Some(json!({ "name": "get_weather", "parameters": { "city": "Paris" } }))
        );
        // Single-quoted keys are repaired the same way, as long as the values
        // themselves are well-formed.
        assert_eq!(
            recover_relaxed_object(r#"{'city':"Paris"}"#),
            Some(json!({ "city": "Paris" }))
        );
    }

    /// Single-quoted *values* are deliberately **not** repaired.
    ///
    /// A key is a short identifier, so reading `'` as a delimiter there is
    /// safe. A value is free text where an apostrophe is ordinary English
    /// (`"it's sunny"`), and treating those as delimiters would corrupt real
    /// arguments. Such a blob stays unrecovered, the call is marked invalid,
    /// and the agent loop hands the model a precise error to retry against —
    /// the same path every other unrepairable blob takes.
    #[test]
    fn single_quoted_values_are_left_unrepaired() {
        assert_eq!(recover_relaxed_object(r#"{'city':'Paris'}"#), None);
    }

    #[test]
    fn an_apostrophe_inside_a_well_formed_key_is_not_a_delimiter() {
        // `'` here is followed by ` fine"`, not a colon, so the key survives
        // whole rather than being truncated at the apostrophe.
        assert_eq!(
            recover_relaxed_object(r#"{"it's fine":1,bare:2}"#),
            Some(json!({ "it's fine": 1, "bare": 2 }))
        );
    }

    #[test]
    fn quotes_unquoted_keys() {
        assert_eq!(
            recover_relaxed_object(r#"{toolkits:["discord"]}"#),
            Some(json!({ "toolkits": ["discord"] }))
        );
    }

    #[test]
    fn quotes_multiple_unquoted_keys_and_bool_value() {
        assert_eq!(
            recover_relaxed_object(r#"{include_unconnected:true,toolkits:["discord"]}"#),
            Some(json!({ "include_unconnected": true, "toolkits": ["discord"] }))
        );
    }

    #[test]
    fn substitutes_leaked_quote_tokens_in_values() {
        assert_eq!(
            recover_relaxed_object(r#"{toolkits:[<|">discord<|">]}"#),
            Some(json!({ "toolkits": ["discord"] }))
        );
    }

    #[test]
    fn substitutes_symmetric_leaked_quote_token_variant() {
        assert_eq!(
            recover_relaxed_object(r#"{toolkits:[<|"|>discord<|"|>]}"#),
            Some(json!({ "toolkits": ["discord"] }))
        );
    }

    #[test]
    fn peels_one_redundant_brace_layer() {
        assert_eq!(
            recover_relaxed_object(r#"{{"tool":"X","arguments":{"guild_id":"1"}}}"#),
            Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
        );
    }

    #[test]
    fn peels_and_quotes_together() {
        assert_eq!(
            recover_relaxed_object(
                r#"{{tool:"DISCORD_LIST_CHANNELS",arguments:{"guild_id":"1470856511193616498"}}}"#
            ),
            Some(json!({
                "tool": "DISCORD_LIST_CHANNELS",
                "arguments": { "guild_id": "1470856511193616498" }
            }))
        );
    }

    #[test]
    fn recovers_full_composio_execute_with_leaked_quote_tokens() {
        assert_eq!(
            recover_relaxed_object(
                r#"{arguments:{guild_id:<|">1470856511193616498<|">},tool:<|">DISCORD_GET_GUILD_CHANNELS<|">}"#
            ),
            Some(json!({
                "arguments": { "guild_id": "1470856511193616498" },
                "tool": "DISCORD_GET_GUILD_CHANNELS"
            }))
        );
    }

    #[test]
    fn peels_several_redundant_layers() {
        assert_eq!(
            recover_relaxed_object(r#"{{{{tool:"X",arguments:{"guild_id":"1"}}}}}"#),
            Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
        );
    }

    #[test]
    fn handles_reordered_relaxed_keys() {
        assert_eq!(
            recover_relaxed_object(r#"{{arguments:{guild_id:"1"},tool:"X"}}"#),
            Some(json!({ "arguments": { "guild_id": "1" }, "tool": "X" }))
        );
    }

    #[test]
    fn preserves_brace_inside_string_value() {
        assert_eq!(
            recover_relaxed_object(r#"{{note:"see {ref:1}"}}"#),
            Some(json!({ "note": "see {ref:1}" }))
        );
    }

    #[test]
    fn does_not_quote_array_elements() {
        assert_eq!(recover_relaxed_object(r#"{tags:[hi,bye]}"#), None);
    }

    #[test]
    fn rejects_keyless_nested_object() {
        assert_eq!(recover_relaxed_object(r#"{tool:"X",{guild_id:"Y"}}"#), None);
    }

    #[test]
    fn rejects_non_object_scalar() {
        assert_eq!(recover_relaxed_object("42"), None);
        assert_eq!(recover_relaxed_object(r#""just a string""#), None);
        assert_eq!(recover_relaxed_object("[1,2,3]"), None);
    }

    #[test]
    fn rejects_unrecoverable_garbage() {
        assert_eq!(recover_relaxed_object(r#"{"a":1]"#), None);
        assert_eq!(recover_relaxed_object("not json at all"), None);
    }

    #[test]
    fn already_valid_object_passes_through() {
        assert_eq!(
            recover_relaxed_object(r#"{"a":1,"b":{"c":2}}"#),
            Some(json!({ "a": 1, "b": { "c": 2 } }))
        );
    }

    #[test]
    fn does_not_unwrap_legitimate_single_object() {
        assert_eq!(
            recover_relaxed_object(r#"{guild_id:"1",limit:50}"#),
            Some(json!({ "guild_id": "1", "limit": 50 }))
        );
    }

    #[test]
    fn quote_bare_keys_leaves_quoted_keys_untouched() {
        assert_eq!(quote_bare_keys(r#"{"a":1,"b":2}"#), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn normalize_leaked_quote_tokens_is_noop_without_tokens() {
        assert_eq!(normalize_leaked_quote_tokens(r#"{"a":1}"#), r#"{"a":1}"#);
    }

    #[test]
    fn object_spans_all_respects_strings_and_trailing() {
        assert!(object_spans_all(r#"{"a":"}"}"#));
        assert!(!object_spans_all(r#"{"a":1},{"b":2}"#));
        assert!(!object_spans_all(r#"{"a":1}trailing"#));
    }
}
