//! P-Format ("Parameter-Format") tool calls — compact, positional,
//! pipe-delimited tool invocations designed to slash the token cost of
//! text-based tool calling.
//!
//! # Why
//!
//! Standard JSON tool calls are heavy on tokens for what's actually a
//! simple instruction:
//!
//! ```text
//! {"name": "get_weather", "arguments": {"location": "London", "unit": "metric"}}
//! ```
//!
//! That's roughly 25 tokens. The same call in P-Format:
//!
//! ```text
//! get_weather[London|metric]
//! ```
//!
//! is ~5 tokens — an 80% reduction. Across a long agent loop with many
//! tool calls per turn, that compounds dramatically.
//!
//! # Spec
//!
//! - One call per `<tool_call>...</tool_call>` tag body.
//! - Form: `name[index|value|index|value|...]` — each argument carries the
//!   slot index it belongs to, so **only the arguments actually being sent
//!   appear**.
//! - `name` is the tool's registered name (alphanumerics + `_`).
//! - Slot indices number the parameters **required first** (in the order the
//!   schema declares them), then the optional ones alphabetically. Both halves
//!   are deterministic across rebuilds and workspaces: a JSON array preserves
//!   order, and `Map` iterates as a `BTreeMap` because this build does not
//!   enable `preserve_order`.
//! - The renderer exposes the numbering in the tool catalogue, each slot marked
//!   as a placeholder to fill:
//!   `get_weather[0|<location>|1|<unit>]`, `math[0|<verbose>|1|<x>|2|<y>]`.
//!   The brackets matter: rendered as bare names the signature reads as a call
//!   to copy, and a live model duly sent the parameter names as the argument
//!   values.
//! - Empty calls: `tool_name[]` for zero-arg tools, and for a call that sends
//!   no arguments at all.
//!
//!   ## Why indices, rather than counting empty slots
//!
//!   The form used to be bare positional — `name[arg1|arg2|...]` — with skipped
//!   arguments written as empty slots (`name[||value]`). That made the *count
//!   of leading delimiters* load-bearing, and it is the single thing models get
//!   wrong most. Two failures observed on a live host:
//!
//!   - `GMAIL_LIST_THREADS[||50|<query>]` failed schema validation **12 times in
//!     one turn** before the turn was cut short.
//!   - A `GMAIL_LIST_THREADS` call wrote four leading empties where three were
//!     needed, so `query` and `user_id` each landed one slot late, in `user_id`
//!     and `verbose`. The call **ran**, with the search text as the account id.
//!
//!   Both are off-by-one on a delimiter, and both bound arguments to the wrong
//!   parameter **silently**. Indices remove the counting: a sparse call names
//!   its slots, and there is nothing to miscount. An index that is missing,
//!   non-numeric, or out of range is **rejected** rather than guessed at, so the
//!   failure mode moves from a wrong call that succeeds to a malformed call the
//!   model is told about.
//!
//!   Required-first ordering is why the natural minimal call — the one required
//!   value — is `name[0|value]` rather than an arbitrary index. An alphabetical
//!   layout put the optional parameters first for most tools, and a live model
//!   wrote `memory_recall[Colorado]` six times in one turn against
//!   `[limit|namespace|query]` and never got a tool to run.
//! - Escapes: `\|` → `|`, `\]` → `]`, `\\` → `\`. Other backslashes
//!   pass through verbatim so URLs and Windows paths remain readable.
//! - Type coercion: schema property `type: integer | number | boolean`
//!   triggers parsing the string into the matching JSON value. Failed
//!   coercion falls back to a string so the model still gets *something*
//!   useful into the tool argument.
//!
//! # Trade-offs
//!
//! - **Positional only** — nested objects or arrays can't be expressed
//!   directly. Tools that need rich payloads should either flatten their
//!   schema, accept a JSON-blob string parameter, or be invoked via the
//!   legacy JSON-in-tag fallback (which the dispatcher attempts when
//!   p-format parsing returns `None`).
//! - **Tool registry required at parse time** — without the schema we
//!   can't reconstruct named arguments. The dispatcher caches a
//!   pre-computed `name → params` map at construction time so this
//!   stays fast and avoids holding a reference to the live tool slice.

use serde_json::{Map, Value};
use std::collections::HashMap;

/// JSON-schema primitive type used for argument coercion. Anything we
/// don't recognise (objects, arrays, custom types) is treated as
/// `Other`, which preserves the raw string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PFormatParamType {
    String,
    Integer,
    Number,
    Boolean,
    Other,
}

impl PFormatParamType {
    /// Map a JSON-schema `type` value to the coercion enum. Schemas may
    /// expose `type` as either a single string (`"integer"`) or an
    /// array (`["integer", "null"]`); we accept both and pick the first
    /// non-`null` entry.
    pub fn from_schema_type(value: Option<&Value>) -> Self {
        let label = match value {
            Some(Value::String(s)) => s.as_str(),
            Some(Value::Array(items)) => items
                .iter()
                .find_map(|v| v.as_str().filter(|s| *s != "null"))
                .unwrap_or(""),
            _ => "",
        };
        match label {
            "string" => Self::String,
            "integer" => Self::Integer,
            "number" => Self::Number,
            "boolean" => Self::Boolean,
            _ => Self::Other,
        }
    }
}

/// One tool's positional parameter list, as the dispatcher needs it
/// at parse time.
#[derive(Debug, Clone)]
pub struct PFormatToolParams {
    /// Parameter names in declaration order.
    pub names: Vec<String>,
    /// Parallel slice of JSON types for coercion.
    pub types: Vec<PFormatParamType>,
}

impl PFormatToolParams {
    /// Pull the ordered parameter names + types out of a tool's
    /// JSON schema. Non-object schemas (rare, but possible for
    /// shell-style tools) return an empty list — the renderer falls
    /// back to `name[]`.
    ///
    /// Order is required-first, then optional alphabetically. The renderer
    /// always shows the resulting order in the tool catalogue so the model — and
    /// the parser — agree on the layout; both read it from here, so they cannot
    /// disagree. See the module-level docs for why required comes first.
    pub fn from_schema(schema: &Value) -> Self {
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
            return Self {
                names: Vec::new(),
                types: Vec::new(),
            };
        };
        // Required parameters first, in the order the schema declares them, then
        // the optional ones alphabetically. Both halves are deterministic (a JSON
        // array preserves order; `Map` is a `BTreeMap` in this build), which is the
        // property the layout actually needs.
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        let mut ordered: Vec<&String> = Vec::with_capacity(props.len());
        for name in &required {
            if let Some((key, _)) = props.get_key_value(*name)
                && !ordered.contains(&key)
            {
                ordered.push(key);
            }
        }
        for key in props.keys() {
            if !ordered.contains(&key) {
                ordered.push(key);
            }
        }

        let mut names = Vec::with_capacity(ordered.len());
        let mut types = Vec::with_capacity(ordered.len());
        for key in ordered {
            names.push(key.clone());
            types.push(PFormatParamType::from_schema_type(props[key].get("type")));
        }
        Self { names, types }
    }
}

/// Pre-computed lookup of every tool's parameter list. Built once at
/// dispatcher construction time so the parser doesn't need to hold a
/// reference to the live tool list (which the host owns).
///
/// The map preserves the spec contract: the parser refuses to invent
/// argument names for an unknown tool, so an LLM can't tunnel
/// arbitrary JSON in by guessing tool names that don't exist.
pub type PFormatRegistry = HashMap<String, PFormatToolParams>;

/// Build a [`PFormatRegistry`] from `(name, schema)` pairs.
///
/// Takes schemas rather than a tool trait object on purpose: a host's tool
/// type is its own vocabulary, and requiring it here would make this module
/// depend on the very thing it exists to stay independent of. Hosts keep a
/// one-line adapter over their own tool slice.
///
/// The schema is `Borrow<Value>` rather than `&Value` so a host whose tool
/// trait *returns* a schema by value — the common shape — can map straight
/// into this without collecting into a temporary first.
pub fn build_registry<I, N, S>(tools: I) -> PFormatRegistry
where
    I: IntoIterator<Item = (N, S)>,
    N: Into<String>,
    S: std::borrow::Borrow<Value>,
{
    tools
        .into_iter()
        .map(|(name, schema)| (name.into(), PFormatToolParams::from_schema(schema.borrow())))
        .collect()
}

/// Render a single tool's p-format signature, e.g. `get_weather[0|<location>|1|<unit>]`.
///
/// This signature is included in the tool catalogue within the system prompt
/// to tell the LLM exactly how to order positional arguments for a tool.
pub fn render_signature(name: &str, params: &PFormatToolParams) -> String {
    if params.names.is_empty() {
        format!("{name}[]")
    } else {
        // Each slot carries its index, and the name is wrapped in angle brackets
        // so it reads as a placeholder to fill rather than a call to copy. Bare
        // names do get copied: a live model answered `memory_recall[limit|
        // namespace|query]` — the signature verbatim, the parameter names sent as
        // the argument *values*. Backticking the whole signature does not help
        // either; that made it copy the backticks instead. `<…>` marks the slot
        // without decorating the form.
        let slots: Vec<String> = params
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{i}|<{n}>"))
            .collect();
        format!("{name}[{}]", slots.join("|"))
    }
}

/// Render a signature straight from a tool's JSON schema.
///
/// The schema-taking counterpart to [`render_signature`], for callers that
/// have a schema but no prebuilt [`PFormatToolParams`].
pub fn render_signature_from_schema(name: &str, schema: &Value) -> String {
    render_signature(name, &PFormatToolParams::from_schema(schema))
}

/// Parse a single p-format call body and reconstruct named JSON arguments.
///
/// This function:
/// 1. Locates the positional arguments within the `[...]` brackets.
/// 2. Splits them by the `|` delimiter (respecting escapes).
/// 3. Maps each positional value to its parameter name from the tool registry.
/// 4. Performs type coercion (e.g., string to integer) based on the tool's schema.
///
/// Returns `(tool_name, args_json)` on success, or `None` if the format is invalid
/// or the tool is unknown.
pub fn parse_call(body: &str, registry: &PFormatRegistry) -> Option<(String, Value)> {
    let trimmed = body.trim();

    // Locate the opening bracket. The closing bracket must be the
    // **last** character of the trimmed body — anything trailing it
    // (e.g. extra whitespace, JSON, prose) means this isn't a valid
    // p-format call and we leave it for the JSON fallback.
    let open = trimmed.find('[')?;
    if !trimmed.ends_with(']') {
        return None;
    }

    let name = trimmed[..open].trim();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    let inner = &trimmed[open + 1..trimmed.len() - 1];

    // Look up the parameter spec — required so we can map positional
    // values back to named JSON keys with the correct types.
    let params = registry.get(name)?;

    let tokens = split_pipes(inner);
    // Index/value pairs, so an odd token count means the model dropped or added
    // a delimiter. Reject rather than guess: the whole point of the indices is
    // that a miscounted delimiter can no longer bind a value to the wrong
    // parameter, and silently keeping the pairs that happen to line up would put
    // that failure right back.
    if !tokens.len().is_multiple_of(2) {
        tinyagents_tracing::debug!(
            tool = name,
            tokens = tokens.len(),
            "[pformat] odd token count — not index/value pairs, refusing to parse"
        );
        return None;
    }

    let mut args = Map::with_capacity(tokens.len() / 2);
    for pair in tokens.chunks_exact(2) {
        let (raw_index, raw) = (pair[0].trim(), &pair[1]);
        let Ok(slot) = raw_index.parse::<usize>() else {
            // A non-numeric index is a call in the old bare-positional form (or
            // simply malformed). Refusing is deliberate: parsing it positionally
            // would silently resurrect the off-by-one this format exists to end.
            tinyagents_tracing::debug!(
                tool = name,
                index = raw_index,
                "[pformat] slot index is not a number — refusing to parse"
            );
            return None;
        };
        let Some(param_name) = params.names.get(slot) else {
            tinyagents_tracing::debug!(
                tool = name,
                slot,
                slots = params.names.len(),
                "[pformat] slot index out of range — refusing to parse"
            );
            return None;
        };
        // An empty value is an argument the model did not send, so the key is
        // left out entirely rather than set to `""`. Inserting `""` makes every
        // non-string parameter fail schema validation — a typed `max_results`
        // arriving as `""` means the tool never runs, and the error names a field
        // the model deliberately left blank, which it cannot satisfy.
        if raw.trim().is_empty() {
            tinyagents_tracing::debug!(
                tool = name,
                slot,
                param = param_name.as_str(),
                "[pformat] empty value for a named slot — argument omitted"
            );
            continue;
        }
        let coerced = coerce_value(
            raw,
            params
                .types
                .get(slot)
                .copied()
                .unwrap_or(PFormatParamType::String),
        );
        // Last write wins on a repeated slot. Rare enough not to be worth
        // rejecting the whole call over, and the later value is the model's
        // latest intent.
        args.insert(param_name.clone(), coerced);
    }

    Some((name.to_string(), Value::Object(args)))
}

/// Split a p-format argument body on unescaped `|`. Honours `\|`,
/// `\]`, and `\\` escapes. An empty body produces an empty `Vec` (NOT
/// `vec![""]`) so a tool with zero parameters parses cleanly.
fn split_pipes(input: &str) -> Vec<String> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('|') => {
                    current.push('|');
                    chars.next();
                }
                Some(']') => {
                    current.push(']');
                    chars.next();
                }
                Some('\\') => {
                    current.push('\\');
                    chars.next();
                }
                _ => current.push('\\'),
            }
        } else if c == '|' {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }

    out.push(current);
    out
}

/// Coerce a raw string argument into the JSON type the schema expects.
/// Falls back to `Value::String` for any failed coercion so the model
/// still gets a usable value into the tool argument map.
fn coerce_value(raw: &str, ty: PFormatParamType) -> Value {
    match ty {
        PFormatParamType::Integer => raw
            .trim()
            .parse::<i64>()
            .map(|n| Value::Number(n.into()))
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        PFormatParamType::Number => raw
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String(raw.to_string())),
        PFormatParamType::Boolean => match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Value::Bool(true),
            "false" | "no" | "0" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        PFormatParamType::String | PFormatParamType::Other => Value::String(raw.to_string()),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_registry() -> PFormatRegistry {
        let mut reg = PFormatRegistry::new();
        reg.insert(
            "get_weather".to_string(),
            PFormatToolParams::from_schema(&json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "unit": { "type": "string" }
                }
            })),
        );
        reg.insert(
            "shell".to_string(),
            PFormatToolParams::from_schema(&json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                }
            })),
        );
        reg.insert(
            "ping".to_string(),
            PFormatToolParams::from_schema(&json!({
                "type": "object",
                "properties": {}
            })),
        );
        reg.insert(
            "math".to_string(),
            PFormatToolParams::from_schema(&json!({
                "type": "object",
                "properties": {
                    "x": { "type": "integer" },
                    "y": { "type": "number" },
                    "verbose": { "type": "boolean" }
                }
            })),
        );
        reg
    }

    #[test]
    fn renders_zero_arg_signature() {
        let reg = make_registry();
        assert_eq!(render_signature("ping", &reg["ping"]), "ping[]");
    }

    #[test]
    fn renders_multi_arg_signature() {
        let reg = make_registry();
        assert_eq!(
            render_signature("get_weather", &reg["get_weather"]),
            "get_weather[0|<location>|1|<unit>]"
        );
    }

    #[test]
    fn parses_simple_call() {
        let reg = make_registry();
        let (name, args) = parse_call("get_weather[0|London|1|metric]", &reg).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args, json!({"location": "London", "unit": "metric"}));
    }

    #[test]
    fn parses_zero_arg_call() {
        let reg = make_registry();
        let (name, args) = parse_call("ping[]", &reg).unwrap();
        assert_eq!(name, "ping");
        assert_eq!(args, json!({}));
    }

    #[test]
    fn parses_single_arg_with_spaces() {
        let reg = make_registry();
        let (name, args) = parse_call("shell[0|ls -la /tmp]", &reg).unwrap();
        assert_eq!(name, "shell");
        assert_eq!(args, json!({"command": "ls -la /tmp"}));
    }

    #[test]
    fn handles_pipe_escape() {
        let reg = make_registry();
        let (_, args) = parse_call(r"shell[0|cat foo \| grep bar]", &reg).unwrap();
        assert_eq!(args, json!({"command": "cat foo | grep bar"}));
    }

    #[test]
    fn handles_bracket_escape() {
        let reg = make_registry();
        let (_, args) = parse_call(r"shell[0|echo \]done\]]", &reg).unwrap();
        assert_eq!(args, json!({"command": "echo ]done]"}));
    }

    #[test]
    fn handles_backslash_escape() {
        let reg = make_registry();
        let (_, args) = parse_call(r"shell[0|C:\\Users\\bob]", &reg).unwrap();
        assert_eq!(args, json!({"command": r"C:\Users\bob"}));
    }

    #[test]
    fn coerces_typed_arguments() {
        let reg = make_registry();
        // No `required` list on `math`, so all three are optional and sort
        // alphabetically: verbose, x, y. The signature the model sees is
        // `math[0|<verbose>|1|<x>|2|<y>]`, so these are the slots it would emit.
        let (_, args) = parse_call("math[0|true|1|42|2|2.75]", &reg).unwrap();
        assert_eq!(args, json!({"verbose": true, "x": 42, "y": 2.75}));
    }

    #[test]
    fn coercion_falls_back_to_string_on_failure() {
        let reg = make_registry();
        let (_, args) = parse_call("math[0|maybe|1|notanumber|2|alsonotanumber]", &reg).unwrap();
        assert_eq!(
            args,
            json!({
                "verbose": "maybe",
                "x": "notanumber",
                "y": "alsonotanumber"
            })
        );
    }

    #[test]
    fn optional_only_signature_is_alphabetical() {
        let reg = make_registry();
        // `math` declares no `required`, so every parameter is optional and the
        // layout falls back to `BTreeMap` order: {verbose, x, y}.
        assert_eq!(
            render_signature("math", &reg["math"]),
            "math[0|<verbose>|1|<x>|2|<y>]"
        );
    }

    #[test]
    fn rejects_unknown_tool() {
        let reg = make_registry();
        assert!(parse_call("nope[arg]", &reg).is_none());
    }

    #[test]
    fn rejects_missing_brackets() {
        let reg = make_registry();
        assert!(parse_call("get_weather London metric", &reg).is_none());
    }

    #[test]
    fn rejects_trailing_garbage() {
        let reg = make_registry();
        // Closing bracket isn't last char → invalid p-format, dispatcher
        // should try the JSON fallback path.
        assert!(parse_call("get_weather[0|London|1|metric] // comment", &reg).is_none());
    }

    #[test]
    fn rejects_a_slot_the_schema_has_no_parameter_for() {
        let reg = make_registry();
        // `get_weather` has two slots, 0 and 1. Slot 2 is not a value to drop —
        // it means the model is working from a layout this schema does not have,
        // so every *other* slot in the call is suspect too.
        assert!(parse_call("get_weather[0|London|1|metric|2|extra]", &reg).is_none());
    }

    #[test]
    fn an_empty_value_omits_the_key_rather_than_sending_a_blank() {
        let reg = make_registry();
        let (_, args) = parse_call("get_weather[0|London|1|]", &reg).unwrap();
        // `unit` is absent, not `""`. A blank string is what the old form sent,
        // and for a typed parameter it fails schema validation naming a field the
        // model deliberately left empty — an error it cannot act on.
        assert_eq!(args, json!({"location": "London"}));
    }

    // ── Indexed slots: the behaviour the format exists for ──────────────────

    fn sparse_registry() -> PFormatRegistry {
        let mut reg = PFormatRegistry::new();
        // Mirrors the shape that produced the live failures: one required
        // parameter, several optional ones that sort ahead of it alphabetically.
        reg.insert(
            "list_threads".to_string(),
            PFormatToolParams::from_schema(&json!({
                "type": "object",
                "properties": {
                    "max_results": { "type": "integer" },
                    "query": { "type": "string" },
                    "user_id": { "type": "string" },
                    "verbose": { "type": "boolean" }
                },
                "required": ["query"]
            })),
        );
        reg
    }

    #[test]
    fn required_parameters_are_numbered_first() {
        let reg = sparse_registry();
        // Alphabetically `query` would be slot 1, behind `max_results`. Required
        // first puts it at 0, which is what makes the one-argument call
        // `list_threads[0|…]` rather than an index the model has to look up.
        assert_eq!(
            render_signature("list_threads", &reg["list_threads"]),
            "list_threads[0|<query>|1|<max_results>|2|<user_id>|3|<verbose>]"
        );
    }

    #[test]
    fn a_sparse_call_sends_only_the_slots_it_names() {
        let reg = sparse_registry();
        let (_, args) = parse_call("list_threads[0|from:alice|1|50]", &reg).unwrap();
        // No empty slots to count, and the absent parameters are absent rather
        // than blank.
        assert_eq!(args, json!({"query": "from:alice", "max_results": 50}));
    }

    #[test]
    fn the_live_misbinding_is_now_a_refusal_rather_than_a_wrong_call() {
        let reg = sparse_registry();
        // The failure this format replaces: a bare-positional call whose leading
        // delimiter count was off by one bound the search text to `user_id` and
        // *ran*. In the indexed form the same body has no numeric index at slot
        // 0, so it is rejected and the model is told, instead of a wrong call
        // succeeding.
        assert!(parse_call("list_threads[|||from:alice]", &reg).is_none());
    }

    #[test]
    fn rejects_a_bare_positional_call() {
        let reg = make_registry();
        // Deliberate: parsing this positionally is exactly the silent misbinding
        // the indices exist to end, so the old form is refused rather than
        // accepted for compatibility.
        assert!(parse_call("get_weather[London|metric]", &reg).is_none());
    }

    #[test]
    fn rejects_an_odd_token_count() {
        let reg = make_registry();
        // A dropped or added delimiter. Keeping the pairs that happen to line up
        // would put the off-by-one straight back.
        assert!(parse_call("get_weather[0|London|1]", &reg).is_none());
    }

    #[test]
    fn rejects_a_non_numeric_index() {
        let reg = make_registry();
        assert!(parse_call("get_weather[location|London]", &reg).is_none());
    }

    #[test]
    fn a_repeated_slot_takes_the_last_value() {
        let reg = make_registry();
        // Rare enough not to be worth failing the whole call over, and the later
        // value is the model's latest intent.
        let (_, args) = parse_call("get_weather[0|London|0|Berlin]", &reg).unwrap();
        assert_eq!(args, json!({"location": "Berlin"}));
    }

    #[test]
    fn an_empty_body_is_a_call_with_no_arguments() {
        let reg = make_registry();
        // Distinct from `ping[]`: `get_weather` *has* slots, and sending none of
        // them is a valid (if incomplete) call the schema validator should judge,
        // not a parse failure.
        let (name, args) = parse_call("get_weather[]", &reg).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args, json!({}));
    }

    #[test]
    fn an_escaped_pipe_stays_inside_its_value() {
        let reg = make_registry();
        // The index/value split runs over `split_pipes`, which honours escapes —
        // so an escaped pipe is part of one value and does not shift the pairing.
        let (_, args) = parse_call(r"shell[0|cat a \| grep b]", &reg).unwrap();
        assert_eq!(args, json!({"command": "cat a | grep b"}));
    }

    #[test]
    fn signature_round_trips_with_parser() {
        let reg = make_registry();
        let sig = render_signature("get_weather", &reg["get_weather"]);
        // Render uses the same identifier the parser expects.
        assert!(sig.starts_with("get_weather["));
        let synthesised = "get_weather[0|Berlin|1|imperial]";
        let (name, args) = parse_call(synthesised, &reg).unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], json!("Berlin"));
        assert_eq!(args["unit"], json!("imperial"));
    }
}
