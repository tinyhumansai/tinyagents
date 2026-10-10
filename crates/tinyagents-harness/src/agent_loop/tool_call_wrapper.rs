//! Unwrapping a real call a model buried inside a pseudo "tool call" tool.
//!
//! Some models (DeepSeek, in production captures) do not call the tool they
//! mean. They call a tool literally named `tool_call` — or `call_tool`,
//! `function_call`, `tool_use`, `invoke` — and put the real call in its
//! arguments, in one of a few shapes:
//!
//! * `{"name": "use_skill", "arguments": {...}}` (arguments sometimes a JSON
//!   string), with `tool` / `tool_name` / `function` spelling the name key;
//! * `{"function": {"name": "use_skill", "arguments": "..."}}`, the OpenAI
//!   wire shape copied verbatim;
//! * `{"use_skill": {...}}`, a single key naming the tool;
//! * `{"skill": "...", "tool": "...", "args": {...}}`, `use_skill`'s own
//!   argument object with the tool name dropped.
//!
//! No such tool exists, so every one of these failed as an unknown tool. The
//! intent is unambiguous and the rewrite is mechanical, so the loop retargets
//! the call **before** admission: every rule, `before_tool` hook, approval
//! gate and schema check then sees the real tool name and its arguments, as
//! if the model had called it directly. Nothing here decides policy; a call
//! whose payload names no callable tool is left alone and gets the ordinary
//! unknown-tool corrective.

use serde_json::{Map, Value};
use tinytools_agent::repair::args::ARGUMENT_KEYS;

/// Names a model uses for a pseudo tool that wraps the real call.
pub(super) const WRAPPER_NAMES: &[&str] = &[
    "tool_call",
    "tool_calls",
    "call_tool",
    "use_tool",
    "function_call",
    "tool_use",
    "invoke",
];

/// Keys that may carry the wrapped tool's name, in priority order.
const NAME_KEYS: &[&str] = &["name", "tool", "tool_name", "function"];

/// The pack tool the `{skill, tool, args}` shape belongs to.
const USE_SKILL: &str = crate::tool::packs::USE_SKILL;

/// Whether `name` is one of the pseudo wrapper tool names.
pub(super) fn is_wrapper_name(name: &str) -> bool {
    let name = name.trim();
    WRAPPER_NAMES
        .iter()
        .any(|wrapper| wrapper.eq_ignore_ascii_case(name))
}

/// The real call inside a wrapper's `arguments`: the target tool's name and
/// its arguments. `None` when the payload does not identify a tool
/// `is_callable` admits, so the caller keeps the original call.
pub(super) fn unwrap_wrapped_call(
    arguments: &Value,
    is_callable: &dyn Fn(&str) -> bool,
) -> Option<(String, Value)> {
    let payload = decode_object(arguments)?;
    unwrap_object(&payload, is_callable, 0)
}

fn unwrap_object(
    payload: &Map<String, Value>,
    is_callable: &dyn Fn(&str) -> bool,
    depth: usize,
) -> Option<(String, Value)> {
    // `use_skill`'s own argument object. Checked before the name keys: here
    // `tool` names a tool *inside* a skill, not the wrapped tool itself.
    if payload.contains_key("skill") && is_callable(USE_SKILL) {
        let wrapped_name = NAME_KEYS
            .iter()
            .filter_map(|key| payload.get(*key).and_then(Value::as_str))
            .any(|name| name == USE_SKILL);
        // `{name: "web_search", skill: ..., arguments: ...}`: `name` (unlike
        // `tool`) never belongs to the `{skill, tool, args}` shape, so a
        // callable tool named there owns the `skill` key as its own argument.
        let named_real_tool = payload
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .is_some_and(|name| !is_wrapper_name(name) && is_callable(name));
        if !wrapped_name && !named_real_tool {
            return Some((USE_SKILL.to_string(), Value::Object(payload.clone())));
        }
    }

    for key in NAME_KEYS {
        match payload.get(*key) {
            Some(Value::String(name)) => {
                let name = name.trim();
                if is_wrapper_name(name) || !is_callable(name) {
                    continue;
                }
                return Some((name.to_string(), wrapped_arguments(payload, key)));
            }
            // `{"function": {"name": ..., "arguments": ...}}`, one level only.
            Some(Value::Object(inner)) if depth == 0 => {
                if let Some(found) = unwrap_object(inner, is_callable, depth + 1) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }

    // `{"use_skill": {...}}`: a single key naming a callable tool.
    if payload.len() == 1
        && let Some((name, value)) = payload.iter().next()
        && !is_wrapper_name(name)
        && is_callable(name)
    {
        let inner = match value {
            Value::Null => Value::Object(Map::new()),
            Value::String(raw) => decode_lenient(raw),
            other => other.clone(),
        };
        return Some((name.clone(), inner));
    }
    None
}

/// The wrapped call's arguments: the first [`ARGUMENT_KEYS`] entry, decoded
/// when it is a JSON string; otherwise every key but the name key, which is
/// the "flattened" spelling `{"tool": "x", "query": "..."}`.
fn wrapped_arguments(payload: &Map<String, Value>, name_key: &str) -> Value {
    for key in ARGUMENT_KEYS {
        match payload.get(*key) {
            Some(Value::String(raw)) => return decode_lenient(raw),
            Some(Value::Null) => return Value::Object(Map::new()),
            Some(value) => return value.clone(),
            None => {}
        }
    }
    let rest: Map<String, Value> = payload
        .iter()
        .filter(|(key, _)| key.as_str() != name_key)
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Value::Object(rest)
}

/// A JSON string decoded strictly, then through the relaxed-object ladder.
/// An undecodable string is kept as-is so schema validation reports it.
fn decode_lenient(raw: &str) -> Value {
    let candidate = tinytools_agent::repair::json::strip_code_fence(raw);
    serde_json::from_str::<Value>(candidate)
        .ok()
        .or_else(|| tinytools_agent::repair::json::recover_object(candidate))
        .unwrap_or_else(|| Value::String(raw.to_string()))
}

/// The wrapper's own arguments as an object, decoding a string payload (the
/// provider could not parse it, or the model stringified it).
fn decode_object(arguments: &Value) -> Option<Map<String, Value>> {
    match arguments {
        Value::Object(map) => Some(map.clone()),
        Value::String(raw) => match decode_lenient(raw) {
            Value::Object(map) => Some(map),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
#[path = "tool_call_wrapper_tests.rs"]
mod tests;
