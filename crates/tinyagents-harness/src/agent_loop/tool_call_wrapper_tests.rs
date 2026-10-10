use super::*;
use serde_json::json;

fn callable(name: &str) -> bool {
    matches!(name, "use_skill" | "web_search" | "tool_search")
}

fn unwrap(arguments: Value) -> Option<(String, Value)> {
    unwrap_wrapped_call(&arguments, &callable)
}

#[test]
fn wrapper_names_match_case_insensitively() {
    for name in [
        "tool_call",
        "TOOL_CALL",
        " call_tool ",
        "function_call",
        "tool_use",
        "invoke",
    ] {
        assert!(is_wrapper_name(name), "{name}");
    }
    assert!(!is_wrapper_name("use_skill"));
    assert!(!is_wrapper_name("tool_search"));
}

#[test]
fn name_and_arguments_object() {
    assert_eq!(
        unwrap(json!({ "name": "web_search", "arguments": { "query": "rust" } })),
        Some(("web_search".into(), json!({ "query": "rust" })))
    );
}

#[test]
fn stringified_arguments_are_decoded_strictly_then_leniently() {
    assert_eq!(
        unwrap(json!({ "name": "web_search", "arguments": "{\"query\":\"rust\"}" })),
        Some(("web_search".into(), json!({ "query": "rust" })))
    );
    assert_eq!(
        unwrap(json!({ "tool": "web_search", "args": "{query: \"rust\",}" })),
        Some(("web_search".into(), json!({ "query": "rust" })))
    );
}

#[test]
fn an_undecodable_argument_string_is_kept_for_the_validator() {
    assert_eq!(
        unwrap(json!({ "name": "web_search", "arguments": "rust please" })),
        Some(("web_search".into(), json!("rust please")))
    );
}

#[test]
fn flattened_arguments_drop_only_the_name_key() {
    assert_eq!(
        unwrap(json!({ "tool_name": "web_search", "query": "rust" })),
        Some(("web_search".into(), json!({ "query": "rust" })))
    );
}

#[test]
fn openai_function_shape_is_unwrapped_one_level() {
    assert_eq!(
        unwrap(json!({ "function": { "name": "web_search", "arguments": "{\"query\":\"x\"}" } })),
        Some(("web_search".into(), json!({ "query": "x" })))
    );
}

#[test]
fn a_single_key_naming_a_tool() {
    assert_eq!(
        unwrap(json!({ "use_skill": { "skill": "email" } })),
        Some(("use_skill".into(), json!({ "skill": "email" })))
    );
    assert_eq!(
        unwrap(json!({ "web_search": "{\"query\":\"x\"}" })),
        Some(("web_search".into(), json!({ "query": "x" })))
    );
}

#[test]
fn the_skill_shape_means_use_skill_even_with_a_tool_key() {
    let payload = json!({ "skill": "email", "tool": "send", "args": { "to": "a" } });
    assert_eq!(unwrap(payload.clone()), Some(("use_skill".into(), payload)));
}

#[test]
fn a_wrapper_naming_use_skill_still_takes_its_arguments() {
    assert_eq!(
        unwrap(json!({ "name": "use_skill", "skill": "email" })),
        Some(("use_skill".into(), json!({ "skill": "email" })))
    );
}

#[test]
fn a_stringified_wrapper_payload_is_decoded() {
    assert_eq!(
        unwrap(json!(
            "{\"name\":\"web_search\",\"arguments\":{\"query\":\"x\"}}"
        )),
        Some(("web_search".into(), json!({ "query": "x" })))
    );
}

#[test]
fn nothing_callable_means_no_rewrite() {
    assert_eq!(
        unwrap(json!({ "name": "frobnicate", "arguments": {} })),
        None
    );
    assert_eq!(
        unwrap(json!({ "name": "tool_call", "arguments": {} })),
        None
    );
    assert_eq!(unwrap(json!({ "query": "rust" })), None);
    assert_eq!(unwrap(json!({})), None);
    assert_eq!(unwrap(json!("not json")), None);
    assert_eq!(unwrap(json!([1, 2])), None);
}

#[test]
fn the_skill_shape_needs_use_skill_to_be_callable() {
    let only_search = |name: &str| name == "web_search";
    assert_eq!(
        unwrap_wrapped_call(&json!({ "skill": "email" }), &only_search),
        None
    );
}

#[test]
fn a_callable_tool_named_by_name_keeps_its_own_skill_argument() {
    let callable = |name: &str| matches!(name, "use_skill" | "calendar");
    assert_eq!(
        unwrap_wrapped_call(
            &json!({ "name": "calendar", "skill": "email", "arguments": { "q": "x" } }),
            &callable
        )
        .map(|(name, _)| name),
        Some("calendar".to_string())
    );
}
