//! Tests for injected (host-supplied) tool arguments.
//!
//! Cover the two halves of the feature: the declaration side — an injected key
//! is projected out of the model-facing schema, from `properties` *and* from
//! `required` — and the enforcement primitive that discards a model-supplied
//! value for such a key before it can be used.

use async_trait::async_trait;
use serde_json::json;

use super::*;

/// A tool that receives its caller's thread id from the host rather than the
/// model — the shape `SubAgentTool` has to hand-roll today.
struct ThreadScopedTool;

#[async_trait]
impl Tool<()> for ThreadScopedTool {
    fn name(&self) -> &str {
        "thread_scoped"
    }

    fn description(&self) -> &str {
        "Does something in the caller's thread"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "thread_scoped",
            "Does something in the caller's thread",
            json!({
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "thread_id": {"type": "string"},
                },
                "required": ["note", "thread_id"],
            }),
        )
    }

    fn injected_arguments(&self) -> &[&str] {
        &["thread_id"]
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, call.name, "ok"))
    }
}

#[test]
fn injected_arguments_are_hidden_from_the_model_facing_schema() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(std::sync::Arc::new(ThreadScopedTool));

    let model_facing = registry.schemas();
    let properties = &model_facing[0].parameters["properties"];
    assert!(properties.get("note").is_some());
    assert!(
        properties.get("thread_id").is_none(),
        "an injected argument was advertised to the model"
    );

    // Leaving it in `required` would demand an argument the model cannot see.
    let required = model_facing[0].parameters["required"].as_array().unwrap();
    assert_eq!(required, &vec![json!("note")]);
}

#[test]
fn declared_schemas_keep_injected_arguments_for_introspection() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(std::sync::Arc::new(ThreadScopedTool));

    let declared = registry.declared_schemas();
    assert!(declared[0].parameters["properties"]["thread_id"].is_object());
    assert_eq!(
        registry.injected_arguments().get("thread_scoped"),
        Some(&vec!["thread_id".to_string()])
    );
}

#[test]
fn a_model_supplied_value_for_an_injected_key_is_discarded() {
    // The forgery this prevents: the model names a hidden key in its own
    // arguments, hoping the host will honour it.
    let mut arguments = json!({"note": "hi", "thread_id": "victim-thread"});
    let removed = strip_injected_arguments(&mut arguments, &["thread_id"]);

    assert_eq!(removed, vec!["thread_id".to_string()]);
    assert_eq!(arguments, json!({"note": "hi"}));
}

#[test]
fn stripping_is_a_no_op_without_injected_keys_or_object_arguments() {
    let mut arguments = json!({"note": "hi"});
    assert!(strip_injected_arguments(&mut arguments, &[]).is_empty());
    assert_eq!(arguments, json!({"note": "hi"}));

    // An `invalid` call preserves raw text as a JSON string; there is no key to
    // forge in a scalar, and it must not be mangled.
    let mut raw = json!("{not json");
    assert!(strip_injected_arguments(&mut raw, &["thread_id"]).is_empty());
    assert_eq!(raw, json!("{not json"));
}

#[test]
fn projection_tolerates_schemas_without_properties_or_required() {
    let schema = ToolSchema::new("bare", "no args", json!({"type": "object"}));
    let projected = project_injected_arguments(schema.clone(), &["thread_id"]);
    assert_eq!(projected.parameters, schema.parameters);

    let scalar = ToolSchema::new("odd", "odd", json!("nonsense"));
    let projected = project_injected_arguments(scalar.clone(), &["thread_id"]);
    assert_eq!(projected.parameters, scalar.parameters);
}

#[test]
fn tools_declare_no_injected_arguments_by_default() {
    struct Plain;

    #[async_trait]
    impl Tool<()> for Plain {
        fn name(&self) -> &str {
            "plain"
        }
        fn description(&self) -> &str {
            "plain"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("plain", "plain", json!({"type": "object"}))
        }
        async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::text(call.id, call.name, "ok"))
        }
    }

    assert!(Tool::<()>::injected_arguments(&Plain).is_empty());
}
