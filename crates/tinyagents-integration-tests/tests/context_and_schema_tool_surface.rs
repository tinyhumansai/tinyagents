//! End-to-end tests for the tool-layer surfaces added alongside the compaction
//! fixes: unique synthetic call ids, tool-result artifacts, the provider schema
//! projection seam, injected arguments, and per-tool error policy.
//!
//! Unlike `context_and_schema_compaction.rs`, most of these exercise APIs that
//! did not exist before, so they cannot be run red against the unfixed crate —
//! except [`synthetic_tool_call_ids_are_unique_across_turns`], which uses only
//! the pre-existing `parse_prompt_tool_calls_from_text` and does fail there.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::Result;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::tool::{
    SchemaPreparation, Tool, ToolErrorPolicy, ToolRegistry, ToolResult,
    parse_prompt_tool_calls_from_text, prepare_tool_schemas, strip_injected_arguments,
};
use tinyinference::message::Message;
use tinyinference::tool::{ToolCall, ToolSchema};

// ---------------------------------------------------------------------------
// TOOL-2 — synthetic call ids
// ---------------------------------------------------------------------------

/// Two turns of one run must not both mint `call_1`. When they did, the next
/// request declared the same tool-call id twice and answered it twice, leaving
/// the pairing unresolvable for the provider and for the harness alike.
#[test]
fn synthetic_tool_call_ids_are_unique_across_turns() {
    let turn = r#"<tool_call>{"name":"lookup","arguments":{"id":1}}</tool_call>"#;
    let (_, first) = parse_prompt_tool_calls_from_text(turn);
    let (_, second) = parse_prompt_tool_calls_from_text(turn);

    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_ne!(
        first[0].id, second[0].id,
        "the second turn reused the first turn's synthetic tool-call id"
    );
}

// ---------------------------------------------------------------------------
// C7 — content_and_artifact
// ---------------------------------------------------------------------------

/// A tool returns a short model-facing summary and a large structured payload;
/// the transcript must carry both, with only the summary visible as text.
#[test]
fn a_tool_result_artifact_survives_into_the_transcript() {
    let mut result = ToolResult::text("c1", "query_rows", "3 rows matched");
    result.raw = Some(json!({"rows": [{"id": 1}, {"id": 2}, {"id": 3}]}));

    let message = tinyagents_harness::tool::message_from_result(&result);

    assert_eq!(message.text(), "3 rows matched");
    assert_eq!(
        message
            .artifact()
            .and_then(|a| a["rows"].as_array())
            .map(Vec::len),
        Some(3)
    );
    // The payload is host-side state and is not charged to the context window.
    assert_eq!(
        message.estimated_char_weight(),
        "3 rows matched".len() + "c1".len()
    );
}

// ---------------------------------------------------------------------------
// TOOL-6 — the provider projection seam
// ---------------------------------------------------------------------------

/// A schema written the way every JSON-Schema generator writes one — with
/// `$defs` and a `$ref` — must be projected into a shape Anthropic accepts.
#[test]
fn tool_schemas_are_projected_for_the_target_provider() {
    let declared = vec![ToolSchema::new(
        "lookup",
        "Look a record up",
        json!({
            "type": "object",
            "$defs": {"Id": {"type": "string"}},
            "properties": {"id": {"$ref": "#/$defs/Id"}},
            "required": ["id"],
        }),
    )];

    let wire = prepare_tool_schemas(&declared, &SchemaPreparation::anthropic());
    assert_eq!(wire[0].parameters["properties"]["id"]["type"], "string");
    assert!(wire[0].parameters.get("$defs").is_none());
}

/// A tool that declares no arguments must not be able to serialise
/// `"parameters": null` and break the whole request.
#[test]
fn null_tool_parameters_are_normalized_rather_than_sent() {
    let declared = vec![ToolSchema::new("ping", "Ping", json!(null))];
    let wire = prepare_tool_schemas(&declared, &SchemaPreparation::openai());

    assert!(!wire[0].parameters.is_null());
    assert_eq!(wire[0].parameters["type"], "object");
}

// ---------------------------------------------------------------------------
// C8 — injected arguments
// ---------------------------------------------------------------------------

struct ContextualTool;

#[async_trait]
impl Tool<()> for ContextualTool {
    fn name(&self) -> &str {
        "contextual"
    }
    fn description(&self) -> &str {
        "Acts within the caller's thread"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "contextual",
            "Acts within the caller's thread",
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
fn injected_arguments_never_reach_the_model_and_cannot_be_forged() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(Arc::new(ContextualTool));

    // Declaration side: hidden from `properties` and from `required`.
    let wire = registry.schemas();
    assert!(wire[0].parameters["properties"].get("thread_id").is_none());
    assert_eq!(wire[0].parameters["required"], json!(["note"]));

    // Enforcement primitive: a model-supplied value for the hidden key is
    // discarded before anything else looks at the arguments.
    let mut forged = json!({"note": "hi", "thread_id": "someone-elses-thread"});
    let removed = strip_injected_arguments(&mut forged, &["thread_id"]);
    assert_eq!(removed, vec!["thread_id".to_string()]);
    assert_eq!(forged, json!({"note": "hi"}));
}

// ---------------------------------------------------------------------------
// C9 — per-tool error policy
// ---------------------------------------------------------------------------

#[test]
fn a_recoverable_tool_failure_can_be_handed_back_to_the_model() {
    let call = ToolCall::new("c1", "lookup", json!({}));
    let handled = ToolErrorPolicy::ReturnToError
        .apply(&call, Err(TinyAgentsError::Tool("no such record".into())))
        .expect("a handled error must not fail the run");

    assert!(handled.is_error());
    assert!(handled.content.contains("no such record"));
}

/// The rule that must never be relaxed: a policy may convert tool failures, but
/// never a cancellation or an interrupt. Swallowing one lets the loop keep
/// running after it was told to stop.
#[test]
fn no_error_policy_can_swallow_a_cancellation_or_an_interrupt() {
    let call = ToolCall::new("c1", "lookup", json!({}));

    for policy in [
        ToolErrorPolicy::Fail,
        ToolErrorPolicy::ReturnToError,
        ToolErrorPolicy::Message("masked".into()),
    ] {
        assert!(
            policy
                .apply(&call, Err(TinyAgentsError::Cancelled))
                .is_err(),
            "{policy:?} swallowed a cancellation"
        );
        assert!(
            policy
                .apply(
                    &call,
                    Err(TinyAgentsError::Interrupted {
                        node: "approval".into(),
                        message: "waiting".into(),
                    })
                )
                .is_err(),
            "{policy:?} swallowed an interrupt"
        );
    }
}
