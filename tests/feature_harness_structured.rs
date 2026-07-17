//! Feature coverage for the harness structured-output surface.
//!
//! These tests exercise the *public* structured-output feature two ways:
//!
//! 1. the standalone [`StructuredExtractor`] API — both the `ProviderSchema`
//!    (parse JSON text) and `ToolCall` (read tool-call arguments) strategies,
//!    their error paths, strategy selection from a [`ModelProfile`], and the
//!    [`response_format_for_strategy`] mapping; and
//! 2. the end-to-end harness path — a run configured with a
//!    [`ResponseFormat`] surfaces a typed value on [`AgentRun::structured`],
//!    including after a tool round-trip.
//!
//! Everything is deterministic and offline (testkit doubles / `MockModel`);
//! no network or live provider is touched. These scenarios are additive to the
//! module-level unit tests: they focus on typed `parse::<T>()` round-trips and
//! the post-tool-call structured surface rather than raw JSON-value asserts.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::{ModelProfile, ModelResponse, ResponseFormat};
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::structured::{
    StructuredExtractor, StructuredStrategy, response_format_for_strategy,
};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;

/// A typed target used to prove `StructuredOutput::parse::<T>()` round-trips.
#[derive(Debug, Deserialize, PartialEq)]
struct Answer {
    value: String,
    score: i64,
}

fn object_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "value": { "type": "string" },
            "score": { "type": "integer" }
        },
        "required": ["value", "score"]
    })
}

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(6, 2)),
        },
        usage: Some(Usage::new(6, 2)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

// ── Standalone extractor: ProviderSchema strategy ─────────────────────────────

#[tokio::test]
async fn provider_schema_extracts_and_parses_typed_struct() {
    let extractor = StructuredExtractor::new(
        StructuredStrategy::ProviderSchema,
        "answer",
        object_schema(),
    );

    let response = ModelResponse::assistant(r#"{"value":"forty-two","score":42}"#);
    let output = extractor
        .extract(&response)
        .expect("valid JSON text extracts");

    // The JSON value is available directly ...
    assert_eq!(output.as_value()["value"], "forty-two");
    assert_eq!(output.as_value()["score"], 42);
    // ... and the raw text is retained for the provider-schema strategy.
    assert_eq!(
        output.raw_text.as_deref(),
        Some(r#"{"value":"forty-two","score":42}"#)
    );

    // The typed `parse::<T>()` round-trip is the headline feature.
    let parsed: Answer = output.parse().expect("parses into the typed struct");
    assert_eq!(
        parsed,
        Answer {
            value: "forty-two".into(),
            score: 42,
        }
    );
}

#[tokio::test]
async fn provider_schema_rejects_non_json_text() {
    let extractor = StructuredExtractor::new(
        StructuredStrategy::ProviderSchema,
        "answer",
        object_schema(),
    );

    let response = ModelResponse::assistant("this is not JSON at all");
    let err = extractor
        .extract(&response)
        .expect_err("non-JSON text must fail closed");

    assert!(
        matches!(err, TinyAgentsError::StructuredOutput(_)),
        "expected a StructuredOutput error, got {err:?}"
    );
}

#[tokio::test]
async fn provider_schema_parse_type_mismatch_errors() {
    // Valid JSON, but the shape does not match `Answer` (score is a string).
    let extractor = StructuredExtractor::new(
        StructuredStrategy::ProviderSchema,
        "answer",
        object_schema(),
    );
    let response = ModelResponse::assistant(r#"{"value":"x","score":"not-a-number"}"#);

    let output = extractor.extract(&response).expect("valid JSON extracts");
    let err = output
        .parse::<Answer>()
        .expect_err("a type mismatch must surface as an error");
    assert!(
        matches!(err, TinyAgentsError::StructuredOutput(_)),
        "expected a StructuredOutput error, got {err:?}"
    );
}

// ── Standalone extractor: ToolCall strategy ───────────────────────────────────

#[tokio::test]
async fn tool_call_strategy_reads_named_tool_arguments() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ToolCall, "answer", object_schema());

    // The structured value arrives as the arguments of a tool call named after
    // the schema; a differently named call is ignored.
    let response = ModelResponse {
        message: AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![
                ToolCall::new("c0", "unrelated", json!({ "noise": true })),
                ToolCall::new("c1", "answer", json!({ "value": "tooled", "score": 7 })),
            ],
            usage: None,
        },
        usage: None,
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    };

    let output = extractor
        .extract(&response)
        .expect("the matching tool call is found");
    let parsed: Answer = output.parse().expect("arguments parse into the struct");
    assert_eq!(
        parsed,
        Answer {
            value: "tooled".into(),
            score: 7,
        }
    );
    // Tool-call extraction carries no raw text (the value came from arguments).
    assert!(output.raw_text.is_none());
}

#[tokio::test]
async fn tool_call_strategy_errors_when_no_matching_call() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ToolCall, "answer", object_schema());
    let response = tool_call_response("c1", "something_else", json!({ "value": "x", "score": 1 }));

    let err = extractor
        .extract(&response)
        .expect_err("no tool call named `answer` is present");
    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "expected a Validation error, got {err:?}"
    );
}

// ── Strategy selection and response-format mapping ────────────────────────────

#[tokio::test]
async fn strategy_selection_follows_the_model_profile() {
    // No profile -> the conservative provider-native default.
    assert_eq!(
        StructuredStrategy::for_profile(None),
        StructuredStrategy::ProviderSchema
    );

    // A tool-calling model without native structured output -> tool-call mode.
    let mut profile = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ToolCall
    );

    // Native structured output + JSON schema support -> provider-native mode.
    profile.native_structured_output = true;
    profile.json_schema = true;
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ProviderSchema
    );
}

#[tokio::test]
async fn response_format_maps_from_strategy() {
    let schema = object_schema();

    let provider =
        response_format_for_strategy(StructuredStrategy::ProviderSchema, "answer", schema.clone());
    assert!(
        matches!(provider, ResponseFormat::JsonSchema { .. }),
        "provider-schema strategy asks for a JSON-schema response format"
    );

    let tool = response_format_for_strategy(StructuredStrategy::ToolCall, "answer", schema);
    assert!(
        matches!(tool, ResponseFormat::Text),
        "tool-call strategy uses plain text (structure arrives via arguments)"
    );
}

// ── End-to-end harness integration ────────────────────────────────────────────

#[tokio::test]
async fn harness_run_surfaces_structured_value_after_tool_round_trip() {
    // Turn 1 requests a tool; turn 2 returns the structured JSON answer. The
    // harness must extract the structured value from the *final* response even
    // though the run took a tool detour first.
    let scripted = MockModel::with_responses(vec![
        tool_call_response("c1", "lookup", json!({ "q": "x" })),
        ModelResponse::assistant(r#"{"value":"final","score":99}"#),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(scripted))
        .set_default_model("mock")
        .register_tool(Arc::new(tinyagents::harness::testkit::FakeTool::returning(
            "lookup",
            "looked-up",
        )))
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema("answer", object_schema())),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer me")])
        .await
        .expect("run succeeds");

    assert_eq!(
        run.model_calls, 2,
        "one tool turn plus the final answer turn"
    );
    assert_eq!(run.tool_calls, 1);

    let structured = run.structured.expect("structured output is surfaced");
    let parsed: Answer =
        serde_json::from_value(structured).expect("typed parse from the run value");
    assert_eq!(
        parsed,
        Answer {
            value: "final".into(),
            score: 99,
        }
    );
}

#[tokio::test]
async fn harness_run_fails_when_final_text_is_not_valid_structured_output() {
    // The model emits prose, not JSON, but the run demands a JSON-schema
    // structured output: extraction must fail the run closed.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(MockModel::constant("just some prose")))
        .set_default_model("mock")
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema("answer", object_schema())),
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("answer me")])
        .await
        .expect_err("non-JSON final text cannot satisfy a structured-output run");

    assert!(
        matches!(err, TinyAgentsError::StructuredOutput(_)),
        "expected a StructuredOutput error, got {err:?}"
    );
}

// ── ContentBlock passthrough sanity ───────────────────────────────────────────

#[tokio::test]
async fn provider_schema_reads_text_content_blocks() {
    // A response whose text lives in an explicit `ContentBlock::Text` (not the
    // `assistant` convenience) is still extracted correctly.
    let extractor = StructuredExtractor::new(
        StructuredStrategy::ProviderSchema,
        "answer",
        object_schema(),
    );
    let response = ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(r#"{"value":"blocky","score":5}"#.into())],
            tool_calls: Vec::new(),
            usage: None,
        },
        usage: None,
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    };

    let parsed: Answer = extractor
        .extract(&response)
        .expect("extracts from a text content block")
        .parse()
        .expect("parses");
    assert_eq!(parsed.value, "blocky");
    assert_eq!(parsed.score, 5);
}
