//! Regression coverage for the wave-2 structured-output defects (REASON-5,
//! REASON-9) and the typed context-overflow error (C15).
//!
//! | Test | Defect |
//! |------|--------|
//! | `extraction_validates_against_the_stored_schema` | REASON-5(a): the schema was stored and never read |
//! | `validation_error_names_the_failing_instance_path` | REASON-5(a) |
//! | `extraction_repairs_a_fenced_response` | REASON-5(b) |
//! | `extraction_repairs_a_truncated_response` | REASON-5(b) |
//! | `extraction_repairs_relaxed_json` | REASON-5(b) |
//! | `extract_outcome_records_a_failure_instead_of_raising` | REASON-5(c) |
//! | `non_tool_calling_profile_does_not_select_the_tool_call_strategy` | REASON-9 |
//! | `context_overflow_is_a_typed_variant` | C15 |

use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::model::{ModelProfile, ModelResponse};
use tinyagents::harness::structured::{StructuredExtractor, StructuredStrategy};

fn score_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "score": { "type": "integer" } },
        "required": ["score"],
        "additionalProperties": false
    })
}

fn extractor() -> StructuredExtractor {
    StructuredExtractor::new(StructuredStrategy::ProviderSchema, "score", score_schema())
}

// ── REASON-5(a): the schema is now enforced ───────────────────────────────────

#[test]
fn extraction_validates_against_the_stored_schema() {
    // Well-formed JSON of entirely the wrong shape used to succeed, leaving
    // `run.structured` holding something no caller asked for.
    let err = extractor()
        .extract(&ModelResponse::assistant(r#"{"wrong_key": 1}"#))
        .expect_err("a value that does not conform must not be reported as success");
    assert!(
        matches!(err, TinyAgentsError::StructuredOutput(_)),
        "{err:?}"
    );
}

#[test]
fn validation_error_names_the_failing_instance_path() {
    let err = extractor()
        .extract(&ModelResponse::assistant(r#"{"score": "four"}"#))
        .expect_err("a string is not an integer");
    let message = err.to_string();
    assert!(message.contains("schema 'score'.score"), "{message}");
    assert!(message.contains("integer"), "{message}");
}

#[test]
fn a_conforming_value_still_extracts() {
    let output = extractor()
        .extract(&ModelResponse::assistant(r#"{"score": 4}"#))
        .expect("a conforming value extracts");
    assert_eq!(output.value["score"], 4);
}

// ── REASON-5(b): the repair ladder ────────────────────────────────────────────

#[test]
fn extraction_repairs_a_fenced_response() {
    let output = extractor()
        .extract(&ModelResponse::assistant("```json\n{\"score\": 4}\n```"))
        .expect("a fenced value must not end the run");
    assert_eq!(output.value["score"], 4);
}

#[test]
fn extraction_repairs_a_truncated_response() {
    let schema = json!({
        "type": "object",
        "properties": { "summary": { "type": "string" } },
        "required": ["summary"]
    });
    let extractor = StructuredExtractor::new(StructuredStrategy::ProviderSchema, "review", schema);
    let output = extractor
        .extract(&ModelResponse::assistant(
            r#"{"summary": "cut off mid-sente"#,
        ))
        .expect("a truncated value is closed rather than discarding the run");
    assert_eq!(output.value["summary"], "cut off mid-sente");
}

#[test]
fn extraction_repairs_relaxed_json() {
    let output = extractor()
        .extract(&ModelResponse::assistant("{score: 4}"))
        .expect("unquoted keys are repaired, as they already are for tool arguments");
    assert_eq!(output.value["score"], 4);
}

#[test]
fn extraction_still_fails_on_text_that_is_not_json() {
    let err = extractor()
        .extract(&ModelResponse::assistant("I could not answer that."))
        .expect_err("the repair ladder must not launder prose into a value");
    assert!(err.to_string().contains("no conservative repair"), "{err}");
}

// ── REASON-5(c): non-fatal extraction ─────────────────────────────────────────

#[test]
fn extract_outcome_records_a_failure_instead_of_raising() {
    let response = ModelResponse::assistant("I could not answer that.");
    let outcome = extractor().extract_outcome(&response);

    assert!(!outcome.is_success());
    assert!(outcome.value.is_none());
    assert!(
        outcome
            .error
            .as_deref()
            .is_some_and(|e| e.contains("score")),
        "the recorded error must be usable as a repair prompt: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.raw.text(),
        "I could not answer that.",
        "the raw response must survive so a caller can re-ask or return it"
    );
}

#[test]
fn extract_outcome_carries_the_value_on_success() {
    let outcome = extractor().extract_outcome(&ModelResponse::assistant(r#"{"score": 4}"#));
    assert!(outcome.is_success());
    assert_eq!(outcome.into_result().unwrap()["score"], 4);
}

// ── REASON-9: strategy selection respects `tool_calling` ──────────────────────

#[test]
fn non_tool_calling_profile_does_not_select_the_tool_call_strategy() {
    // A profile that can do neither native structured output nor tool calling
    // used to route into `ToolCall`, whose forced `ToolChoice::Tool` is dropped
    // in prompt-guided mode — an extractor waiting for a call that can never
    // arrive.
    let profile = ModelProfile {
        tool_calling: false,
        json_schema: false,
        native_structured_output: false,
        ..ModelProfile::default()
    };
    assert_ne!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ToolCall
    );
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ProviderSchema
    );
}

#[test]
fn a_tool_calling_profile_still_selects_the_tool_call_strategy() {
    let profile = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ToolCall
    );
}

// ── C15: typed context overflow ───────────────────────────────────────────────

#[test]
fn context_overflow_is_a_typed_variant() {
    let error = TinyAgentsError::ContextOverflow {
        provider: "ollama".to_string(),
        model: Some("qwen3:8b".to_string()),
        message: "this model's maximum context length is 8192 tokens".to_string(),
    };
    assert!(error.is_context_overflow());
    assert!(error.to_string().contains("context overflow"), "{error}");
}

#[test]
fn a_provider_error_carrying_the_overflow_code_classifies_the_same() {
    // Providers construct `TinyAgentsError::Provider` directly today, so
    // classification must recognise the code as well as the typed variant —
    // otherwise the two paths disagree about the same failure.
    use tinyagents::harness::model::ProviderError;
    use tinyagents::harness::providers::openai::CONTEXT_OVERFLOW_CODE;

    let provider_error = ProviderError {
        provider: "openai".to_string(),
        status: Some(400),
        code: Some(CONTEXT_OVERFLOW_CODE.to_string()),
        message: "context length exceeded".to_string(),
        ..ProviderError::default()
    };
    let error = TinyAgentsError::from_provider_error(provider_error);

    assert!(error.is_context_overflow(), "{error:?}");
    assert!(matches!(error, TinyAgentsError::ContextOverflow { .. }));
}

#[test]
fn an_unrelated_provider_error_is_not_a_context_overflow() {
    use tinyagents::harness::model::ProviderError;

    let error = TinyAgentsError::from_provider_error(ProviderError {
        provider: "openai".to_string(),
        status: Some(429),
        message: "rate limited".to_string(),
        ..ProviderError::default()
    });
    assert!(!error.is_context_overflow());
    assert!(matches!(error, TinyAgentsError::Provider(_)));
}
