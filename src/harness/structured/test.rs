//! Tests for structured-output extraction.
//!
//! Cover [`StructuredStrategy::for_profile`] selection, both extraction
//! strategies (provider-schema JSON parsing and tool-call argument reading,
//! including their error paths), [`response_format_for_strategy`] mapping, and
//! [`StructuredOutput::parse`] deserialisation.

use super::*;
use crate::harness::model::{ModelProfile, ModelResponse, ResponseFormat};
use crate::harness::tool::ToolCall;
use serde_json::json;

#[test]
fn auto_strategy_defaults_to_provider_schema_without_profile() {
    assert_eq!(
        StructuredStrategy::for_profile(None),
        StructuredStrategy::ProviderSchema
    );
}

#[test]
fn auto_strategy_uses_tool_call_when_no_native_structured_output() {
    let profile = ModelProfile {
        tool_calling: true,
        json_schema: false,
        native_structured_output: false,
        ..ModelProfile::default()
    };
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ToolCall
    );
}

#[test]
fn auto_strategy_uses_provider_schema_with_native_structured_output() {
    let profile = ModelProfile {
        native_structured_output: true,
        json_schema: true,
        ..ModelProfile::default()
    };
    assert_eq!(
        StructuredStrategy::for_profile(Some(&profile)),
        StructuredStrategy::ProviderSchema
    );
}

#[test]
fn provider_schema_parses_json_text() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "result", json!({}));
    let response = ModelResponse::assistant(r#"{"answer":42}"#);
    let output = extractor.extract(&response).unwrap();
    assert_eq!(output.value["answer"], 42);
    assert!(output.raw_text.is_some());
}

#[test]
fn provider_schema_errors_on_invalid_json() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "result", json!({}));
    let response = ModelResponse::assistant("not json");
    assert!(extractor.extract(&response).is_err());
}

#[test]
fn tool_call_strategy_reads_matching_call() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ToolCall, "extract_answer", json!({}));
    let mut response = ModelResponse::assistant("");
    response.message.tool_calls.push(ToolCall {
        id: "tc-1".to_string(),
        name: "extract_answer".to_string(),
        arguments: json!({"answer": "yes"}),
        invalid: None,
    });
    let output = extractor.extract(&response).unwrap();
    assert_eq!(output.value["answer"], "yes");
    assert!(output.raw_text.is_none());
}

#[test]
fn tool_call_strategy_errors_when_no_match() {
    let extractor = StructuredExtractor::new(StructuredStrategy::ToolCall, "my_tool", json!({}));
    let response = ModelResponse::assistant("");
    assert!(extractor.extract(&response).is_err());
}

#[test]
fn response_format_for_provider_schema_is_json_schema() {
    let fmt = response_format_for_strategy(StructuredStrategy::ProviderSchema, "foo", json!({}));
    assert!(matches!(fmt, ResponseFormat::JsonSchema { .. }));
}

#[test]
fn response_format_for_tool_call_is_text() {
    let fmt = response_format_for_strategy(StructuredStrategy::ToolCall, "foo", json!({}));
    assert_eq!(fmt, ResponseFormat::Text);
}

#[test]
fn structured_output_parse_deserialises() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Answer {
        value: String,
    }
    let output = StructuredOutput {
        value: json!({"value": "hello"}),
        raw_text: None,
    };
    let parsed: Answer = output.parse().unwrap();
    assert_eq!(parsed.value, "hello");
}

// -- truncation diagnostics --

/// A response whose content is `text`, stopped for `finish_reason`.
fn stopped(text: &str, finish_reason: &str) -> ModelResponse {
    ModelResponse::assistant(text).with_finish_reason(finish_reason)
}

#[test]
fn empty_content_stopped_for_length_reports_the_output_limit() {
    // A reasoning model can spend its whole output budget thinking and return
    // no content. Parsing the empty string reports "expected value at line 1
    // column 1", which sends the reader hunting for a malformed response that
    // was never sent, and hides the one-line fix.
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "review", json!({}));
    let err = extractor
        .extract(&stopped("", "length"))
        .expect_err("empty content is not extractable");

    let message = err.to_string();
    assert!(message.contains("output limit"), "{message}");
    assert!(message.contains("max_tokens"), "{message}");
    assert!(
        !message.contains("line 1 column 1"),
        "the misleading parse error must not survive: {message}"
    );
}

#[test]
fn empty_content_stopped_normally_reports_what_is_known() {
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "review", json!({}));
    let err = extractor
        .extract(&stopped("   ", "stop"))
        .expect_err("empty content is not extractable");
    assert!(err.to_string().contains("no content"), "{err}");
}

#[test]
fn genuinely_malformed_json_still_reports_a_parse_error() {
    // The new branch must not swallow the case it was not written for.
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ProviderSchema, "review", json!({}));
    let err = extractor
        .extract(&stopped("not json at all", "stop"))
        .expect_err("malformed content is not extractable");
    assert!(err.to_string().contains("not valid JSON"), "{err}");
}
