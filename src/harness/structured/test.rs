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
