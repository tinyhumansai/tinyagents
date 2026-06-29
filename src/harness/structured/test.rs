//! Tests added in a later pass.

use super::*;
use crate::harness::message::AssistantMessage;
use crate::harness::model::{ModelResponse, ResponseFormat};
use crate::harness::tool::ToolCall;
use serde_json::json;

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
    let extractor =
        StructuredExtractor::new(StructuredStrategy::ToolCall, "my_tool", json!({}));
    let response = ModelResponse::assistant("");
    assert!(extractor.extract(&response).is_err());
}

#[test]
fn response_format_for_provider_schema_is_json_schema() {
    let fmt = response_format_for_strategy(
        StructuredStrategy::ProviderSchema,
        "foo",
        json!({}),
    );
    assert!(matches!(fmt, ResponseFormat::JsonSchema { .. }));
}

#[test]
fn response_format_for_tool_call_is_text() {
    let fmt =
        response_format_for_strategy(StructuredStrategy::ToolCall, "foo", json!({}));
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
