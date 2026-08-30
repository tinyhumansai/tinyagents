//! Regression coverage for TOOL-4: an unknown tool name and invalid tool
//! arguments used to abort the whole run by default.
//!
//! That made the crate inconsistent with itself — an *unparseable* arguments
//! blob has always recovered unconditionally, so `{city:` survived while
//! `{"city": 5}` killed the run — and diverged from LangGraph, which never
//! fails on either.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::runtime::{AgentHarness, InvalidArgsPolicy, RunPolicy, UnknownToolPolicy};
use tinyagents_harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest, ModelResponse};

/// The defaults are the whole point of this file, so pin them directly too.
#[test]
fn recoverable_variants_are_the_defaults() {
    assert_eq!(
        UnknownToolPolicy::default(),
        UnknownToolPolicy::ReturnToolError
    );
    assert_eq!(
        InvalidArgsPolicy::default(),
        InvalidArgsPolicy::ReturnToolError
    );
    assert_eq!(
        RunPolicy::default().unknown_tool,
        UnknownToolPolicy::ReturnToolError
    );
    assert_eq!(
        RunPolicy::default().invalid_args,
        InvalidArgsPolicy::ReturnToolError
    );
}

/// A model that emits one scripted tool call and then a plain final answer.
struct TwoTurnModel {
    call: Mutex<Option<ToolCall>>,
}

impl TwoTurnModel {
    fn new(call: ToolCall) -> Self {
        Self {
            call: Mutex::new(Some(call)),
        }
    }
}

#[async_trait]
impl ChatModel<()> for TwoTurnModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        match self.call.lock().expect("poisoned").take() {
            Some(call) => {
                let mut response = ModelResponse::assistant("");
                response.message.content.clear();
                response.message.tool_calls = vec![call];
                response.finish_reason = Some("tool_calls".to_string());
                Ok(response)
            }
            None => Ok(ModelResponse::assistant("recovered")),
        }
    }
}

/// A tool whose schema genuinely requires a string `city`.
struct WeatherTool;

#[async_trait]
impl Tool<()> for WeatherTool {
    fn name(&self) -> &str {
        "weather"
    }

    fn description(&self) -> &str {
        "Looks up the weather for a city."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "weather".to_string(),
            "Looks up the weather for a city.".to_string(),
            json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        )
    }

    async fn call(&self, _state: &(), call: ToolCall) -> tinyagents_harness::Result<ToolResult> {
        Ok(ToolResult::text(call.id, call.name, "sunny"))
    }
}

/// TOOL-4a: a hallucinated tool name is answered with a tool-error message the
/// model can act on, and the run continues.
#[tokio::test]
async fn an_unknown_tool_name_recovers_by_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(TwoTurnModel::new(ToolCall::new("c1", "nope", json!({})))),
    );
    harness.register_tool(Arc::new(WeatherTool));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("an unknown tool name must not abort the run");

    assert_eq!(run.text(), Some("recovered".to_string()));
    let tool_text = run
        .messages
        .iter()
        .find(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text())
        .expect("a tool-error message must be injected");
    assert!(
        tool_text.contains("nope"),
        "the recovery message should name the unknown tool, got: {tool_text}"
    );
}

/// TOOL-4b: arguments that fail schema validation are answered with the
/// validation detail rather than aborting the run.
#[tokio::test]
async fn invalid_tool_arguments_recover_by_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(TwoTurnModel::new(ToolCall::new(
            "c1",
            "weather",
            json!({ "city": 5 }),
        ))),
    );
    harness.register_tool(Arc::new(WeatherTool));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("invalid arguments must not abort the run");

    assert_eq!(run.text(), Some("recovered".to_string()));
    assert!(
        run.messages
            .iter()
            .any(|m| matches!(m, Message::Tool(_)) && !m.text().is_empty()),
        "a validation-error tool message must be injected, got {:?}",
        run.messages
    );
}

/// `Fail` is still available for callers that genuinely want a hard stop.
#[tokio::test]
async fn the_fail_variants_are_still_available() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(TwoTurnModel::new(ToolCall::new("c1", "nope", json!({})))),
    );
    harness.register_tool(Arc::new(WeatherTool));
    harness.with_policy(RunPolicy {
        unknown_tool: UnknownToolPolicy::Fail,
        ..RunPolicy::default()
    });

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("an opted-in Fail policy still aborts");
}
