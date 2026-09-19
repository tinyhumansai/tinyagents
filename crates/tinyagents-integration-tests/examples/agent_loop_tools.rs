//! End-to-end agent loop with a real tool.
//!
//! Builds an [`AgentHarness`] whose model is scripted to first request a tool
//! call and then, after seeing the tool result, produce a final answer. A small
//! real [`Tool`] (a calculator) is registered so the loop has something to run.
//!
//! Run with:
//!
//! ```text
//! cargo run -p tinyagents-integration-tests --example agent_loop_tools
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::Result;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::testkit::ScriptedModel;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolResult};

/// A tiny calculator tool that adds two numbers from its JSON arguments.
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers `a` and `b` and returns their sum."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        let a = arguments.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = arguments.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sum = a + b;
        Ok(ToolResult::success(format!("{sum}")))
    }
}

/// Builds an assistant response that requests a single tool call.
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(12, 4)),
        },
        usage: Some(Usage::new(12, 4)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// Builds a final, plain-text assistant response.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(20, 8)),
        },
        usage: Some(Usage::new(20, 8)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // The scripted model returns a tool call first, then a final answer once it
    // has seen the tool's output.
    let model = ScriptedModel::new(vec![
        tool_call_response("call-1", "add", json!({ "a": 2, "b": 40 })),
        text_response("The answer is 42."),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(model))
        .set_default_model("mock")
        .register_tool(Arc::new(CalculatorTool));

    let run = harness
        .invoke_default(&(), vec![Message::user("What is 2 + 40?")])
        .await?;

    println!("=== Agent loop with tools ===");
    println!("final text : {}", run.text().unwrap_or_default());
    println!("model calls: {}", run.model_calls);
    println!("tool calls : {}", run.tool_calls);

    // Surface which tools ran by scanning the transcript for tool result
    // messages.
    let tool_messages: Vec<String> = run
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(_) => Some(m.text()),
            _ => None,
        })
        .collect();
    println!("tool results: {tool_messages:?}");

    println!(
        "usage      : {} input + {} output = {} total tokens (over {} calls)",
        run.usage.usage.input_tokens,
        run.usage.usage.output_tokens,
        run.usage.usage.total_tokens,
        run.usage.calls,
    );

    Ok(())
}
