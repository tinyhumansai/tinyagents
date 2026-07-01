//! End-to-end tool-calling loop against the real OpenAI API.
//!
//! Registers a real local [`Tool`] (a `get_weather` lookup) alongside an
//! [`OpenAiModel`], then asks a question that should trigger the tool. The
//! harness drives the full model -> tool -> model loop: OpenAI requests the
//! tool, the harness runs it locally, feeds the result back, and OpenAI
//! produces the final answer.
//!
//! Run with:
//!
//! ```text
//! cargo run --example openai_tools
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::Result;
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

/// A tiny canned weather tool. In a real app this would call a weather API; for
/// the example it returns a deterministic string so the loop is reproducible.
struct WeatherTool;

#[async_trait]
impl Tool<()> for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Returns the current weather for a given city."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "get_weather",
            "Returns the current weather for a given city.",
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. \"Paris\"."
                    }
                },
                "required": ["city"]
            }),
        )
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        let city = call
            .arguments
            .get("city")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        eprintln!("[tool] get_weather(city = {city:?})");
        Ok(ToolResult::text(
            call.id,
            "get_weather",
            format!("It is sunny and 21C in {city}."),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let model = OpenAiModel::from_env()?;
    println!("=== OpenAI tool-calling loop ===");
    println!("model: {}", model.model());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("openai", Arc::new(model))
        .set_default_model("openai")
        .register_tool(Arc::new(WeatherTool));

    let question = "What is the weather in Paris right now? Use the tool.";
    println!("question: {question}\n");

    let run = harness
        .invoke_default(&(), vec![Message::user(question)])
        .await?;

    println!("final answer: {}", run.text().unwrap_or_default());
    println!("\nmodel calls : {}", run.model_calls);
    println!("tool calls  : {}", run.tool_calls);

    // Surface which tool results came back by scanning the transcript.
    let tool_results: Vec<String> = run
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(_) => Some(m.text()),
            _ => None,
        })
        .collect();
    println!("tool results: {tool_results:?}");

    println!(
        "usage       : {} input + {} output = {} total tokens over {} call(s)",
        run.usage.usage.input_tokens,
        run.usage.usage.output_tokens,
        run.usage.usage.total_tokens,
        run.usage.calls,
    );

    Ok(())
}
