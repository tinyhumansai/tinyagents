//! Compatibility probe for local OpenAI-compatible servers (Ollama, LM Studio).
//!
//! Runs a matrix of tool-calling, streaming, and structured-output scenarios
//! directly against an [`OpenAiModel`] and prints a PASS/FAIL line per
//! scenario, so regressions against local backends are easy to spot.
//!
//! Point it at a server with the usual env vars:
//!
//! ```text
//! OPENAI_BASE_URL=http://localhost:1234/v1 \
//! OPENAI_MODEL=qwen/qwen3-4b \
//! OPENAI_API_KEY=local \
//! cargo run --example local_model_probe
//! ```

use futures::StreamExt;
use serde_json::json;

use tinyagents::harness::message::{AssistantMessage, Message};
use tinyagents::harness::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStreamItem, ResponseFormat, ToolChoice,
};
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::tool::{ToolCall, ToolSchema};

fn weather_tool() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Returns the current weather for a given city.",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name, e.g. \"Paris\"." }
            },
            "required": ["city"]
        }),
    )
}

fn time_tool() -> ToolSchema {
    ToolSchema::new(
        "get_time",
        "Returns the current local time. Takes no arguments.",
        json!({ "type": "object", "properties": {} }),
    )
}

fn base_request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        messages,
        max_tokens: Some(2048),
        ..ModelRequest::default()
    }
}

struct Outcome {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn ok(name: &'static str, detail: impl Into<String>) -> Outcome {
    Outcome {
        name,
        passed: true,
        detail: detail.into(),
    }
}

fn fail(name: &'static str, detail: impl Into<String>) -> Outcome {
    Outcome {
        name,
        passed: false,
        detail: detail.into(),
    }
}

fn first_call(response: &ModelResponse) -> Option<&ToolCall> {
    response.message.tool_calls.first()
}

fn msg_text(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .collect::<Vec<_>>()
        .join("")
}

/// 1. Plain text request/response.
async fn basic_text(model: &OpenAiModel) -> Outcome {
    let req = base_request(vec![Message::user("Reply with exactly: hello world")]);
    match model.invoke(&(), req).await {
        Ok(resp) => {
            let text = msg_text(&resp.message);
            if text.trim().is_empty() {
                fail("basic-text", "empty response text")
            } else {
                ok("basic-text", text.chars().take(60).collect::<String>())
            }
        }
        Err(e) => fail("basic-text", e.to_string()),
    }
}

/// 2. Model emits a tool call for an obvious tool prompt.
async fn tool_call(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "What is the weather in Paris? Use the get_weather tool.",
    )]);
    req.tools = vec![weather_tool()];
    match model.invoke(&(), req).await {
        Ok(resp) => match first_call(&resp) {
            Some(call) if call.arguments.get("city").is_some() => ok(
                "tool-call",
                format!("id={:?} args={}", call.id, call.arguments),
            ),
            Some(call) => fail("tool-call", format!("no city arg: {}", call.arguments)),
            None => fail(
                "tool-call",
                format!(
                    "no tool_calls; finish={:?} text={:?}",
                    resp.finish_reason,
                    msg_text(&resp.message)
                        .chars()
                        .take(120)
                        .collect::<String>()
                ),
            ),
        },
        Err(e) => fail("tool-call", e.to_string()),
    }
}

/// 3. Full round-trip: tool call -> tool result -> final answer.
async fn tool_roundtrip(model: &OpenAiModel) -> Outcome {
    let user = Message::user("What is the weather in Paris? Use the get_weather tool.");
    let mut req = base_request(vec![user.clone()]);
    req.tools = vec![weather_tool()];
    let resp = match model.invoke(&(), req).await {
        Ok(r) => r,
        Err(e) => return fail("tool-roundtrip", format!("turn 1: {e}")),
    };
    let Some(call) = first_call(&resp).cloned() else {
        return fail("tool-roundtrip", "turn 1 produced no tool call");
    };
    let assistant = Message::Assistant(AssistantMessage {
        id: None,
        content: resp.message.content.clone(),
        tool_calls: resp.message.tool_calls.clone(),
        usage: None,
    });
    let mut req2 = base_request(vec![
        user,
        assistant,
        Message::tool(call.id.clone(), r#"{"temp_c": 22, "condition": "sunny"}"#),
    ]);
    req2.tools = vec![weather_tool()];
    match model.invoke(&(), req2).await {
        Ok(final_resp) => {
            let text = msg_text(&final_resp.message).to_lowercase();
            if text.contains("22") || text.contains("sunny") {
                ok("tool-roundtrip", text.chars().take(80).collect::<String>())
            } else if !final_resp.message.tool_calls.is_empty() {
                // A second identical tool call means the model did not accept
                // the tool result we sent back.
                fail(
                    "tool-roundtrip",
                    "model re-requested the tool instead of answering",
                )
            } else {
                fail("tool-roundtrip", format!("answer ignored result: {text}"))
            }
        }
        Err(e) => fail("tool-roundtrip", format!("turn 2: {e}")),
    }
}

/// 4. Forced named tool choice (`tool_choice = {"type":"function",...}`).
async fn forced_tool_choice(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user("Tell me about the weather in Tokyo.")]);
    req.tools = vec![weather_tool()];
    req.tool_choice = ToolChoice::Tool("get_weather".to_string());
    match model.invoke(&(), req).await {
        Ok(resp) => match first_call(&resp) {
            Some(call) => ok("forced-tool-choice", format!("args={}", call.arguments)),
            None => fail("forced-tool-choice", "no tool call despite forced choice"),
        },
        Err(e) => fail("forced-tool-choice", e.to_string()),
    }
}

/// 5. `tool_choice = "required"`.
async fn required_tool_choice(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user("Tell me about the weather in Tokyo.")]);
    req.tools = vec![weather_tool()];
    req.tool_choice = ToolChoice::Required;
    match model.invoke(&(), req).await {
        Ok(resp) => match first_call(&resp) {
            Some(call) => ok("required-tool-choice", format!("args={}", call.arguments)),
            None => fail("required-tool-choice", "no tool call despite required"),
        },
        Err(e) => fail("required-tool-choice", e.to_string()),
    }
}

/// 6. Zero-argument tool.
async fn zero_arg_tool(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "What time is it? Use the get_time tool.",
    )]);
    req.tools = vec![time_tool()];
    match model.invoke(&(), req).await {
        Ok(resp) => match first_call(&resp) {
            Some(call) if call.arguments.is_object() => {
                ok("zero-arg-tool", format!("args={}", call.arguments))
            }
            Some(call) => fail(
                "zero-arg-tool",
                format!("non-object args: {}", call.arguments),
            ),
            None => fail("zero-arg-tool", "no tool call"),
        },
        Err(e) => fail("zero-arg-tool", e.to_string()),
    }
}

/// 7. Parallel tool calls in one turn.
async fn parallel_tools(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "Get the weather for BOTH Paris and Tokyo. Call get_weather once per city, \
         both calls in the same turn.",
    )]);
    req.tools = vec![weather_tool()];
    match model.invoke(&(), req).await {
        Ok(resp) => {
            let n = resp.message.tool_calls.len();
            let ids: Vec<&str> = resp
                .message
                .tool_calls
                .iter()
                .map(|c| c.id.as_str())
                .collect();
            let unique = ids.iter().collect::<std::collections::HashSet<_>>().len();
            if n >= 2 && unique == n {
                ok("parallel-tools", format!("{n} calls, ids={ids:?}"))
            } else if n >= 2 {
                fail("parallel-tools", format!("duplicate ids: {ids:?}"))
            } else {
                // Small models often serialize calls across turns; only flag
                // this as informational, not a harness bug.
                ok(
                    "parallel-tools",
                    format!("model made {n} call(s) (model behavior)"),
                )
            }
        }
        Err(e) => fail("parallel-tools", e.to_string()),
    }
}

/// 8. Streaming plain text.
async fn streaming_text(model: &OpenAiModel) -> Outcome {
    let req = base_request(vec![Message::user("Count from 1 to 5, digits only.")]);
    match model.stream(&(), req).await {
        Ok(mut stream) => {
            let mut deltas = 0usize;
            let mut completed = None;
            let mut failed = None;
            while let Some(item) = stream.next().await {
                match item {
                    ModelStreamItem::MessageDelta(_) => deltas += 1,
                    ModelStreamItem::Completed(resp) => completed = Some(resp),
                    ModelStreamItem::Failed(e) => failed = Some(e),
                    ModelStreamItem::ProviderFailed(e) => failed = Some(e.message),
                    _ => {}
                }
            }
            match (completed, failed) {
                (Some(resp), None) if !msg_text(&resp.message).trim().is_empty() => ok(
                    "streaming-text",
                    format!("{deltas} deltas, text={:?}", msg_text(&resp.message)),
                ),
                (Some(_), None) => fail("streaming-text", "completed with empty text"),
                (_, Some(e)) => fail("streaming-text", e),
                (None, None) => fail("streaming-text", "stream ended without Completed"),
            }
        }
        Err(e) => fail("streaming-text", e.to_string()),
    }
}

/// 9. Streaming with tool calls.
async fn streaming_tools(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "What is the weather in Paris? Use the get_weather tool.",
    )]);
    req.tools = vec![weather_tool()];
    match model.stream(&(), req).await {
        Ok(mut stream) => {
            let mut completed = None;
            let mut failed = None;
            while let Some(item) = stream.next().await {
                match item {
                    ModelStreamItem::Completed(resp) => completed = Some(resp),
                    ModelStreamItem::Failed(e) => failed = Some(e),
                    ModelStreamItem::ProviderFailed(e) => failed = Some(e.message),
                    _ => {}
                }
            }
            match (completed, failed) {
                (Some(resp), None) => match first_call(&resp) {
                    Some(call) if call.arguments.get("city").is_some() => ok(
                        "streaming-tools",
                        format!("id={:?} args={}", call.id, call.arguments),
                    ),
                    Some(call) => fail("streaming-tools", format!("bad args: {}", call.arguments)),
                    None => fail(
                        "streaming-tools",
                        format!(
                            "no tool call; finish={:?} text={:?}",
                            resp.finish_reason,
                            msg_text(&resp.message)
                                .chars()
                                .take(120)
                                .collect::<String>()
                        ),
                    ),
                },
                (_, Some(e)) => fail("streaming-tools", e),
                (None, None) => fail("streaming-tools", "stream ended without Completed"),
            }
        }
        Err(e) => fail("streaming-tools", e.to_string()),
    }
}

/// 10. JSON-object response format.
async fn json_object(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "Return a JSON object with keys \"name\" (string) and \"age\" (number) \
         for a fictional person.",
    )]);
    req.response_format = Some(ResponseFormat::JsonObject);
    match model.invoke(&(), req).await {
        Ok(resp) => {
            let text = msg_text(&resp.message);
            match serde_json::from_str::<serde_json::Value>(text.trim()) {
                Ok(v) if v.is_object() => {
                    ok("json-object", text.chars().take(80).collect::<String>())
                }
                _ => fail("json-object", format!("not a JSON object: {text:?}")),
            }
        }
        Err(e) => fail("json-object", e.to_string()),
    }
}

/// 11. JSON-schema response format (sends `strict: true` on the wire).
async fn json_schema(model: &OpenAiModel) -> Outcome {
    let mut req = base_request(vec![Message::user(
        "Rate this review from 1-5 and classify sentiment: \"Great product!\"",
    )]);
    req.response_format = Some(ResponseFormat::JsonSchema {
        name: "review".to_string(),
        schema: json!({
            "type": "object",
            "properties": {
                "score": { "type": "integer" },
                "sentiment": { "type": "string" }
            },
            "required": ["score", "sentiment"],
            "additionalProperties": false
        }),
    });
    match model.invoke(&(), req).await {
        Ok(resp) => {
            let text = msg_text(&resp.message);
            match serde_json::from_str::<serde_json::Value>(text.trim()) {
                Ok(v) if v.get("score").is_some() => {
                    ok("json-schema", text.chars().take(80).collect::<String>())
                }
                _ => fail("json-schema", format!("schema not honored: {text:?}")),
            }
        }
        Err(e) => fail("json-schema", e.to_string()),
    }
}

/// 12. Reasoning models must not leak `<think>` blocks into final text.
async fn thinking_leak(model: &OpenAiModel) -> Outcome {
    let req = base_request(vec![Message::user(
        "What is 17 * 23? Reply with just the number.",
    )]);
    match model.invoke(&(), req).await {
        Ok(resp) => {
            let text = msg_text(&resp.message);
            if text.contains("<think>") || text.contains("</think>") {
                fail(
                    "thinking-leak",
                    format!(
                        "<think> leaked into text: {:?}",
                        text.chars().take(120).collect::<String>()
                    ),
                )
            } else {
                ok("thinking-leak", text.chars().take(60).collect::<String>())
            }
        }
        Err(e) => fail("thinking-leak", e.to_string()),
    }
}

/// 13. Streaming must not leak `<think>` into content deltas either.
async fn streaming_thinking_leak(model: &OpenAiModel) -> Outcome {
    let req = base_request(vec![Message::user(
        "What is 12 + 30? Reply with just the number.",
    )]);
    match model.stream(&(), req).await {
        Ok(mut stream) => {
            let mut content = String::new();
            let mut completed_text = String::new();
            while let Some(item) = stream.next().await {
                match item {
                    ModelStreamItem::MessageDelta(d) => content.push_str(&d.text),
                    ModelStreamItem::Completed(resp) => {
                        completed_text = msg_text(&resp.message).to_string();
                    }
                    _ => {}
                }
            }
            if content.contains("<think>") || completed_text.contains("<think>") {
                fail(
                    "streaming-thinking-leak",
                    format!(
                        "<think> leaked; deltas={:?}",
                        content.chars().take(120).collect::<String>()
                    ),
                )
            } else {
                ok(
                    "streaming-thinking-leak",
                    completed_text.chars().take(60).collect::<String>(),
                )
            }
        }
        Err(e) => fail("streaming-thinking-leak", e.to_string()),
    }
}

#[tokio::main]
async fn main() -> tinyagents::Result<()> {
    dotenvy::dotenv().ok();
    let model = OpenAiModel::from_env()?;
    println!("=== local model compatibility probe ===");
    println!(
        "endpoint: {}  model: {}\n",
        std::env::var("OPENAI_BASE_URL").unwrap_or_default(),
        model.model()
    );

    let outcomes = vec![
        basic_text(&model).await,
        tool_call(&model).await,
        tool_roundtrip(&model).await,
        forced_tool_choice(&model).await,
        required_tool_choice(&model).await,
        zero_arg_tool(&model).await,
        parallel_tools(&model).await,
        streaming_text(&model).await,
        streaming_tools(&model).await,
        json_object(&model).await,
        json_schema(&model).await,
        thinking_leak(&model).await,
        streaming_thinking_leak(&model).await,
    ];

    let mut failures = 0;
    for o in &outcomes {
        let mark = if o.passed { "PASS" } else { "FAIL" };
        if !o.passed {
            failures += 1;
        }
        println!("[{mark}] {:>24} : {}", o.name, o.detail);
    }
    println!("\n{} scenario(s), {} failure(s)", outcomes.len(), failures);
    Ok(())
}
