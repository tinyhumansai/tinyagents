//! Serde-mapping unit tests for the OpenAI provider.
//!
//! These tests exercise the request/response translation in isolation and never
//! touch the network: the request side asserts the JSON shape produced for a
//! representative [`ModelRequest`], and the response side feeds hand-written
//! OpenAI-shaped JSON through [`parse_response`].

use serde_json::json;

use super::*;
use crate::harness::message::Message;
use crate::harness::model::{
    ChatModel, ModelRequest, ModelStreamItem, ProviderError, ResponseFormat, StreamAccumulator,
    ToolChoice,
};
use crate::harness::providers::{ProviderKind, ProviderSpec};
use crate::harness::tool::ToolSchema;

/// Builds a model with a fixed key/model so translation output is deterministic.
fn model() -> OpenAiModel {
    OpenAiModel::new("test-key").with_model("gpt-4.1-mini")
}

#[test]
fn translates_request_to_openai_json_shape() {
    let request = ModelRequest::new(vec![
        Message::system("You are a sentiment classifier."),
        Message::user("I love this product!"),
    ])
    .with_tools(vec![ToolSchema::new(
        "get_weather",
        "Look up the weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    )])
    .with_tool_choice(ToolChoice::Required)
    .with_response_format(ResponseFormat::json_schema(
        "sentiment",
        json!({
            "type": "object",
            "properties": {
                "sentiment": { "type": "string" },
                "score": { "type": "number" }
            },
            "required": ["sentiment", "score"]
        }),
    ))
    .with_temperature(0.2)
    .with_top_p(0.8)
    .with_stop_sequences(["END"])
    .with_seed(7)
    .with_max_tokens(256);

    let body = model().translate_request(&request).unwrap();
    let value = serde_json::to_value(&body).unwrap();

    assert_eq!(value["model"], json!("gpt-4.1-mini"));

    // Messages: roles and content map straight through.
    assert_eq!(value["messages"][0]["role"], json!("system"));
    assert_eq!(
        value["messages"][0]["content"],
        json!("You are a sentiment classifier.")
    );
    assert_eq!(value["messages"][1]["role"], json!("user"));
    assert_eq!(
        value["messages"][1]["content"],
        json!("I love this product!")
    );

    // Tools: ToolSchema -> {type:"function", function:{...}}.
    assert_eq!(value["tools"][0]["type"], json!("function"));
    assert_eq!(value["tools"][0]["function"]["name"], json!("get_weather"));
    assert_eq!(
        value["tools"][0]["function"]["description"],
        json!("Look up the weather for a city.")
    );
    assert_eq!(
        value["tools"][0]["function"]["parameters"]["properties"]["city"]["type"],
        json!("string")
    );

    // tool_choice: Required -> "required".
    assert_eq!(value["tool_choice"], json!("required"));

    // response_format: JsonSchema -> json_schema with strict:true.
    assert_eq!(value["response_format"]["type"], json!("json_schema"));
    assert_eq!(
        value["response_format"]["json_schema"]["name"],
        json!("sentiment")
    );
    assert_eq!(
        value["response_format"]["json_schema"]["strict"],
        json!(true)
    );
    assert_eq!(
        value["response_format"]["json_schema"]["schema"]["properties"]["score"]["type"],
        json!("number")
    );

    // Sampling params.
    assert_eq!(value["temperature"], json!(0.2));
    assert_eq!(value["top_p"], json!(0.8));
    assert_eq!(value["max_tokens"], json!(256));
    assert_eq!(value["stop"], json!(["END"]));
    assert_eq!(value["seed"], json!(7));
}

#[test]
fn translates_provider_options_for_local_openai_compatible_models() {
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_temperature(0.1)
        .with_provider_options(json!({
            "temperature": 9.9,
            "stream": true,
            "hotness": "spicy",
            "reasoning": { "effort": "high" },
            "options": {
                "num_ctx": 8192,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "mirostat": 2
            },
            "keep_alive": "10m"
        }));

    let value = serde_json::to_value(
        OpenAiModel::ollama()
            .with_model("qwen2.5")
            .translate_request(&request)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(value["model"], json!("qwen2.5"));
    assert_eq!(value["temperature"], json!(0.1));
    assert!(value.get("stream").is_none());
    assert_eq!(value["hotness"], json!("spicy"));
    assert_eq!(value["reasoning"]["effort"], json!("high"));
    assert_eq!(value["options"]["num_ctx"], json!(8192));
    assert_eq!(value["options"]["top_k"], json!(40));
    assert_eq!(value["options"]["repeat_penalty"], json!(1.1));
    assert_eq!(value["options"]["mirostat"], json!(2));
    assert_eq!(value["keep_alive"], json!("10m"));
}

#[test]
fn rejects_non_object_provider_options_for_openai_compatible_models() {
    let request =
        ModelRequest::new(vec![Message::user("hi")]).with_provider_options(json!(["top_k", 40]));

    let error = model().translate_request(&request).unwrap_err();

    assert!(matches!(error, TinyAgentsError::Validation(_)));
    assert!(
        error
            .to_string()
            .contains("provider_options for OpenAI-compatible providers must be a JSON object")
    );
}

#[test]
fn translates_named_tool_choice_and_omits_when_no_tools() {
    // Named tool -> structured object.
    let with_tool = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("t", "d", json!({}))])
        .with_tool_choice(ToolChoice::Tool("t".to_string()));
    let value = serde_json::to_value(model().translate_request(&with_tool).unwrap()).unwrap();
    assert_eq!(
        value["tool_choice"],
        json!({ "type": "function", "function": { "name": "t" } })
    );

    // No declared tools -> tool_choice (and tools) omitted entirely.
    let no_tools = ModelRequest::new(vec![Message::user("hi")]);
    let value = serde_json::to_value(model().translate_request(&no_tools).unwrap()).unwrap();
    assert!(value.get("tool_choice").is_none());
    assert!(value.get("tools").is_none());
    assert!(value.get("response_format").is_none());
}

#[test]
fn translates_assistant_tool_calls_to_stringified_arguments() {
    let request = ModelRequest::new(vec![
        Message::user("What is the weather in Paris?"),
        Message::Assistant(AssistantMessage {
            id: Some("msg-1".to_string()),
            content: Vec::new(),
            tool_calls: vec![ToolCall {
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                arguments: json!({ "city": "Paris" }),
            }],
            usage: None,
        }),
        Message::tool("call-1", "sunny, 21C"),
    ]);

    let value = serde_json::to_value(model().translate_request(&request).unwrap()).unwrap();

    // Assistant message: no content, one tool call with stringified arguments.
    let assistant = &value["messages"][1];
    assert_eq!(assistant["role"], json!("assistant"));
    assert!(assistant.get("content").is_none());
    assert_eq!(assistant["tool_calls"][0]["id"], json!("call-1"));
    assert_eq!(assistant["tool_calls"][0]["type"], json!("function"));
    assert_eq!(
        assistant["tool_calls"][0]["function"]["name"],
        json!("get_weather")
    );
    // Arguments are a JSON *string*.
    assert_eq!(
        assistant["tool_calls"][0]["function"]["arguments"],
        json!("{\"city\":\"Paris\"}")
    );

    // Tool result message carries the correlation id.
    let tool = &value["messages"][2];
    assert_eq!(tool["role"], json!("tool"));
    assert_eq!(tool["tool_call_id"], json!("call-1"));
    assert_eq!(tool["content"], json!("sunny, 21C"));
}

#[test]
fn parses_openai_response_with_content_tool_call_and_usage() {
    // Hand-written OpenAI-shaped response JSON.
    let body = json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call-99",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 8,
            "total_tokens": 50,
            "prompt_tokens_details": { "cached_tokens": 30 }
        }
    });

    let response = parse_response(body.clone()).unwrap();

    // Message id + text content.
    assert_eq!(response.message.id.as_deref(), Some("chatcmpl-abc123"));
    assert_eq!(response.text(), "Let me check the weather for you.");

    // Tool call parsed back into structured arguments.
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-99");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, json!({ "city": "Paris" }));

    // Finish reason.
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));

    // Usage mapping, including cached -> cache_read_tokens.
    let usage = response.usage.expect("usage present");
    assert_eq!(usage.input_tokens, 42);
    assert_eq!(usage.output_tokens, 8);
    assert_eq!(usage.total_tokens, 50);
    assert_eq!(usage.cache_read_tokens, 30);

    // Raw JSON preserved verbatim.
    assert_eq!(response.raw, Some(body));
}

#[test]
fn parse_response_errors_on_invalid_tool_argument_json() {
    let body = json!({
        "id": "chatcmpl-badargs",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-bad",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\":"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let err = parse_response(body).expect_err("invalid arguments must fail");
    let message = err.to_string();
    assert!(matches!(err, TinyAgentsError::Model(_)));
    assert!(message.contains("call-bad"), "{message}");
    assert!(message.contains("lookup"), "{message}");
    assert!(message.contains("raw arguments"), "{message}");
}

#[test]
fn parses_text_only_response_without_usage_details() {
    let body = json!({
        "id": "chatcmpl-xyz",
        "choices": [
            {
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7
        }
    });

    let response = parse_response(body).unwrap();
    assert_eq!(response.text(), "Hello!");
    assert!(response.tool_calls().is_empty());
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.cache_read_tokens, 0);
}

#[test]
fn parses_empty_tool_arguments_as_empty_object() {
    // Some compat backends send an empty arguments string for a zero-argument
    // tool call; it must map to `{}`, not fail as malformed JSON.
    let body = json!({
        "id": "chatcmpl-noargs",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-empty",
                            "type": "function",
                            "function": { "name": "ping", "arguments": "" }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });

    let response = parse_response(body).unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "ping");
    assert_eq!(calls[0].arguments, json!({}));
}

#[tokio::test]
async fn sse_stream_empty_tool_arguments_reconstruct_as_empty_object() {
    // A streamed tool call whose only arguments fragment is empty. The merged
    // call must carry `{}` rather than fail terminally.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-x\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "ping");
    assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn parse_response_errors_on_empty_choices() {
    let body = json!({ "id": "x", "choices": [] });
    let err = parse_response(body).unwrap_err();
    assert!(matches!(err, TinyAgentsError::Model(_)));
}

#[test]
fn compatible_presets_set_base_url_and_default_model() {
    let deepseek = OpenAiModel::deepseek("k");
    assert_eq!(deepseek.provider(), "deepseek");
    assert_eq!(deepseek.base_url(), "https://api.deepseek.com/v1");
    assert_eq!(deepseek.model(), "deepseek-chat");

    let anthropic = OpenAiModel::anthropic("k");
    assert_eq!(anthropic.provider(), "anthropic");
    assert_eq!(anthropic.base_url(), "https://api.anthropic.com/v1");
    assert_eq!(anthropic.model(), "claude-3-5-sonnet-latest");

    let ollama = OpenAiModel::ollama();
    assert_eq!(ollama.provider(), "ollama");
    assert_eq!(ollama.base_url(), "http://localhost:11434/v1");
    assert_eq!(ollama.model(), "llama3.2");

    // A custom model override still wins over the preset default.
    let custom = OpenAiModel::groq("k").with_model("mixtral-8x7b");
    assert_eq!(custom.base_url(), "https://api.groq.com/openai/v1");
    assert_eq!(custom.model(), "mixtral-8x7b");

    // Fully generic compatible endpoint.
    let generic = OpenAiModel::compatible("k", "https://example.test/v1/", "my-model");
    assert_eq!(generic.provider(), "openai");
    assert_eq!(generic.base_url(), "https://example.test/v1");
    assert_eq!(generic.model(), "my-model");

    let named = OpenAiModel::compatible_provider(
        "custom",
        "k",
        "https://custom.example/v1/",
        "custom-model",
    );
    assert_eq!(named.provider(), "custom");
    assert_eq!(named.base_url(), "https://custom.example/v1");
    assert_eq!(named.model(), "custom-model");
}

#[test]
fn provider_spec_builds_compatible_model() {
    let spec = ProviderSpec::for_kind(ProviderKind::Ollama).with_model("qwen2.5");
    let model = OpenAiModel::from_spec(spec, "ignored").unwrap();

    assert_eq!(model.provider(), "ollama");
    assert_eq!(model.base_url(), "http://localhost:11434/v1");
    assert_eq!(model.model(), "qwen2.5");
    assert_eq!(
        <OpenAiModel as ChatModel<()>>::profile(&model)
            .unwrap()
            .provider
            .as_deref(),
        Some("ollama")
    );
}

#[test]
fn provider_failed_stream_item_finishes_as_model_error() {
    let mut accumulator = StreamAccumulator::new();
    accumulator.push(&ModelStreamItem::ProviderFailed(ProviderError {
        provider: "groq".to_string(),
        model: Some("llama-3.3-70b-versatile".to_string()),
        status: Some(429),
        code: Some("rate_limit".to_string()),
        message: "too many requests".to_string(),
        retryable: true,
        raw: None,
    }));

    let error = accumulator.finish().unwrap_err().to_string();
    assert!(error.contains("groq provider error (rate_limit): too many requests"));
}

#[tokio::test]
async fn sse_stream_parses_text_tool_calls_and_usage() {
    use futures::StreamExt;

    // Two text fragments, a tool-call split across chunks, then usage + [DONE].
    // Chunk boundaries deliberately split a `data:` line so the buffer-join path
    // is exercised.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arg".to_vec(),
        b"uments\":\"{\\\"q\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"42}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let bytes = futures::stream::iter(raw.into_iter().map(Ok::<Vec<u8>, TinyAgentsError>));

    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: std::collections::VecDeque::new(),
        acc: OpenAiStreamAcc::default(),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        terminal_emitted: false,
    };
    let items: Vec<ModelStreamItem> = futures::stream::unfold(state, sse_next).collect().await;

    // First item is Started; last is Completed.
    assert!(matches!(items.first(), Some(ModelStreamItem::Started)));
    assert!(matches!(items.last(), Some(ModelStreamItem::Completed(_))));

    // Two text message deltas reconstruct "Hello".
    let text: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");

    // Tool-call argument fragments streamed as ToolCallDelta items.
    let tool_deltas = items
        .iter()
        .filter(|item| matches!(item, ModelStreamItem::ToolCallDelta(_)))
        .count();
    assert_eq!(tool_deltas, 2);

    // A usage delta was emitted.
    assert!(
        items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::UsageDelta(_)))
    );

    // The merged final response carries the reassembled tool call and usage.
    let merged = StreamAccumulator::new();
    let mut merged = merged;
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "Hello");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-1");
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, json!({ "q": 42 }));
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.usage.unwrap().total_tokens, 8);
}

#[tokio::test]
async fn sse_stream_invalid_tool_argument_json_fails_terminally() {
    use futures::StreamExt;

    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-bad\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let bytes = futures::stream::iter(raw.into_iter().map(Ok::<Vec<u8>, TinyAgentsError>));

    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: std::collections::VecDeque::new(),
        acc: OpenAiStreamAcc::default(),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        terminal_emitted: false,
    };
    let items: Vec<ModelStreamItem> = futures::stream::unfold(state, sse_next).collect().await;

    let failed = items
        .iter()
        .find_map(|item| match item {
            ModelStreamItem::ProviderFailed(error) => Some(error),
            _ => None,
        })
        .expect("invalid arguments should emit ProviderFailed");
    assert_eq!(failed.code.as_deref(), Some("invalid_tool_arguments"));
    assert!(!failed.retryable);
    assert!(failed.message.contains("call-bad"), "{}", failed.message);
    assert!(failed.message.contains("lookup"), "{}", failed.message);

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let err = merged
        .finish()
        .expect_err("provider failure must reach accumulator");
    assert!(err.to_string().contains("invalid_tool_arguments"));
}

/// Drives an SSE byte stream through the parser and returns every item.
async fn collect_sse(raw: Vec<Vec<u8>>) -> Vec<ModelStreamItem> {
    use futures::StreamExt;

    let bytes = futures::stream::iter(raw.into_iter().map(Ok::<Vec<u8>, TinyAgentsError>));
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: std::collections::VecDeque::new(),
        acc: OpenAiStreamAcc::default(),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        terminal_emitted: false,
    };
    futures::stream::unfold(state, sse_next).collect().await
}

#[tokio::test]
async fn sse_stream_reassembles_multibyte_char_split_across_chunks() {
    // A 4-byte emoji in the content payload, split down the middle across two
    // network chunks. A lossy per-chunk decode would corrupt it into U+FFFD
    // replacement characters; the byte buffer must reassemble it first.
    let line = "data: {\"choices\":[{\"delta\":{\"content\":\"hi😀\"}}]}\n\n";
    let bytes = line.as_bytes();
    let split = line.find('😀').unwrap() + 2; // inside the 4-byte sequence
    let raw: Vec<Vec<u8>> = vec![
        bytes[..split].to_vec(),
        bytes[split..].to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "hi😀");
}

#[tokio::test]
async fn sse_stream_drains_final_line_without_trailing_newline() {
    // The provider ends the stream with a final `data:` event that has no
    // trailing newline and no `[DONE]` sentinel. The leftover buffer must be
    // drained at EOF so the last fragment is not dropped.
    let raw: Vec<Vec<u8>> =
        vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}".to_vec()];

    let items = collect_sse(raw).await;

    assert!(matches!(items.last(), Some(ModelStreamItem::Completed(_))));
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "tail");
}

#[tokio::test]
async fn sse_stream_surfaces_mid_stream_error_payload() {
    // The provider streams a text delta, then an `{"error": ...}` payload
    // instead of a chunk. It must surface as a terminal ProviderFailed rather
    // than be swallowed as an empty chunk.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".to_vec(),
        b"data: {\"error\":{\"message\":\"upstream exploded\",\"code\":\"server_error\"}}\n\n"
            .to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    let failed = items
        .iter()
        .find_map(|item| match item {
            ModelStreamItem::ProviderFailed(error) => Some(error),
            _ => None,
        })
        .expect("mid-stream error should emit ProviderFailed");
    assert_eq!(failed.code.as_deref(), Some("server_error"));
    assert!(failed.message.contains("upstream exploded"));
    // No terminal Completed is emitted after the failure.
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let err = merged
        .finish()
        .expect_err("stream error must reach accumulator");
    assert!(err.to_string().contains("upstream exploded"));
}

#[tokio::test]
async fn sse_stream_correlates_indexless_parallel_tool_calls_by_id() {
    // A compat backend that omits `index` entirely and interleaves two parallel
    // tool calls, correlating fragments only by `id`. Without id-based slotting
    // both would collapse onto slot 0; the streamed delta ids must also match the
    // final reconstructed call ids.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call-a\",\"function\":{\"name\":\"alpha\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call-b\",\"function\":{\"name\":\"beta\",\"arguments\":\"{\\\"y\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call-a\",\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call-b\",\"function\":{\"arguments\":\"2}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    // Every streamed tool-call delta id must be a real call id, never a slot-0
    // collapse.
    let delta_ids: Vec<String> = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::ToolCallDelta(delta) => Some(delta.call_id.clone()),
            _ => None,
        })
        .collect();
    assert!(delta_ids.contains(&"call-a".to_string()));
    assert!(delta_ids.contains(&"call-b".to_string()));

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    let calls = response.tool_calls();
    assert_eq!(
        calls.len(),
        2,
        "parallel calls must not collapse: {calls:?}"
    );
    assert_eq!(calls[0].id, "call-a");
    assert_eq!(calls[0].name, "alpha");
    assert_eq!(calls[0].arguments, json!({ "x": 1 }));
    assert_eq!(calls[1].id, "call-b");
    assert_eq!(calls[1].name, "beta");
    assert_eq!(calls[1].arguments, json!({ "y": 2 }));
}

#[tokio::test]
async fn sse_stream_indexless_fallback_ids_match_between_delta_and_final() {
    // No `index` and no `id` at all (arguments-only continuation on the same
    // slot). The synthetic fallback id streamed in the delta must equal the id on
    // the final reconstructed call.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"name\":\"solo\",\"arguments\":\"{\\\"n\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"7}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    let delta_id = items
        .iter()
        .find_map(|item| match item {
            ModelStreamItem::ToolCallDelta(delta) => Some(delta.call_id.clone()),
            _ => None,
        })
        .expect("a tool-call delta is emitted");

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, delta_id, "delta id must match final call id");
    assert_eq!(calls[0].name, "solo");
    assert_eq!(calls[0].arguments, json!({ "n": 7 }));
}

#[test]
fn user_image_blocks_render_as_content_parts() {
    use crate::harness::message::{ContentBlock, ImageRef, UserMessage};

    let request = ModelRequest::new(vec![Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("What is in this image?".to_string()),
            ContentBlock::Image(ImageRef {
                url: "https://example.test/cat.png".to_string(),
                mime_type: Some("image/png".to_string()),
            }),
        ],
    })]);

    let value = serde_json::to_value(model().translate_request(&request).unwrap()).unwrap();
    let content = &value["messages"][0]["content"];

    // Content is an array of parts, not a dropped/plain string.
    assert!(content.is_array(), "expected content parts, got {content}");
    assert_eq!(content[0]["type"], json!("text"));
    assert_eq!(content[0]["text"], json!("What is in this image?"));
    assert_eq!(content[1]["type"], json!("image_url"));
    assert_eq!(
        content[1]["image_url"]["url"],
        json!("https://example.test/cat.png")
    );
}

#[test]
fn text_only_user_message_stays_a_plain_string() {
    // The common text-only case keeps its historical plain-string wire shape.
    let request = ModelRequest::new(vec![Message::user("hi")]);
    let value = serde_json::to_value(model().translate_request(&request).unwrap()).unwrap();
    assert_eq!(value["messages"][0]["content"], json!("hi"));
}

#[test]
fn provider_extension_block_fails_closed_instead_of_dropping() {
    use crate::harness::message::{ContentBlock, UserMessage};

    let request = ModelRequest::new(vec![Message::User(UserMessage {
        content: vec![ContentBlock::ProviderExtension(json!({ "opaque": true }))],
    })]);

    let error = model().translate_request(&request).unwrap_err();
    assert!(matches!(error, TinyAgentsError::Validation(_)));
    assert!(error.to_string().contains("provider-extension"));
}

#[test]
fn routes_max_tokens_to_max_completion_tokens_for_o_series() {
    // o-series reasoning models reject `max_tokens` and require
    // `max_completion_tokens`.
    let request = ModelRequest::new(vec![Message::user("hi")]).with_max_tokens(128);
    let model = OpenAiModel::new("k").with_model("o3-mini");
    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();

    assert!(value.get("max_tokens").is_none());
    assert_eq!(value["max_completion_tokens"], json!(128));
}

#[test]
fn keeps_max_tokens_for_classic_models() {
    let request = ModelRequest::new(vec![Message::user("hi")]).with_max_tokens(128);
    let value = serde_json::to_value(model().translate_request(&request).unwrap()).unwrap();

    assert_eq!(value["max_tokens"], json!(128));
    assert!(value.get("max_completion_tokens").is_none());
}

#[test]
fn request_timeout_prefers_explicit_override() {
    // An explicit per-request timeout wins for both unary and streaming calls.
    assert_eq!(
        request_timeout(Some(1_500), false),
        Some(Duration::from_millis(1_500))
    );
    assert_eq!(
        request_timeout(Some(1_500), true),
        Some(Duration::from_millis(1_500))
    );
}

#[test]
fn request_timeout_defaults_by_call_kind() {
    // Unary calls fall back to a sane overall default; streaming calls get no
    // overall cap so a long stream is not truncated.
    assert_eq!(
        request_timeout(None, false),
        Some(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS))
    );
    assert_eq!(request_timeout(None, true), None);
}

#[test]
fn from_env_errors_when_api_key_missing() {
    // Snapshot and clear the key so the missing-key path is exercised
    // deterministically, then restore the prior value.
    let previous = std::env::var("OPENAI_API_KEY").ok();
    // SAFETY: tests in this module are the only place this crate mutates
    // OPENAI_API_KEY; the value is restored before returning.
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }

    let result = OpenAiModel::from_env();

    if let Some(value) = previous {
        unsafe {
            std::env::set_var("OPENAI_API_KEY", value);
        }
    }

    assert!(matches!(result, Err(TinyAgentsError::Validation(_))));
}

#[test]
fn parses_model_listing_envelope() {
    // The `GET /models` shape shared by OpenAI and OpenAI-compatible providers.
    let body = json!({
        "object": "list",
        "data": [
            { "id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "openai" },
            { "id": "llama3.1", "object": "model" }
        ]
    });
    let listing: ModelListWire = serde_json::from_value(body).unwrap();
    assert_eq!(listing.data.len(), 2);
    assert_eq!(listing.data[0].id, "gpt-4o");
    assert_eq!(listing.data[0].created, Some(1715367049));
    assert_eq!(listing.data[0].owned_by.as_deref(), Some("openai"));
    // Providers that omit optional fields still parse (id only).
    assert_eq!(listing.data[1].id, "llama3.1");
    assert_eq!(listing.data[1].created, None);
    assert_eq!(listing.data[1].owned_by, None);
}
