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
                invalid: None,
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
            "prompt_tokens_details": { "cached_tokens": 30 },
            "completion_tokens_details": { "reasoning_tokens": 6 }
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
    assert_eq!(usage.reasoning_tokens, 6);

    // Raw JSON preserved verbatim.
    assert_eq!(response.raw, Some(body));
}

#[test]
fn parse_response_marks_invalid_tool_argument_json_instead_of_failing() {
    // Malformed argument JSON must not fail the whole model call (which made
    // small local models appear "broken"). Instead the call is surfaced as an
    // `invalid` ToolCall with the raw arguments preserved so the agent loop can
    // feed the error back to the model.
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

    let response = parse_response(body).expect("malformed args must not fail the call");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-bad");
    assert_eq!(calls[0].name, "lookup");
    // Raw arguments preserved verbatim as a JSON string value.
    assert_eq!(calls[0].arguments, json!("{\"q\":"));
    let reason = calls[0].invalid.as_deref().expect("call marked invalid");
    assert!(reason.contains("call-bad"), "{reason}");
    assert!(reason.contains("lookup"), "{reason}");
    assert!(reason.contains("raw arguments"), "{reason}");
}

#[test]
fn parses_id_less_tool_call_with_synthesized_fallback_id() {
    // Ollama's /v1 endpoint omitted the tool-call `id` entirely until v0.12.11,
    // and some servers omit `type`. Neither may fail deserialization; a
    // missing/empty id gets the same `tool-{index}` fallback the streaming path
    // uses so the agent loop can still correlate the result.
    let body = json!({
        "id": "chatcmpl-noid",
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [
                    // No `id`, no `type`.
                    { "function": { "name": "ping", "arguments": "{}" } },
                    // Explicit empty id is treated as absent.
                    { "id": "", "type": "function", "function": { "name": "pong", "arguments": "{\"n\":1}" } }
                ]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let response = parse_response(body).unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "tool-0");
    assert_eq!(calls[0].name, "ping");
    assert_eq!(calls[0].arguments, json!({}));
    assert_eq!(calls[1].id, "tool-1");
    assert_eq!(calls[1].name, "pong");
    assert_eq!(calls[1].arguments, json!({ "n": 1 }));
}

#[test]
fn parses_object_form_tool_arguments() {
    // Some OpenAI-compatible servers send `function.arguments` as a JSON object
    // instead of the OpenAI-standard stringified JSON. It must normalize to the
    // same parsed arguments, not fail the response.
    let body = json!({
        "id": "chatcmpl-obj",
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-obj",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let response = parse_response(body).unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, json!({ "city": "Paris" }));
    assert!(
        calls[0].invalid.is_none(),
        "object args are valid, not invalid"
    );
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
fn total_tokens_falls_back_to_prompt_plus_completion_when_omitted() {
    // Some OpenAI-compatible backends omit `total_tokens` entirely; it must
    // not silently deserialize to a misleading `0` when prompt/completion
    // tokens were clearly reported.
    let body = json!({
        "id": "chatcmpl-omit",
        "choices": [
            {
                "message": { "role": "assistant", "content": "Hi!" },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2
        }
    });

    let response = parse_response(body).unwrap();
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.total_tokens, 7);
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
fn parse_error_body_classifies_retryability_by_http_status() {
    // Regression test: retry used to see every provider failure flattened
    // into `TinyAgentsError::Model(String)`, so it could not distinguish a
    // retryable 429 from a non-retryable 401 and retried both. The status
    // code alone must fully determine `ProviderError::retryable`.
    let m = model();

    let unauthorized = m.parse_error_body(
        401,
        r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}"#,
    );
    assert_eq!(unauthorized.status, Some(401));
    assert!(!unauthorized.retryable, "401 must not be retryable");

    let rate_limited = m.parse_error_body(
        429,
        r#"{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}"#,
    );
    assert_eq!(rate_limited.status, Some(429));
    assert!(rate_limited.retryable, "429 must be retryable");

    let server_error = m.parse_error_body(500, r#"{"error":{"message":"internal error"}}"#);
    assert!(server_error.retryable, "5xx must be retryable");

    // A 400 with "Stream must be set to true" — the common proxy-enforced
    // streaming check that triggers the invoke -> stream fallback.
    let stream_required = m.parse_error_body(
        400,
        r#"{"detail":"Stream must be set to true"}"#,
    );
    assert_eq!(stream_required.status, Some(400));
    assert!(
        stream_required.message.contains("Stream must be set to true"),
        "error message must contain the trigger phrase: got {}",
        stream_required.message
    );
}

#[test]
fn requires_streaming_flag_skips_non_streaming_attempt() {
    // The flag is `false` by default for all built-in presets.
    let default = OpenAiModel::new("k");
    assert!(!default.requires_streaming(), "default must be false");
    let ollama = OpenAiModel::ollama();
    assert!(!ollama.requires_streaming(), "ollama default must be false");

    // When the flag is set, `invoke` should skip the non-streaming wire
    // call. We verify the flag value and that it round-trips through
    // the builder.
    let flagged = OpenAiModel::new("k").with_requires_streaming(true);
    assert!(
        flagged.requires_streaming(),
        "with_requires_streaming(true) must set the field"
    );
    let unflagged = flagged.with_requires_streaming(false);
    assert!(
        !unflagged.requires_streaming(),
        "with_requires_streaming(false) must clear the field"
    );
}

#[test]
fn reasoning_tag_extraction_defaults_off_for_hosted_openai_only() {
    // Hosted OpenAI never emits inline `<think>` reasoning; unconditional
    // extraction there would silently strip legitimate content mentioning a
    // literal tag. The built-in default therefore only takes effect for
    // non-hosted base URLs, while an explicit override always wins.
    let hosted = OpenAiModel::new("k");
    assert!(
        hosted.effective_reasoning_tags().is_none(),
        "hosted default must not extract inline tags"
    );

    let local = OpenAiModel::compatible("k", "http://localhost:1234/v1", "m");
    assert!(
        local.effective_reasoning_tags().is_some(),
        "compat endpoints get extraction by default"
    );
    assert!(OpenAiModel::ollama().effective_reasoning_tags().is_some());

    let forced = OpenAiModel::new("k")
        .with_reasoning_tag_extraction(Some(ReasoningTagExtraction::default()));
    assert!(
        forced.effective_reasoning_tags().is_some(),
        "explicit override forces extraction on for the hosted base URL"
    );

    let disabled = OpenAiModel::ollama().with_reasoning_tag_extraction(None);
    assert!(
        disabled.effective_reasoning_tags().is_none(),
        "explicit None disables extraction everywhere"
    );
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
fn provider_failed_stream_item_finishes_as_provider_error() {
    // A streamed `ProviderFailed` must finish as `TinyAgentsError::Provider` with
    // the structured error preserved (status/code/`retryable`), so the retry
    // layer classifies it from `retryable` instead of the old behavior that
    // stringified it into `Model` and always retried it as transient.
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

    match accumulator.finish().unwrap_err() {
        TinyAgentsError::Provider(error) => {
            assert_eq!(error.provider, "groq");
            assert_eq!(error.status, Some(429));
            assert_eq!(error.code.as_deref(), Some("rate_limit"));
            assert_eq!(error.message, "too many requests");
            assert!(error.retryable);
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
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
    let bytes = futures::stream::iter(
        raw.into_iter()
            .map(|v| Ok::<bytes::Bytes, TinyAgentsError>(bytes::Bytes::from(v))),
    );

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
async fn sse_stream_preserves_reasoning_content_as_side_channel() {
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"reasoning\":\"carefully\"}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n"
            .to_vec(),
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12,\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    let reasoning: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.reasoning.clone()),
            _ => None,
        })
        .collect();
    let text: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "think carefully");
    assert_eq!(text, "answer");

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    assert_eq!(merged.reasoning(), "think carefully");
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "answer");
    // The merged response leads with the preserved (unsigned) thinking block.
    assert_eq!(
        response.message.content.first(),
        Some(&crate::harness::message::ContentBlock::Thinking {
            text: "think carefully".into(),
            signature: None,
        })
    );
    let usage = response.usage.unwrap();
    assert_eq!(usage.reasoning_tokens, 4);
}

#[tokio::test]
async fn sse_stream_invalid_tool_argument_json_reconstructs_as_invalid_call() {
    // A streamed tool call whose reassembled arguments are malformed must not
    // fail the stream terminally: it reconstructs as an `invalid` ToolCall (raw
    // arguments preserved) so the agent loop can feed the error back to the
    // model and the call still resolves instead of stalling the loop.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-bad\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;

    // No terminal failure is emitted; the stream completes normally.
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::ProviderFailed(_))),
        "malformed args must not emit ProviderFailed"
    );
    assert!(matches!(items.last(), Some(ModelStreamItem::Completed(_))));

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged
        .finish()
        .expect("stream must reconstruct an invalid call, not fail");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-bad");
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, json!("{\"q\":"));
    let reason = calls[0].invalid.as_deref().expect("call marked invalid");
    assert!(reason.contains("call-bad"), "{reason}");
    assert!(reason.contains("lookup"), "{reason}");
}

/// Drives an SSE byte stream through the parser and returns every item.
async fn collect_sse(raw: Vec<Vec<u8>>) -> Vec<ModelStreamItem> {
    use futures::StreamExt;

    let bytes = futures::stream::iter(
        raw.into_iter()
            .map(|v| Ok::<bytes::Bytes, TinyAgentsError>(bytes::Bytes::from(v))),
    );
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

/// Like [`collect_sse`], but with a specific inline reasoning-tag extraction
/// config (`None` disables inline extraction).
async fn collect_sse_with(
    raw: Vec<Vec<u8>>,
    reasoning_tags: Option<ReasoningTagExtraction>,
) -> Vec<ModelStreamItem> {
    use futures::StreamExt;

    let bytes = futures::stream::iter(
        raw.into_iter()
            .map(|v| Ok::<bytes::Bytes, TinyAgentsError>(bytes::Bytes::from(v))),
    );
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: std::collections::VecDeque::new(),
        acc: OpenAiStreamAcc::new(reasoning_tags),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        terminal_emitted: false,
    };
    futures::stream::unfold(state, sse_next).collect().await
}

/// Concatenates the reasoning fragments across every streamed `MessageDelta`.
fn stream_reasoning(items: &[ModelStreamItem]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.reasoning.clone()),
            _ => None,
        })
        .collect()
}

/// Concatenates the text of every leading `Thinking` block on a response.
fn response_reasoning(response: &ModelResponse) -> String {
    response
        .message
        .content
        .iter()
        .filter_map(|block| block.as_thinking().map(|(text, _)| text.to_string()))
        .collect()
}

/// Concatenates the visible-text fragments across every streamed `MessageDelta`.
fn stream_text(items: &[ModelStreamItem]) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.clone()),
            _ => None,
        })
        .collect()
}

/// Builds a one-content-delta SSE body carrying `content`, then a stop chunk.
fn content_chunks(deltas: &[&str]) -> Vec<Vec<u8>> {
    let mut raw: Vec<Vec<u8>> = deltas
        .iter()
        .map(|d| {
            format!(
                "data: {}\n\n",
                json!({ "choices": [{ "delta": { "content": d } }] })
            )
            .into_bytes()
        })
        .collect();
    raw.push(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_vec());
    raw.push(b"data: [DONE]\n\n".to_vec());
    raw
}

#[tokio::test]
async fn sse_stream_extracts_inline_think_tags() {
    let items = collect_sse_with(
        content_chunks(&["<think>reasoning here</think>", "the answer"]),
        Some(ReasoningTagExtraction::default()),
    )
    .await;

    // Live deltas keep chain-of-thought off the visible channel.
    assert_eq!(stream_reasoning(&items), "reasoning here");
    assert_eq!(stream_text(&items), "the answer");

    // The terminal response carries a leading Thinking block + clean text.
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "the answer");
    assert_eq!(response_reasoning(&response), "reasoning here");
    assert_eq!(
        response.message.content.first(),
        Some(&crate::harness::message::ContentBlock::Thinking {
            text: "reasoning here".into(),
            signature: None,
        })
    );
}

#[tokio::test]
async fn sse_stream_inline_think_tag_split_across_deltas() {
    // The opening tag is split across three network chunks: `<th`, `ink`, `>`.
    let items = collect_sse_with(
        content_chunks(&["before<th", "ink", ">secret</think>after"]),
        Some(ReasoningTagExtraction::default()),
    )
    .await;

    assert_eq!(stream_reasoning(&items), "secret");
    assert_eq!(stream_text(&items), "beforeafter");

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "beforeafter");
    assert_eq!(response_reasoning(&response), "secret");
}

#[tokio::test]
async fn sse_stream_disabled_extraction_leaks_think_tags() {
    // With extraction disabled the inline tags pass straight through — the
    // documented opt-out behavior.
    let items = collect_sse_with(content_chunks(&["<think>cot</think>answer"]), None).await;

    assert_eq!(stream_text(&items), "<think>cot</think>answer");
    assert_eq!(stream_reasoning(&items), "");
}

#[tokio::test]
async fn sse_stream_side_channel_and_inline_reasoning_combine() {
    // A response carrying BOTH a side-channel reasoning fragment and an inline
    // <think> section: side-channel leads, inline follows, joined by the
    // configured separator.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"side\"}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"content\":\"<think>inline</think>done\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let items = collect_sse_with(raw, Some(ReasoningTagExtraction::default())).await;

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(response_reasoning(&response), "side\ninline");
}

#[tokio::test]
async fn sse_stream_start_with_reasoning_deepseek_template() {
    // DeepSeek-R1 template: output begins mid-reasoning, only a closing tag.
    let items = collect_sse_with(
        content_chunks(&["chain of thought", "</think>final"]),
        Some(ReasoningTagExtraction::default().with_start_with_reasoning(true)),
    )
    .await;

    assert_eq!(stream_reasoning(&items), "chain of thought");
    assert_eq!(stream_text(&items), "final");

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    assert_eq!(response.text(), "final");
    assert_eq!(response_reasoning(&response), "chain of thought");
}

#[test]
fn parse_chat_response_extracts_inline_think_and_side_channel() {
    let body = json!({
        "id": "chatcmpl-think",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "side",
                    "content": "<think>inline</think>\n\nThe answer is 42."
                },
                "finish_reason": "stop"
            }
        ]
    });

    let cfg = ReasoningTagExtraction::default();
    let response = parse_chat_response(body, Some(&cfg)).unwrap();
    assert_eq!(response.text(), "The answer is 42.");
    // Side-channel leads, inline follows, separator-joined.
    assert_eq!(response_reasoning(&response), "side\ninline");
}

#[test]
fn parse_chat_response_without_config_leaves_inline_tags_in_text() {
    let body = json!({
        "id": "chatcmpl-plain",
        "choices": [
            { "message": { "role": "assistant", "content": "<think>x</think>y" }, "finish_reason": "stop" }
        ]
    });

    let response = parse_chat_response(body, None).unwrap();
    assert_eq!(response.text(), "<think>x</think>y");
    assert_eq!(response_reasoning(&response), "");
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
async fn sse_stream_processes_many_lines_in_one_chunk() {
    // A single network chunk carrying several complete `data:` lines exercises
    // the batched multi-line drain: every line is parsed in order and the buffer
    // is consumed exactly once. Each line contributes one content fragment.
    let mut chunk = Vec::new();
    for frag in ["a", "b", "c", "d", "e"] {
        chunk.extend_from_slice(
            format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{frag}\"}}}}]}}\n\n")
                .as_bytes(),
        );
    }
    let raw: Vec<Vec<u8>> = vec![chunk, b"data: [DONE]\n\n".to_vec()];

    let items = collect_sse(raw).await;

    let text: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "abcde", "all five lines in one chunk parsed in order");

    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    assert_eq!(merged.finish().unwrap().text(), "abcde");
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
async fn sse_stream_index_zero_parallel_tool_calls_do_not_merge() {
    // Ollama's /v1 endpoint emitted parallel tool calls all carrying `index: 0`
    // (ollama/ollama#15457). Two distinct ids at the same index must open two
    // separate slots instead of silently merging into one. Interleaved
    // continuation fragments (still `index: 0`, carrying the id) must route back
    // to the correct already-open slot rather than the occupant of slot 0.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"function\":{\"name\":\"alpha\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-b\",\"function\":{\"name\":\"beta\",\"arguments\":\"{\\\"y\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-b\",\"function\":{\"arguments\":\"2}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    let calls = response.tool_calls();
    assert_eq!(
        calls.len(),
        2,
        "index-0 parallel calls must not merge: {calls:?}"
    );
    assert_eq!(calls[0].id, "call-a");
    assert_eq!(calls[0].name, "alpha");
    assert_eq!(calls[0].arguments, json!({ "x": 1 }));
    assert_eq!(calls[1].id, "call-b");
    assert_eq!(calls[1].name, "beta");
    assert_eq!(calls[1].arguments, json!({ "y": 2 }));
}

#[tokio::test]
async fn sse_stream_index_zero_id_less_continuations_follow_latest_call() {
    // Worst-case combination of two local-server defects: parallel calls all
    // reuse `index: 0` (ollama/ollama#15457) AND continuation fragments carry
    // no id (the standard OpenAI streaming shape). An id-less continuation
    // under a reused index must follow the call most recently opened at that
    // index — not the original occupant of slot 0.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"function\":{\"name\":\"alpha\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-b\",\"function\":{\"name\":\"beta\",\"arguments\":\"{\\\"y\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"2}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged.finish().unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 2, "expected two distinct calls: {calls:?}");
    assert_eq!(calls[0].id, "call-a");
    assert_eq!(calls[0].arguments, json!({ "x": 1 }));
    assert!(calls[0].invalid.is_none(), "call-a must stay valid");
    assert_eq!(calls[1].id, "call-b");
    assert_eq!(
        calls[1].arguments,
        json!({ "y": 2 }),
        "id-less continuation must land on call-b, the latest call at index 0"
    );
    assert!(calls[1].invalid.is_none(), "call-b must stay valid");
}

#[tokio::test]
async fn sse_stream_empty_name_continuation_never_overwrites_recorded_name() {
    // LM Studio (lmstudio-bug-tracker#649) can send a later tool-call fragment
    // whose `function.name` is an empty string. It must never clobber the name
    // already recorded from the call-opening fragment.
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n".to_vec(),
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"\",\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
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
    assert_eq!(
        calls[0].name, "lookup",
        "an empty-name continuation must not overwrite the recorded name"
    );
    assert_eq!(calls[0].arguments, json!({ "q": 1 }));
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

// ── Auth styles ───────────────────────────────────────────────────────

#[test]
fn auth_headers_bearer_is_the_default() {
    // A freshly-constructed model authenticates with a bearer token.
    let headers = auth_headers(&AuthStyle::default(), "secret");
    assert_eq!(
        headers,
        vec![("Authorization".to_string(), "Bearer secret".to_string())]
    );
    assert_eq!(AuthStyle::default(), AuthStyle::Bearer);
}

#[test]
fn auth_headers_x_api_key_sends_bare_key_without_authorization() {
    let headers = auth_headers(&AuthStyle::XApiKey, "secret");
    assert_eq!(
        headers,
        vec![("x-api-key".to_string(), "secret".to_string())]
    );
    // No bearer `Authorization` header for this style.
    assert!(!headers.iter().any(|(name, _)| name == "Authorization"));
}

#[test]
fn auth_headers_anthropic_pairs_key_with_version() {
    let headers = auth_headers(&AuthStyle::Anthropic, "secret");
    assert_eq!(
        headers,
        vec![
            ("x-api-key".to_string(), "secret".to_string()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ]
    );
}

#[test]
fn auth_headers_custom_uses_the_named_header() {
    let headers = auth_headers(&AuthStyle::Custom("api-key".to_string()), "secret");
    assert_eq!(headers, vec![("api-key".to_string(), "secret".to_string())]);
}

#[test]
fn auth_headers_none_sends_nothing() {
    assert!(auth_headers(&AuthStyle::None, "secret").is_empty());
}

#[test]
fn with_auth_style_and_with_header_build_without_panicking() {
    // The builders compose; a non-bearer style + a static attribution header is a
    // valid model (the field mapping is covered by `auth_headers_*` above).
    let _model = OpenAiModel::new("secret")
        .with_base_url("https://example.test/v1")
        .with_model("some-model")
        .with_auth_style(AuthStyle::XApiKey)
        .with_header("HTTP-Referer", "https://openhuman.example")
        .with_header("X-Title", "OpenHuman");
}
#[test]
fn derive_profile_populates_known_context_windows() {
    // A recognized id gets its context window from the shared provider-neutral
    // hint table, so context-window-aware compaction has a real window to gate
    // on instead of falling back to a fixed threshold.
    assert_eq!(
        super::transport::derive_profile("openai", "gpt-4o-mini").max_input_tokens,
        Some(128_000)
    );
    assert_eq!(
        super::transport::derive_profile("openai", "gpt-4.1").max_input_tokens,
        Some(1_047_576)
    );
    // An unrecognized id stays `None` rather than guessing a window.
    assert_eq!(
        super::transport::derive_profile("openai", "totally-unknown-model").max_input_tokens,
        None
    );
}

// ── Temperature suppression / override ────────────────────────────────

#[test]
fn glob_match_handles_prefix_suffix_infix_and_exact() {
    assert!(glob_match("o1*", "o1-mini"));
    assert!(glob_match("o3*", "o3"));
    assert!(glob_match("gpt-5*", "GPT-5-Turbo")); // case-insensitive
    assert!(glob_match("*turbo", "gpt-4-turbo"));
    assert!(glob_match("*mid*", "a-middle-b"));
    assert!(glob_match("gpt-4o", "gpt-4o")); // no wildcard → exact
    assert!(!glob_match("o1*", "gpt-4o"));
    assert!(!glob_match("gpt-4o", "gpt-4o-mini")); // exact, not prefix
    assert!(!glob_match("*turbo", "turbo-x"));
}

#[test]
fn effective_temperature_omits_for_unsupported_models() {
    let unsupported = vec!["o1*".to_string(), "gpt-5*".to_string()];
    // Matching model → temperature omitted regardless of request/override.
    assert_eq!(
        effective_temperature("o1-mini", Some(0.7), Some(0.2), &unsupported),
        None
    );
    // Non-matching model → request temperature passes through.
    assert_eq!(
        effective_temperature("gpt-4o", Some(0.7), None, &unsupported),
        Some(0.7)
    );
}

#[test]
fn effective_temperature_override_wins_over_request() {
    // Override applies when the model supports temperature.
    assert_eq!(
        effective_temperature("gpt-4o", Some(0.7), Some(0.2), &[]),
        Some(0.2)
    );
    // No override, no request → None.
    assert_eq!(effective_temperature("gpt-4o", None, None, &[]), None);
}

#[test]
fn temperature_builders_compose() {
    let _model = OpenAiModel::new("k")
        .with_model("o1-mini")
        .with_temperature_unsupported_models(["o1*", "o3*"])
        .with_temperature_override(Some(0.0));
}

// ── merge_system_into_user ────────────────────────────────────────────

#[test]
fn merge_system_folds_into_first_user_and_drops_system_role() {
    let merged = merge_system_into_user(&[
        Message::system("You are terse."),
        Message::user("Hi"),
        Message::assistant("Hello"),
        Message::user("Bye"),
    ]);
    // System dropped; its text prefixes the FIRST user turn only.
    assert_eq!(merged.len(), 3);
    assert!(matches!(merged[0], Message::User(_)));
    assert_eq!(merged[0].text(), "You are terse.\n\nHi");
    assert!(matches!(merged[1], Message::Assistant(_)));
    assert_eq!(merged[2].text(), "Bye");
    assert!(!merged.iter().any(|m| matches!(m, Message::System(_))));
}

#[test]
fn merge_system_concatenates_multiple_system_messages() {
    let merged = merge_system_into_user(&[
        Message::system("Rule 1."),
        Message::system("Rule 2."),
        Message::user("Go"),
    ]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text(), "Rule 1.\n\nRule 2.\n\nGo");
}

#[test]
fn merge_system_promotes_to_user_when_no_user_message() {
    let merged = merge_system_into_user(&[Message::system("Only system.")]);
    assert_eq!(merged.len(), 1);
    assert!(matches!(merged[0], Message::User(_)));
    assert_eq!(merged[0].text(), "Only system.");
}

#[test]
fn merge_system_preserves_user_image_blocks() {
    use crate::harness::message::{ImageRef, UserMessage};
    let user_with_image = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("caption".to_string()),
            ContentBlock::Image(ImageRef {
                url: "https://example.test/x.png".to_string(),
                mime_type: None,
            }),
        ],
    });
    let merged = merge_system_into_user(&[Message::system("sys"), user_with_image]);
    assert_eq!(merged.len(), 1);
    if let Message::User(u) = &merged[0] {
        assert_eq!(u.content.len(), 2); // text (merged) + image preserved
        assert!(matches!(u.content[0], ContentBlock::Text(ref t) if t == "sys\n\ncaption"));
        assert!(matches!(u.content[1], ContentBlock::Image(_)));
    } else {
        panic!("expected a user message");
    }
}

#[test]
fn merge_system_is_noop_without_system_messages() {
    let input = vec![Message::user("just user")];
    assert_eq!(merge_system_into_user(&input), input);
}

/// Builds an OpenAI-shaped non-streaming response body carrying a single tool
/// call whose `function.arguments` string is `raw` (verbatim, including any
/// corruption). Used to exercise `parse_tool_arguments` recovery/fail-fast.
fn tool_call_body(raw: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-toolargs",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": { "name": "composio_execute", "arguments": raw }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    })
}

#[test]
fn recovers_tool_args_with_leaked_trailing_template_marker() {
    // Regression (openhuman#4766): an OpenAI-compatible gateway leaked a
    // model's trailing `<tool_call|>` chat-template delimiter into
    // `function.arguments`. The JSON is otherwise valid, so stripping the
    // marker must recover the call instead of failing the turn.
    let response = parse_response(tool_call_body(
        r#"{"tool":"GMAIL_CREATE_EMAIL_DRAFT","x":1}<tool_call|>"#,
    ))
    .expect("leaked marker must be recovered");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments,
        json!({ "tool": "GMAIL_CREATE_EMAIL_DRAFT", "x": 1 })
    );
}

#[test]
fn recovers_tool_args_wrapped_in_tool_call_tags() {
    // Some templates wrap the whole call in `<tool_call>…</tool_call>`; both
    // delimiters must be stripped before parsing.
    let response = parse_response(tool_call_body(r#"<tool_call>{"q":"hi"}</tool_call>"#))
        .expect("wrapped args must be recovered");
    assert_eq!(response.tool_calls()[0].arguments, json!({ "q": "hi" }));
}

#[test]
fn recovers_first_call_when_fragments_are_concatenated() {
    // A leaked delimiter can also sit *between* two concatenated argument
    // objects (e.g. an accumulator over-merge). After stripping the marker the
    // string is `{...}{...}`; recovery keeps the first complete object.
    let response = parse_response(tool_call_body(
        r#"{"tool":"gmail"}<tool_call|>{"tool":"other"}"#,
    ))
    .expect("leading call must be recovered");
    assert_eq!(
        response.tool_calls()[0].arguments,
        json!({ "tool": "gmail" })
    );
}

#[test]
fn marks_genuinely_malformed_tool_args_invalid_instead_of_hanging() {
    // The exact corruption seen in the wild carries a stray `]` *inside* the
    // JSON — not just a leaked delimiter — so it cannot be safely repaired.
    // Rather than fail the call (or hang on a never-resolving one), it is
    // surfaced as an `invalid` ToolCall the agent loop can bounce back to the
    // model with a clear error.
    let response = parse_response(tool_call_body(
        r#"{"arguments":{"body":"hi"]}}<tool_call|>"#,
    ))
    .expect("unrepairable args must resolve as an invalid call, not fail");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    let reason = calls[0].invalid.as_deref().expect("call marked invalid");
    assert!(reason.contains("call-1"), "{reason}");
    assert!(reason.contains("raw arguments"), "{reason}");
}

#[test]
fn valid_tool_args_containing_marker_substring_are_left_intact() {
    // A legitimate arguments object whose *string value* contains the marker
    // text parses cleanly on the first attempt, so the repair path never runs
    // and the value is preserved verbatim.
    let response =
        parse_response(tool_call_body(r#"{"body":"use <tool_call|> literally"}"#)).unwrap();
    assert_eq!(
        response.tool_calls()[0].arguments,
        json!({ "body": "use <tool_call|> literally" })
    );
}

#[test]
fn recovers_relaxed_json_in_structured_tool_args() {
    // Small local models emit a relaxed JSON dialect for arguments — unquoted
    // keys and redundant wrapping braces (`{{…}}`, escalating on each bounced
    // retry). Strict parse rejects them; the relaxed-JSON repair recovers an
    // executable call instead of looping.
    let response = parse_response(tool_call_body(
        r#"{{tool:"DISCORD_LIST_CHANNELS",arguments:{"guild_id":"1"}}}"#,
    ))
    .expect("relaxed structured args must recover");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].invalid.is_none(), "{:?}", calls[0].invalid);
    assert_eq!(
        calls[0].arguments,
        json!({ "tool": "DISCORD_LIST_CHANNELS", "arguments": { "guild_id": "1" } })
    );
}

#[test]
fn recovers_leaked_quote_tokens_in_structured_tool_args() {
    // A gateway leaked the model's string-delimiter token as literal `<|">` in
    // place of `"`, so every value arrived wrapped in it. Combined with unquoted
    // keys this looped forever; it must recover as an executable call.
    let response = parse_response(tool_call_body(
        r#"{arguments:{guild_id:<|">1470856511193616498<|">},tool:<|">DISCORD_GET_GUILD_CHANNELS<|">}"#,
    ))
    .expect("leaked quote-token args must recover");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].invalid.is_none(), "{:?}", calls[0].invalid);
    assert_eq!(
        calls[0].arguments,
        json!({
            "arguments": { "guild_id": "1470856511193616498" },
            "tool": "DISCORD_GET_GUILD_CHANNELS"
        })
    );
}

#[test]
fn keyless_nested_object_still_resolves_invalid_not_fabricated() {
    // The one relaxed shape deliberately NOT repaired: a value with no key
    // (`{tool:"X",{…}}`) is genuinely broken and must resolve as an `invalid`
    // call the model can retry — never a fabricated best-guess object.
    let response = parse_response(tool_call_body(r#"{tool:"X",{guild_id:"Y"}}"#))
        .expect("unrepairable args resolve as invalid, not a hard failure");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert!(
        calls[0].invalid.is_some(),
        "keyless nested object must not be silently fabricated into args"
    );
}

#[tokio::test]
async fn sse_stream_recovers_tool_args_with_leaked_template_marker() {
    // The streaming reduce (`OpenAiStreamAcc::into_response`) shares
    // `parse_tool_arguments`, so a leaked trailing marker across the terminal
    // must reconstruct a usable call instead of finishing as a model error
    // (which is what orphaned the parent's join/reduce in openhuman#4766).
    let raw: Vec<Vec<u8>> = vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-x\",\"function\":{\"name\":\"composio_execute\",\"arguments\":\"{\\\"q\\\":1}<tool_call|>\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];

    let items = collect_sse(raw).await;
    let mut merged = StreamAccumulator::new();
    for item in &items {
        merged.push(item);
    }
    let response = merged
        .finish()
        .expect("stream must not fail on a recoverable marker");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "composio_execute");
    assert_eq!(calls[0].arguments, json!({ "q": 1 }));
}
// `ChatModel::profile` is generic over `State`; pin `State = ()` so the concrete
// `OpenAiModel` handle disambiguates without a turbofish at every call site.
fn profile_of(model: &OpenAiModel) -> &crate::harness::model::ModelProfile {
    ChatModel::<()>::profile(model).expect("openai models expose a profile")
}

#[test]
fn with_native_tool_calling_false_clears_tool_profile_flags() {
    let model = OpenAiModel::new("k")
        .with_model("qwen2.5")
        .with_native_tool_calling(false);
    let profile = profile_of(&model);
    assert!(!profile.tool_calling);
    assert!(!profile.parallel_tool_calls);
    assert!(!profile.streaming_tool_chunks);

    // Re-enabling restores native tool calling on the profile.
    let re = OpenAiModel::new("k")
        .with_model("qwen2.5")
        .with_native_tool_calling(false)
        .with_native_tool_calling(true);
    assert!(profile_of(&re).tool_calling);
}

#[test]
fn prompt_guided_request_replays_calls_and_results_as_text() {
    let model = OpenAiModel::new("k")
        .with_model("qwen2.5")
        .with_native_tool_calling(false);
    let mut assistant = Message::assistant("");
    let Message::Assistant(message) = &mut assistant else {
        unreachable!()
    };
    message.tool_calls = vec![crate::harness::tool::ToolCall::new(
        "call-1",
        "lookup",
        json!({"q":"weather"}),
    )];
    let request = ModelRequest::new(vec![
        Message::user("look it up"),
        assistant,
        Message::tool("call-1", "sunny"),
    ])
    .with_tools(vec![ToolSchema::new(
        "lookup",
        "look something up",
        json!({"type":"object"}),
    )]);

    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();
    assert_eq!(value["messages"][2]["role"], "assistant");
    assert!(
        value["messages"][2]["content"]
            .as_str()
            .unwrap()
            .contains("<tool_call>")
    );
    assert!(value["messages"][2].get("tool_calls").is_none());
    assert_eq!(value["messages"][3]["role"], "user");
    assert!(
        value["messages"][3]["content"]
            .as_str()
            .unwrap()
            .contains("<tool_result>\nsunny\n</tool_result>")
    );
    assert!(value.get("tools").is_none());
}

#[test]
fn with_vision_toggles_image_in_modality() {
    let off = OpenAiModel::new("k")
        .with_model("qwen2.5")
        .with_vision(false);
    assert!(!profile_of(&off).modalities.image_in);

    let on = OpenAiModel::new("k")
        .with_model("gpt-4.1-mini")
        .with_vision(true);
    assert!(profile_of(&on).modalities.image_in);
}

#[test]
fn default_provider_options_are_baked_onto_every_request() {
    let model = OpenAiModel::ollama()
        .with_model("qwen2.5")
        .with_default_provider_options(json!({ "options": { "num_ctx": 8192 } }));

    // A request with no provider_options still carries the baked options.
    let request = ModelRequest::new(vec![Message::user("hi")]);
    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();
    assert_eq!(value["options"]["num_ctx"], json!(8192));
}

#[test]
fn request_provider_options_win_over_baked_defaults() {
    let model = OpenAiModel::ollama()
        .with_model("qwen2.5")
        .with_default_provider_options(
            json!({ "options": { "num_ctx": 8192 }, "keep_alive": "5m" }),
        );

    // The per-call `options` key overrides the baked one; unrelated baked keys
    // (`keep_alive`) still flow through.
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_provider_options(json!({ "options": { "num_ctx": 2048 } }));
    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();
    assert_eq!(value["options"]["num_ctx"], json!(2048));
    assert_eq!(value["keep_alive"], json!("5m"));
}

#[test]
fn merge_provider_options_prefers_overrides_and_handles_nulls() {
    let defaults = json!({ "a": 1, "b": 2 });
    let overrides = json!({ "b": 20, "c": 30 });
    assert_eq!(
        merge_provider_options(&defaults, &overrides),
        json!({ "a": 1, "b": 20, "c": 30 })
    );
    assert_eq!(
        merge_provider_options(&serde_json::Value::Null, &overrides),
        overrides
    );
    assert_eq!(
        merge_provider_options(&defaults, &serde_json::Value::Null),
        defaults
    );
    assert_eq!(
        merge_provider_options(&serde_json::Value::Null, &serde_json::Value::Null),
        serde_json::Value::Null
    );
    // A non-null, non-object override is passed through untouched (not merged), so
    // downstream validation still rejects it rather than the merge hiding it.
    assert_eq!(
        merge_provider_options(&defaults, &json!(["top_k", 40])),
        json!(["top_k", 40])
    );
}

// ---------------------------------------------------------------------------
// Local-server request-shape degradation (named tool_choice, json_object)
// ---------------------------------------------------------------------------

#[test]
fn degrades_named_tool_choice_to_required_and_filters_tools() {
    // The endpoint rejects the object form: send `"required"` and drop every
    // tool except the named one so the model has no other tool to pick.
    let model = OpenAiModel::new("k")
        .with_model("m")
        .with_named_tool_choice(false);
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![
            ToolSchema::new("get_weather", "w", json!({})),
            ToolSchema::new("get_time", "t", json!({})),
        ])
        .with_tool_choice(ToolChoice::Tool("get_weather".to_string()));

    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();

    assert_eq!(value["tool_choice"], json!("required"));
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], json!("get_weather"));
}

#[test]
fn degraded_named_tool_choice_keeps_tools_when_named_tool_absent() {
    // Named tool not in the list: leave `tools` intact but still send "required".
    let model = OpenAiModel::new("k")
        .with_model("m")
        .with_named_tool_choice(false);
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("get_time", "t", json!({}))])
        .with_tool_choice(ToolChoice::Tool("missing".to_string()));

    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();

    assert_eq!(value["tool_choice"], json!("required"));
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], json!("get_time"));
}

#[test]
fn degrades_json_object_to_permissive_json_schema() {
    let model = OpenAiModel::new("k")
        .with_model("m")
        .with_json_object_format(false);
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_response_format(ResponseFormat::JsonObject);

    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();

    assert_eq!(
        value["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "json_object",
                "schema": { "type": "object" },
                "strict": false,
            }
        })
    );
}

#[test]
fn baseline_knobs_default_to_supported_wire_shapes() {
    // Default knobs preserve the hosted-OpenAI object/json_object wire shapes, so
    // an accidental default flip would fail here.
    let model = model();
    assert_eq!(model.baseline_degrade(), Degrade::default());

    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("t", "d", json!({}))])
        .with_tool_choice(ToolChoice::Tool("t".to_string()))
        .with_response_format(ResponseFormat::JsonObject);
    let value = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();

    assert_eq!(
        value["tool_choice"],
        json!({ "type": "function", "function": { "name": "t" } })
    );
    assert_eq!(value["response_format"], json!({ "type": "json_object" }));
}

#[test]
fn degrade_for_400_targets_only_the_shape_the_request_used() {
    let named = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("t", "d", json!({}))])
        .with_tool_choice(ToolChoice::Tool("t".to_string()));
    assert_eq!(
        degrade_for_400(
            "Invalid tool_choice type: 'object'. Supported string values: none, auto, required",
            &named,
            Degrade::default(),
        ),
        Some(Degrade {
            named_tool_choice: true,
            json_object: false,
        })
    );

    let json = ModelRequest::new(vec![Message::user("hi")])
        .with_response_format(ResponseFormat::JsonObject);
    assert_eq!(
        degrade_for_400(
            "'response_format.type' must be 'json_schema' or 'text'",
            &json,
            Degrade::default(),
        ),
        Some(Degrade {
            named_tool_choice: false,
            json_object: true,
        })
    );
}

#[test]
fn degrade_for_400_ignores_unrelated_or_already_degraded_failures() {
    let named = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("t", "d", json!({}))])
        .with_tool_choice(ToolChoice::Tool("t".to_string()));

    // A tool_choice message but the request used `Required` (not a named tool):
    // there is nothing to degrade, so no retry.
    let required = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![ToolSchema::new("t", "d", json!({}))])
        .with_tool_choice(ToolChoice::Required);
    assert_eq!(
        degrade_for_400("Invalid tool_choice type", &required, Degrade::default()),
        None
    );

    // An unrelated 400 never triggers a degraded retry.
    assert_eq!(
        degrade_for_400("context length exceeded", &named, Degrade::default()),
        None
    );

    // Already degraded on the first attempt -> no repeat (prevents a retry loop).
    assert_eq!(
        degrade_for_400(
            "Invalid tool_choice type",
            &named,
            Degrade {
                named_tool_choice: true,
                json_object: false,
            },
        ),
        None
    );
}

#[test]
fn degrade_for_400_unions_with_existing_baseline_degrade() {
    // A json_object 400 while the named-tool-choice degrade is already baked on
    // keeps that baseline flag set for the retry.
    let json = ModelRequest::new(vec![Message::user("hi")])
        .with_response_format(ResponseFormat::JsonObject);
    assert_eq!(
        degrade_for_400(
            "response_format must be json_schema",
            &json,
            Degrade {
                named_tool_choice: true,
                json_object: false,
            },
        ),
        Some(Degrade {
            named_tool_choice: true,
            json_object: true,
        })
    );
}
