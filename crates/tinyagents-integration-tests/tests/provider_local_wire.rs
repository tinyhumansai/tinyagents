//! Regression tests for the local-runtime provider fixes, asserted **at the
//! wire**.
//!
//! # Why these go through a real socket
//!
//! The pre-existing local tests assert that a request body *contains* a field —
//! for example that `{"options": {"num_ctx": 8192}}` appears in the JSON. That
//! shape of assertion is what let LOCAL-2 survive: `num_ctx` was present in the
//! body of a request sent to `POST /chat/completions`, an endpoint that does not
//! read it. The field was there; the behaviour was not.
//!
//! So these tests assert the thing that actually matters — **which URL the bytes
//! went to, and what the adapter did with the answer** — by standing up a
//! throwaway HTTP server on a loopback port and inspecting what arrives.
//!
//! The server is hand-rolled on `std::net` rather than a mock crate because the
//! dev-dependency `tokio` is built without the `net` feature, and adding an HTTP
//! mocking dependency to exercise four endpoints is not a trade worth making.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use tinyinference::tool::ToolSchema;
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest, ReasoningEffort, ResponseFormat, ToolChoice};
use tinyinference::providers::openai::OpenAiModel;

// ---------------------------------------------------------------------------
// Minimal recording HTTP server
// ---------------------------------------------------------------------------

/// One request the server received.
#[derive(Clone, Debug)]
struct Recorded {
    method: String,
    path: String,
    body: Value,
}

/// A canned reply.
#[derive(Clone)]
struct Canned {
    status: u16,
    body: String,
}

impl Canned {
    fn ok(body: Value) -> Self {
        Self {
            status: 200,
            body: body.to_string(),
        }
    }

    fn error(status: u16, body: Value) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }
}

/// A loopback HTTP server that replies from a scripted queue and records every
/// request it saw.
struct MockServer {
    base_url: String,
    seen: Arc<Mutex<Vec<Recorded>>>,
}

impl MockServer {
    /// Starts a server that answers each successive request with the next
    /// scripted reply, repeating the last one once the script is exhausted.
    fn start(script: Vec<Canned>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
        let port = listener.local_addr().expect("local addr").port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        std::thread::spawn(move || {
            for (index, stream) in listener.incoming().enumerate() {
                let Ok(stream) = stream else { break };
                let reply = script
                    .get(index)
                    .or_else(|| script.last())
                    .cloned()
                    .unwrap_or_else(|| Canned::ok(json!({})));
                serve_one(stream, &reply, &recorder);
            }
        });

        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            seen,
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.seen.lock().expect("recorder lock").clone()
    }

    /// The single request sent to `path`, panicking with the full request log
    /// when there is not exactly one — a far more useful failure than
    /// `Option::unwrap`.
    fn request_to(&self, path: &str) -> Recorded {
        let all = self.requests();
        let matched: Vec<&Recorded> = all.iter().filter(|r| r.path.starts_with(path)).collect();
        assert_eq!(
            matched.len(),
            1,
            "expected exactly one request to {path}; saw {:?}",
            all.iter()
                .map(|r| format!("{} {}", r.method, r.path))
                .collect::<Vec<_>>()
        );
        matched[0].clone()
    }

    fn paths(&self) -> Vec<String> {
        self.requests().into_iter().map(|r| r.path).collect()
    }
}

/// Reads one HTTP/1.1 request off `stream`, records it, then writes `reply`.
///
/// **Recording happens before the reply is written**, and that ordering is
/// load-bearing rather than incidental. Recording afterwards is a race the
/// client always wins: `invoke` can receive the full response body and return to
/// the test while the server thread has not yet reached its `push`, so an
/// assertion made immediately after the call sees an empty log. That is exactly
/// how two of these tests came out flaky under a loaded parallel test run and
/// green when the file ran alone.
fn serve_one(mut stream: TcpStream, reply: &Canned, recorder: &Arc<Mutex<Vec<Recorded>>>) {
    let Some(record) = read_request(&mut stream) else {
        return;
    };
    recorder.lock().expect("recorder lock").push(record);

    let response = format!(
        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reply.status,
        reply.body.len(),
        reply.body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Reads one HTTP/1.1 request off `stream`. Returns `None` for a malformed one.
fn read_request(stream: &mut TcpStream) -> Option<Recorded> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);

    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            break;
        }
        if header.trim().is_empty() {
            break;
        }
        if let Some(value) = header
            .to_ascii_lowercase()
            .strip_prefix("content-length:")
            .map(str::trim)
            .and_then(|v| v.parse::<usize>().ok())
        {
            content_length = value;
        }
    }

    let mut raw = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut raw).ok()?;
    }
    let body = serde_json::from_slice::<Value>(&raw).unwrap_or(Value::Null);

    Some(Recorded { method, path, body })
}

/// A minimal, valid Chat Completions reply.
fn chat_reply(text: &str) -> Value {
    json!({
        "id": "chatcmpl-test",
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 }
    })
}

fn user(text: &str) -> ModelRequest {
    ModelRequest::new(vec![Message::user(text)])
}

// ---------------------------------------------------------------------------
// LOCAL-2 — `num_ctx` must reach an endpoint that reads it
// ---------------------------------------------------------------------------

/// Before the fix this was unreachable: `num_ctx` was flattened onto the
/// `/chat/completions` body, where Ollama's compatibility layer ignores it, and
/// the only tests asserted the field's *presence in that body*. The behaviour
/// under test is that the value goes to `/api/chat`, the endpoint that reads it.
#[tokio::test]
async fn num_ctx_and_keep_alive_reach_the_native_ollama_endpoint() {
    let server = MockServer::start(vec![Canned::ok(json!({ "model": "llama3.2" }))]);

    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2")
        .expect("valid local URL")
        .with_local_num_ctx(8192)
        .with_keep_alive("30m");

    model.warm_up().await.expect("warm-up succeeds");

    let request = server.request_to("/api/chat");
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path, "/api/chat",
        "num_ctx is an /api/chat field; sending it to /chat/completions is how it went nowhere"
    );
    assert_eq!(request.body["options"]["num_ctx"], json!(8192));
    assert_eq!(request.body["keep_alive"], json!("30m"));
    // An empty `messages` array is Ollama's documented load request.
    assert_eq!(request.body["messages"], json!([]));
}

/// `with_local_num_ctx` must also move the *advertised* window. A window you
/// requested and a window you advertise being different numbers is precisely the
/// LOCAL-1 failure: compaction is gated on the advertised one.
#[test]
fn requesting_a_context_window_also_advertises_it() {
    let model = OpenAiModel::ollama().with_local_num_ctx(8192);
    let profile = <OpenAiModel as ChatModel<()>>::profile(&model).expect("local profile");
    assert_eq!(profile.max_input_tokens, Some(8192));
}

/// A runtime with no native API must not attempt one.
#[tokio::test]
async fn warm_up_is_a_no_op_for_a_runtime_without_a_native_api() {
    let server = MockServer::start(vec![Canned::ok(json!({}))]);
    let model = OpenAiModel::llama_cpp(&server.base_url, "local-model").expect("valid URL");
    model.warm_up().await.expect("warm-up is a no-op");
    assert!(
        server.requests().is_empty(),
        "llama.cpp-server has no /api/chat; inventing one would 404 every startup"
    );
}

// ---------------------------------------------------------------------------
// LOCAL-1 / C10 — the real context window comes from the server
// ---------------------------------------------------------------------------

/// `llama3.2` matches the generic hint table's `("llama3", Substring, 128_000)`
/// entry. Ollama's real default `num_ctx` is 2048. The probe must replace the
/// guess with what the server reports, and the un-probed default must be `None`
/// rather than the guess.
#[tokio::test]
async fn probing_replaces_the_model_id_guess_with_the_servers_own_window() {
    let server = MockServer::start(vec![Canned::ok(json!({
        "model_info": { "general.architecture": "llama", "llama.context_length": 8192 },
        "capabilities": ["completion", "tools"]
    }))]);

    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    // Un-probed: unknown, not 128 000.
    assert_eq!(
        <OpenAiModel as ChatModel<()>>::profile(&model)
            .expect("profile")
            .max_input_tokens,
        None,
        "an invented window is worse than None: compaction is gated on it"
    );

    let model = model.probed().await.expect("probe succeeds");
    let profile = <OpenAiModel as ChatModel<()>>::profile(&model).expect("profile");
    assert_eq!(profile.max_input_tokens, Some(8192));
    assert!(
        profile.tool_calling,
        "the server reported the `tools` capability"
    );
    assert!(
        !profile.modalities.image_in,
        "no `vision` capability reported"
    );

    assert_eq!(
        server.request_to("/api/show").body["model"],
        json!("llama3.2")
    );
}

/// A server without the probe endpoint must degrade to "learned nothing", not
/// fail the caller's startup.
#[tokio::test]
async fn a_probe_against_an_older_server_is_not_an_error() {
    let server = MockServer::start(vec![Canned::error(404, json!({ "error": "not found" }))]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");
    let probe = model
        .probe_local_profile()
        .await
        .expect("a 404 is not fatal");
    assert!(probe.is_empty());
}

/// Probing a hosted endpoint is a programming error, and says so.
#[tokio::test]
async fn probing_a_hosted_endpoint_is_rejected() {
    let error = OpenAiModel::new("k")
        .probe_local_profile()
        .await
        .expect_err("hosted OpenAI is not a local runtime");
    assert!(error.to_string().contains("local runtime"), "{error}");
}

// ---------------------------------------------------------------------------
// LOCAL-3 / C11 — native tools are sent, then degraded only on evidence
// ---------------------------------------------------------------------------

/// Native tools used to be hard-disabled for every local runtime, so every call
/// took the prompt-guided branch and injected the protocol plus each tool's JSON
/// Schema into the system prompt. They must now go on the wire.
#[tokio::test]
async fn a_local_runtime_sends_native_tools() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("done"))]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let request = user("hi").with_tools(vec![ToolSchema::new(
        "get_weather",
        "look up weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )]);
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let sent = server.request_to("/v1/chat/completions");
    let tools = sent.body["tools"]
        .as_array()
        .expect("native tools on the wire");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], json!("get_weather"));

    // The prompt-guided protocol block must be absent — it is what the tiny real
    // context window could not afford.
    let system = sent.body["messages"][0]["content"].as_str().unwrap_or("");
    assert!(
        !system.contains("<tool_call>"),
        "prompt-guided protocol leaked into a native-tools request: {system}"
    );
}

/// The auto-degrade half: a 400 that implicates `tools` must retry
/// prompt-guided **and latch**, so the rejection is paid once per process rather
/// than being assumed forever up front.
#[tokio::test]
async fn a_tools_rejection_degrades_once_and_then_latches() {
    let server = MockServer::start(vec![
        Canned::error(
            400,
            json!({ "error": { "message": "registry.ollama.ai/library/gemma3 does not support tools" } }),
        ),
        Canned::ok(chat_reply("done")),
        Canned::ok(chat_reply("again")),
    ]);
    let model = OpenAiModel::ollama_at(&server.base_url, "gemma3").expect("valid local URL");
    let tools = vec![ToolSchema::new("t", "d", json!({"type": "object"}))];

    let first = user("hi").with_tools(tools.clone());
    ChatModel::<()>::invoke(&model, &(), first)
        .await
        .expect("the degraded retry succeeds");

    let seen = server.requests();
    assert_eq!(
        seen.len(),
        2,
        "one rejected attempt, then one degraded retry"
    );
    assert!(
        seen[0].body["tools"]
            .as_array()
            .is_some_and(|t| !t.is_empty()),
        "the first attempt must try native tools"
    );
    assert!(
        seen[1]
            .body
            .get("tools")
            .is_none_or(|t| t.as_array().is_none_or(Vec::is_empty)),
        "the retry must drop native tools"
    );

    // Latched: the next call goes straight to the degraded shape.
    let second = user("again").with_tools(tools);
    ChatModel::<()>::invoke(&model, &(), second)
        .await
        .expect("second call succeeds");
    let seen = server.requests();
    assert_eq!(
        seen.len(),
        3,
        "the latch must skip the doomed baseline attempt"
    );
    assert!(
        seen[2]
            .body
            .get("tools")
            .is_none_or(|t| t.as_array().is_none_or(Vec::is_empty)),
        "the latch was not applied to the following call"
    );
}

// ---------------------------------------------------------------------------
// LOCAL-6 — self-hosted servers other than Ollama/LM Studio get local treatment
// ---------------------------------------------------------------------------

#[test]
fn llama_cpp_and_vllm_are_treated_as_local_runtimes() {
    for model in [
        OpenAiModel::llama_cpp("127.0.0.1:8080", "local-model").expect("valid URL"),
        OpenAiModel::vllm("127.0.0.1:8000", "", "meta-llama/Llama-3.3-70B").expect("valid URL"),
    ] {
        // `/v1` normalisation, which the hosted `Compatible` path did not do.
        assert!(
            model.base_url().ends_with("/v1"),
            "unexpected base url: {}",
            model.base_url()
        );
        assert!(model.local_runtime_kind().is_some());

        let profile = <OpenAiModel as ChatModel<()>>::profile(&model).expect("profile");
        // No invented window even for `Llama-3.3-70B`, which the hint table
        // matches through `("llama-3", Substring, 128_000)`.
        assert_eq!(profile.max_input_tokens, None);
    }
}

/// The degrade knobs the local presets exist for must be pre-set, so the first
/// call does not pay a guaranteed 400 to rediscover a documented rejection.
#[tokio::test]
async fn local_presets_pre_set_the_shapes_local_servers_reject() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("{}"))]);
    let model = OpenAiModel::llama_cpp(&server.base_url, "local-model").expect("valid URL");

    let request = user("hi")
        .with_tools(vec![ToolSchema::new("t", "d", json!({"type": "object"}))])
        .with_tool_choice(ToolChoice::Tool("t".to_string()))
        .with_response_format(ResponseFormat::JsonObject);
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let sent = server.request_to("/v1/chat/completions");
    assert_eq!(
        sent.body["tool_choice"],
        json!("required"),
        "the named tool_choice object must already be degraded on the first call"
    );
    assert_eq!(
        sent.body["response_format"]["type"],
        json!("json_schema"),
        "json_object must already be degraded on the first call"
    );
}

// ---------------------------------------------------------------------------
// C12 — an actionable missing-model error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_missing_local_model_names_the_fix() {
    let server = MockServer::start(vec![Canned::error(
        404,
        json!({ "error": { "message": "model 'llama3.2' not found" } }),
    )]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let error = ChatModel::<()>::invoke(&model, &(), user("hi"))
        .await
        .expect_err("a 404 fails the call");
    let rendered = error.to_string();
    assert!(
        rendered.contains("ollama pull llama3.2"),
        "an opaque 404 tells the operator nothing: {rendered}"
    );
}

#[tokio::test]
async fn validate_model_lists_what_the_server_actually_serves() {
    let server = MockServer::start(vec![Canned::ok(json!({
        "object": "list",
        "data": [{ "id": "qwen3:8b" }, { "id": "bge-m3" }]
    }))]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let error = model
        .validate_model()
        .await
        .expect_err("llama3.2 is not served");
    let rendered = error.to_string();
    assert!(rendered.contains("ollama pull llama3.2"), "{rendered}");
    assert!(rendered.contains("qwen3:8b"), "{rendered}");
}

// ---------------------------------------------------------------------------
// REASON-4 — `strict` is a knob, and the schema is sanitized when it is on
// ---------------------------------------------------------------------------

/// `strict: true` was hardcoded and paired with the caller's raw schema, so a
/// valid JSON Schema 400d. Local runtimes must default it off entirely.
#[tokio::test]
async fn a_local_runtime_does_not_send_strict_structured_output() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("{}"))]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let request = user("hi").with_response_format(ResponseFormat::json_schema(
        "answer",
        json!({"type": "object", "properties": {"a": {"type": "string"}}}),
    ));
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let sent = server.request_to("/v1/chat/completions");
    assert_eq!(
        sent.body["response_format"]["json_schema"]["strict"],
        json!(false)
    );
}

/// On hosted OpenAI strict stays on, but the schema is now sanitized to satisfy
/// it: every property in `required`, `additionalProperties: false`.
#[tokio::test]
async fn hosted_strict_mode_sanitizes_the_schema_it_sends() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("{}"))]);
    let model = OpenAiModel::new("k").with_base_url(format!("{}/v1", server.base_url));

    let request = user("hi").with_response_format(ResponseFormat::json_schema(
        "answer",
        json!({
            "type": "object",
            "properties": { "a": {"type": "string"}, "b": {"type": "number"} }
        }),
    ));
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let schema = &server.request_to("/v1/chat/completions").body["response_format"]["json_schema"];
    assert_eq!(schema["strict"], json!(true));
    assert_eq!(
        schema["schema"]["additionalProperties"],
        json!(false),
        "strict mode rejects an object without this"
    );
    let required = schema["schema"]["required"]
        .as_array()
        .expect("strict mode requires every property to be listed")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(
        required.contains(&"a") && required.contains(&"b"),
        "{required:?}"
    );
}

/// A `JsonSchema` request had no degradation path at all — only `JsonObject` was
/// considered — so a 400 on it was terminal.
#[tokio::test]
async fn a_strict_schema_rejection_retries_without_strict() {
    let server = MockServer::start(vec![
        Canned::error(
            400,
            json!({ "error": { "message": "response_format: 'strict' is not supported" } }),
        ),
        Canned::ok(chat_reply("{}")),
    ]);
    let model = OpenAiModel::new("k").with_base_url(format!("{}/v1", server.base_url));

    let request = user("hi").with_response_format(ResponseFormat::json_schema(
        "answer",
        json!({"type": "object", "properties": {"a": {"type": "string"}}}),
    ));
    ChatModel::<()>::invoke(&model, &(), request)
        .await
        .expect("the degraded retry succeeds");

    let seen = server.requests();
    assert_eq!(seen.len(), 2, "a JsonSchema 400 must have a retry path");
    assert_eq!(
        seen[0].body["response_format"]["json_schema"]["strict"],
        json!(true)
    );
    assert_eq!(
        seen[1].body["response_format"]["json_schema"]["strict"],
        json!(false)
    );
}

// ---------------------------------------------------------------------------
// REASON-6 — the gpt-5 family
// ---------------------------------------------------------------------------

/// gpt-5 was routed to `max_tokens`, which OpenAI rejects outright.
#[tokio::test]
async fn gpt5_sends_max_completion_tokens_not_max_tokens() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("hi"))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_model("gpt-5-mini");

    let _ = ChatModel::<()>::invoke(&model, &(), user("hi").with_max_tokens(64)).await;

    let sent = server.request_to("/v1/chat/completions");
    assert_eq!(sent.body["max_completion_tokens"], json!(64));
    assert!(
        sent.body.get("max_tokens").is_none(),
        "OpenAI rejects max_tokens for gpt-5: {}",
        sent.body
    );
}

#[test]
fn gpt5_profiles_as_a_reasoning_model_with_native_structured_output() {
    let model = OpenAiModel::new("k").with_model("gpt-5");
    let profile = <OpenAiModel as ChatModel<()>>::profile(&model).expect("profile");
    assert!(
        profile.reasoning,
        "CapabilitySet{{reasoning:true}} used to reject gpt-5"
    );
    assert!(profile.native_structured_output);
    assert!(profile.reasoning_effort);
    assert_eq!(profile.max_input_tokens, Some(400_000));
}

// ---------------------------------------------------------------------------
// C13 — provider-neutral reasoning config
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reasoning_effort_lowers_onto_the_chat_completions_field() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("hi"))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_model("gpt-5");

    let _ = ChatModel::<()>::invoke(
        &model,
        &(),
        user("hi").with_reasoning_effort(ReasoningEffort::High),
    )
    .await;

    assert_eq!(
        server.request_to("/v1/chat/completions").body["reasoning_effort"],
        json!("high")
    );
}

/// `provider_options` stays the escape hatch and wins, rather than the key
/// landing on the wire twice.
#[tokio::test]
async fn provider_options_win_over_the_typed_reasoning_field() {
    let server = MockServer::start(vec![Canned::ok(chat_reply("hi"))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_model("gpt-5");

    let request = user("hi")
        .with_reasoning_effort(ReasoningEffort::High)
        .with_provider_option("reasoning_effort", json!("minimal"));
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    assert_eq!(
        server.request_to("/v1/chat/completions").body["reasoning_effort"],
        json!("minimal")
    );
}

// ---------------------------------------------------------------------------
// CACHE-6b — cache tokens are accounted for
// ---------------------------------------------------------------------------

/// `Usage::cache_creation_tokens` is summed and priced, and no provider ever set
/// it, so cache writes were billed as ordinary input everywhere.
#[tokio::test]
async fn cache_write_tokens_are_recorded() {
    let server = MockServer::start(vec![Canned::ok(json!({
        "id": "c1",
        "choices": [{ "message": { "role": "assistant", "content": "hi" }, "finish_reason": "stop" }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": { "cached_tokens": 40, "cache_write_tokens": 25 }
        }
    }))]);
    let model = OpenAiModel::new("k").with_base_url(format!("{}/v1", server.base_url));

    let response = ChatModel::<()>::invoke(&model, &(), user("hi"))
        .await
        .expect("call succeeds");
    let usage = response.usage.expect("usage reported");
    assert_eq!(usage.cache_read_tokens, 40);
    assert_eq!(usage.cache_creation_tokens, 25);
    assert_eq!(
        usage.input_tokens, 100,
        "OpenAI includes cache tokens in the input total"
    );
}

// ---------------------------------------------------------------------------
// TOOL-2b — synthetic tool-call ids are unique across a run
// ---------------------------------------------------------------------------

/// A build that omits `id` used to emit `tool-0` on every turn, so one
/// transcript held several assistant messages declaring the same id.
#[tokio::test]
async fn id_less_tool_calls_get_distinct_ids_on_successive_turns() {
    let reply = json!({
        "id": "c1",
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{ "function": { "name": "ping", "arguments": "{}" } }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let server = MockServer::start(vec![Canned::ok(reply.clone()), Canned::ok(reply)]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let mut ids = Vec::new();
    for _ in 0..2 {
        let response = ChatModel::<()>::invoke(&model, &(), user("hi"))
            .await
            .expect("call succeeds");
        ids.push(response.tool_calls()[0].id.clone());
    }

    assert_ne!(
        ids[0], ids[1],
        "two turns sharing a synthetic id is an unresolvable pairing"
    );
    for id in &ids {
        assert!(id.starts_with("tacall-"), "{id}");
        // The prompt-guided protocol mints `ptc_{seq}_{slot}`; the schemes must
        // be unmistakably disjoint.
        assert!(!id.starts_with("ptc_"), "{id}");
    }
}

/// Gateways emit ids other providers reject on the way back
/// (`functions.write_todos:0`). Normalizing at the provider boundary keeps the
/// id and its paired result consistent, and must be deterministic.
#[tokio::test]
async fn non_conforming_provider_ids_are_normalized_deterministically() {
    let reply = json!({
        "id": "c1",
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "functions.write_todos:0",
                    "function": { "name": "ping", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let server = MockServer::start(vec![Canned::ok(reply.clone()), Canned::ok(reply)]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let mut ids = Vec::new();
    for _ in 0..2 {
        let response = ChatModel::<()>::invoke(&model, &(), user("hi"))
            .await
            .expect("call succeeds");
        ids.push(response.tool_calls()[0].id.clone());
    }

    assert_eq!(ids[0], ids[1], "normalization must be deterministic");
    assert!(
        ids[0]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "unexpected id: {}",
        ids[0]
    );
}

// ---------------------------------------------------------------------------
// C15 — context overflow is classified
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_context_overflow_is_classified_with_a_stable_code() {
    let server = MockServer::start(vec![Canned::error(
        400,
        json!({ "error": { "message": "This model's maximum context length is 4096 tokens" } }),
    )]);
    let model = OpenAiModel::ollama_at(&server.base_url, "llama3.2").expect("valid local URL");

    let error = ChatModel::<()>::invoke(&model, &(), user("hi"))
        .await
        .expect_err("a 400 fails the call");
    let tinyinference::Error::Provider(error) = error else {
        panic!("expected a typed provider error");
    };
    assert_eq!(error.provider, "ollama");
    assert_eq!(error.model.as_deref(), Some("llama3.2"));
    assert_eq!(error.code.as_deref(), Some("context_overflow"));
    assert!(error.message.contains("maximum context length"));
}

// ---------------------------------------------------------------------------
// REASON-7 — the Responses path carries the whole request
// ---------------------------------------------------------------------------

/// The body used to be `{model, input, instructions, stream, store,
/// max_output_tokens}` and everything else was silently dropped.
#[tokio::test]
async fn the_responses_path_no_longer_drops_the_request() {
    let server = MockServer::start(vec![Canned::ok(json!({ "output_text": "hi" }))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_model("gpt-5")
        .with_responses_api_primary();

    let request = user("hi")
        .with_tools(vec![ToolSchema::new("t", "d", json!({"type": "object"}))])
        .with_response_format(ResponseFormat::json_schema(
            "answer",
            json!({"type": "object"}),
        ))
        .with_temperature(0.3)
        .with_top_p(0.9)
        .with_seed(7)
        .with_stop_sequences(["STOP"])
        .with_continuation_id("resp_123")
        .with_reasoning_effort(ReasoningEffort::Low);
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let body = server.request_to("/v1/responses").body;
    assert_eq!(body["tools"][0]["name"], json!("t"));
    assert!(body.get("tool_choice").is_some());
    assert_eq!(body["text"]["format"]["type"], json!("json_schema"));
    assert_eq!(body["temperature"], json!(0.3));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["seed"], json!(7));
    assert_eq!(body["stop"], json!(["STOP"]));
    // `continuation_id` had a builder and no reader anywhere in the crate.
    assert_eq!(body["previous_response_id"], json!("resp_123"));
    assert_eq!(body["reasoning"]["effort"], json!("low"));
    // With `store: false`, reasoning is droppable unless it carries
    // `encrypted_content`, which only arrives when explicitly requested.
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
}

/// Reasoning items, the encrypted payload, and the usage breakdowns were all
/// unread on this path.
#[tokio::test]
async fn the_responses_path_reads_reasoning_and_cache_usage() {
    let server = MockServer::start(vec![Canned::ok(json!({
        "output": [
            {
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": "thinking about it" }],
                "encrypted_content": "enc-abc"
            },
            {
                "type": "message",
                "content": [{ "type": "output_text", "text": "the answer" }]
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": { "cached_tokens": 30, "cache_write_tokens": 10 },
            "output_tokens_details": { "reasoning_tokens": 12 }
        }
    }))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_responses_api_primary();

    let response = ChatModel::<()>::invoke(&model, &(), user("hi"))
        .await
        .expect("call succeeds");

    assert_eq!(
        response.text(),
        "the answer",
        "reasoning must not leak into the answer"
    );

    let usage = response.usage.expect("usage reported");
    assert_eq!(
        usage.cache_read_tokens, 30,
        "every cached token was billed at full rate"
    );
    assert_eq!(usage.cache_creation_tokens, 10);
    assert_eq!(usage.reasoning_tokens, 12);

    let thinking = response
        .message
        .content
        .iter()
        .find_map(|block| match block {
            tinyinference::message::ContentBlock::Thinking { text, signature } => {
                Some((text.clone(), signature.clone()))
            }
            _ => None,
        })
        .expect("reasoning surfaces as a Thinking block");
    assert_eq!(thinking.0, "thinking about it");
    assert_eq!(
        thinking.1.as_deref(),
        Some("enc-abc"),
        "the encrypted payload is what makes reasoning replayable under store:false"
    );
}

/// Tool results folded into anonymous assistant turns, erasing which call they
/// answered.
#[tokio::test]
async fn tool_results_keep_their_call_identity_on_the_responses_path() {
    let server = MockServer::start(vec![Canned::ok(json!({ "output_text": "ok" }))]);
    let model = OpenAiModel::new("k")
        .with_base_url(format!("{}/v1", server.base_url))
        .with_responses_api_primary();

    let request = ModelRequest::new(vec![
        Message::user("what is the weather"),
        Message::tool("call_abc", "sunny"),
    ]);
    let _ = ChatModel::<()>::invoke(&model, &(), request).await;

    let input = server.request_to("/v1/responses").body["input"].clone();
    let rendered = input.to_string();
    assert!(
        rendered.contains("call_abc"),
        "the tool call id must survive into the input: {rendered}"
    );
    let tool_item = input
        .as_array()
        .expect("input items")
        .last()
        .expect("last item");
    assert_eq!(
        tool_item["role"],
        json!("user"),
        "a tool result is not the assistant asserting a fact"
    );
}

// ---------------------------------------------------------------------------
// Cross-cutting: probing never happens on its own
// ---------------------------------------------------------------------------

/// Construction must stay free of network I/O — a constructor that blocks on a
/// round trip is unusable where this crate is embedded.
#[test]
fn construction_performs_no_network_io() {
    let server = MockServer::start(vec![Canned::ok(json!({}))]);
    let _ = OpenAiModel::ollama_at(&server.base_url, "llama3.2")
        .expect("valid local URL")
        .with_local_num_ctx(8192)
        .with_keep_alive("30m");
    assert!(
        server.paths().is_empty(),
        "probing and warm-up must be opt-in, not a side effect of construction"
    );
}
