//! LIVE local-runtime coverage: Ollama and LM Studio, through the same
//! [`OpenAiModel`] adapter that serves every hosted provider.
//!
//! [`live_provider_matrix`] already proves that a *configured* endpoint answers
//! chat, streaming, and a one-shot tool call. That is necessary but not
//! sufficient for a local runtime, because the ways local servers break are
//! specific to them and mostly invisible to a single-call probe:
//!
//! - they reject request shapes the hosted APIs accept (a named `tool_choice`
//!   object, `response_format: {"type": "json_object"}`) with an HTTP 400,
//! - they send **no** `Authorization` header and 401 on an unexpected one,
//! - they serve whatever model the operator loaded, under an id no preset can
//!   guess, and
//! - a small quantised model can emit one syntactically valid tool call and
//!   still fail to *use the result*, which is what an agent actually needs.
//!
//! So this file drives the whole loop rather than one call: discovery →
//! chat → streaming → one-shot tool call → **a full tool round trip through
//! [`AgentHarness`]** (model asks for the tool, the harness runs it, the model
//! consumes the result and answers) → structured JSON output.
//!
//! Every test runs against every runtime that is reachable, so one file covers
//! both Ollama and LM Studio and any future OpenAI-compatible local server.
//!
//! # Configuration
//!
//! | Variable | Default |
//! |---|---|
//! | `LOCAL_MODEL_TESTS` | unset — **required** to dial anything |
//! | `LOCAL_OLLAMA_BASE_URL` | `http://localhost:11434/v1` |
//! | `LOCAL_OLLAMA_MODEL` | first tool-capable id `GET /v1/models` advertises |
//! | `LOCAL_LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` |
//! | `LOCAL_LMSTUDIO_MODEL` | first non-embedding id `GET /v1/models` advertises |
//!
//! # Skips gracefully
//!
//! Dialling is opt-in via `LOCAL_MODEL_TESTS=1`, so a bare `cargo test` never
//! starts a multi-second local inference run. Set it and any runtime that is
//! not listening — or that has no usable model loaded — is reported and
//! skipped rather than failed, so a box running only Ollama still passes.
//!
//! # Run
//!
//! ```text
//! LOCAL_MODEL_TESTS=1 cargo test --test live_local_models -- --nocapture
//! ```

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use tinyagents_harness::Result;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::{AgentHarness, InvalidArgsPolicy, RunPolicy};
use tinyagents_harness::testkit::{EventRecorder, Trajectory};
use tinyagents_harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};
use tinyinference::message::Message;
use tinyinference::model::{
    ChatModel, ModelRequest, ModelStreamItem, ResponseFormat, StreamAccumulator, ToolChoice,
};
use tinyinference::providers::openai::OpenAiModel;
use tinyinference::providers::{ProviderKind, ProviderSpec};

/// Per-call ceiling. A cold local model has to load several GB off disk before
/// it emits its first token, so this is deliberately generous.
const TIMEOUT_MS: u64 = 180_000;

/// Token budget for every call here, sized for a **reasoning** model.
///
/// This is not padding. Local reasoning models (`qwen3` in both Ollama and LM
/// Studio) spend tokens on a hidden reasoning channel *before* emitting a
/// single visible character, and that channel draws from the same `max_tokens`
/// budget. Asking `qwen3-4b` for one word with `max_tokens: 32` returns
/// `finish_reason: "length"`, 30 reasoning tokens, and **empty** visible
/// content — a completion that looks like a provider bug and is really a budget
/// that never reached the answer.
///
/// The crate already treats this as a first-class failure mode — see
/// [`RunPolicy::truncated_empty_retries`], whose documentation names `qwen3` via
/// Ollama specifically — so the tests must not reintroduce it by being frugal.
/// A budget this size is what a host talking to local models should use.
///
/// [`RunPolicy::truncated_empty_retries`]: tinyagents_harness::runtime::RunPolicy::truncated_empty_retries
const MAX_TOKENS: u32 = 1024;

/// The weather the fake tool always reports. Distinctive enough that finding it
/// in the model's final answer proves the tool *result* reached the model,
/// rather than the model inventing a plausible temperature.
///
/// Both sentinels are chosen to survive paraphrase, which a live assertion
/// against a 3B model must account for. The temperature is positive because a
/// leading minus sign is routinely dropped when the model restates the value,
/// and the condition is matched on its stem (`sandstorm`) because the model
/// freely re-inflects the word it was given (`hailing` came back as `hail`).
const SENTINEL_TEMPERATURE: &str = "41";
const SENTINEL_CONDITION: &str = "sandstorms";
/// The substring actually searched for in the model's prose.
const SENTINEL_CONDITION_STEM: &str = "sandstorm";

// ---------------------------------------------------------------------------
// Runtime discovery
// ---------------------------------------------------------------------------

/// One local runtime under test.
struct LocalRuntime {
    /// Display name used in skip/failure messages.
    name: &'static str,
    kind: ProviderKind,
    base_url: String,
    model: String,
}

impl LocalRuntime {
    /// Builds the adapter for this runtime.
    ///
    /// Local runtimes need no credential, so the key is a placeholder that
    /// `from_spec` never puts on the wire: the [`ProviderKind`] presets carry
    /// `requires_api_key: false`, which selects `AuthStyle::None`.
    fn model(&self) -> OpenAiModel {
        let spec = ProviderSpec::for_kind(self.kind.clone())
            .with_base_url(&self.base_url)
            .with_model(&self.model);
        OpenAiModel::from_spec(spec, "local").expect("local runtime spec is complete")
    }
}

/// Ids that are embedding-only or otherwise not chat models. LM Studio and
/// Ollama both list embedding models alongside chat models in `/v1/models`,
/// and asking one to chat is an error that has nothing to do with the harness.
fn is_embedding_id(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    id.contains("embed") || id.contains("bge") || id.contains("rerank")
}

/// Chat model ids that are known **not** to support tool calling, so the
/// tool-loop tests would fail on the model rather than on the code under test.
fn is_tool_incapable_id(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    // Base/text-completion and vision-only tags in the common local catalogues.
    id.contains("-base") || id.contains("stable-diffusion") || id.contains("whisper")
}

/// Resolves one runtime, or returns the reason it cannot be tested.
///
/// Model choice is explicit (`*_MODEL`) or discovered from the server: a local
/// runtime serves whatever the operator loaded, so no default id can be
/// hard-coded without 404ing on most installs.
async fn discover(
    name: &'static str,
    kind: ProviderKind,
    url_var: &str,
    model_var: &str,
    default_url: &str,
) -> std::result::Result<LocalRuntime, String> {
    let base_url = std::env::var(url_var)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_url.to_string());

    // A model id is needed to construct the adapter at all, so probe with a
    // placeholder purely to reach `list_models`, which is model-independent.
    let probe_spec = ProviderSpec::for_kind(kind.clone())
        .with_base_url(&base_url)
        .with_model("probe");
    let probe = OpenAiModel::from_spec(probe_spec, "local")
        .map_err(|e| format!("{name}: invalid configuration: {e}"))?;

    let listed = probe
        .list_models()
        .await
        .map_err(|e| format!("{name}: not reachable at {base_url} ({e})"))?;

    if let Some(explicit) = std::env::var(model_var)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        return Ok(LocalRuntime {
            name,
            kind,
            base_url,
            model: explicit.trim().to_string(),
        });
    }

    let model = listed
        .iter()
        .map(|entry| entry.id.clone())
        .find(|id| !is_embedding_id(id) && !is_tool_incapable_id(id))
        .ok_or_else(|| {
            format!(
                "{name}: reachable at {base_url} but serves no chat model \
                 (saw {} id(s)); load one or set {model_var}",
                listed.len()
            )
        })?;

    Ok(LocalRuntime {
        name,
        kind,
        base_url,
        model,
    })
}

/// Every local runtime that is reachable and usable right now.
///
/// Returns an empty vector (after explaining why, on stderr) when the opt-in
/// switch is unset or nothing is listening, which is what lets these tests pass
/// on a machine with no local runtime at all.
async fn reachable_runtimes() -> Vec<LocalRuntime> {
    if std::env::var("LOCAL_MODEL_TESTS")
        .ok()
        .filter(|v| !v.trim().is_empty() && v != "0")
        .is_none()
    {
        eprintln!(
            "skipping live local-model tests: set LOCAL_MODEL_TESTS=1 to dial local runtimes \
             (LOCAL_MODEL_TESTS=1 cargo test --test live_local_models -- --nocapture)"
        );
        return Vec::new();
    }

    let candidates = [
        (
            "ollama",
            ProviderKind::Ollama,
            "LOCAL_OLLAMA_BASE_URL",
            "LOCAL_OLLAMA_MODEL",
            "http://localhost:11434/v1",
        ),
        (
            "lmstudio",
            ProviderKind::LmStudio,
            "LOCAL_LMSTUDIO_BASE_URL",
            "LOCAL_LMSTUDIO_MODEL",
            "http://localhost:1234/v1",
        ),
    ];

    let mut ready = Vec::new();
    for (name, kind, url_var, model_var, default_url) in candidates {
        match discover(name, kind, url_var, model_var, default_url).await {
            Ok(runtime) => {
                eprintln!(
                    "local runtime `{}` ready: {} @ {}",
                    runtime.name, runtime.model, runtime.base_url
                );
                ready.push(runtime);
            }
            Err(reason) => eprintln!("skipping {reason}"),
        }
    }
    // Opting in explicitly and then reaching nothing must not look like success.
    // Every assertion in this file is inside a `for` over this list, so an empty
    // list makes the whole suite pass while testing nothing at all — the exact
    // failure mode these tests exist to rule out.
    assert!(
        !ready.is_empty(),
        "LOCAL_MODEL_TESTS is set but no local runtime is reachable. Start Ollama \
         (`ollama serve` + `ollama pull llama3.2:3b`) or LM Studio (`lms server start` + \
         load a model), or unset LOCAL_MODEL_TESTS to skip."
    );
    ready
}

/// The [`RunPolicy`] a host should use to drive a small local model.
///
/// The crate default is now [`InvalidArgsPolicy::ReturnToolError`], which hands
/// the validation error back to the model as a tool result instead of aborting
/// the run. That change removed the original reason this helper existed: the
/// default used to be [`InvalidArgsPolicy::Fail`], which killed the whole run
/// the first time a model called a registered tool with schema-invalid
/// arguments — reasonable for a frontier model, where that is nearly always a
/// genuine bug, but unusably brittle for a 3B quantised model that omits a
/// required argument often enough to end most runs on the first tool call.
///
/// The helper still earns its keep, for a narrower reason.
/// [`InvalidArgsPolicy::NormalizeThenReturnToolError`] additionally repairs the
/// common provider-shape defects — JSON emitted as a string, a scalar sent
/// where an array is declared — *before* deciding the call is invalid. Those
/// defects are characteristic of small local models specifically, so the extra
/// normalization pass is worth requesting locally and not worth paying for by
/// default. Either way the recovery consumes a tool-call budget slot, so the
/// correction loop stays bounded.
///
/// This is observed behaviour, not a hypothetical: under the old default,
/// `llama3.2:3b` failed this file's tool-loop test with
/// `tool `get_weather` arguments.city is required`.
fn local_run_policy() -> RunPolicy {
    RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    }
}

fn base_request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        messages,
        max_tokens: Some(MAX_TOKENS),
        timeout_ms: Some(TIMEOUT_MS),
        ..ModelRequest::default()
    }
}

/// How many times a tool-calling assertion may re-roll the model.
///
/// `ToolChoice::Required` is a **request**, not a guarantee, and a 3B model
/// declines it often enough to matter: measured over 12 consecutive
/// `tool_choice: "required"` calls, `llama3.2:3b` via Ollama returned a
/// structured tool call 11 times and plain prose once. Sometimes it answers the
/// question from parametric memory instead.
///
/// That is a property of the model, not of the code under test, so asserting it
/// per-call would make this suite a coin flip. These tests ask the question that
/// is actually about our code — *can this runtime, through this adapter,
/// produce a well-formed tool call and consume its result?* — and a bounded
/// re-roll answers exactly that: a runtime that genuinely cannot do it fails all
/// `TOOL_ATTEMPTS` times, while one that can will not fail all of them.
///
/// Any host driving a local model needs the same allowance. A tool call is not
/// something you can assume happened — check, and retry.
const TOOL_ATTEMPTS: usize = 4;

/// Runs `attempt` up to [`TOOL_ATTEMPTS`] times, returning once one succeeds.
///
/// Panics with the final attempt's message when every try fails, so a genuine
/// breakage still surfaces the real error rather than a bare count.
async fn with_tool_reroll<F, Fut>(runtime: &LocalRuntime, label: &str, mut attempt: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<(), String>>,
{
    let mut last = String::new();
    for round in 1..=TOOL_ATTEMPTS {
        match attempt().await {
            Ok(()) => {
                if round > 1 {
                    eprintln!(
                        "  {}: {label} succeeded on attempt {round}/{TOOL_ATTEMPTS} \
                         (the model declined the tool on earlier tries)",
                        runtime.name
                    );
                }
                return;
            }
            Err(error) => last = error,
        }
    }
    panic!(
        "{}: {label} failed all {TOOL_ATTEMPTS} attempts against `{}`; last error: {last}",
        runtime.name, runtime.model
    );
}

// ---------------------------------------------------------------------------
// A real tool, with a real schema, that records what it was asked
// ---------------------------------------------------------------------------

/// A weather tool that returns a fixed, deliberately implausible reading.
///
/// Unlike [`FakeTool`](tinyagents_harness::testkit::FakeTool) this declares a
/// real JSON Schema with a required argument, which is the thing a small local
/// model actually has to get right.
struct WeatherTool {
    /// Every `city` argument the model supplied, in call order.
    seen: Mutex<Vec<String>>,
}

impl WeatherTool {
    fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }

    fn cities(&self) -> Vec<String> {
        self.seen.lock().expect("weather tool lock").clone()
    }

    fn schema_json() -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name, e.g. \"Paris\"." }
            },
            "required": ["city"]
        })
    }
}

#[async_trait]
impl Tool<()> for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Returns the current weather for a given city. Always use this tool for weather questions."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), Self::schema_json())
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        let city = call
            .arguments
            .get("city")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.seen.lock().expect("weather tool lock").push(city);
        Ok(ToolResult::text(
            call.id,
            "get_weather",
            format!("{SENTINEL_TEMPERATURE} degrees Celsius and {SENTINEL_CONDITION}",),
        ))
    }
}

/// Forces [`ToolChoice::Required`] on the **first** model call of a run, then
/// gets out of the way.
///
/// Without this the tool-loop test is really two tests wearing one coat: does
/// the 3B model *choose* to call the tool, and does the harness then feed the
/// result back correctly. The first is model judgment and is genuinely
/// stochastic — `llama3.2:3b` answers the weather question from parametric
/// memory roughly one run in six no matter how the system prompt is worded.
/// Only the second is a property of the code under test, so the choice is
/// forced and the loop behaviour is what gets asserted.
///
/// The force must not persist: with `Required` on every call the model can
/// never emit a final answer and the run would spin until it hit the tool-call
/// cap.
struct ForceFirstToolCall {
    calls: Mutex<usize>,
}

impl ForceFirstToolCall {
    fn new() -> Self {
        Self {
            calls: Mutex::new(0),
        }
    }
}

#[async_trait]
impl tinyagents_harness::middleware::Middleware<(), ()> for ForceFirstToolCall {
    fn name(&self) -> &str {
        "force_first_tool_call"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        let mut calls = self.calls.lock().expect("force-first-tool-call lock");
        if *calls == 0 {
            request.tool_choice = ToolChoice::Required;
        }
        *calls += 1;
        Ok(())
    }
}

/// The same tool as a bare schema, for the one-shot (harness-free) probe.
fn weather_schema() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Returns the current weather for a given city.",
        WeatherTool::schema_json(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `GET /v1/models` must answer, and advertise at least one usable id.
///
/// This is the discovery path a host uses to populate a model picker, and it is
/// the only way to learn a local runtime's model ids — there is no catalogue to
/// hard-code.
#[tokio::test]
async fn local_runtimes_advertise_their_loaded_models() {
    for runtime in reachable_runtimes().await {
        let listed = runtime
            .model()
            .list_models()
            .await
            .unwrap_or_else(|e| panic!("{}: list_models failed: {e}", runtime.name));

        assert!(
            !listed.is_empty(),
            "{}: a reachable runtime should advertise at least one model",
            runtime.name
        );
        assert!(
            listed.iter().any(|entry| entry.id == runtime.model),
            "{}: the selected model `{}` should appear in its own /models listing",
            runtime.name,
            runtime.model
        );
    }
}

/// A single-turn chat call must return non-empty assistant text.
#[tokio::test]
async fn local_runtimes_answer_a_single_turn_chat() {
    for runtime in reachable_runtimes().await {
        let response = runtime
            .model()
            .invoke(
                &(),
                base_request(vec![Message::user(
                    "Reply with exactly the single word: hello",
                )]),
            )
            .await
            .unwrap_or_else(|e| panic!("{}: chat failed: {e}", runtime.name));

        assert!(
            !response.text().trim().is_empty(),
            "{}: chat returned empty assistant text",
            runtime.name
        );
    }
}

/// Streaming must produce genuinely incremental deltas, not one final chunk.
///
/// A local server that buffers the whole completion and emits it as a single
/// SSE event technically "streams" but breaks every incremental consumer, so
/// the delta count is asserted, not just the merged text.
#[tokio::test]
async fn local_runtimes_stream_incremental_deltas() {
    for runtime in reachable_runtimes().await {
        let mut stream = runtime
            .model()
            .stream(
                &(),
                base_request(vec![Message::user("Count from one to five, in words.")]),
            )
            .await
            .unwrap_or_else(|e| panic!("{}: stream failed to start: {e}", runtime.name));

        let mut deltas = 0usize;
        let mut accumulator = StreamAccumulator::new();
        while let Some(item) = stream.next().await {
            if matches!(item, ModelStreamItem::MessageDelta(_)) {
                deltas += 1;
            }
            accumulator.push(&item);
        }
        let response = accumulator
            .finish()
            .unwrap_or_else(|e| panic!("{}: stream produced no response: {e}", runtime.name));

        assert!(
            deltas > 1,
            "{}: expected incremental deltas, got {deltas}",
            runtime.name
        );
        assert!(
            !response.text().trim().is_empty(),
            "{}: streamed text was empty",
            runtime.name
        );
    }
}

/// The model must emit one well-formed tool call with parseable arguments.
///
/// `ToolChoice::Required` is used deliberately: local servers historically
/// reject a *named* tool choice object, and the transport degrades that shape
/// for local runtimes. This asserts the degradation actually works end to end.
#[tokio::test]
async fn local_runtimes_emit_a_parseable_tool_call() {
    for runtime in reachable_runtimes().await {
        let model = runtime.model();
        with_tool_reroll(&runtime, "one-shot tool call", || async {
            let mut request = base_request(vec![Message::user(
                "What is the weather in Paris right now? Call the get_weather tool.",
            )]);
            request.tools = vec![weather_schema()];
            request.tool_choice = ToolChoice::Required;

            let response = model
                .invoke(&(), request)
                .await
                .map_err(|e| format!("forced tool call failed: {e}"))?;

            let call = response
                .message
                .tool_calls
                .first()
                .ok_or_else(|| "no tool call returned".to_string())?;

            if call.name != "get_weather" {
                return Err(format!("called {:?}, expected get_weather", call.name));
            }
            if let Some(invalid) = &call.invalid {
                return Err(format!("tool arguments did not parse: {invalid}"));
            }
            // The requested city must have reached us, but *where* in the
            // arguments is a model-quality question this layer cannot fix.
            // Small models bury it under a wrapper key or echo the tool's own
            // JSON Schema with the value filled in — all captured shapes are
            // listed on `unwrap_wrapped_arguments`. Recovering them is the
            // harness's job and is asserted by
            // `local_runtimes_complete_a_full_tool_loop`; the provider
            // adapter's own contract is only that it surfaced a well-formed
            // call carrying the argument somewhere.
            let rendered = call.arguments.to_string();
            if !rendered.contains("Paris") {
                return Err(format!("tool call carried no `city` argument: {rendered}"));
            }
            Ok(())
        })
        .await;
    }
}

/// The full agent loop: the model asks for a tool, the harness runs it, and the
/// model answers **using the result**.
///
/// This is the test that a one-shot tool probe cannot replace. Emitting a tool
/// call is easy; correctly consuming a `tool` role message and producing a
/// grounded final answer is where small quantised local models — and any bug in
/// how the adapter serialises tool results back onto the wire — actually break.
#[tokio::test]
async fn local_runtimes_complete_a_full_tool_loop() {
    for runtime in reachable_runtimes().await {
        with_tool_reroll(&runtime, "full tool loop", || async {
            let tool = Arc::new(WeatherTool::new());

            let mut harness: AgentHarness<()> = AgentHarness::new();
            harness.register_tool(tool.clone());
            harness
                .register_model("local", Arc::new(runtime.model()))
                .set_default_model("local")
                .with_policy(local_run_policy())
                .push_middleware(Arc::new(ForceFirstToolCall::new()));

            let recorder = EventRecorder::new();
            let ctx = RunContext::new(RunConfig::new("live-local-tool-loop"), ())
                .with_events(recorder.sink());

            let run = harness
                .invoke_in_context(
                    &(),
                    ctx,
                    vec![
                        Message::system(
                            "You answer weather questions. You must call the get_weather tool \
                             to obtain the weather, then state the temperature and condition it \
                             returned. Never invent weather data.",
                        ),
                        Message::user("What is the weather in Paris right now?"),
                    ],
                )
                .await
                .map_err(|e| format!("tool-loop run failed: {e}"))?;

            let traj = Trajectory::from_events(recorder.events());
            if !traj.completed() {
                return Err("the run did not reach completion".to_string());
            }
            if !traj.tool_was_called("get_weather") {
                return Err("the model never called get_weather".to_string());
            }

            let cities = tool.cities();
            if !cities.iter().any(|c| c.to_lowercase().contains("paris")) {
                return Err(format!(
                    "the tool was not asked about Paris, got {cities:?}"
                ));
            }

            // Two model calls minimum: one to request the tool, one to consume
            // its result. A single call would mean the loop never fed the
            // result back — the whole point of this test.
            if run.model_calls < 2 {
                return Err(format!(
                    "expected at least 2 model calls (request + consume), got {}",
                    run.model_calls
                ));
            }

            let final_text = run.text().unwrap_or_default();
            if final_text.trim().is_empty() {
                return Err("the loop produced no final answer".to_string());
            }
            // The answer must be grounded in what the tool returned rather than
            // in what the model already believes about the weather in Paris.
            let grounded = final_text.contains(SENTINEL_TEMPERATURE)
                || final_text.to_lowercase().contains(SENTINEL_CONDITION_STEM);
            if !grounded {
                return Err(format!(
                    "the final answer ignored the tool result (expected \
                     {SENTINEL_TEMPERATURE} or {SENTINEL_CONDITION_STEM}): {final_text}"
                ));
            }
            Ok(())
        })
        .await;
    }
}

/// Structured JSON output must come back as parseable JSON.
///
/// Local servers reject `response_format: {"type": "json_object"}` with a 400
/// and want a `json_schema` instead; the transport degrades that shape for
/// local runtimes, and this asserts the degraded request is both accepted and
/// honoured.
#[tokio::test]
async fn local_runtimes_produce_structured_json_output() {
    for runtime in reachable_runtimes().await {
        let mut request = base_request(vec![Message::user(
            "Paris is the capital of France and has about 2.1 million residents. \
                 Return it as JSON with keys `city` and `country`.",
        )]);
        request.response_format = Some(ResponseFormat::JsonObject);

        let response = runtime
            .model()
            .invoke(&(), request)
            .await
            .unwrap_or_else(|e| panic!("{}: structured output call failed: {e}", runtime.name));

        let text = response.text();
        let parsed: Value = serde_json::from_str(text.trim()).unwrap_or_else(|e| {
            panic!(
                "{}: json_object response did not parse as JSON ({e}): {text}",
                runtime.name
            )
        });
        assert!(
            parsed.is_object(),
            "{}: expected a JSON object, got {parsed}",
            runtime.name
        );
    }
}

// ---------------------------------------------------------------------------
// Offline unit coverage
//
// These run on every `cargo test` with no network and no local server, so the
// discovery/classification rules stay pinned even when the live tests skip.
// ---------------------------------------------------------------------------

#[test]
fn embedding_ids_are_excluded_from_chat_model_discovery() {
    for id in [
        "nomic-embed-text:latest",
        "text-embedding-nomic-embed-text-v1.5",
        "bge-m3",
        "BAAI/bge-reranker-v2-m3",
    ] {
        assert!(
            is_embedding_id(id),
            "{id} should be treated as embedding-only"
        );
    }
    for id in ["llama3.2:3b", "qwen3-4b", "gpt-oss-20b"] {
        assert!(!is_embedding_id(id), "{id} is a chat model");
    }
}

#[test]
fn tool_incapable_ids_are_excluded_from_chat_model_discovery() {
    assert!(is_tool_incapable_id("llama-3.2-3b-base"));
    assert!(is_tool_incapable_id("whisper-large-v3"));
    assert!(!is_tool_incapable_id("qwen3-4b"));
}

#[test]
fn local_presets_need_no_credential_and_default_to_their_own_ports() {
    let ollama = ProviderSpec::for_kind(ProviderKind::Ollama);
    assert!(!ollama.requires_api_key);
    assert_eq!(ollama.base_url, "http://localhost:11434/v1");

    let lmstudio = ProviderSpec::for_kind(ProviderKind::LmStudio);
    assert!(!lmstudio.requires_api_key);
    assert_eq!(lmstudio.base_url, "http://localhost:1234/v1");
    // LM Studio serves whatever GGUF is loaded, so there is no default id to
    // guess; callers must supply one.
    assert!(
        lmstudio.model.is_empty(),
        "the LM Studio preset must not guess a model id"
    );
}

/// Pins the recovering crate default, and the stronger normalizing policy a
/// local host still opts into on top of it.
///
/// This test previously asserted the default was [`InvalidArgsPolicy::Fail`]
/// and said that if the default ever became a recovering policy, this
/// assertion should not simply be flipped — [`local_run_policy`]'s rationale
/// had to be revisited first. The default did change, and it was: the helper's
/// doc comment no longer justifies itself by "the default aborts the run", but
/// by the provider-shape normalization pass that `ReturnToolError` alone does
/// not perform. The assertion below is updated only because that reasoning was
/// re-derived, not to make a red test green.
#[test]
fn invalid_tool_arguments_recover_by_default_and_local_hosts_add_normalization() {
    assert_eq!(
        RunPolicy::default().invalid_args,
        InvalidArgsPolicy::ReturnToolError,
        "the crate default hands the validation error back to the model \
         rather than aborting the run"
    );
    assert_eq!(
        local_run_policy().invalid_args,
        InvalidArgsPolicy::NormalizeThenReturnToolError,
        "local runtimes need the recovering policy so a small model can self-correct"
    );
}

#[test]
fn a_local_runtime_spec_builds_without_an_api_key() {
    let spec = ProviderSpec::for_kind(ProviderKind::LmStudio).with_model("qwen3-4b");
    let model = OpenAiModel::from_spec(spec, "local").expect("local spec builds");
    assert_eq!(model.base_url(), "http://localhost:1234/v1");
    assert_eq!(model.model(), "qwen3-4b");
}
