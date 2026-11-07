//! End-to-end coverage for deferred tool discovery in the agent loop.
//!
//! A `Deferred` tool stays out of the initial request's `tools` array; then
//! the loop advertises the `tool_search` bridge, answers `tool_search` from
//! its own catalogue, admits a deferred tool called by its own name exactly
//! like any registered tool, and promotes a search match's typed declaration
//! on the next request. The transcript records the promotion for resumed runs.
//! There is no call wrapper: `tool_search` is the only intrinsic tool.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::events::{AgentEvent, RecordingListener};
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy, UnknownToolPolicy};
use tinyagents_harness::testkit::FakeTool;
use tinyagents_harness::tool::discover::{
    DiscoveryOverrides, TOOL_SEARCH_NAME, ToolDiscoveryPolicy,
};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolExposure, ToolResult};

/// A tool with a chosen exposure that records what it was called with.
struct ExposedTool {
    name: &'static str,
    description: &'static str,
    exposure: ToolExposure,
    calls: Mutex<Vec<Value>>,
}

impl ExposedTool {
    fn new(name: &'static str, description: &'static str, exposure: ToolExposure) -> Arc<Self> {
        Arc::new(Self {
            name,
            description,
            exposure,
            calls: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Tool for ExposedTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker symbol."}},
            "required": ["symbol"]
        })
    }

    fn exposure(&self) -> ToolExposure {
        self.exposure
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.calls.lock().unwrap().push(args.clone());
        Ok(ToolResult::success(format!(
            "{} → {}",
            self.name,
            args["symbol"].as_str().unwrap_or("?")
        )))
    }
}

struct ChangedDeferredTool;

#[async_trait]
impl Tool for ChangedDeferredTool {
    fn name(&self) -> &str {
        "stock_quote"
    }
    fn description(&self) -> &str {
        "A changed description after restart."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{"symbol":{"type":"integer"}},"required":["symbol"]})
    }
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }
    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("unused"))
    }
}

/// A scripted model that records the `tools` array of every request.
struct RecordingModel {
    responses: Mutex<Vec<ModelResponse>>,
    tools_seen: Mutex<Vec<String>>,
    messages_seen: Mutex<Vec<Vec<Message>>>,
}

impl RecordingModel {
    fn new(responses: Vec<ModelResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into_iter().rev().collect()),
            tools_seen: Mutex::new(Vec::new()),
            messages_seen: Mutex::new(Vec::new()),
        })
    }

    /// The transcript of every request, in order.
    fn messages_seen(&self) -> Vec<Vec<Message>> {
        self.messages_seen.lock().unwrap().clone()
    }

    fn tools_seen(&self) -> Vec<String> {
        self.tools_seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.tools_seen
            .lock()
            .unwrap()
            .push(serde_json::to_string(&request.tools).expect("tools serialise"));
        self.messages_seen
            .lock()
            .unwrap()
            .push(request.messages.clone());
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| text("out of script")))
    }
}

fn tool_call(id: &str, name: &str, arguments: Value) -> ModelResponse {
    tinyagents_harness::testkit::tool_call_response(ToolCall::new(id, name, arguments))
        .with_usage(Usage::new(1, 1))
}

fn text(body: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(body.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(1, 1)),
            origin: None,
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

struct CaptureMiddleware {
    listener: Arc<RecordingListener>,
}

#[async_trait]
impl Middleware<(), ()> for CaptureMiddleware {
    fn name(&self) -> &str {
        "capture"
    }

    async fn before_agent(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
    ) -> tinyagents_harness::Result<()> {
        ctx.events.subscribe(self.listener.clone());
        Ok(())
    }
}

/// A `before_tool` hook that records the tool names it is asked about.
struct BeforeToolSpy {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for BeforeToolSpy {
    fn name(&self) -> &str {
        "before_tool_spy"
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        call: &mut ToolCall,
    ) -> tinyagents_harness::Result<()> {
        self.seen.lock().unwrap().push(call.name.clone());
        Ok(())
    }
}

fn tool_names(tools_json: &str) -> Vec<String> {
    serde_json::from_str::<Vec<Value>>(tools_json)
        .expect("tools array")
        .into_iter()
        .map(|tool| tool["name"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn deferred_tool_is_promoted_after_search_and_restored_on_resume() {
    let listener = Arc::new(RecordingListener::new());
    let before_tool = Arc::new(Mutex::new(Vec::new()));
    let deferred = ExposedTool::new(
        "stock_quote",
        "Fetch the latest price for a ticker symbol.",
        ToolExposure::Deferred,
    );
    let unrelated = ExposedTool::new(
        "weather_forecast",
        "Report the weather for a city.",
        ToolExposure::Deferred,
    );
    let hidden = ExposedTool::new("internal_step", "Host-only step.", ToolExposure::Hidden);
    let model = RecordingModel::new(vec![
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "price of a ticker", "limit": 1}),
        ),
        // A revealed tool is called by its own name, with no wrapper.
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        tool_call("c3", "stock_quote", json!({"symbol": "XYZ"})),
        // A hidden tool is unknown to the model even by name.
        tool_call("c4", "internal_step", json!({"symbol": "no"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(deferred.clone())
        .register_tool(unrelated.clone())
        .register_tool(hidden.clone())
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(BeforeToolSpy {
            seen: before_tool.clone(),
        }))
        .with_policy(RunPolicy {
            // `ToolSearched.query` follows the same `capture.tool_io` gate as
            // a normal tool call's arguments; enable it so this test's
            // assertion on the recorded query is meaningful.
            capture: tinyagents_harness::runtime::PayloadCapture {
                tool_io: true,
                ..Default::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("what is ACME trading at?")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("done".to_string()));

    // The first request stays small. Search promotes the matched typed schema
    // on the next request, and that declaration remains stable thereafter.
    let seen = model.tools_seen();
    assert_eq!(seen.len(), 5);
    assert_eq!(tool_names(&seen[0]), vec!["read_file", TOOL_SEARCH_NAME]);
    assert!(seen[1..].iter().all(|tools| tools == &seen[1]));
    assert_eq!(
        tool_names(&seen[1]),
        vec!["read_file", "stock_quote", TOOL_SEARCH_NAME]
    );
    let promoted: Vec<Value> = serde_json::from_str(&seen[1]).unwrap();
    let stock = promoted
        .iter()
        .find(|tool| tool["name"] == "stock_quote")
        .unwrap();
    assert_eq!(
        stock["parameters"]["properties"]["symbol"]["type"],
        "string"
    );
    assert_eq!(stock["parameters"]["required"], json!(["symbol"]));
    assert!(!seen[1].contains("\"name\":\"weather_forecast\""));
    assert!(!seen[0].contains("Fetch the latest price for a ticker symbol."));
    // The manifest names the deferred tool without its schema.
    assert!(seen[0].contains("- stock_quote: Fetch the latest price for a ticker symbol"));
    assert!(!seen[0].contains("internal_step"));

    let (_, recorded) = tinyinference_llm::message::replay_system_state(&run.messages);
    assert!(recorded.iter().any(|tool| tool.name == "stock_quote"));
    let resumed_model = RecordingModel::new(vec![text("resumed")]);
    let mut resumed_harness: AgentHarness<()> = AgentHarness::new();
    resumed_harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", resumed_model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(Arc::new(ChangedDeferredTool));
    resumed_harness.register_tool(unrelated);
    let mut resumed_messages = run.messages.clone();
    resumed_messages.push(Message::user("quote another stock"));
    resumed_harness
        .invoke_default(&(), resumed_messages)
        .await
        .unwrap();
    let resumed_tools: Vec<Value> = serde_json::from_str(&resumed_model.tools_seen()[0]).unwrap();
    let resumed_stock = resumed_tools
        .iter()
        .find(|tool| tool["name"] == "stock_quote")
        .unwrap();
    assert_eq!(
        resumed_stock["parameters"]["properties"]["symbol"]["type"],
        "string"
    );

    // Both by-name calls reached the real tool.
    let calls = deferred.calls.lock().unwrap().clone();
    assert_eq!(
        calls,
        vec![json!({"symbol": "ACME"}), json!({"symbol": "XYZ"})]
    );
    assert!(hidden.calls.lock().unwrap().is_empty());

    // `before_tool` saw the real name of every deferred call; `tool_search`
    // is answered intrinsically before any hook runs.
    let hooks = before_tool.lock().unwrap().clone();
    assert_eq!(hooks, vec!["stock_quote", "stock_quote", "internal_step"]);

    // Transcript: the search answer carries the full schema; the hidden call
    // was answered as unknown, listing only model-callable names.
    let tool_messages: Vec<String> = run
        .messages
        .iter()
        .filter(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .collect();
    assert!(tool_messages[0].contains("\"name\": \"stock_quote\""));
    assert!(tool_messages[0].contains("\"symbol\""));
    assert!(tool_messages[1].contains("stock_quote → ACME"));
    assert!(tool_messages[3].contains("unknown tool `internal_step`"));
    // The corrective sends the model to discovery rather than dumping every
    // callable name, and never leaks the hidden tool as a suggestion.
    assert!(tool_messages[3].contains("call `tool_search`"));
    assert!(!tool_messages[3].contains("valid tools"));
    let rest = tool_messages[3]
        .split_once("unknown tool `internal_step`")
        .map(|(_, rest)| rest)
        .expect("corrective names the requested tool");
    assert!(!rest.contains("internal_step"));

    // Events make the surface auditable.
    let events: Vec<AgentEvent> = listener.events().into_iter().map(|r| r.event).collect();
    assert!(events.iter().any(|event| matches!(
        event,
        // `direct` counts only the `read_file` Direct-exposure tool: the
        // intrinsic bridge schema is implied by `deferred: 1`, not
        // double-counted into `direct` (see `ToolsAdvertised`'s doc comment).
        AgentEvent::ToolsAdvertised { direct: 1, deferred: 2, schema_bytes } if *schema_bytes > 0
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolSearched { matched: 1, query, .. } if query == "price of a ticker"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::DeferredToolCall { tool_name, .. } if tool_name == "stock_quote"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolStarted { tool_name, .. } if tool_name == "stock_quote"
    )));
    // Each by-name call to a deferred tool is reported once.
    let deferred_calls = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::DeferredToolCall { tool_name, .. } if tool_name == "stock_quote"
            )
        })
        .count();
    assert_eq!(deferred_calls, 2);
}

#[tokio::test]
async fn disabled_discovery_drops_the_bridge_and_keeps_direct_calls_working() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"query": "quote"})),
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(deferred.clone())
        .with_policy(RunPolicy {
            discovery: ToolDiscoveryPolicy {
                enabled: false,
                ..ToolDiscoveryPolicy::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(tool_names(&model.tools_seen()[0]), vec!["read_file"]);
    // `tool_search` is unknown when nothing advertised it …
    let first_tool_message = run
        .messages
        .iter()
        .find(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .unwrap();
    assert!(first_tool_message.contains("unknown tool `tool_search`"));
    // … but a deferred tool called by name still runs: deferral only subtracts
    // from the wire, never from what the host registered.
    assert_eq!(deferred.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn no_deferred_tools_means_no_bridge() {
    let model = RecordingModel::new(vec![text("done")]);
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")));
    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");
    assert_eq!(tool_names(&model.tools_seen()[0]), vec!["read_file"]);
}

/// Regression: `ToolSearched.query` used to be recorded verbatim regardless
/// of `RunPolicy::capture.tool_io`, so a run left at the payload-free default
/// still journaled/exported the model's raw search text — the same privacy
/// class as a normal tool call's arguments, which *do* honor that gate. With
/// the default (disabled) capture policy the query must come through empty.
#[tokio::test]
async fn tool_searched_query_is_payload_free_by_default() {
    let listener = Arc::new(RecordingListener::new());
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let model = RecordingModel::new(vec![
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "sensitive tenant text"}),
        ),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(deferred)
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));
    // No `RunPolicy` override: default `capture.tool_io` is `false`.

    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    let events: Vec<AgentEvent> = listener.events().into_iter().map(|r| r.event).collect();
    let searched = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolSearched { query, .. } => Some(query.clone()),
            _ => None,
        })
        .expect("a ToolSearched event was emitted");
    assert_eq!(
        searched, "",
        "the query must not be captured under the payload-free default policy"
    );
}

#[tokio::test]
async fn host_registered_tool_search_wins_over_the_intrinsic_bridge() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let host_search = ExposedTool::new(
        TOOL_SEARCH_NAME,
        "The host's own search tool.",
        ToolExposure::Direct,
    );
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"symbol": "anything"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(deferred.clone())
        .register_tool(host_search.clone());

    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // The host's `tool_search` keeps its slot and its description, and the
    // intrinsic bridge adds nothing: it has no other tool to advertise.
    let tools = model.tools_seen()[0].clone();
    assert_eq!(tool_names(&tools), vec![TOOL_SEARCH_NAME]);
    assert!(tools.contains("The host's own search tool."));
    assert!(!tools.contains("deferred tool(s) are searchable"));
    // …and the call went to the host's tool, not the intrinsic answer.
    assert_eq!(host_search.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn unknown_tool_corrective_does_not_advertise_a_host_registered_tool_search() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let host_search = ExposedTool::new(
        TOOL_SEARCH_NAME,
        "The host's own search tool.",
        ToolExposure::Direct,
    );
    let model = RecordingModel::new(vec![
        tool_call("c1", "nonexistent_tool", json!({"x": 1})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(deferred)
        .register_tool(host_search)
        .with_policy(RunPolicy {
            unknown_tool: UnknownToolPolicy::ReturnToolError,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    let message = run
        .messages
        .iter()
        .find(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .unwrap();
    assert!(
        message.starts_with("unknown tool `nonexistent_tool`"),
        "{message}"
    );
    // The host's `tool_search` shadows the intrinsic bridge, so the corrective
    // must not promise the bridge's discovery behaviour.
    assert!(!message.contains("call `tool_search`"), "{message}");
}

/// Regression: the collision check that decides whether to advertise an
/// intrinsic bridge schema used to look only at `tool_schemas` (the `Direct`
/// set), while admission's own collision check (`self.tools.dispatch`) sees
/// every exposure. A `Hidden` tool registered as `tool_search` therefore used
/// to be missed here: the loop still advertised the intrinsic `tool_search`
/// schema, but admission suppressed the intrinsic handler for a name it
/// recognized as registered, so a model that used the advertised bridge got
/// an unknown-tool answer instead of a search result. Both paths must use the
/// same collision rule.
#[tokio::test]
async fn hidden_registration_suppresses_the_matching_bridge_schema() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let hidden_search = ExposedTool::new(
        TOOL_SEARCH_NAME,
        "Host-internal, never model-visible.",
        ToolExposure::Hidden,
    );
    let model = RecordingModel::new(vec![
        tool_call("c1", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(deferred.clone())
        .register_tool(hidden_search.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("done".to_string()));

    // The intrinsic `tool_search` schema is suppressed (the registry already
    // owns that name, even though it is Hidden and unreachable itself), and it
    // was the only bridge schema, so nothing is advertised.
    let tools = model.tools_seen()[0].clone();
    assert!(tool_names(&tools).is_empty(), "{tools}");
    // The deferred tool is still reachable directly by its own name, and the
    // hidden registration never ran.
    assert_eq!(deferred.calls.lock().unwrap().len(), 1);
    assert!(hidden_search.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn tool_schemas_projection_applies_to_wire_and_catalog() {
    use tinyagents_harness::tool::{SchemaCompaction, SchemaPreparation};

    let long = "d".repeat(400);
    let direct = Arc::new(FakeTool::returning("read_file", "contents"));
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"query": "quote"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(direct)
        .register_tool(deferred)
        .register_tool(ExposedTool::new(
            "verbose_direct",
            Box::leak(long.clone().into_boxed_str()),
            ToolExposure::Direct,
        ))
        .with_policy(RunPolicy {
            tool_schemas: Some(
                SchemaPreparation::openai().with_compaction(SchemaCompaction {
                    max_description_bytes: Some(40),
                    max_schema_bytes: None,
                }),
            ),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // The verbose direct description was clipped on the wire…
    let tools: Vec<Value> = serde_json::from_str(&model.tools_seen()[0]).unwrap();
    let verbose = tools
        .iter()
        .find(|t| t["name"] == "verbose_direct")
        .unwrap();
    assert!(verbose["description"].as_str().unwrap().len() <= 40);
    assert!(!model.tools_seen()[0].contains(&long));
    // …and the search answer for the deferred tool went through the same
    // projection (its short description is untouched but present).
    let answer = run
        .messages
        .iter()
        .find(|m| matches!(m, Message::Tool(_)))
        .map(Message::text)
        .unwrap();
    assert!(answer.contains("\"name\": \"stock_quote\""));

    // Regression: the intrinsic `tool_search` bridge schema used
    // to be appended *after* provider preparation ran, so it reached the
    // wire unprojected — a `max_description_bytes` budget (or a Gemini
    // `minimum`/`maximum` strip) never applied to them even though the same
    // policy is configured for the whole run. `tool_search`'s description
    // embeds the deferred-tool manifest, which is comfortably over 40 bytes
    // unprojected, so this is a real assertion, not a vacuous one.
    let bridge_search = tools
        .iter()
        .find(|t| t["name"] == TOOL_SEARCH_NAME)
        .unwrap();
    assert!(
        bridge_search["description"].as_str().unwrap().len() <= 40,
        "bridge schema description was not projected through the run's SchemaPreparation"
    );
}

#[tokio::test]
async fn a_deferred_event_is_emitted_only_for_a_deferred_tool_called_by_name() {
    let listener = Arc::new(RecordingListener::new());
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let direct = ExposedTool::new("read_file", "Read.", ToolExposure::Direct);
    let hidden = ExposedTool::new("internal_step", "Host-only.", ToolExposure::Hidden);
    let model = RecordingModel::new(vec![
        // A *direct* tool: runs, but is not a deferred call.
        tool_call("c1", "read_file", json!({"symbol": "A"})),
        // A *hidden* tool: rejected as unknown, not a deferred call.
        tool_call("c2", "internal_step", json!({"symbol": "B"})),
        // A deferred tool: the one case that is a deferred call.
        tool_call("c3", "stock_quote", json!({"symbol": "C"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(direct.clone())
        .register_tool(deferred.clone())
        .register_tool(hidden.clone())
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(direct.calls.lock().unwrap().len(), 1);
    assert!(hidden.calls.lock().unwrap().is_empty());
    assert_eq!(deferred.calls.lock().unwrap().len(), 1);

    let deferred_events: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::DeferredToolCall { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert_eq!(deferred_events, vec!["stock_quote"]);
}

/// A harness over one read tool, a matched deferred tool and an unmatched one,
/// scripted with `model`.
fn discovery_harness(
    model: &Arc<RecordingModel>,
    discovery: ToolDiscoveryPolicy,
) -> (AgentHarness<()>, Arc<ExposedTool>, Arc<ExposedTool>) {
    let stock = ExposedTool::new(
        "stock_quote",
        "Fetch the latest price for a ticker symbol.",
        ToolExposure::Deferred,
    );
    let weather = ExposedTool::new(
        "weather_forecast",
        "Report the weather for a city.",
        ToolExposure::Deferred,
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(stock.clone())
        .register_tool(weather.clone())
        .with_policy(RunPolicy {
            discovery,
            ..RunPolicy::default()
        });
    (harness, stock, weather)
}

fn search_then_call_script(arguments: Value) -> Vec<ModelResponse> {
    vec![
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "price of a ticker", "limit": 1}),
        ),
        tool_call("c2", "stock_quote", arguments),
        text("done"),
    ]
}

fn tool_messages(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .collect()
}

#[tokio::test]
async fn promotion_off_keeps_the_tools_list_byte_identical_and_validates_direct_calls() {
    let model = RecordingModel::new(vec![
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "price of a ticker", "limit": 1}),
        ),
        // Wrong type: the matched tool's schema still validates a direct call.
        tool_call("c2", "stock_quote", json!({"symbol": 5})),
        tool_call("c3", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
    ]);
    let (harness, stock, _) = discovery_harness(
        &model,
        ToolDiscoveryPolicy::default().with_promote_found_tools(false),
    );

    let run = harness
        .invoke_default(&(), vec![Message::user("what is ACME trading at?")])
        .await
        .expect("run succeeds");

    let seen = model.tools_seen();
    assert_eq!(seen.len(), 4);
    assert!(
        seen.iter().all(|tools| tools == &seen[0]),
        "tools never change"
    );
    assert_eq!(tool_names(&seen[0]), vec!["read_file", TOOL_SEARCH_NAME]);
    // No tool-change patch was written to the transcript either.
    assert!(
        tinyinference_llm::message::replay_system_state(&run.messages)
            .1
            .is_empty()
    );

    let results = tool_messages(&run.messages);
    // The schema travels in the search result instead.
    assert!(results[0].contains("\"name\": \"stock_quote\""));
    assert!(results[0].contains("\"symbol\""));
    // The invalid call was refused by schema validation and never ran; the
    // valid one reached the tool.
    assert!(results[1].contains("invalid arguments for tool `stock_quote`"));
    assert!(results[2].contains("stock_quote \u{2192} ACME"));
    assert_eq!(
        stock.calls.lock().unwrap().clone(),
        vec![json!({"symbol": "ACME"})]
    );
}

#[tokio::test]
async fn promotion_off_admits_an_unmatched_deferred_tool_exactly_as_promotion_on_does() {
    // Admission never depended on a prior search: a deferred tool is callable
    // by its own name whether or not it was matched (see `tool-discovery.md`).
    // Turning promotion off must not change that rule in either direction.
    let script = || {
        vec![
            tool_call(
                "c1",
                TOOL_SEARCH_NAME,
                json!({"query": "price of a ticker", "limit": 1}),
            ),
            tool_call("c2", "weather_forecast", json!({"symbol": "Oslo"})),
            text("done"),
        ]
    };
    let mut outcomes = Vec::new();
    for promote in [true, false] {
        let model = RecordingModel::new(script());
        let (harness, _, weather) = discovery_harness(
            &model,
            ToolDiscoveryPolicy::default().with_promote_found_tools(promote),
        );
        let run = harness
            .invoke_default(&(), vec![Message::user("weather?")])
            .await
            .expect("run succeeds");
        outcomes.push((
            tool_messages(&run.messages)[1].clone(),
            weather.calls.lock().unwrap().len(),
        ));
    }
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(outcomes[1].1, 1, "the unmatched tool still ran once");
}

#[tokio::test]
async fn sessions_in_one_process_each_keep_their_own_promotion_setting() {
    // One harness, one process: the harness-wide policy promotes (today's
    // behaviour), and a session opts out through its run config. The sessions
    // run back to back on the same harness; each keeps its own behaviour.
    let mut script = search_then_call_script(json!({"symbol": "A"}));
    script.extend(search_then_call_script(json!({"symbol": "B"})));
    let model = RecordingModel::new(script);
    let (harness, _, _) = discovery_harness(&model, ToolDiscoveryPolicy::default());

    harness
        .invoke(
            &(),
            (),
            RunConfig::new("promoting"),
            vec![Message::user("a")],
        )
        .await
        .expect("promoting session");
    let off = RunConfig::new("quiet").with_discovery_overrides(DiscoveryOverrides {
        promote_found_tools: Some(false),
        ..DiscoveryOverrides::default()
    });
    harness
        .invoke(&(), (), off, vec![Message::user("b")])
        .await
        .expect("quiet session");

    let seen = model.tools_seen();
    assert_eq!(seen.len(), 6);
    assert_eq!(
        tool_names(&seen[1]),
        vec!["read_file", "stock_quote", TOOL_SEARCH_NAME]
    );
    assert!(seen[3..].iter().all(|tools| tools == &seen[3]));
    assert_eq!(tool_names(&seen[3]), vec!["read_file", TOOL_SEARCH_NAME]);
}

#[tokio::test]
async fn a_session_override_beats_the_harness_wide_policy_in_both_directions() {
    // Harness-wide: promotion off. The session opts in.
    let model = RecordingModel::new(search_then_call_script(json!({"symbol": "A"})));
    let (harness, _, _) = discovery_harness(
        &model,
        ToolDiscoveryPolicy::default().with_promote_found_tools(false),
    );
    let on = RunConfig::new("on").with_discovery_overrides(DiscoveryOverrides {
        promote_found_tools: Some(true),
        ..DiscoveryOverrides::default()
    });
    harness
        .invoke(&(), (), on, vec![Message::user("a")])
        .await
        .expect("run succeeds");
    assert_eq!(
        tool_names(&model.tools_seen()[1]),
        vec!["read_file", "stock_quote", TOOL_SEARCH_NAME]
    );
}

#[tokio::test]
async fn compact_search_results_are_a_per_session_option() {
    let mut rendered = Vec::new();
    for compact in [false, true] {
        let model = RecordingModel::new(search_then_call_script(json!({"symbol": "A"})));
        let (harness, _, _) = discovery_harness(&model, ToolDiscoveryPolicy::default());
        let config = RunConfig::new("r").with_discovery_overrides(DiscoveryOverrides {
            compact_search_results: Some(compact),
            ..DiscoveryOverrides::default()
        });
        let run = harness
            .invoke(&(), (), config, vec![Message::user("a")])
            .await
            .expect("run succeeds");
        rendered.push(tool_messages(&run.messages)[0].clone());
    }
    assert!(rendered[0].contains("\n  "), "default stays pretty-printed");
    assert_eq!(
        rendered[1].lines().count(),
        2,
        "compact is one payload line"
    );
}

/// A harness whose three `stock_*` deferred tools all match a ticker search,
/// next to one direct tool and one deferred tool the search does not match.
fn collapse_harness(
    model: &Arc<RecordingModel>,
    discovery: ToolDiscoveryPolicy,
) -> (AgentHarness<()>, Arc<ExposedTool>) {
    let quote = ExposedTool::new(
        "stock_quote",
        "Fetch the latest price for a ticker symbol.",
        ToolExposure::Deferred,
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("mock", model.clone()),
        ))
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(quote.clone())
        .register_tool(ExposedTool::new(
            "stock_history",
            "Fetch the price history for a ticker symbol.",
            ToolExposure::Deferred,
        ))
        .register_tool(ExposedTool::new(
            "weather_forecast",
            "Report the weather for a city.",
            ToolExposure::Deferred,
        ))
        .register_tool(ExposedTool::new(
            "stock_news",
            "Fetch headlines and the price move for a ticker symbol.",
            ToolExposure::Deferred,
        ))
        .with_policy(RunPolicy {
            discovery,
            ..RunPolicy::default()
        });
    (harness, quote)
}

fn collapsing(on: bool) -> RunConfig {
    RunConfig::new("session").with_discovery_overrides(DiscoveryOverrides {
        collapse_search_results: Some(on),
        ..DiscoveryOverrides::default()
    })
}

fn ticker_search() -> ModelResponse {
    tool_call(
        "c1",
        TOOL_SEARCH_NAME,
        json!({"query": "price of a ticker", "limit": 3}),
    )
}

fn hit_names(result: &str) -> Vec<String> {
    let payload = result.split_once('\n').expect("header then payload").1;
    serde_json::from_str::<Vec<Value>>(payload)
        .expect("payload is a JSON array")
        .iter()
        .map(|hit| hit["name"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn collapse_keeps_the_winning_schema_once_the_run_ends() {
    let model = RecordingModel::new(vec![
        ticker_search(),
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
    ]);
    let (harness, quote) = collapse_harness(&model, ToolDiscoveryPolicy::default());

    let run = harness
        .invoke(&(), (), collapsing(true), vec![Message::user("price?")])
        .await
        .expect("run succeeds");

    // Stored: the query line and only the used tool's schema.
    let results = tool_messages(&run.messages);
    assert!(
        results[0].starts_with("tool_search \"price of a ticker\" \u{2192} used stock_quote\n")
    );
    assert_eq!(hit_names(&results[0]), vec!["stock_quote"]);
    assert!(!results[0].contains("stock_news"));
    // The tool itself ran as normal.
    assert_eq!(quote.calls.lock().unwrap().len(), 1);
    // While the run was going the model still saw every matched schema.
    let seen = model.messages_seen();
    let during = seen[1]
        .iter()
        .find(|m| matches!(m, Message::Tool(_)))
        .unwrap();
    assert_eq!(hit_names(&during.text()).len(), 3);
}

#[tokio::test]
async fn collapse_reduces_an_unused_search_to_one_line_of_names() {
    let model = RecordingModel::new(vec![ticker_search(), text("no luck")]);
    let (harness, _) = collapse_harness(&model, ToolDiscoveryPolicy::default());

    let run = harness
        .invoke(&(), (), collapsing(true), vec![Message::user("price?")])
        .await
        .expect("run succeeds");

    let result = &tool_messages(&run.messages)[0];
    assert_eq!(result.lines().count(), 1);
    assert!(result.starts_with("tool_search \"price of a ticker\" \u{2192} none used; matched "));
    for name in ["stock_quote", "stock_history", "stock_news"] {
        assert!(result.contains(name), "{name} is listed");
    }
    assert!(!result.contains("parameters"));
}

#[tokio::test]
async fn collapse_is_off_by_default_and_a_session_can_switch_it_on_or_off() {
    let script = || vec![ticker_search(), text("done")];
    let run_with = |policy: ToolDiscoveryPolicy, config: RunConfig| async move {
        let model = RecordingModel::new(script());
        let (harness, _) = collapse_harness(&model, policy);
        let run = harness
            .invoke(&(), (), config, vec![Message::user("price?")])
            .await
            .expect("run succeeds");
        tool_messages(&run.messages)[0].clone()
    };

    let default = run_with(ToolDiscoveryPolicy::default(), RunConfig::new("s")).await;
    assert_eq!(hit_names(&default).len(), 3, "today's behaviour: unchanged");

    let on_by_policy = run_with(
        ToolDiscoveryPolicy::default().with_collapse_search_results(true),
        RunConfig::new("s"),
    )
    .await;
    assert!(on_by_policy.contains("none used"));

    let off_by_session = run_with(
        ToolDiscoveryPolicy::default().with_collapse_search_results(true),
        collapsing(false),
    )
    .await;
    assert_eq!(hit_names(&off_by_session).len(), 3);
}

#[tokio::test]
async fn a_resumed_run_sees_the_collapsed_result_and_can_call_the_kept_tool_directly() {
    let model = RecordingModel::new(vec![
        ticker_search(),
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
        // The second run calls the kept tool again with no new search, one
        // call valid and one invalid against the tool's schema.
        tool_call("c3", "stock_quote", json!({"symbol": "XYZ"})),
        tool_call("c4", "stock_quote", json!({"symbol": 7})),
        text("done again"),
    ]);
    let (harness, quote) = collapse_harness(&model, ToolDiscoveryPolicy::default());
    let first = harness
        .invoke(&(), (), collapsing(true), vec![Message::user("price?")])
        .await
        .expect("first run");

    let mut history = first.messages.clone();
    history.push(Message::user("and XYZ?"));
    let second = harness
        .invoke(&(), (), collapsing(true), history)
        .await
        .expect("resumed run");

    // The resumed run's first request carried the collapsed result.
    let seen = model.messages_seen();
    let resumed_request = &seen[3];
    let carried = resumed_request
        .iter()
        .find(|m| matches!(m, Message::Tool(t) if t.tool_call_id == "c1"))
        .expect("search result is in the resumed history");
    assert_eq!(hit_names(&carried.text()), vec!["stock_quote"]);
    assert!(!carried.text().contains("stock_news"));
    // The kept tool is callable directly and still validated against its schema.
    let results = tool_messages(&second.messages);
    assert!(
        results
            .iter()
            .any(|r| r.contains("stock_quote \u{2192} XYZ"))
    );
    assert!(
        results
            .iter()
            .any(|r| r.contains("invalid arguments for tool `stock_quote`"))
    );
    assert_eq!(
        quote.calls.lock().unwrap().len(),
        2,
        "ACME then XYZ; the invalid call never ran"
    );
    // The collapsed result was not touched again.
    assert_eq!(
        tool_messages(&second.messages)[0],
        tool_messages(&first.messages)[0]
    );
}

#[tokio::test]
async fn turning_collapse_on_leaves_results_already_in_the_thread_untouched() {
    let model = RecordingModel::new(vec![
        ticker_search(),
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "price of a ticker", "limit": 3}),
        ),
        tool_call("c2", "stock_news", json!({"symbol": "XYZ"})),
        text("done again"),
    ]);
    let (harness, _) = collapse_harness(&model, ToolDiscoveryPolicy::default());

    // Before the switch: today's behaviour, the full result is stored.
    let before = harness
        .invoke(&(), (), collapsing(false), vec![Message::user("price?")])
        .await
        .expect("run before the switch");
    let old_result = tool_messages(&before.messages)[0].clone();
    assert_eq!(hit_names(&old_result).len(), 3);

    // After the switch, the same thread continues. Its new search reuses the
    // call ids of the old one, as a provider that numbers calls would.
    let mut history = before.messages.clone();
    history.push(Message::user("news too?"));
    let old_len = history.len();
    let after = harness
        .invoke(&(), (), collapsing(true), history)
        .await
        .expect("run after the switch");

    let mut expected = before.messages.clone();
    expected.push(Message::user("news too?"));
    assert_eq!(
        after.messages[..old_len],
        expected[..],
        "old history is byte-identical"
    );
    let results = tool_messages(&after.messages);
    assert_eq!(results[0], old_result);
    assert!(results[2].starts_with("tool_search \"price of a ticker\" \u{2192} used stock_news\n"));
    assert_eq!(hit_names(&results[2]), vec!["stock_news"]);
}
