//! The per-session `repair_tool_call_wrapper` option: whether a call a model
//! buried inside a pseudo `tool_call` tool is retargeted at the real tool.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::runtime::AgentHarness;
use crate::testkit::EventRecorder;
use crate::tool::discover::DiscoveryOverrides;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolPolicy, ToolResult};

/// A tool that records the arguments it ran with.
struct Recorder {
    name: &'static str,
    seen: Mutex<Vec<Value>>,
}

impl Recorder {
    fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            seen: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Tool for Recorder {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "recording tool"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }
    async fn execute(&self, arguments: Value) -> anyhow::Result<ToolResult> {
        self.seen.lock().unwrap().push(arguments);
        Ok(ToolResult::success("ran"))
    }
}

fn response(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(text.to_string())]
    };
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
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

fn harness(model_calls: Vec<ToolCall>, tools: &[Arc<Recorder>]) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            response(model_calls, ""),
            response(Vec::new(), "done"),
        ])),
    )));
    for tool in tools {
        harness.register_tool(tool.clone());
    }
    harness
}

fn wrapped(wrapper: &str, target: &str) -> ToolCall {
    ToolCall::new(
        "call-1",
        wrapper,
        json!({"name": target, "arguments": {"q": "x"}}),
    )
}

fn config(repair: Option<bool>) -> RunConfig {
    RunConfig::new("wrapper").with_discovery_overrides(DiscoveryOverrides {
        repair_tool_call_wrapper: repair,
        ..DiscoveryOverrides::default()
    })
}

/// Runs one session and returns the text of the tool result for `call-1` plus
/// the recorded events.
async fn run_session(harness: &AgentHarness<()>, config: RunConfig) -> (String, Vec<AgentEvent>) {
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(config, ()).with_events(recorder.sink());
    let result = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("an unknown tool is recoverable");
    let text = result
        .run
        .messages
        .iter()
        .find_map(|message| match message {
            Message::Tool(tool) if tool.tool_call_id == "call-1" => Some(message.text()),
            _ => None,
        })
        .expect("the call was answered");
    (text, recorder.events())
}

fn was_repaired(events: &[AgentEvent]) -> bool {
    events.iter().any(|event| {
        matches!(event, AgentEvent::InvalidToolArgs { recovery, .. } if recovery.starts_with("unwrapped:"))
    })
}

#[tokio::test]
async fn the_repair_is_on_by_default_and_runs_the_wrapped_tool() {
    let lookup = Recorder::new("lookup");
    let harness = harness(
        vec![wrapped("tool_call", "lookup")],
        std::slice::from_ref(&lookup),
    );
    let (text, events) = run_session(&harness, config(None)).await;
    assert_eq!(lookup.calls(), vec![json!({"q": "x"})]);
    assert_eq!(text, "ran");
    assert!(was_repaired(&events));
}

#[tokio::test]
async fn with_the_repair_off_a_wrapped_call_gets_the_unknown_tool_corrective() {
    let lookup = Recorder::new("lookup");
    let harness = harness(
        vec![wrapped("tool_call", "lookup")],
        std::slice::from_ref(&lookup),
    );
    let (text, events) = run_session(&harness, config(Some(false))).await;
    assert!(lookup.calls().is_empty(), "the wrapped tool must not run");
    assert!(text.starts_with("unknown tool `tool_call`"), "{text}");
    assert!(!was_repaired(&events));
}

#[tokio::test]
async fn an_explicit_on_matches_the_default() {
    let lookup = Recorder::new("lookup");
    let harness = harness(
        vec![wrapped("tool_call", "lookup")],
        std::slice::from_ref(&lookup),
    );
    let (_, events) = run_session(&harness, config(Some(true))).await;
    assert_eq!(lookup.calls().len(), 1);
    assert!(was_repaired(&events));
}

#[tokio::test]
async fn two_sessions_with_different_settings_share_one_harness() {
    let lookup = Recorder::new("lookup");
    let call = wrapped("tool_call", "lookup");
    let harness = {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        // One response script per session: the mock replays them in order.
        harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                response(vec![call.clone()], ""),
                response(Vec::new(), "done"),
                response(vec![call], ""),
                response(Vec::new(), "done"),
            ])),
        )));
        harness.register_tool(lookup.clone());
        harness
    };

    let (off_text, _) = run_session(&harness, config(Some(false))).await;
    assert!(
        off_text.starts_with("unknown tool `tool_call`"),
        "{off_text}"
    );
    assert!(lookup.calls().is_empty());

    let (on_text, _) = run_session(&harness, config(None)).await;
    assert_eq!(on_text, "ran");
    assert_eq!(lookup.calls().len(), 1);
}

#[tokio::test]
async fn a_host_tool_named_tool_call_runs_whatever_the_setting() {
    for repair in [None, Some(true), Some(false)] {
        let host_tool = Recorder::new("tool_call");
        let lookup = Recorder::new("lookup");
        let harness = harness(
            vec![wrapped("tool_call", "lookup")],
            &[host_tool.clone(), lookup.clone()],
        );
        let (text, events) = run_session(&harness, config(repair)).await;
        assert_eq!(host_tool.calls().len(), 1, "{repair:?}");
        assert!(lookup.calls().is_empty(), "{repair:?}");
        assert_eq!(text, "ran");
        assert!(!was_repaired(&events));
    }
}
