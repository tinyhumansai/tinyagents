//! TRUE end-to-end: agent-calling-agent composition (sub-agents).
//!
//! A parent [`AgentHarness`] whose scripted [`MockModel`] calls a
//! [`SubAgentTool`] drives a child [`AgentHarness`] (also a `MockModel`) and
//! composes the child's answer into a final assistant reply. The parent run's
//! [`EventSink`] is wired to a testkit [`EventRecorder`] so we can reconstruct
//! a [`Trajectory`] and assert *structurally* that the sub-agent really ran —
//! never on model prose.
//!
//! A second test exercises the deterministic recursion-depth guard: nesting a
//! sub-agent past the harness's `max_depth` fails fast with
//! [`TinyAgentsError::SubAgentDepth`] *before* any model call, both through the
//! direct invoke path and through the tool path.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents::error::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::limits::RunLimits;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::{Middleware, PromptCacheGuardMiddleware};
use tinyagents::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::testkit::{EventRecorder, Trajectory};
use tinyagents::harness::tool::{Tool, ToolCall};
use tinyagents::harness::usage::Usage;
use tinyagents::{SubAgent, SubAgentTool};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// A tool-call assistant turn: no text, a single tool call.
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(9, 4)),
        },
        usage: Some(Usage::new(9, 4)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

/// A plain-text assistant turn.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(5, 3)),
        },
        usage: Some(Usage::new(5, 3)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

/// A child harness whose model always answers with `answer`.
fn child_harness(answer: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child-model", Arc::new(MockModel::constant(answer)));
    harness
}

/// A child harness capped at `max_depth`.
fn child_harness_with_max_depth(answer: &str, max_depth: usize) -> AgentHarness<()> {
    let mut harness = child_harness(answer);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(max_depth),
        ..RunPolicy::default()
    });
    harness
}

#[derive(Default)]
struct StablePromptSegments;

#[async_trait]
impl Middleware<(), ()> for StablePromptSegments {
    fn name(&self) -> &str {
        "stable_prompt_segments"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents::Result<()> {
        let mut segments = vec![PromptSegment {
            id: "system".into(),
            role: SegmentRole::System,
            cacheable: true,
        }];
        if !request.tools.is_empty() {
            segments.push(PromptSegment {
                id: "tools".into(),
                role: SegmentRole::Tools,
                cacheable: true,
            });
        }
        segments.push(PromptSegment {
            id: "turn".into(),
            role: SegmentRole::Volatile,
            cacheable: false,
        });
        request.cache_segments = segments;
        Ok(())
    }
}

#[derive(Default)]
struct RequestProbe {
    max_tokens: Mutex<Vec<Option<u32>>>,
    stable_prefixes: Mutex<Vec<Vec<String>>>,
}

impl RequestProbe {
    fn max_tokens(&self) -> Vec<Option<u32>> {
        self.max_tokens.lock().expect("max_tokens mutex").clone()
    }

    fn stable_prefixes(&self) -> Vec<Vec<String>> {
        self.stable_prefixes
            .lock()
            .expect("stable_prefixes mutex")
            .clone()
    }
}

#[async_trait]
impl Middleware<(), ()> for RequestProbe {
    fn name(&self) -> &str {
        "request_probe"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents::Result<()> {
        self.max_tokens
            .lock()
            .expect("max_tokens mutex")
            .push(request.max_tokens);
        self.stable_prefixes
            .lock()
            .expect("stable_prefixes mutex")
            .push(request.cacheable_prefix_ids());
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn parent_drives_subagent_and_composes_answer() {
    // Child agent that "researches" and returns a fixed answer.
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("RUST_IS_A_SYSTEMS_LANGUAGE")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    // Parent: first turn delegates to the sub-agent tool, second turn composes
    // the final answer.
    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("Based on the researcher: a systems language."),
        ])),
    );

    // Wire the parent run's events into a shared recorder so we can assert on
    // the trajectory (the sub-agent tool really ran) afterward.
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("parent-run"), ()).with_events(recorder.sink());

    let run = parent
        .invoke_in_context(&(), ctx, vec![Message::user("delegate this")])
        .await
        .expect("parent run succeeds");

    // Behavior / structure assertions — never on the exact prose.
    assert_eq!(run.tool_calls, 1, "parent invoked the sub-agent tool once");
    assert_eq!(run.model_calls, 2, "parent made two model calls");
    assert_eq!(
        run.text(),
        Some("Based on the researcher: a systems language.".to_string())
    );

    // The child's answer was woven into the parent transcript as a tool message.
    let child_answer_present = run
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool(_)) && m.text() == "RUST_IS_A_SYSTEMS_LANGUAGE");
    assert!(
        child_answer_present,
        "the sub-agent's answer should appear as a tool result in the parent transcript"
    );

    // Trajectory assertion: the sub-agent (exposed as the `researcher` tool)
    // really ran, and the run completed cleanly.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_tool_called("researcher");
    assert_eq!(traj.tool_call_count("researcher"), 1);
    traj.assert_model_called_times(3);
    traj.assert_completed();
    traj.assert_order(&["run.started", "researcher", "run.completed"])
        .expect("sub-agent tool runs between run start and completion");
}

#[tokio::test]
async fn child_subagents_derive_unique_thread_ids_from_parent_thread() {
    let child = SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("RUST_IS_A_SYSTEMS_LANGUAGE")),
    );
    let recorder = EventRecorder::new();
    let parent = RunContext::new(
        RunConfig::new("parent-run").with_thread("parent-thread"),
        (),
    )
    .with_events(recorder.sink());

    child
        .invoke_in_parent(&(), (), &parent, "first question")
        .await
        .expect("first child run succeeds");
    child
        .invoke_in_parent(&(), (), &parent, "second question")
        .await
        .expect("second child run succeeds");

    let child_threads: Vec<String> = recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::RunStarted {
                thread_id: Some(thread_id),
                ..
            } => Some(thread_id.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(child_threads.len(), 2);
    assert_ne!(
        child_threads[0], child_threads[1],
        "each child run should get an isolated thread"
    );
    for thread in child_threads {
        assert!(
            thread.starts_with("parent-thread-subagent-researcher-d1-"),
            "child thread should inherit the parent thread as a hyphenated prefix: {thread}"
        );
        assert!(
            !thread.contains('/'),
            "child thread id should not use slash separators: {thread}"
        );
    }
}

#[tokio::test]
async fn nested_subagent_turns_preserve_kv_layout_thread_lineage_and_output_cap() {
    let researcher_guard = Arc::new(PromptCacheGuardMiddleware::new());
    let researcher_probe = Arc::new(RequestProbe::default());
    let mut researcher_harness: AgentHarness<()> = AgentHarness::new();
    researcher_harness
        .register_model(
            "researcher-model",
            Arc::new(MockModel::constant("research-result")),
        )
        .push_middleware(Arc::new(StablePromptSegments))
        .push_middleware(researcher_guard.clone())
        .push_middleware(researcher_probe.clone());
    let researcher = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(researcher_harness),
    ));

    let worker_probe = Arc::new(RequestProbe::default());
    let mut worker_harness: AgentHarness<()> = AgentHarness::new();
    worker_harness
        .register_tool(Arc::new(SubAgentTool::new(researcher)))
        .register_model(
            "worker-model",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("r1", "researcher", json!({ "input": "first fact" })),
                text_response("worker returned first fact"),
                tool_call_response("r2", "researcher", json!({ "input": "more data" })),
                text_response("worker returned more data"),
            ])),
        )
        .push_middleware(Arc::new(StablePromptSegments))
        .push_middleware(worker_probe.clone());
    let worker = Arc::new(SubAgent::new(
        "worker",
        "delegates research",
        Arc::new(worker_harness),
    ));

    let orchestrator_probe = Arc::new(RequestProbe::default());
    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator
        .register_tool(Arc::new(SubAgentTool::new(worker)))
        .register_model(
            "orchestrator-model",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("w1", "worker", json!({ "input": "ask researcher" })),
                tool_call_response("w2", "worker", json!({ "input": "ask for more" })),
                text_response("orchestrator composed both research turns"),
            ])),
        )
        .push_middleware(Arc::new(StablePromptSegments))
        .push_middleware(orchestrator_probe.clone());

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(
        RunConfig::new("orchestrator-run")
            .with_thread("root-thread")
            .with_max_model_calls(4)
            .with_max_tool_calls(4)
            .with_max_turn_output_tokens(16),
        (),
    )
    .with_events(recorder.sink());

    let run = orchestrator
        .invoke_in_context(&(), ctx, vec![Message::user("coordinate research")])
        .await
        .expect("nested orchestrator run succeeds");

    assert_eq!(
        run.text(),
        Some("orchestrator composed both research turns".to_string())
    );
    assert_eq!(run.tool_calls, 2);

    let researcher_prefixes = researcher_probe.stable_prefixes();
    assert_eq!(researcher_prefixes, vec![vec!["system"], vec!["system"]]);
    assert!(
        researcher_guard.layout_events().is_empty(),
        "researcher KV-cache stable prefix should stay aligned across turns"
    );
    assert_eq!(researcher_probe.max_tokens(), vec![Some(16), Some(16)]);
    assert_eq!(
        worker_probe.max_tokens(),
        vec![Some(16), Some(16), Some(16), Some(16)]
    );
    assert_eq!(
        orchestrator_probe.max_tokens(),
        vec![Some(16), Some(16), Some(16)]
    );

    let child_threads: Vec<String> = recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::RunStarted {
                thread_id: Some(thread_id),
                ..
            } if thread_id.as_str() != "root-thread" => Some(thread_id.to_string()),
            _ => None,
        })
        .collect();
    let worker_threads: Vec<&String> = child_threads
        .iter()
        .filter(|thread| {
            thread.starts_with("root-thread-subagent-worker-d1-")
                && !thread.contains("-subagent-researcher-d2-")
        })
        .collect();
    let researcher_threads: Vec<&String> = child_threads
        .iter()
        .filter(|thread| thread.contains("-subagent-researcher-d2-"))
        .collect();

    assert_eq!(worker_threads.len(), 2);
    assert_ne!(worker_threads[0], worker_threads[1]);
    assert_eq!(researcher_threads.len(), 2);
    assert_ne!(researcher_threads[0], researcher_threads[1]);
    for thread in researcher_threads {
        assert!(thread.starts_with("root-thread-subagent-worker-d1-"));
        assert!(thread.contains("-subagent-researcher-d2-"));
        assert!(!thread.contains('/'));
    }
}

#[tokio::test]
async fn nesting_past_max_depth_is_a_deterministic_error() {
    // Cap the child harness at depth 1: a child run is allowed at depth 1
    // (parent_depth 0) but not at depth 2 (parent_depth 1).
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));

    // Within the cap: parent_depth 0 -> child depth 1.
    let ok_run = subagent
        .invoke(&(), (), 0, "ok")
        .await
        .expect("child run at depth 1 is within the cap");
    assert_eq!(ok_run.text(), Some("ok".to_string()));

    // Direct invoke past the cap: parent_depth 1 -> child depth 2 > cap of 1.
    let err = subagent
        .invoke(&(), (), 1, "too deep")
        .await
        .expect_err("child depth 2 exceeds the cap");
    assert!(
        matches!(err, TinyAgentsError::SubAgentDepth(1)),
        "expected SubAgentDepth(1), got {err:?}"
    );

    // Tool path past the cap: constructing the tool at parent_depth 1 makes the
    // child run at depth 2, which also exceeds the cap deterministically.
    let tool = SubAgentTool::new(subagent).with_parent_depth(1);
    let tool_result = tool
        .call(&(), ToolCall::new("c1", "deep", json!({ "input": "x" })))
        .await
        .expect("the tool returns a failed tool result");
    let tool_error = tool_result.error.expect("tool result carries an error");
    assert!(
        tool_error.contains("recursion depth limit") && tool_error.contains("maximum depth of 1"),
        "expected SubAgentDepth(1) from the tool path, got {tool_error:?}"
    );
}
