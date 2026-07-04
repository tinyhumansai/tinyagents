//! Tests for first-class sub-agents and recursion-depth tracking.
//!
//! These exercise:
//! - a parent harness whose scripted [`MockModel`] calls a [`SubAgentTool`],
//!   driving a child harness and composing the child's answer,
//! - direct [`SubAgent::invoke`] returning the child [`AgentRun`] at depth 1,
//! - the depth guard producing [`TinyAgentsError::SubAgentDepth`] when nested
//!   too deep (both via direct invoke and via the tool path),
//! - sub-agent lifecycle events emitted onto a shared sink.

use std::sync::Arc;

use serde_json::json;

use crate::error::TinyAgentsError;
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, EventSink, RecordingListener};
use crate::harness::limits::RunLimits;
use crate::harness::message::{AssistantMessage, ContentBlock, Message};
use crate::harness::model::ModelResponse;
use crate::harness::providers::MockModel;
use crate::harness::runtime::{AgentHarness, RunPolicy};
use crate::harness::subagent::{SubAgent, SubAgentSession, SubAgentTool};
use crate::harness::testkit::ScriptedModel;
use crate::harness::tool::{Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSchema};
use crate::harness::usage::Usage;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Builds a tool-call assistant response (no text, one tool call).
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds a plain-text assistant response.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds a child harness whose model always answers with `answer`.
fn child_harness(answer: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child-model", Arc::new(MockModel::constant(answer)));
    harness
}

/// Builds a child harness with a custom `max_depth` policy.
fn child_harness_with_max_depth(answer: &str, max_depth: usize) -> AgentHarness<()> {
    let mut harness = child_harness(answer);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(max_depth),
        ..RunPolicy::default()
    });
    harness
}

struct SpinTool;

#[async_trait::async_trait]
impl Tool<()> for SpinTool {
    fn name(&self) -> &str {
        "spin"
    }

    fn description(&self) -> &str {
        "keeps the child loop running"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new("spin", "keeps the child loop running", json!({}))
    }

    async fn call(&self, _state: &(), call: ToolCall) -> crate::Result<ToolResult> {
        Ok(ToolResult::text(call.id, "spin", "again"))
    }
}

fn looping_child_harness_with_max_model_calls(max_model_calls: usize) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "child-model",
            Arc::new(MockModel::with_tool_call("spin", json!({}))),
        )
        .register_tool(Arc::new(SpinTool))
        .with_policy(RunPolicy {
            limits: RunLimits::default().with_max_model_calls(max_model_calls),
            ..RunPolicy::default()
        });
    harness
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn subagent_tool_drives_child_and_composes_answer() {
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("the child answer")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("composed final answer"),
        ])),
    );

    let run = parent
        .invoke_default(&(), vec![Message::user("delegate this")])
        .await
        .expect("parent run succeeds");

    assert_eq!(run.tool_calls, 1);
    assert_eq!(run.model_calls, 2);
    assert_eq!(run.text(), Some("composed final answer".to_string()));

    // The child's answer is woven into the parent transcript as a tool message.
    let has_child_answer = run
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool(_)) && m.text() == "the child answer");
    assert!(
        has_child_answer,
        "child answer should appear as a tool result"
    );
}

#[tokio::test]
async fn direct_invoke_returns_child_run_at_depth_one() {
    let subagent = SubAgent::new(
        "helper",
        "a helper agent",
        Arc::new(child_harness("hi from child")),
    )
    .with_system_prompt("You are a helper.");

    let run = subagent
        .invoke(&(), (), 0, "hello")
        .await
        .expect("child run succeeds");

    assert_eq!(run.text(), Some("hi from child".to_string()));
    // system prompt + user input + assistant reply.
    assert_eq!(run.messages.len(), 3);
    assert!(matches!(run.messages[0], Message::System(_)));
}

#[tokio::test]
async fn invoke_at_max_depth_errors() {
    // Child harness caps depth at 1: a child run is allowed at depth 1
    // (parent_depth 0) but not at depth 2 (parent_depth 1).
    let subagent = SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    );

    // parent_depth 0 -> child depth 1 -> within the cap.
    subagent
        .invoke(&(), (), 0, "ok")
        .await
        .expect("depth 1 within cap");

    // parent_depth 1 -> child depth 2 -> exceeds the cap of 1.
    let err = subagent
        .invoke(&(), (), 1, "too deep")
        .await
        .expect_err("depth 2 exceeds cap");
    assert!(matches!(err, TinyAgentsError::SubAgentDepth(1)));
}

#[tokio::test]
async fn tool_path_enforces_depth_limit() {
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));
    // Invoke the child at parent_depth 1 -> child depth 2 -> exceeds cap of 1.
    let tool = SubAgentTool::new(subagent).with_parent_depth(1);

    let result = tool
        .call(&(), ToolCall::new("c1", "deep", json!({ "input": "x" })))
        .await
        .expect("tool reports depth limit as a parent-visible tool result");
    assert!(result.is_error());
    assert!(
        result.content.contains("recursion depth limit")
            && result.content.contains("delegated-agent limit signal"),
        "unexpected depth limit message: {}",
        result.content
    );
}

#[tokio::test]
async fn call_with_context_enforces_depth_limit_as_a_tool_result() {
    // Regression test: `SubAgentTool::call_with_context` used to propagate a
    // `child_config` depth-limit rejection raw via `?`, bypassing the
    // limit-to-tool-error conversion `call` already applies and aborting the
    // whole parent run instead of surfacing a tool-error the model can react
    // to. This exercises the `call_with_context` path directly (the one the
    // agent loop actually uses), not `call`.
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));
    let tool = SubAgentTool::new(subagent);

    // Caller (parent) depth 1 -> child depth 2 -> exceeds the cap of 1.
    let parent_ctx: RunContext<()> = RunContext::new(RunConfig::new("parent").with_depth(1), ());
    let context = ToolExecutionContext::from_run_context(&parent_ctx);

    let result = tool
        .call_with_context(
            &(),
            ToolCall::new("c1", "deep", json!({ "input": "too deep" })),
            context,
        )
        .await
        .expect("depth limit is converted into a parent-visible tool result, not a fatal error");

    assert!(result.is_error());
    assert_eq!(result.call_id, "c1");
    assert!(
        result.content.contains("recursion depth limit")
            && result.content.contains("delegated-agent limit signal"),
        "unexpected depth limit message: {}",
        result.content
    );
}

#[tokio::test]
async fn subagent_tool_reports_child_limit_to_parent_as_tool_error() {
    let subagent = Arc::new(SubAgent::new(
        "worker",
        "does bounded work",
        Arc::new(looping_child_harness_with_max_model_calls(1)),
    ));
    let tool = SubAgentTool::new(subagent);

    let result = tool
        .call(
            &(),
            ToolCall::new("c1", "worker", json!({ "input": "loop" })),
        )
        .await
        .expect("child limit is converted into a parent-visible tool result");

    assert!(result.is_error());
    assert_eq!(result.call_id, "c1");
    assert_eq!(result.name, "worker");
    assert!(
        result.content.contains("Sub-agent `worker` stopped")
            && result.content.contains("configured run limit")
            && result.content.contains("delegated-agent limit signal"),
        "unexpected limit message: {}",
        result.content
    );
}

#[tokio::test]
async fn parent_can_continue_after_subagent_tool_hits_child_limit() {
    let subagent = Arc::new(SubAgent::new(
        "worker",
        "does bounded work",
        Arc::new(looping_child_harness_with_max_model_calls(1)),
    ));

    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent
        .register_tool(Arc::new(SubAgentTool::new(subagent)))
        .register_model(
            "parent-model",
            Arc::new(MockModel::with_responses(vec![
                tool_call_response("c1", "worker", json!({ "input": "loop" })),
                text_response("worker hit its limit; I will narrow the task"),
            ])),
        );

    let run = parent
        .invoke_default(&(), vec![Message::user("delegate bounded work")])
        .await
        .expect("parent receives the sub-agent limit as a tool result and continues");

    assert_eq!(
        run.text(),
        Some("worker hit its limit; I will narrow the task".to_string())
    );
    assert_eq!(run.model_calls, 2);
    assert_eq!(run.tool_calls, 1);
    assert!(
        run.messages
            .iter()
            .any(|message| matches!(message, Message::Tool(_))
                && message.text().contains("delegated-agent limit signal")),
        "parent transcript should include the child limit tool result"
    );
}

#[tokio::test]
async fn invoke_with_events_emits_lifecycle_on_shared_sink() {
    let subagent = SubAgent::new(
        "observed",
        "an observed agent",
        Arc::new(child_harness("done")),
    );

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    subagent
        .invoke_with_events(&(), (), 0, "go", &sink)
        .await
        .expect("child run succeeds");

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();

    // Sub-agent lifecycle brackets the child run, and the child's own RunStarted
    // also lands on the shared sink.
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::SubAgentStarted { name, depth } if name == "observed" && *depth == 1
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::SubAgentCompleted { name, depth } if name == "observed" && *depth == 1
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunStarted { .. }))
    );
}

#[tokio::test]
async fn invoke_in_parent_derives_unique_child_thread_ids() {
    let subagent = SubAgent::new(
        "observed",
        "an observed agent",
        Arc::new(child_harness("done")),
    );

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());
    let parent = RunContext::new(
        RunConfig::new("parent-run").with_thread("parent-thread"),
        (),
    )
    .with_events(sink);

    subagent
        .invoke_in_parent(&(), (), &parent, "first")
        .await
        .expect("first child run succeeds");
    subagent
        .invoke_in_parent(&(), (), &parent, "second")
        .await
        .expect("second child run succeeds");

    let child_threads: Vec<String> = recorder
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::RunStarted {
                thread_id: Some(thread_id),
                ..
            } => Some(thread_id.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(child_threads.len(), 2);
    assert_ne!(child_threads[0], child_threads[1]);
    for thread in child_threads {
        assert!(thread.starts_with("parent-thread-subagent-observed-d1-"));
        assert!(!thread.contains('/'));
    }
}

/// Each invocation must mint a unique child run id (readable
/// `{name}-d{depth}-{seq}`): a reused `{name}-d{depth}` interleaved journals
/// and status stores keyed by run id across invocations.
#[tokio::test]
async fn repeated_invocations_mint_unique_child_run_ids() {
    let subagent = SubAgent::new(
        "observed",
        "an observed agent",
        Arc::new(child_harness("done")),
    );

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    subagent
        .invoke_with_events(&(), (), 0, "first", &sink)
        .await
        .expect("first child run succeeds");
    subagent
        .invoke_with_events(&(), (), 0, "second", &sink)
        .await
        .expect("second child run succeeds");

    let run_ids: Vec<String> = recorder
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::RunStarted { run_id, .. } => Some(run_id.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(run_ids.len(), 2);
    assert_ne!(
        run_ids[0], run_ids[1],
        "each invocation must get its own run id"
    );
    for run_id in run_ids {
        assert!(
            run_id.starts_with("observed-d1-"),
            "run id keeps the readable name/depth prefix: {run_id}"
        );
    }
}

/// Two sessions reusing the same sub-agent must not share run ids for the same
/// turn number.
#[tokio::test]
async fn parallel_sessions_mint_unique_run_ids_per_turn() {
    let subagent = Arc::new(SubAgent::new(
        "sess",
        "a session agent",
        Arc::new(child_harness("done")),
    ));

    let mut run_ids = Vec::new();
    for _ in 0..2 {
        let sink = EventSink::new();
        let recorder = Arc::new(RecordingListener::new());
        sink.subscribe(recorder.clone());
        let mut session = SubAgentSession::new(subagent.clone()).with_events(sink);
        session
            .send(&(), (), vec![Message::user("hello")])
            .await
            .expect("send succeeds");
        run_ids.extend(
            recorder
                .events()
                .into_iter()
                .filter_map(|record| match record.event {
                    AgentEvent::RunStarted { run_id, .. } => Some(run_id.to_string()),
                    _ => None,
                }),
        );
    }

    assert_eq!(run_ids.len(), 2);
    assert_ne!(
        run_ids[0], run_ids[1],
        "turn 0 of two sessions must not share a run id"
    );
    for run_id in run_ids {
        assert!(run_id.starts_with("sess-t0-d1-"), "run id: {run_id}");
    }
}

#[tokio::test]
async fn session_reuses_subagent_and_carries_context_across_sends() {
    // A scripted child model that records every request it receives, so we can
    // prove the SECOND send carried the FIRST turn's messages.
    let model = Arc::new(ScriptedModel::replies(vec!["Paris", "French"]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("scripted", model.clone());

    let subagent = Arc::new(
        SubAgent::new(
            "geographer",
            "answers geography questions",
            Arc::new(harness),
        )
        .with_system_prompt("You are a concise geography expert."),
    );
    // Keep a separate handle to confirm the SAME Arc is reused after sends.
    let subagent_handle = Arc::clone(&subagent);

    let mut session = SubAgentSession::new(subagent);

    // First send: the user's question.
    let first = session
        .send(
            &(),
            (),
            vec![Message::user("What is the capital of France?")],
        )
        .await
        .expect("first send succeeds");
    assert_eq!(first.text(), Some("Paris".to_string()));
    assert_eq!(session.turns(), 1);

    // Second send: a follow-up human message that depends on the first answer.
    let second = session
        .send(
            &(),
            (),
            vec![Message::user("What language do they speak there?")],
        )
        .await
        .expect("second send succeeds");
    assert_eq!(second.text(), Some("French".to_string()));
    assert_eq!(session.turns(), 2);

    // The SAME underlying SubAgent (and harness) was reused — never rebuilt.
    assert!(
        Arc::ptr_eq(session.subagent(), &subagent_handle),
        "the session must reuse the same SubAgent Arc across sends"
    );

    // The scripted model saw exactly two requests (one per send) — a single,
    // reused harness instance accumulated both.
    let requests = model.requests();
    assert_eq!(
        requests.len(),
        2,
        "one model request per send on one harness"
    );

    // The SECOND request CONTAINED the first turn's messages (context carried):
    // the original question, the first assistant answer, and the system prompt.
    let second_texts: Vec<String> = requests[1].messages.iter().map(Message::text).collect();
    assert!(
        second_texts.contains(&"What is the capital of France?".to_string()),
        "second request should retain the first user question, got {second_texts:?}"
    );
    assert!(
        second_texts.contains(&"Paris".to_string()),
        "second request should retain the first assistant answer, got {second_texts:?}"
    );
    assert!(
        second_texts.contains(&"What language do they speak there?".to_string()),
        "second request should include the follow-up human message, got {second_texts:?}"
    );
    // The fixed system prompt is seeded exactly once (not duplicated per send).
    let system_count = requests[1]
        .messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .count();
    assert_eq!(system_count, 1, "system prompt seeded once, not per send");
}

#[tokio::test]
async fn session_emits_reuse_event_only_after_first_send() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("scripted", Arc::new(ScriptedModel::replies(vec!["a", "b"])));
    let subagent = Arc::new(SubAgent::new("helper", "helps", Arc::new(harness)));

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    let mut session = SubAgentSession::new(subagent).with_events(sink);

    session
        .send(&(), (), vec![Message::user("one")])
        .await
        .expect("first send");
    session
        .send(&(), (), vec![Message::user("two")])
        .await
        .expect("second send");

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    let reused: Vec<usize> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::SubAgentReused { name, turn } if name == "helper" => Some(*turn),
            _ => None,
        })
        .collect();
    assert_eq!(
        reused,
        vec![1],
        "exactly one reuse event, for the second send (turn 1)"
    );
    // Both sends still emit the started/completed bracket (depth 1).
    let started = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::SubAgentStarted { depth, .. } if *depth == 1))
        .count();
    assert_eq!(started, 2, "each send brackets with SubAgentStarted");
}

#[tokio::test]
async fn session_reset_clears_transcript_and_turns() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("scripted", Arc::new(ScriptedModel::replies(vec!["x", "y"])));
    let subagent = Arc::new(SubAgent::new("r", "resets", Arc::new(harness)));

    let mut session = SubAgentSession::new(subagent);
    session
        .send(&(), (), vec![Message::user("first")])
        .await
        .expect("send");
    assert_eq!(session.turns(), 1);
    assert!(!session.transcript().is_empty());

    session.reset();
    assert_eq!(session.turns(), 0);
    assert!(session.transcript().is_empty());
}

#[test]
fn subagent_tool_schema_uses_name_and_description() {
    let subagent = Arc::new(SubAgent::new(
        "writer",
        "writes prose",
        Arc::new(child_harness("x")),
    ));
    let tool = SubAgentTool::new(subagent);
    let schema = tool.schema();
    assert_eq!(schema.name, "writer");
    assert_eq!(schema.description, "writes prose");
    assert_eq!(schema.parameters["required"][0], "input");
}

#[tokio::test]
async fn subagent_deltas_propagate_to_parent_stream() {
    use crate::harness::agent_loop::AgentStreamItem;
    use futures::StreamExt;

    // Child streams its answer; parent delegates to it via a tool then finishes.
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("the streamed child answer")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("parent final answer"),
        ])),
    );

    let items: Vec<AgentStreamItem> = parent
        .invoke_stream(
            &(),
            (),
            RunConfig::new("parent"),
            vec![Message::user("delegate")],
        )
        .collect()
        .await;
    let events: Vec<AgentEvent> = items
        .iter()
        .filter_map(|i| match i {
            AgentStreamItem::Event(record) => Some(record.event.clone()),
            _ => None,
        })
        .collect();

    // Sub-agent lifecycle at depth 1 is visible in the parent stream.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::SubAgentStarted { depth: 1, .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::SubAgentCompleted { depth: 1, .. }))
    );

    // The CHILD's own model deltas propagate to the parent stream, stamped with
    // the child's run id (lineage), not the parent's.
    let child_delta_pos = events.iter().position(|e| {
        matches!(e, AgentEvent::ModelDelta { run_id, .. } if run_id.as_str().starts_with("researcher-d1"))
    });
    assert!(
        child_delta_pos.is_some(),
        "child model deltas must appear in the parent stream"
    );
    // The parent's own deltas are stamped with the parent run id.
    assert!(events.iter().any(
        |e| matches!(e, AgentEvent::ModelDelta { run_id, .. } if run_id.as_str() == "parent")
    ));
    // The child delta arrives before the parent's run completes.
    let parent_completed_pos = events.iter().position(
        |e| matches!(e, AgentEvent::RunCompleted { run_id } if run_id.as_str() == "parent"),
    );
    assert!(child_delta_pos < parent_completed_pos);

    match items.last() {
        Some(AgentStreamItem::Completed(run)) => {
            assert_eq!(run.text().as_deref(), Some("parent final answer"))
        }
        other => panic!("expected Completed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn non_streaming_parent_does_not_stream_child_deltas() {
    // A non-streaming parent (invoke, not invoke_stream) must leave the child on
    // the unary path: no child ModelDelta events land on the shared sink.
    let recorder = Arc::new(RecordingListener::new());
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers",
        Arc::new(child_harness("child answer")),
    ));
    let tool = Arc::new(SubAgentTool::new(child));

    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "q" })),
            text_response("done"),
        ])),
    );

    let ctx: RunContext<()> = RunContext::new(RunConfig::new("parent"), ());
    ctx.events.subscribe(recorder.clone());
    parent
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let child_deltas = recorder.events().into_iter().any(|record| {
        matches!(record.event, AgentEvent::ModelDelta { ref run_id, .. } if run_id.as_str().starts_with("researcher-d1"))
    });
    assert!(
        !child_deltas,
        "a non-streaming parent must not emit child model deltas"
    );
}
