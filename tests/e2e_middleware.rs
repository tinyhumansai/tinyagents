//! End-to-end coverage for a stacked built-in middleware pipeline.
//!
//! Builds an [`AgentHarness`] with several middleware composed together and
//! drives it with deterministic testkit doubles (no live provider):
//!
//! - a [`RetryMiddleware`] wrap-hook around a *flaky* model (errors once, then
//!   succeeds) so the run recovers without the caller seeing the transient
//!   failure;
//! - a [`ToolAllowlistMiddleware`] that rejects a tool call whose name is not on
//!   the allowlist before the tool runs;
//! - a [`RedactionMiddleware`] that scrubs a secret from the final model text;
//! - a [`TracingMiddleware`] that records per-phase begin counts.
//!
//! All behavioral assertions use the testkit [`Trajectory`]/[`EventRecorder`]
//! plus the middleware's own inspection helpers — never the exact model prose.
//!
//! The "flaky" model composes a testkit [`ScriptedModel`] for the success path:
//! `ScriptedModel` can only ever yield queued *responses*, so the transient
//! error is injected by a thin [`FlakyModel`] wrapper that fails its first
//! `invoke` and then delegates to the inner scripted model. The harness's own
//! retry path is disabled (a `max_attempts: 1` [`RetryPolicy`]) so the recovery
//! is attributable solely to the [`RetryMiddleware`] wrap hook.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents::TinyAgentsError;
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::{
    RedactionMiddleware, RetryMiddleware, ToolAllowlistMiddleware, TracingMiddleware,
};
use tinyagents::harness::model::{ChatModel, ModelRequest, ModelResponse};
use tinyagents::harness::retry::RetryPolicy;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::testkit::{EventRecorder, FakeTool, ScriptedModel, Trajectory};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;

// ── Test doubles ──────────────────────────────────────────────────────────────

/// A [`ChatModel`] that fails its first `fail_first` invocations with a
/// retryable [`TinyAgentsError::Model`] error and then delegates to an inner
/// [`ScriptedModel`].
///
/// This makes "errors then succeeds" deterministic while still using the
/// testkit [`ScriptedModel`] for the success path.
struct FlakyModel {
    fail_first: usize,
    calls: Mutex<usize>,
    inner: ScriptedModel,
}

impl FlakyModel {
    fn new(fail_first: usize, inner: ScriptedModel) -> Self {
        Self {
            fail_first,
            calls: Mutex::new(0),
            inner,
        }
    }
}

#[async_trait]
impl ChatModel<()> for FlakyModel {
    async fn invoke(&self, state: &(), request: ModelRequest) -> tinyagents::Result<ModelResponse> {
        let n = {
            // Scope the guard so it is dropped before the `.await` below
            // (a `MutexGuard` is not `Send`).
            let mut calls = self.calls.lock().expect("FlakyModel calls lock poisoned");
            *calls += 1;
            *calls
        };
        if n <= self.fail_first {
            return Err(TinyAgentsError::Model(format!(
                "flaky transient failure #{n}"
            )));
        }
        self.inner.invoke(state, request).await
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A model response that requests a single tool call (no text content).
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

/// A final text response carrying the given assistant text.
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

/// Builds a [`Trajectory`] from every event captured by `recorder`.
fn trajectory(recorder: &EventRecorder) -> Trajectory {
    Trajectory::from_events(recorder.events())
}

/// Counts [`AgentEvent::RetryScheduled`] events in the recorder.
fn retry_scheduled_count(recorder: &EventRecorder) -> usize {
    recorder
        .events()
        .iter()
        .filter(|e| matches!(e, AgentEvent::RetryScheduled { .. }))
        .count()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A stacked pipeline (retry wrap + allowlist + redaction + tracing) recovers
/// from a transient model failure, runs an allowed tool, redacts the secret
/// from the final text, and records the expected per-phase trace counts.
#[tokio::test]
async fn stacked_middleware_recovers_redacts_and_traces() {
    let recorder = EventRecorder::new();
    let secret = "SECRET-API-KEY-123";
    let final_text = format!("The answer is {secret}; you are welcome.");

    // First model turn requests the allowed `search` tool; second turn returns
    // the final (secret-bearing) text. The flaky wrapper fails exactly once
    // before the first scripted response is produced.
    let scripted = ScriptedModel::new(vec![
        tool_call_response("call-1", "search", json!({ "q": "x" })),
        text_response(&final_text),
    ]);
    let model = FlakyModel::new(1, scripted);

    let redaction = Arc::new(RedactionMiddleware::new([secret]));
    let tracing = Arc::new(TracingMiddleware::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("flaky", Arc::new(model))
        .set_default_model("flaky")
        .register_tool(Arc::new(FakeTool::returning("search", "tool-output")))
        .push_model_middleware(Arc::new(RetryMiddleware::new(RetryPolicy::default())))
        .push_middleware(Arc::new(ToolAllowlistMiddleware::new(["search"])))
        .push_middleware(redaction.clone())
        .push_middleware(tracing.clone())
        // Disable the harness's *own* retry so recovery is attributable to the
        // RetryMiddleware wrap hook alone.
        .with_policy(RunPolicy {
            retry: RetryPolicy {
                max_attempts: 1,
                ..RetryPolicy::default()
            },
            ..RunPolicy::default()
        });

    let ctx = RunContext::new(RunConfig::new("mw-e2e-recover"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds after the retry middleware recovers the transient failure");

    // The retry recovered: a final response was produced over two model turns.
    assert!(
        run.final_response.is_some(),
        "a final response was produced"
    );
    assert_eq!(run.model_calls, 2, "two model turns completed");
    assert_eq!(run.tool_calls, 1, "the allowed tool ran once");
    assert_eq!(
        retry_scheduled_count(&recorder),
        1,
        "exactly one retry was scheduled by the RetryMiddleware"
    );

    // The secret was redacted out of the final text.
    let text = run.text().expect("final text present");
    assert!(
        !text.contains(secret),
        "the secret must not survive in the final text: {text:?}"
    );
    assert!(
        text.contains("[REDACTED]"),
        "the redaction mask should be present: {text:?}"
    );
    assert_eq!(
        redaction.redactions(),
        1,
        "exactly one occurrence was redacted"
    );

    // Tracing recorded the expected per-phase begin counts.
    let counts = tracing.counts();
    assert_eq!(counts.agent, 1, "one before_agent");
    assert_eq!(counts.model, 2, "one before_model per model turn");
    assert_eq!(counts.tool, 1, "one before_tool for the allowed tool");
    assert_eq!(counts.delta, 0, "no streaming deltas on the unary path");
    assert_eq!(
        counts.error, 0,
        "the run did not surface an error to on_error"
    );

    // Structural trajectory assertions — never the model prose.
    let traj = trajectory(&recorder);
    traj.assert_completed();
    traj.assert_model_called_times(2);
    traj.assert_tool_called("search");
    assert!(!traj.failed(), "the run did not fail");
}

/// The same stacked pipeline rejects a tool call whose name is not on the
/// allowlist *before* the tool executes, failing the run with a validation
/// error.
#[tokio::test]
async fn stacked_middleware_rejects_disallowed_tool() {
    let recorder = EventRecorder::new();

    // The model immediately asks for the `danger` tool, which is registered but
    // not on the allowlist.
    let scripted = ScriptedModel::new(vec![tool_call_response("call-1", "danger", json!({}))]);
    let danger = Arc::new(FakeTool::returning("danger", "should never run"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("scripted", Arc::new(scripted))
        .set_default_model("scripted")
        // Register the tool so the *only* reason for failure is the allowlist,
        // not a missing-tool error.
        .register_tool(danger.clone())
        .register_tool(Arc::new(FakeTool::returning("search", "tool-output")))
        .push_model_middleware(Arc::new(RetryMiddleware::new(RetryPolicy::default())))
        .push_middleware(Arc::new(ToolAllowlistMiddleware::new(["search"])))
        .push_middleware(Arc::new(RedactionMiddleware::new(["nope"])))
        .push_middleware(Arc::new(TracingMiddleware::new()));

    let ctx = RunContext::new(RunConfig::new("mw-e2e-reject"), ()).with_events(recorder.sink());
    let err = harness
        .invoke_in_context(&(), ctx, vec![Message::user("do something dangerous")])
        .await
        .expect_err("the disallowed tool must be rejected");

    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "expected a validation rejection, got {err:?}"
    );

    // The disallowed tool was rejected *before* it could execute.
    assert!(
        danger.calls().is_empty(),
        "the disallowed tool must never be invoked"
    );

    let traj = trajectory(&recorder);
    assert!(traj.failed(), "the run failed");
    assert!(!traj.completed(), "the run did not complete");
    assert!(
        !traj.tool_was_called("danger"),
        "no ToolStarted event for the rejected tool"
    );
}
