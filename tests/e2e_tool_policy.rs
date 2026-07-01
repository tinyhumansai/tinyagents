//! End-to-end coverage of the [`ToolPolicy`] metadata surface and the
//! [`ToolPolicyMiddleware`] enforcement, driven through a real
//! [`AgentHarness`].
//!
//! These tests exercise the two enforcement points together:
//!
//! * **exposure** — `before_model` filters the model-visible toolset so a
//!   denied tool never even appears in `ModelRequest::tools`;
//! * **execution** — `before_tool` fails closed with
//!   [`TinyAgentsError::Validation`] if the model tries to call a denied tool
//!   anyway.
//!
//! A [`ScriptedModel`] is kept behind an `Arc` so, after each run, the exact
//! post-filter toolset the model was shown can be inspected via
//! [`ScriptedModel::requests`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::middleware::ToolPolicyMiddleware;
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::ScriptedModel;
use tinyagents::harness::tool::ToolCall as TC;
use tinyagents::harness::tool::{
    Tool, ToolAccess, ToolCall, ToolPolicy, ToolResult, ToolSchema, ToolSideEffects,
};
use tinyagents::harness::usage::Usage;
use tinyagents::{Result, TinyAgentsError};

// ── Model response builders (verbatim from the task spec) ─────────────────────

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![TC::new(id, name, arguments)],
            usage: Some(Usage::new(6, 2)),
        },
        usage: Some(Usage::new(6, 2)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(3, 1)),
        },
        usage: Some(Usage::new(3, 1)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
    }
}

// ── Test tools ────────────────────────────────────────────────────────────────

/// A classified, read-only tool: exposed and executable under strict policy.
struct SafeTool;

#[async_trait]
impl Tool<()> for SafeTool {
    fn name(&self) -> &str {
        "safe"
    }

    fn description(&self) -> &str {
        "A pure read-only lookup with no side effects."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, self.name(), "out"))
    }
}

/// A classified tool that declares a `payment` side effect: strict policy both
/// hides it at exposure time and denies it at execution time.
struct ChargeTool;

#[async_trait]
impl Tool<()> for ChargeTool {
    fn name(&self) -> &str {
        "charge"
    }

    fn description(&self) -> &str {
        "Moves money; declares a payment side effect."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::classified().with_side_effects(ToolSideEffects {
            payment: true,
            ..Default::default()
        })
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, self.name(), "out"))
    }
}

/// A tool that never overrides [`Tool::policy`], so it keeps the default
/// *unclassified* policy. Strict enforcement treats it as untrusted.
struct UnclassifiedTool;

#[async_trait]
impl Tool<()> for UnclassifiedTool {
    fn name(&self) -> &str {
        "unclassified"
    }

    fn description(&self) -> &str {
        "Carries no declared policy at all."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    // No `policy()` override: defaults to an unclassified `ToolPolicy`.

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, self.name(), "out"))
    }
}

/// A classified tool that is explicitly *not* background-safe.
struct ForegroundOnlyTool;

#[async_trait]
impl Tool<()> for ForegroundOnlyTool {
    fn name(&self) -> &str {
        "bg_unsafe"
    }

    fn description(&self) -> &str {
        "Classified but not safe to run in a background run."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::classified().with_access(ToolAccess {
            background_safe: false,
            ..Default::default()
        })
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, self.name(), "out"))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Strict policy hides both an unclassified tool and a payment tool from the
/// model at exposure time; only the classified read-only tool survives.
#[tokio::test]
async fn strict_policy_exposes_only_the_classified_safe_tool() {
    let scripted = Arc::new(ScriptedModel::new(vec![text_response("finished")]));

    let mut h: AgentHarness<()> = AgentHarness::new();
    h.register_model("mock", scripted.clone());
    h.register_tool(Arc::new(SafeTool));
    h.register_tool(Arc::new(ChargeTool));
    h.register_tool(Arc::new(UnclassifiedTool));

    // Build the middleware AFTER registering tools so it captures every policy.
    let mw = ToolPolicyMiddleware::strict(h.tools().policies());
    h.push_middleware(Arc::new(mw));

    let run = h
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.text(), Some("finished".to_string()));

    let requests = scripted.requests();
    assert_eq!(requests.len(), 1, "exactly one model call");
    let exposed: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        exposed,
        vec!["safe"],
        "strict policy must hide the payment and unclassified tools"
    );
}

/// Strict policy rejects execution of a denied tool the model tries to call
/// anyway, surfacing [`TinyAgentsError::Validation`] from the run.
#[tokio::test]
async fn strict_policy_rejects_execution_of_denied_payment_tool() {
    let scripted = Arc::new(ScriptedModel::new(vec![
        tool_call_response("c1", "charge", json!({})),
        text_response("done"),
    ]));

    let mut h: AgentHarness<()> = AgentHarness::new();
    h.register_model("mock", scripted.clone());
    h.register_tool(Arc::new(SafeTool));
    h.register_tool(Arc::new(ChargeTool));

    let mw = ToolPolicyMiddleware::strict(h.tools().policies());
    h.push_middleware(Arc::new(mw));

    let err = h
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("charge tool must be denied at before_tool");

    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "expected a Validation error, got {err:?}"
    );
}

/// A read-only classified tool is exposed and executes normally end-to-end:
/// the model calls it, receives the result, then returns a final text response.
#[tokio::test]
async fn classified_read_only_tool_executes_end_to_end() {
    let scripted = Arc::new(ScriptedModel::new(vec![
        tool_call_response("c1", "safe", json!({})),
        text_response("all done"),
    ]));

    let mut h: AgentHarness<()> = AgentHarness::new();
    h.register_model("mock", scripted.clone());
    h.register_tool(Arc::new(SafeTool));

    let mw = ToolPolicyMiddleware::strict(h.tools().policies());
    h.push_middleware(Arc::new(mw));

    let run = h
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.tool_calls, 1, "the safe tool ran exactly once");
    assert_eq!(run.model_calls, 2, "tool call turn plus final text turn");
    assert_eq!(
        run.final_response.unwrap().text(),
        "all done",
        "final assistant text is returned"
    );

    // The safe tool was exposed on the first model call.
    let requests = scripted.requests();
    assert!(
        requests[0].tools.iter().any(|t| t.name == "safe"),
        "safe tool must be visible to the model"
    );
}

/// `require_background_safe(true)` hides a classified tool whose policy sets
/// `background_safe = false`, while keeping a background-safe tool exposed.
#[tokio::test]
async fn require_background_safe_hides_foreground_only_tool() {
    let scripted = Arc::new(ScriptedModel::new(vec![text_response("finished")]));

    let mut h: AgentHarness<()> = AgentHarness::new();
    h.register_model("mock", scripted.clone());
    h.register_tool(Arc::new(SafeTool));
    h.register_tool(Arc::new(ForegroundOnlyTool));

    let mw = ToolPolicyMiddleware::new(h.tools().policies()).require_background_safe(true);
    h.push_middleware(Arc::new(mw));

    let run = h
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("finished".to_string()));

    let requests = scripted.requests();
    let exposed: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        exposed,
        vec!["safe"],
        "the non-background-safe tool must be hidden"
    );
}

/// Pure unit-style assertions over [`ToolPolicy`] metadata and its serde
/// round-trip.
#[tokio::test]
async fn tool_policy_metadata_and_serde_round_trip() {
    // A read-only policy is classified and declares no side effects.
    let read_only = ToolPolicy::read_only();
    assert!(read_only.classified, "read_only is classified");
    assert!(
        !read_only.has_side_effects(),
        "read_only declares no side effects"
    );
    assert!(
        read_only.access.background_safe,
        "read_only is background-safe"
    );

    // A payment policy does declare a side effect.
    let payment = ToolPolicy::classified().with_side_effects(ToolSideEffects {
        payment: true,
        ..Default::default()
    });
    assert!(payment.classified);
    assert!(
        payment.has_side_effects(),
        "a payment policy has side effects"
    );

    // Full serde round-trip preserves the policy exactly.
    let requiring_approval = ToolPolicy::classified()
        .with_side_effects(ToolSideEffects {
            network: true,
            external_service: true,
            ..Default::default()
        })
        .requiring_approval();
    let encoded = serde_json::to_string(&requiring_approval).expect("policy serializes");
    let decoded: ToolPolicy = serde_json::from_str(&encoded).expect("policy deserializes");
    assert_eq!(
        decoded, requiring_approval,
        "serde round-trip preserves the ToolPolicy"
    );
    assert!(decoded.access.approval_required);
    assert!(decoded.has_side_effects());
}
