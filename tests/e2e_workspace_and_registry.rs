//! End-to-end coverage for two cross-cutting harness surfaces:
//!
//! * **Part A — workspace isolation.** [`WorkspaceDescriptor`] path policy,
//!   the [`SharedRootWorkspace`] [`WorkspaceIsolation`] provider, threading a
//!   descriptor into a tool via [`ToolExecutionContext`], serde round-tripping,
//!   and the workspace lifecycle [`AgentEvent`] kinds.
//! * **Part B — registry diagnostics.** Building a [`CapabilityRegistry`]
//!   through the public API and introspecting it via [`RegistrySnapshot`] and
//!   [`CapabilityRegistry::diagnostics`].

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, FakeTool};
use tinyagents::harness::tool::{
    SandboxMode, Tool, ToolCall, ToolExecutionContext, ToolResult, ToolSchema,
};
use tinyagents::harness::usage::Usage;
use tinyagents::harness::workspace::{cleanup_workspace, prepare_workspace};
use tinyagents::language::Blueprint;
use tinyagents::{
    CapabilityRegistry, ComponentKind, DiagnosticSeverity, RegistrySnapshot, SharedRootWorkspace,
    WorkspaceDescriptor, WorkspaceIsolation,
};

// ---------------------------------------------------------------------------
// Part A — workspace isolation
// ---------------------------------------------------------------------------

/// Test 1: `allows` accepts the root and trusted roots, rejects everything
/// else, and normalizes `..` escapes before comparing.
#[test]
fn descriptor_allows_root_and_trusted_roots_but_blocks_escapes() {
    let ws = WorkspaceDescriptor::new("/work/a")
        .with_trusted_root("/shared")
        .with_sandbox(SandboxMode::Required);

    assert!(ws.allows(Path::new("/work/a/x")));
    assert!(ws.allows(Path::new("/shared/y")));
    assert!(!ws.allows(Path::new("/etc/passwd")));
    // `..` escape is lexically normalized to `/work/b/secret` and rejected.
    assert!(!ws.allows(Path::new("/work/a/../b/secret")));

    assert_eq!(ws.root, std::path::PathBuf::from("/work/a"));
    assert_eq!(ws.trusted_roots, vec![std::path::PathBuf::from("/shared")]);
    assert_eq!(ws.sandbox, SandboxMode::Required);
}

/// Test 2: the shared-root provider prepares a descriptor rooted at the shared
/// dir tagged with the run id, and cleanup is a no-op.
#[tokio::test]
async fn shared_root_workspace_prepares_descriptor_and_cleans_up() {
    let provider = SharedRootWorkspace::new("/shared/root").with_sandbox(SandboxMode::Required);

    let descriptor = provider.prepare("run-77", Some("worker")).await.unwrap();
    assert_eq!(descriptor.root, std::path::PathBuf::from("/shared/root"));
    assert_eq!(descriptor.policy_id, "run-77");
    assert_eq!(descriptor.sandbox, SandboxMode::Required);
    assert!(descriptor.allows(Path::new("/shared/root/out.txt")));

    provider.cleanup(&descriptor).await.unwrap();
}

/// A tool that discovers its allowed root from the execution context instead of
/// an application global.
struct WorkspaceReadingTool;

#[async_trait]
impl Tool<()> for WorkspaceReadingTool {
    fn name(&self) -> &str {
        "workspace_root"
    }

    fn description(&self) -> &str {
        "reports the allowed workspace root"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    async fn call(&self, _state: &(), call: ToolCall) -> tinyagents::Result<ToolResult> {
        // Direct calls have no execution context / workspace.
        Ok(ToolResult::text(call.id, call.name, "no-workspace"))
    }

    async fn call_with_context(
        &self,
        _state: &(),
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> tinyagents::Result<ToolResult> {
        let content = match context.workspace {
            Some(ws) => ws.root.display().to_string(),
            None => "no-workspace".to_owned(),
        };
        Ok(ToolResult::text(call.id, call.name, content))
    }
}

/// Test 3: a tool reads its workspace root from the execution context, and
/// falls back to "no-workspace" when none is attached.
#[tokio::test]
async fn tool_reads_workspace_from_execution_context() {
    let tool = WorkspaceReadingTool;
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("ws-run"), ());

    // With a workspace attached, the tool reports its root.
    let exec_ctx = ToolExecutionContext::from_run_context(&ctx)
        .with_workspace(WorkspaceDescriptor::new("/work"));
    let result = tool
        .call_with_context(&(), ToolCall::new("c", "t", json!({})), exec_ctx)
        .await
        .unwrap();
    assert_eq!(result.content, "/work");

    // Without a workspace, the tool reports the sentinel.
    let bare_ctx = ToolExecutionContext::from_run_context(&ctx);
    let result = tool
        .call_with_context(&(), ToolCall::new("c", "t", json!({})), bare_ctx)
        .await
        .unwrap();
    assert_eq!(result.content, "no-workspace");
}

/// Test 4: descriptors round-trip through serde for audit logging.
#[test]
fn descriptor_serde_round_trip() {
    let ws = WorkspaceDescriptor::new("/work/a")
        .with_trusted_root("/shared")
        .with_policy_id("policy-1")
        .with_sandbox(SandboxMode::Disabled);

    let json = serde_json::to_string(&ws).unwrap();
    let back: WorkspaceDescriptor = serde_json::from_str(&json).unwrap();
    assert_eq!(ws, back);
}

/// Test 5: the workspace lifecycle events expose stable audit kinds.
#[test]
fn workspace_event_kinds_are_stable() {
    let prepared = AgentEvent::WorkspacePrepared {
        policy_id: "p".into(),
        root: "/w".into(),
    };
    assert_eq!(prepared.kind(), "workspace.prepared");

    let violation = AgentEvent::WorkspaceViolation {
        path: "/etc/passwd".into(),
    };
    assert_eq!(violation.kind(), "workspace.violation");

    let cleanup = AgentEvent::WorkspaceCleanup {
        policy_id: "p".into(),
        error: None,
    };
    assert_eq!(cleanup.kind(), "workspace.cleanup");
}

/// A tool that, from its execution context, both reports its allowed root and
/// fail-closed enforces a fixed out-of-root path (emitting `WorkspaceViolation`
/// on the run's event sink). The violation is captured rather than propagated,
/// so the harness run continues and the assertions can inspect it.
struct WorkspaceEnforcingTool;

#[async_trait]
impl Tool<()> for WorkspaceEnforcingTool {
    fn name(&self) -> &str {
        "workspace_guard"
    }

    fn description(&self) -> &str {
        "reports the workspace root and enforces an out-of-root path"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(self.name(), self.description(), json!({"type": "object"}))
    }

    async fn call(&self, _state: &(), call: ToolCall) -> tinyagents::Result<ToolResult> {
        Ok(ToolResult::text(call.id, call.name, "no-workspace"))
    }

    async fn call_with_context(
        &self,
        _state: &(),
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> tinyagents::Result<ToolResult> {
        let ws = context
            .workspace
            .clone()
            .expect("the run was configured with a workspace");
        let root = ws.root.display().to_string();
        // Enforcing an out-of-root path fails closed and emits a violation event.
        let blocked = ws
            .enforce(Path::new("/etc/shadow"), &context.events)
            .is_err();
        Ok(ToolResult::text(
            call.id,
            call.name,
            format!("root={root};blocked={blocked}"),
        ))
    }
}

/// Test 5b: a full harness run threads a [`WorkspaceDescriptor`] through
/// [`RunContext::with_workspace`]; the tool reads its allowed root from the
/// execution context, and enforcing an out-of-root path blocks the operation
/// and emits [`AgentEvent::WorkspaceViolation`] on the run's event sink.
#[tokio::test]
async fn harness_run_threads_workspace_and_enforces_out_of_root() {
    fn tool_call_response(name: &str) -> ModelResponse {
        ModelResponse {
            message: AssistantMessage {
                id: Some("m1".into()),
                content: Vec::new(),
                tool_calls: vec![ToolCall::new("c1", name, json!({}))],
                usage: Some(Usage::new(4, 1)),
            },
            usage: Some(Usage::new(4, 1)),
            finish_reason: Some("tool_calls".into()),
            raw: None,
            resolved_model: None,
            continue_turn: None,
        }
    }
    fn text_response(text: &str) -> ModelResponse {
        ModelResponse {
            message: AssistantMessage {
                id: None,
                content: vec![ContentBlock::Text(text.into())],
                tool_calls: Vec::new(),
                usage: Some(Usage::new(2, 1)),
            },
            usage: Some(Usage::new(2, 1)),
            finish_reason: Some("stop".into()),
            raw: None,
            resolved_model: None,
            continue_turn: None,
        }
    }

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("workspace_guard"),
            text_response("done"),
        ])),
    );
    harness.register_tool(Arc::new(WorkspaceEnforcingTool));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("ws-harness"), ())
        .with_events(recorder.sink())
        .with_workspace(WorkspaceDescriptor::new("/work"));

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("the run completes after one tool call and a final message");

    assert_eq!(run.tool_calls, 1, "the guard tool ran exactly once");
    // The tool discovered its allowed root from the execution context and the
    // out-of-root enforce blocked.
    assert!(
        run.messages
            .iter()
            .any(|m| m.text().contains("root=/work;blocked=true")),
        "the tool result should report the threaded root and a blocked out-of-root path"
    );
    // The blocked path emitted a WorkspaceViolation on the shared sink.
    assert!(
        recorder.kinds().iter().any(|k| k == "workspace.violation"),
        "enforcing an out-of-root path must emit workspace.violation, got {:?}",
        recorder.kinds()
    );
    let violated_path = recorder.events().into_iter().find_map(|e| match e {
        AgentEvent::WorkspaceViolation { path } => Some(path),
        _ => None,
    });
    assert_eq!(violated_path.as_deref(), Some("/etc/shadow"));
}

/// Test 5c: the [`prepare_workspace`]/[`cleanup_workspace`] lifecycle helpers
/// emit the paired `WorkspacePrepared`/`WorkspaceCleanup` audit events on the
/// event sink they are given, driven through a real [`WorkspaceIsolation`]
/// provider ([`SharedRootWorkspace`]).
#[tokio::test]
async fn workspace_lifecycle_helpers_emit_prepared_and_cleanup_events() {
    let provider = SharedRootWorkspace::new("/shared/root").with_sandbox(SandboxMode::Required);
    let recorder = EventRecorder::new();
    let sink = recorder.sink();

    let descriptor = prepare_workspace(&provider, &sink, "run-42", Some("worker"))
        .await
        .expect("prepare succeeds");
    assert_eq!(descriptor.root, std::path::PathBuf::from("/shared/root"));
    assert_eq!(descriptor.policy_id, "run-42");

    cleanup_workspace(&provider, &sink, &descriptor)
        .await
        .expect("cleanup succeeds");

    let kinds = recorder.kinds();
    assert!(
        kinds.iter().any(|k| k == "workspace.prepared"),
        "prepare_workspace must emit workspace.prepared, got {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "workspace.cleanup"),
        "cleanup_workspace must emit workspace.cleanup, got {kinds:?}"
    );

    // The prepared event carries the descriptor's policy id and root.
    let prepared = recorder.events().into_iter().find_map(|e| match e {
        AgentEvent::WorkspacePrepared { policy_id, root } => Some((policy_id, root)),
        _ => None,
    });
    assert_eq!(
        prepared,
        Some(("run-42".to_string(), "/shared/root".to_string()))
    );
}

// ---------------------------------------------------------------------------
// Part B — registry diagnostics
// ---------------------------------------------------------------------------

/// Builds a minimal, valid [`Blueprint`] for graph registration.
fn blueprint(id: &str) -> Blueprint {
    Blueprint {
        graph_id: id.to_owned(),
        start: "a".to_owned(),
        channels: Vec::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
        defaults: Vec::new(),
        ..Blueprint::default()
    }
}

/// Registers one component of each supported kind plus a model alias.
fn populated_registry() -> CapabilityRegistry<()> {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt", Arc::new(MockModel::constant("hi")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool::returning("lookup_user", "ok")))
        .unwrap();
    reg.register_graph_blueprint("flow", blueprint("flow"))
        .unwrap();
    reg.register_router("classify").unwrap();
    reg.register_reducer("append").unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt").unwrap();
    reg
}

/// Test 6: the snapshot lists every registered component with correct counts,
/// kind filtering, and a DOT export that clusters each kind.
#[test]
fn snapshot_lists_all_components_with_counts_and_dot() {
    let reg = populated_registry();
    let snapshot = reg.snapshot();

    assert!(!snapshot.is_empty());
    // model + tool + graph + router + reducer.
    assert_eq!(snapshot.len(), 5);
    assert_eq!(snapshot.count(ComponentKind::Model), 1);
    assert_eq!(snapshot.count(ComponentKind::Tool), 1);
    assert_eq!(snapshot.count(ComponentKind::Graph), 1);
    assert_eq!(snapshot.count(ComponentKind::Router), 1);
    assert_eq!(snapshot.count(ComponentKind::Reducer), 1);

    let tools = snapshot.by_kind(ComponentKind::Tool);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].id.0, "lookup_user");

    // The model carries the alias registered above.
    let models = snapshot.by_kind(ComponentKind::Model);
    assert_eq!(models[0].id.0, "gpt");
    assert!(models[0].aliases.contains(&"default".to_string()));

    let dot = snapshot.to_dot();
    assert!(dot.contains("digraph registry"));
    assert!(dot.contains("cluster_model"));
}

/// Test 7: the snapshot round-trips through serde unchanged.
#[test]
fn snapshot_serde_round_trip() {
    let snapshot = populated_registry().snapshot();

    let json = serde_json::to_string(&snapshot).unwrap();
    let back: RegistrySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, snapshot);
}

/// Test 8: a registry built through the fail-closed public API produces no
/// diagnostics.
#[test]
fn diagnostics_are_empty_for_a_healthy_registry() {
    let reg = populated_registry();
    assert!(reg.diagnostics().is_empty());
}

/// Test 9: registering the same name under two different kinds is legal
/// (same-kind duplicates are rejected at registration), and the snapshot plus
/// diagnostics surface it — the snapshot counts it once per kind and
/// `diagnostics()` flags the cross-kind reuse as a `Warning`.
#[test]
fn snapshot_and_diagnostics_flag_name_reused_across_kinds() {
    let mut reg = CapabilityRegistry::<()>::new();
    // Same id "sync" registered under two distinct kinds.
    reg.register_tool(Arc::new(FakeTool::returning("sync", "ok")))
        .unwrap();
    reg.register_router("sync").unwrap();
    // Plus a newer descriptor-only kind, aliased for good measure.
    reg.register_descriptor(ComponentKind::TaskStore, "queue")
        .unwrap();
    reg.alias(ComponentKind::TaskStore, "jobs", "queue")
        .unwrap();

    // The snapshot lists "sync" once per kind and the new TaskStore kind.
    let snapshot = reg.snapshot();
    assert_eq!(snapshot.count(ComponentKind::Tool), 1);
    assert_eq!(snapshot.count(ComponentKind::Router), 1);
    assert_eq!(snapshot.count(ComponentKind::TaskStore), 1);
    let task_stores = snapshot.by_kind(ComponentKind::TaskStore);
    assert_eq!(task_stores[0].id.0, "queue");
    assert!(task_stores[0].aliases.contains(&"jobs".to_string()));

    // Diagnostics flag "sync" as reused across the Router and Tool kinds.
    let diagnostics = reg.diagnostics();
    let reuse = diagnostics
        .iter()
        .find(|d| d.name == "sync")
        .expect("a diagnostic should flag the cross-kind name reuse");
    assert_eq!(reuse.severity, DiagnosticSeverity::Warning);
    assert!(
        reuse.message.contains("multiple kinds"),
        "the diagnostic should explain the cross-kind reuse, got {:?}",
        reuse.message
    );
}
