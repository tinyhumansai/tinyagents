//! Tests for workspace isolation hooks.

use super::*;
use crate::harness::tool::SandboxMode;
use std::path::Path;

#[test]
fn descriptor_allows_paths_under_root_and_trusted_roots() {
    let ws = WorkspaceDescriptor::new("/work/agent-a")
        .with_trusted_root("/shared/cache")
        .with_sandbox(SandboxMode::Required)
        .with_policy_id("run-1");

    assert!(ws.allows(Path::new("/work/agent-a/src/main.rs")));
    assert!(ws.allows(Path::new("/shared/cache/blob")));
    // Outside every root.
    assert!(!ws.allows(Path::new("/etc/passwd")));
    // Escape attempt via `..` is normalized and rejected.
    assert!(!ws.allows(Path::new("/work/agent-a/../agent-b/secret")));
    assert_eq!(ws.sandbox, SandboxMode::Required);
    assert_eq!(ws.policy_id, "run-1");
}

#[tokio::test]
async fn shared_root_workspace_prepares_and_cleans_up() {
    let provider = SharedRootWorkspace::new("/work").with_sandbox(SandboxMode::Disabled);
    let descriptor = provider.prepare("run-42", Some("worker")).await.unwrap();

    assert_eq!(descriptor.root, std::path::PathBuf::from("/work"));
    assert_eq!(descriptor.policy_id, "run-42");
    assert_eq!(descriptor.sandbox, SandboxMode::Disabled);
    assert!(descriptor.allows(Path::new("/work/output.txt")));

    // Cleanup is a no-op for a shared root.
    provider.cleanup(&descriptor).await.unwrap();
}

#[tokio::test]
async fn prepare_and_cleanup_helpers_emit_lifecycle_events() {
    use crate::harness::events::{AgentEvent, EventSink, RecordingListener};
    use std::sync::Arc;

    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());

    let provider = SharedRootWorkspace::new("/work");
    let descriptor = prepare_workspace(&provider, &events, "run-7", Some("worker"))
        .await
        .unwrap();
    cleanup_workspace(&provider, &events, &descriptor)
        .await
        .unwrap();

    let kinds: Vec<_> = recorder.events().iter().map(|r| r.event.kind()).collect();
    assert_eq!(kinds, vec!["workspace.prepared", "workspace.cleanup"]);
    // The prepared event carries the policy id and root for audit.
    match &recorder.events()[0].event {
        AgentEvent::WorkspacePrepared { policy_id, root } => {
            assert_eq!(policy_id, "run-7");
            assert_eq!(root, "/work");
        }
        other => panic!("expected WorkspacePrepared, got {other:?}"),
    }
}

#[test]
fn enforce_blocks_unsafe_paths_and_emits_violation() {
    use crate::harness::events::{EventSink, RecordingListener};
    use std::sync::Arc;

    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());

    let ws = WorkspaceDescriptor::new("/work/agent-a");
    // Allowed path passes silently with no event.
    ws.enforce(Path::new("/work/agent-a/out.txt"), &events)
        .unwrap();
    assert!(recorder.is_empty());

    // Unsafe path fails closed and emits a violation.
    let err = ws
        .enforce(Path::new("/etc/passwd"), &events)
        .expect_err("path outside root must be blocked");
    assert!(err.to_string().contains("outside the allowed workspace"));
    assert_eq!(recorder.events()[0].event.kind(), "workspace.violation");
}

#[test]
fn run_context_workspace_threads_into_tool_execution_context() {
    use crate::harness::context::{RunConfig, RunContext};
    use crate::harness::tool::ToolExecutionContext;

    let ws = WorkspaceDescriptor::new("/work/agent-a").with_policy_id("run-9");
    let ctx: RunContext = RunContext::new(RunConfig::new("run-9"), ()).with_workspace(ws.clone());

    let tool_ctx = ToolExecutionContext::from_run_context(&ctx);
    assert_eq!(tool_ctx.workspace, Some(ws));
}

#[test]
fn descriptor_serializes_for_audit() {
    let ws = WorkspaceDescriptor::new("/work").with_policy_id("p1");
    let json = serde_json::to_value(&ws).unwrap();
    let back: WorkspaceDescriptor = serde_json::from_value(json).unwrap();
    assert_eq!(ws, back);
}
