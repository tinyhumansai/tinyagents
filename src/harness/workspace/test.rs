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

#[test]
fn descriptor_serializes_for_audit() {
    let ws = WorkspaceDescriptor::new("/work").with_policy_id("p1");
    let json = serde_json::to_value(&ws).unwrap();
    let back: WorkspaceDescriptor = serde_json::from_value(json).unwrap();
    assert_eq!(ws, back);
}
