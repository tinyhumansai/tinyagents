# Harness Workspace Isolation Feature

The workspace feature owns the SDK-neutral hooks agents use when their tools run
over real files or command executors. TinyAgents does **not** own any concrete
sandbox policy; it owns the interface, so parallel agents (and their sub-agents)
can be isolated consistently and a tool can discover its allowed filesystem root
from run context instead of an application global.

Source: `src/harness/workspace/{mod.rs,types.rs}`; tests:
`src/harness/workspace/test.rs`.

## Core Types

```rust
pub struct WorkspaceDescriptor {
    pub root: PathBuf,
    pub trusted_roots: Vec<PathBuf>,
    pub policy_id: String,
    pub sandbox: SandboxMode,
}

#[async_trait]
pub trait WorkspaceIsolation: Send + Sync {
    async fn prepare(&self, run_id: &str, agent: Option<&str>) -> Result<WorkspaceDescriptor>;
    async fn cleanup(&self, descriptor: &WorkspaceDescriptor) -> Result<()>;
}
```

A [`WorkspaceDescriptor`] tells a tool which filesystem root(s) it may touch and
how strictly it must be sandboxed ([`SandboxMode::Inherit`] / `Disabled` /
`Required`, from the tool feature). A [`WorkspaceIsolation`] provider prepares a
per-agent worktree/sandbox and tears it down afterward. `SharedRootWorkspace` is
the built-in provider: it scopes every agent to one shared root without copying
(cleanup is a no-op) — a sensible default and a test double.

Build a descriptor fluently:

```rust
use tinyagents::harness::tool::SandboxMode;
use tinyagents::harness::workspace::WorkspaceDescriptor;

let ws = WorkspaceDescriptor::new("/work/agent-a")
    .with_trusted_root("/shared/cache")
    .with_sandbox(SandboxMode::Required)
    .with_policy_id("run-1");

assert!(ws.allows(std::path::Path::new("/work/agent-a/src/main.rs")));
assert!(ws.allows(std::path::Path::new("/shared/cache/blob")));
assert!(!ws.allows(std::path::Path::new("/etc/passwd")));
// Escape via `..` is normalized away, so it is rejected.
assert!(!ws.allows(std::path::Path::new("/work/agent-a/../agent-b/secret")));
```

`allows` is a **lexical** gate (it normalizes `.`/`..` without touching the
filesystem), so it works for paths that do not exist yet and never triggers a
canonicalizing syscall.

## Threading a workspace into tools

`RunContext::with_workspace(descriptor)` stores the descriptor on the context;
every `ToolExecutionContext` the run builds then carries it, so a tool reads its
allowed root from `context.workspace` rather than an application global:

```rust
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::tool::ToolExecutionContext;
use tinyagents::harness::workspace::WorkspaceDescriptor;

let ws = WorkspaceDescriptor::new("/work/agent-a").with_policy_id("run-9");
let ctx: RunContext =
    RunContext::new(RunConfig::new("run-9"), ()).with_workspace(ws.clone());

let tool_ctx = ToolExecutionContext::from_run_context(&ctx);
assert_eq!(tool_ctx.workspace, Some(ws));
```

A `None` workspace means no workspace policy is in effect. The
[`ToolPolicyMiddleware`](tool.md#tool-policy-enforcement) `require_sandbox`
enforcement reads this descriptor to decide whether a `SandboxMode::Required`
tool may run.

## Lifecycle helpers and events

Two helpers drive a provider and make the setup/teardown observable on the run's
event sink:

- `prepare_workspace(&isolation, &events, run_id, agent)` prepares an environment
  and emits `AgentEvent::WorkspacePrepared { policy_id, root }`, returning the
  descriptor to thread into the run.
- `cleanup_workspace(&isolation, &events, &descriptor)` tears it down and emits
  `AgentEvent::WorkspaceCleanup { policy_id, error }` (`error` set only when
  cleanup fails).

```rust
use std::sync::Arc;
use tinyagents::harness::events::{EventSink, RecordingListener};
use tinyagents::harness::workspace::{
    cleanup_workspace, prepare_workspace, SharedRootWorkspace,
};

let events = EventSink::new();
let recorder = Arc::new(RecordingListener::new());
events.subscribe(recorder.clone());

let provider = SharedRootWorkspace::new("/work");
let descriptor = prepare_workspace(&provider, &events, "run-7", Some("worker")).await?;
cleanup_workspace(&provider, &events, &descriptor).await?;

let kinds: Vec<_> = recorder.events().iter().map(|r| r.event.kind()).collect();
assert_eq!(kinds, vec!["workspace.prepared", "workspace.cleanup"]);
```

## Fail-closed path enforcement

Before a tool touches a path, call `WorkspaceDescriptor::enforce(path, &events)`.
It is a fail-closed gate: an allowed path returns `Ok(())` silently; a path
outside every allowed root emits `AgentEvent::WorkspaceViolation { path }` and
returns `TinyAgentsError::Validation`, so the caller blocks the operation.

```rust
let ws = WorkspaceDescriptor::new("/work/agent-a");
ws.enforce(std::path::Path::new("/work/agent-a/out.txt"), &events)?; // allowed, no event

let err = ws
    .enforce(std::path::Path::new("/etc/passwd"), &events)
    .expect_err("path outside root must be blocked");
assert!(err.to_string().contains("outside the allowed workspace"));
// A `workspace.violation` event was emitted for audit.
```

## Emitted events

| Event | When |
| --- | --- |
| `WorkspacePrepared { policy_id, root }` | `prepare_workspace` succeeds |
| `WorkspaceCleanup { policy_id, error }` | `cleanup_workspace` runs (`error` set on failure) |
| `WorkspaceViolation { path }` | `enforce` blocks an out-of-root path |

The descriptor is fully `Serialize`/`Deserialize` for registry introspection and
audit journaling.
