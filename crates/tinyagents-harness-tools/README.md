# `tinyagents-harness-tools`

This crate provides reusable tool implementations and stateful support for
`tinyagents-harness`. It keeps product policy and presentation outside the
agent loop while making common harness capabilities available to Rust hosts.

## Features

The default feature set enables `todos`, `goals`, and `skills`. The `subagents`
feature enables the generic job and message tools and brings in
`tinyagents-orchestration` and its task registry dependencies. Time,
clarification, plan-exit, per-call binding, and thread-lock support are
available without optional features. The `state-block` feature enables the
run-state middleware and its required `goals` and `todos` modules. The
`json-schema` feature enables schema derives on tool-owned data types.

The opt-in `fs`, `net`, and `exec` features expose the standard tool families
under `tinyagents_harness_tools::{fs,net,exec}`. The `media` feature exposes
the generic artifact-filing wrapper at `tinyagents_harness_tools::media`, and
`plan-review` exposes the host-backed review tool at
`tinyagents_harness_tools::plan_review`. These features do not change the
default set.

## Public modules and tools

- `goals` provides `GoalStore`, `GoalTool`, observer notifications, atomic
  turn accounting, budget guards, and goal context. A host calls
  `GoalStore::prepare_turn` and composes a budget stop hook for the turn.
- `todos` provides `TodoTool`, the persistent sequence model, and
  `TodoSequenceMiddleware`.
- `state_block` provides `StateBlockMiddleware`, which rebuilds the current
  goal, plan, and tool-call facts in each model request.
- `skills` provides `UseSkillTool`, `PackCatalog`, `PackRegistryHandle`, and
  `UseSkillDispatch` for binding a selected tool from a named pack.
  Hosts can recover the shared registry from an erased registered tool with
  `PackRegistryHandle::of` (or the equivalent free function
  `skills::pack_registry_handle`); the generic `ToolCallBinding` still pins each
  call's admission and execution target.
- `subagents` provides `DriverSubAgentTool`, `SubAgentJobsTool`, and
  `SubAgentMessageTool` over the registry owned by
  `tinyagents_orchestration::subagent::SubAgentJobRegistry`.
- `thread_locks` provides the public `ThreadLockMap` concurrency primitive.
- `fs` provides filesystem tools and file-state operations in `fs`.
- `net` provides network tools in `net`.
- `exec` provides `ExecTool`, `InstallTool`, process timeout utilities, and
  host-supplied execution ports in `exec`.
- `media` provides `MediaArtifactTool` and its `ArtifactFiling` port in
  `media`.
- `plan-review` provides `RequestPlanReviewTool` and its `PlanReviewPort` in
  `plan_review`.
- The crate root exports `CurrentTimeTool`, `ResolveTimeTool`,
  `AskClarificationTool`, `PlanExitTool`, `WaitTool`, `WaitLoopTool`, and the
  `per_call` binding adapter.

`WaitTool` and `WaitLoopTool` provide bounded delayed callback ticks for
orchestrator-controlled polling. They return the caller's callback message
without owning scheduling state.

Subagent registry policy stays in `tinyagents-orchestration`; this crate owns
the reusable tool declarations. The harness owns tool registration,
admission, schema preparation, discovery, and dispatch infrastructure.

### Host-backed tools

`ExecTool` and `InstallTool` receive an `ExecGate` and an `ExecBackend` from the
host. The gate authorizes the parsed request before the backend can spawn a
process; the permit carries the backend selection, capability, working
directory, and environment chosen for that call. `NativeExecBackend::new`
selects a shell flavor, but does not choose whether native execution is safe.
An embedding host wires the pair with
`ExecTool::new(Arc<dyn ExecGate>, Arc<dyn ExecBackend>)` and supplies the same
host-owned gate/backend ports to `InstallTool`.
Hosts own approval, security policy, sandbox selection, and audit policy and
may instead provide a sandbox or remote backend. Keep those decisions in the
host adapter.

For turns that can be cancelled, hosts should create a
`exec::timeout::ProcessCleanup`, run the turn future inside `scope`, drop that
future when cancelling, and await `wait` before acknowledging cancellation.
The native backend registers its child waiter with the active scope; the wait
ensures the process and captured output pipes are finished first. Work handed
to a host-owned task must explicitly inherit cleanup with
`ProcessCleanup::hold` and drop the hold after stopping.

`MediaArtifactTool` takes an `ArtifactFiling` implementation. The host owns
artifact persistence, file-root checks, and blob lifecycle; the generic wrapper
only maps returned artifact records and rewrites their paths. Likewise,
`RequestPlanReviewTool` uses `PlanReviewPort` to request a host review and
returns its resolution without owning a UI or approval policy.
