# Graph Interrupts And Resume

Interrupts pause execution and return control to the caller. Basic
interrupt/resume — the `Interrupt` type, `Command::resume`, and
`CompiledGraph::resume`/`resume_from` — is implemented today
(`crates/tinyagents-graph/src/command/types.rs`,
`crates/tinyagents-graph/src/compiled/executor.rs`). Everything under
"Targeted Human Steering" below, plus `interrupt_before`/`interrupt_after`
selectors and resume-by-interrupt-id maps, is a **target — not implemented**
(verified by grep against `crates/tinyagents-graph/src`; see
`docs/runtime-comparison/plan.md`).

The struct actually shipped today is smaller than the one below — `id` is a
bare `String` (not `InterruptId`) and there is no `order` field, but
`task_id` is now a typed field (R5), stamped by the interrupt boundary with
the pausing branch's task id:

```rust
pub struct Interrupt {
    pub id: String,
    pub node: NodeId,
    pub payload: serde_json::Value,
    pub task_id: Option<TaskId>,
}
```

The fuller shape this doc originally described (target, not implemented):

```rust
pub struct Interrupt {
    pub id: InterruptId,
    pub node: NodeId,
    pub task_id: TaskId,
    pub payload: serde_json::Value,
    pub order: usize,
}
```

Resume API:

```rust
compiled_graph
    .resume(
        RunConfig::thread("support-123"),
        Command::resume(json!({ "approved": true })),
    )
    .await?;
```

Rules (implemented today unless marked target):

- interrupts require both a checkpointer and a `thread_id`
- if a node emits an interrupt without resumable durability, the run returns a
  resume error instead of an interrupted execution
- interrupted executions are returned only after the checkpoint needed for
  resume has been persisted
- the interrupted node restarts from the beginning
- every branch of a step that interrupts is surfaced (`GraphExecution::interrupts`
  carries all of them, not just the lowest-index one) — a `Send` fan-out of
  one node interrupting on several concurrent activations is matched by task
  id, each stamped onto its own `Interrupt::task_id`, rather than by an
  `order` field
- resume values as a map from task id to value: `Command::resume_tasks(..)` /
  `Command::resume_by_task` deliver a distinct value per interrupted task in
  one resume call, keyed by `TaskId` — `Command::resume(value)` (one value,
  fanned to every task named by the checkpoint's stamped `interrupted_nodes`
  or, absent that, to every pending task) still works and is consulted as the
  fallback for any task the map does not name
- **Target (not implemented):** resume values as a map keyed by interrupt id
  specifically (rather than task id)
- node code before an interrupt must be deterministic or idempotent
- side effects before an interrupt must be guarded by idempotency keys
- **Target (not implemented):** interrupts configured before or after named
  nodes (see below)

**Target (not implemented; see `docs/runtime-comparison/plan.md`).**
Compile-time `interrupt_before` and `interrupt_after` selectors — useful for
debugging, approvals, and human review at arbitrary graph boundaries without
editing node code — do not exist in `crates/tinyagents-graph/src` today; a
node must call the interrupt itself.

## Targeted Human Steering

**Target (not implemented; see `docs/runtime-comparison/plan.md`).** Nothing
below this point — `ResumeTarget` as a targeted-steering struct,
`resume_targeted`, or per-run/per-task/per-namespace resume routing — exists
in `crates/tinyagents-graph/src` today. The `ResumeTarget` type that does
exist (`crates/tinyagents-graph/src/compiled/types.rs`) is unrelated: it is a
`Latest`/`Checkpoint(CheckpointId)` enum selecting which checkpoint a resume
replays from, not a target-selection struct with `run_id`/`task_id`/
`interrupt_id`/`namespace` fields.

Human input during an interrupt is one form of steering. A control surface
should be able to target:

- the parent orchestrator run
- a specific child sub-agent run
- a graph task id
- a node namespace inside a subgraph
- a specific interrupt id

Targeted resume shape:

```rust
pub struct ResumeTarget {
    pub run_id: RunId,
    pub task_id: Option<TaskId>,
    pub interrupt_id: Option<InterruptId>,
    pub namespace: Vec<String>,
}

compiled_graph
    .resume_targeted(
        ResumeTarget {
            run_id,
            task_id: Some(child_task),
            interrupt_id: Some(approval_interrupt),
            namespace: vec!["supervisor".into(), "research_agent".into()],
        },
        Command::resume(json!({ "approved": true })),
    )
    .await?;
```

Rules:

- resuming a child interrupt resumes that child task, not all paused siblings
- resuming the parent orchestrator may leave child interrupts pending unless
  policy cancels or resolves them
- a human can add steering instructions while resuming, but those instructions
  must be recorded separately from the interrupt answer
- stale resume targets are rejected with the latest run/checkpoint metadata
- UI clients should present pending interrupts with run tree path, node id,
  task id, sub-agent id, and checkpoint id so humans can steer the intended
  target
