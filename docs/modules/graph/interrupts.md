# Graph Interrupts And Resume

Interrupts pause execution and return control to the caller.

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

Rules:

- interrupts require a checkpointer
- resume requires a `thread_id`
- the interrupted node restarts from the beginning
- multiple interrupts inside one task are matched by order or interrupt id
- resume values can be a single value or a map from interrupt id to value
- node code before an interrupt must be deterministic or idempotent
- side effects before an interrupt must be guarded by idempotency keys
- interrupts can be configured before or after named nodes

Compile-time `interrupt_before` and `interrupt_after` selectors are useful for
debugging, approvals, and human review at arbitrary graph boundaries without
editing node code.

## Targeted Human Steering

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
