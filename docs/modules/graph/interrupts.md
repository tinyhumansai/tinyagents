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
