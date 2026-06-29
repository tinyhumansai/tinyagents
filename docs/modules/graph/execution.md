# Graph Execution Model And Parallelization

Milestone 1 executor:

- sequential
- one active node at a time
- whole-state updates
- direct or string conditional routes
- recursion limit

Target executor:

- Pregel-like bulk synchronous parallel supersteps
- multiple active tasks per step
- immutable state snapshot during a step
- writes visible in the next step only
- deterministic write ordering before reducer application
- reducer/channel updates at step boundaries
- checkpoint after step completion
- pending writes from completed tasks
- cached writes replay without rerunning nodes
- resume from checkpoint
- recursive call tracking
- child graph and child agent run tracking

Superstep lifecycle:

1. Load checkpoint, active tasks, and pending writes.
2. Emit step started event.
3. Match cached task writes when cache policy allows it.
4. Run active tasks under concurrency, timeout, retry, and cancellation policy.
5. Collect writes, commands, sends, interrupts, and errors.
6. Persist task writes as pending writes when checkpointing supports it.
7. Apply channel reducers at the step boundary.
8. Select next active tasks from channel version changes and routing commands.
9. Persist the checkpoint according to durability mode.
10. Emit checkpoint, update, task, and step completion events.

Checkpointing mid-node should be avoided. Async Rust stack suspension is not a
stable persistence primitive; rerunning a node from the beginning is easier to
reason about and matches interrupt semantics.

## Parallelization

Parallel execution is represented as multiple active tasks in one superstep. A
node can route to more than one next node through conditional routing, a
command, or `Send` packets.

```rust
Command::new()
    .update(update)
    .goto([
        Send::new("retrieve_docs", json!({ "query": "billing" })),
        Send::new("retrieve_docs", json!({ "query": "refund" })),
        Send::new("score_risk", json!({ "account_id": 42 })),
    ])
```

Parallel execution rules:

- all active nodes in a superstep read the same committed state snapshot
- node-local reads may optionally include that node's own pending writes for
  branch decisions
- each node returns partial updates, commands, interrupts, sends, or errors
- channels merge successful writes at the step boundary
- conflicting writes produce reducer/channel errors, not arbitrary last-writer
  behavior
- a failed required node fails the step unless an error handler or policy routes
  it elsewhere
- completed writes can be preserved as pending writes when other nodes fail
- concurrency is bounded by graph defaults and run config

Parallelism must be visible in events:

- `StepStarted { active: [...] }`
- `TaskStarted`
- `TaskCompleted`
- `TaskFailed`
- `TaskCached`
- `StateUpdated`
- `RouteSelected`
- `CheckpointSaved`
- `StepCompleted`

For agent-specific fanout, forked runtime context, and shared-cache semantics,
see [Parallel agents and context forking](parallel-agents-forking.md).
