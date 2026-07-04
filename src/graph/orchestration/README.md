# graph::orchestration

Graph-level orchestration controls.

This module is the graph runtime's managed child-work surface. It gives
language-model orchestrators stable task ids and typed controls — `spawn`,
`await`, `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, and
`steer` — without exposing raw executor handles such as `tokio::JoinHandle`.
A model asks for work by task id and observes lifecycle status; it never
touches an in-process future/handle directly, so the same tool surface works
whether the "child" is an in-process subgraph, a sub-agent, or (in principle)
out-of-process work fronted by a `TaskStore` implementation.

The controls are ordinary harness tools. Use `OrchestrationTool` directly,
call `orchestration_tools` to build the full set, or call
`register_orchestration_tools` to insert them into a
`harness::tool::ToolRegistry` alongside any other tools.

## Public surface

### Tools (`tool.rs`)

- `OrchestrationTool` — a single control, constructed with an
  `OrchestrationToolKind` and a `TaskStore`; `.with_steering(..)` wires it to a
  `SteeringRegistry` for controls that need to reach a running task (e.g.
  `steer`, `cancel`).
- `orchestration_tools(store)` — builds the full default set of controls
  against one store.
- `orchestration_tools_with_steering(store, steering)` — same, with steering
  wired.
- `register_orchestration_tools(registry, store, ..)` — inserts the full set
  into a `ToolRegistry`.
- `orchestration_tool_schema(kind)` / `orchestration_tool_schemas()` — the
  JSON tool schemas, independent of a concrete store (useful for prompting or
  schema inspection without constructing tools).
- `SteeringRegistry` — a concurrent map from `TaskId` to `SteeringHandle`
  (`register` / `deregister` / `get`), letting a `steer`/`cancel` control reach
  a task that is currently running.

### Task model (`types.rs`)

- `OrchestrationTaskKind` — what a task *is* (subgraph run, sub-agent run,
  ...); `as_str()`.
- `OrchestrationTaskStatus` — lifecycle state; `is_terminal()` / `is_live()`
  predicates.
- `OrchestrationTaskSpec` — the request to spawn a task: kind, lineage
  (thread/node), timeout, input payload, metadata. Built with `new` +
  `with_lineage` / `with_thread` / `with_node` / `with_timeout_ms` /
  `with_input` / `with_metadata`.
- `OrchestrationTaskResult` — a completed task's output (`text` / `output`
  constructors for the common shapes).
- `OrchestrationTaskRecord` — the durable record a `TaskStore` holds: spec +
  current status + result once terminal. `pending(spec)` constructs the
  initial record; `task_id()` / `is_terminal()` accessors.
- `OrchestrationTaskFilter` — filters for `list` (`with_kind`,
  `created_between`, `matches(&record)`).
- `OrchestrationToolKind` — the control identifiers (`spawn`, `await`,
  `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, `steer`);
  `name()` / `description()` give the tool-facing strings.
- `OrchestrationControlOutcome` — the typed result of invoking a control.

### Storage (`store.rs`)

- `TaskStore` (trait) — durable task bookkeeping: create/update/list/get
  records by id, apply an `OrchestrationTaskFilter`.
  - `InMemoryTaskStore` — in-process implementation; `from_records(..)` seeds
    it for tests.
  - `JsonlTaskStore` — append-only JSONL-backed implementation;
    `JsonlTaskStore::open(path)`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Task kind/status/spec/result/record/filter types, `OrchestrationToolKind`, `OrchestrationControlOutcome`. |
| `tool.rs` | `OrchestrationTool`, `SteeringRegistry`, tool constructors and schemas. |
| `store.rs` | `TaskStore` trait, `InMemoryTaskStore`, `JsonlTaskStore`. |
| `test.rs` | Unit tests (spawn/await/cancel/timeout/race semantics, store round-trips, filters). |

## Operational constraints

- Controls that reach a *running* task (`steer`, `cancel`) require a
  `SteeringRegistry` populated with that task's `SteeringHandle`; without one,
  those controls only affect terminal-state bookkeeping in the `TaskStore`,
  not the in-flight task itself.
- `JsonlTaskStore` is append-only: task-record updates are appended, not
  rewritten in place, so a long-lived store grows monotonically. Compact or
  rotate externally if that matters for your deployment.
- `OrchestrationTaskFilter::matches` is evaluated in-process against loaded
  records; a `TaskStore` backend is not required to push the filter down, so
  `list` cost scales with total record count unless a given implementation
  optimizes it.
