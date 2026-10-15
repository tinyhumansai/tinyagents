# tinyagents-tasks

Graph-independent detached-task machinery: task model, `TaskStore`
implementations, the detached task registry, restart reconciliation and the
orchestration control tools. `tinyagents-graph` and `tinyagents-orchestration`
both depend on this crate; it does not depend on the graph engine. The old
`tinyagents_graph::orchestration::*` paths remain as plain re-exports.

This crate is the managed child-work surface. It gives
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
- `OrchestrationTaskStatus` — the one durable lifecycle status for a subagent
  or task run; `is_terminal()` / `is_live()` predicates. See
  [Status mapping](#status-mapping).
- `NoEquivalentStatus` — the error of every fallible status conversion.
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
  - `DriverTaskStore` (`driver_store.rs`) — durable implementation over a
    driver `DocumentStore`; each transition writes the record and an atomic
    history snapshot. `DriverCompletionStore` does the same for completions.
- `TaskStoreRegistry<K>` (`store_registry.rs`) — process-wide cache mapping a
  host-defined scope key to a lazily-opened `Arc<dyn TaskStore>`, so a
  multi-tenant host keeps exactly one store per scope. `get_or_open` /
  `get` / `values` / `len` / `is_empty` / `clear`.
  `open_jsonl_task_store_or_memory(path)` is the standard opener: it degrades
  to an `InMemoryTaskStore` if the durable log can't be created or read.

### Process-local runtime (`runtime.rs`)

- `DetachedTaskRegistry<Metadata, Status>` — tracks the executor-only pieces
  of a detached task that cannot survive a process restart: status watch
  channel, cancellation token, abort handle, owner id, and live steering
  lookup. `TaskStore` remains the durable source of lifecycle truth; this
  registry is what an executor consults to `wait`, `cancel`, `cancel_where`,
  `cancel_all`, `snapshot(s)`, or fetch a `steering_handle` for a task it
  still owns in-process. `sweep_terminal` / the `soft_cap` passed to `new`
  bound unbounded growth from tasks nobody ever waited on.

### Orphan reconciliation (`reconcile.rs`)

- `reconcile_orphaned_tasks(store, filter, reason)` — settles every live task
  matching `filter` into a terminal state (`Cancelled` if a cancellation was
  already requested, otherwise `Failed` with the caller-supplied `reason`).
  Meant to run once at host startup against a `TaskStore` whose executor
  process may have died since the last run, before any `DetachedTaskRegistry`
  is repopulated.
- `ReconcileReport` / `ReconciledTask` / `ReconcileOutcome` — the sweep's
  per-task and aggregate results; `task_status_label(status)` gives the
  stable lowercase label used in logs.

### Restart recovery note (`recovery.rs`)

Reconciliation fixes the store; the *parent agent* still has to be told its
children died. These pure functions turn a `ReconcileReport` into a note for the
parent's next turn (wiring it into a host is the host's job):

- `recovery_children(report, reason)` — selects the interrupted children
  (everything the sweep failed or could not settle; tasks settled as cancelled
  are user intent and are skipped), ordered by creation time then task id. Each
  `RecoveryChild` carries `task_id`, `kind`, `label` (the `label` metadata, else
  the agent/graph/tool name), `last_status`, and `interrupted_reason` (the sweep's persisted reason).
  Labels and reasons are truncated to `MAX_RECOVERY_LABEL_CHARS` = 256 chars
  here, in `recovery_children`; the builder does not re-truncate rows a caller
  constructs by hand.
- `build_restart_recovery_note(children)` — a capped roster (at most
  `MAX_RECOVERY_CHILDREN` = 32 rows, then a `+N more` line) as JSON inside
  a `<child_task_facts>` block (`<` is escaped so labels cannot close it),
  followed by `RESTART_RECOVERY_INSTRUCTION`: reconcile against saved results,
  verify uncertain side effects, never blindly re-run. Empty input gives `""`.

Nothing here ever relaunches a task. For the per-tool-call side of recovery see
`tinyagents_session::run_ledger::classify_recovery`.

### Durable completion router (`completions/`)

A detached child finishes whenever it finishes; its parent still has to hear
about it exactly once, even across a restart. The completion router owns that
hand-off and nothing product-specific:

- `CompletionRecord` — one finished child keyed by `task_id`: `parent_key`,
  `agent_id`, `label`, `status` (`Success` / `Failed` / `Cancelled` /
  `Incomplete`), `result` (`CompletionResult`: text already capped by the
  result policy, `omitted_chars`, optional `CompletionArtifact`),
  `finished_at`, `attempts`, `notify_mode` and `state` (`Pending` /
  `Delivered` / `Tombstoned` / `GaveUp`).
- `CompletionStore` — `InMemoryCompletionStore` and `DriverCompletionStore`.
  The driver store keeps the latest record per task id as one document in a
  `tinystoragedrivers` collection (`orchestration_completions` by default)
  with `parentKey`, `state` and `updatedAtMs` projected for queries; an
  unreadable document is skipped with a warning. A record that settles
  (`Delivered`, `Tombstoned`, `GaveUp`, including the records `cancel_parent`
  withdraws) is written back without its `result`, so a delivered or
  withdrawn child's text does not sit in the store until compaction; the id,
  state and attempts that dedupe and tombstones need stay. `compact(retain)`
  drops settled records older than `retain`, dedupe entries included.
- `CompletionRouter` — `record` (idempotent per task id), `tombstone(task_id)`
  (the parent waited on or collected the child, so nothing further is routed; a message already pushed to a live queue is not retracted; works
  before the completion exists), `claim_pending(parent, max)` /
  `begin_turn(parent)` / `pull(parent, max)` (claim a batch and count an
  attempt; a claimed record is leased until resolved), `mark_delivered(ids)`,
  `mark_failed(ids)` (moves a record to `GaveUp` once `attempts >=
  max_attempts`, default `DEFAULT_MAX_ATTEMPTS` = 5, and returns those records
  with their full result for the host's give-up policy; the stored copy keeps
  none), `pending_for(parent)`, `cancel_parent` /
  `resume_parent` (the flag is a durable marker record, so it survives a restart), `attach_parent` / `detach_parent`, `compact`, and
  `restart_recovery_note(parent, children)`.
- `NotifyMode`, set per spawn: `Followup` (default) pushes onto the parent's
  `QueueLane::Followup` when the parent is attached, `Collect` onto
  `QueueLane::Collect`, `HoldForNextTurn` waits for `begin_turn`, `Off` is
  record-only for `pull`. A `Followup`/`Collect` record whose parent is not
  attached stays pending for `claim_pending`. A push onto a live queue counts
  as attempt one and leases the record; it settles only when the host calls
  `mark_delivered`, so a crash, a cleared queue or `detach_parent` (which
  releases that parent's leases) leaves it claimable. Delivery is at-least-once.
  `in_flight_for` lists unacknowledged pushes and `release(ids)` frees an
  abandoned claim without counting a failure.
- `CompletionFormatter` — the wording. `NeutralCompletionFormatter` is the
  default (escaped JSON in a `<completed_child_tasks>` block); a host supplies
  its own with `CompletionRouter::with_formatter`.

Task ids are the dedupe key and must be unique across parents: recording an id
already held by another parent returns `RecordOutcome::IdCollision` and stores
nothing.

After a restart, `pending_for(parent)` returns what was never delivered and
`restart_recovery_note` folds it into the recovery note beside the interrupted
children. Nothing relaunches a child. The host keeps: deciding when a parent is
idle, the delivery turn, its formatter, and what to do with a `GaveUp` record.
`tinyagents-orchestration` records into a router from `SubagentDriver` and from
`spawn_status_watcher_with_completions`; with no router configured its behaviour
is unchanged.

## Status mapping

`OrchestrationTaskStatus` is canonical. Every other subagent-run status enum
keeps its own (unchanged) wire format and converts to it; the payload-free ones
(`AgentRunStatus`, `TranscriptSubagentStatus`, `CompletionStatus`,
`SubAgentJobStatus`) also convert back. `From` is total; `TryFrom` fails with `NoEquivalentStatus`. `DetachedSubagentStatus`
carries a payload (output, question, error), as does `SubagentOutcomeKind`; both only convert out, by
reference (`From<&DetachedSubagentStatus>`, and `TryFrom<&_>` for `CompletionStatus`/`SubAgentJobStatus`); there is no reverse conversion because the payload cannot be reconstructed from a bare status.

| Type (crate) | -> `OrchestrationTaskStatus` | `OrchestrationTaskStatus` -> |
| --- | --- | --- |
| `DetachedSubagentStatus` (orchestration) | `From<&_>`: `Running`, `Completed`, `Failed`; `AwaitingUser` -> `Awaiting` | none (no payload to rebuild) |
| `SubAgentJobStatus` (orchestration) | `From`: `Queued` -> `Pending`; `Incomplete` -> `Failed` (`SubAgentJob::task_status()` refines a timeout to `TimedOut`) | `TryFrom`: `CancelRequested` -> `Running`; `TimedOut` -> `Incomplete`; `Awaiting`, `Abandoned` -> error |
| `SubagentOutcomeKind` (orchestration) | `From<&_>`: `AwaitingInput` -> `Awaiting`; `Incomplete` + `Timeout` -> `TimedOut`, other `Incomplete` -> `Failed` | none (payload) |
| `AgentRunStatus` (session) | `From`: `AwaitingUser`, `Paused` -> `Awaiting`; `Interrupted` -> `Abandoned` | `From`: `CancelRequested` -> `Running`; `TimedOut` -> `Failed`; `Abandoned` -> `Interrupted`; `Awaiting` -> `AwaitingUser` |
| `TranscriptSubagentStatus` (session) | `From`: `Incomplete` -> `Failed`; `Interrupted` -> `Abandoned` | `TryFrom`: live states -> `Running`; `TimedOut` -> `Incomplete`; `Abandoned` -> `Interrupted`; `Cancelled` -> error |
| `CompletionStatus` (tasks) | `From`: `Success` -> `Completed`, `Failed` -> `Failed`, `Cancelled` -> `Cancelled`, `Incomplete` -> `Failed` | `TryFrom`: `Completed` -> `Success`, `Failed` -> `Failed`, `Cancelled` -> `Cancelled`, `TimedOut`, `Abandoned` -> `Incomplete`; live states (`Pending`, `Running`, `Awaiting`, `CancelRequested`) -> error |

The other pairs convert directly where both sides have a meaningful value:
`SubAgentJobStatus` <-> `AgentRunStatus` (`From` out, `TryFrom` back; `Incomplete`
-> `Failed`; `AwaitingUser`, `Paused`, `Interrupted` -> error), `SubAgentJobStatus`
<-> `CompletionStatus` (`From` in, `TryFrom` out; `Queued`, `Running` -> error),
and `DetachedSubagentStatus` / `SubagentOutcomeKind` -> `AgentRunStatus`,
`SubAgentJobStatus` and `CompletionStatus` (`AwaitingUser` / `AwaitingInput`
pause states and `Running` -> error where there is no equivalent). Two caveats: the canonical status has no `Incomplete`, so the direct job/outcome <-> `CompletionStatus` pairs keep the incomplete-vs-failed distinction that a trip through `OrchestrationTaskStatus` loses; and `detached/ledger.rs` (`record_to_wait_outcome`) folds a stored task status back into `DetachedSubagentStatus`, mapping `TimedOut`, `Abandoned` and `Cancelled` to `Failed`.

Terminality is preserved by every mapping, except that
`DetachedSubagentStatus::is_terminal` treats `AwaitingUser` as terminal (the
run will not progress by itself) while the task status keeps it live so a
follow-up can resume it. Lossy collapses are the ones named above; each is
pinned by an exhaustive test. Serde forms (`snake_case` strings, plus the
externally-tagged `SubagentOutcomeKind`) are unchanged and pinned by golden
tests; the deprecated aliases and helpers are listed in the orchestration
README.

## Files

| File | Role |
| --- | --- |
| `lib.rs` | Module declarations and re-exports. |
| `status.rs` | The `CompletionStatus` <-> `OrchestrationTaskStatus` conversions (tests in `status_tests.rs`); `NoEquivalentStatus` itself is in `types.rs`. |
| `types.rs` | `NoEquivalentStatus`, task kind/status/spec/result/record/filter types, `OrchestrationToolKind`, `OrchestrationControlOutcome`, and the `DetachedTaskRegistry` snapshot/error types. |
| `tool.rs` | `OrchestrationTool`, `SteeringRegistry`, tool constructors and schemas. |
| `store.rs` | `TaskStore` trait, `InMemoryTaskStore`, and the shared transition rules. |
| `store_registry.rs` | `TaskStoreRegistry<K>`, `open_driver_task_store_or_memory`. |
| `runtime.rs` | `DetachedTaskRegistry<Metadata, Status>` — process-local executor handles keyed by task id. |
| `reconcile.rs` | `reconcile_orphaned_tasks` and its report types, for settling orphans left by a dead executor. |
| `recovery.rs` | `recovery_children` / `build_restart_recovery_note`: the parent-facing roster of interrupted children after a reconcile sweep (tests in `recovery_tests.rs`). |
| `completions/` | `types.rs`, `store.rs`, `driver.rs`, `router.rs`, `format.rs`: the completion router (tests in `router_tests.rs`, `driver_tests.rs`). |
| `lib_tests.rs` | Unit tests (spawn/await/cancel/timeout/race semantics, store round-trips, filters, reconciliation, detached-task registry). |

## Operational constraints

- Controls that reach a *running* task (`steer`, `cancel`) require a
  `SteeringRegistry` populated with that task's `SteeringHandle`; without one,
  those controls only affect terminal-state bookkeeping in the `TaskStore`,
  not the in-flight task itself.
- `TaskStore` is a synchronous trait, so `DriverTaskStore` bridges the async
  driver ports with `tinystoragedrivers_core::Blocking`; build it from a
  multi-thread runtime or outside one. History snapshots grow one per
  transition, so prune old records through the backend if that matters.
- `OrchestrationTaskFilter::matches` is evaluated in-process against loaded
  records; a `TaskStore` backend is not required to push the filter down, so
  `list` cost scales with total record count unless a given implementation
  optimizes it.
