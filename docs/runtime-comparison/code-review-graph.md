# tinyagents-graph / -orchestration / -session: durable-runtime code review

Reviewed at v2.1.2 (`fc33c43`), read-only. Paths are relative to the repository root.

`cargo clippy -p tinyagents-graph -p tinyagents-orchestration -p tinyagents-session --all-targets -- -W clippy::pedantic` is clean under default lints; 564 pedantic warnings for graph (mostly `must_use`, `missing_errors_doc`, casts), notably: `execute_run` 499 lines, `run_active_parallel` 121 lines, `state_api::update_state` 121 lines.

## 1. Architecture as-built

Entry points `CompiledGraph::{run, run_with_thread, run_with_inputs, resume, resume_from, retry}`
(`compiled/executor.rs:18-244`) all funnel into `execute` → `execute_run` (`executor.rs:505`).
One run = a loop over supersteps on an `active: Vec<Activation>` (`compiled/mod.rs:176`,
`Activation {node, send_arg, task_id}`):

1. Guards: `steps >= min(recursion_limit, policy.max_total_steps)` (`executor.rs:601-616`),
   wall-clock `run_deadline` (`:623`), `max_visits_per_node` (`:643`).
2. Assign task ids `"{steps}:{index}:{node}"` (`:661-665`), emit `StepStarted`.
3. Run handlers: sequential (`run_active_sequential`, `:1377`) or, with `with_parallel(true)`
   and >1 active, `run_active_parallel` (`:1483`; `join_all` or a `select_all` rolling window
   under `max_concurrency`). Every handler gets `state.clone()` and a fresh `NodeContext`
   (`:1297`, `:1247`). Per-node timeout + retry policy wrap each attempt (`:1265-1310`).
4. Fold results in index order (`fold_result`, `:1320`): updates → `Vec<Update>`, `Command.goto`
   → `goto_map[index]`, first `Interrupt`/error → stop folding (`break` at `:1444/:1458/:1617/:1631`).
5. Boundary: `state = reducer.apply(state, update)` for each update (`:726`).
6. Failure boundary (`:764-830`): route the completed prefix `active[..failed_index]`, then
   `pending = successors ++ active[failed_index..]`, persist a failure checkpoint, status Failed.
7. Interrupt boundary (`:838-926`): same shape with `active[..index]`, persist checkpoint with
   `interrupts` + `metadata.interrupted_nodes`, return `GraphExecution{interrupts}`.
8. Otherwise `route_completed(active)` (`compiled/routing.rs:15`): `goto` > static edge >
   conditional branch; waiting-edge barriers accumulate in `barrier_arrivals`; `Send` targets
   are not deduped, plain targets are. Persist per `DurabilityMode` (`:947-1038`), chain
   `parent_checkpoint`, `active = next`.

Checkpoint record: `Checkpoint<State>` (`checkpoint/types.rs:151`) = full `state` clone +
`next_nodes` + `completed_tasks` + inline `pending_writes` (completion markers only) +
`pending_activations` + `barrier_arrivals` + `interrupts` + free-form `metadata` JSON. The
executor calls `put` then `put_writes` (`executor.rs:1696-1697`), so the same markers are
stored twice. There is no format version, no timestamp, no channel versions, no delta scheme:
every superstep snapshots the whole state.

Resume (`resume_from_inner`, `executor.rs:246`): load the scoped checkpoint, take
`pending_activations` (fallback `next_nodes`), filter out tasks with a completion marker,
key the resume value on `metadata.interrupted_nodes` (fallback: fan across pending), then call
`execute` with `steps = 0` and `initial_parent = loaded id`.

State API (`compiled/state_api.rs`): `get_state`, `get_state_history`, `update_state`
(latest only), `bulk_update_state`, `fork_state` (new root, no parent).

Subgraphs (`subgraph/mod.rs`): a node handler that clones the child graph with namespace
`parent_ns ++ [node_id]` and calls `run_with_thread` / `resume` depending on `ctx.resume`
(`drive_child`, `:135-164`). A child interrupt is re-emitted as the parent node's result.

Other crates: `tinyagents-orchestration/workflow/engine.rs` is a *second* scheduler: phases
are a JSON `phase_states` blob persisted through the session run ledger (CAS + lease), with
`map_reduce` for fan-out; it does not use `CompiledGraph` at all (the graph in
`workflow/graph.rs` is a topology preview only). `teams/graph.rs` wraps one worker future in
an unthreaded, uncheckpointed 2-node graph. `tinyagents-session` is a SQLite ledger
(one `Connection::open` per operation, `store.rs:65`, WAL + busy_timeout).

Divergence from docs (see also findings): `docs/modules/graph/execution.md` lists a target
executor with channel versions, cached-write replay, task-level `TaskStarted/TaskCached`
events; none exist. `checkpointing.md:56-77` lists checkpoint fields (version, timestamp,
channel versions, versions seen, task outcomes) that the record does not have.
`interrupts.md` shows `interrupt_before/after`, resume-by-interrupt-id maps, and
`resume_targeted`; none are implemented (`mark_interrupt` is export-only,
`builder/mod.rs:365`). `parallel-agents-forking.md:156-166` promises "resuming from
interrupt restarts the interrupted child task, not unrelated completed siblings", which is
false for higher-index siblings (Critical 1).

## 2. Findings

### Critical

**C1. Completed parallel siblings after the interrupting/failing index are discarded and
re-executed on resume.**
`executor.rs:1600-1633` (parallel fold):
```rust
for (index, (activation, result)) in active.iter().zip(results).enumerate() {
    ...
    Err(error) => { ... failure = Some(StepFailure{failed_index: index, error}); break; }
    ...
    if let Some(found) = self.fold_result(...) { interrupt = Some(found); break; }
```
and `executor.rs:869` / `:790`: `pending.extend(active[index..].iter().cloned());`.
`join_all` has already driven every branch to completion, so branches `index+1..` finished
(LLM calls, tool side effects, sub-agent runs), yet their `NodeResult`s are dropped and they
are scheduled to run again. The docs' own contract ("failed sibling tasks do not force
successful child agents to rerun once pending writes are saved") is not met; the pending-write
ledger only records the *lower-index* prefix. Fix: fold every `Ok` result regardless of
position; record all completed tasks in `completed_tasks`/`pending_writes`; make `pending`
= successors of *all* completed + only the interrupted/failed activations. The test
`parallel_interrupt_pauses_at_lowest_index_branch` (`compiled/test.rs:1002`) pins the current
behaviour and would need updating. Effort: M.

**C2. Successors of completed siblings run in the *same* superstep as the re-run of the
interrupted/failed node, observing a different state than an uninterrupted run would.**
`executor.rs:856-869`: `pending = route_completed(active[..index]) ++ active[index..]`.
Uninterrupted: step N = [A, B]; step N+1 = [S_A, S_B] and S_A sees B's update.
Interrupted at B: resume step 1 = [S_A, B]; S_A now reads a state *without* B's update, and
B's update lands one step later. Any node that aggregates over sibling output through a plain
edge (not a waiting edge) silently produces different results depending on whether an
interrupt happened. LangGraph avoids this by re-running only the incomplete tasks of the
*same* step and applying all writes together. Fix: persist the completed tasks' updates as
real pending writes (requires `Update: Serialize` for durable backends, or store the
already-reduced partial state plus "tasks still owed for step N"), and on resume finish step N
before routing anyone. Effort: L (ties to R2 below).

**C3. No per-thread execution lock: two concurrent `run_with_thread`/`resume` on the same
thread interleave one lineage.**
`executor.rs:246-390` takes no lock; `delegation/run.rs:139-160` and `:200-212` each add their
own `ThreadLockMap` and say why: "neither this wrapper nor the `CompiledGraph` run/resume
entry points hold a per-thread lock across that gap on their own". Backends allow duplicate
`(thread_id, checkpoint_id)` rows (`sqlite.rs:135-147` has no UNIQUE constraint) and `get(None)`
returns the last inserted row, so two writers produce a thread whose "latest" flips between
two histories and whose `parent_checkpoint_id` chains cross. Fix: an in-process
`ThreadLockMap` guard in `execute` keyed by `(thread_id, namespace)` plus an optional durable
lease in `Checkpointer` (`try_claim(thread, owner, ttl)`) like `run_ledger::try_claim_workflow_run`
(`session/run_ledger/ops.rs:169`). Effort: S (in-process) / M (durable).

**C4. Subgraph failures are not resumable through the parent; `retry` restarts the child from
scratch and re-runs its completed nodes.**
`subgraph/mod.rs:161`: `(Some(thread_id), None, None) => child.run_with_thread(thread_id, state)`.
After a child node fails, the child writes a failure checkpoint in its namespace and the parent
writes one scheduling the subgraph node. `parent.retry()` re-runs the node with `resume = None`
→ fresh `run_with_thread` with `initial_parent = None`: the child's partial progress is ignored,
its completed nodes (with side effects) re-execute, and a second root-less lineage is appended
to the same namespace. Fix: in `drive_child`, before running fresh, check
`get_scoped(thread, None, child_ns)` for a checkpoint with non-empty pending activations
whose `metadata.failed_node` is set (or any pending), and `retry` it; also record the child's
checkpoint id in the parent's activation so the association is explicit. Effort: M.

### Important

**I1. `Send` fan-out of the same subgraph node shares one checkpoint namespace.**
`subgraph/mod.rs:109-113`: namespace = parent ++ `[ctx.node_id]` only. N concurrent
activations of node `worker` (map-reduce over a subgraph) write N interleaved lineages under
`["worker"]`; on parent resume each activation calls `child.resume(thread)` → `get_scoped(...,
None)` → all N resume from whichever child wrote last. Fix: namespace by
`[node_id, task_id]` (task id is already on the `Activation`, `compiled/mod.rs:179`) and expose
`NodeContext.task_id`. Effort: S. (Same identity gap: `resume_map` is keyed by `NodeId`,
`executor.rs:361-374`, `:1247`, so only the first same-node activation receives the value.)

**I2. `update_state` erases interrupt provenance, so a later `resume(value)` fans the value to
every pending node.** `state_api.rs:228-231` writes `interrupts: Vec::new()` and metadata
`{source, step}` without `interrupted_nodes`; `compiled/mod.rs:229-251` then finds nothing
stamped and nothing in `interrupts`, and `executor.rs:363-366` falls back to fanning across
`active`. The documented flow "inspect → `update_state` → `resume`" therefore hands
`ctx.resume` to nodes that never interrupted. Fix: carry `interrupts` and `interrupted_nodes`
through an `update` checkpoint unless `as_node` names the interrupted node. Effort: S.

**I3. Step counter, node-visit counts and recursion caps reset on every resume.**
`executor.rs:518` `let mut steps = 0usize;` and `:513-520` fresh `node_visits`. A loop that
interrupts (or fails and is auto-retried by a host) every step is unbounded; checkpoint
`metadata.step` restarts at 1 after each resume, so `get_state_history` shows non-monotonic
steps and `update_state`'s `parent_step + 1` (`state_api.rs:231`) is meaningless across
resumes. Fix: seed `steps` from the loaded checkpoint's `to_metadata().step` and persist
`node_visits` in metadata. Effort: S.

**I4. Node panics and mid-superstep cancellation leave the run `Running` forever with no
terminal event.** No `catch_unwind` anywhere in `compiled/` (grep). A panic in a handler
unwinds through `join_all`/`fut.await` (`:1297`); dropping the run future (host timeout,
`tokio::select!`) skips `fail_run`, `save_status`, `RunFailed`, and drops
`AsyncCheckpointWrites` (detaching in-flight writes, contrary to its own contract at
`types.rs:141-149`). The executor also has no `CancellationToken` input at all (`map_reduce`
and `SubAgentPolicy` do). Fix: wrap handler futures in `AssertUnwindSafe(..).catch_unwind()`
mapped to `TinyAgentsError::Graph`; add `run_with_cancel(token)` checked at step boundaries and
raced against the handler set; add a `Drop` guard on `execute_run` that writes a `Cancelled`
status. Effort: M.

**I5. The channel state model is not durable.** `channel/types.rs:240` `ChannelState` (and
`ChannelSet`, holding `Box<dyn Channel>`) has no `Serialize`/`Deserialize`, so the only state
model with per-key reducers, conflict detection and barrier channels cannot be used with
`FileCheckpointer` or `SqliteCheckpointer` (both bound `State: Serialize + DeserializeOwned`,
`file.rs:372`, `sqlite.rs:203`). `state-channels.md` presents channels as the way to get
"checkpoints can store pending writes". Fix: serialize `ChannelSet` as `{kind, config}` per
channel + values (each `Channel` already has `kind()`), with a registry for
`BinaryAggregate` closures. Effort: M.

**I6. Checkpoint format has no version and four overlapping projections of "what runs next".**
`checkpoint/types.rs:151-201`: `next_nodes`, `completed_tasks`, `pending_writes`,
`pending_activations` all derive from the same activation list and can disagree when written
by hand/`update_state` (`state_api.rs` has a long comment defending the merge for that reason).
No `format_version` field means a future incompatible change can only be detected by serde
`Category::Data` heuristics (`checkpoint/mod.rs:70-99` documents this as an accepted risk).
`delegation` works around it with its own `schema_version` in the state (`delegation/run.rs`).
Fix: v2 record with `version: u32`, `tasks: Vec<PendingActivation>` as the single source, and
`completed: Vec<TaskId>`; keep a v1 → v2 decoder. Effort: M.

**I7. `Interrupt::new` ids are process-local counters, unlike checkpoint/run ids.**
`command/mod.rs:110-112`: `INTERRUPT_SEQ.fetch_add` + `format!("interrupt-{node}-{seq}")`.
After a restart the same `(node, seq)` is re-minted; `GraphRunStatus.pending_interrupts`
(`status/types.rs`) and any UI keyed on interrupt id conflate two pauses. `ids::new_checkpoint_id`
already solved this with a process nonce (`harness/src/ids/mod.rs:135`). Fix: use it. Effort: S.

**I8. SQLite backend: synchronous I/O on the async executor path, no WAL, no busy timeout,
and full-state loads for `state_history(limit)`.** Only `put` uses `spawn_blocking`
(`sqlite.rs:207`); `get`, `get_scoped`, `list`, `state_history`, `get_thread`, `put_writes`,
`delete_*` lock a `std::sync::Mutex<Connection>` and run queries inline in `async fn`
(`:268`, `:314`, `:359`, `:436`, `:472`, `:491`, `:619`). `from_connection` never sets
`journal_mode=WAL`/`busy_timeout`/`synchronous` (the session crate does, `session/store.rs:49-55`).
`state_history` reads and deserializes *every* record of the namespace even for `limit: Some(1)`
(`:353-374`). `put` and `put_writes` are two transactions. Fix: `spawn_blocking` everywhere,
pragmas on open, `LIMIT`-driven lineage query (walk parents with a recursive CTE), one
transaction per boundary. Effort: M.

**I9. File backend: `list`/`get_scoped`/resume deserialize every full record; `put_writes`
rewrites and fsyncs the whole sidecar every superstep.** `file.rs:451`
`list` → `read_records` (`:247`, full `Checkpoint<State>` decode); `FileCheckpointer` does not
override `get_scoped` (no match in `file.rs`), so `resume`/`update_state`/`fork_state` cost
O(H) full-state decodes plus a second streaming pass in `get`. The trait doc at
`checkpoint/mod.rs:129` claims "both durable backends do [override]" — false. `put_writes`
(`:621-653`) is read-modify-rewrite with `write_atomic` (fsync), synchronously on the executor
task. Fix: header-only decode for `list` (as `get` already does), override `get_scoped`,
append-only writes sidecar keyed by checkpoint id, `spawn_blocking`. Effort: M.

**I10. `GraphBuilder::add_edge` silently overwrites a previous edge from the same node.**
`builder/mod.rs:195-197`: `self.edges.insert(from.into(), to.into())` (also `add_waiting_edge`,
`:226`). `edges: HashMap<NodeId, NodeId>` means static fan-out is impossible and
`add_edge("a","b").add_edge("a","c")` compiles to `a → c` with no error; `routing.md:49-52`
lists "one or more node names" as a routing output. Fix: `HashMap<NodeId, Vec<NodeId>>`, or at
least return `Validation` on a second insert. Effort: S/M.

**I11. `#![cfg_attr(not(feature = "tracing"), allow(dead_code, unused_imports,
unused_variables))]` at crate root hides dead code in the default build.**
`tinyagents-graph/src/lib.rs:25-28`, `tinyagents-session/src/lib.rs:1-4`. The default feature
set is `tracing` off, so every unused item in both crates is invisible to `cargo clippy -D
warnings`. Evidence of accumulation: `StreamMode` (`stream/types.rs:220`) is exported and
consumed nowhere; `WRITES_IDX_RESUME/ERROR/INTERRUPT` (`checkpoint/types.rs:272-278`) are only
used by the conformance suite — the executor never persists a resume value, so a resumed node
that crashes loses it and `retry` re-interrupts; `CompiledGraph.command_nodes` and
`BuilderNode.id` carry explicit `#[allow(dead_code)]`. Fix: make the tracing macros expand to
`{ let _ = (...); }` in the off configuration and delete the crate-level allow. Effort: S.

**I12. Two durable schedulers with different semantics.** `orchestration/workflow/engine.rs`
implements phase DAG scheduling, lease/CAS persistence, cancellation and resume in ~830 lines
of JSON manipulation (`state.rs` string-typed statuses `"running"`, `"completed"`), emitting
`GraphEvent::NodeStarted{node: "run_phase", step: total_spawned+1}` (`engine.rs:428`) to look
like a graph. `workflow/graph.rs:37` says "`WorkflowEngine` installs its effectful variants
below" but the engine never runs that graph. The graph runtime already has waiting edges,
`Send` fan-out, `max_concurrency`, checkpoints and resume; the workflow engine re-implements
them without any of the executor's tests. Fix: lower a `WorkflowDefinition` to a
`CompiledGraph` (phase = node, `depends_on` = waiting edges, agents = `Send` fan-out) and keep
the run ledger as a status projection. Effort: L.

### Minor

**M1. `execute_run` is 499 lines with 12 `#[allow(clippy::too_many_arguments)]`** across
`compiled/` (`executor.rs`, `mod.rs`). Thirteen positional args of `Option<&..>`/`&HashMap`
(`persist_checkpoint`, `:1652-1666`) are error-prone. Introduce a `RunCtx` struct (run id,
thread, started_at, recursion meta, barriers, async writes) and a `Boundary` struct. Effort: M.

**M2. Hot-path cloning.** `executor.rs:1297` clones `State` per handler *per attempt*;
`:1188`/`:1855` clone it again per checkpoint; `serde_json::to_string(&checkpoint)` per
boundary; `Send` args cloned into `Activation`, `PendingActivation`, `NodeContext`. For
`serde_json::Value` states with message histories this is O(history) per node per step. Fix:
`NodeHandler = Fn(Arc<State>, NodeContext)` (one clone per step), `Arc<Value>` send args.
Effort: M (API break).

**M3. `run_deadline` uses `SystemTime::elapsed`** (`executor.rs:624`); wall-clock jumps make
runs fail or never time out. Use `Instant`. Effort: S.

**M4. Status-store errors are swallowed** (`executor.rs:500` `let _ = store.put_status`), so a
dead status backend is invisible. Log at warn at minimum. Effort: S.

**M5. Async durability skips `put_writes`.** `executor.rs:1764` background task only calls
`put`; sync path calls both. Ledger tooling sees no markers for async-mode runs. Effort: S.

**M6. `GraphEvent` carries no run id/task id/namespace on step/node events**
(`stream/types.rs:46-104`); the module doc (`stream/mod.rs:8-10`) claims nested streams can be
"attributed back up the run tree", but only the journal wrapper adds that. Add a `RunRef`
header or emit `(run_id, event)`. Effort: S/M.

**M7. `GraphEventJournal` is lossy under load.** `observability/mod.rs:464` uses the harness
`AppendWorker`, which `try_send`s and increments `dropped` when the bounded queue is full
(`harness/src/observability/worker.rs:250-253`). A "durable ... replay" journal that drops
events should at least expose `dropped()` on `JournalGraphSink` and document it. Effort: S.

**M8. Router closures return `String`, `Route` is unused by the executor** (`builder/types.rs`
`RouterFn = dyn Fn(&State) -> String`; `Route` newtype defined but branches key on `String`).
A typo in a route label is a runtime `MissingRoute`. Effort: S.

**M9. `language::build_graph` drops declared conditional routes**: `language.rs:41`
`Routing::Conditional(_) => builder.mark_command_routing(...)` ignores the route table, so a
blueprint's routing is not validated against what the node's `Command` may return. Effort: S.

**M10. Session ledger opens a new SQLite connection per operation** (`session/store.rs:65`),
including the lease heartbeat every `lease/3` (`engine.rs:585`), and all ledger calls are sync
inside `async fn drive`. A small connection cache or `spawn_blocking` would help. Effort: S.

**M11. `PhaseRegistration::register` performs a blocking DB CAS under a `parking_lot::Mutex`
inside an async executor callback** (`engine.rs:217-224`). Effort: S.

**M12. Subgraph doc lists unimplemented requirements as if present** (`subgraphs.md:15-25`:
isolated child thread ids, checkpointer override, `Command::Parent`, child state in parent
task metadata). Mark them as target or remove. Effort: S.

## 3. Structural refactors worth doing

**R1. Split `compiled/executor.rs` along boundaries, not along "public vs private".**
`RunCtx` (identity, clocks, recursion meta, async writes), `StepRunner` (sequential/parallel,
returning a `StepOutcome` with *all* results), `Boundary` (apply reducer, route, persist), and
`Resume` (load + filter + resume map). This is the precondition for C1/C2 because the "which
results do we keep" decision is currently spread over four `break`s and two `extend`s. Risk:
low, pure code motion with the existing 100 unit tests as a net.

**R2. Real pending-writes: make the executor persist task outputs, not just markers.**
Add `Update: Serialize + DeserializeOwned` bounds on the durable checkpointers only (the
in-memory path stays bound-free), store `PendingWrite.payload = serde_json::to_value(update)`,
and on resume replay stored writes for completed tasks of the interrupted step instead of
re-running them (this is what `execution.md`'s "cached writes replay without rerunning nodes"
describes). Fixes C1 and C2 and gives `WRITES_IDX_RESUME` a purpose (persist the resume value
so `retry` after a crash still has it). Migration risk: medium; the `Update` bound is a public
API change for durable users, hidden behind a `DurableUpdate` marker trait with a blanket impl.

**R3. Checkpoint format v2 + versioned channels.** Add `version`, `created_at`, `task_id`
on `Interrupt`, one `tasks` list, and an optional `channel_versions: BTreeMap<String, u64>`
written by `ChannelState` (I5). Versioned channels are what make subgraph output merging
correct (`shared_subgraph_node` returns the child's *whole* final state as the parent update,
so an append reducer double-applies inherited messages) and what enables delta snapshots
(`prune` already documents "delta-channel" semantics that nothing implements,
`checkpoint/mod.rs:462-476`). Risk: medium; needs a decode shim for v1 rows and a
`FileCheckpointer`/`SqliteCheckpointer` migration test.

**R4. Per-thread run lease in the executor (C3)** with an in-process `ThreadLockMap` (already
exists at `thread_locks/mod.rs`) and an optional `Checkpointer::try_claim/renew/release`
defaulting to no-op; `delegation/run.rs` can then drop its private lock map. Risk: low.

**R5. Typed task identity.** `TaskId` exists in `harness::ids` (used by `RecursionFrame`),
but the executor uses `String` task ids, `Interrupt` has none, and `NodeContext` does not
expose it. Threading `TaskId` through `Activation`, `PendingActivation`, `PendingWrite`,
`Interrupt`, `NodeContext`, and the subgraph namespace (I1) closes several identity bugs at
once. Risk: low-medium (public struct fields).

**R6. Retire the workflow engine's private scheduler (I12)** in favour of a
`WorkflowDefinition → CompiledGraph` lowering; keep `WorkflowStore` as the status projection
and the lease as R4's durable lock. Risk: high (host-facing `phase_states` JSON contract);
do it behind a feature flag with the existing `workflow/tests.rs` as the acceptance suite.

## 4. Test-quality assessment

Strengths: `compiled/test.rs` (100 tests) covers routing, `Send` args across resume, barrier
arrivals across resume, async-durability ordering/failure/drain, attributed `update_state`
merges, retry/failure checkpoints, and legacy-checkpoint fallbacks. `testkit/conformance.rs`
has checkpointer contract suites (writes, lineage, concurrency) run against all three
backends. `delegation/test.rs` (1.3k lines) exercises schema-version guards and lock
serialization. Tests are descriptive and mostly assert observable durable state, not
internals.

Gaps (each maps to a finding):
- No test that a *higher-index* completed parallel branch is not re-run after an interrupt
  or failure, nor that its update survives (C1). `parallel_interrupt_pauses_at_lowest_index_branch`
  asserts the lossy behaviour.
- No test comparing final state of an interrupted-then-resumed run against the same graph
  run uninterrupted (C2) — the single most valuable durability property test.
- No concurrent-same-thread test for the executor (C3); only delegation tests it indirectly.
- No subgraph failure → parent `retry` test (C4); `subgraph/test.rs` covers interrupt resume
  only. No `Send` fan-out of a checkpointed subgraph node (I1).
- No `update_state` (as_node None) followed by `resume(value)` test asserting the value
  reaches only the interrupted node (I2).
- No test for step monotonicity across resume or cumulative recursion limits (I3).
- No node-panic test and no cancellation (dropped future) test asserting status/checkpoint
  consistency (I4). `orchestration/test.rs:254` is the only `catch_unwind` in the crate.
- No channel-graph durability test (I5) — impossible today, which is the point.
- No test exercising a `FileCheckpointer`/`SqliteCheckpointer` behind the *executor* for
  interrupt → resume across a fresh `Connection`/process nonce (the "restart" path); the
  durable-backend tests are conformance-only and executor tests use `InMemoryCheckpointer`.
- No performance guard on `state_history(limit)` or on `list` decode cost (I8/I9).
- `workflow/tests.rs` (850 lines) is thorough on lease/CAS races but relies on string statuses;
  a typo in `"completed"` would pass type-check and fail only at runtime.

## 5. Things that are genuinely good

- Deterministic fold order and explicit `goto_map` keyed by index (repeated `Send` of one node
  keeps its own routing) — a real correctness win over node-keyed maps.
- `AsyncCheckpointWrites` chaining with first-error propagation and drain-on-every-exit is
  carefully reasoned and tested (`compiled/types.rs:153-249`).
- Barrier-relief with `reaches_deterministically` (`routing.rs:68-158`) solves a real
  deadlock without weakening the barrier, and the comment explains the multi-hop pitfall.
- Failure boundaries are resumable and `retry`/`update_state` compose with them; the
  `interrupted_nodes` metadata keeps resume values off nodes that never paused.
- Checkpoint id minting uses a process nonce; `state_history` has a cycle guard;
  `copy_thread` refuses to interleave lineages; torn trailing JSONL lines are tolerated.
- `ThreadLockMap` weak-value map is a neat leak-free per-key mutex.
- Schema-version fencing and per-thread locking in `delegation/run.rs` show the team knows
  the executor's gaps; the fixes above mostly move that discipline into the runtime.
- `map_reduce` gets ordering, fail-fast-in-input-order, cancellation and timeouts right.
- The session ledger's lease + revision CAS (`run_ledger/ops.rs:169-320`) is a solid pattern
  the graph checkpointer should borrow.
