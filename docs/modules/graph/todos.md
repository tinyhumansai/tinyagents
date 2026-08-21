# Per-thread task board (kanban todos)

`graph::todos` gives a graph a per-thread **task board**: an ordered *list* of
task cards with a small kanban lifecycle. It is the concrete-work-items
counterpart to the single-objective [`graph::goals`](goals.md), a
provider-neutral port of OpenHuman's task board / `todos` modules.

See the source module README at `src/graph/todos/README.md` for the full public
surface; this spec captures the design contract.

## Model

- `TaskBoardCard { id, title, status, objective, plan, assigned_agent,
  allowed_tools, approval_mode, acceptance_criteria, evidence, notes, blocker,
  session_thread_id, source_metadata, order, updated_at }`.
- `TaskCardStatus`: `Todo`, `AwaitingApproval`, `Ready`, `InProgress`,
  `Blocked`, `Done`, `Rejected`. `TaskApprovalMode`: `Required`, `NotRequired`.
- `TaskBoard { thread_id, cards, updated_at }`; `TodosSnapshot` (cards +
  markdown) is returned by every CRUD op.
- `render_markdown` renders GitHub-flavored markers
  (`[ ]`/`[x]`/`[~]`/`[!]`/`[?]`/`[-]`) with indented metadata; `parse_status`
  accepts aliases; `normalise_board` generates ids, trims, drops empty-title
  cards, backfills a blocker from notes, and recomputes order.

## Persistence

One serialized `TaskBoard` per thread in the `graph.todos` namespace of a
`harness::store::Store`, keyed by `hex(thread_id)`. Normal CRUD mutations run
`load → mutate → normalise → put` under a per-thread async mutex (atomic within
one process, same caveat as `graph::goals`).

Host integrations can use three raw lifecycle operations: `get` reads without
normalising and preserves absent versus present-empty; `delete` removes the
value outright under the thread lock; and `import_if_absent` atomically writes
a complete board only when no value exists, leaving existing or undecodable
values untouched for safe one-time migrations.

### Invariants

- **Single in-progress:** at most one card may be `InProgress`; a violation is a
  `Validation` error on `add` / `edit` / `replace` / `claim_card` — never
  silently fixed.
- `decide_plan` only transitions an `AwaitingApproval` card; a stale decision
  errors. `revise_plan` rejects all awaiting cards and is a lenient no-op when
  none awaits.
- `claim_card` is an atomic compare-and-set: transition from one of `expected`
  to `target` under the lock, else reject.

## Tool

`TodoTool` is a single multiplexer harness `Tool` dispatching on an `op` field
(`add`/`edit`/`update_status`/`decide_plan`/`revise_plan`/`remove`/`replace`/
`clear`/`list`), built with `todo_tools` / `register_todo_tools`. The board is
bound to `ToolExecutionContext::thread_id` (never a tool argument). Domain errors
(unknown id, invariant violation) are surfaced to the model as tool errors
rather than failing the run.

## Runs and dispatch

Two layers sit on top of the board for hosts that run cards autonomously.

`graph::todos::runs` is the claim/heartbeat/reclaim log (`task_run_store`). A
worker claims a card, opens a `TaskRun`, ticks a heartbeat while it works, and
closes the run with a `RunOutcome`. `reclaim_stale` sweeps runs whose heartbeat
or claim aged out under `RunLimits`, closes them `Reclaimed`, and returns their
card to `Todo` — or parks it at `Blocked` once the card has exceeded
`max_reclaim_count`, so a card that keeps killing workers stops cycling. The
staleness policy itself is the pure, clock-injected `staleness_reason`. See
[`src/graph/todos/runs/README.md`](../../../src/graph/todos/runs/README.md).

`graph::todos::dispatch` is the scheduling policy: `pick_next_card` (urgency,
then board order, optionally agent-assigned only), `requires_plan_approval`
(the card's own mode outranks the global gate), `PollCadence` (idle backoff),
`build_task_prompt` / `build_progress_instruction`, and `ActiveRunRegistry`
(in-flight runs with race-free removal, so a terminal write-back happens once).
See [`src/graph/todos/dispatch/README.md`](../../../src/graph/todos/dispatch/README.md).

Executing a card is out of scope for the crate — that needs a host's agent and
tool belt. `tests/e2e_graph_task_dispatch.rs` is the reference assembly.

## Testing

Unit tests in `src/graph/todos/test.rs` (types, store invariants, tool),
`src/graph/todos/runs/test.rs`, and `src/graph/todos/dispatch/test.rs`; an
end-to-end model-driven tool run in `tests/e2e_graph_todos.rs`; feature coverage
for the run lifecycle in `tests/feature_graph_task_runs.rs`; and the full
dispatch loop in `tests/e2e_graph_task_dispatch.rs`.
