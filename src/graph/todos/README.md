# graph::todos

A per-thread **task board** (kanban todos): an ordered *list* of task cards per
thread. Ported from OpenHuman's task board / `todos` modules and re-hosted on
the TinyAgents graph runtime — provider-neutral, offline-testable, with no
app-specific coupling (no progress events, RPC envelopes, or scratch fallback).

Distinct from [`graph::goals`](../goals) (a *single* durable objective per
thread): a goal is the completion contract the graph pursues; the board holds
the concrete work items.

## Data model (`types.rs`)

- `TaskCardStatus { Todo, AwaitingApproval, Ready, InProgress, Blocked, Done,
  Rejected }` and `TaskApprovalMode { Required, NotRequired }` (each `as_str`).
- `TaskBoardCard { id, title, status, objective, plan, assigned_agent,
  allowed_tools, approval_mode, acceptance_criteria, evidence, notes, blocker,
  session_thread_id, source_metadata, order, updated_at }` (serde `camelCase`).
- `TaskBoard { thread_id, cards, updated_at }`.
- `CardPatch` — optional `add`/`edit` fields; `approval_mode` is doubly-optional
  (`None` untouched, `Some(None)` clears, `Some(Some(_))` sets).
- `TodosSnapshot { thread_id, cards, markdown }` — every CRUD op returns one.
- `parse_status` (accepts aliases like `pending`→`Todo`, `approved`→`Ready`),
  `render_markdown` (`[ ]`/`[x]`/`[~]`/`[!]`/`[?]`/`[-]` markers + indented
  metadata), `normalise_board` (id generation, trimming, empty-title drop,
  blocker-from-notes, order recompute).

`id`s are `task-<n>` (via `next_seq()`); `updated_at` is unix-epoch millis as a
string (dependency-free, no `chrono`).

## Persistence (`store.rs`)

One serialized `TaskBoard` per thread under the `graph.todos` namespace of a
`crate::harness::store::Store`, keyed by `hex(thread_id)`. Every mutation runs
`load → mutate → normalise → put` under a per-thread async mutex (a weak-value
`graph::thread_locks::ThreadLockMap`, so idle threads' mutexes are reclaimed
instead of leaking) — atomic within one process (same single-process caveat as
`graph::goals::store`). Ops:
`add` / `edit` / `update_status` / `decide_plan` / `revise_plan` / `remove` /
`replace` / `clear` / `list` / `claim_card` / `set_session_thread`.

Invariants preserved from OpenHuman:

- **Single in-progress:** at most one card may be `InProgress`; a violation is a
  `Validation` error on `add` / `edit` / `replace` / `claim_card` — never
  silently fixed.
- `decide_plan` only transitions an `AwaitingApproval` card (approve → `Ready`,
  reject → `Rejected`); a stale decision errors.
- `revise_plan` rejects every `AwaitingApproval` card and is a lenient no-op
  when none is awaiting.
- `claim_card` is an atomic compare-and-set: it transitions a card from one of
  `expected` to `target` under the lock, rejecting the claim otherwise.

## Tool (`tool.rs`)

`TodoTool` is a single multiplexer harness `Tool` dispatching on an `op` field
(`add`/`edit`/`update_status`/`decide_plan`/`revise_plan`/`remove`/`replace`/
`clear`/`list`). Build it with `todo_tools(store)` or `register_todo_tools`. The
target thread comes from `ToolExecutionContext::thread_id` (never a tool
argument); the bare `Tool::call` entry point errors without a thread. Domain
errors (unknown id, invariant violation) are surfaced to the model as tool
errors rather than failing the run.

## Example

```rust,ignore
use std::sync::Arc;
use tinyagents::{TodoTool, todo_store};
use tinyagents::harness::store::{InMemoryStore, Store};

let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

// Programmatic:
let snap = todo_store::add(&store, "thread-1", "Write the RFC", Default::default()).await?;
println!("{}", snap.markdown);

// Or register the `todo` tool for a model to drive:
let tool = TodoTool::new(store.clone());
```

## Files

| File | Role |
| --- | --- |
| `types.rs` | Card/board model, `parse_status`, `render_markdown`, `normalise_board`, `CardPatch`, `TodosSnapshot`. |
| `store.rs` | `Store`-backed CRUD, per-thread RMW lock, single-in-progress invariant, CAS `claim_card`. |
| `tool.rs` | The `todo` multiplexer tool. |
| `test.rs` | Unit tests (types, store, tool). |
