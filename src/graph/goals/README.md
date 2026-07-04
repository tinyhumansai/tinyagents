# graph::goals

A per-thread **goal**: exactly one durable objective per thread, carried across
supersteps, interrupts, and resumes. Ported from OpenHuman's `thread_goals` and
re-hosted on the TinyAgents graph runtime — provider-neutral, offline-testable,
with no app-specific coupling (no event bus, RPC envelopes, or heartbeat).

Distinct from [`graph::todos`](../todos) (a *list* of task cards per thread): a
goal is the single "completion contract" the graph pursues; the board holds the
concrete work items.

## Data model (`types.rs`)

- `ThreadGoalStatus { Active, Paused, BudgetLimited, Complete }` — `is_active` /
  `is_terminal` predicates. Ownership is asymmetric: a model creates/replaces a
  goal and marks it `Complete`; `Paused` / `BudgetLimited` are system-driven.
- `ThreadGoal { thread_id, goal_id, objective, status, token_budget, tokens_used,
  time_used_seconds, created_at_ms, updated_at_ms, continuation_suppressed }`
  (serde `camelCase`) with `budget_remaining` / `over_budget`.
- `GoalProgress { tokens_used, elapsed_secs, made_progress }` — usage a work
  iteration reports to the continuation gate. `TurnOutcome` is an alias used at
  the driver boundary.
- `active_goal_context_block(&ThreadGoal) -> Option<String>` — a prompt block to
  prepend to a work node's input (steer text for Active / stop text for
  BudgetLimited; `None` for Paused/Complete).

## Persistence (`store.rs`)

One serialized `ThreadGoal` per thread under the `graph.goals` namespace of a
`crate::harness::store::Store`, keyed by `hex(thread_id)` (hex keeps arbitrary
thread ids valid across `InMemoryStore` / `FileStore` / Sqlite). CRUD:
`get` / `set` / `set_if_absent` / `complete` / `pause` / `resume` / `clear` /
`list_all` / `account_usage` / `set_continuation_suppressed_if`.

Semantics preserved from OpenHuman:

- `set` mints a fresh `goal_id` and resets counters when the objective changes;
  a same-objective re-set preserves counters and re-opens to `Active` unless
  still over budget.
- `account_usage` folds token/time usage and flips an active goal to
  `BudgetLimited` at the cap. Its `expected_goal_id` **compare-and-set** guard
  silently drops stale accounting from a replaced goal.
- `set_continuation_suppressed_if` writes only when the current goal still
  matches `expected_goal_id` and is active.

**Concurrency / single-process caveat.** The `Store` trait offers no CAS and no
cross-key transaction, so each mutation runs `load → mutate → put` under a
per-thread async mutex (a weak-value `graph::thread_locks::ThreadLockMap`, so
idle threads' mutexes are reclaimed instead of leaking) — atomic **within one
process**. Across processes sharing
a `FileStore`, two concurrent read-modify-writes can lose an update; the
`goal_id` guard still prevents logical corruption from stale accounting but not
lost updates. Funnel goal mutations through one process, or wait for a future
`Store::compare_and_swap`.

## Tools (`tool.rs`)

`GoalTool` dispatches on `GoalToolKind`. The default model-facing set
(`goal_tools` / `register_goal_tools`) is `goal_get`, `goal_set`,
`goal_complete`; `goal_pause` / `goal_resume` / `goal_clear` are constructible
host controls. The target thread comes from `ToolExecutionContext::thread_id`
(never a tool argument), so a model can't address another thread's goal; the
bare `Tool::call` entry point (no context) errors.

## Continuation (`continuation.rs`)

OpenHuman's idle heartbeat becomes graph-native, three ways:

- **`goal_gate_node`** (primary) — a command-routing node forming a self-driving
  bounded loop. Wire `work_node -> gate` and register `gate` with
  `with_command_destinations([work_node, END])`. Each pass reads the thread id
  from `NodeContext`, folds the iteration's `GoalProgress` via `account_usage`,
  and routes back to `work_node` while the goal is Active and under budget, else
  to `END`. `recursion_limit` is the hard backstop; a zero-progress iteration
  sets the one-shot suppression and stops.
- **`run_continuation_tick`** (driver) — for callers with an external scheduler:
  selects idle, active, non-suppressed goals (oldest-first, `max_per_tick`) and
  runs one turn each through a `run_turn` closure, then accounts + one-shot
  suppresses on no progress.
- **`note_user_turn`** — call at the start of a user-initiated run to clear the
  one-shot suppression and reactivate a paused goal. A loop iteration never
  clears its own suppression.

**Token accounting boundary.** The graph runtime does not meter tokens per node,
so accounting is explicit: a work node writes what it spent into `State` and the
`progress` / `run_turn` closure reports it. `made_progress == false` is the
graph analogue of OpenHuman's "the turn produced no tool calls".

## Example: a self-driving goal loop

```rust,ignore
use std::sync::Arc;
use tinyagents::{GraphBuilder, END, GoalProgress, goal_gate_node, goal_store};
use tinyagents::harness::store::{InMemoryStore, Store};

let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());
goal_store::set(&store, "thread-1", "summarise the repo", Some(50_000)).await?;

let gate = goal_gate_node::<St, St>(store.clone(), "work", |s: &St| GoalProgress {
    tokens_used: s.last_turn_tokens,
    elapsed_secs: 0,
    made_progress: s.made_progress,
});

let graph = GraphBuilder::<St, St>::overwrite()
    .with_recursion_limit(64)
    .add_node("work", work_node)          // a subagent_node that calls goal_complete when done
    .add_node("gate", gate)
    .set_entry("work")
    .add_edge("work", "gate")
    .with_command_destinations("gate", ["work", END])
    .compile()?;

let exec = graph.run_with_thread("thread-1", St::default()).await?;
```

## Files

| File | Role |
| --- | --- |
| `types.rs` | Data model: `ThreadGoal`, `ThreadGoalStatus`, `GoalProgress`. |
| `prompt.rs` | `active_goal_context_block` — renders the per-iteration prompt context block. |
| `store.rs` | `Store`-backed CRUD, per-thread RMW lock, budget + CAS guards. |
| `tool.rs` | `GoalTool` / `GoalToolKind` harness tools. |
| `continuation.rs` | `goal_gate_node`, `run_continuation_tick`, `note_user_turn`. |
| `test.rs` | Unit tests (types, store, tools, continuation loop). |
