# Per-thread goal and graph-native continuation

`graph::goals` gives a graph a single durable **objective per thread** — a
"completion contract" carried across supersteps, interrupts, and resumes — plus
a graph-native way to keep working it. It is a provider-neutral port of
OpenHuman's `thread_goals`, re-hosted on the harness `Store` and driven off the
graph runtime rather than an out-of-band heartbeat.

See the source module README at `src/graph/goals/README.md` for the full public
surface; this spec captures the design contract.

## Model

- Exactly **one** `ThreadGoal` per thread, keyed by `thread_id`.
- `ThreadGoalStatus`: `Active` → the graph may work it and auto-continue;
  `Paused` (host control); `BudgetLimited` (accounting reached the token cap);
  `Complete` (model-confirmed success). Ownership is asymmetric: a model
  creates/replaces and completes a goal; pause/budget-limit are system-driven.
- Optional `token_budget`; `account_usage` folds usage and flips an active goal
  to `BudgetLimited` at the cap.

## Persistence

One serialized `ThreadGoal` per thread in the `graph.goals` namespace of a
`harness::store::Store`, keyed by `hex(thread_id)`. The `Store` trait has no CAS,
so each mutation runs `load → mutate → put` under a per-thread async mutex —
atomic within one process. A `goal_id` compare-and-set guard drops stale
accounting from a replaced goal. Cross-process lost-update is a documented
limitation (a future `Store::compare_and_swap` is the clean fix).

## Tools

`GoalTool` / `GoalToolKind` expose `goal_get`, `goal_set`, `goal_complete` as
the default model-facing set (`goal_tools` / `register_goal_tools`);
`goal_pause` / `goal_resume` / `goal_clear` are host controls. The target thread
comes from `ToolExecutionContext::thread_id` — never a tool argument — so a
model can't address another thread's goal.

## Continuation (heartbeat → graph)

OpenHuman's idle heartbeat becomes three graph-native primitives:

1. **`goal_gate_node`** (primary) — a command-routing node forming a self-driving
   bounded loop. Wired `work_node -> gate` with the gate a command node whose
   destinations are `[work_node, END]`, it folds each iteration's `GoalProgress`
   via `account_usage` and routes back to `work_node` while the goal is Active
   and under budget, else to `END`. The graph `recursion_limit` is the hard
   backstop; a zero-progress iteration sets a one-shot suppression and stops.
2. **`run_continuation_tick`** — a faithful heartbeat port for callers that have
   an external scheduler: selects idle, active, non-suppressed goals (oldest
   first, `max_per_tick`) and runs one turn each through a caller closure.
3. **`note_user_turn`** — clears the one-shot suppression and reactivates a
   paused goal on a user-initiated run. A loop iteration never clears its own
   suppression, so user-vs-continuation is distinguished structurally.

### Token accounting boundary

The graph runtime is provider-neutral and does not meter tokens per node, so
accounting is **explicit**: a work node (typically a `subagent_node`) writes what
it spent into `State`, and the caller's `progress` / `run_turn` closure reports
it. `made_progress == false` is the graph analogue of OpenHuman's "the turn
produced no tool calls".

## Budget enforcement

`store::account_usage` is the raw write; `graph::goals::budget` is the policy
around it — the two halves of enforcement a host would otherwise reimplement.

- `account_turn(store, thread_id, input, output, secs, user_initiated)` charges
  a finished turn against the thread's **active** goal (a paused, complete, or
  budget-limited goal accrues nothing from incidental conversation) and returns
  the goal as it stands afterwards, including a flip to `BudgetLimited`. A
  `user_initiated` turn also clears the one-shot `continuation_suppressed` flag;
  a continuation must not clear its own, or it would loop.
- `GoalBudgetGuard::for_goal(goal)` arms only for an active goal that has a
  budget. `check(store, in_flight_tokens)` adds the turn's spend so far to the
  accounted total and returns `BudgetVerdict::Stop` once it reaches the ceiling
  — checked mid-turn, this bounds a run to a small overshoot instead of
  discovering the overrun after the fact. `verdict_for` is the same decision
  with no store read. The guard captures the `goal_id` it was armed for, so a
  goal replaced mid-turn quietly disarms it rather than enforcing a ceiling that
  no longer describes the work.

Neither aborts anything: `account_turn` reports, the guard returns a verdict,
and wiring a stop into a turn stays the host's call.

## Testing

Unit tests in `src/graph/goals/test.rs` (types, store, tools, the gate loop, and
budget enforcement on `InMemoryStore`); an end-to-end self-driving loop in
`tests/e2e_graph_goals.rs`; feature coverage for budget accounting and the
mid-turn guard in `tests/feature_graph_goal_budget.rs`.
