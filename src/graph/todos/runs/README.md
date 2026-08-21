# graph::todos::runs

The **claim / heartbeat / reclaim** layer over a task board. A
[`TaskBoardCard`](../types.rs) says what needs doing; a `TaskRun` says who is
doing it right now, since when, and how it ended.

Ported from OpenHuman's `threads::todos::runs`, minus the app coupling (domain
event bus, workspace-file layout). Runs are addressed like boards —
`(Store, thread_id)` — and share the board's timestamp format (unix-epoch
milliseconds as a string), so a board and its run log live side by side in one
store.

## Why it exists

A worker that dies mid-card leaves the card `InProgress` forever, and the
board's single-in-progress rule then wedges the whole thread. The run log makes
that recoverable: a sweep notices the silence, closes the run, and hands the
card back.

The bound matters as much as the recovery. A card that keeps killing its workers
would otherwise cycle through them forever, so after
`RunLimits::max_reclaim_count` reclaims it parks at `Blocked` with a blocker
naming the last reason.

## Data model (`types.rs`)

- `TaskRun { run_id, card_id, claimed_by, claim_token, started_at,
  last_heartbeat_at, completed_at, outcome, error, evidence }` (serde
  `camelCase`); `is_active()` is "no `completed_at`".
- `RunOutcome { Success, Failed, Reclaimed }`.
- `RunLimits { heartbeat_stale_secs, claim_ttl_secs, max_reclaim_count }`,
  defaulting to 300s / 3600s / 3.
- `ReclaimResult { reclaimed_count, blocked_count, details }` and
  `ReclaimDetail { run_id, card_id, reason, new_card_status }` — the crate emits
  no events, so a host derives its own from `details`.
- `staleness_reason(run, now_ms, limits)` — the policy as a pure,
  clock-injected function: TTL is checked before heartbeat (a run that is both
  reports the more fundamental reason), and an unparsable stamp reads as
  healthy so a corrupt record never yanks a card from a live worker.

## Store (`store.rs`)

One `Vec<TaskRun>` per thread under the `graph.todos.runs` namespace, keyed by
the hex-encoded thread id. Every mutation is `load → mutate → put` under a
per-thread async mutex (single-process, like the board store); the run lock map
is separate from the board's, so a run write never blocks a card write.

| Function | Role |
| --- | --- |
| `create_run` | Open a claim. Rejects a duplicate caller-supplied `run_id`. |
| `update_heartbeat` | Tick liveness. Errors on an unknown or finished run. |
| `complete_run` | Close with an outcome, error, and evidence. |
| `list_runs` / `get_run` | Read the log, optionally filtered to one card. |
| `find_stale_runs` | Active runs judged stale, each with its reason. |
| `count_reclaims_for_card` | How often a card has been handed back. |
| `reclaim_stale` | The sweep: close stale runs, return or park their cards. |
| `spawn_heartbeat_task` | Background ticker; stops on cancel or completion. |

Claiming does **not** touch the card. The caller moves it through
`todo_store::claim_card`, which is what enforces the single-`InProgress` rule.

## Example

```rust,ignore
use tinyagents::{RunLimits, RunOutcome, TaskCardStatus, task_run_store, todo_store};

todo_store::claim_card(&store, thread, &card_id,
    &[TaskCardStatus::Todo], TaskCardStatus::InProgress).await?;
let run = task_run_store::create_run(&store, thread, None, &card_id, "worker-a").await?;

task_run_store::spawn_heartbeat_task(
    store.clone(), thread.into(), run.run_id.clone(), cancel_rx,
    task_run_store::DEFAULT_HEARTBEAT_TICK,
);

// … work …
task_run_store::complete_run(
    &store, thread, &run.run_id, RunOutcome::Success, None, vec!["pr #12".into()],
).await?;

// Elsewhere, on a timer: hand back anything whose worker went quiet.
task_run_store::reclaim_stale(&store, thread, &RunLimits::default()).await?;
```

## Files

| File | Role |
| --- | --- |
| `types.rs` | `TaskRun`, `RunOutcome`, `RunLimits`, reclaim reports, `staleness_reason`. |
| `store.rs` | `Store`-backed lifecycle, the reclaim sweep, the heartbeat task. |
| `test.rs` | Unit tests (lifecycle, staleness policy, reclaim, serialization). |

Feature coverage lives in `tests/feature_graph_task_runs.rs`; the dispatch loop
that drives it end to end is in `tests/e2e_graph_task_dispatch.rs`.
