# graph::todos::dispatch

Dispatch policy for a task board: **what runs next, what it is told to do, and
how a run in flight is tracked and cancelled.** `store` owns the board and
`runs` owns claims; this is the layer between them a scheduler is built from.

Ported from OpenHuman's `agent::task_dispatcher`, keeping the parts that are
policy and leaving behind the parts that are a host (its agent registry, config
loading, personality profiles, and event bus).

## Selection (`select.rs`)

Pure functions over a board snapshot — no store, so they are trivially testable
and can be applied to cards a host already holds.

- `pick_next_card(cards, agent_assigned_only)` — the highest-urgency
  dispatchable card (`Todo` or approved `Ready`). Urgency comes from
  `source_metadata.urgency` (`card_urgency`, default `0.0`); ties break toward
  the lower board `order`, so equal-priority work runs in planned order.
  `agent_assigned_only` restricts the pick to cards with an `assigned_agent`,
  which is how a host keeps an autonomous sweep off a person's own todos.
- `has_card_in_progress(cards)` — the board already has a card being worked, so
  there is nothing to claim this tick.
- `requires_plan_approval(global_required, approval_mode)` — the card's own
  mode is authoritative; `global_required` is only the fallback. `Required`
  parks the card **even when the global gate is off**, or a plan stamped for
  interactive review would execute before anyone saw it.
- `PollCadence { base, max_backoff, grace_ticks }` — diminishing-returns
  polling. `next_delay(idle_ticks)` holds `base` through the grace window, then
  doubles per idle tick, saturating at `max_backoff`. Monotonic and
  overflow-free for any `u32` streak. Defaults: 60s base, 15min ceiling, 2 ticks
  of grace.

## Prompts (`prompt.rs`)

- `build_task_prompt(card, tools)` — objective (falling back to the title), then
  numbered plan steps and acceptance criteria. A card carrying
  `source_metadata` with a `provider` also gets its provenance, a pointer at the
  host's memory-recall tool, and an instruction to record the outcome back on
  the upstream item. An id-only card gets no provenance block at all, since a
  bare `#123` tells the model nothing.
- `build_progress_instruction(card_id, thread_id, tools)` — the addendum that
  asks the run to append notes/evidence as it works and, crucially, to **block
  rather than guess** when it needs a decision it cannot make. A run that blocks
  leaves the card paused for a human instead of force-completed.
- `TaskPromptTools { memory_recall, update_task }` names the tools those prompts
  point at, so a host that registers them under other names (or has no memory
  tool) still gets coherent text.

## Registry (`registry.rs`)

`ActiveRunRegistry<Context>` maps a session thread id to an `ActiveRun`
(`run_id`, `card_id`, an `AbortHandle`, a heartbeat cancel sender, and the
host's own `Context`).

Its real job is **deciding who cleans up**. A run finishing naturally and a
cancel arriving at the same moment both want to write the card's terminal
state; both must go through `take` / `take_if`, and only one gets `Some`.
`take_if(thread_id, Some(run_id))` matches and removes under a single lock, so
a cancel for a superseded request cannot tear down the run that replaced it.

## Composing a dispatcher

```text
reclaim_stale → has_card_in_progress? → pick_next_card
  → requires_plan_approval? → park at AwaitingApproval
  → claim_card → create_run → build_task_prompt → spawn + register
  → complete_run → card write-back → take from the registry
```

Executing the card is deliberately out of scope: that needs an agent, a model,
and a host's tool belt. `tests/e2e_graph_task_dispatch.rs` wires the whole loop
against a `MockModel` and is the reference assembly.

## Files

| File | Role |
| --- | --- |
| `select.rs` | Card selection, approval gate, polling cadence. |
| `prompt.rs` | Task prompt and progress-instruction rendering. |
| `registry.rs` | In-flight run tracking with race-free removal. |
| `test.rs` | Unit tests for all three. |
