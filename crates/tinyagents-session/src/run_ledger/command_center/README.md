# session::run_ledger::command_center

A read-only projection of the durable run ledger into the five buckets a
background-work UI shows, plus control verbs that move one run through the
ledger.

## Design

Live run state already reaches the ledger through the host's spawn tools and
progress bridge. This module only projects and transitions that state. It owns
no executor, scheduler or cancellation token, and it holds no product strings:
the host passes an agent-id to display-name closure and owns the RPC surface.

## Public surface

- `view`: `bucket_for` (status to bucket), `list_agent_work` (load and
  project), `build_view` (pure grouping, always five buckets in display
  order).
- `types`: `AgentWorkBucket` (needs-input, working, completed, failed,
  stopped), `AgentWorkRow`, `CommandCenterGroup`, `CommandCenterView`.
- `control`: `ControlVerb` (`stop`, `retry`, `continue`, `follow_up`),
  `ControlError`, `apply_control`. The allowed-transition matrix is the pure
  `plan_transition`.

## Operational constraints

- A control verb is a **durable ledger transition, not an execution action**.
  `apply_control` records the new status and a `control_*` run event. The host
  delivers the effect to the live executor: cancel the run's token on `stop`,
  inject the message on `continue` / `follow_up`, re-dispatch on `retry`. A
  host that only calls `apply_control` has changed the record, not the run.
- Transitions are compare-and-set. `transition_agent_run_status_from` updates
  the row only if it is still in the status the verb was planned against, and
  appends the run event in the same transaction, so a status never commits
  without its timeline entry and never overwrites a status another writer
  (the worker, the progress bridge) recorded meanwhile. On a miss
  `apply_control` re-plans against the new status, up to three attempts, and
  returns `InvalidTransition` when the verb no longer applies.
- `stop` stamps `completed_at` and stores the optional reason as `error`;
  `retry` and `continue` clear both; `follow_up` leaves the status unchanged.
# Durable addressed input draft

`run_ledger::inbox::RunInboxStore` is a separate optional capability implemented
by `DriverRunLedger`. `enqueue_input` validates the exact persisted recipient
lineage and commits the queued record, deduplication identity and `input_queued`
event in one checked batch. Reusing the root/submission id for different content
returns a conflict. Receipts survive process restart and list in event-sequence
order. A backend without checked atomic batches returns its native error.

`pending_inputs(root, recipient, after_sequence, limit)` returns at most 500
records in ascending sequence order. Start with `None`, then pass the last
receipt's sequence as `Some(sequence)` to read the next page. The cursor is
exclusive; paging never consumes or acknowledges input. Each page validates
the recipient's persisted root lineage, including empty pages.

This is an admission draft, not a complete delivery implementation. No operation
marks input delivered, changes a run to running, or launches a worker. Recipient
integration must atomically commit admitted input into the durable session and
record consumption before exposing delivery/resumption controls. Live job
request-id deduplication alone is not sufficient for this contract. Entries have
no implicit expiry; retention must preserve receipt identities while retries are
possible. Core host consumption awaits upstream availability.
