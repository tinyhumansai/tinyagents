# `graph::delegation`

A multi-stage sub-agent delegation pipeline expressed as a durable state graph.

```text
  plan ─▶ execute ─▶ review ──approved / budget spent──▶ [approval] ─▶ finalize ─▶ END
            ▲                   │
            └──────revise───────┘
```

The `approval` node exists only when the run is human-gated
(`DelegationConfig::require_review_approval`).

## Why it is here

Chaining sub-agents ad hoc gives you no revision budget, no resume after a
crash, and no place to put a human in the loop. This module is that pipeline
written once against the graph runtime, so the routing, the bounded revision
loop, the checkpoint classification and the durable approval pause are one
reviewed implementation rather than a per-host reinvention.

## Public surface

| Item | Role |
| --- | --- |
| `run_delegation` | Run to completion, return the terminal `DelegationState`. The convenience wrapper for the non-gated shape, where the graph never interrupts. |
| `run_delegation_durable` | Run, reporting via `DelegationOutcome::pending` whether it finalized or parked on a human-approval interrupt. |
| `run_or_resume_delegation` | Classify the thread's latest checkpoint, then resume / return / start fresh accordingly. |
| `resume_delegation` | Deliver an approver's decision to a parked run. |
| `deny_decision` | The canonical deny value, for resume-on-timeout. |
| `delegation_graph_topology` | Structure-only export (nodes, edges, routing) for inspection. |

## The stage worker is injected

Every entry point takes a `run_stage` closure:

```rust,ignore
Fn(DelegationStage, DelegationState) -> impl Future<Output = Result<DelegationStageOutput, String>>
```

This module owns *when* a stage runs and *where its result routes*; it owns
nothing about *how* a stage runs. A host passes a closure dispatching each
`DelegationStage` to its own sub-agent runner; tests pass a deterministic mock,
which is what makes the orchestration mechanics unit-testable without a model.

The same reasoning applies to observability: `DelegationConfig::event_sink` is
an optional host-supplied `GraphEventSink`. The module never reaches for a
host-specific logger.

## `DelegationState` is an on-disk format

This is the constraint to read before editing anything in `types.rs`.

A checkpoint written by one release is decoded by a later one. So the serde
representation of `DelegationState` — field names, `#[serde(...)]` attributes,
defaults, and the resulting JSON — is a compatibility contract with every
installation that has a run in flight.

`CURRENT_SCHEMA_VERSION` is what makes that contract enforceable rather than
merely documented:

- a fresh run stamps its state with the current version;
- pre-versioned checkpoints decode to `0` via `#[serde(default)]`;
- `run_or_resume_delegation` **expires** any checkpoint below the current
  version instead of resuming or returning it.

That last point closes a gap a decode failure alone cannot: a stale record whose
fields still happen to decode (an empty `executions` list, say) is structurally
readable but semantically wrong for the current graph, and would otherwise be
resumed as if it were current.

`test.rs` pins the exact serialized JSON of a fully-populated state, of the
default state, and of a pre-versioned record. A field addition that drifts the
shape fails those tests rather than reaching a user's disk unnoticed. Changing
the shape deliberately means bumping `CURRENT_SCHEMA_VERSION` in the same
change.

## Bounds and failure handling

- **Revision budget.** `max_revisions` caps reviewer-requested revisions; at the
  cap the review is force-approved so a never-satisfied reviewer still
  terminates.
- **Recursion policy.** A `RecursionPolicy` bounds visits per node and total
  steps as a backstop to the in-state counter — the loop terminates even if the
  counter logic is wrong.
- **Retry.** `max_attempts(1)` preserves single-attempt node semantics; the seam
  is wired so raising it is a one-line change.
- **Cancellation** is checked at each node boundary and routes to `finalize`,
  so a cancelled run still produces a state rather than an error.
- **Checkpoint read errors are split by kind.** A *decode* failure expires the
  checkpoint and starts fresh; an *operational* failure (SQLite busy, I/O, a
  poisoned lock) is propagated. Treating the second as the first would silently
  discard valid durable work on a transient fault.

## Durable pause vs. interactive pause

The `approval` node is a **checkpointed graph interrupt**: the pause lives
entirely in the checkpointer and survives a process restart, and only
`resume_delegation` releases it. That is a durability mechanism for the pause —
it grants no approval authority and bypasses no host security boundary. A host's
own in-memory, TTL-bounded prompt for a live turn is a different mechanism and
is unaffected by anything here.
