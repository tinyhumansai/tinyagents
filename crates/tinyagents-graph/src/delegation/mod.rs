//! Multi-stage sub-agent delegation as a durable, resumable state graph.
//!
//! Where a single harness call drives *one* agent turn, this module composes
//! several sub-agent stages into a checkpointed state machine — the graph-native
//! alternative to ad-hoc sub-agent chaining:
//!
//! ```text
//!   plan ─▶ execute ─▶ review ──approved/maxed──▶ finalize ─▶ END
//!             ▲                   │
//!             └─────revise────────┘
//! ```
//!
//! It exercises most of the graph layer:
//! - **conditional routing** — `review` returns a [`Command`](crate::Command)
//!   routing to `execute` (revise) or `finalize` (done) based on the stage result;
//! - **recursion bounds** — a [`RecursionPolicy`](crate::RecursionPolicy)
//!   caps the `execute` ⇄ `review` loop as a backstop to the in-state
//!   `revisions` counter;
//! - **durable checkpoint/resume** — an optional
//!   [`Checkpointer`](crate::Checkpointer) persists the typed
//!   [`DelegationState`] at every super-step boundary, so a crashed or paused
//!   run resumes from its last node;
//! - **human-in-the-loop** — with [`DelegationConfig::require_review_approval`]
//!   an approved review parks on a durable
//!   [`Interrupt`](crate::Interrupt) that survives a process restart and
//!   is released by [`resume_delegation`];
//! - **cooperative cancellation** — a
//!   [`CancellationToken`](crate::CancellationToken) short-circuits the pipeline
//!   to `finalize` at the next node boundary.
//!
//! # The per-stage worker is injected
//!
//! [`run_delegation`] and friends take a `run_stage` closure, so this module owns
//! the orchestration mechanics and nothing about *how* a stage runs. A host
//! passes a closure dispatching each [`DelegationStage`] to its own sub-agent
//! runner; tests pass a deterministic mock.
//!
//! # [`DelegationState`] is an on-disk format
//!
//! Checkpointed state is read back by a later process — potentially a later
//! release. [`CURRENT_SCHEMA_VERSION`] stamps fresh runs and
//! [`run_or_resume_delegation`] expires any checkpoint below it rather than
//! misreading a stale shape. Changing a field name, a `#[serde(...)]` attribute,
//! or a default breaks resume for existing installations; `test.rs` pins the
//! exact serialized JSON so such a change fails loudly. See `README.md`.
//!
//! # Entry points
//!
//! - [`run_delegation`] — run to completion, return the terminal state. The
//!   convenience wrapper for the non-gated shape.
//! - [`run_delegation_durable`] — run, reporting whether it finalized or parked
//!   on a human-approval interrupt.
//! - [`run_or_resume_delegation`] — resume a thread's last checkpoint boundary
//!   when one is live and compatible, else start fresh.
//! - [`resume_delegation`] — deliver an approver's decision to a parked run.
//! - [`delegation_graph_topology`] — structure-only export for inspection.

mod graph;
mod run;
mod types;

pub use graph::delegation_graph_topology;
pub use run::{
    deny_decision, resume_delegation, run_delegation, run_delegation_durable,
    run_or_resume_delegation,
};
pub use types::{
    CURRENT_SCHEMA_VERSION, DelegationConfig, DelegationOutcome, DelegationStage,
    DelegationStageOutput, DelegationState, PendingApproval, StepRecord,
};

#[cfg(test)]
mod test;
