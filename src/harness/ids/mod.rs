//! Identifier newtypes and lifecycle enums.
//!
//! These ids are the keys that make recursion observable and correlatable: a
//! [`RunId`] names one run, and pairing it with the `root_run_id` /
//! `parent_run_id` recorded in [`crate::harness::events::HarnessRunStatus`] lets
//! a child run (a sub-agent or sub-graph invocation) be traced back up to the
//! top-level run that spawned it. A `From`/`Display`/`as_str` surface is
//! generated for each newtype by a single macro so the ids stay cheap to clone,
//! log, and serialize.
//!
//! See [`types`] for the type definitions. This module provides the shared
//! constructors, accessors, and conversions for every id newtype.

mod types;

pub use types::*;

/// Implements the common surface (`new`, `as_str`, `Display`, `From`) for a
/// string-backed id newtype.
macro_rules! impl_string_id {
    ($name:ident) => {
        impl $name {
            /// Creates a new id from anything convertible into a `String`.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Returns the id as a string slice.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }
    };
}

impl_string_id!(RunId);
impl_string_id!(ThreadId);
impl_string_id!(CallId);
impl_string_id!(EventId);
impl_string_id!(ComponentId);
impl_string_id!(GraphId);
impl_string_id!(NodeId);
impl_string_id!(TaskId);
impl_string_id!(SessionId);
impl_string_id!(CellId);
impl_string_id!(CheckpointId);
impl_string_id!(InterruptId);

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-unique monotonic sequence source for deterministic, dependency-free
/// id generation.
///
/// Reused by the id constructors below instead of wall-clock time or randomness
/// so ids stay reproducible across platforms and easy to assert in tests.
static ID_SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns the next process-unique monotonic sequence number.
///
/// This is the single id-allocation primitive shared by the `new_*_id`
/// helpers; it never repeats within a process and requires no `rand`,
/// `SystemTime`, or `Date` dependency.
pub fn next_seq() -> u64 {
    ID_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Returns a per-process nonce, stable for the lifetime of the process and
/// (in practice) distinct across restarts.
///
/// The bare monotonic [`next_seq`] restarts at `0` in every new process, so ids
/// built from it alone (`ckpt-0`, `ckpt-1`, …) *collide* across a restart — a
/// resumed thread would re-mint checkpoint ids it already used, corrupting the
/// parent-lineage map and time-travel resume. Mixing in this nonce makes ids
/// collision-free across restarts while [`next_seq`] keeps them ordered within
/// a process. Seeded once from the wall clock (nanoseconds since the epoch);
/// only ever used as an opaque uniqueness component, never parsed or compared
/// for time.
pub fn process_nonce() -> u64 {
    static NONCE: OnceLock<u64> = OnceLock::new();
    *NONCE.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    })
}

/// Returns the current wall-clock time in milliseconds since the Unix epoch,
/// or `0` if the clock is set before the epoch.
///
/// The single `now_ms` used across the crate for timestamping records,
/// checkpoints, goals, and observability events, so the epoch/`unwrap_or(0)`
/// convention lives in exactly one place instead of being re-hand-rolled in
/// every module that needs a millisecond timestamp.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Allocates a fresh [`RunId`] of the form `run-<nonce>-<n>`, collision-free
/// across process restarts (see [`process_nonce`]).
pub fn new_run_id() -> RunId {
    RunId(format!("run-{}-{}", process_nonce(), next_seq()))
}

/// Allocates a fresh [`CheckpointId`] of the form `ckpt-<nonce>-<n>`,
/// collision-free across process restarts (see [`process_nonce`]).
///
/// Restart-safety is essential here: checkpoint ids are the keys of the
/// parent-lineage spine that `prune`, `get_state_history`, and
/// `ResumeTarget::Checkpoint` all walk, so a duplicate id across a restart can
/// delete a live record's ancestor or resume the wrong checkpoint.
pub fn new_checkpoint_id() -> CheckpointId {
    CheckpointId(format!("ckpt-{}-{}", process_nonce(), next_seq()))
}

/// Allocates a fresh, process-unique [`SessionId`] of the form `session-<n>`.
pub fn new_session_id() -> SessionId {
    SessionId(format!("session-{}", next_seq()))
}

/// Allocates a fresh, process-unique [`CellId`] of the form `cell-<n>`.
pub fn new_cell_id() -> CellId {
    CellId(format!("cell-{}", next_seq()))
}

/// Allocates a fresh, process-unique [`CallId`] of the form `call-<n>`.
pub fn new_call_id() -> CallId {
    CallId(format!("call-{}", next_seq()))
}

#[cfg(test)]
mod test;
