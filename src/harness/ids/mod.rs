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

use std::sync::atomic::{AtomicU64, Ordering};

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
