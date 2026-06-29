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
impl_string_id!(CheckpointId);
impl_string_id!(InterruptId);

#[cfg(test)]
mod test;
