//! Error-context helper for the session store's SQLite surface.
//!
//! The store and run ledger are dense with driver calls whose bare failures
//! (`no such table`, `database is locked`) say nothing about which operation
//! raised them. This trait attaches that operation context while funnelling
//! everything into [`TinyAgentsError::Storage`], so the crate keeps one error
//! type without every call site writing the same closure.
//!
//! It deliberately mirrors the shape of `anyhow::Context` — including the
//! `Option` impl for "row expected but absent" — because this module was
//! ported from a host that used `anyhow`, and matching the shape kept that
//! port mechanical and reviewable.

use std::fmt::Display;

use crate::error::{Result, TinyAgentsError};

/// Attaches operation context to a fallible storage call, producing a
/// [`TinyAgentsError::Storage`].
pub(crate) trait StorageContext<T> {
    /// Wraps the failure as a storage error prefixed with `context`.
    fn storage_context(self, context: &str) -> Result<T>;
}

impl<T, E: Display> StorageContext<T> for std::result::Result<T, E> {
    fn storage_context(self, context: &str) -> Result<T> {
        self.map_err(|err| TinyAgentsError::Storage(format!("{context}: {err}")))
    }
}

/// A `None` where a row was expected is a storage inconsistency, not a
/// user-facing absence — the callers using this read back a record they just
/// wrote, so `None` means the write silently failed.
impl<T> StorageContext<T> for Option<T> {
    fn storage_context(self, context: &str) -> Result<T> {
        self.ok_or_else(|| TinyAgentsError::Storage(context.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_error_is_prefixed_with_context() {
        let failed: std::result::Result<(), _> = Err("disk full");
        let err = failed.storage_context("write session").unwrap_err();
        assert!(matches!(err, TinyAgentsError::Storage(_)));
        assert_eq!(err.to_string(), "storage error: write session: disk full");
    }

    #[test]
    fn result_ok_passes_through() {
        let ok: std::result::Result<u8, &str> = Ok(7);
        assert_eq!(ok.storage_context("read").unwrap(), 7);
    }

    #[test]
    fn none_becomes_storage_error_without_a_source_suffix() {
        let absent: Option<u8> = None;
        let err = absent
            .storage_context("run missing after upsert")
            .unwrap_err();
        assert_eq!(err.to_string(), "storage error: run missing after upsert");
    }

    #[test]
    fn some_passes_through() {
        assert_eq!(Some(3).storage_context("read").unwrap(), 3);
    }
}
