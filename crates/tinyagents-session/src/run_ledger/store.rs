//! Run-ledger schema entry point.
//!
//! The tables this module used to create on every operation now live in the
//! versioned migration list (`crate::migrations`, migration 2), which
//! is applied once per database when a connection is opened. Re-running
//! `CREATE TABLE IF NOT EXISTS` per call was not just wasted work: with no
//! version marker there was no way to ever *add* a column to a workspace
//! database that already existed.

use rusqlite::Connection;

use tinyagents_harness::error::Result;

/// No-op retained as the run-ledger's schema entry point.
///
/// Every `crate::store::with_connection` handle is already migrated by
/// the time it reaches a caller, so there is nothing left to do here. The
/// function is kept (rather than deleted along with its ~30 call sites) so the
/// ledger operations still read as "ensure my schema exists", and so a future
/// ledger-only bootstrap has an obvious place to land.
#[inline]
pub(crate) fn init_run_ledger_schema(_conn: &Connection) -> Result<()> {
    Ok(())
}
