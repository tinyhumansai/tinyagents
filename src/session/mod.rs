//! Durable session database and run ledger.
//!
//! SQLite-backed store (WAL + FTS5) for sessions, messages, tool calls, cost
//! metadata, and parent/child lineage, plus a [`run_ledger`] for background
//! agent/workflow execution state. This is the runtime's *history* layer: what
//! ran, what it cost, what it called, and how runs nest.
//!
//! # Why this is a top-level module
//!
//! Session history is a persistence domain in its own right, not a part of the
//! agent loop. Nothing in [`crate::harness`] reads from it, and a host can use
//! it without running a harness at all — indexing sessions produced elsewhere,
//! or recovering orchestration state at boot before any agent exists. Filing it
//! under `harness::` would imply a dependency that does not exist in either
//! direction.
//!
//! # Relationship to the other persistence layers
//!
//! - [`crate::harness::store`] is namespaced key-value storage for live
//!   runtime data. It is a substrate runs read and write during execution.
//! - [`crate::graph::checkpoint`] is durability for *resuming* an interrupted
//!   graph run.
//! - This module is queryable history. Nothing resumes from it; it answers
//!   "what happened", supports cross-session search, and lets a host recover
//!   orchestration state after a restart.
//!
//! A host that keeps its own transcript files (the source of truth for
//! KV-cache resume) still wants this module for indexing and search over them.
//!
//! # Layout
//!
//! Every entry point takes the workspace root and derives the database path,
//! so a host chooses only where its workspace lives:
//!
//! ```text
//! {workspace_dir}/session_db/sessions.db
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use tinyagents::session::{self, SessionStatus};
//!
//! # fn main() -> tinyagents::Result<()> {
//! let workspace = Path::new("/tmp/workspace");
//!
//! session::record_session_start(
//!     workspace, "sess-1", "researcher", "Researcher", "sess-1",
//!     None, None, None, Some("gpt-5"), None,
//! )?;
//! session::record_message(
//!     workspace, "sess-1", "user", "summarize the repo", None, None, None, None,
//! )?;
//! session::record_session_end(
//!     workspace, "sess-1", SessionStatus::Completed, 1, 120, 340, 0, 0.004,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! Requires the `sqlite` feature.
//!
//! See [`README.md`](./README.md) for the schema, the FTS behaviour, and the
//! coordination guarantees.

mod migrations;
mod context;
mod ops;
pub mod run_ledger;
mod store;
pub mod types;

pub use ops::{
    get_session, list_children, list_messages, list_sessions, list_tool_calls, mark_interrupted,
    record_message, record_session_end, record_session_start, record_tool_call, search_sessions,
};
pub use store::{db_path, with_connection, with_transaction};
pub use types::{
    SessionMessage, SessionRecord, SessionSearchParams, SessionSearchResult, SessionStatus,
    SessionToolCall,
};

#[cfg(test)]
mod test;
