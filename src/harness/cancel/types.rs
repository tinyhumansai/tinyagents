//! Type definitions for cooperative run cancellation.
//!
//! A [`CancellationToken`] is a cheap, clonable handle that an orchestrator (a
//! parent agent, a human UI, a graph supervisor, a tool, a middleware, or a
//! test) uses to request that an in-flight run stop at its next safe
//! checkpoint. The token is *cooperative*: it never aborts work mid-flight.
//! Instead, the agent loop and the streaming pipeline poll
//! [`CancellationToken::is_cancelled`] (or `.await` on
//! [`CancellationToken::cancelled`]) at well-defined points and unwind cleanly
//! with [`crate::error::TinyAgentsError::Cancelled`].
//!
//! Implementations and tests live in the sibling `mod.rs` and `test.rs`; this
//! file owns the data layout.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::Notify;

/// Shared cancellation state behind a [`CancellationToken`].
///
/// Kept private: callers only ever interact through the cheap [`Arc`]-wrapped
/// [`CancellationToken`] handle. The `AtomicBool` provides a lock-free
/// `is_cancelled` poll, and the [`Notify`] wakes any task currently parked in
/// [`CancellationToken::cancelled`].
pub(super) struct CancelState {
    /// Set once, irreversibly, when [`CancellationToken::cancel`] is called.
    pub(super) cancelled: AtomicBool,
    /// Wakes tasks awaiting [`CancellationToken::cancelled`] on transition.
    pub(super) notify: Notify,
}

/// A cheap, clonable handle used to request cooperative cancellation of a run.
///
/// Cloning a token is `O(1)` (an [`Arc`] bump) and every clone observes the
/// same underlying state: cancelling any clone cancels them all. Cancellation
/// is *latching* — once requested it can never be undone — so a token only ever
/// transitions from "live" to "cancelled".
///
/// Construct a fresh, never-cancelled token with [`CancellationToken::new`]
/// (or [`Default`]). Attach one to a run via
/// [`crate::harness::context::RunContext::with_cancellation`]; the default
/// [`RunContext`][crate::harness::context::RunContext] carries a fresh token
/// that is never cancelled, so cancellation is strictly opt-in.
///
/// # Example
///
/// ```
/// use tinyagents::harness::cancel::CancellationToken;
///
/// let token = CancellationToken::new();
/// assert!(!token.is_cancelled());
///
/// let clone = token.clone();
/// clone.cancel();
/// assert!(token.is_cancelled()); // visible through every clone
/// ```
#[derive(Clone)]
pub struct CancellationToken {
    pub(super) state: Arc<CancelState>,
}
