//! Cooperative, runtime-agnostic run cancellation.
//!
//! In the recursive runtime a single [`CancellationToken`] can be shared down a
//! tree of nested runs (a parent agent, its sub-agents, and their sub-graphs),
//! so an orchestrator cancels an entire recursion with one `cancel()` and every
//! level unwinds at its next safe checkpoint with
//! [`crate::error::TinyAgentsError::Cancelled`].
//!
//! This module provides [`CancellationToken`], a lightweight, self-contained
//! cancellation primitive (an `Arc<AtomicBool>` paired with a [`tokio::sync::Notify`])
//! used to request that an in-flight harness run stop at its next safe
//! checkpoint. It deliberately avoids pulling in a heavier dependency such as
//! `tokio-util`: the harness only needs `new`/`cancel`/`is_cancelled` plus an
//! async `cancelled().await` future, all of which are a few lines over the
//! `tokio` `sync` feature already in the dependency tree.
//!
//! # Cooperative, not preemptive
//!
//! Cancelling a token never aborts a running future. Instead the agent loop
//! polls [`CancellationToken::is_cancelled`] at the same safe checkpoints it
//! uses for steering — before each model call and before each tool call — and
//! the streaming pipeline races [`CancellationToken::cancelled`] against the
//! provider stream. On observing cancellation the run unwinds cleanly with
//! [`crate::error::TinyAgentsError::Cancelled`]. This guarantees a token is
//! never observed in the middle of a side-effecting tool call or a partially
//! consumed provider stream chunk.
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::cancel::CancellationToken;
//!
//! # tokio_test_block(async {
//! let token = CancellationToken::new();
//! let waiter = token.clone();
//!
//! // In another task, request cancellation.
//! token.cancel();
//!
//! // The `cancelled` future resolves once cancellation is requested.
//! waiter.cancelled().await;
//! assert!(waiter.is_cancelled());
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread()
//! #         .build()
//! #         .unwrap()
//! #         .block_on(f)
//! # }
//! ```

mod types;

pub use types::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

use types::CancelState;

impl CancellationToken {
    /// Creates a fresh token that has **not** been cancelled.
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancelState {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    /// Requests cancellation.
    ///
    /// This latches the token into the cancelled state (it can never be
    /// undone) and wakes every task currently awaiting
    /// [`CancellationToken::cancelled`]. Calling `cancel` more than once is
    /// harmless and idempotent. Because every clone shares one state, this
    /// cancels the token observed through all clones.
    pub fn cancel(&self) {
        // `Release` so the flag write is visible to any `Acquire` poll in
        // `is_cancelled`; pair the wake-up after the store so a waiter that
        // re-checks the flag on wake always sees `true`.
        self.state.cancelled.store(true, Ordering::Release);
        self.state.notify.notify_waiters();
    }

    /// Returns `true` once [`CancellationToken::cancel`] has been called on this
    /// token or any of its clones.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Resolves as soon as the token is (or becomes) cancelled.
    ///
    /// If the token is already cancelled the returned future completes
    /// immediately. Otherwise it parks until [`CancellationToken::cancel`] is
    /// called. The implementation registers its interest with the underlying
    /// [`Notify`] *before* re-checking the flag, closing the race where a
    /// `cancel` lands between the initial check and parking.
    ///
    /// This future is cancel-safe and may be used in a `select!` arm; dropping
    /// it simply deregisters the waiter.
    pub async fn cancelled(&self) {
        // Fast path: already cancelled.
        if self.is_cancelled() {
            return;
        }

        let notify: &Notify = &self.state.notify;
        loop {
            let notified = notify.notified();
            tokio::pin!(notified);
            // Register as a waiter *before* re-checking the flag so a `cancel`
            // (which calls `notify_waiters`) racing with this check cannot be
            // missed.
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
            if self.is_cancelled() {
                return;
            }
            // Spurious wake (no transition observed): loop and re-arm.
        }
    }
}

/// A [`Default`] [`CancellationToken`] is a fresh, never-cancelled token.
impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod test;
