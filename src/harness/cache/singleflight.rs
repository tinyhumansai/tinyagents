//! Stampede protection: collapse concurrent identical cache misses into one
//! provider call.
//!
//! # The gap this fills
//!
//! A cache in front of a slow provider does nothing for the *first* N callers.
//! Ten sub-agents that ask the same sub-question at the same moment all miss,
//! all call the provider, and nine of those calls are pure waste — paid for,
//! rate-limit consuming, and then thrown away when the last writer wins.
//!
//! Neither reference implementation solves this: LangChain's own unit test
//! documents the race rather than preventing it. So this is greenfield, and
//! deliberately small — one in-flight map plus a `tokio::sync::broadcast`
//! channel per key.
//!
//! # Semantics
//!
//! The first caller for a key becomes the **leader** and runs the closure. Any
//! caller arriving while the leader is in flight becomes a **follower** and
//! waits for the leader's outcome instead of running the closure. Followers
//! receive a clone of the leader's success.
//!
//! Errors are **not** shared: a follower whose leader failed re-runs the
//! closure itself. Sharing the failure would turn one caller's transient 503
//! into every concurrent caller's failure while hiding the fact that each had
//! its own retry budget — and an error is not a value worth caching.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::Result;
use crate::harness::model::ModelResponse;

/// What a leader broadcasts to its followers when it finishes.
#[derive(Clone, Debug)]
enum Outcome {
    /// The leader succeeded; followers take this response.
    Ready(Box<ModelResponse>),
    /// The leader failed; followers must run the call themselves.
    Failed,
}

/// Collapses concurrent duplicate model calls that share a cache key.
///
/// Cheap to clone; clones share the same in-flight map.
#[derive(Clone, Default)]
pub struct SingleFlight {
    inflight: Arc<Mutex<HashMap<String, broadcast::Sender<Outcome>>>>,
}

impl std::fmt::Debug for SingleFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inflight = self.inflight.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("SingleFlight")
            .field("inflight", &inflight)
            .finish()
    }
}

impl SingleFlight {
    /// Creates an empty single-flight gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keys currently in flight. Intended for tests and diagnostics.
    pub fn inflight_len(&self) -> usize {
        self.inflight.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Runs `call` for `key`, or waits for an already in-flight call with the
    /// same key and returns its result.
    ///
    /// Returns `(response, was_follower)` so the caller can skip a redundant
    /// cache write when it merely rode along on someone else's call.
    pub async fn run<F, Fut>(&self, key: &str, call: F) -> Result<(ModelResponse, bool)>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<ModelResponse>>,
    {
        // The lock is acquired and released inside this helper so no
        // `MutexGuard` is ever alive across an `await` — which would make the
        // whole future `!Send` and unusable from `tokio::spawn`.
        let claim = self.claim(key);
        let Some(claim) = claim else {
            // A poisoned map must never take the run down: fall back to simply
            // making the call, which is the un-collapsed behaviour.
            tracing::warn!("[cache] single-flight map poisoned; issuing the model call directly");
            return call().await.map(|response| (response, false));
        };
        let mut receiver = claim;

        // Follower: wait for the leader rather than duplicating the call.
        if let Some(receiver) = receiver.as_mut() {
            tracing::debug!(key = %key, "[cache] joining an in-flight identical model call");
            match receiver.recv().await {
                Ok(Outcome::Ready(response)) => return Ok((*response, true)),
                // Leader failed, or dropped the channel without sending (a
                // cancelled or panicking leader). Either way, run it ourselves.
                Ok(Outcome::Failed) | Err(_) => {
                    tracing::debug!(
                        key = %key,
                        "[cache] in-flight leader did not produce a response; issuing our own call"
                    );
                    return call().await.map(|response| (response, false));
                }
            }
        }

        // Leader: run the call, then publish the outcome and retire the key.
        let result = call().await;
        let sender = self
            .inflight
            .lock()
            .ok()
            .and_then(|mut inflight| inflight.remove(key));
        if let Some(sender) = sender {
            let outcome = match &result {
                Ok(response) => Outcome::Ready(Box::new(response.clone())),
                Err(_) => Outcome::Failed,
            };
            // `send` fails only when no follower is subscribed, which is the
            // common (uncontended) case — not an error.
            let _ = sender.send(outcome);
        }
        result.map(|response| (response, false))
    }
}
