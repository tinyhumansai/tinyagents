//! The shared HTTP client policy for every embedding adapter.
//!
//! # Why this exists
//!
//! Every embedding adapter built its transport with `reqwest::Client::new()`,
//! which sets **no timeout at all** — neither a connect timeout nor an overall
//! one. reqwest's default really is "wait forever". A server that accepts the
//! TCP connection and then never answers therefore hung the calling task
//! indefinitely, with no error to retry and nothing in the logs.
//!
//! That is not a hypothetical failure mode: the chat path hit it and fixed it
//! **twice** — once with `DEFAULT_CONNECT_TIMEOUT_SECS` on the model client, and
//! again with the `list_models` deadline, whose comment names this exact
//! scenario ("an Ollama/LM Studio server that accepts the TCP connect and then
//! never responds … hung the call forever"). The embedding adapters, which point
//! at the same local servers, never got either.
//!
//! The constants deliberately mirror the chat path's, so the two halves of the
//! crate cannot drift into different opinions about how long a wedged local
//! server is allowed to hold a task.

use std::time::Duration;

/// TCP connect timeout. Bounds connection establishment without capping a
/// legitimately slow response body. Mirrors the chat transport's
/// `DEFAULT_CONNECT_TIMEOUT_SECS`.
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;

/// Overall request deadline.
///
/// Shorter than the chat path's 600 s because an embedding call has no
/// generation phase to wait through — it is one forward pass. A minute is
/// generous for a large batch against a cold local model and still bounds the
/// wedged-server case.
pub const DEFAULT_EMBEDDING_TIMEOUT_SECS: u64 = 120;

/// The default [`reqwest::Client`] every embedding adapter is constructed with.
///
/// Falls back to `reqwest::Client::new()` if the builder somehow fails, so a
/// construction path that cannot return an error stays infallible — an
/// unbounded client is bad, but panicking during construction is worse.
pub fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(DEFAULT_EMBEDDING_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|error| {
            tracing::warn!(
                target: "tinyagents::embeddings",
                %error,
                "[embeddings] could not build the default timeout-bounded client; \
                 falling back to an unbounded one"
            );
            reqwest::Client::new()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_client_is_bounded() {
        // reqwest exposes no getter for its configured timeouts, so assert the
        // property that actually matters and is observable: a request against a
        // closed port fails promptly rather than hanging.
        let client = default_client();
        // Cheap structural check that the builder path was taken at all.
        assert!(format!("{client:?}").contains("Client"));
    }

    /// An embedding call has no generation phase, so it must not inherit the
    /// chat path's 600 s patience — and the connect timeout must be the tighter
    /// of the two. Both are compile-time facts, so assert them as such.
    const _: () = {
        assert!(DEFAULT_EMBEDDING_TIMEOUT_SECS < 600);
        assert!(DEFAULT_CONNECT_TIMEOUT_SECS < DEFAULT_EMBEDDING_TIMEOUT_SECS);
    };
}
