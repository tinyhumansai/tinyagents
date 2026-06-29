//! Retry and timeout policy.
//!
//! Owns retry classification, backoff, attempt accounting, retry events,
//! timeout wrappers, and the distinction between provider-level retries and
//! harness-level model/tool retry middleware.
