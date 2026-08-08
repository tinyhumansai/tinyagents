//! Harness cache module — prompt, response, and layout caches.
//!
//! In the recursive runtime the same request can recur many times — a
//! sub-agent re-asked an identical sub-question, a graph node replayed during
//! recovery, or a deterministic test driving the loop twice. This module makes
//! that recursion cheap and deterministic: the local response cache short-
//! circuits an identical model call entirely (the agent loop emits
//! [`crate::harness::events::AgentEvent::CacheHit`] /
//! [`crate::harness::events::AgentEvent::CacheMiss`]), while the prompt-cache
//! layout tooling protects the stable prefix the *provider* itself caches.
//!
//! # Two distinct caching concerns
//!
//! ## 1. Local response cache
//! [`ResponseCache`] + [`InMemoryResponseCache`] (and, behind the `sqlite`
//! feature, [`SqliteResponseCache`]) let the harness skip provider API calls
//! entirely when it has already seen an identical request.
//!
//! The key is a **two-part composition**, never the prompt alone:
//!
//! ```text
//! scoped_cache_key(cache_key(request), model.cache_identity(), streaming, ns)
//! ```
//!
//! [`cache_key`] hashes an explicit allowlist projection of the request;
//! [`scoped_cache_key`] folds in the *resolved* model's identity (provider,
//! model id, endpoint, credential fingerprint — never a raw credential), the
//! streaming mode, and the policy namespace. Without the identity half, one
//! `Arc<InMemoryResponseCache>` shared between a hosted and a local harness
//! serves either's answer to the other.
//!
//! ## 2. Provider prompt / KV-cache layout protection
//! [`PromptCacheLayout`] records the ordered cacheable prefix of a request
//! *and* a digest of the material it carries, so a middleware that rewrites a
//! stable segment's text can no longer report "prefix stable".
//! [`CacheLayoutEvent`] describes mutations, and
//! [`CacheLayoutEvent::under_policy`] plus [`apply_prompt_cache_breakpoints`]
//! make [`CachePolicy::protect_prompt_prefix`] load-bearing rather than inert.
//!
//! ## 3. Stampede protection
//! [`SingleFlight`] collapses concurrent identical misses into one provider
//! call.

mod hash;
mod key;
mod layout;
mod memory;
mod singleflight;
#[cfg(feature = "sqlite")]
mod sqlite;
mod types;

pub use key::{
    PROMPT_CACHE_KEY_OPTION, apply_prompt_cache_breakpoints, cache_key, credential_fingerprint,
    model_cache_identity, prompt_cache_key, scoped_cache_key,
};
pub use singleflight::SingleFlight;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteResponseCache;
pub use types::*;

#[cfg(test)]
mod test;
