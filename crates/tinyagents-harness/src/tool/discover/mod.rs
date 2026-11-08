//! On-demand tool discovery: keeping deferred tool schemas out of the context
//! window until a model asks for them.
//!
//! See `README.md` in this directory for the design and the cache rationale.
//! The pieces:
//!
//! - [`DeferredCatalog`] — the run's deferred schemas, BM25-indexed, ranked
//!   by the host's [`tinytools::ToolRanker`] when one is installed.
//! - [`ToolDiscoveryPolicy`] — the knobs, carried on
//!   [`crate::runtime::RunPolicy::discovery`], including the ranker and the
//!   [`DiscoveryRankMode`] that says how it is used.
//! - [`bridge_schemas`] / [`answer_tool_search`] — the intrinsic `tool_search`
//!   bridge tool the agent loop advertises and answers.
//! - [`EmbeddingToolRanker`] — a semantic [`tinytools::ToolRanker`] over any
//!   [`tinyinference_embeddings::EmbeddingModel`], with an in-memory and
//!   optional host-supplied embedding cache.
//! - [EmbeddingCache] — async persistence for rebuildable catalogue vectors;
//!   the host binds its namespace and the ranker keeps embedding spaces separate.
//!   [`DriverEmbeddingCache`] stores them on any `tinystoragedrivers` backend.
//! - [`render_manifest`] — the budgeted listing inside `tool_search`'s
//!   description.

mod bridge;
mod collapse;
mod embedding_cache;
mod embedding_ranker;
mod manifest;
mod types;

pub use bridge::{SearchAnswer, TOOL_SEARCH_NAME, answer_tool_search, bridge_schemas};
pub use collapse::collapse_search_results;
pub use embedding_cache::DriverEmbeddingCache;
pub use embedding_ranker::{EmbeddingCache, EmbeddingToolRanker};
// The BM25 arithmetic lives in `tinytools::rank` now, so a host ranks with the
// same index the bridge does; the old paths keep resolving.
pub use manifest::{MANIFEST_DESCRIPTION_CHARS, first_sentence, render_manifest};
pub use tinytools::rank::{Bm25Index, tokenize};
pub use types::{
    DeferredCatalog, DeferredTool, DiscoveryOverrides, DiscoveryRankMode, PriorSearchResults,
    RankedSearch, ToolDiscoveryPolicy,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
