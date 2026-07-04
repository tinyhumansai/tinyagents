//! Type definitions for the harness embeddings + retrieval module.
//!
//! These are the building blocks of retrieval-augmented context — the
//! [`EmbeddingModel`] / [`VectorStore`] / [`Retriever`] triad that lets a
//! recursive agent fetch external knowledge on demand rather than carrying it
//! all in-context.
//!
//! All public types declared here are re-exported through [`super`] so callers
//! import them from `crate::harness::embeddings` directly.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;

// ── EmbeddingModel ────────────────────────────────────────────────────────────

/// Provider-neutral embedding model.
///
/// An embedding model turns text into dense [`f32`] vectors that downstream
/// vector stores and retrievers can compare with a distance metric (this module
/// uses cosine similarity). Implementations must be `Send + Sync` so they can be
/// shared across async task boundaries behind an [`Arc`].
///
/// The harness keeps embedding generation separate from the chat model
/// abstraction: chat models produce messages, embedding models produce vectors.
///
/// # Contract
/// - [`embed`](EmbeddingModel::embed) returns exactly one vector per input
///   text, in the same order as the inputs.
/// - Every returned vector has length [`dimensions`](EmbeddingModel::dimensions).
/// - Embedding the same text twice should produce the same vector for
///   deterministic implementations such as [`MockEmbeddingModel`].
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Embeds a batch of texts, returning one vector per input in input order.
    ///
    /// Returning an empty `Vec` for empty input is valid.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Returns the fixed dimensionality of every vector this model produces.
    fn dimensions(&self) -> usize;
}

// ── MockEmbeddingModel ────────────────────────────────────────────────────────

/// Deterministic, offline embedding model for tests and examples.
///
/// `MockEmbeddingModel` hashes the input text to derive a stable vector without
/// any network access. Identical text always maps to an identical vector (so
/// the cosine similarity of a text with itself is exactly `1.0`), while
/// different texts map to different vectors. This makes retrieval behaviour
/// testable offline: querying with the exact text of an indexed document ranks
/// that document first.
///
/// The vectors are **not** semantically meaningful — this model exists purely
/// for deterministic shape/retrieval tests, mirroring LangChain's
/// `DeterministicFakeEmbedding`.
///
/// # Example
/// ```
/// use tinyagents::harness::embeddings::{EmbeddingModel, MockEmbeddingModel};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let model = MockEmbeddingModel::new(16);
/// let vectors = model.embed(&["hello".to_string()]).await.unwrap();
/// assert_eq!(vectors.len(), 1);
/// assert_eq!(vectors[0].len(), 16);
/// # });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MockEmbeddingModel {
    /// Fixed dimensionality of every produced vector.
    pub(crate) dimensions: usize,
}

impl MockEmbeddingModel {
    /// Creates a deterministic mock embedding model producing vectors of length
    /// `dimensions`.
    ///
    /// # Panics
    /// Panics if `dimensions` is `0`, since a zero-length vector cannot be
    /// compared with cosine similarity.
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "embedding dimensions must be non-zero");
        Self { dimensions }
    }

    /// Computes the deterministic vector for a single `text`.
    ///
    /// Each component is derived by hashing `(text, component_index)` and
    /// mapping the result into the half-open range `[-1.0, 1.0)`. The mapping is
    /// pure, so repeated calls with the same `text` always return the same
    /// vector.
    pub(crate) fn embed_one(&self, text: &str) -> Vec<f32> {
        (0..self.dimensions)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                text.hash(&mut hasher);
                i.hash(&mut hasher);
                let raw = hasher.finish();
                // Map into [0, 1) then shift/scale into [-1, 1).
                let frac = (raw % 1_000_000) as f32 / 1_000_000.0;
                frac * 2.0 - 1.0
            })
            .collect()
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ── ScoredDoc ─────────────────────────────────────────────────────────────────

/// A document returned from a vector-store or retriever query, with its
/// relevance score.
///
/// `score` is a cosine similarity in `[-1.0, 1.0]` where **higher is more
/// similar**. Results from this module's stores are returned in descending
/// score order (most similar first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredDoc {
    /// Caller-supplied identifier of the matched document.
    pub id: String,
    /// Cosine similarity to the query vector; higher is more similar.
    pub score: f32,
    /// Arbitrary metadata associated with the document at index time.
    pub metadata: Value,
}

// ── VectorStore ───────────────────────────────────────────────────────────────

/// A store of dense vectors that supports nearest-neighbour search.
///
/// Implementations associate an `id` and arbitrary `metadata` with each vector,
/// and answer top-`k` similarity queries. This module's [`InMemoryVectorStore`]
/// ranks results by cosine similarity.
///
/// Implementations must be `Send + Sync` so they can be shared behind an
/// [`Arc`].
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds (or overwrites, when `id` already exists) a vector with associated
    /// `metadata`.
    async fn add(&self, id: String, vector: Vec<f32>, metadata: Value) -> Result<()>;

    /// Returns up to `top_k` documents most similar to `vector`, sorted by
    /// descending similarity score.
    ///
    /// Returns fewer than `top_k` documents when the store holds fewer entries,
    /// and an empty `Vec` when `top_k` is `0` or the store is empty.
    async fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<ScoredDoc>>;
}

// ── InMemoryVectorStore ───────────────────────────────────────────────────────

/// A single stored vector together with its id and metadata.
#[derive(Clone, Debug)]
pub(crate) struct VectorEntry {
    /// Caller-supplied document id.
    pub(crate) id: String,
    /// The stored dense vector.
    pub(crate) vector: Vec<f32>,
    /// Arbitrary metadata attached at insert time.
    pub(crate) metadata: Value,
}

/// Shared interior state of an [`InMemoryVectorStore`]: the entries in
/// insertion order plus an id → index map for O(1) upsert-by-id.
///
/// Entries are never removed, only appended or replaced in place, so the
/// indices in `index` stay valid for the lifetime of the store.
#[derive(Debug, Default)]
pub(crate) struct VectorStoreInner {
    /// All stored entries, in insertion order.
    pub(crate) entries: Vec<VectorEntry>,
    /// Maps each entry id to its position in `entries`.
    pub(crate) index: HashMap<String, usize>,
}

/// Thread-safe in-process [`VectorStore`] backed by a plain [`Vec`].
///
/// Search is a linear scan computing cosine similarity against every stored
/// vector, which is appropriate for tests, examples, and small corpora. The
/// store is cheaply clonable through the inner [`Arc`]; clones share the same
/// underlying data.
///
/// Adding a vector whose `id` already exists **replaces** the previous entry
/// (an O(1) id-indexed upsert), so re-indexing a document updates it in place.
///
/// # Example
/// ```
/// use tinyagents::harness::embeddings::{InMemoryVectorStore, VectorStore};
/// use serde_json::json;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let store = InMemoryVectorStore::new();
/// store.add("a".into(), vec![1.0, 0.0], json!({})).await.unwrap();
/// store.add("b".into(), vec![0.0, 1.0], json!({})).await.unwrap();
/// let hits = store.query(&[1.0, 0.0], 1).await.unwrap();
/// assert_eq!(hits[0].id, "a");
/// # });
/// ```
#[derive(Clone, Debug, Default)]
pub struct InMemoryVectorStore {
    /// Entries plus their id index, protected by a standard mutex.
    pub(crate) inner: Arc<Mutex<VectorStoreInner>>,
}

// ── Retriever ─────────────────────────────────────────────────────────────────

/// Query-to-document component tying an [`EmbeddingModel`] to a [`VectorStore`].
///
/// A `Retriever` embeds documents at index time and embeds queries at retrieval
/// time using the **same** embedding model, then delegates nearest-neighbour
/// search to the vector store. Both collaborators are held behind [`Arc`] so a
/// retriever is cheap to clone and share.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tinyagents::harness::embeddings::{InMemoryVectorStore, MockEmbeddingModel, Retriever};
/// use serde_json::json;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let retriever = Retriever::new(
///     Arc::new(MockEmbeddingModel::new(32)),
///     Arc::new(InMemoryVectorStore::new()),
/// );
/// retriever
///     .index(vec![
///         ("d1".into(), "cats are great".into(), json!({})),
///         ("d2".into(), "the stock market crashed".into(), json!({})),
///     ])
///     .await
///     .unwrap();
/// let hits = retriever.retrieve("cats are great", 1).await.unwrap();
/// assert_eq!(hits[0].id, "d1");
/// # });
/// ```
#[derive(Clone)]
pub struct Retriever {
    /// Embedding model used for both documents and queries.
    pub(crate) model: Arc<dyn EmbeddingModel>,
    /// Backing vector store searched at retrieval time.
    pub(crate) store: Arc<dyn VectorStore>,
}
