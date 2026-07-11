//! Feature/integration tests for the harness embeddings + retrieval
//! infrastructure (`harness::embeddings`).
//!
//! Complements the existing `e2e_embeddings.rs` (which covers `Retriever`
//! ranking, `top_k`, and re-indexing) by exercising the lower-level surfaces
//! directly and offline: the `cosine_similarity` metric edge cases, the
//! `MockEmbeddingModel` determinism/shape contract, and the `InMemoryVectorStore`
//! dimensionality guards (first-vector-fixes-dimension, zero-length rejection,
//! query-dimension mismatch, empty-store behaviour).
//!
//! Deterministic and offline via `MockEmbeddingModel`.

use std::sync::Arc;

use serde_json::json;
use tinyagents::harness::embeddings::{
    EmbeddingModel, InMemoryVectorStore, MockEmbeddingModel, Retriever, VectorStore,
    cosine_similarity,
};

// ── cosine_similarity ───────────────────────────────────────────────────────

#[test]
fn cosine_similarity_covers_aligned_orthogonal_and_opposite() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
    // Magnitude-invariant: scaling a vector does not change the direction score.
    assert_eq!(cosine_similarity(&[2.0, 0.0], &[5.0, 0.0]), 1.0);
}

#[test]
fn cosine_similarity_degenerate_inputs_return_zero_not_nan() {
    // Mismatched lengths, empty inputs, and zero-magnitude vectors are all 0.0.
    assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
}

// ── MockEmbeddingModel ──────────────────────────────────────────────────────

#[tokio::test]
async fn mock_embedding_is_deterministic_and_correctly_shaped() {
    let model = MockEmbeddingModel::new(16);
    assert_eq!(model.dimensions(), 16);

    let a = model.embed(&["hello".to_string()]).await.unwrap();
    let b = model.embed(&["hello".to_string()]).await.unwrap();
    // One vector per input, each of the fixed dimensionality.
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].len(), 16);
    // Identical text → identical vector (offline determinism).
    assert_eq!(a, b);

    // Different text → different vector.
    let c = model.embed(&["world".to_string()]).await.unwrap();
    assert_ne!(a[0], c[0]);

    // A text embedded against itself scores a perfect cosine similarity.
    assert!((cosine_similarity(&a[0], &b[0]) - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn mock_embedding_batches_preserve_order() {
    let model = MockEmbeddingModel::new(8);
    let vectors = model
        .embed(&["one".to_string(), "two".to_string(), "three".to_string()])
        .await
        .unwrap();
    assert_eq!(vectors.len(), 3);
    // The per-input vector matches the single-input embedding for that text.
    let two = model.embed(&["two".to_string()]).await.unwrap();
    assert_eq!(vectors[1], two[0]);
}

// ── InMemoryVectorStore dimensionality guards ───────────────────────────────

#[tokio::test]
async fn vector_store_add_and_query_by_cosine() {
    let store = InMemoryVectorStore::new();
    assert!(store.is_empty());
    store
        .add("a".into(), vec![1.0, 0.0], json!({"t": "x"}))
        .await
        .unwrap();
    store
        .add("b".into(), vec![0.0, 1.0], json!({}))
        .await
        .unwrap();
    assert_eq!(store.len(), 2);

    let hits = store.query(&[1.0, 0.0], 1).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[0].metadata, json!({"t": "x"}));
    assert!((hits[0].score - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn vector_store_upserts_by_id_in_place() {
    let store = InMemoryVectorStore::new();
    store
        .add("a".into(), vec![1.0, 0.0], json!({"v": 1}))
        .await
        .unwrap();
    // Re-adding the same id replaces the entry rather than appending.
    store
        .add("a".into(), vec![0.0, 1.0], json!({"v": 2}))
        .await
        .unwrap();
    assert_eq!(store.len(), 1);

    let hits = store.query(&[0.0, 1.0], 1).await.unwrap();
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[0].metadata, json!({"v": 2}));
}

#[tokio::test]
async fn vector_store_rejects_zero_length_and_mismatched_dimensions() {
    let store = InMemoryVectorStore::new();
    // A zero-dimensional vector can never be compared.
    assert!(store.add("z".into(), vec![], json!({})).await.is_err());

    // The first vector fixes the store's dimensionality; a later mismatch fails.
    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();
    assert!(
        store
            .add("b".into(), vec![1.0, 0.0, 0.0], json!({}))
            .await
            .is_err()
    );

    // A wrong-length query vector is rejected rather than scoring 0.0 silently.
    assert!(store.query(&[1.0], 1).await.is_err());
}

#[tokio::test]
async fn empty_store_answers_any_query_with_no_hits() {
    let store = InMemoryVectorStore::new();
    // No entries → no dimensionality to validate against, so any query is empty.
    assert!(store.query(&[1.0, 2.0, 3.0], 5).await.unwrap().is_empty());
    // top_k of 0 is always empty.
    store.add("a".into(), vec![1.0], json!({})).await.unwrap();
    assert!(store.query(&[1.0], 0).await.unwrap().is_empty());
}

// ── Retriever cross-model dimensionality mismatch ───────────────────────────

#[tokio::test]
async fn retriever_surfaces_dimension_mismatch_from_a_different_model() {
    // Index with a 32-dim model, then query the same store through a 16-dim
    // model: the store rejects the wrong-length query vector.
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    let index_retriever = Retriever::new(Arc::new(MockEmbeddingModel::new(32)), store.clone());
    index_retriever
        .index(vec![("d1".into(), "cats".into(), json!({}))])
        .await
        .unwrap();

    let query_retriever = Retriever::new(Arc::new(MockEmbeddingModel::new(16)), store);
    assert!(query_retriever.retrieve("cats", 1).await.is_err());
}
