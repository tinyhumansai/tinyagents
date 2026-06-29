//! TRUE end-to-end (offline): index documents through a [`Retriever`] backed by
//! a deterministic [`MockEmbeddingModel`] and an [`InMemoryVectorStore`], then
//! assert that `retrieve()` ranks the most relevant document first.
//!
//! This composes the **embeddings** subsystem (embedding model + vector store +
//! retriever) without any network access. The mock model hashes text to a
//! stable vector, so querying with the exact text of an indexed document yields
//! cosine similarity `1.0` and ranks that document first — a deterministic,
//! prose-free relevance assertion.

use std::sync::Arc;

use tinyagents::{InMemoryVectorStore, MockEmbeddingModel, Retriever};
use serde_json::json;

/// Builds a retriever over a small, topically-distinct corpus.
async fn indexed_retriever() -> Retriever {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(64)),
        Arc::new(InMemoryVectorStore::new()),
    );
    retriever
        .index(vec![
            (
                "cats".into(),
                "cats are great pets that purr".into(),
                json!({ "topic": "animals" }),
            ),
            (
                "finance".into(),
                "the stock market crashed today".into(),
                json!({ "topic": "finance" }),
            ),
            (
                "rust".into(),
                "rust is a systems programming language".into(),
                json!({ "topic": "programming" }),
            ),
        ])
        .await
        .expect("indexing succeeds");
    retriever
}

#[tokio::test]
async fn retrieve_ranks_exact_match_first() {
    let retriever = indexed_retriever().await;

    // Querying with the exact indexed text must rank that document first with a
    // (near-)perfect cosine score.
    let hits = retriever
        .retrieve("rust is a systems programming language", 3)
        .await
        .expect("retrieve succeeds");

    assert_eq!(hits.len(), 3, "top_k returns all three docs");
    assert_eq!(hits[0].id, "rust", "the exact match ranks first");
    assert!(
        (hits[0].score - 1.0).abs() < 1e-3,
        "exact match scores ~1.0, got {}",
        hits[0].score
    );
    // Carried metadata is intact on the top hit.
    assert_eq!(hits[0].metadata, json!({ "topic": "programming" }));

    // Results are sorted by descending score (most similar first).
    for pair in hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "scores are non-increasing: {} then {}",
            pair[0].score,
            pair[1].score
        );
    }
}

#[tokio::test]
async fn retrieve_respects_top_k() {
    let retriever = indexed_retriever().await;

    let top1 = retriever
        .retrieve("the stock market crashed today", 1)
        .await
        .expect("retrieve succeeds");
    assert_eq!(top1.len(), 1, "top_k=1 returns a single hit");
    assert_eq!(top1[0].id, "finance", "finance doc is the closest match");

    // top_k larger than the corpus returns only what exists.
    let all = retriever
        .retrieve("cats are great pets that purr", 10)
        .await
        .expect("retrieve succeeds");
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].id, "cats");
}

#[tokio::test]
async fn reindexing_same_id_updates_in_place() {
    // Keep a concrete handle to the store so we can assert on its `len()` (an
    // inherent method on `InMemoryVectorStore`, not on the `VectorStore` trait).
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = Retriever::new(Arc::new(MockEmbeddingModel::new(64)), store.clone());
    retriever
        .index(vec![
            (
                "cats".into(),
                "cats are great pets that purr".into(),
                json!({ "topic": "animals" }),
            ),
            (
                "finance".into(),
                "the stock market crashed today".into(),
                json!({ "topic": "finance" }),
            ),
            (
                "rust".into(),
                "rust is a systems programming language".into(),
                json!({ "topic": "programming" }),
            ),
        ])
        .await
        .expect("indexing succeeds");
    assert_eq!(store.len(), 3, "three distinct documents indexed");

    // Re-index an existing id with new text + metadata; the store updates in
    // place rather than appending a duplicate.
    retriever
        .index(vec![(
            "cats".into(),
            "kittens love to nap in the sun".into(),
            json!({ "topic": "animals", "v": 2 }),
        )])
        .await
        .expect("re-indexing succeeds");
    assert_eq!(store.len(), 3, "re-indexing does not add a duplicate entry");

    // The updated text is now the exact match for the "cats" id.
    let hits = retriever
        .retrieve("kittens love to nap in the sun", 1)
        .await
        .expect("retrieve succeeds");
    assert_eq!(hits[0].id, "cats");
    assert_eq!(hits[0].metadata, json!({ "topic": "animals", "v": 2 }));
}
