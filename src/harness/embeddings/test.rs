//! Unit tests for the embeddings + retrieval module.

use std::sync::Arc;

use serde_json::json;

use super::*;

#[test]
fn cosine_similarity_identical_is_one() {
    assert_eq!(cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]), 1.0);
    assert_eq!(cosine_similarity(&[3.0, 4.0], &[6.0, 8.0]), 1.0);
}

#[test]
fn cosine_similarity_orthogonal_is_zero() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
}

#[test]
fn cosine_similarity_opposite_is_negative_one() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
}

#[test]
fn cosine_similarity_known_value() {
    // 45 degrees between (1,0) and (1,1) -> cos = 1/sqrt(2).
    let s = cosine_similarity(&[1.0, 0.0], &[1.0, 1.0]);
    assert!((s - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn cosine_similarity_degenerate_inputs_return_zero() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
}

#[tokio::test]
async fn mock_model_is_deterministic_and_correct_shape() {
    let model = MockEmbeddingModel::new(24);
    assert_eq!(model.dimensions(), 24);
    let a = model.embed(&["hello world".to_string()]).await.unwrap();
    let b = model.embed(&["hello world".to_string()]).await.unwrap();
    assert_eq!(a, b, "identical text must yield identical vectors");
    assert_eq!(a[0].len(), 24);

    // A text embedded against itself has cosine similarity 1.0.
    assert!((cosine_similarity(&a[0], &b[0]) - 1.0).abs() < 1e-6);

    // Different texts produce different vectors.
    let c = model
        .embed(&["totally different".to_string()])
        .await
        .unwrap();
    assert_ne!(a[0], c[0]);
}

#[tokio::test]
async fn mock_model_batches_in_order() {
    let model = MockEmbeddingModel::new(8);
    let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];
    let vectors = model.embed(&texts).await.unwrap();
    assert_eq!(vectors.len(), 3);
    // Each batched vector matches the individually-embedded vector.
    for (text, v) in texts.iter().zip(vectors.iter()) {
        let single = model.embed(std::slice::from_ref(text)).await.unwrap();
        assert_eq!(&single[0], v);
    }
}

#[tokio::test]
async fn mock_model_empty_input_returns_empty() {
    let model = MockEmbeddingModel::new(8);
    let vectors = model.embed(&[]).await.unwrap();
    assert!(vectors.is_empty());
}

#[tokio::test]
async fn vector_store_ranks_by_cosine_similarity() {
    let store = InMemoryVectorStore::new();
    assert!(store.is_empty());
    store
        .add("a".into(), vec![1.0, 0.0], json!({"k": "a"}))
        .await
        .unwrap();
    store
        .add("b".into(), vec![0.0, 1.0], json!({"k": "b"}))
        .await
        .unwrap();
    store
        .add("c".into(), vec![0.9, 0.1], json!({"k": "c"}))
        .await
        .unwrap();
    assert_eq!(store.len(), 3);

    let hits = store.query(&[1.0, 0.0], 2).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "c");
    assert!(hits[0].score >= hits[1].score);
    assert_eq!(hits[0].metadata, json!({"k": "a"}));
}

#[tokio::test]
async fn vector_store_top_k_zero_and_overflow() {
    let store = InMemoryVectorStore::new();
    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();
    assert!(store.query(&[1.0, 0.0], 0).await.unwrap().is_empty());
    // Requesting more than stored returns all entries.
    assert_eq!(store.query(&[1.0, 0.0], 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn vector_store_add_replaces_existing_id() {
    let store = InMemoryVectorStore::new();
    store
        .add("x".into(), vec![1.0, 0.0], json!({"v": 1}))
        .await
        .unwrap();
    store
        .add("x".into(), vec![0.0, 1.0], json!({"v": 2}))
        .await
        .unwrap();
    assert_eq!(store.len(), 1);
    let hits = store.query(&[0.0, 1.0], 1).await.unwrap();
    assert_eq!(hits[0].id, "x");
    assert_eq!(hits[0].metadata, json!({"v": 2}));
}

#[tokio::test]
async fn vector_store_rejects_mismatched_query_dimension() {
    let store = InMemoryVectorStore::new();
    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();

    let err = store.query(&[1.0, 0.0, 0.0], 1).await.unwrap_err();
    assert!(
        matches!(err, crate::error::TinyAgentsError::Validation(_)),
        "{err:?}"
    );
    assert!(err.to_string().contains("dimensions"), "{err}");
}

#[tokio::test]
async fn vector_store_rejects_mismatched_or_empty_add() {
    let store = InMemoryVectorStore::new();
    // Zero-dimensional vectors are rejected outright.
    let err = store.add("z".into(), vec![], json!({})).await.unwrap_err();
    assert!(
        matches!(err, crate::error::TinyAgentsError::Validation(_)),
        "{err:?}"
    );

    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();
    // The first vector fixes the store's dimensionality.
    let err = store
        .add("b".into(), vec![1.0, 0.0, 0.0], json!({}))
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::error::TinyAgentsError::Validation(_)),
        "{err:?}"
    );
    assert_eq!(store.len(), 1, "rejected vectors must not be stored");
}

#[tokio::test]
async fn vector_store_empty_store_accepts_any_query_dimension() {
    let store = InMemoryVectorStore::new();
    // No stored dimensionality to compare against: any query returns no hits.
    assert!(store.query(&[1.0, 2.0, 3.0], 5).await.unwrap().is_empty());
    assert!(store.query(&[], 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn retriever_rejects_query_of_wrong_dimension() {
    // Index with a 8-dim model, then retrieve with a 4-dim model over the same
    // store: the mismatch must surface as a Validation error, not zero-score
    // arbitrary hits.
    let store = Arc::new(InMemoryVectorStore::new());
    let indexer = Retriever::new(Arc::new(MockEmbeddingModel::new(8)), store.clone());
    indexer
        .index(vec![("doc".into(), "some text".into(), json!({}))])
        .await
        .unwrap();

    let querier = Retriever::new(Arc::new(MockEmbeddingModel::new(4)), store);
    let err = querier.retrieve("some text", 1).await.unwrap_err();
    assert!(
        matches!(err, crate::error::TinyAgentsError::Validation(_)),
        "{err:?}"
    );
}

#[tokio::test]
async fn retriever_index_and_retrieve_most_similar_first() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(64)),
        Arc::new(InMemoryVectorStore::new()),
    );
    retriever
        .index(vec![
            (
                "cats".into(),
                "cats are great pets".into(),
                json!({"topic": "animals"}),
            ),
            (
                "dogs".into(),
                "dogs are loyal companions".into(),
                json!({"topic": "animals"}),
            ),
            (
                "finance".into(),
                "the stock market crashed today".into(),
                json!({"topic": "finance"}),
            ),
        ])
        .await
        .unwrap();

    // Querying with the exact text of an indexed doc ranks it first (cosine 1.0).
    let hits = retriever.retrieve("cats are great pets", 3).await.unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "cats");
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].metadata, json!({"topic": "animals"}));
}

#[tokio::test]
async fn retriever_empty_index_is_noop() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(8)),
        Arc::new(InMemoryVectorStore::new()),
    );
    retriever.index(vec![]).await.unwrap();
    assert!(retriever.retrieve("anything", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn retriever_accessors_expose_collaborators() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(8)),
        Arc::new(InMemoryVectorStore::new()),
    );
    assert_eq!(retriever.model().dimensions(), 8);
    let _ = retriever.store();
}
#[test]
fn embedding_identity_signature_is_stable() {
    let model = MockEmbeddingModel::new(8);
    assert_eq!(model.name(), "mock");
    assert_eq!(model.model_id(), "deterministic-hash");
    assert_eq!(
        model.signature(),
        "provider=mock;model=deterministic-hash;dims=8"
    );
    assert_eq!(
        format_embedding_signature("openai", "text-embedding-3-small", 1536),
        "provider=openai;model=text-embedding-3-small;dims=1536"
    );
}

#[test]
fn voyage_and_noop_identity_match_host_contract() {
    let voyage = VoyageEmbeddingModel::new("test-key");
    assert_eq!(voyage.name(), "voyage");
    assert_eq!(voyage.model_id(), VOYAGE_DEFAULT_MODEL);
    assert_eq!(
        voyage.signature(),
        "provider=voyage;model=voyage-3-large;dims=1024"
    );

    let noop = NoopEmbeddingModel;
    assert_eq!(noop.signature(), "provider=none;model=none;dims=0");
}
