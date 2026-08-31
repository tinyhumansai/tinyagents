//! LIVE local-embedding coverage: Ollama's native `/api/embed` and LM Studio's
//! OpenAI-compatible `/v1/embeddings`, end to end through [`Retriever`].
//!
//! `e2e_embeddings.rs` already proves the retrieval *plumbing* — but it does so
//! against [`MockEmbeddingModel`], which hashes text to a stable vector. That
//! makes the ranking assertion tautological: querying with a document's exact
//! text scores `1.0` by construction, whatever the embedding space is like. It
//! cannot catch a real embedding adapter that returns vectors in the wrong
//! order, silently truncates a batch, reports a dimensionality that disagrees
//! with the vectors it produces, or returns embeddings so degenerate that
//! retrieval is no better than chance.
//!
//! So this file drives real local embedding servers and asserts the properties
//! that actually matter for retrieval:
//!
//! - **dimensional honesty** — `dimensions()` agrees with every vector emitted,
//!   and stays stable across calls, because the vector store partitions on it,
//! - **positional integrity** — the *n*th vector belongs to the *n*th input,
//!   asserted by embedding a batch and comparing against one-at-a-time calls,
//! - **semantic separation** — a paraphrase scores higher than an unrelated
//!   sentence, so the vectors carry meaning rather than noise, and
//! - **retrieval** — a real query against a real index returns the right
//!   document first, with no exact-text shortcut.
//!
//! # Configuration
//!
//! | Variable | Default |
//! |---|---|
//! | `LOCAL_MODEL_TESTS` | unset — **required** to dial anything |
//! | `LOCAL_OLLAMA_EMBED_URL` | `http://localhost:11434` (server root, not `/v1`) |
//! | `LOCAL_OLLAMA_EMBED_MODEL` | `nomic-embed-text` |
//! | `LOCAL_LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` |
//! | `LOCAL_LMSTUDIO_EMBED_MODEL` | first embedding id `GET /v1/models` advertises |
//!
//! # Skips gracefully
//!
//! Opt-in via `LOCAL_MODEL_TESTS=1`, and any server that is not listening — or
//! has no embedding model installed — is reported and skipped rather than
//! failed.
//!
//! # Run
//!
//! ```text
//! LOCAL_MODEL_TESTS=1 cargo test --test live_local_embeddings -- --nocapture
//! ```

use std::sync::Arc;

use serde_json::json;
use tinyinference::embeddings::{
    EmbeddingModel, InMemoryVectorStore, OllamaEmbeddingModel, OpenAiEmbeddingModel,
    RECOMMENDED_OLLAMA_CONTEXT_TOKENS, Retriever, cosine_similarity,
};
use tinyinference::providers::openai::OpenAiModel;
use tinyinference::providers::{ProviderKind, ProviderSpec};

/// A local embedding backend under test, behind the provider-neutral trait so
/// every assertion below is written once and runs against both.
struct LocalEmbedder {
    name: &'static str,
    model: Arc<dyn EmbeddingModel>,
}

/// Ollama exposes embeddings on its **native** `/api/embed`, not the
/// OpenAI-compatible surface, so it gets the dedicated adapter and its base URL
/// is the server root (`OllamaEmbeddingModel` rejects a `/v1` or `/api` suffix).
async fn ollama_embedder() -> std::result::Result<LocalEmbedder, String> {
    let base_url = env_or("LOCAL_OLLAMA_EMBED_URL", "http://localhost:11434");
    let model_id = env_or("LOCAL_OLLAMA_EMBED_MODEL", "nomic-embed-text");

    // The width is a property of the installed GGUF, not something a caller can
    // know: `nomic-embed-text` is 768 wide while the adapter's own default
    // (`bge-m3`) is 1024, and declaring the wrong one makes every call fail
    // dimension validation. `embed_discovering_dimensions` is the supported way
    // to learn it — passing `0` to `try_new` does *not* mean "discover", it
    // means "use the default". This probe therefore doubles as the reachability
    // and model-installed check.
    let (width, _) = OllamaEmbeddingModel::embed_discovering_dimensions(
        &base_url,
        &model_id,
        reqwest::Client::new(),
        &["probe".to_string()],
        RECOMMENDED_OLLAMA_CONTEXT_TOKENS,
        RECOMMENDED_OLLAMA_CONTEXT_TOKENS,
    )
    .await
    .map_err(|e| format!("ollama embeddings ({model_id}): {e}"))?;

    let model = OllamaEmbeddingModel::try_new(&base_url, &model_id, width)
        .map_err(|e| format!("ollama: invalid embedding configuration: {e}"))?;

    Ok(LocalEmbedder {
        name: "ollama",
        model: Arc::new(model),
    })
}

/// LM Studio serves embeddings on the OpenAI-compatible `/v1/embeddings`, so the
/// hosted adapter reaches it with only the base URL changed — and the API key
/// requirement switched off, since a local server neither needs nor checks one.
async fn lmstudio_embedder() -> std::result::Result<LocalEmbedder, String> {
    let base_url = env_or("LOCAL_LMSTUDIO_BASE_URL", "http://localhost:1234/v1");

    let model_id = match std::env::var("LOCAL_LMSTUDIO_EMBED_MODEL")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        Some(explicit) => explicit.trim().to_string(),
        None => discover_lmstudio_embedding_model(&base_url).await?,
    };

    let configure = |dimensions: usize| {
        OpenAiEmbeddingModel::new("local")
            .with_base_url(&base_url)
            .with_model(&model_id)
            .with_required_api_key(false)
            // `dimensions` is an OpenAI-specific *request* parameter for
            // Matryoshka-style truncation; llama.cpp-backed servers reject or
            // ignore it, and the width is whatever the GGUF produces regardless.
            .with_send_dimensions(false)
            .with_dimensions(dimensions)
    };

    // Discover the width before declaring one. The adapter defaults to
    // `text-embedding-3-small`'s 1536 and validates every vector against it, so
    // probing with the default would reject a 768-wide local model as a
    // mismatch. Zero disables that check, which is exactly what a discovery
    // probe needs.
    let probe = configure(0)
        .embed(&["probe".to_string()])
        .await
        .map_err(|e| format!("lmstudio embeddings ({model_id}): {e}"))?;
    let width = probe.first().map(Vec::len).unwrap_or(0);
    if width == 0 {
        return Err(format!("lmstudio: `{model_id}` returned an empty vector"));
    }

    Ok(LocalEmbedder {
        name: "lmstudio",
        model: Arc::new(configure(width)),
    })
}

/// Asks LM Studio which models it serves and picks an embedding one.
///
/// There is no default to hard-code: the id is whatever GGUF the operator
/// loaded.
async fn discover_lmstudio_embedding_model(base_url: &str) -> std::result::Result<String, String> {
    let spec = ProviderSpec::for_kind(ProviderKind::LmStudio)
        .with_base_url(base_url)
        .with_model("probe");
    let client = OpenAiModel::from_spec(spec, "local")
        .map_err(|e| format!("lmstudio: invalid configuration: {e}"))?;

    let listed = client
        .list_models()
        .await
        .map_err(|e| format!("lmstudio: not reachable at {base_url} ({e})"))?;

    listed
        .iter()
        .map(|entry| entry.id.clone())
        .find(|id| {
            let id = id.to_ascii_lowercase();
            id.contains("embed") || id.contains("bge")
        })
        .ok_or_else(|| {
            format!(
                "lmstudio: reachable at {base_url} but serves no embedding model \
                 (saw {} id(s)); load one or set LOCAL_LMSTUDIO_EMBED_MODEL",
                listed.len()
            )
        })
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Every local embedding backend that is reachable and usable right now.
async fn reachable_embedders() -> Vec<LocalEmbedder> {
    if std::env::var("LOCAL_MODEL_TESTS")
        .ok()
        .filter(|v| !v.trim().is_empty() && v != "0")
        .is_none()
    {
        eprintln!(
            "skipping live local-embedding tests: set LOCAL_MODEL_TESTS=1 to dial local servers \
             (LOCAL_MODEL_TESTS=1 cargo test --test live_local_embeddings -- --nocapture)"
        );
        return Vec::new();
    }

    let mut ready = Vec::new();
    for probe in [ollama_embedder().await, lmstudio_embedder().await] {
        match probe {
            Ok(embedder) => {
                eprintln!(
                    "local embedder `{}` ready: {} ({} dims)",
                    embedder.name,
                    embedder.model.model_id(),
                    embedder.model.dimensions()
                );
                ready.push(embedder);
            }
            Err(reason) => eprintln!("skipping {reason}"),
        }
    }
    // Opting in explicitly and then reaching nothing must not look like success.
    // Every assertion in this file is inside a `for` over this list, so an empty
    // list makes the whole suite pass while testing nothing at all — the exact
    // failure mode these tests exist to rule out.
    assert!(
        !ready.is_empty(),
        "LOCAL_MODEL_TESTS is set but no local embedding server is reachable. \
         Start Ollama (`ollama serve` + `ollama pull nomic-embed-text`) or LM Studio \
         (`lms server start` + load an embedding model), or unset LOCAL_MODEL_TESTS to skip."
    );
    ready
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The reported dimensionality must match the vectors actually produced, and
/// must not drift between calls.
///
/// This is load-bearing rather than cosmetic: [`InMemoryVectorStore`] fixes its
/// width on the first insert and rejects mismatches, and
/// [`EmbeddingModel::signature`] — which embeds `dimensions()` — is what
/// partitions persisted vectors between embedding spaces. A model whose
/// declared width disagrees with its output silently corrupts both.
#[tokio::test]
async fn local_embedders_report_the_width_they_actually_produce() {
    for embedder in reachable_embedders().await {
        let declared = embedder.model.dimensions();
        assert!(
            declared > 0,
            "{}: dimensions() must be resolved after a successful embed",
            embedder.name
        );

        let vectors = embedder
            .model
            .embed(&["alpha".to_string(), "beta".to_string()])
            .await
            .unwrap_or_else(|e| panic!("{}: embed failed: {e}", embedder.name));

        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(
                vector.len(),
                declared,
                "{}: vector {index} is {} wide but the model declares {declared}",
                embedder.name,
                vector.len()
            );
        }

        // A second call must not renegotiate the width.
        let again = embedder
            .model
            .embed_query("gamma")
            .await
            .unwrap_or_else(|e| panic!("{}: embed_query failed: {e}", embedder.name));
        assert_eq!(
            again.len(),
            declared,
            "{}: the width changed between calls",
            embedder.name
        );
        assert!(
            embedder.model.signature().contains(&declared.to_string()),
            "{}: the signature should pin the dimensionality: {}",
            embedder.name,
            embedder.model.signature()
        );
    }
}

/// A batch must return one vector per input, **in input order**.
///
/// Asserted by comparing each batched vector against the same text embedded on
/// its own. A backend that reorders or drops a row would still return the right
/// *count*, so counting alone cannot catch it — and a silent reorder poisons an
/// index in a way that only shows up later as bad retrieval.
#[tokio::test]
async fn local_embedders_return_one_vector_per_input_in_order() {
    let texts: Vec<String> = [
        "the cat sat on the mat",
        "quarterly revenue exceeded expectations",
        "rust is a systems programming language",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    for embedder in reachable_embedders().await {
        let batched = embedder
            .model
            .embed(&texts)
            .await
            .unwrap_or_else(|e| panic!("{}: batched embed failed: {e}", embedder.name));

        assert_eq!(
            batched.len(),
            texts.len(),
            "{}: expected one vector per input",
            embedder.name
        );

        for (index, text) in texts.iter().enumerate() {
            let alone = embedder
                .model
                .embed_query(text)
                .await
                .unwrap_or_else(|e| panic!("{}: single embed failed: {e}", embedder.name));

            // Not asserted bit-identical: batching can change the arithmetic
            // (padding, batch-size-dependent kernels) by a hair. Alignment is
            // what matters, and a misplaced row scores nowhere near 1.0.
            let alignment = cosine_similarity(&batched[index], &alone);
            assert!(
                alignment > 0.99,
                "{}: batched vector {index} does not match `{text}` embedded alone \
                 (cosine {alignment:.4}) — the batch is misaligned",
                embedder.name
            );
        }
    }
}

/// The vectors must carry meaning: a paraphrase has to sit closer to a sentence
/// than an unrelated sentence does.
///
/// Without this every other assertion here would still pass for a backend that
/// returned constant or random vectors of the right shape.
#[tokio::test]
async fn local_embedders_place_paraphrases_closer_than_unrelated_text() {
    for embedder in reachable_embedders().await {
        let vectors = embedder
            .model
            .embed(&[
                "a small dog is barking loudly in the garden".to_string(),
                "the little puppy is making a lot of noise outside".to_string(),
                "compile times regressed after the dependency upgrade".to_string(),
            ])
            .await
            .unwrap_or_else(|e| panic!("{}: embed failed: {e}", embedder.name));

        let paraphrase = cosine_similarity(&vectors[0], &vectors[1]);
        let unrelated = cosine_similarity(&vectors[0], &vectors[2]);

        assert!(
            paraphrase > unrelated,
            "{}: a paraphrase ({paraphrase:.4}) should score above unrelated text \
             ({unrelated:.4}) — the embedding space carries no meaning",
            embedder.name
        );
    }
}

/// The full retrieval path: index real documents, query with words that appear
/// in **none** of them, and get the topically right document first.
///
/// The query deliberately shares no content words with the target document, so
/// nothing here can be satisfied by lexical overlap or by the exact-text
/// shortcut that makes the mock-backed test tautological.
#[tokio::test]
async fn local_embedders_rank_the_right_document_first() {
    for embedder in reachable_embedders().await {
        let retriever = Retriever::new(
            Arc::clone(&embedder.model),
            Arc::new(InMemoryVectorStore::new()),
        );

        retriever
            .index(vec![
                (
                    "animals".into(),
                    "Cats purr when they are content and knead soft blankets with their paws."
                        .into(),
                    json!({ "topic": "animals" }),
                ),
                (
                    "finance".into(),
                    "The central bank raised interest rates to curb accelerating inflation.".into(),
                    json!({ "topic": "finance" }),
                ),
                (
                    "programming".into(),
                    "Ownership and borrowing let the compiler prove memory safety without a \
                     garbage collector."
                        .into(),
                    json!({ "topic": "programming" }),
                ),
            ])
            .await
            .unwrap_or_else(|e| panic!("{}: indexing failed: {e}", embedder.name));

        for (query, expected) in [
            ("Why do felines make a rumbling sound?", "animals"),
            ("monetary policy and rising prices", "finance"),
            ("how does the language avoid a GC?", "programming"),
        ] {
            let hits = retriever
                .retrieve(query, 3)
                .await
                .unwrap_or_else(|e| panic!("{}: retrieval failed: {e}", embedder.name));

            assert_eq!(
                hits.len(),
                3,
                "{}: expected all 3 documents back",
                embedder.name
            );
            assert_eq!(
                hits[0].id,
                expected,
                "{}: `{query}` should rank `{expected}` first, got `{}` \
                 (scores: {:?})",
                embedder.name,
                hits[0].id,
                hits.iter()
                    .map(|h| (h.id.as_str(), h.score))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                hits[0].metadata["topic"], expected,
                "{}: metadata should survive the round trip",
                embedder.name
            );
        }
    }
}

/// Blank input is handled **differently by the two adapters**, and this pins
/// both so the divergence is visible rather than discovered in production.
///
/// A corpus routinely contains empty documents, and the danger is a backend
/// that silently *drops* them: the remaining vectors shift up by one, every id
/// after the blank binds to the wrong vector, and the index is permanently
/// corrupted in a way that only surfaces later as bad retrieval. Neither
/// adapter does that — but they avoid it in opposite ways:
///
/// - [`OllamaEmbeddingModel`] returns an **empty vector per blank input**,
///   preserving positions, and never dials the server.
/// - [`OpenAiEmbeddingModel`] **rejects the batch** with a validation error
///   naming the offending index, because the hosted OpenAI endpoint rejects
///   empty strings outright.
///
/// Both are position-safe. They are not, however, interchangeable: code that
/// indexes a corpus containing blanks works against Ollama and fails against
/// any OpenAI-compatible endpoint, including a local LM Studio. Callers must
/// filter blanks themselves rather than rely on either behaviour.
#[tokio::test]
async fn blank_input_is_position_safe_on_both_adapters() {
    let blanks = ["   ".to_string(), "\n".to_string()];

    for embedder in reachable_embedders().await {
        match embedder.model.embed(&blanks).await {
            Ok(vectors) => {
                assert_eq!(
                    embedder.name, "ollama",
                    "only the Ollama adapter is expected to accept an all-blank batch"
                );
                assert_eq!(
                    vectors.len(),
                    blanks.len(),
                    "ollama: an all-blank batch must still return one slot per input"
                );
                assert!(
                    vectors.iter().all(Vec::is_empty),
                    "ollama: blank inputs should yield empty vectors, not fabricated ones"
                );
            }
            Err(error) => {
                let error = error.to_string();
                assert_eq!(
                    embedder.name, "lmstudio",
                    "only the OpenAI-compatible adapter is expected to reject blanks"
                );
                // Rejecting is acceptable; rejecting without saying which input
                // is at fault would leave the caller unable to fix their corpus.
                assert!(
                    error.contains("index 0"),
                    "lmstudio: the rejection should name the offending index, got: {error}"
                );
            }
        }
    }
}

/// A missing model must fail with a message that says how to fix it.
///
/// "model not found" is the single most common local-embedding failure — the
/// operator simply has not pulled it — so the error is expected to name the
/// remedy rather than surface a bare 404.
#[tokio::test]
async fn ollama_reports_a_missing_embedding_model_with_remediation() {
    if reachable_embedders()
        .await
        .iter()
        .all(|embedder| embedder.name != "ollama")
    {
        return;
    }

    let base_url = env_or("LOCAL_OLLAMA_EMBED_URL", "http://localhost:11434");
    let missing = OllamaEmbeddingModel::new(&base_url, "definitely-not-installed-abc123", 768);

    let error = missing
        .embed(&["hello".to_string()])
        .await
        .expect_err("an uninstalled model must not silently succeed")
        .to_string();

    assert!(
        error.contains("ollama pull definitely-not-installed-abc123"),
        "the error should tell the operator how to install it, got: {error}"
    );
}

// ---------------------------------------------------------------------------
// Offline unit coverage
//
// These run on every `cargo test` with no network, pinning the configuration
// rules that the live tests above depend on.
// ---------------------------------------------------------------------------

/// [`OllamaEmbeddingModel`] takes the **server root**, not an API endpoint.
///
/// This is easy to get wrong because the *chat* side of the same server is
/// configured with a `/v1` suffix, and Ollama's embedding API is not under
/// `/v1` at all. Pointing the adapter at `/v1` or `/api` yields a 404 on every
/// call, so it is rejected at construction with a message naming the mistake.
#[test]
fn the_ollama_embedding_adapter_rejects_an_endpoint_url() {
    for bad in [
        "http://localhost:11434/v1",
        "http://localhost:11434/api",
        "http://localhost:11434/v1/embeddings",
    ] {
        let error = OllamaEmbeddingModel::try_new(bad, "nomic-embed-text", 768)
            .expect_err(&format!("{bad} should be rejected"))
            .to_string();
        assert!(
            error.contains("server root"),
            "{bad} should be rejected with a message naming the fix, got: {error}"
        );
    }

    assert!(
        OllamaEmbeddingModel::try_new("http://localhost:11434", "nomic-embed-text", 768).is_ok()
    );
}

/// The OpenAI embedding adapter must be usable against a local server: no
/// credential, and no OpenAI-specific `dimensions` parameter on the wire.
#[test]
fn the_openai_embedding_adapter_can_be_pointed_at_a_local_server() {
    let model = OpenAiEmbeddingModel::new("")
        .with_base_url("http://localhost:1234/v1")
        .with_model("text-embedding-nomic-embed-text-v1.5")
        .with_required_api_key(false)
        .with_send_dimensions(false)
        .with_dimensions(768);

    assert_eq!(model.dimensions(), 768);
    assert_eq!(model.model_id(), "text-embedding-nomic-embed-text-v1.5");
    // The signature partitions persisted vectors, so a local model must not
    // collide with a hosted OpenAI one of the same width.
    assert!(
        model
            .signature()
            .contains("text-embedding-nomic-embed-text-v1.5")
    );
}

/// Blank-only batches short-circuit without dialling, so the positional
/// guarantee holds even with no server running.
#[tokio::test]
async fn blank_batches_are_answered_without_a_server() {
    let model = OllamaEmbeddingModel::new("http://127.0.0.1:9", "nomic-embed-text", 768);
    let vectors = model
        .embed(&["".to_string(), "  ".to_string()])
        .await
        .expect("an all-blank batch never reaches the network");
    assert_eq!(vectors, vec![Vec::<f32>::new(), Vec::new()]);
}
