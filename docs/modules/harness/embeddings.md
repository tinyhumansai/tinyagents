# Harness Embeddings And Retrieval Feature

Embeddings are provider-neutral vector representations used for retrieval,
semantic search, deduplication, document compression, reranking, and
retrieval-augmented prompt context. This feature owns embedding models, vector
stores, retrievers, indexing records, retrieval events, and deterministic test
utilities.

## Source Inspiration

LangChain keeps embeddings and retrieval as core primitives rather than
embedding them inside the chat model abstraction:

- embedding interface:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/embeddings/embeddings.py>
- fake and deterministic fake embeddings:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/embeddings/fake.py>
- vector store interface and in-memory vector store:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/base.py>
  and
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/in_memory.py>
- retriever interface:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/retrievers.py>
- indexing API and record managers:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/indexing>
- vector math utilities for cosine similarity and maximal marginal relevance:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/utils.py>
- partner vector stores and embedding providers such as Chroma, Qdrant, OpenAI,
  Ollama, and others:
  <https://github.com/langchain-ai/langchain/tree/master/libs/partners>

RustAgents should follow the separation: chat models generate messages,
embedding models generate vectors, vector stores search vectors, retrievers
return documents, and prompt/middleware code decides what retrieved context
enters a model request.

## Responsibilities

- Define provider-neutral embedding traits.
- Register named embedding providers.
- Embed documents and queries separately.
- Support batch embedding and async-native embedding.
- Track vector dimensionality and distance metric.
- Normalize provider metadata, usage, cost, and errors.
- Cache embeddings when policy allows.
- Store and query dense vectors.
- Support sparse vectors and hybrid retrieval where backends support them.
- Support similarity search, score-threshold search, and maximal marginal
  relevance search.
- Support metadata filters.
- Support add, update, delete, get-by-id, and search operations.
- Support incremental indexing with content hashes and record managers.
- Expose retrievers to middleware, tools, and prompt assembly.
- Emit embedding, indexing, vector-store, and retriever events.
- Provide deterministic fake embeddings and in-memory vector stores for tests.

## Non-Responsibilities

- It does not decide graph routing.
- It does not automatically inject retrieved documents into every prompt.
- It does not replace durable application stores.
- It does not require a vector store for every embedding call.
- It does not assume query embeddings and document embeddings are identical.
- It does not hide backend-specific vector-store capabilities when users need
  them.

## Package Shape

Target layout:

```text
src/harness/embeddings.rs
```

The first implementation can keep the feature in one module. If it grows large,
split it into:

```text
src/harness/embeddings/
  mod.rs
  embedding.rs
  vector_store.rs
  retriever.rs
  indexing.rs
  sparse.rs
  testkit.rs
```

## Core Types

```rust
#[async_trait]
pub trait EmbeddingModel<Ctx = ()>: Send + Sync {
    fn profile(&self) -> Option<&EmbeddingProfile>;

    async fn embed_documents(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: EmbedDocumentsRequest,
    ) -> Result<EmbedDocumentsResponse>;

    async fn embed_query(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: EmbedQueryRequest,
    ) -> Result<EmbedQueryResponse>;
}

pub struct EmbeddingRegistry<Ctx = ()> {
    models: HashMap<EmbeddingName, Arc<dyn EmbeddingModel<Ctx>>>,
    default: Option<EmbeddingName>,
}
```

The trait separates document embeddings from query embeddings because some
providers optimize them differently. Even when the current provider uses the
same endpoint for both, the public contract should preserve the distinction.

## Requests And Responses

```rust
pub struct EmbedDocumentsRequest {
    pub model: EmbeddingName,
    pub texts: Vec<String>,
    pub input_type: EmbeddingInputType,
    pub dimensions: Option<usize>,
    pub provider_options: serde_json::Value,
    pub cache_policy: Option<CachePolicy>,
    pub metadata: serde_json::Value,
}

pub struct EmbedDocumentsResponse {
    pub vectors: Vec<EmbeddingVector>,
    pub usage: Option<UsageRecord>,
    pub provider: ProviderMetadata,
    pub cache: Option<CacheDecision>,
}

pub struct EmbeddingVector {
    pub values: Vec<f32>,
    pub dimension: usize,
    pub norm: Option<f32>,
}
```

`Vec<f32>` should be the default in-memory representation. Provider adapters may
receive or store `f64`, quantized, binary, or backend-native vectors, but the
harness should normalize the common path to a predictable Rust type.

## Embedding Profiles

```rust
pub struct EmbeddingProfile {
    pub provider: ProviderName,
    pub model: EmbeddingName,
    pub dimensions: Option<usize>,
    pub allowed_dimensions: Vec<usize>,
    pub max_batch_size: Option<usize>,
    pub max_input_tokens: Option<usize>,
    pub supports_documents: bool,
    pub supports_queries: bool,
    pub supports_sparse: bool,
    pub supports_image_input: bool,
    pub distance_metrics: Vec<DistanceMetric>,
    pub provider_extras: serde_json::Value,
}
```

Profiles are used to:

- validate requested dimensions
- choose batch sizes
- enforce provider input limits
- choose dense, sparse, or hybrid search paths
- report retriever provenance in events
- reject incompatible vector-store/index configurations early

## Vector Stores

```rust
#[async_trait]
pub trait VectorStore<Ctx = ()>: Send + Sync {
    async fn add_documents(
        &self,
        ctx: &mut RunContext<Ctx>,
        documents: Vec<IndexedDocument>,
        options: AddDocumentsOptions,
    ) -> Result<Vec<DocumentId>>;

    async fn delete(&self, ctx: &mut RunContext<Ctx>, ids: Vec<DocumentId>) -> Result<DeleteResult>;

    async fn get_by_ids(
        &self,
        ctx: &mut RunContext<Ctx>,
        ids: Vec<DocumentId>,
    ) -> Result<Vec<IndexedDocument>>;

    async fn search(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: VectorSearchRequest,
    ) -> Result<Vec<ScoredDocument>>;
}
```

Search types:

- `similarity`: return nearest documents
- `similarity_with_score`: include backend score or normalized relevance score
- `similarity_score_threshold`: filter results below a configured relevance
  threshold
- `mmr`: maximal marginal relevance, balancing query similarity and diversity
- `by_vector`: search with a caller-provided vector
- `hybrid`: combine dense and sparse retrieval where supported

The store must document score semantics. Some backends return distance where
lower is better; others return similarity where higher is better. RustAgents
should normalize relevance scores when possible and preserve raw backend scores
for auditability.

## Retrievers

Retrievers are query-to-document components. They are more general than vector
stores because they can wrap vector stores, keyword search, hybrid search,
rerankers, document compressors, or application-specific logic.

```rust
#[async_trait]
pub trait Retriever<Ctx = ()>: Send + Sync {
    async fn retrieve(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: RetrievalRequest,
    ) -> Result<Vec<ScoredDocument>>;
}
```

Retriever requests should carry:

- query text
- optional query vector
- search type
- `k`
- `fetch_k`
- score threshold
- MMR lambda
- metadata filter
- tags and metadata

Retrievers should emit `retriever.started`, `retriever.completed`, and
`retriever.failed` events. Events should include retriever name, embedding
provider/model, vector-store provider, search type, result count, timings, and
redacted query metadata.

## Indexed Documents

```rust
pub struct IndexedDocument {
    pub id: Option<DocumentId>,
    pub text: String,
    pub metadata: serde_json::Value,
    pub source_id: Option<String>,
    pub embedding: Option<EmbeddingVector>,
}
```

Document ids matter for deletion, updates, deduplication, and provenance. When
ids are not supplied, stores may generate ids, but indexing workflows should
prefer deterministic ids derived from source identity or content hashes.

## Indexing And Record Managers

LangChain's indexing layer hashes document content and metadata, stores record
manager entries, and avoids re-indexing unchanged documents. RustAgents should
make that pattern explicit.

```rust
#[async_trait]
pub trait RecordManager: Send + Sync {
    async fn get_time(&self) -> Result<SystemTime>;
    async fn update(&self, records: Vec<IndexRecord>) -> Result<()>;
    async fn exists(&self, keys: Vec<IndexKey>) -> Result<Vec<bool>>;
    async fn list_keys(&self, filter: RecordFilter) -> Result<Vec<IndexKey>>;
    async fn delete_keys(&self, keys: Vec<IndexKey>) -> Result<()>;
}

pub struct IndexPolicy {
    pub key_encoder: KeyEncoder,
    pub cleanup: CleanupPolicy,
    pub batch_size: usize,
    pub force_update: bool,
}
```

Indexing should support:

- deterministic content and metadata hashing
- configurable hash algorithms
- source id grouping
- deduplication while preserving order
- incremental indexing
- full cleanup
- scoped cleanup by source id
- monotonic server time checks where the backend supports them
- failure handling when vector-store writes and record-manager writes diverge

SHA-1-style defaults should be documented as compatibility-oriented rather than
collision-resistant. New RustAgents implementations should prefer `sha256`,
`sha512`, `blake3`, or caller-provided encoders.

## Sparse And Hybrid Retrieval

Qdrant and other backends support sparse vectors and hybrid dense+sparse
retrieval. RustAgents should not force sparse support into the dense embedding
trait. Use separate types:

```rust
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[async_trait]
pub trait SparseEmbeddingModel<Ctx = ()>: Send + Sync {
    async fn embed_sparse_documents(
        &self,
        ctx: &mut RunContext<Ctx>,
        texts: Vec<String>,
    ) -> Result<Vec<SparseVector>>;

    async fn embed_sparse_query(
        &self,
        ctx: &mut RunContext<Ctx>,
        text: String,
    ) -> Result<SparseVector>;
}
```

Hybrid search should record the dense model, sparse model, fusion strategy, and
backend-specific scoring metadata.

## Caching

Embedding cache keys should include:

- provider
- model
- input text
- input type: document or query
- requested dimensions
- provider options
- preprocessing version
- tokenizer/chunker version when applicable

Cached embeddings should record vector dimension, provider metadata, and
creation time. Cache hits should still emit usage/cache events, but should not
pretend provider tokens were consumed.

## Usage And Cost

Embedding providers may bill by input tokens, characters, requests, or batches.
Usage records should support:

- input tokens
- input characters
- number of texts
- number of vectors
- dimensions
- batch count
- provider-specific extras

Cost records should support per-token, per-character, per-vector, and per-request
pricing. Pricing data must be updateable outside provider adapters.

## Events

Event kinds:

- `embedding.started`
- `embedding.completed`
- `embedding.failed`
- `vector_store.added`
- `vector_store.deleted`
- `vector_store.searched`
- `retriever.started`
- `retriever.completed`
- `retriever.failed`
- `indexing.started`
- `indexing.batch_completed`
- `indexing.completed`
- `indexing.failed`

Events should include run ids, component ids, provider/model names, vector-store
namespace, batch sizes, result counts, timing, usage, cost, and redacted query or
document fingerprints.

## Retrieval Context Assembly

Retrievers return candidate documents. Prompt and middleware code decide how to
use them. A retrieval middleware may:

- run one or more retrievers before a model call
- deduplicate documents by source id or content hash
- rerank or compress results
- drop documents when context pressure is high
- summarize large documents
- attach citation metadata to context blocks
- emit provenance events for every injected context block

The harness must preserve provenance so model outputs can cite retrieved
sources, and so tests can assert which documents entered the prompt.

## Built-In Implementations

Initial implementations:

- `DeterministicFakeEmbedding` for tests
- `RandomFakeEmbedding` for shape tests only
- `InMemoryVectorStore` for examples and unit tests
- `InMemoryRecordManager` for indexing tests

Feature-gated future adapters:

- `provider-openai-embeddings`
- `provider-ollama-embeddings`
- `vector-chroma`
- `vector-qdrant`
- `vector-pgvector`
- `vector-mongodb-atlas`
- `vector-redis`

## Conformance Tests

Embedding providers should pass tests for:

- document embedding count matches input count
- query embedding returns one vector
- dimensions match profile or requested dimension
- empty input behavior is documented
- Unicode input
- batch sizing
- async behavior
- cache hit/miss behavior
- usage and cost records
- cancellation and timeout
- provider error classification

Vector stores should pass tests for:

- add documents
- add texts
- generated ids
- caller-supplied ids
- delete by id
- get by ids without assuming return order
- similarity search
- similarity search with score
- score-threshold search
- MMR search
- metadata filters
- update replaces vector when text changes
- persistence behavior where supported

Indexing should pass tests for:

- deterministic hashes
- metadata hash changes
- deduplication
- incremental no-op on unchanged documents
- cleanup of stale documents
- record-manager/vector-store failure handling
