# `tool::discover`

On-demand tool discovery for the agent loop. The design doc is
[`docs/modules/harness/tool-discovery.md`](../../../../../docs/modules/harness/tool-discovery.md);
this file is the map of the module.

| File          | Owns                                                                 |
|---------------|----------------------------------------------------------------------|
| `types.rs`    | `ToolDiscoveryPolicy` (the knobs, the host `ranker` and `DiscoveryRankMode`), `DiscoveryOverrides` (the per-session overrides), `DeferredCatalog` (a run's deferred schemas, BM25-indexed, name-sorted, ranked through the policy), `DeferredTool`, `RankedSearch` |
| `manifest.rs` | `render_manifest` — the budgeted listing inside `tool_search`'s description: full → names → count |
| `bridge.rs`   | The one intrinsic tool, `tool_search`: `bridge_schemas`, `answer_tool_search` (async; returns a `SearchAnswer`) |
| `embedding_ranker.rs` | `EmbeddingToolRanker` — a semantic `tinytools::ToolRanker` over any `EmbeddingModel`, cached in memory and optionally through `EmbeddingCache` or disk by embedding-space signature |
| `mod_tests.rs` | Unit tests for all of the above                                      |

The agent loop (`agent_loop/run_loop.rs`, `agent_loop/tools.rs`) is the only
consumer: it builds the catalogue once per run, appends the `tool_search`
schema after the name-sorted direct set, and answers `tool_search` from the
catalogue without running a tool. There is no call wrapper: a deferred tool is
called by its own name, so every `before_tool` gate sees the real tool as is.

Invariants worth keeping:

- By default a search promotes its matches into `tools` (one cache-prefix
  change per newly found tool). With `promote_found_tools: false` the `tools`
  array never changes within a run because of discovery: revealed schemas
  travel in a tool result only (cache stability). Both that and
  `compact_search_results` are per-session options, read as
  `ToolDiscoveryPolicy::with_overrides(&RunConfig::discovery)`.
- Everything rendered from the catalogue is name-sorted and deterministic.
- Discovery only subtracts: a deferred tool is callable by name whether or not
  the bridge is enabled; a `Hidden` tool is never callable by the model.
- The manifest is bounded by `manifest_token_budget`; the search answer clips
  descriptions to 500 chars and `limit` to `max_limit`.
- A search never fails. The host ranker (`ToolDiscoveryPolicy::ranker`,
  any `tinytools::ToolRanker`) is served when active; on error or an empty
  answer BM25 answers instead and `RankedSearch::fallback` says why.
  `DiscoveryRankMode::Compare` serves the host ranker and carries the BM25
  ranking alongside for comparison. BM25 itself lives in `tinytools::rank`.

## Persisting catalogue embeddings

`EmbeddingToolRanker::with_cache(Arc<dyn EmbeddingCache>)` accepts a host's
async cache handle. The ranker requests missing summary/family hashes under its
embedding-space signature and stores only newly computed catalogue vectors.
Hosts bind that handle to the tenant, credentials and backend; create a new
ranker when those change because its in-memory vectors belong to that handle.

Cache implementations merge deltas rather than replace unrelated entries.
Unreadable caches, failed writes and malformed vectors are rebuildable misses.
The ranker never persists search intent vectors and never switches a failed
host cache to a filesystem cache. with_disk_cache remains available for local
hosts. Selecting either cache replaces a previously selected cache.
