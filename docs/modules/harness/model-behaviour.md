# ModelProfile Behaviour, Deferred Responses, and Request Hooks

New `tinyinference-llm` capabilities that let an adapter's behaviour — not just its advertised capabilities — be driven by data, plus a way to resolve an async/batch call and to hook the OpenAI/Anthropic transport.

### `ModelProfile` as behaviour (implemented)

`tinyinference_llm::model::ModelProfile`
(`vendor/tinyinference/crates/tinyinference-llm/src/model/types.rs`) carries
fields that let an adapter's *behaviour* — not just its advertised
capabilities (see [model.md](model.md)) — be driven by data:

- `schema_transform: Option<SchemaTransform>` — a serializable, named JSON-schema
  transform (`StripDefs`, `InlineRefs`, `NoAdditionalProperties`, `GeminiCompat`,
  `OpenAiStrict`, or `Chain(Vec<SchemaTransform>)`) with a real `apply(&self,
  &Value) -> Value`. `StripDefs`/`InlineRefs` resolve `$ref`s into `$defs`;
  `NoAdditionalProperties` recursively forces `additionalProperties: false`;
  `GeminiCompat` strips keywords Gemini's schema dialect rejects;
  `OpenAiStrict` composes inlining + `additionalProperties: false` + requiring
  every property, matching OpenAI's strict JSON-schema mode.
- `default_structured_mode: Option<StructuredMode>` (`Tool` / `Native` /
  `Prompted`) and `prompted_output_template: Option<String>` for the prompted
  fallback.
- `thinking_tags: Option<(String, String)>` — an open/close tag pair (for
  example `("<think>", "</think>")`) a model wraps chain-of-thought in.
- `ignore_streamed_leading_whitespace: bool` and
  `thinking_level_map: BTreeMap<String, ReasoningConfig>` (a named reasoning
  level, e.g. `"low"`, mapped to the `ReasoningConfig` it expands to).
- `compat: ProviderCompat` — provider-family quirks that do not fit the
  capability model: `mid_conversation_system_messages`, `strict_tools`,
  `cache_retention`, `session_affinity`, `max_tool_name_length`,
  `tool_id_pattern`.

All fields are additive and `serde(default)`, so existing serialized profiles
deserialize unchanged. See `model/mod_tests.rs` in the vendor crate for the
transform and round-trip tests. Consuming these fields from harness
`tool/schema_prepare.rs`, `structured/`, and `agent_loop/model_call.rs` is not
yet wired up — the data model exists but the harness does not yet read it.

### Deferred (async/batch) responses (implemented)

`ModelStreamItem::Deferred(DeferredHandle)` is a new terminal stream item for
a provider that accepts a request but finishes it asynchronously (an OpenAI
batch or background response). `DeferredHandle { provider, id, kind,
metadata }` is a serializable, provider-neutral handle a host can persist and
resume polling with. `ChatModel::fetch_deferred(&self, &DeferredHandle) ->
Result<DeferredStatus>` (`DeferredStatus::{Pending, Completed(Box<ModelResponse>),
Failed(String)}`) resolves it later; the default implementation returns
`Error::Unsupported`, so only adapters that can actually defer need to
override it. No provider adapter overrides it yet (OpenAI batch/background
mapping was scoped out of this pass) — the shape exists for a future adapter
to fill in. `StreamAccumulator::deferred()` surfaces a folded-in handle, and
`StreamAccumulator::finish()` returns `Error::Unsupported` rather than
silently discarding it when one was seen.

### `ProviderRequestOptions` (implemented)

`tinyinference_llm::providers::ProviderRequestOptions { on_payload, on_response,
http }` lets a host hook the OpenAI and Anthropic adapters' Chat
Completions/Messages transport without the harness knowing about the hook:
`on_payload: Option<Arc<dyn Fn(&mut Value) + Send + Sync>>` mutates the wire
body immediately before it is sent, `on_response: Option<Arc<dyn Fn(&Value) +
Send + Sync>>` observes the raw response JSON after a successful call (an
OpenAI `invoke` fulfilled by streaming, as `local-openai` models do, passes a
Chat Completions body assembled from the stream instead, which parses back to
the same answer), and `http: Option<reqwest::Client>` overrides the adapter's
own client. Set it at
construction with `AnthropicModel::with_request_options` /
`OpenAiModel::with_request_options`. On the OpenAI adapter it currently covers
the Chat Completions path only (not the Responses/Codex path).
