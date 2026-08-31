# Local Models and Embeddings

Running against a local runtime — Ollama or LM Studio — is not the same as
running against a hosted provider with a different base URL. The wire format is
identical, but small quantised models and llama.cpp-backed servers fail in ways
the hosted APIs never do, and a host that ignores those differences gets a loop
that works in development and stalls in production.

This page documents what actually differs, what the crate already handles, and
what a caller still has to configure. Everything here is asserted by
[`tests/live_local_models.rs`](../../../crates/tinyagents-integration-tests/tests/live_local_models.rs)
and
[`tests/live_local_embeddings.rs`](../../../crates/tinyagents-integration-tests/tests/live_local_embeddings.rs)
against real servers.

## Presets

| Preset | Default base URL | Default model | Credential |
| ------ | ---------------- | ------------- | ---------- |
| `ProviderKind::Ollama` | `http://localhost:11434/v1` | `llama3.2` | none |
| `ProviderKind::LmStudio` | `http://localhost:1234/v1` | **none — you must set one** | none |

Both resolve through `OpenAiModel::from_spec`, which routes local kinds onto the
local-runtime path: `AuthStyle::None` (no `Authorization` header at all, which
some servers reject), base-URL normalisation to the `/v1` root, and the
request-shape degradations described below.

Reaching LM Studio through a bare `PROVIDER_LMSTUDIO_BASE_URL` instead of the
preset resolves to `ProviderKind::Compatible` and gets **none** of that. Use the
preset.

LM Studio has no default model on purpose: the served id is whatever GGUF the
operator loaded, so any guess would 404 on most installs. Discover it at runtime:

```rust
let spec = ProviderSpec::for_kind(ProviderKind::LmStudio).with_model("probe");
let ids = OpenAiModel::from_spec(spec, "local")?.list_models().await?;
```

## What breaks, and where it is handled

### Request shapes local servers reject

A named `tool_choice` object and `response_format: {"type": "json_object"}` both
draw an HTTP 400 from llama.cpp-backed servers. The transport degrades them to
shapes those servers accept — `tool_choice: "required"` with the `tools` array
filtered to the named tool, and a permissive `json_schema` — either eagerly for
local presets or as a single retry when a 400 body implicates the shape.

### Tool arguments that are not the arguments

Small models frequently send something other than a bare arguments object. All
of these are real captures from `llama3.2:3b` for a tool declaring one required
`city` string:

```text
{"type":"object","required":["city"],"properties":{"city":"Paris"}}   # schema echo
{"properties":{…},"required":[…],"arguments":{"city":"Paris"}}        # nested under `arguments`
{"param":{"city":"Paris"}}                                            # invented wrapper
```

`normalize_tool_arguments` unwraps a single envelope level for a known set of
wrapper keys, but only when the outer object is already schema-invalid, the tool
does not itself declare an argument of that name, and the unwrapped value
validates. Failing any of those, the original arguments survive so the model
sees a precise error rather than a rewritten one.

This only runs under a recovering `InvalidArgsPolicy` — see below.

### Tool calls emitted as text

Roughly one response in a dozen, `llama3.2:3b` under `tool_choice: "required"`
puts the call in `content` instead of the wire's `tool_calls` array, with no
`<tool_call>` markup and often malformed JSON:

```text
{"name":"get_weather","parameters':{'city':"Paris"}}
```

Left alone this is catastrophic rather than merely lossy: the loop sees an
assistant message with no tool calls, treats it as the final answer, and returns
JSON-looking prose to the user while the tool never runs.
`apply_prompt_tool_calls` recovers it, requiring the **entire** message content
to parse as one object naming a tool so prose that merely quotes JSON is never
swallowed. Mismatched and single-quoted *keys* are repaired; single-quoted
*values* deliberately are not, because an apostrophe in a value is ordinary
English.

### Invalid arguments abort the run by default

`RunPolicy::invalid_args` defaults to `InvalidArgsPolicy::Fail`: the first
schema-invalid tool call kills the whole run. That is defensible for a frontier
model, where such a call is nearly always a genuine bug. For a 3B model it makes
the loop unusable — and it disables the argument recovery above, which only runs
under the recovering policy.

**A host driving a local model should opt in:**

```rust
harness.with_policy(RunPolicy {
    invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
    ..RunPolicy::default()
});
```

Recovery still consumes a tool-call budget slot, so `RunLimits::max_tool_calls`
bounds any repair loop.

### Reasoning models and the token budget

`qwen3` (both runtimes) spends tokens on a hidden reasoning channel before
emitting a single visible character, and that channel draws from the same
`max_tokens` budget. Asking `qwen3-4b` for one word with `max_tokens: 32`
returns `finish_reason: "length"`, 30 reasoning tokens, and **empty** visible
content.

Budget for the reasoning channel — the live tests use 1024 tokens for prompts
whose answers are a few words. `RunPolicy::empty_response_retries` exists for the
residual stochastic case where a model burns the whole budget anyway.

Inline `<think>` blocks are extracted into `ContentBlock::Thinking` rather than
leaking into the visible answer; see `ReasoningTagExtraction`.

### Tool choice is a request, not a guarantee

`ToolChoice::Required` is honoured by Ollama roughly eleven times in twelve for
`llama3.2:3b`; the remainder emit the call as text (recovered as above) or
answer directly. Any host that depends on a tool actually running must check,
not assume.

## Embeddings

The two runtimes are reached by **different adapters**, because Ollama does not
serve embeddings on its OpenAI-compatible surface:

| Runtime | Adapter | Endpoint | Base URL |
| ------- | ------- | -------- | -------- |
| Ollama | `OllamaEmbeddingModel` | `POST /api/embed` | server **root**, e.g. `http://localhost:11434` |
| LM Studio | `OpenAiEmbeddingModel` | `POST /v1/embeddings` | `http://localhost:1234/v1` |

`OllamaEmbeddingModel` rejects a base URL carrying a `/v1` or `/api` suffix at
construction, because the chat side of the *same server* is configured with
`/v1` and copying it there yields a 404 on every call.

### Discover the width; do not declare it

Vector width is a property of the installed GGUF. `nomic-embed-text` is 768 wide
while `OllamaEmbeddingModel`'s own default (`bge-m3`) is 1024, and a declared
width that disagrees with the model fails every call on dimension validation.

- Ollama: `OllamaEmbeddingModel::embed_discovering_dimensions(...)`. Passing `0`
  to `try_new` does **not** mean "discover" — it means "use the default".
- OpenAI-compatible: construct `with_dimensions(0)` to disable validation, probe,
  then rebuild with the observed width. Also set `with_send_dimensions(false)`:
  `dimensions` is an OpenAI request parameter for Matryoshka truncation that
  llama.cpp-backed servers reject or ignore.

Width matters beyond the immediate call: `InMemoryVectorStore` fixes its width on
first insert, and `EmbeddingModel::signature()` embeds the width to partition
persisted vectors between embedding spaces.

### Blank inputs diverge between adapters

Both adapters are position-safe — neither silently drops a blank and shifts every
later vector onto the wrong id — but they achieve it differently:

- `OllamaEmbeddingModel` returns an empty vector per blank input, preserving
  positions, without dialling the server.
- `OpenAiEmbeddingModel` rejects the whole batch with a validation error naming
  the offending index.

They are therefore **not** interchangeable: code that indexes a corpus containing
blanks works against Ollama and fails against any OpenAI-compatible endpoint,
LM Studio included. Filter blanks in the caller rather than relying on either.

## Running the tests

```bash
# Ollama
ollama serve &
ollama pull llama3.2:3b && ollama pull nomic-embed-text

# LM Studio
lms server start
lms load qwen/qwen3-4b && lms load text-embedding-nomic-embed-text-v1.5

LOCAL_MODEL_TESTS=1 cargo test --test live_local_models --test live_local_embeddings -- --nocapture
```

Both files skip entirely without `LOCAL_MODEL_TESTS=1`, and skip any individual
runtime that is not listening. With the variable set and **nothing** reachable
they fail rather than pass, because every assertion lives inside a loop over the
reachable runtimes — an empty list would otherwise be a green run that tested
nothing.

Per-runtime overrides: `LOCAL_OLLAMA_BASE_URL`, `LOCAL_OLLAMA_MODEL`,
`LOCAL_OLLAMA_EMBED_URL`, `LOCAL_OLLAMA_EMBED_MODEL`, `LOCAL_LMSTUDIO_BASE_URL`,
`LOCAL_LMSTUDIO_MODEL`, `LOCAL_LMSTUDIO_EMBED_MODEL`. Models are otherwise
discovered from each server's own `/v1/models`.
