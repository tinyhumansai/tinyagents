# harness::providers::openai

Real OpenAI Chat Completions provider (feature `openai`). This is one of the
concrete leaves the recursive runtime bottoms out in: a single `OpenAiModel`
backs hosted OpenAI *and* every OpenAI-compatible endpoint (Anthropic
compatibility, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together, Mistral) via
the preset constructors, so the sub-agent / sub-graph layers above never need
to know which provider actually answered.

## What it does

`OpenAiModel` implements `ChatModel` against `POST {base_url}/chat/completions`.
It:

1. Translates the provider-neutral `ModelRequest` into OpenAI's JSON wire
   format (message content blocks, tool schemas, tool choice, response format,
   image content parts).
2. Performs the HTTP call with a shared, reusable `reqwest::Client` (unary or
   streamed).
3. Maps the response back into a `ModelResponse` with a fully populated
   `AssistantMessage`, `ToolCall`s, `Usage`, and finish reason — or, for
   streaming, a `ModelStream` of `ModelStreamItem`s.

The wire (de)serialization shapes live in `types.rs`; `transport.rs`,
`convert.rs`, and `sse.rs` own the HTTP transport, request/response
translation, and SSE decoding respectively, keeping OpenAI-specific JSON out
of the rest of the harness.

## Construction

- `OpenAiModel::new(api_key)` — bare constructor, hosted OpenAI base URL and
  `DEFAULT_MODEL` (`gpt-4.1-mini`).
- `.with_model(..)` / `.with_provider(..)` / `.with_base_url(..)` — builder
  overrides.
- `OpenAiModel::from_env()` — reads `OPENAI_API_KEY` (required) and optional
  `OPENAI_MODEL` / `OPENAI_BASE_URL`.
- `OpenAiModel::from_spec(spec, api_key)` / `from_spec_env(spec)` — build from a
  `providers::ProviderSpec` (base URL, default model, provider id already
  resolved).
- **Compatibility presets** — thin wrappers over `new` + `with_base_url` +
  `with_model` for endpoints that speak the same Chat Completions wire format:
  `compatible(base_url, model)` / `compatible_provider(..)` (arbitrary
  endpoint), `deepseek`, `anthropic` (compat endpoint, not the native
  Anthropic API), `groq`, `xai`, `openrouter`, `together`, `mistral`, `ollama`.
  Override the preset's default model with `.with_model(..)`.

Accessors: `.model()`, `.provider()`, `.base_url()`.

## Model discovery

`list_models()` calls `GET {base_url}/models` with the same credentials as
chat calls. Every OpenAI-compatible endpoint serves the same shape, so this
doubles as runtime model discovery for local/self-hosted providers (Ollama,
Together, Groq, OpenRouter, ...); returned ids can be fed straight into
`.with_model(..)`.

## Streaming (SSE)

Streaming responses are decoded by a small state machine (`SseState` /
`OpenAiStreamAcc` in `sse.rs`) built on `futures::stream::unfold`:

- Bytes are accumulated across chunk boundaries and only lossily decoded once
  a complete line is available, so a multi-byte UTF-8 character split across
  two HTTP chunks is never corrupted into replacement characters.
- Index-less streamed tool-call fragments (some providers omit the tool-call
  index on continuation chunks) are correlated by id rather than position.
- Parallel tool-call fragments that all arrive with `index: 0` (Ollama's `/v1`
  bug, ollama/ollama#15457) are kept distinct: when an explicit index carries a
  non-empty id that conflicts with the id already recorded at that slot, a fresh
  slot is opened instead of merging two calls onto one.
- A later fragment with an empty `function.name` never overwrites a name already
  recorded for the slot (LM Studio, lmstudio-bug-tracker#649).
- A trailing partial line without a final newline (providers that terminate
  the last SSE event without a trailing newline) is still flushed.
- Mid-stream error payloads (`data: {"error": ...}`) surface as a stream error
  instead of being silently swallowed.

## Local-model tolerance (tool calls)

Small local servers (Ollama, LM Studio, llama.cpp, vLLM) routinely violate the
OpenAI tool-call contract. A single non-conforming field used to fail
deserialization of the *whole* response, so the model appeared "broken". The
wire structs (`types.rs`) and the translator (`convert.rs`) are deliberately
lenient — leniency is additive to the serde shapes (`#[serde(default)]` /
custom deserializers), so hosted OpenAI is unaffected:

- **Missing/empty `id`** — Ollama omitted the tool-call `id` until v0.12.11. An
  empty or absent id is treated as absent and a stable `tool-{index}` fallback
  is synthesized (the same one the streaming path uses) so tool results still
  correlate back to the call.
- **Missing `type`** — defaulted rather than required.
- **`arguments` as an object** — some servers send `function.arguments` as a
  JSON object instead of the OpenAI-standard stringified JSON; it is normalized
  to the parsed arguments either way.
- **Malformed `arguments` JSON** — when arguments cannot be parsed even after the
  conservative chat-template-marker repair (`recover_tool_arguments`), the call
  does **not** fail the model call. It is surfaced as a `ToolCall` with
  `invalid: Some(reason)` and the raw string preserved in `arguments`. The agent
  loop feeds `reason` back to the model as an error tool result (an
  `AgentEvent::InvalidToolArgs` recovery, mirroring LangChain's
  `invalid_tool_calls` and the AI SDK's invalid dynamic tool parts) so the model
  can retry, and — because the call always resolves — a malformed blob can never
  become a never-resolving tool call that stalls the loop. This leniency is
  unconditional: unlike `InvalidArgsPolicy` (which governs *schema* validation of
  well-formed arguments), an unparseable payload is a transport-level defect.

## Error handling

Non-2xx responses are normalized through `parse_error_body` / `provider_error`
into a structured `ProviderError` (HTTP status, provider error code, and a
`retryable` flag derived from the status: 408/409/429/5xx are retryable,
everything else — including 401/400 — is not) and surfaced as
`TinyAgentsError::Provider`, so `harness::retry::is_retryable` can classify
retryability instead of retrying every provider failure indiscriminately.
Transport-level failures (connection errors, body-read failures) have no such
structure to preserve and surface as a plain `TinyAgentsError::Model` string
via `provider_failure_message`. Malformed JSON bodies surface as
`TinyAgentsError::Serialization`.

## Operational constraints

- `DEFAULT_CONNECT_TIMEOUT_SECS` (30s) bounds TCP connection establishment on
  every call, including streaming, without capping response body time.
- `DEFAULT_REQUEST_TIMEOUT_SECS` (600s) is an overall timeout applied to
  **unary** calls only when `ModelRequest::timeout_ms` is unset; streaming
  calls get no overall cap by default — callers relying on a hard streaming
  deadline must set `timeout_ms` explicitly or enforce one externally.
- o-series models (`o1`, `o3`, ...) require `max_completion_tokens` instead of
  `max_tokens`; this is handled internally based on the model id.
- The client is reused across calls for connection pooling — construct one
  `OpenAiModel` per logical provider/account rather than per request.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Module wiring: shared imports/constants and re-exports (`OpenAiModel`). |
| `transport.rs` | `OpenAiModel` construction, provider presets, request building, and the `ChatModel` impl (`invoke`/`stream`). |
| `convert.rs` | Request/response translation between harness types and the OpenAI wire format. |
| `sse.rs` | SSE stream parsing and incremental accumulation (`SseState`, `OpenAiStreamAcc`, `sse_next`). |
| `types.rs` | Wire (de)serialization shapes (`ModelListWire`, `ModelListing`, request/response bodies). |
| `test.rs` | Unit tests (SSE boundary decoding, tool-call correlation, error mapping, presets). |
