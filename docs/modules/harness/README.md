# Harness Module Specification

The harness is the orchestration layer around LLM calls. It owns model
registration, tool registration, prompt assembly, middleware, memory, event
streaming, tracing, retries, limits, summarization, caching, usage accounting,
pricing, sub-agent/orchestrator steering, and test support.

The harness should be usable in three modes:

1. direct model invocation
2. model plus tools agent loop
3. graph node runtime dependency

It should not require the graph module. The graph module can depend on harness
traits, but a user should be able to call a model or run a tool loop without
constructing a graph.

## Source Inspiration

Primary references:

- <https://docs.langchain.com/oss/python/langchain/agents>
- <https://docs.langchain.com/oss/python/langchain/models>
- <https://docs.langchain.com/oss/python/langchain/tools>
- <https://docs.langchain.com/oss/python/langchain/runtime>
- <https://docs.langchain.com/oss/python/langchain/short-term-memory>
- <https://docs.langchain.com/oss/python/langchain/structured-output>
- <https://docs.langchain.com/oss/python/langchain/middleware/built-in>
- <https://docs.langchain.com/oss/python/langchain/streaming>
- <https://docs.langchain.com/oss/python/langchain/observability>
- <https://docs.langchain.com/oss/python/langchain/test>
- LangChain callback usage tracking code:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- LangChain store and chat history code:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/stores.py>
  and
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_history.py>
- LangChain v1 agent factory:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py>
- LangChain v1 agent middleware types and built-ins:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware>
- LangChain structured output strategies:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py>
- LangChain model profiles:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/model_profile.py>
  and
  <https://github.com/langchain-ai/langchain/tree/master/libs/model-profiles>
- LangChain message and content-block model:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/messages>
- LangChain embeddings, vector stores, retrievers, and indexing:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/embeddings>
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/vectorstores>
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/retrievers.py>
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/indexing>
- LangChain runnable config, fallbacks, retry, and event streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/runnables>
- OpenHuman PR #4261 agent graph implementation:
  <https://github.com/tinyhumansai/openhuman/pull/4261>
- OpenHuman PR #4261 state graph files:
  `src/openhuman/agent_graph/graph/`, `checkpoint/`, `hitl/`,
  `observability/`, `definitions/`, `blueprint/`, `live/`, `ops.rs`, and
  `schemas.rs`
- LangChain callbacks, tracers, and usage accounting:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/callbacks>
  and
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- LangChain standard integration tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

LangChain separates durable core primitives (`libs/core`), the v1 agent facade
(`libs/langchain_v1`), legacy/classic integrations (`libs/langchain`), partner
providers (`libs/partners`), model capability data (`libs/model-profiles`), and
standard provider test suites (`libs/standard-tests`). TinyAgents should keep a
similar separation of concerns even though it is a single Rust crate today:
harness traits first, feature-gated provider adapters second, and compatibility
tests for every adapter.

## Responsibilities

- Normalize user input into structured messages.
- Build model requests from messages, prompts, tools, memory, and config.
- Track context-window pressure and choose trimming or summarization policies.
- Preserve provider prompt/KV-cache stability by making stable prompt prefixes
  explicit and keeping volatile context out of those prefixes by default.
- Dispatch model calls through provider-neutral traits.
- Resolve each agent/model call through request overrides, reusable state,
  model hints, agent defaults, registry defaults, and fallback policy.
- Dispatch embedding calls through provider-neutral traits.
- Dispatch tool calls through a registry with schema validation.
- Expose retrievers and vector stores for retrieval-augmented context assembly.
- Run the standard model-tool-model agent loop.
- Represent the standard agent loop as an explicit state machine when callers
  need inspection, checkpointing, HITL, branch/fork semantics, or graph-node
  execution.
- Apply middleware before and after model calls, tool calls, retries, and errors.
- Enforce model-call limits, tool-call limits, timeouts, and retry policy.
- Emit typed events for tracing, streaming, and tests.
- Persist short-term thread memory when configured.
- Expose durable stores through runtime context.
- Store run data, messages, events, tool artifacts, and application records
  through pluggable backends.
- Track prompt tokens, completion tokens, cached tokens, model prices, and
  run-level cost.
- Enforce optional token budgets and dollar budgets.
- Cache reusable prompts, model responses, tool artifacts, and summaries when
  policy allows.
- Distinguish local response caching from provider prompt/KV-cache reuse and
  expose cache layout events when middleware changes model-visible prompt
  segments.
- Provide deterministic test utilities.
- Describe provider capability profiles so middleware can choose safe defaults.
- Translate between provider-native message formats and TinyAgents messages.
- Persist resolved model identity in response metadata, run status, events,
  usage/cost rows, and durable agent or graph state for reuse.
- Support dynamic runtime context injection into tools and middleware without
  exposing private state to model-visible schemas.
- Support model fallback, tool retry, rate limiting, and human interruption as
  explicit policies rather than ad hoc callbacks.
- Support parent-orchestrator and human steering of sub-agents, orchestrator
  agents, graph tasks, and harness loops through typed commands delivered at
  safe boundaries.
- Queue host-owned payloads in steer, follow-up, and collected-context lanes
  without importing host transport or UI metadata into the harness.
- Support durable graph runs with pause/resume, checkpoint listing, and
  inspectable node transitions.
- Support per-agent execution blueprints that describe how an agent runs
  separately from what its prompt says.
- Provide a standard conformance test suite for model, tool, store, stream, and
  middleware implementations.

## Non-Responsibilities

- It does not own graph topology.
- It does not decide graph routing except inside an agent-loop node.
- It does not persist graph checkpoints; that belongs to the graph module.
- It does not hide provider-specific metadata when users need it.
- It does not execute arbitrary workflow language source directly.
- It does not require every provider to support every modality or output
  strategy; capability profiles describe those differences.
- It does not make hidden network calls from tools, middleware, or stores unless
  the configured implementation does so explicitly.

## Package Shape

Each substantial harness feature gets its own module. This is not just file
organization; it is an ownership rule. If a feature will need its own traits,
errors, tests, middleware, or provider adapters, it belongs in its own submodule.

Target layout:

```text
crates/tinyagents-harness/src/
  mod.rs
  agent_loop.rs
  cache.rs
  context.rs
  cost.rs
  embeddings.rs
  events.rs
  graph_runtime.rs
  limits.rs
  memory.rs
  message.rs
  middleware.rs
  model.rs
  prompt.rs
  providers.rs
  retry.rs
  run_queue/
    mod.rs
    test.rs
    types.rs
  runtime.rs
  steering.rs
  stream.rs
  summarization.rs
  structured.rs
  store.rs
  testkit.rs
  tool.rs
  usage.rs
```

The current crate already has top-level `chat.rs`, `model.rs`, and `tool.rs`.
Those can either stay as public re-exports or move under `harness/` once the API
settles.

Feature ownership:

- `agent_loop`: default model-tool-model loop.
- `cache`: prompt, provider prompt/KV-cache, response, summary, and artifact
  cache policy.
- `cancel`: cooperative `CancellationToken` observed at agent-loop checkpoints
  (before each model and tool call) and in the streaming/retry paths.
- `context`: `RunConfig`, `RunContext`, inherited metadata, runtime values.
- `cost`: model pricing, budget policy, and cost rollups.
- `embeddings`: embedding providers, vector stores, retrievers, indexing, and
  retrieval-context records.
- `events`: typed harness events, sinks, streams, redaction adapters.
- `graph_runtime`: explicit state graphs, node commands, reducers,
  checkpointing, HITL, run records, and graph execution blueprints.
- `limits`: model-call, tool-call, timeout, retry, and recursion policy.
- `memory`: short-term thread memory and long-term stores.
- `message`: structured messages, content blocks, tool call correlation.
- `middleware`: before/after/wrap hooks and middleware stack ordering.
- `model`: provider-neutral model traits, requests, responses, streams.
- `prompt`: prompt templates, rendering, and dynamic prompt context.
- `providers`: feature-gated provider adapters.
- `retry`: retry classification, backoff, attempt accounting.
- `run_queue`: generic FIFO mechanics for steer, follow-up, and
  collected-context payloads; hosts retain ownership of payload metadata.
- `runtime`: high-level `AgentHarness` builder/facade.
- `steering`: policy-checked parent/human steering of orchestrators,
  sub-agents, graph tasks, and harness loops.
- `stream`: token streams, tool progress streams, event streams, adapters.
- `summarization`: context summaries, message compaction, summary provenance.
- `structured`: typed response formats and validation.
- `store`: JSONL, file, MongoDB, in-memory, and other persistence backends.
- `testkit`: fakes, recorders, deterministic ids, trajectory assertions.
- `tool`: tool traits, schemas, validation, execution, result formatting.
- `usage`: token accounting, cached token tracking, context-window estimates.
- `workspace`: per-agent filesystem/sandbox isolation, allowed-root descriptors,
  and fail-closed path enforcement for tools that touch real files.

### Host-authorized invocations and tool timeouts

Split into a focused doc: how a host capability bundle is bound to one
invocation, and how per-tool timeouts are resolved. See
[hosting.md](hosting.md).

Continued specification: [runtime.md](runtime.md) (tool registry, agent loop,
middleware, memory/stores) and
[observability-overview.md](observability-overview.md) (structured output,
events/streaming, errors, testkit, milestones).

Feature details:

- [Context feature](context.md)
- [Model and provider feature](model.md)
- [Local models and embeddings (Ollama, LM Studio)](local-models.md)
- [Embeddings and retrieval feature](embeddings.md)
- [State graph runtime feature](state-graph.md)
- [Prompt feature](prompt.md)
- [Tool feature](tool.md)
- [Tool exposure, discovery, and schema budgets](tool-discovery.md)
- [Tool dialects](tool-dialect.md)
- [Workspace isolation feature](workspace.md)
- [Middleware feature](middleware.md)
- [Sub-agent and orchestrator steering](subagent-steering.md)
- [Structured output feature](structured-output.md)
- [Limits, retry, fallback, and rate limiting](limits-retry.md)
- [Summarization feature](summarization.md)
- [Usage feature](usage.md)
- [Cost feature](cost.md)
- [Cache feature](cache.md)
- [Streaming feature](streaming.md)
- [Store feature](store.md)
- [Observability and events](observability.md)
- [Testkit feature](testkit.md)
- [Host authorization and tool timeouts](hosting.md)
- [LangChain feature parity map](langchain-parity.md)
- [Design notes: harness core-type sketch](design-notes.md)

## Core Types (moved)

See [`design-notes.md`](design-notes.md) for the harness core-type sketch
(`AgentHarness`, `RunConfig`, `RunContext`) — moved out of this file to keep
it under the repo's 500-line Markdown limit.

## Messages

Messages are the harness's internal data model. Raw strings should only appear
at API boundaries.

```rust
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

pub enum ContentBlock {
    Text(String),
    Json(serde_json::Value),
    Image(ImageRef),
    Audio(AudioRef),
    File(FileRef),
    ToolCall(ToolCallBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
    Citation(CitationBlock),
    Refusal(RefusalBlock),
    ProviderExtension(serde_json::Value),
}

pub struct AssistantMessage {
    pub id: Option<String>,
    pub content: Vec<ContentBlock>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub provider: Option<ProviderMetadata>,
}

pub struct ToolMessage {
    pub tool_call_id: String,
    pub name: String,
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}
```

Required message properties:

- stable role
- structured content blocks
- assistant tool calls
- tool call ids
- tool result correlation
- usage metadata
- provider extension escape hatch
- invalid or partially parsed tool calls for streaming/provider repair
- reasoning, citation, refusal, and safety blocks when providers expose them
- provider response ids for continuation/resume APIs

The first public API can keep `ChatMessage` as a simple compatibility type, but
the harness internals should move toward richer messages before provider
integrations are added.

Provider adapters are responsible for converting between provider payloads and
this model. The conversion must be round-trip safe for supported fields and must
preserve unknown provider fields in `ProviderExtension` rather than dropping
them. Streaming adapters must merge message chunks deterministically.

## Model Registry

The model registry maps names to provider-neutral implementations.

```rust
pub struct ModelRegistry<State, Ctx = ()> {
    models: HashMap<ModelName, Arc<dyn ChatModel<State, Ctx>>>,
    default: Option<ModelName>,
}

#[async_trait]
pub trait ChatModel<State, Ctx = ()>: Send + Sync {
    async fn invoke(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelResponse>;

    async fn stream(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelStream>;
}
```

`ModelRequest` should contain:

- model id or registry alias
- model hints and capability requirements for smart resolution
- previous resolved-model reuse policy
- messages
- tool declarations
- tool choice policy
- response format
- temperature
- max tokens
- stop sequences
- timeout
- retry policy
- tags and metadata
- provider options
- capability requirements such as `requires_tool_calling`,
  `requires_structured_output`, `requires_image_input`, or
  `requires_tool_call_streaming`
- cache policy
- rate-limit policy
- continuation or previous-response id where a provider supports it

`ModelResponse` should contain:

- resolved model identity
- assistant message
- usage
- finish reason
- raw provider metadata
- structured response when requested
- provider response id
- safety/refusal metadata
- retry and cache metadata
- elapsed time and timing breakdown

Provider integrations should be optional features:

- `provider-openai`
- `provider-anthropic`
- `provider-ollama`
- `provider-mock`

Every provider adapter must expose a `ModelProfile`. Middleware and builders
should use profiles to reject impossible requests early, choose provider-native
structured output only when supported, reserve context window budget, and decide
whether streamed tool-call chunks can be trusted.

Model selection should be explicit and reusable. The harness resolves each model
call through request override, durable prior state, model hints, agent default,
registry default, and fallback policy. The selected model is recorded as a
`ResolvedModel` in the response, event stream, run status, usage/cost records,
and durable state when configured.


---

Continues in [`runtime.md`](runtime.md) (tool registry, agent loop,
middleware, memory/stores) and [`observability-overview.md`](observability-overview.md)
(structured output, events/streaming, errors, testkit, milestones).
