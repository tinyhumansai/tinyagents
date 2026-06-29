# Harness Model And Provider Feature

The model feature owns provider-neutral chat model traits, request/response
shapes, streaming model chunks, provider capability profiles, provider-specific
options, and adapter conformance requirements.

## Source Inspiration

LangChain separates chat model interfaces, message types, provider integrations,
and model capability metadata:

- `BaseChatModel` and model streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/language_models>
- provider adapters such as OpenAI, Anthropic, Ollama, Mistral, Fireworks, Groq,
  Perplexity, and xAI:
  <https://github.com/langchain-ai/langchain/tree/master/libs/partners>
- model profiles:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/model_profile.py>
- model profile generation package:
  <https://github.com/langchain-ai/langchain/tree/master/libs/model-profiles>
- standard chat model tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

RustAgents should adapt this into a strongly typed Rust surface with
feature-gated provider crates/modules and a shared conformance suite.

## Responsibilities

- Define provider-neutral chat model traits.
- Normalize request options across providers without hiding provider escape
  hatches.
- Expose provider capability profiles.
- Validate requests against profiles before network calls.
- Convert provider responses into canonical `Message` and `ContentBlock` values.
- Preserve provider response ids, raw metadata, safety/refusal metadata, and
  usage details.
- Support invoke and stream paths with equivalent final semantics.
- Support model aliases, default models, and dynamic model selection.
- Support feature-gated provider adapters.
- Provide fake and replay models for tests.

## Non-Responsibilities

- It does not execute tools.
- It does not choose graph routes.
- It does not own provider API keys globally.
- It does not hardcode time-sensitive model prices.
- It does not assume all models support tool calling, structured output,
  multimodal input, or streaming.

## Core Types

```rust
#[async_trait]
pub trait ChatModel<State, Ctx = ()>: Send + Sync {
    fn profile(&self) -> Option<&ModelProfile>;

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

pub struct ModelRequest {
    pub model: ModelName,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDeclaration>,
    pub tool_choice: ToolChoice,
    pub response_format: Option<ResponseFormat>,
    pub model_settings: ModelSettings,
    pub provider_options: serde_json::Value,
    pub cache_policy: Option<CachePolicy>,
    pub required_capabilities: CapabilitySet,
    pub metadata: serde_json::Value,
}

pub struct ModelResponse {
    pub message: AssistantMessage,
    pub usage: Option<UsageRecord>,
    pub finish_reason: Option<FinishReason>,
    pub structured: Option<serde_json::Value>,
    pub provider: ProviderMetadata,
    pub timings: TimingBreakdown,
    pub cache: Option<CacheDecision>,
}
```

## Model Profiles

Every provider adapter should expose a `ModelProfile` when data is known.

```rust
pub struct ModelProfile {
    pub provider: ProviderName,
    pub model: ModelName,
    pub display_name: Option<String>,
    pub status: ModelStatus,
    pub release_date: Option<Date>,
    pub last_updated: Option<Date>,
    pub max_input_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub modalities: Modalities,
    pub supports_tool_calling: bool,
    pub supports_tool_choice: bool,
    pub supports_tool_call_streaming: bool,
    pub supports_native_structured_output: bool,
    pub supports_reasoning_output: bool,
    pub supports_temperature: bool,
    pub supports_attachments: bool,
    pub provider_extras: serde_json::Value,
}
```

Profiles are used to:

- reject impossible requests before making a provider call
- choose native structured output versus tool-based structured output
- decide whether tool-call chunks are streamable or must be reconstructed from
  a final response
- estimate context pressure
- expose capability information in docs and UIs
- select fallbacks that satisfy required capabilities

Profiles are not a pricing table. Prices belong to the cost feature and should
be updateable independently.

## Provider Options

`ModelSettings` should contain normalized settings:

- temperature
- max output tokens
- stop sequences
- top-p or equivalent sampling controls
- timeout
- seed
- reasoning effort or budget
- response format

`provider_options` should preserve provider-specific knobs such as OpenAI
Responses API options, Anthropic thinking config, Ollama local options, or
future provider extensions. Provider adapters should document which normalized
fields they honor and which provider options they pass through.

## Adapter Requirements

Provider adapters must:

- map RustAgents messages to provider request payloads
- map provider response payloads back to RustAgents messages
- preserve provider message ids and response ids
- preserve tool call ids and arguments
- produce `invalid_tool_call` data when tool calls cannot be parsed
- surface usage metadata when available
- mark estimated usage when usage is locally estimated
- emit model lifecycle events through `RunContext`
- respect cancellation and timeouts
- distinguish transport, authentication, rate-limit, provider, safety,
  validation, and parsing errors
- expose deterministic fake/replay fixtures for tests

Adapters should avoid global mutable provider clients unless those clients are
thread-safe and explicitly configured.

## Streaming

Streaming model responses should produce typed chunks:

```rust
pub enum ModelStreamItem {
    Started(ModelCallStarted),
    MessageDelta(MessageDelta),
    ToolCallDelta(ToolCallDelta),
    UsageDelta(UsageDelta),
    ProviderEvent(serde_json::Value),
    Completed(ModelResponse),
    Failed(RustAgentsError),
}
```

The final merged stream must be equivalent to `invoke` for the same provider
where the provider supports both paths. If a provider emits cumulative usage
rather than delta usage, the adapter must normalize it before sending usage
events.

## Conformance Tests

Each provider adapter should pass standard tests for:

- simple text invocation
- system message handling
- Unicode input/output
- tool declaration binding
- tool call id preservation
- no-argument tool calls
- structured output by native provider mode when supported
- structured output by tool strategy when supported
- streaming text chunks
- streaming tool-call chunks when supported
- multimodal input where supported
- usage metadata normalization
- provider options passthrough
- callback/event lifecycle
- cancellation and timeout behavior
- retryable versus non-retryable error classification
