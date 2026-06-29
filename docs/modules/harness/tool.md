# Harness Tool Feature

The tool feature owns typed capabilities exposed to agents. It defines tool
metadata, JSON-schema-compatible model-visible inputs, hidden runtime injection,
validation, execution, retry policy, artifacts, and result formatting.

## Source Inspiration

LangChain tool behavior is spread across core tools, v1 tool-node re-exports,
agent middleware, and standard tests:

- core tools:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tools>
- v1 tool-node compatibility exports:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/tools/tool_node.py>
- agent middleware tool wrappers:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py>
- standard tool-call tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

LangChain and LangGraph also distinguish model-visible tool arguments from
runtime-injected values such as state, store, context, and stream writers.
RustAgents should make that distinction explicit in Rust types.

## Responsibilities

- Register named tools.
- Validate tool names and reject duplicates.
- Expose model-visible JSON schemas.
- Hide injected runtime parameters from model-visible schemas.
- Validate model-provided arguments before execution.
- Execute tools with access to state, runtime context, stores, cancellation, and
  event streams.
- Record tool lifecycle events.
- Format tool results as canonical messages.
- Preserve structured outputs and artifact references.
- Classify tool errors for retry, user-visible repair, or hard failure.
- Support serial and bounded-concurrent execution.
- Support tool selection middleware and dynamic tool exposure.

## Core Types

```rust
#[async_trait]
pub trait Tool<State, Ctx = ()>: Send + Sync {
    fn spec(&self) -> ToolSpec;

    async fn call(
        &self,
        state: &State,
        runtime: ToolRuntime<'_, Ctx>,
        call: ToolCall,
    ) -> Result<ToolResult>;
}

pub struct ToolSpec {
    pub name: ToolName,
    pub description: String,
    pub input_schema: JsonSchema,
    pub output_schema: Option<JsonSchema>,
    pub injected: Vec<InjectedArgSpec>,
    pub safety: ToolSafety,
    pub timeout: Option<Duration>,
    pub retry: Option<RetryPolicy>,
    pub idempotency: Idempotency,
}

pub struct ToolRuntime<'a, Ctx = ()> {
    pub ctx: &'a mut RunContext<Ctx>,
    pub stores: &'a StoreRegistry,
    pub events: &'a EventSink,
    pub cancellation: CancellationToken,
}
```

## Tool Names

Tool names should default to ASCII `snake_case`. The registry should reject:

- empty names
- duplicate names
- names with spaces
- names that exceed provider-safe length limits
- names requiring provider-specific escaping

The registry may support provider-specific aliases, but canonical events and
stores should use the RustAgents tool name.

## Schema Rules

The model-visible input schema must include only arguments the model may choose.
Hidden runtime values include:

- current run context
- thread id and run id
- state references
- store handles
- event emitters
- stream writers
- cancellation handles
- secrets or provider clients

Hidden values must never appear in the tool schema sent to a model. This avoids
teaching the model about internal implementation details and prevents accidental
secret exposure.

## Execution Lifecycle

1. Validate tool name exists.
2. Validate arguments against the input schema.
3. Check tool-call limits and concurrency limits.
4. Emit `tool.started`.
5. Run `before_tool` middleware.
6. Execute the tool with `ToolRuntime`.
7. Run `after_tool` middleware.
8. Format result into a `ToolMessage`.
9. Persist artifacts if configured.
10. Emit `tool.completed` or `tool.failed`.

Validation failures should produce a model-consumable error message when the
agent loop can recover, and a hard error when policy forbids repair.

## Results And Artifacts

```rust
pub struct ToolResult {
    pub tool_call_id: ToolCallId,
    pub name: ToolName,
    pub content: Vec<ContentBlock>,
    pub value: Option<serde_json::Value>,
    pub artifacts: Vec<ArtifactRef>,
    pub is_error: bool,
    pub provider: Option<ProviderMetadata>,
    pub elapsed: Duration,
}
```

Large outputs should be stored as artifacts and summarized for model context.
The full artifact key should be available to application code and events, while
the model-facing content should stay bounded and redacted.

## Safety Metadata

Tools should declare safety metadata:

- read-only versus mutating
- idempotent versus non-idempotent
- local-only versus networked
- filesystem access
- shell/process access
- payment or external spend
- requires user confirmation
- allowed workspace root
- redaction policy

Middleware can use this metadata to enforce confirmation, sandboxing, allowlist,
or human-in-the-loop policies.
