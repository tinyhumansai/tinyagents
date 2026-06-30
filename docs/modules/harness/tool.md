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
TinyAgents should make that distinction explicit in Rust types.

## Responsibilities

- Register named tools.
- Validate tool names and reject duplicates.
- Expose model-visible JSON schemas.
- Hide injected runtime parameters from model-visible schemas.
- Validate model-provided arguments before execution.
- Validate provider-supplied tool calls against the tools advertised for the
  current turn before execution.
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
stores should use the TinyAgents tool name.

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

The local execution boundary validates the schema subset TinyAgents relies on
for fail-closed tool dispatch:

- primitive and compound `type` checks
- object `properties`
- `required` object fields
- `additionalProperties: false`
- array `items`
- exact-value `enum`

Richer provider-facing JSON Schema keywords may still be present; the harness
passes them through to providers but only enforces the subset above locally.

## Tool Call Formats

`ToolSchema` carries a `format: ToolFormat` field so a tool definition can state
how it should be shown to a model. The execution boundary is still one typed
shape: after parsing a model emission, the harness invokes tools through
`ToolCall { id, name, arguments }`, where `arguments` is `serde_json::Value`.
That keeps validation, middleware, replay, and provider normalization stable
even when the model-facing syntax changes.

TinyAgents supports three model-facing tool formats:

- `ToolFormat::Json` — JSON/function-call format. This is the default and is
  omitted during serialization for backward compatibility. Providers with
  native function/tool calling, such as OpenAI Chat Completions, can map this
  directly to their native tool declaration shape.
- `ToolFormat::Xml` — XML tag format. A renderer may expose the same tool as
  `<tool_name><field>value</field></tool_name>`. The parser must normalize the
  emitted tag body back into JSON arguments before schema validation.
- `ToolFormat::PType { parameters }` — parametric p-type format. This is a
  compact ordered-parameter syntax for token-sensitive prompts, for example
  `search("rust agents", 5)`. `parameters` records the ordered field names that
  map positional values back into the JSON argument object.

Example:

```rust
use serde_json::json;
use tinyagents::harness::tool::{ToolFormat, ToolSchema};

let json_tool = ToolSchema::new(
    "weather",
    "Look up weather for a city.",
    json!({
        "type": "object",
        "required": ["city"],
        "properties": { "city": { "type": "string" } }
    }),
);

let xml_tool = json_tool.clone().with_format(ToolFormat::Xml);

let ptype_tool = ToolSchema::new(
    "search",
    "Search documents.",
    json!({
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": { "type": "string" },
            "limit": { "type": "integer" }
        }
    }),
)
.with_format(ToolFormat::PType {
    parameters: vec!["query".to_string(), "limit".to_string()],
});
```

Provider adapters should treat `ToolFormat` as a capability-aware rendering
preference:

- If the provider has native JSON/function calling, send `ToolFormat::Json`
  tools as native tool declarations.
- If the provider does not support the requested format natively, render the
  declaration into prompt text and parse the model's emitted call back into
  `ToolCall`.
- If a provider only accepts JSON tool declarations, it may fall back to the
  JSON schema while preserving `ToolSchema::format` in harness metadata.

## Execution Lifecycle

1. Check cancellation, wall-clock deadline, and tool-call limits.
2. Run `before_tool` middleware, allowing policy middleware to reject or adjust
   the pending call.
3. Validate the final tool name exists.
4. Validate final arguments against the input schema.
5. Emit `tool.started`.
6. Execute the tool with `ToolRuntime`.
7. Run `after_tool` middleware.
8. Format result into a `ToolMessage`.
9. Persist artifacts if configured.
10. Emit `tool.completed` or `tool.failed`.

Validation failures should produce a model-consumable error message when the
agent loop can recover, and a hard error when policy forbids repair.

Provider-supplied tool calls must fail closed:

- unknown tool names are not executed
- malformed JSON arguments are not replaced with empty defaults for
  side-effecting tools
- tool call ids are preserved in error tool messages
- allowlist violations emit events and append repairable tool-result messages
  only when the agent loop policy allows recovery

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
