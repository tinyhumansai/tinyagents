# Harness Structured Output Feature

Structured output lets callers request a typed response instead of free-form
assistant text. The harness should support both provider-native structured
output and tool-call-based structured output.

## Source Inspiration

LangChain v1 implements structured output with provider and tool strategies:

- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py>
- provider implementations such as OpenAI:
  <https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py>
- standard structured-output tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

## Responsibilities

- Accept Rust types and JSON schema as response schemas.
- Support provider-native response-format APIs.
- Support artificial tool-call schemas when provider-native mode is unavailable.
- Validate responses into typed values.
- Preserve raw assistant messages alongside parsed output.
- Handle validation errors according to policy.
- Support union/oneOf variants.
- Support strict and non-strict schema modes.
- Emit structured-output events.

## Strategies

```rust
pub enum ResponseFormat {
    Auto(ResponseSchema),
    Provider(ProviderStructuredOutput),
    Tool(ToolStructuredOutput),
    JsonSchema(JsonSchema),
}

pub struct ResponseSchema {
    pub name: String,
    pub description: Option<String>,
    pub schema: JsonSchema,
    pub strict: Option<bool>,
}
```

`Auto` should choose provider-native mode only when the selected model profile
declares support. Otherwise it should fall back to tool strategy when the model
supports tool calling.

## Provider Strategy

Provider strategy sends the schema to the model provider through a native
response-format API. It is preferred when:

- the provider validates server-side
- the model profile declares structured output support
- the schema is accepted by that provider
- streaming behavior is understood

Provider strategy must still validate the returned value locally. Provider
validation errors should be surfaced as provider errors; local parsing errors
should be surfaced as structured-output validation errors.

## Tool Strategy

Tool strategy exposes an artificial output tool to the model. The final
structured output is parsed from the tool-call arguments.

Tool strategy must handle:

- multiple structured-output tool calls when only one is expected
- missing structured-output tool calls
- invalid JSON arguments
- schema validation errors
- union variant selection
- optional repair messages back into the agent loop

The artificial tool should not execute application side effects. It is a parse
carrier only.

## Error Policy

```rust
pub enum StructuredOutputErrorPolicy {
    ReturnError,
    RetryWithDefaultMessage,
    RetryWithMessage(String),
    RetryWithFormatter(Arc<dyn StructuredErrorFormatter>),
}
```

Validation retries must count against model-call limits and retry budgets. Every
retry should emit an event containing the schema name, error kind, and attempt.

## Return Shape

```rust
pub struct StructuredRun<T> {
    pub parsed: T,
    pub raw: AssistantMessage,
    pub validation: ValidationRecord,
}
```

When callers request `include_raw`, the harness should return both raw and
parsed values. When they do not, the raw value should still be available through
events and stores if recording is enabled.
