# Harness Observability: Structured Output, Events, Testkit

Continues from [`README.md`](README.md) and [`runtime.md`](runtime.md):
structured output, events and streaming, errors, testkit, and
implementation milestones.

## Structured Output

Structured output should support:

- provider-native schema mode
- tool-call fallback mode
- JSON parsing mode for simple local models

```rust
pub enum ResponseFormat {
    Text,
    JsonSchema(JsonSchema),
    ProviderNative(JsonSchema),
    ToolStrategy { tool_name: String, schema: JsonSchema },
}
```

The final run result should keep messages and structured output separate:

```rust
pub struct AgentRun<State, Output = ()> {
    pub state: State,
    pub messages: Vec<Message>,
    pub structured_response: Option<Output>,
    pub events: Vec<AgentEvent>,
}
```

## Events And Streaming

The harness event stream should be typed, not a string callback.

```rust
pub enum AgentEvent {
    RunStarted { run_id: RunId, thread_id: Option<ThreadId> },
    ModelStarted { call_id: CallId, model: ModelName },
    ModelDelta { call_id: CallId, delta: MessageDelta },
    ModelCompleted { call_id: CallId, started_at_ms: Option<u64>, usage: Option<Usage> },
    ToolStarted { call_id: CallId, tool_name: ToolName },
    ToolDelta { call_id: CallId, delta: ToolDelta },
    ToolCompleted { call_id: CallId, tool_name: ToolName, started_at_ms: Option<u64> },
    MiddlewareStarted { name: String },
    MiddlewareCompleted { name: String },
    RetryScheduled { call_id: CallId, attempt: usize },
    Custom { name: String, payload: serde_json::Value },
    RunCompleted { run_id: RunId },
    RunFailed { run_id: RunId, error: String },
}
```

Streaming modes:

- `messages`: model deltas and final messages
- `tools`: tool start, progress, result
- `updates`: state or memory updates
- `events`: all low-level events
- `final`: final output only

## Errors

Harness errors should distinguish:

- invalid request
- missing model
- missing tool
- invalid tool schema
- invalid tool arguments
- provider authentication failure
- provider rate limit
- provider server error
- timeout
- retry exhausted
- structured output validation failure
- middleware failure
- memory failure

Retry policy should only retry explicitly retryable classes by default:

- network interruption
- timeout
- rate limit
- provider 5xx

Do not retry authentication, schema, malformed request, or missing tool errors
unless a user explicitly overrides policy.

## Testkit

`harness::testkit` should be part of the early API.

Utilities:

- `FakeChatModel`
- `ScriptedChatModel`
- `FakeStreamingModel`
- `FakeTool`
- `InMemoryShortTermMemory`
- `InMemoryStore`
- `EventRecorder`
- deterministic ids
- deterministic clock
- trajectory assertions

Example trajectory assertion:

```rust
assert_trajectory(run.events())
    .model_called("default")
    .tool_called("lookup_user")
    .model_called("default")
    .completed();
```

## Implementation Milestones

### H1: Current Minimal Traits

- Keep `ChatMessage`.
- Keep `ChatModel`.
- Keep `Tool`.
- Add better tool call ids.

### H2: Registries And Context

- Add `ModelRegistry`.
- Add `ToolRegistry`.
- Add `RunConfig`.
- Add `RunContext`.

### H3: Agent Loop

- Implement model-tool loop.
- Enforce limits.
- Add fake model and fake tool tests.

### H4: Middleware And Events

- Add middleware stack.
- Add typed events.
- Add event recorder.

### H5: Memory And Structured Output

- Add short-term memory trait.
- Add store trait.
- Add structured response format.

### H6: Providers

- Add feature-gated provider crates or modules.
- Start with mock and one hosted provider.
