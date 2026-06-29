# Harness Middleware Feature

Middleware is the main extension point for behavior that cuts across models,
tools, memory, stores, streaming, and graph nodes.

## Source Inspiration

LangChain v1 middleware provides typed model requests, tool-call wrappers,
before/after hooks, wrap hooks, dynamic prompts, human-in-the-loop control, PII
redaction, retry/fallback, summarization, context editing, and tool selection:

- middleware types:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py>
- built-in middleware:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware>
- agent factory composition:
  <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/factory.py>

RustAgents should provide equivalent extension power without requiring users to
understand graph internals for normal harness usage.

## Responsibilities

- Provide stable middleware ordering.
- Support before/after hooks for observation and simple mutation.
- Support wrap hooks for replacement, retry, fallback, short-circuit, and
  human-interrupt behavior.
- Allow middleware to modify model requests, tool calls, and responses.
- Allow middleware to emit events.
- Allow middleware to add local state updates without mutating unrelated state.
- Allow middleware to jump to model, tools, or end when used inside an agent loop.
- Expose errors to middleware for logging, redaction, fallback, or recovery.
- Keep middleware testable with fake models, tools, and event sinks.

## Hook Types

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_agent(&self, state: &State, ctx: &mut RunContext<Ctx>) -> Result<()>;
    async fn after_agent(&self, state: &State, ctx: &mut RunContext<Ctx>, run: &mut AgentRun) -> Result<()>;

    async fn before_model(&self, state: &State, ctx: &mut RunContext<Ctx>, request: &mut ModelRequest) -> Result<()>;
    async fn after_model(&self, state: &State, ctx: &mut RunContext<Ctx>, response: &mut ModelResponse) -> Result<()>;

    async fn before_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, call: &mut ToolCall) -> Result<()>;
    async fn after_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, result: &mut ToolResult) -> Result<()>;

    async fn on_error(&self, state: &State, ctx: &mut RunContext<Ctx>, error: &RustAgentsError) -> Result<()>;
}
```

Wrap hooks need separate traits because they receive a handler:

```rust
#[async_trait]
pub trait ModelMiddleware<State, Ctx = ()>: Send + Sync {
    async fn wrap_model(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<ModelMiddlewareOutcome>;
}

#[async_trait]
pub trait ToolMiddleware<State, Ctx = ()>: Send + Sync {
    async fn wrap_tool(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ToolCallRequest,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<ToolMiddlewareOutcome>;
}
```

## Ordering

Before hooks run in registration order. After hooks run in reverse registration
order. Wrap hooks compose so the first registered middleware is the outermost
layer. This mirrors common web middleware stacks and keeps cleanup symmetrical.

## Control Outcomes

Middleware should be able to return:

- continue with modified request
- replace model/tool response
- emit state update
- retry current call
- fallback to another model or tool
- jump to `model`
- jump to `tools`
- jump to `end`
- interrupt for human input
- fail with classified error

Graph-specific commands should be translated at the graph boundary. The harness
should expose harness-native control outcomes so it remains usable without a
graph.

## Built-In Middleware

Initial built-ins should include:

- tracing/event middleware
- timeout middleware
- retry middleware
- model fallback middleware
- model-call limit middleware
- tool-call limit middleware
- rate limiter middleware
- dynamic prompt middleware
- context editing middleware
- message trimming middleware
- summarization middleware
- structured output validation middleware
- PII detection/redaction middleware
- tool allowlist middleware
- dynamic tool selection middleware
- human-in-the-loop middleware
- privileged shell/filesystem guard middleware

Each built-in must document:

- hook points used
- mutation behavior
- emitted events
- failure mode
- interaction with streaming
- interaction with retries and fallbacks

## State And Request Mutation

Middleware should prefer immutable request replacement for large changes and
small explicit mutation for local fields. It must not mutate shared registries or
global config during a run. Runtime state updates should be explicit and
observable.
