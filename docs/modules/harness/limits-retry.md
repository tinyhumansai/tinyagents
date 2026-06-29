# Harness Limits, Retry, Fallback, And Rate Limiting

This feature owns policy for stopping runaway agent loops, retrying transient
failures, falling back between models/tools, and pacing provider calls.

## Source Inspiration

LangChain and LangChain v1 provide related behavior in several places:

- runnable retry:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/retry.py>
- runnable fallbacks:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/fallbacks.py>
- in-memory rate limiter:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/rate_limiters.py>
- model and tool retry/fallback middleware:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware>

TinyAgents should centralize these policies so every model call, tool call,
agent loop, and graph node can share predictable behavior.

## Responsibilities

- Enforce maximum model calls.
- Enforce maximum tool calls.
- Enforce maximum retries.
- Enforce maximum wall-clock duration.
- Enforce per-call timeouts.
- Enforce maximum concurrency.
- Enforce optional recursion depth.
- Enforce graph super-step caps.
- Support cancellation.
- Classify retryable and non-retryable errors.
- Support exponential backoff with jitter.
- Support model fallback chains.
- Support tool fallback or emulator policies.
- Support token-bucket rate limiting.
- Emit wait, retry, fallback, limit, timeout, and cancellation events.

## Limit Policy

```rust
pub struct RunLimits {
    pub max_model_calls: usize,
    pub max_tool_calls: usize,
    pub max_retries: usize,
    pub max_concurrency: usize,
    pub max_recursion_depth: usize,
    pub max_graph_steps: usize,
    pub wall_clock_timeout: Option<Duration>,
    pub model_call_timeout: Option<Duration>,
    pub tool_call_timeout: Option<Duration>,
}
```

Limits should fail closed. When a limit is reached, the harness should stop the
agent loop and return a classified error that includes the current counters.

For graph-backed runs, max-step failures should include graph name, run id,
current super-step, frontier nodes, and the configured cap. The run should be
persisted as failed when a checkpointer is configured.

## Retry Policy

```rust
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub jitter: JitterPolicy,
    pub retry_on: RetryClassifier,
}
```

Retry classifiers should distinguish:

- transport errors
- provider rate-limit errors
- provider 5xx errors
- provider 4xx errors
- authentication errors
- timeout errors
- local validation errors
- structured-output validation errors
- tool validation errors
- tool execution errors
- budget errors
- cancellation

Authentication, budget, schema definition, and cancellation errors should not be
retried by default.

## Fallback Policy

```rust
pub struct ModelFallbackPolicy {
    pub candidates: Vec<ModelName>,
    pub require_same_capabilities: bool,
    pub retry_before_fallback: usize,
}
```

Fallback selection must honor required capabilities. A request that requires
tool calling must not silently fall back to a model without tool calling.
Fallback events should include the failed model, selected model, error class,
and attempt count.

## Rate Limiting

Rate limiting should be separate from LLM token accounting. It paces requests;
it does not count prompt tokens.

```rust
#[async_trait]
pub trait RateLimiter: Send + Sync {
    async fn acquire(&self, permit: RateLimitPermit) -> Result<RateLimitLease>;
}

pub struct TokenBucketRateLimiter {
    pub requests_per_second: f64,
    pub max_bucket_size: f64,
}
```

Rate-limit wait time should be emitted as an event and included in timing
breakdowns. This avoids hiding queueing latency inside provider latency.

## Cancellation

`RunContext` should carry a cancellation token. Model adapters, tools, stores,
streams, retry sleeps, and rate-limit waits must observe it. Cancellation should
produce a distinct error class and should not be retried.
