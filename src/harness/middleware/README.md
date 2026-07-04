# harness::middleware

Cross-cutting extension points that wrap agent, model, and tool execution.

In the recursive (RLM-style) harness a sub-agent or sub-graph is just another
agent loop, so the same before/after hooks bracket the parent run *and* every
nested model/tool/agent call beneath it. That uniform wrapping is what lets
concerns like tracing, usage/cost roll-up, and guardrails compose consistently
as models call models and graphs run graphs.

## Two extension shapes

- **Lifecycle middleware** (`Middleware` trait) — observes and optionally
  mutates values flowing past fixed points: `before_agent` / `after_agent`,
  `before_model` / `on_model_delta` / `after_model`, `before_tool` /
  `on_tool_delta` / `after_tool`, and `on_error`. Every hook has a no-op
  default, is async, and returns `Result<()>`; an `Err` short-circuits the
  stack.
- **Wrap ("around-call") middleware** (`ModelMiddleware`, `ToolMiddleware`) —
  surrounds the inner call with a `next` handler (`ModelHandler` /
  `ToolHandler`) instead of only observing before/after values. A wrap
  middleware can proceed (call `next.run(..)` once), short-circuit (never call
  it, returning a replacement `MiddlewareModelOutcome` / `MiddlewareToolOutcome`
  directly), retry (call `next.run(..)` in a loop), or fall back (call it, then
  substitute a response on error). This is the only extension point expressive
  enough for retry/fallback/caching semantics.

Both shapes are composed by `MiddlewareStack`, which holds three ordered
lists: `Middleware`, `ModelMiddleware`, `ToolMiddleware`.

## Onion ordering

`before_*` lifecycle hooks run in registration order; `after_*` hooks run in
**reverse** registration order, so the first-registered middleware is the
outermost layer — it sets up first and tears down last. Wrap middleware
compose the same way as a nested onion around the real model/tool call
(`ModelBaseCall` / `ToolBaseCall`), with the first-registered middleware
outermost and the base call innermost.

Every per-middleware hook invocation is bracketed by
`AgentEvent::MiddlewareStarted` / `MiddlewareCompleted` events emitted through
the `RunContext`, so hook activity is independently observable via the event
sink regardless of what a middleware itself records. The pair is always
balanced: a hook that returns `Err` still emits its `MiddlewareCompleted`
before the error short-circuits the stack, so an observer never sees a dangling
`Started`. (The per-delta streaming hook `on_model_delta` is the one deliberate
exception — it emits no bracketing events, to stay cheap on the token hot path.)

## Error handling

The first hook that returns `Err` short-circuits the stack: every
middleware's `Middleware::on_error` is invoked (so all middleware get a chance
to log/redact/react), then the *original* error is returned to the caller.
Errors raised from `on_error` itself are ignored — they cannot mask the root
cause or replace it with a different error.

## Public surface

- `Middleware<State, Ctx = ()>` — the lifecycle trait described above.
- `ModelMiddleware<State, Ctx>` / `ToolMiddleware<State, Ctx>` — the wrap
  traits, each with a single `wrap_model` / `wrap_tool` method.
- `MiddlewareStack<State, Ctx>` — the composer; `push` / `push_model` /
  `push_tool` register middleware, `run_before_agent` / `run_after_agent` /
  `run_wrapped_model` / `run_wrapped_tool` / etc. run them.
- `AgentRun` — the accumulated result of a run (messages, final response,
  structured output, usage, call/step counters) threaded through
  `after_agent`.
- `ModelHandler<'a, State, Ctx>` / `ToolHandler<'a, State, Ctx>` — the `next`
  handle passed to wrap middleware; `.run(ctx, state, request/call)` proceeds
  to the next layer.
- `ModelBaseCall<State, Ctx>` / `ToolBaseCall<State, Ctx>` — the innermost real
  call, supplied by the agent loop as the base of the onion.
- `MiddlewareModelOutcome` / `MiddlewareToolOutcome` — `#[non_exhaustive]`
  result enums wrap middleware resolve to; construct via `.into()` from a
  `ModelResponse` / `ToolResult`.

### Built-in middleware (`library/`)

- `LoggingMiddleware` — observation-only; counts how often each lifecycle hook
  fires (`HookCounts`, readable via `.counts()`). Emits no events of its own —
  the stack already emits start/completed events.
- `MessageTrimMiddleware` — replaces `request.messages` with the result of
  `summarization::trim_messages` under a configured `TrimStrategy` in
  `before_model`.
- `ContextCompressionMiddleware` — consults a `SummarizationPolicy` in
  `before_model` and is a complete no-op below the context-window threshold;
  above it, condenses older messages via a `Summarizer` (default
  `ConcatSummarizer`) into a summary message, keeps the recent window and
  system messages verbatim, records a `SummaryRecord`, and emits
  `AgentEvent::Compressed`.
- `PromptCacheGuardMiddleware` — computes the request's
  `cache::PromptCacheLayout` in `before_model` and records a
  `CacheLayoutEvent` whenever the cacheable prefix changes from the previous
  call, making KV-cache/prompt-cache regressions observable.
- `UsageAccountingMiddleware` — folds each `response.usage` into a running
  `UsageTotals` in `after_model`, readable via `.totals()`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: traits, `MiddlewareStack`, built-in middleware structs. |
| `mod.rs` | Behavioral code: `AgentRun` helpers, the stack runner. |
| `library/` | Constructors and impls for every built-in middleware, split by concern: `resilience.rs` (retry/timeout/fallback/rate-limit), `budget.rs` (token/cost tracking and enforcement), `tool_policy.rs` (allowlisting, policy, dynamic/contextual selection, human approval), `context.rs` (message trim, summarization-based compression, prompt-cache guard), `observe.rs` (structured-output validation, dynamic prompt, redaction, tracing, logging, usage accounting). |
| `test.rs` | Unit tests (ordering, short-circuiting, each built-in middleware). |

## Operational constraints

- Hooks are invoked with `&mut RunContext<Ctx>` and a shared `&State`; a
  middleware must not assume exclusive ownership of `State` — it is read-only
  from the middleware's perspective.
- Because `on_error` failures are swallowed, a middleware must not rely on
  `on_error` for anything beyond best-effort side effects (logging, metrics).
- Wrap middleware that retry `next.run(..)` are responsible for their own
  budget/backoff; the stack does not cap retry attempts.
