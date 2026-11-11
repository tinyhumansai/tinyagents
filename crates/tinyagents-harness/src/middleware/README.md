# harness::middleware

Cross-cutting extension points that wrap agent, model, and tool execution.

In the recursive harness a sub-agent or sub-graph is just another
agent loop, so the same before/after hooks bracket the parent run *and* every
nested model/tool/agent call beneath it. That uniform wrapping is what lets
concerns like tracing, usage/cost roll-up, and guardrails compose consistently
as models call models and graphs run graphs.

## Extension shapes

- **Lifecycle middleware** (`Middleware` trait) — observes and optionally
  mutates values flowing past fixed points: `before_agent` / `after_agent`,
  `before_model` / `on_model_delta` / `after_model`, `before_tool` /
  `on_tool_delta` / `after_tool`, and `on_error`. Every hook has a no-op
  default, is async, and returns `Result<()>`; an `Err` short-circuits the
  stack.
- **Executor observation** (`Middleware::observe_tool_execution`) — a
  synchronous, read-only hook inside the terminal tool future. It receives the
  admitted outer identity plus the actual bound target and arguments after
  gates and wrappers. It records attempts, including retries and nested calls,
  without requesting control, rewriting calls, or claiming successful effects.
- **Around-agent middleware** (`AgentMiddleware`) — surrounds a complete run,
  receives the mutable partial `AgentRun`, and can rewrite initial input,
  prepare run-scoped resources, short-circuit, or finalize after either success
  or failure. Host-owned memory, workspace lifecycle, and session policy belong
  here.
- **Around-call middleware** (`ModelMiddleware`, `ToolMiddleware`) —
  surrounds the inner call with a `next` handler (`ModelHandler` /
  `ToolHandler`) instead of only observing before/after values. A wrap
  middleware can proceed (call `next.run(..)` once), short-circuit (never call
  it, returning a replacement `MiddlewareModelOutcome` / `MiddlewareToolOutcome`
  directly), retry (call `next.run(..)` in a loop), or fall back (call it, then
  substitute a response on error). This is the only extension point expressive
  enough for retry/fallback/caching semantics.

All shapes are composed by `MiddlewareStack`, which keeps independent ordered
lists for lifecycle, agent, model, and tool middleware.

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
`Started`. The per-delta hooks `on_model_delta` and `on_tool_delta`, and the
synchronous executor-entry observer, emit no bracketing events. These observers
can run repeatedly within one invocation and keep their own lightweight records.

## Error handling

The first hook that returns `Err` short-circuits the stack: every
middleware's `Middleware::on_error` is invoked (so all middleware get a chance
to log/redact/react), then the *original* error is returned to the caller.
Errors raised from `on_error` itself are ignored — they cannot mask the root
cause or replace it with a different error.

One failure delivers exactly one `on_error` per middleware. The stack marks the
`RunContext` when it fans a hook failure out, and a driver that also handles the
propagated error (the agent loop does, for failures raised by models, tools, or
the loop itself) skips its own dispatch for that error.

## Public surface

- `Middleware<State, Ctx = ()>` — the lifecycle trait described above.
- `AgentMiddleware<State, Ctx>` / `AgentHandler` / `AgentBaseCall` — the
  complete-run onion used for host-owned behavior and resource lifecycles.
- `ModelMiddleware<State, Ctx>` / `ToolMiddleware<State, Ctx>` — the wrap
  traits, each with a single `wrap_model` / `wrap_tool` method.
  `wrap_tool` and `ToolHandler::run` / `ToolBaseCall::call` take a shared
  `&RunContext` (previously `&mut`; migrate by dropping the `mut`), so a
  multi-call tool batch runs the wrap onion inside each concurrent call. A wrap
  that cannot tolerate overlap returns `false` from
  `ToolMiddleware::concurrent_safe` to keep batches serial. See
  `docs/modules/harness/middleware-tool-wrap.md`.
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

### Middleware types defined directly in this module (`types.rs`)

These structs are *defined* in `middleware/types.rs` (not `library/`) because
their fields (records, layout cache, counters) are considered part of this
module's core surface — but their constructors and trait impls actually live
in `library/context.rs` and `library/observe.rs` alongside the rest of the
built-in catalog:

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
- `MicrocompactMiddleware` — the micro-compaction companion to
  `ContextCompressionMiddleware`: blanks the bodies of older tool-result
  messages (keeping the newest `keep_recent` verbatim) instead of summarizing
  chat history, optionally gated on a token budget to preserve the
  prompt-cache prefix. It only rewrites the outgoing request and, with a token
  budget, only near the window. To expire old tool results by age in the stored
  transcript instead, see `result_expiry` (`RunPolicy::result_expiry`); the two
  compose, and microcompact then has fewer bodies left to blank.
- `PromptCacheGuardMiddleware` — computes the request's
  `cache::PromptCacheLayout` in `before_model` and records a
  `CacheLayoutEvent` whenever the cacheable segment ids or content change from
  the previous call. For canonical declared layouts, a trimmed or compacted
  history can lose its cached tail while retaining the stable prefix, so it
  does not raise this event. Layouts without a known message boundary check
  the full message stream for byte-prefix preservation; pure tail appends
  remain valid.
- `UsageAccountingMiddleware` — folds each `response.usage` into a running
  `UsageTotals` in `after_model`, readable via `.totals()`.

### Built-in middleware library (`library/`)

A much larger catalog of ready-to-use middleware — retry/timeout/fallback/rate
limiting, budget enforcement, tool policy/allowlisting/selection, human
approval, structured-output validation, dynamic prompts, redaction, and
tracing — lives in `library/` and is re-exported through this module. See
[`library/README.md`](library/README.md) for the full list.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: traits, `MiddlewareStack`, and the middleware listed above. |
| `mod.rs` | Behavioral code: `AgentRun` helpers, the stack runner. |
| `library/` | The built-in middleware library — see [`library/README.md`](library/README.md). |
| `mod_tests.rs` | Unit tests (ordering, short-circuiting, each middleware defined in this module). |

## Operational constraints

- Hooks are invoked with `&mut RunContext<Ctx>` and a shared `&State`; a
  middleware must not assume exclusive ownership of `State` — it is read-only
  from the middleware's perspective. The executor-entry observer and tool-wrap
  onion use shared `&RunContext`; their invocations can overlap.
- Because `on_error` failures are swallowed, a middleware must not rely on
  `on_error` for anything beyond best-effort side effects (logging, metrics).
- Wrap middleware that retry `next.run(..)` are responsible for their own
  budget/backoff; the stack does not cap retry attempts.

`before_model_dispatch` observes or transforms ephemeral content on the final
request after model-control admission and final profile resolution. It runs
before the model wrap pipeline, in registration order. Admission/control policy
belongs in `before_model_control`; final request observers do not request control.
