# harness::agent_loop

The default model-tool-model agent loop: the innermost turn of the recursive
harness.

This loop is where one model call is driven to completion. Because a whole
harness can be exposed as a tool (`harness::subagent::SubAgentTool`), the
tools this loop executes may themselves be other agents — "a model calling a
model" is just this loop nested inside one of its own tool calls. Each
invocation runs inside a `RunContext` that tracks recursion depth, fans
usage/cost up to a parent run, and observes cooperative cancellation and
steering at safe checkpoints.

The module is implemented as inherent methods on
`harness::runtime::AgentHarness<State, Ctx>` rather than a free function or
separate type — there is no standalone "AgentLoop" struct to construct.

## Lifecycle

1. Build a `RunContext` from the `RunConfig` and emit `AgentEvent::RunStarted`.
2. Run `before_agent` middleware.
3. Repeatedly:
   - enforce the model-call cap and wall-clock deadline (fail-closed),
   - build the `ModelRequest` from the working messages, registered tool
     schemas, and the policy's default response format,
   - run `before_model` middleware, emit `AgentEvent::ModelStarted`,
   - resolve and invoke the model with retry + fallback,
   - run `after_model` middleware, emit `AgentEvent::ModelCompleted`, fold
     usage into the `AgentRun`, append the assistant message,
   - if the assistant requested tools, execute them (enforcing the tool-call
     cap, running `before_tool`/`after_tool`, emitting tool events) and append
     the tool results, then continue,
   - otherwise extract structured output when configured and break.
4. Run `after_agent` middleware and emit `AgentEvent::RunCompleted`.

On any error the loop emits `AgentEvent::RunFailed`, fans the error out
through `on_error` middleware, and returns the error.

## Tool execution: serial vs. concurrent

A turn's tool calls are driven in three phases — serial **admission**
(cancellation/deadline/limit checks, `before_tool`, unknown-tool policy,
schema validation, `ToolStarted`), **execution**, and a serial **fold** in
original call order (`after_tool`, `ToolCompleted`, transcript append).
The fold also records a result's host-only `metadata` on the event and on
`AgentRun::tool_metadata`, and hands back its `follow_up` content as a user
message that the batch driver appends only after the batch's last tool row
(B2) — a provider requires every tool row to follow its assistant row
directly, so follow-ups never interleave with tool rows. The same holds for
the deferred-resume batch in `apply_deferred_results`.

Execution runs concurrently only when *all* of the following hold: the turn
requests two or more tools, zero tool-wrap middleware (`ToolMiddleware`) is
registered, and every call's tool reports `Tool::is_concurrency_safe() ==
true` (the trait default is `false`, so a tool must opt in). See
`should_execute_tools_concurrently` and `batch_is_canonical_parallel_safe` in
`tools.rs`. Tool-wrap middleware holds `&mut RunContext` across each wrapped
call — part of its public contract — so its presence keeps the historical
serial path. Lifecycle middleware does **not** force the serial path: every
`before_tool` hook runs during serial admission, which completes in full for
every call in the batch before any concurrent future is built, so there is
nothing left for a lifecycle middleware to mutate once execution starts
(I-8; this used to force serial execution unconditionally).

When concurrency does trigger, the batch runs via `futures::stream::iter(..)
.buffered(n)` — not an unbounded `join_all` — where `n` is
`RunPolicy::limits.max_tool_concurrency` (unbounded, i.e. every eligible call
starts at once, when `None`, the default). `buffered` yields results in input
order, same as `join_all` did, so downstream folding is unaffected; it just
caps how many calls are in flight simultaneously. Turn latency is then the
slowest *batch* of at most `n` tools instead of the slowest single tool. In
both modes results are attached to their original `tool_call_id` in the
calls' original order, every call's `ToolStarted` precedes its
`ToolCompleted`, and `ToolCompleted` events are emitted in call order. The
first failing call (in call order) fails the turn; in concurrent mode
already-launched siblings run to completion before the error surfaces. See
`tools.rs` for the full design notes.

## Limits

Model- and tool-call caps come from `runtime::RunPolicy::limits` and are
enforced *before* each call, returning `TinyAgentsError::LimitExceeded`. The
wall-clock deadline (from the run config) is checked each iteration and
surfaces as `TinyAgentsError::Timeout`. The run context's own
`limits::LimitTracker` is also advanced so its counters stay consistent with
the enforced caps.

## Cancellation and wall-clock bounding

Every host/provider I/O boundary on the loop path (model resolution, budget
admission and usage recording, tool authorization, tool-output screening,
host turn preparation, the unary provider call) races cooperative
cancellation against an optional wall-clock deadline through one shared
helper, `context::RunContext::bounded(deadline, fut, timeout_message)`,
instead of each call site copying its own `tokio::select! { biased; _ =
cancelled() => .., _ = timeout(remaining, fut) => .. }` block (R-1 from
`docs/runtime-comparison/code-review-harness.md`). `timeout_message` is a
closure so the call-specific message is only built on the timeout path, not
on every call. The streaming provider loop's per-chunk pull races
cancellation against `stream.next()` directly — its future yields an
`Option`, not a `Result`, so it does not fit `bounded`'s signature and stays
a bespoke `select!`.

## Backoff

Retry backoff durations are *computed* via
`retry::RetryPolicy::backoff_for_attempt`, but whether the loop actually
sleeps for that duration is opt-in — off by default (keeping tests fast and
deterministic) and enabled per policy via
`retry::RetryPolicy::with_backoff_sleep`. A real provider integration retries
after a genuine, growing delay while unit tests stay sleep-free.

## Truncated-empty recovery

Local reasoning models (for example `qwen3` via Ollama) intermittently spend
their entire token budget on the hidden reasoning channel and return
`finish_reason == "length"` with no visible text, no tool calls, and no
structured output — a result useless to every caller. Before finalizing such a
turn (and before structured extraction, which would otherwise fail on the empty
completion) the loop retries the model call up to
`runtime::RunPolicy::truncated_empty_retries` times (default `1`, so two
attempts total). Each retry drops the useless assistant row, doubles the
request's `max_tokens` when one was set — clamped at 4x the original cap, and a
deliberate override of the per-turn output cap that caused the truncation — or
re-issues unchanged when no budget was set (the failure is stochastic, so a
plain retry still helps). Each attempt counts as a model call and emits
`AgentEvent::RetryScheduled`. The retry runs *before*
`RunPolicy::error_on_empty_response`; only once the retries are exhausted does
that guard (if enabled) turn the still-blank final into
`TinyAgentsError::EmptyResponse`. Set `truncated_empty_retries` to `0` to
restore exact-replay behavior. The recovery lives in the shared `run_loop`, so
it applies identically to the unary and streaming paths.

A provider can also end a stream normally after emitting only reasoning, with
no visible text or tool call. Hosts may set
`RunPolicy::empty_response_retries` to retry that non-truncated blank result;
it defaults to `0` because the extra model call may be billed. The retry drops
the unusable assistant row, keeps the original output-token cap, counts toward
the model-call limit, and emits `RetryScheduled`. It never promotes reasoning
into visible answer text. Such a blank result is excluded from the local
response cache, and a prior blank cache hit is ignored when retries are
enabled, so the next attempt reaches the provider. A still-empty response
follows the caller's normal
`error_on_empty_response` policy.

## Public surface

- `AgentHarness::invoke(state, ctx_data, config, input) -> Result<AgentRun>` —
  runs the loop, returns only the accumulated `AgentRun`.
- `AgentHarness::invoke_with_status(..) -> Result<AgentLoopResult>` — same run,
  also returns a compact `HarnessRunStatus` snapshot (phase, counters, timing,
  error summary) alongside the `AgentRun`.
- `AgentLoopResult { run: AgentRun, status: HarnessRunStatus }` — the richer
  return type; also returned by `AgentHarness::invoke_in_context_with_status`.
- `AgentHarness::invoke_collecting_partial(..) -> PartialRunOutcome` (and its
  `invoke_in_context_collecting_partial` / streaming counterparts) — like
  `invoke`, but never discards the accumulated `AgentRun` on error; useful for
  inspecting, repairing, or resuming a run that hit a limit or a tool failure.
- `AgentHarness::invoke_streaming` / `invoke_streaming_default` /
  `invoke_streaming_in_context[_with_status]` — the streaming counterparts of
  the `invoke*` family: each model call goes through
  `ChatModel::stream` instead of `ChatModel::invoke`, threading deltas through
  every middleware's `on_model_delta` hook. Visible text also crosses the
  streamed-text stall detector; repeated process narration drops the provider
  stream and returns non-retryable `GenerationStalled`.
- `AgentHarness::invoke_stream` / `invoke_stream_in_context` — a caller-facing
  event stream (`stream.rs`): yields every `AgentEvent` emitted during the run
  as `AgentStreamItem::Event`, then a single terminal
  `AgentStreamItem::Completed`/`Failed`. Driving the loop and consuming the
  stream are the same task, so a caller that stops polling pauses the run.
- `AgentStreamItem { Event(EventRecord), Completed(Box<AgentRun>), Failed { error, run } }`
  — the item type yielded by `invoke_stream`.

## Errors

`TinyAgentsError::LimitExceeded` (model/tool cap reached),
`TinyAgentsError::Timeout` (wall-clock deadline elapsed),
`TinyAgentsError::GenerationStalled` (repetitive visible model stream),
`TinyAgentsError::ModelNotFound` (no model resolvable),
`TinyAgentsError::ToolNotFound` (model called an unregistered tool), or any
error surfaced by a model, tool, middleware, or structured-output extraction.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Module wiring: shared imports and the module-level doc comment. |
| `entry.rs` | Public entry points (`invoke`/`invoke_with_status`/`invoke_streaming*`/`invoke_collecting_partial`) and the shared `drive`/`drive_collecting` lifecycle wrapper. |
| `run_loop.rs` | The core loop body (`run_loop`), response-cache decision logic, and host budget/prompt-cache helpers. |
| `tools.rs` | Tool execution for one turn: serial admission, serial or concurrent execution, ordered fold. |
| `model_call.rs` | Cache-aware retry/fallback model dispatch, the streaming variant, host model resolution, and the innermost `ModelBaseCall`/`ToolBaseCall` impls the middleware wrap-onion terminates into. |
| `stream.rs` | Caller-consumable streaming entry point (`invoke_stream`/`invoke_stream_in_context`) that projects the run's `EventSink` into an `AgentStreamItem` stream. |
| `types.rs` | `AgentLoopResult`, `PartialRunOutcome`, and the private `LoopExit`. |
| `test.rs` | Unit tests (limits, retry/fallback, tool execution, structured extraction). |

## Operational constraints

- The loop assumes `state: &State` is safe to read concurrently with any
  nested sub-agent call — it never mutates it directly.
- Unregistered-tool calls fail closed per `runtime::UnknownToolPolicy`; there
  is no silent skip.
- Identifiers (`CallId`, `ComponentId`) are derived deterministically from the
  `RunConfig`, not randomly or from wall-clock time, so repeated calls with the
  same input and config produce the same ids.
