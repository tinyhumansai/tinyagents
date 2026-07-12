# harness::agent_loop

The default model-tool-model agent loop: the innermost turn of the recursive
(RLM-style) harness.

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

When a turn requests two or more tools and **no tool-wrap middleware**
(`ToolMiddleware`) is registered, execution runs concurrently (`join_all`),
so turn latency is the slowest tool instead of the sum. Tool-wrap middleware
holds `&mut RunContext` across each wrapped call — part of its public
contract — so its presence keeps the historical serial path. In both modes
results are attached to their original `tool_call_id` in the calls' original
order, every call's `ToolStarted` precedes its `ToolCompleted`, and
`ToolCompleted` events are emitted in call order. The first failing call (in
call order) fails the turn; in concurrent mode already-launched siblings run
to completion before the error surfaces. See `tools.rs` for the full design
notes.

## Limits

Model- and tool-call caps come from `runtime::RunPolicy::limits` and are
enforced *before* each call, returning `TinyAgentsError::LimitExceeded`. The
wall-clock deadline (from the run config) is checked each iteration and
surfaces as `TinyAgentsError::Timeout`. The run context's own
`limits::LimitTracker` is also advanced so its counters stay consistent with
the enforced caps.

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

## Public surface

- `AgentHarness::invoke(state, ctx_data, config, input) -> Result<AgentRun>` —
  runs the loop, returns only the accumulated `AgentRun`.
- `AgentHarness::invoke_with_status(..) -> Result<AgentLoopResult>` — same run,
  also returns a compact `HarnessRunStatus` snapshot (phase, counters, timing,
  error summary) alongside the `AgentRun`.
- `AgentLoopResult { run: AgentRun, status: HarnessRunStatus }` — the richer
  return type; the only public type this module owns beyond the
  `AgentHarness` methods themselves.

## Errors

`TinyAgentsError::LimitExceeded` (model/tool cap reached),
`TinyAgentsError::Timeout` (wall-clock deadline elapsed),
`TinyAgentsError::ModelNotFound` (no model resolvable),
`TinyAgentsError::ToolNotFound` (model called an unregistered tool), or any
error surfaced by a model, tool, middleware, or structured-output extraction.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Module wiring: shared imports and the module-level doc comment. |
| `entry.rs` | Public entry points (`invoke`/`invoke_with_status`/`invoke_streaming*`) and the shared `drive` lifecycle wrapper. |
| `run_loop.rs` | The core loop body (`run_loop`) and response-cache decision logic. |
| `tools.rs` | Tool execution for one turn: serial admission, serial or concurrent execution, ordered fold. |
| `model_call.rs` | Cache-aware retry/fallback model dispatch, the streaming variant, and the innermost `ModelBaseCall`/`ToolBaseCall` impls the middleware wrap-onion terminates into. |
| `types.rs` | `AgentLoopResult`. |
| `test.rs` | Unit tests (limits, retry/fallback, tool execution, structured extraction). |

## Operational constraints

- The loop assumes `state: &State` is safe to read concurrently with any
  nested sub-agent call — it never mutates it directly.
- Unregistered-tool calls fail closed per `runtime::UnknownToolPolicy`; there
  is no silent skip.
- Identifiers (`CallId`, `ComponentId`) are derived deterministically from the
  `RunConfig`, not randomly or from wall-clock time, so repeated calls with the
  same input and config produce the same ids.
