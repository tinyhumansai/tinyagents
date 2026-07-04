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
   - if the assistant requested tools, execute each (enforcing the tool-call
     cap, running `before_tool`/`after_tool`, emitting tool events) and append
     the tool results, then continue,
   - otherwise extract structured output when configured and break.
4. Run `after_agent` middleware and emit `AgentEvent::RunCompleted`.

On any error the loop emits `AgentEvent::RunFailed`, fans the error out
through `on_error` middleware, and returns the error.

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
