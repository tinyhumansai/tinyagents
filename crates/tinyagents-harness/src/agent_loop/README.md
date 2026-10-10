# harness::agent_loop

The default model-tool-model agent loop: the innermost turn of the recursive
harness.

This loop is where one model call is driven to completion. Because a whole
harness can be exposed as a tool (`tinyagents_orchestration::subagent::SubAgentTool`), the
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
requests two or more tools, every registered tool-wrap middleware
(`ToolMiddleware`) reports `concurrent_safe() == true`, and every call's tool reports `Tool::is_concurrency_safe() ==
true` (the trait default is `false`, so a tool must opt in). See
`should_execute_tools_concurrently` and `batch_is_canonical_parallel_safe` in
`tools.rs`. Tool-wrap middleware takes a shared `&RunContext`, so the wrap onion runs
inside each concurrent call; a wrap returning `concurrent_safe() == false`
keeps the historical serial path. Lifecycle middleware does **not** force the serial path: every
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

When a dropped or undecodable tool call needs recovery but no model call
remains, the normal limit policy still ends the run. No unused recovery
prompt is appended to the transcript and no `RetryScheduled` event is emitted.

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

## Calls written on a turn with no callable tool

A request can withdraw its tools — a final wrap-up call, a spent research
budget — or set `ToolChoice::None`, while its transcript is still full of tool
calls. A native model then has no tool channel and writes its next call as
plain text in its own markup (DeepSeek V4: `<｜DSML｜invoke name="shell">…`).
Declaring the tools with `tool_choice: "none"` does not prevent it. Taken as
the final answer, that markup ends the turn with a stray command where the
result should be.

On such a turn `TextRecovery::withholding` is in force: the streaming
scrubber and, for unary replies, `withhold_text_calls` remove any complete
call the tool-call grammars recognise from the visible text, keep reasoning,
and count it in `DroppedBlocks::withheld`. Nothing is dispatched — the
I-2 rule (prose never becomes a call when none could be accepted) holds. If a
call was withheld and a model call remains, the loop drops that assistant row
and re-prompts once with `WITHHELD_TOOL_CALL_NUDGE`, which says plainly that
tools are unavailable, emitting `ControlApplied { control:
"withheld_tool_call" }` and `RetryScheduled`. Re-prompts are bounded by
`RunPolicy::dropped_tool_call_nudges` and run before the empty-reply retries,
since a bare re-send of the same transcript leaks the same way. With no call
left, the scrubbed reply stands. Language-tagged fenced examples and bare JSON
answers are left untouched.

## Truncated-empty recovery

Local reasoning models (for example `qwen3` via Ollama) intermittently spend
their entire token budget on the hidden reasoning channel and return
a length stop (`finish_reason` of `length`, or `max_tokens` / `MAX_TOKENS`
as Anthropic and some gateways spell it — an empty `max_tokens` reply takes this
path too) with no visible text, no tool calls, and no structured output — a result useless to every caller. Before finalizing such a
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
`TinyAgentsError::EmptyResponse`.

A hosted reasoning model on a high effort setting can deliberate past the
boosted budget too. Once the retries are spent the loop does not finish on the
blank reply: up to `RunPolicy::truncated_empty_nudges` times (default `1`) it
drops the row, appends a user message saying the reply ran out of output tokens
while reasoning, and asks for the next tool call now, written incrementally (or,
on a turn with no callable tool, a short answer). It then continues the loop
at the boosted cap and emits `AgentEvent::ControlApplied`
(`truncated_empty_nudge`). Both the retry and the nudge are skipped when
`limits.max_model_calls` leaves no room for another call, so a run on its last
allowed call ends on the blank reply instead of failing with
`LimitExceeded`. Set `truncated_empty_retries` and `truncated_empty_nudges` to
`0` to restore exact-replay behavior. The recovery lives in the shared
`run_loop`, so it applies identically to the unary and streaming paths.

Neither a smaller cap nor a lower effort label reliably stops a hosted model
that deliberates past its cap; `reasoning.effort = none` does. So after a dead
call the retry or nudged call goes out with reasoning switched off
(`RunPolicy::truncated_empty_reasoning_fallback`, default `true`), at the same
cap (the cap was for the deliberation), and the nudge tells the model to do its
working-out in the workspace. The hold-off backs off per death (1, 2, 4, 8, 16
live calls without reasoning, no ceiling) and the configured effort always
returns once it is spent: kept off for good, a model spends the rest of a run
writing probe programs instead of the deliverable;
the switch is announced as `AgentEvent::ControlApplied` (`reasoning_fallback`).
The state is run-wide (`agent_loop/reasoning_fallback.rs`): it outlives the
turn that set it.

A request's `reasoning.budget_tokens` is a promise the provider may not keep,
and every streamed call measured that reasoned past its budget with nothing
visible went on to die at the output cap. The reasoning watchdog
(`RunPolicy::reasoning_watchdog`, default `RequestBudget`) ends such a call at
the budget instead, dropping the stream, and hands the loop the same
`finish_reason = length`, no-content response the cap would have produced, so
the recovery above runs after a fraction of the wait
(`AgentEvent::ControlApplied`, `reasoning_watchdog`). Visible text or a
tool-call fragment disarms it; unary calls are not bounded.

The reasoning a call dies in is usually real work (a correct derivation, cut
off), and every retry used to begin it again from nothing. The tail of that
reasoning (`RunPolicy::truncated_empty_carry_reasoning_chars`, default 8,000
characters, `0` carries nothing) rides into the transcript as a user message
ahead of the retry or nudged call, framed as the model's own interrupted notes
with the instruction to continue from there in code rather than re-derive
(`AgentEvent::ControlApplied`, `truncated_empty_reasoning_carried`). The
watchdog's synthetic dead response keeps the reasoning that streamed, so a call
it ends is carried the same way as one the cap ended.

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
  stream and returns non-retryable `GenerationStalled`. A stream that goes
  silent past `RunLimits::stream_idle_timeout_ms` after its first output event
  (a deadline advanced only by output events, not `Started`/`UsageDelta`)
  fails with retryable `CallTimeout`. The wait for the first output event is
  unbounded unless `stream_first_event_timeout_ms` opts in. After
  `max_consecutive_stream_idle_timeouts` in a row on one model, retries on it
  stop with `StreamIdleTimeout` and the fallback chain continues with a fresh
  count; the run fails with that error only when the chain is exhausted.
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
| `model_switch.rs` | Applies a steered `SwitchModel` to the turn request before binding resolution, and decides where a fallback walk starts for a switched model. |
| `stream.rs` | Caller-consumable streaming entry point (`invoke_stream`/`invoke_stream_in_context`) that projects the run's `EventSink` into an `AgentStreamItem` stream. |
| `types.rs` | `AgentLoopResult`, `PartialRunOutcome`, and the private `LoopExit`. |
| `mod_tests.rs` | Unit tests (limits, retry/fallback, tool execution, structured extraction). |

## Operational constraints

- The loop assumes `state: &State` is safe to read concurrently with any
  nested sub-agent call — it never mutates it directly.
- Unregistered-tool calls fail closed per `runtime::UnknownToolPolicy`; there
  is no silent skip.
- Identifiers (`CallId`, `ComponentId`) are derived deterministically from the
  `RunConfig`, not randomly or from wall-clock time, so repeated calls with the
  same input and config produce the same ids.
