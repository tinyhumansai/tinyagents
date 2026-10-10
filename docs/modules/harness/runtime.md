# Harness Runtime: Tools, Agent Loop, Middleware, Host-Owned State

Continues from [`README.md`](README.md): tool registry, agent loop,
middleware, host-owned state, and stores.

## Tool Registry

The tool registry owns available tools and their schemas.

```rust
pub struct ToolRegistry<State, Ctx = ()> {
    tools: HashMap<ToolName, Arc<dyn Tool<State, Ctx>>>,
}

#[async_trait]
pub trait Tool<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> ToolSchema;

    async fn call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call: ToolCall,
    ) -> Result<ToolResult>;
}
```

Tool schema requirements:

- name
- description
- JSON schema compatible input shape
- optional output schema
- safety metadata
- timeout override
- retry override
- model-visible flag for each argument
- injected-runtime argument declarations that are hidden from model schemas
- side-effect and idempotency metadata
- confirmation policy for destructive operations
- artifact output policy

Tool call requirements:

- `id`
- `name`
- `arguments`
- provider metadata
- originating model call id
- validation status
- retry attempt

Tool result requirements:

- `tool_call_id`
- `name`
- content
- raw structured value
- elapsed time
- error flag
- artifact references
- user-visible summary
- redacted event payload

Tool names should default to ASCII `snake_case`. The registry should reject
duplicate names and invalid names.

## Agent Loop

The default loop is the LangChain-style model-tool loop:

```text
input messages
  -> build request
  -> call model
  -> if assistant has tool calls:
       validate tool calls
       execute tools
       append tool messages
       repeat
  -> final assistant message
```

Detailed lifecycle:

1. Create `RunConfig` and `RunContext`.
2. Registered host middleware may load short-term memory for `thread_id`.
3. Normalize input into messages.
4. Apply prompt templates and dynamic context.
5. Select model.
6. Select exposed tools.
7. Run `before_model` middleware, including prompt/cache-layout guards and
   pre-call compression.
8. Invoke or stream the model through `wrap_model` middleware.
9. Run `on_model_delta` middleware for streamed chunks.
10. Run `after_model` middleware, including post-call compression and any
    host-selected summary persistence.
11. Emit model events and append assistant message.
12. If tool calls exist, validate name, schema, and limits.
13. Run `before_tool` middleware per call.
14. Execute tools — concurrently only when *all* of: the turn has two or more
    calls, every registered tool-wrap (`ToolMiddleware`) reports
    `concurrent_safe() == true` (wraps take `&RunContext` and run inside each
    concurrent call; `false` forces the serial path), and every call's tool reports `is_concurrency_safe() ==
    true` (the trait default is `false`, so concurrency is opt-in per tool);
    see `should_execute_tools_concurrently` in
    `crates/tinyagents-harness/src/agent_loop/tools.rs`. Lifecycle middleware
    no longer forces the serial path: every `before_tool` hook runs during
    serial admission, which completes in full before any concurrent future is
    built, so there is nothing left for it to mutate once execution starts.
    When the concurrent path runs, it is bounded by
    `RunLimits::max_tool_concurrency` (`futures::stream::iter(..)
    .buffered(n)`; `None`, the default, is unbounded). Results always fold
    back in original call order.
15. Run `after_tool` middleware per result.
16. Append tool messages.
17. Repeat until no tool calls remain.
18. Validate structured output if configured.
19. Registered host middleware may persist short-term memory.
20. Emit final event and return `AgentRun`.

Hard limits:

- `max_model_calls`
- `max_tool_calls`
- wall-clock timeout
- per-call timeout
- retry budget

The loop must fail closed when a limit is reached.

A per-call ceiling (`RunLimits::max_model_call_ms`) firing raises
`TinyAgentsError::CallTimeout`, distinct from a run-deadline
`TinyAgentsError::Timeout`: `CallTimeout` is retryable and is still consulted
against the fallback chain (the model wedged, not the run), while `Timeout`
is terminal (the run itself is out of wall-clock budget).

Streaming model calls also have an inactivity timeout
(`RunLimits::stream_idle_timeout_ms`, default 120 s, `None`/`Some(0)`
disables). It applies only **after the first output event** and is a deadline
that each further output event pushes to `now + idle`; the stream-opened marker
and usage updates never extend it. Before the first output event there is no
separate bound by default (`stream_first_event_timeout_ms: None`), because
hidden-reasoning models and local CPU prefill are legitimately silent for many
minutes: that wait is limited only by `max_model_call_ms` and the run deadline.
Set `Some(ms)` to opt in to a first-output bound. A stream that goes silent
fails with a retryable `CallTimeout`, so the normal retry/fallback path
applies. After `RunLimits::max_consecutive_stream_idle_timeouts` (default 5,
`None`/`Some(0)` disables) idle timeouts in a row on one model (since its last
output event) the circuit breaker stops retrying that model with
`TinyAgentsError::StreamIdleTimeout`; the fallback chain is still consulted with a
fresh count per model, and the run fails with that error only when the chain is
exhausted. Because any output event resets the count, the breaker is aimed at
streams that stall before producing output (pair it with
`stream_first_event_timeout_ms`). Non-streaming calls are bounded by
`max_model_call_ms` only.

Step 12's text-dialect recovery (parsing `<tool_call>`-style markup out of
an assistant's visible text through the `tinytools-agent` grammars, both on
the streamed deltas and on the terminal response) always runs under a forced
text dialect (`RunPolicy::tool_dialect` of `Xml` / `Pformat`, or `Auto`
falling back to Xml for a model without native tool calling; `Python` and
`Typescript` are forced text dialects too) — there, parsing text is the
protocol. Under a native dialect it is gated by
`RunPolicy::text_dialect_recovery` (`TextDialectRecovery::Off | On | Auto`,
default `Auto`): it only runs when the resolved model's profile does not
report native tool calling, and it always skips markup that appears only
inside a fenced code block. A model that quotes the syntax while explaining
it (or answers under a model that *does* support native tool calling) is
never executed as a real call.

Every safe checkpoint above — before a model call is dispatched, after the
model response, and after tool execution — also drains any pending
`MiddlewareControl` (see
[middleware control outcomes](middleware.md#middleware-control-a1)) and, at
the tool-execution checkpoint, first asks
`MiddlewareStack::any_should_stop_after_turn` whether any registered
middleware wants to end the run based on the whole turn's results. Step 19's
structured-output validation is the output-validation retry loop (A3, see
[structured-output.md](structured-output.md#error-policy-the-output-validation-retry-loop-a3)):
a schema failure or an `OutputValidator` rejection re-asks the model (bounded
by `RunPolicy::output_retry.max_attempts`) instead of failing the run on the
first attempt. Step 12's tool-call handling additionally honors
`RunPolicy::end_strategy` (A6, see
[structured-output.md](structured-output.md#endstrategy-output-tool--function-tools-in-one-turn-a6))
when a turn returns both a structured-output tool call and real tool calls.

### Loop exits: finished, limit stop, paused, deferred

The loop distinguishes four deliberate stops. A normal finish and a
`LimitBehavior::StopWithPartial` limit stop complete the run
(`HarnessRunStatus` `Completed`, `AgentEvent::RunCompleted`). A steering
**pause** sets `AgentRun::paused` and a **deferred** tool batch (A2) sets
`AgentRun::deferred`; both report the run `Interrupted`, emit
`ControlApplied { control: "paused" | "deferred" }`, leave `final_response`
unset, and are resumed from `run.messages`. The working transcript is written
onto the `AgentRun` on every exit path, including errors.

Resuming a deferred run is `AgentHarness::resume_deferred(state, ctx,
run.messages, DeferredToolResults)` (sugar over
`RunContext::with_deferred_results`), or on the hosted path
`AgentTurnRequest::new(agent, run.messages).with_deferred_results(results)`.
The loop applies the decisions to the unanswered tool calls on the last
assistant row *before* its first model call, then proceeds normally. See
[tool.md](tool.md#deferred-tool-calls-approval-and-external-execution-a2)
for the triggers, decision vocabulary, and the inline `DeferredToolHandler`.

**Durability is the host's responsibility.** The harness does not write to
the session run ledger (`tinyagents-session` depends on the harness, not the
other way round), and the only state a resume needs is `run.messages` plus
`run.deferred` — both `serde` types. Persist them wherever the run's other
state lives; `tinyagents_session::run_ledger::AgentRun::checkpoint` (a JSON
column keyed by run id, alongside a `Paused`/`Interrupted` status) is the
natural slot, and a host that also wants per-call approval rows keeps those
in its own tables keyed by `DeferredToolRequests` call ids.

### Queued steering and follow-ups (A4)

A run can carry a `RunQueueHandle` (`Arc<RunQueue<Message>>`, attached with
`RunContext::with_run_queue`). Anything with a clone of the handle — a UI, a
parent agent, a tool — pushes `Message`s onto one of three lanes while the
run is in flight, and the loop drains them only at safe boundaries:

| Lane | Drained when | Effect |
|---|---|---|
| `Steer` | after a tool batch's results are all on the transcript (never mid-batch), and at a natural finish before any follow-up | appended to the transcript; the next model call sees it |
| `Followup` | at a natural finish, only when no steer is pending | appended; the loop runs another turn instead of returning |
| `Collect` | once, at run end, on every exit path | delivered on `AgentRun::collected`; never enters the transcript |

`RunPolicy::queue_mode` picks how many items a boundary takes:
`QueueMode::All` (default) applies every pending item; `OneAtATime` applies
the oldest and leaves the rest for the next boundary. Each application emits
`AgentEvent::QueuedMessageApplied { lane, count, first_index, messages }` (`messages` follows the
capture policy per message: `tool` messages under `PayloadCapture::tool_io`,
all others under `PayloadCapture::model_io`). A "natural finish" is
the model producing a final answer (including a structured-output finish
under `EndStrategy::Early`/`Graceful`); a middleware `StopWithFinal` /
`JumpTo(End)`, a limit stop, a pause, or a deferral is terminal and leaves
the queue untouched for the host. Follow-up turns count against
`max_model_calls` like any other, which bounds a host that keeps queueing.

The queue is content injection only. `SteeringHandle` / `SteeringCommand`
(pause, resume, cancel, `InjectMessage`, `Redirect`) is the unchanged control
channel and is drained at its own checkpoint before each model call; the two
mechanisms are independent. A child context never inherits its parent's
queue — a queue has no per-run addressing, so a child draining it would
steal the parent's messages.

```rust
let queue: RunQueueHandle = Arc::new(RunQueue::new());
let ctx = RunContext::new(RunConfig::new("r"), ()).with_run_queue(queue.clone());
// ...from another task while the run is in flight:
queue.push(QueueLane::Steer, Message::user("prefer the cheaper option")).await;
queue.push(QueueLane::Followup, Message::user("now summarize what you did")).await;
```

### `RunPolicy` fields added by Phase 2 (A1/A3/A4/A6)

| Field | Type | Default | Purpose |
|---|---|---|---|
| `output_retry` | `OutputRetryPolicy { max_attempts: u8, message_template: String }` | `max_attempts = 1` | Bounds the output-validation retry loop. |
| `end_strategy` | `EndStrategy` | `Graceful` | Resolves output-tool + function-tool turns. |
| `structured_strategy_override` | `Option<StructuredStrategyOverride>` | `None` | Forces `Prompted`/`ToolCallUnion` for `ResponseFormat::Auto`. |
| `queue_mode` | `QueueMode` | `All` | How many `RunQueue` items each boundary applies (A4). |

`AgentHarness::with_output_validator(Arc<dyn OutputValidator<State, Ctx>>)`
registers the validator the output-retry loop consults; only one may be
installed (calling it again replaces the previous one).
`AgentHarness::with_deferred_tool_handler(Arc<dyn DeferredToolHandler>)`
(A2) likewise installs the single inline resolver for deferred tool calls.

## Middleware

Middleware is the main extension point for behavior that cuts across providers,
tools, and graph nodes.

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_model(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &mut ModelRequest,
    ) -> Result<()>;

    async fn on_model_delta(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        delta: &mut ModelDelta,
    ) -> Result<()>;

    async fn after_model(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        response: &mut ModelResponse,
    ) -> Result<()>;

    async fn before_tool(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call: &mut ToolCall,
    ) -> Result<()>;

    async fn on_tool_delta(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        delta: &mut ToolDelta,
    ) -> Result<()>;

    async fn after_tool(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> Result<()>;

    async fn on_error(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        error: &TinyAgentsError,
    ) -> Result<()>;
}
```

Middleware ordering is stable and explicit. Middleware runs in registration
order for `before_*` hooks, registration order for streaming delta hooks, and
reverse order for `after_*` hooks. Wrap hooks should surround the full model or
tool operation when middleware needs setup, streaming inspection, and teardown
as one unit.

Built-in middleware candidates:

- tracing middleware
- retry middleware
- timeout middleware
- model fallback middleware
- token-bucket rate limiter middleware
- prompt cache layout guard middleware
- message trimming middleware
- summarization middleware
- context compression middleware
- transcript compression middleware
- retrieval compression middleware
- streaming delta compression middleware
- output compression middleware
- context editing middleware
- tool allowlist middleware
- dynamic tool selection middleware
- guardrail middleware
- PII detection/redaction middleware
- human-in-the-loop middleware
- shell/filesystem privilege boundary middleware
- structured output validator
- rate limiter

Wrap hooks should exist in addition to before/after hooks. A wrap hook receives a
request plus a handler and can call the handler, replace the request, retry,
fallback to another model/tool, short-circuit with a response, or return a
control command. Before/after hooks are simpler and should remain available for
common mutation and observation cases.

## Host-Owned Memory And Stores

Memory and storage are related but not the same feature. Hosts own conversation
semantics and choose their memory policy; the harness exposes store primitives
and middleware seams rather than a harness-owned memory implementation.

Memory has two layers conceptually:

```text
short-term memory: thread-scoped conversation state
long-term store: cross-thread application data
```

Short-term memory:

- keyed by `thread_id`
- loaded before an agent loop
- updated after successful loop completion
- optionally trimmed or summarized
- useful for conversation continuity

Stores:

- available through `RunContext`
- namespaced
- typed where possible
- usable by tools and middleware
- not automatically injected into prompts unless middleware does it
- reusable by memory, event recording, tool artifacts, and web UIs

Memory is host policy, implemented as around-agent middleware instead of a
harness-owned storage interface:

```rust
#[async_trait]
impl AgentMiddleware<AppState, AppContext> for MemoryMiddleware {
    async fn wrap_agent(&self, ctx: &mut RunContext<AppContext>, state: &AppState,
        mut request: AgentRequest, run: &mut AgentRun,
        next: AgentHandler<'_, AppState, AppContext>) -> Result<()> {
        request.input.splice(0..0, self.load(ctx).await?);
        let result = next.run(ctx, state, request, run).await;
        let save_result = self.save(ctx, run).await;
        match (result, save_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(save_error)) => Err(save_error),
            (Err(run_error), Ok(())) => Err(run_error),
            (Err(run_error), Err(save_error)) => {
                self.record_persistence_failure(ctx, &save_error);
                Err(run_error)
            }
        }
    }
}
```

`record_persistence_failure` is host-defined telemetry (for example, an event
or log). It must not replace a typed agent error: callers need the failure from
the run to decide whether to retry, repair, or report it.

The storage layer should be a separate harness feature:

```rust
#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: StoreKey) -> Result<Option<StoreValue>>;
    async fn put(&self, key: StoreKey, value: StoreValue) -> Result<()>;
    async fn delete(&self, key: StoreKey) -> Result<()>;
    async fn scan(&self, prefix: StoreKeyPrefix) -> Result<Vec<StoreRecord>>;
}

#[async_trait]
pub trait AppendStore: Send + Sync {
    async fn append(&self, stream: StoreStream, value: StoreValue) -> Result<StoreOffset>;
    async fn read_from(&self, stream: StoreStream, offset: StoreOffset) -> Result<Vec<StoreRecord>>;
}

pub enum StoreValue {
    Json(serde_json::Value),
    Bytes(Vec<u8>),
    Text(String),
}
```

Initial store backends:

- `InMemoryStore`: deterministic tests and examples.
- `JsonlStore`: append-only local development, replayable event logs, and cheap
  debugging.
- `FileStore`: local artifacts such as tool outputs, provider payload snapshots,
  and prompt fixtures.
- `MongoStore`: durable application/runtime records for server deployments.

Later store backends:

- SQLite for single-node durable local apps.
- Postgres for multi-tenant production apps.
- S3-compatible blob store for large artifacts.
- Redis for short-lived cache/session data.

Store data classes:

- run records
- thread records
- normalized messages
- event envelopes
- tool call records
- model call records
- structured outputs
- user/application memory
- tool artifacts and blobs

Backend selection should be per store namespace:

```rust
let stores = StoreRegistry::new()
    .register("events", JsonlStore::new("./data/events.jsonl"))
    .register("threads", MongoStore::new(mongo, "threads"))
    .register("artifacts", FileStore::new("./data/artifacts"));
```

Store events should flow through `harness::events` or the registry event bus:

- `store.read`
- `store.write`
- `store.append`
- `store.delete`
- `store.error`

Sensitive store fields must support redaction before event emission.
