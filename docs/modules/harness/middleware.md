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

TinyAgents should provide equivalent extension power without requiring users to
understand graph internals for normal harness usage.

## Responsibilities

- Provide stable middleware ordering.
- Support before/after hooks for observation and simple mutation.
- Support wrap hooks for replacement, retry, fallback, short-circuit, and
  human-interrupt behavior.
- Support streaming hooks for model deltas and tool progress so middleware can
  act during long-running calls.
- Support steering hooks so parent orchestrators or humans can safely guide
  agent loops and sub-agent runs without direct state mutation.
- Support prompt/cache-layout hooks so middleware can compress context without
  accidentally invalidating provider prompt/KV-cache prefixes.
- Allow middleware to modify model requests, tool calls, and responses.
- Allow middleware to emit events.
- Allow middleware to add local state updates without mutating unrelated state.
- Allow middleware to jump to model, tools, or end when used inside an agent loop.
- Translate middleware control outcomes into state-graph commands when a run is
  graph-backed.
- Expose errors to middleware for logging, redaction, fallback, or recovery.
- Keep middleware testable with fake models, tools, and event sinks.

## Hook Types

```rust
#[async_trait]
pub trait Middleware<State, Ctx = ()>: Send + Sync {
    async fn before_agent(&self, state: &State, ctx: &mut RunContext<Ctx>) -> Result<()>;
    async fn after_agent(&self, state: &State, ctx: &mut RunContext<Ctx>, run: &mut AgentRun) -> Result<()>;
    async fn before_steering(&self, state: &State, ctx: &mut RunContext<Ctx>, command: &mut SteeringCommand) -> Result<()>;
    async fn after_steering(&self, state: &State, ctx: &mut RunContext<Ctx>, outcome: &mut SteeringOutcome) -> Result<()>;

    async fn before_model(&self, state: &State, ctx: &mut RunContext<Ctx>, request: &mut ModelRequest) -> Result<()>;
    async fn before_model_stream(&self, state: &State, ctx: &mut RunContext<Ctx>, request: &mut ModelRequest) -> Result<()>;
    async fn on_model_delta(&self, state: &State, ctx: &mut RunContext<Ctx>, delta: &mut ModelDelta) -> Result<()>;
    async fn after_model(&self, state: &State, ctx: &mut RunContext<Ctx>, response: &mut ModelResponse) -> Result<()>;

    async fn before_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, call: &mut ToolCall) -> Result<()>;
    async fn on_tool_delta(&self, state: &State, ctx: &mut RunContext<Ctx>, delta: &mut ToolDelta) -> Result<()>;
    async fn after_tool(&self, state: &State, ctx: &mut RunContext<Ctx>, result: &mut ToolResult) -> Result<()>;

    async fn on_error(&self, state: &State, ctx: &mut RunContext<Ctx>, error: &TinyAgentsError) -> Result<()>;
}
```

Wrap hooks need separate traits because they receive a handler. These are
implemented in `crate::harness::middleware`:

```rust
#[async_trait]
pub trait ModelMiddleware<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    async fn wrap_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: ModelRequest,
        next: ModelHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareModelOutcome>;
}

#[async_trait]
pub trait ToolMiddleware<State, Ctx = ()>: Send + Sync {
    fn name(&self) -> &str;
    async fn wrap_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome>;
}
```

`next` is a borrowed handle to the rest of the onion (`ModelHandler` /
`ToolHandler`); calling `next.run(ctx, state, request_or_call)` proceeds to the
inner layer and ultimately the real model/tool call. Because `run` borrows
`&self`, a wrap middleware can call it **zero** times (short-circuit /
replace), **once** (proceed), or **many** times (retry / fallback). The
innermost layer is supplied by the agent loop via the `ModelBaseCall` /
`ToolBaseCall` traits, and the stack composes the onion through
`MiddlewareStack::run_wrapped_model` / `run_wrapped_tool` (registration order =
outermost first). `MiddlewareModelOutcome::Response(ModelResponse)` and
`MiddlewareToolOutcome::Result(ToolResult)` carry the resolved value; both are
`#[non_exhaustive]`. The agent loop runs each lifecycle `before_*` hook, then
the wrap onion, then each lifecycle `after_*` hook.

## Ordering

Before hooks run in registration order. After hooks run in reverse registration
order. Wrap hooks compose so the first registered middleware is the outermost
layer. This mirrors common web middleware stacks and keeps cleanup symmetrical.

Streaming hooks run in registration order for each delta before the delta is
forwarded to subscribers or accumulated into the final response. Middleware that
needs symmetrical setup and teardown for a stream should use `wrap_model`; delta
hooks are for per-chunk inspection or transformation.

Prompt/cache-layout middleware should run after static prompt rendering and
before model dispatch. It must declare whether it changed stable prefix segments
or only volatile tail segments so provider prompt-cache behavior is observable.

## Control Outcomes

Middleware should be able to return:

- continue with modified request
- replace model/tool response
- replace or suppress a streaming delta
- emit state update
- retry current call
- fallback to another model or tool
- accept, reject, transform, or defer a steering command
- jump to `model`
- jump to `tools`
- jump to `end`
- interrupt for human input
- persist checkpoint
- resume from checkpoint
- fail with classified error

Graph-specific commands should be translated at the graph boundary. The harness
should expose harness-native control outcomes so it remains usable without a
graph.

## Graph Boundary

Middleware must not need to know whether the caller is using the simple loop or
the state-graph runtime. The runtime adapter maps harness-native outcomes onto
graph commands:

- continue -> `Command::Continue`
- jump to model/tools/end -> `Command::Goto(...)` or `Command::End`
- human interrupt -> `Command::Interrupt`
- accepted steering -> `Command::Update`, `Command::Goto`,
  `Command::Interrupt`, or queued child-run delivery depending on target
- branch/fan-out middleware -> `Command::Fork`
- retry/fallback -> handled inside the node or wrap hook before command return

When middleware mutates graph-visible state, it must emit an explicit state
update event so checkpoint replay can explain the change.

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
- prompt cache layout guard middleware
- context editing middleware
- context compression middleware
- transcript compression middleware
- retrieval compression middleware
- streaming delta compression middleware
- output compression middleware
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
- interaction with provider prompt/KV-cache layout
- interaction with retries and fallbacks

## Tool policy enforcement

`ToolPolicyMiddleware` (`src/harness/middleware/library/`) enforces the
per-tool [`ToolPolicy`](tool.md#tool-policy-enforcement) metadata at two hooks:
`before_model` (exposure — a blocked tool is hidden from the model) and
`before_tool` (execution — a blocked call is rejected with
`TinyAgentsError::Validation`). Both hooks share one decision so a hidden tool
can never be executed by a divergent path.

Build it from a registry snapshot (`ToolRegistry::policies()`), then compose
enforcement builders:

- `ToolPolicyMiddleware::strict(policies)` — fail-closed baseline: unclassified
  or unknown tools are rejected, and `destructive`/`payment` side effects denied.
- `.require_classification(bool)` / `.require_background_safe(bool)`
- `.deny_side_effects(mask)` — deny any tool declaring a side effect in `mask`.
- `.require_sandbox(true)` — block a tool whose `runtime.sandbox ==
  SandboxMode::Required` **unless** the run carries a workspace whose `sandbox`
  is `Required` (see [workspace isolation](workspace.md)); fail closed otherwise.
- `.require_approval([names])` — block any tool declaring
  `access.approval_required` unless its name is in the approved set.
- `.enforce_result_bytes(true)` — in `after_tool`, truncate a result exceeding
  the tool's `runtime.max_result_bytes` and flag it (`result.error` mentions
  `max_result_bytes`).

```rust
use std::sync::Arc;
use tinyagents::harness::middleware::{MiddlewareStack, ToolPolicyMiddleware};
use tinyagents::harness::tool::{SandboxMode, ToolPolicy, ToolRuntime};
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::workspace::WorkspaceDescriptor;

let mut policies = std::collections::HashMap::new();
policies.insert(
    "shell".to_string(),
    ToolPolicy::classified().with_runtime(ToolRuntime {
        sandbox: SandboxMode::Required,
        ..ToolRuntime::default()
    }),
);
use tinyagents::harness::tool::ToolCall;
let call = || ToolCall::new("c1", "shell", serde_json::json!({}));

let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
stack.push(Arc::new(ToolPolicyMiddleware::new(policies).require_sandbox(true)));

// No workspace -> the sandboxed tool is blocked (fail closed).
let mut bare: RunContext = RunContext::new(RunConfig::new("no-sandbox"), ());
assert!(stack.run_before_tool(&mut bare, &(), &mut call()).await.is_err());

// A sandboxed workspace satisfies the requirement.
let mut ok: RunContext = RunContext::new(RunConfig::new("sandboxed"), ())
    .with_workspace(WorkspaceDescriptor::new("/work").with_sandbox(SandboxMode::Required));
stack.run_before_tool(&mut ok, &(), &mut call()).await?; // admitted
```

Emitted events: rejections surface as `TinyAgentsError::Validation` (not events).
`enforce_result_bytes` mutates the `ToolResult` in place.

## Tool exposure

`ContextualToolSelectionMiddleware` filters the model-visible tool set on each
`before_model`, using a predicate that sees both the `ToolSchema` and a live
`ToolSelectionContext { run_id, depth, tags, requested_model }` — so exposure can
vary by recursion depth, run tags (security tier / background marker), or the
target model. When it withholds any tools it emits
`AgentEvent::ToolsFiltered { by, excluded, remaining }`, making the exposure
decision auditable.

Two constructors:

- `from_lists(allow, deny)` — deny always hides; when `allow` is `Some`, a tool
  must be listed to be exposed (fail-closed for unknown tools).
- `inheriting(parent_allow, parent_deny, child_allow, child_deny)` — composes a
  child policy against an inherited parent policy so a sub-agent can only
  **narrow**, never widen: **deny is additive** (`parent ∪ child`) and **allow
  is intersective** (the effective allowlist is the intersection when both
  restrict; the single restriction when only one does; unrestricted otherwise).

```rust
use tinyagents::harness::middleware::ContextualToolSelectionMiddleware;

// Parent allows {a,b,c} and denies {c}; child tries to allow {b,c,d}.
// Effective allow = {a,b,c} ∩ {b,c,d} = {b,c}; deny adds parent's c -> {c}.
// So only `b` survives (`d` was never parent-allowed, `c` is parent-denied).
let mw = ContextualToolSelectionMiddleware::inheriting(
    Some(["a", "b", "c"]), ["c"],
    Some(["b", "c", "d"]), Vec::<String>::new(),
);
// After run_before_model, request.tools == [schema("b")].
```

Exposure only changes what the model *sees*; pair it with
[tool policy enforcement](#tool-policy-enforcement) or `ToolAllowlistMiddleware`
so a model that calls a hidden tool is still stopped at execution.

## Middleware control

Any middleware (or step) can steer the loop out-of-band via
`RunContext::request_control(MiddlewareControl)`
([context feature](context.md#middleware-control-outcomes)):

- `MiddlewareControl::StopWithFinal(text)` — stop now, using `text` as the final
  assistant response.
- `MiddlewareControl::Interrupt { node, message }` — pause at the next safe
  checkpoint, surfacing `TinyAgentsError::Interrupted` so a caller can checkpoint
  and resume.

Requests are resolved by **precedence, not last-writer**: `request_control`
keeps the highest-`precedence()` pending request within a turn (`Interrupt` (2)
outranks `StopWithFinal` (1)), so a stronger pause is never silently downgraded
to a stop by a later weaker request. The agent loop drains the request at its
safe checkpoint (after each model response) via `RunContext::take_control` and,
when it honors one, emits `AgentEvent::ControlApplied { control, detail }` where
`control` is the outcome's `kind()` label.

## State And Request Mutation

Middleware should prefer immutable request replacement for large changes and
small explicit mutation for local fields. It must not mutate shared registries or
global config during a run. Runtime state updates should be explicit and
observable.

## Compression Middleware

Compression is not one hook. A useful compression implementation may need to run
at several boundaries:

- `before_agent`: load previous compression state and policy.
- `before_model`: compress old messages, retrieved context, examples, or tool
  artifacts before the request is sent.
- `wrap_model`: measure full call timing, retry behavior, cache layout, and
  provider usage while preserving setup/teardown symmetry.
- `on_model_delta`: compact, redact, sample, or classify streaming output before
  it is persisted or forwarded.
- `after_model`: commit response summaries, update transcript compression state,
  and attach provenance to the final response.
- `before_tool` and `after_tool`: compress large tool arguments/results and
  decide what enters model-visible context.
- `after_agent`: persist durable summaries, compression indexes, and audit
  events.

Compression middleware must preserve enough provenance for debugging and replay:
source message ids, source artifact ids, original token estimates, compressed
token estimates, prompt segment ids, cache prefix fingerprints, policy version,
and whether the stable provider prompt-cache prefix was preserved.
