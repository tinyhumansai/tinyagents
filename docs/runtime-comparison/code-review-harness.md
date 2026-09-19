# `tinyagents-harness` — deep code-quality and design review

Reviewed at v2.1.2 (`fc33c43`). All paths below are relative to `crates/tinyagents-harness/src/` unless they start with `docs/`, `vendor/` or `crates/`.

---

## 1. Architecture as-built

**Entry.** `runtime/types.rs` — `AgentHarness<State, Ctx>` = `ModelRegistry` + `ToolRegistry<State,Ctx>` + `MiddlewareStack` + `RunPolicy`
(+ optional `ResponseCache`, `ToolTimeoutSettings`). Two front doors:

* SDK path — `agent_loop/entry.rs`: `invoke*` / `invoke_streaming*` / `invoke_stream*` / `*_collecting_partial` (10 variants) all funnel
  into `drive_collecting` → `run_loop` (`run_loop.rs:15`). `TerminalRunGuard` (`entry.rs:15`) owns the `AgentRun` so a dropped
  future still fires the host terminal observer.
* Hosted path — `runtime/agent.rs`: `AgentInvocation { host: Arc<HostCapabilities<State>>, request, context }` → `prepare_agent_turn`
  (definition lookup, input screening, system-prompt composition, memory/experience recall) → installs a type-erased
  `HostInvocationAuthority<State,Ctx>` on `RunContext::host_authority` → same `run_loop`. The loop re-reads that authority through
  `host_invocation_binding()` (`agent.rs:679`) for model routing, tool allowlists, budget gates, authorization and output screening.

**One turn** (`run_loop.rs:206-837`): cancel check → steering drain (`steering/mod.rs`) → pending `MiddlewareControl` → deadline +
model-call cap (`limits/`) → `PromptBuilder` builds `ModelRequest` (system segment + name-sorted tool segment + tail, fingerprinted for
provider prompt caching) → `before_model` lifecycle hooks → model resolution (host resolver or local registry) → structured-output plan
(`structured/`) → host budget permit → `ModelStarted` → **model-wrap onion** (`middleware/mod.rs:315`) whose base is
`ModelCallBase::call` (`model_call.rs:1093`): re-fingerprint, re-bind if a wrap layer changed `request.model`, then
`invoke_model_with_retry` (response cache → `invoke_model_resolving` = per-model retry loop + fallback chain; unary or streaming via
`invoke_model_streaming_once`, which fans each chunk through `on_model_delta` and `AgentEvent::ModelDelta`) → text-dialect tool-call
recovery → usage fold → `after_model` → `ModelCompleted` → assistant appended → tool calls split into structured hits vs real →
`execute_tools` (`tools.rs:174`): serial admission (cancel/deadline/cap, `before_tool`, provider-invalid args, host allowlist,
unknown-tool policy, injected-arg stripping, normalisation, schema validation, host authorization) → execution (serial through the
tool-wrap onion, or `join_all` when eligible) → ordered fold (`after_tool`, host screening/classification, `ToolCompleted`, transcript
append) → loop. Termination: no tool calls (after truncated-empty retry and `continue_turn` nudge) → structured extraction →
`LoopExit::Finished`; or `LimitStop`, `Paused`, or `Err`.

**Sub-agents** (`subagent/mod.rs`): `SubAgentTool` is a `ToolDispatch` whose `execute` builds `parent.child(...)` and runs the child
harness inside the parent's tool call, sharing `EventSink`, `CancellationToken`, `SteeringHandle`, stores and workspace.

**Providers.** Only two adapters live in this crate (`providers/claude_code`, `providers/claude_agent_sdk` — subprocess/JSON-RPC
drivers). OpenAI/Anthropic/local adapters live in `vendor/tinyinference/crates/tinyinference-llm`; the harness never sees wire JSON.

**Divergence from docs (verified):**

| Doc claim | Reality |
|---|---|
| `agent_loop/README.md:23-35`, `runtime.md:103`, module doc `tools.rs:15-19`: multi-call turns run concurrently "when no tool-wrap middleware is registered" | `tools.rs:1015-1022` also requires `lifecycle_middleware == 0` **and** every tool to opt in via `Tool::is_concurrency_safe` (default `false`, `vendor/tinytools/.../tool/types.rs:163`). Any real deployment (observe/budget/logging middleware) is serial. |
| `runtime.md:119` lists `max_concurrency` as a hard limit | No such field on `RunLimits` (`limits/types.rs:23`); the concurrent path is an unbounded `join_all` (`tools.rs:971`). |
| `docs/modules/harness/tool.md:234` "`UnknownToolPolicy::Fail` (default, historical)"; `agent_loop/README.md:113,132` | `runtime/types.rs:118` `#[default] ReturnToolError`; `InvalidArgsPolicy` likewise defaults to `ReturnToolError` (`:147`). |
| `agent_loop/stream.rs:23-25` "streaming a child agent's own deltas … tracked as follow-up" | Done: `subagent/mod.rs:246` passes `parent.streaming`; `runtime/test.rs` has `hosted_streaming_child_keeps_model_deltas_in_the_parent_stream`. |
| `runtime.md:112` "Run `on_tool_delta` middleware for tool progress streams" | `MiddlewareStack::run_on_tool_delta` (`middleware/mod.rs:245`) has **no caller**; `AgentEvent::ToolProgress` is never emitted. |
| `docs/modules/harness/structured-output.md:84-95` `StructuredOutputErrorPolicy { RetryWithDefaultMessage, … }` with retry events | Nothing of the kind exists. The loop does `extractor.extract(&response)?` (`run_loop.rs:791`) — one shot, run fails. The only in-loop retry is truncated-empty recovery. |
| `docs/audit.md:37-49` "malformed OpenAI tool-call JSON fails closed", citing `src/providers/openai/mod.rs` | That path no longer exists in this crate (moved to vendor), and the behaviour is now the opposite by design: provider-marked `ToolCall::invalid` is *recovered* as a tool-error result (`tools.rs:276-287`). The "resolved" entry describes a state that has since been reversed. |
| `docs/sdk-gaps.md` §2 "Recoverable unknown tool calls — Status: missing" | Implemented (`tools.rs:305-371`, `UnknownToolPolicy::{Fail,ReturnToolError,Rewrite}`); the gaps doc is stale. §3 reasoning deltas: `MessageDelta.reasoning` exists; tool-call start/complete channels do not. |
| `structured/repair.rs:11-14` links `tinyinference_llm::providers::openai::relaxed_json` | No such module; `cargo doc` reports it as a broken link. Two lenient JSON repair ladders now exist (`relaxed_json.rs`, vendor `convert.rs:497`) and neither is applied to `call.invalid` in admission (see I-13). |

`cargo clippy -p tinyagents-harness --all-targets -- -W clippy::pedantic`: 1 042 warnings (386 `must_use`, 103 backtick docs,
79 missing `# Errors`, 38 missing `# Panics`, 19 lossy `u64→f64`, 7 `f64→u64` sign-loss casts, 6 functions > 100 lines — `run_loop_body`
is 499). `cargo doc --no-deps`: 77 warnings, all broken intra-doc links (`CachePolicy::protect_prompt_prefix` ×3, `EventId` ×4,
`ModelRequest` ×3, `SqliteResponseCache`, `crate::graph::command::Command`, private `types` modules linked from public docs).

---

## 2. Findings

### Critical

**C-1. Unchecked pointer cast of the host authority can read the wrong type (UB).**
`runtime/agent.rs:679-702`:
```rust
pub(crate) fn host_invocation_binding<State: Send + Sync, Ctx: Send + Sync>(context: &RunContext<Ctx>) -> ... {
    let Some(authority) = context.host_authority.as_ref() else { return Ok(None) };
    let authority = unsafe {
        &*(std::sync::Arc::as_ptr(authority) as *const HostInvocationAuthority<State, Ctx>)
    };
    Ok(Some(authority.binding.clone()))
```
`host_authority` is `Option<Arc<dyn Any + Send + Sync>>` (`context/types.rs:268`) and the SAFETY comment claims only same-`State`,
same-`Ctx` contexts can carry it. That is false: `RunContext::child<ChildCtx>` is `pub` (`context/mod.rs:320-337`) and copies
`host_authority` into a `RunContext<ChildCtx>` for **any** `ChildCtx`; and `State` is not tracked by `RunContext` at all, so
`AgentHarness<OtherState, Ctx>::invoke_in_context(hosted_ctx, …)` compiles. Either way `host_invocation_binding` reinterprets
`HostInvocationAuthority<S1,C1>` as `HostInvocationAuthority<S2,C2>` — which contains `Arc<HostCapabilities<S1>>` and
`Option<Arc<InvocationRuntime<S1,C1>>>` — and `.clone()`s through the wrong vtables. Repro: hosted parent, then
`parent.child(cfg, OtherCtx)` + a second harness with a different `Ctx`. `invoke_in_parent` (`subagent/mod.rs:231`) guards only its own
entry; the primitive is unguarded.
*Fix (S/M):* make the erased slot checkable. Simplest sound option: store `Arc<dyn Any>` only when `State: 'static, Ctx: 'static`
(already true at install time) and have `host_invocation_binding` require the same bounds and use `downcast_ref`, returning
`Err(Validation("host authority type mismatch"))` on `None`. The "borrowed `State` support" the comment defends is not exercised by the
hosted path (which already requires `'static`), so make `host_invocation_binding` two functions: a `'static` one used by hosted code,
and a trivially-`None` one for the generic loop when no authority is installed. Also split `child` into `child(&self, cfg, data: Ctx)`
(propagates authority) and `child_with_data<ChildCtx>` (does **not**).

**C-2. Streaming path discards Anthropic thinking signatures → next turn cannot be replayed.**
`model_call.rs:936-964` rebuilds the terminal message from the deltas that crossed middleware:
```rust
if !streamed_reasoning.is_empty() {
    content.push(ContentBlock::Thinking { text: std::mem::take(&mut streamed_reasoning), signature: None });
}
```
and the comment says signatures are "intentionally discarded". The vendor Anthropic renderer drops any unsigned thinking block
(`vendor/tinyinference/crates/tinyinference-llm/src/providers/anthropic/request.rs:235-258`). Anthropic requires the signed thinking
block to precede a `tool_use` block when thinking is enabled, so **streaming + extended thinking + any tool call fails on the second
model call** (the assistant row has `tool_calls` but no thinking block). Unary runs keep the signature and work. Same defect in the
cache-replay path `model_call.rs:427-448`. No test mentions `signature` (`grep -c signature agent_loop/test.rs` = 0).
*Fix (S):* only synthesise an unsigned block when a delta middleware actually changed the reasoning text. Compare
`streamed_reasoning` against the concatenated `Thinking` text of the terminal response; if equal, keep the terminal blocks verbatim
(signature intact). Same for `RedactedThinking` ordering. Add a streaming test with a signed `Thinking` block that asserts the
signature survives into `run.messages`.

**C-3. Concurrent tool path breaks its own started/terminal invariant on first failure.**
`tools.rs:976-1005`: after `join_all`, the fold returns at the first `Err`:
```rust
Err(err) => { self.fail_tool_call(ctx, status, &prepared.call_id, …); return Err(err); }
```
Every later `Execute` slot already had `ToolStarted` emitted (`tools.rs:929`) and `status.active_tool_calls` populated, but never
gets `ToolFailed`/`ToolCompleted`, contradicting the module doc (`tools.rs:28-35`, "every `ToolStarted` is followed by exactly one
terminal partner"). Exporters that pair by `call_id` (Langfuse span pairing in `observability/langfuse`) leak open spans, and
`HarnessRunStatus.active_tool_calls` reports in-flight tools after the run failed.
*Fix (S):* on the first fatal error, drain the remaining `executed` pairs and call `fail_tool_call` for each (error =
"aborted: sibling tool call failed") before returning; `release_active_tool_call` is already positional.

### Important

**I-1. A per-call timeout neither retries nor falls back.** `model_call.rs:626-628`:
```rust
if matches!(error, TinyAgentsError::Timeout(_)) { return Err(error); }
```
The comment says this is for the *run* deadline, but `with_call_budget` labels per-call ceilings (`PER_CALL_BOUND_LABEL`) with the
same variant, and `is_retryable` (`retry/mod.rs:346-366`) returns `false` for `Timeout`. So `RunLimits::max_model_call_ms` — whose
whole purpose is "this one call wedged, run time still left" — aborts the run instead of trying the next attempt/fallback.
*Fix (S):* add `TinyAgentsError::CallTimeout { .. }` (or a `bound` field) so the retry classifier and fallback gate can distinguish
the two; keep run-deadline timeouts terminal.

**I-2. Text-dialect tool-call recovery is unconditional and runs on final answers.** `run_loop.rs:565` →
`recover_text_dialect_calls` (`:1077-1131`) parses `<tool_call>` XML out of *any* assistant text whenever tools were offered and the
provider returned no native calls. A model that quotes the XML format in its answer (explaining tools, echoing a user's example,
writing docs) gets that text executed as a real tool call, with the visible text silently stripped. There is no policy switch, no
event, and the module doc for the loop does not mention it.
*Fix (S):* gate on `RunPolicy::text_dialect_recovery: bool` (default `false` for models whose profile reports native tool calling),
emit `AgentEvent::ControlApplied { control: "text_dialect_recovered" }`, and skip fenced code blocks.

**I-3. `TinyAgentsError` and `AgentEvent` are exhaustive public enums with placeholder variants.**
`events/types.rs:317,550-585` — `StateUpdate`, `MemoryLoaded`, `MemorySaved`, `ToolProgress`, `StreamClosed`, `MiddlewareFailed`
are documented "defined for future emit"; `grep` shows only `MiddlewareFailed` is emitted, from one middleware
(`middleware/library/context.rs:159`), never by the stack. Neither enum is `#[non_exhaustive]` (`error.rs:19`, `events/types.rs:38`),
so every downstream `match` breaks when a real variant lands, and the "future" variants are already part of the JSON contract.
`TinyAgentsError` also carries graph/language variants (`MissingStart`, `RecursionLimit`, `Checkpoint`, `Compile`, `Parse`…) inside the
harness crate — the one-error-type design the CLAUDE.md "no facade crate" rule was supposed to retire.
*Fix (M):* `#[non_exhaustive]` on both; delete never-emitted variants or emit them (`MiddlewareFailed` from `run_stack_hook`
is a 3-line change); long-term split `TinyAgentsError` into per-crate errors with `From` impls.

**I-4. Blocking I/O inside `async fn` on the tokio worker.** `cache/sqlite.rs:138-190` runs rusqlite queries directly inside
`async fn get/put_with_ttl/clear`; `store/mod.rs:154-200` `FileStore::get/put/delete` call `std::fs::*` directly, while the same
file's `append` (`store/mod.rs:483-520`) correctly uses `spawn_blocking`. A response-cache hit on the model hot path therefore stalls
a worker; under `SqliteResponseCache` with a contended DB every concurrent run serialises on `Mutex<Connection>` while the runtime
thread is parked.
*Fix (S):* wrap the bodies in the existing `spawn_blocking`-with-fallback helper from `store/mod.rs:512-520`.

**I-5. Steering queue is shared across the whole run tree.** `context/mod.rs:331` `with_optional_steering(self.steering.clone())`
gives every child the parent's `SteeringHandle` (`Arc`). `apply_pending_steering` (`steering/mod.rs`) drains the queue at *whichever*
run reaches a checkpoint first, so an `Inject`/`Pause` meant for the orchestrator can be consumed by a sub-agent mid-tool-call and
injected into the child's transcript. Also, one policy-rejected command in a batch returns `Err(Steering)` **after** draining, so the
allowed commands in the same batch are lost and the run dies.
*Fix (M):* key commands by target run id (default: root) or give children a derived handle that only sees commands addressed to them;
reject disallowed commands individually (emit `Steered { accepted: false }`, keep the rest).

**I-6. Hosted return values erase every typed error.** `runtime/agent.rs:360-367` and `:390-393`:
```rust
Some(_) => Err(TinyAgentsError::Model("hosted agent invocation failed".to_string())),
```
`LimitExceeded`, `EmptyResponse`, `Validation`, `Interrupted`, `Steering` all become a generic `Model` error, and the partial `run` is
dropped. `sanitize_hosted_event` (`:206-236`) does the same to `RunFailed.error` on the public stream. A host cannot tell "budget
exhausted" from "provider 500" without a private event listener — the very thing `docs/sdk-gaps.md` §7 asks for.
*Fix (S):* return a `HostedError { kind: HostedErrorKind, run: Box<AgentRun> }` where `kind` is a closed, non-leaking enum
(`Cancelled | Timeout | LimitExceeded | Policy | Provider | Internal`); sanitise the message, not the classification.

**I-7. Nested retry layers multiply attempts and emit uncorrelatable events.** `RetryMiddleware::wrap_model`
(`middleware/library/resilience.rs:35-58`) retries `next`, and `next` bottoms out in `invoke_model_resolving` which has its own
retry loop **and** fallback chain. With both configured the worst case is `mw.max_attempts × policy.retry.max_attempts × |fallback|`
provider calls. The middleware also emits `RetryScheduled { call_id: "{run}-model" }` (`:49-52`) while the loop uses
`"{run}-model-{n}"` (`run_loop.rs:517`), so journals show retries for a call id that never started.
*Fix (S):* thread the real `CallId` into wrap middleware via `RunContext` (status already has `active_model_call`), and document
that `RetryMiddleware` is an alternative to `RunPolicy::retry`, or have the base call skip its own retry when a `RetryMiddleware` is
registered.

**I-8. `join_all` concurrency is unbounded and the eligibility rule makes it practically unreachable.** See the doc table:
`tools.rs:1021` requires zero lifecycle middleware. Either the docs are wrong or the feature is. The reason given ("lifecycle
middleware can rewrite calls during admission") is already handled — admission is serial and completes before any future is built
(`tools.rs:893-905`). *Fix (S):* drop `lifecycle_middleware == 0` from the predicate (the `&mut RunContext` argument only applies
to tool-wrap middleware) and add `RunLimits::max_tool_concurrency` with `futures::stream::iter(..).buffered(n)`.

**I-9. Host allow-list is fail-open when empty.** `run_loop.rs:162`, `tools.rs:296,320,353`:
```rust
.is_none_or(|allowed| allowed.is_empty() || allowed.contains(&schema.name))
```
`allowed_tools` is `definition.tools.into_iter().collect()` (`runtime/agent.rs:615`); an agent definition that declares no tools
gets **every** registered tool. `docs/sdk-gaps.md` §9 lists "fail-closed when policy metadata is missing" as the goal.
*Fix (S):* make `allowed_tools: Option<HashSet<String>>` (`None` = definition did not declare; `Some(empty)` = no tools), and treat
`None` as fail-closed under a `HostCapabilities` flag.

**I-10. Per-turn O(transcript) cloning and repeated fingerprinting.** Each model call: `messages[..system_end].to_vec()` and
`messages[system_end..].to_vec()` (`run_loop.rs:312,317`), `tool_schemas.clone()` (`:315`), `PromptBuilder::build` clones the
system messages and tools again (`prompt/mod.rs:364-378`) and SHA-256s them, then `refresh_prompt_cache_fingerprint`
(`run_loop.rs:950-1011`) clones system + tools a third time and hashes again, then unary `model.invoke(state, request.clone())`
(`model_call.rs:536`) clones the whole request per attempt. `response.tool_calls().to_vec()` + `.iter().cloned().partition` +
`tool_calls.clone()` triple-copy the calls (`:642-662`). `EventSink::emit` clones every record into `pending` even with zero
listeners (`events/mod.rs:170-171`), and `ModelCompleted` carries a full `serde_json::to_value(&request.messages)` when capture is on.
A 200-message transcript with 40 tools pays this every turn.
*Fix (M):* build the request once per turn and hand `&ModelRequest` to the wrap onion; cache the tools fingerprint for the run
(tool set is fixed — the code already says so at `:150-152`); make `EventSink::emit` return `EventId` and skip the enqueue when no
listeners are registered.

**I-11. `host_invocation_binding` clones a `HashSet<String>` + 6 fields on every call**, and it is called up to 6× per tool call
(`run_loop.rs:153,453`; `tools.rs:292,468,633,677`; `agent.rs:711` per progress token). *Fix (S):* return `Option<&HostInvocationBinding>`
(or `Arc<HostInvocationBinding>`), which also shrinks the unsafe surface in C-1.

**I-12. `unsafe` lifetime transmute for the hosted stream.** `runtime/agent.rs:668-673` transmutes the boxed stream's lifetime and
relies on field drop order plus a `#[expect(dead_code)]` field to keep the overlay alive; `poll_next` uses `get_unchecked_mut`
(`:160`) although every field is `Unpin` except `PhantomData<(&'a State, Ctx)>`. *Fix (S):* make the stream own its inputs
(`Arc<InvocationRuntime>` moved into `stream::unfold` state; `invoke_stream_in_context` already takes `RunContext` by value), and use
`PhantomData<fn() -> Ctx>` so `Pin::get_mut` is safe. Both `unsafe` blocks disappear.

**I-13. `relaxed_json` is not applied where its own docs say it is needed.** `relaxed_json.rs:1-30` motivates the module with
provider-invalid `function.arguments` looping forever; but admission short-circuits on `call.invalid` (`tools.rs:276-287`) without
trying `recover_relaxed_object`. The module is only reached from `structured/repair.rs:117` and the retired text-dialect prompt
parser (`tool/prompt.rs:603`). *Fix (S):* before returning the tool-error, try `recover_relaxed_object(call.arguments.as_str())`
and, on success, clear `invalid` and proceed to normal validation (emit `InvalidToolArgs { recovery: "repaired" }`).

### Minor

* **M-1** `StopWithFinal` after the model turn (`run_loop.rs:638`) leaves the assistant row's `tool_calls` unanswered in
  `run.messages`; resuming from that transcript is a provider 400. Append synthetic tool results or pop the row.
* **M-2** Child run ids use a process-global counter (`subagent/mod.rs:158` via `ids::next_seq`), contradicting
  `agent_loop/README.md:133` ("ids derived deterministically from the RunConfig") and making replayed journals of nested runs diverge
  across processes.
* **M-3** `map_tool_dispatch_error` (`tools.rs:1109-1116`) collapses every error to `Tool("tool dispatch failed")`, which
  `is_retryable` treats as unconditionally retryable (`retry/mod.rs:363`) — a `RetryMiddleware` around tools will re-run permanently
  failing tools; and `SubAgentDepth`/`LimitExceeded` from a child that escaped `invoke_in_parent_context`'s mapping are also flattened.
* **M-4** `ToolDispatch::tool()` returns a fresh `Arc` (`subagent/mod.rs:704-710` allocates a new declaration with cloned schema
  `Value` each call) and is invoked 3-4× per admitted call plus once per tool per run for `schemas()`. Return `&dyn Tool` or cache.
* **M-5** `ToolRegistry::register` silently overwrites duplicates (`tool/mod.rs:120-133`); `sdk-gaps` §15 asks for duplicate
  diagnostics. Return `Result` or a `Replaced(name)` marker.
* **M-6** `TerminalRunGuard::complete` and `Drop` clone the whole `AgentRun` (`entry.rs:30,39`) for an observer that reads
  `text()`, `usage` and `executed_tools` (`agent.rs:771-806`). Pass `&AgentRun` or a summary.
* **M-7** `apply_pending_steering` errors kill the run on the first disallowed command **after** the batch has been drained (see I-5).
* **M-8** `LimitTracker::started_at` is set in `RunContext::new`, not at `run_loop` start; a context built ahead of time burns
  wall-clock before the run begins (`context/mod.rs:291-310`, `limits/mod.rs:212`).
* **M-9** Hand-rolled civil-date conversion in `observability/langfuse/mod.rs:651-684` while `chrono` is a non-optional dependency;
  8 `LazyLock<Regex>` in `handoff.rs:209-221` for HTML stripping used by no loop code; `apply_handoff` hardcodes host tool name
  `extract_from_result` and `starts_with("Error")` (`handoff.rs:123`). This is OpenHuman policy inside the SDK.
* **M-10** `RunQueue`, `handoff`, `memory::ConversationMemory` are exported but not wired into any loop path; either document them
  as host utilities in `lib.rs` or move them behind a feature.
* **M-11** Dependency weight: `bytes` is unused (0 references); `chrono` is only used by `tools/time.rs` (feature `tools`) but is
  unconditional; `uuid`, `tempfile`, `wait-timeout`, `dirs` exist solely for `providers/claude_code`; `reqwest` (with `http2`,
  `rustls`) is pulled for Langfuse + multimodal. A `claude-code` feature and `langfuse` feature would cut the default graph
  substantially. `tempfile` is listed under both `[dependencies]` and `[dev-dependencies]`.
* **M-12** `run_on_model_delta` emits no `MiddlewareStarted/Completed` while `run_on_tool_delta` does (`middleware/mod.rs:194-256`);
  every other hook emits 2 events per middleware per call, so a 5-middleware stack produces 20+ bookkeeping events per turn that
  `ModelCompleted`-based exporters must filter.
* **M-13** Lossy numeric casts flagged by pedantic: `retry/mod.rs:280` (`attempt as i32` → `powi`), `:289,:435` (`f64 as u64`),
  `cache/sqlite.rs:64,177,215-216` (`u128↔i64` millis), `langfuse/mod.rs:684`. Use `try_from`/`saturating` helpers.
* **M-14** `claude_code/mod.rs:181-186` acquires the global semaphore *before* the per-thread mutex, so N callers on one busy thread
  hold N global permits while waiting — head-of-line blocking for other threads.

---

## 3. Structural refactors worth doing

1. **R-1. Turn-scoped request/response objects instead of `&mut` everything.** `run_loop_body` is 499 lines with 8 mutable locals
   threaded through 20 checkpoints. Introduce `struct Turn<'r> { request: ModelRequest, plan: Option<StructuredPlan>, recovery:
   TruncatedEmptyState, call_id, started_at }` built by `plan_turn()`, consumed by `call_model()`, `settle_response()`,
   `execute_tools()`. Rationale: makes I-10 fixable (build once, borrow), makes the exit paths testable without a harness, and
   gives the six copies of the `tokio::select! { biased; _ = cancel => …, r = timeout(remaining, fut) => … }` block
   (`run_loop.rs:488-507,926-939`; `model_call.rs:48-57`; `tools.rs:477-491,640-653`; `agent.rs:491-505`) one home:
   `RunContext::bounded(&self, what, fut) -> Result<T>`. Migration risk: low — internal only.
2. **R-2. Replace type-erased host authority with a trait object.** `host_authority: Option<Arc<dyn HostAuthority<State>>>` where the
   trait exposes `agent_id()`, `allowed_tools()`, `host() -> &HostCapabilities<State>`, `runtime()`; `Ctx` is only needed for
   `InvocationRuntime<State,Ctx>` — store that as `Arc<dyn Any>` and downcast at the single site that needs it. Removes C-1's
   `unsafe` and I-11's clones. Risk: medium (touches `RunContext` layout, `subagent`, `runtime/agent.rs`); public surface unchanged.
3. **R-3. Unify the four retry/fallback engines.** Loop retry (`invoke_model_resolving`), `RetryMiddleware`, `ModelFallbackMiddleware`,
   and `RunPolicy::fallback` all implement attempt loops with slightly different classification (I-1, I-7). Make the loop's engine
   the only one and turn the middlewares into thin policy overrides (`RunContext::override_retry_policy`). Risk: medium — public
   middleware types stay but their semantics become "configure", not "execute".
4. **R-4. Split `TinyAgentsError`** into `HarnessError` (this crate) with graph/language variants moved to their crates, re-exported via
   `From`. Add `#[non_exhaustive]` to it and `AgentEvent` now (cheap, prevents the next break). Risk: medium for downstream matches.
5. **R-5. Feature-gate heavy optional surfaces**: `claude-code` (subprocess driver + `uuid`, `tempfile`, `wait-timeout`, `dirs`),
   `langfuse` (`reqwest`), keep `tools` owning `chrono`. Risk: low; integration tests already gate `sqlite`/`tools`.
6. **R-6. Move `handoff`, `run_queue`, `no_progress`, `artifacts`, `workspace/git` into a `tinyagents-host-utils` crate** or under a
   `host-utils` feature. They are not on any loop path and carry product-specific heuristics (M-9).

---

## 4. Test-quality assessment

**Strengths.** The big files test behaviour, not implementation: `agent_loop/test.rs` (213 tests) is organised by contract —
limits, cache, retry/backoff schedule, fallback eligibility, truncated-empty recovery, `continue_turn`, argument normalisation
envelopes, streaming middleware transforms, parallel ordering (`parallel_tool_results_keep_original_call_order_and_ids`,
`unknown_tool_recovery_keeps_its_slot_in_a_parallel_turn`), cancellation mid-call, per-call vs run-budget timeouts. `runtime/test.rs`
covers hosted screening/redaction per content block, terminal-observer-on-drop, stream sanitisation, delegate authorisation.
Integration tests (`crates/tinyagents-integration-tests/tests/e2e_*`) exercise public surfaces only. Unit tests live in `test.rs`
files as the repo guideline asks; `#[cfg(test)]` helpers are small and named.

**Gaps (each maps to a finding):**
* No test asserts a `Thinking { signature: Some }` block survives a streaming tool-calling turn (C-2).
* No test asserts `ToolFailed` is emitted for siblings after a fatal concurrent failure, nor that `active_tool_calls` is empty
  afterwards (C-3). `parallel_tool_timeouts_return_ordered_recoverable_errors` covers recoverable errors only.
* No test builds a child context with a different `Ctx` from a hosted parent (C-1); `direct_parent_subagent_entry_fails_closed_for_hosted_authority`
  covers only the `SubAgent` wrapper.
* `per_model_call_ceiling_times_out_a_slow_call_with_run_time_left` asserts the error but not that the fallback chain was
  consulted (I-1).
* No test for text-dialect recovery on a *final* answer that merely quotes `<tool_call>` markup (I-2); the one unit test
  (`run_loop.rs:1169`) covers the no-tools case only.
* No test that an empty definition allow-list denies all tools (I-9); `hosted_definition_tool_allowlist_filters_schemas_and_rejects_fabricated_calls`
  uses a non-empty list.
* No test for steering commands being consumed by a child instead of the parent (I-5); `e2e_steering.rs` is single-level.
* No blocking-in-async detection (I-4); a `tokio::test(flavor = "current_thread")` with `time::pause` and a slow store would show it.
* Middleware-ordering tests assert counts (`HookCounts`) more than order; only one test checks reverse `after_*` order.
* `RetryMiddleware` + `RunPolicy::retry` stacking has no test bounding total attempts (I-7).

---

## 5. Genuinely good — do not touch

* **Exit discipline.** `LoopExit` (`agent_loop/types.rs:39`) separating finish / limit-stop / pause, plus `PartialRunOutcome` so a failed
  run keeps its transcript, is the right shape; `TerminalRunGuard` makes cancellation accounting honest.
* **Cache correctness.** SHA-256 per-message folding (`cache/key.rs:69`), scoping the key by resolved model identity + streaming flag +
  namespace, refusing to write fallback answers under the primary key, and replaying hits as synthetic deltas so warm and cold streaming
  runs are observationally identical (`model_call.rs:337`).
* **EventSink dispatch** (`events/mod.rs:163-197`): offset assignment under lock, single drainer, listeners notified outside the lock,
  re-entrant emits safe, ids stable across restarts. The audit's "resolved" entry is really resolved.
* **Admission-before-announce** in the concurrent path (`tools.rs:887-905`) and positional `release_active_tool_call` are careful.
* **Truncated-empty recovery** and `continue_turn` handling are well-bounded (count against `max_model_calls`, 4× clamp, state reset
  on every resolved turn) and thoroughly tested.
* **Injected-argument ordering rule** (strip → validate against model-facing schema → authorise on raw provider args → execute on
  prepared args, `tools.rs:374-505`) is a sound trust boundary; keep host authorization last.
* **Wrap-onion rebinding** (`ModelCallBase::rebind`, `model_call.rs:1027`) fixes a real class of "fallback re-invokes the same
  model" bugs and is documented with the failure it prevents.
* **`FileStore::append`'s strict/torn-write UTF-8 handling** (`store/mod.rs:380-430`) is the kind of reasoning the rest of the I/O
  layer should adopt (see I-4).
