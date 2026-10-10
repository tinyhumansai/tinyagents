//! Tool execution for one assistant turn.
//!
//! Split out of `agent_loop/run_loop.rs`; see the module doc comment in
//! `agent_loop/mod.rs` for the full loop lifecycle.
//!
//! # Serial vs. concurrent execution
//!
//! A turn's tool calls are driven in three phases:
//!
//! 1. **Admission** (always serial, in call order, under `&mut RunContext`):
//!    cancellation/deadline/limit checks, the lifecycle `before_tool` hooks,
//!    unknown-tool policy resolution, injected-argument stripping, and schema
//!    validation. Admission emits **no** [`AgentEvent::ToolStarted`] — see
//!    "Started/terminal pairing" below.
//! 2. **Execution**: when the turn requests **two or more** tools and **no
//!    tool-wrap middleware** ([`crate::middleware::ToolMiddleware`])
//!    is registered, the admitted calls run **concurrently**, so turn latency
//!    is the slowest tool instead of the sum (bounded by
//!    [`RunLimits::max_tool_concurrency`][crate::limits::RunLimits::max_tool_concurrency]
//!    when set — see I-8; unbounded, i.e. every eligible call starts at once,
//!    when unset). Lifecycle middleware does **not** force the serial path:
//!    admission (phase 1) already ran every `before_tool` hook to completion,
//!    serially, before any concurrent future is built, so there is nothing
//!    left for a lifecycle middleware to mutate once execution starts.
//!    Otherwise execution is serial, preserving the historical semantics.
//!    [`AgentEvent::ToolStarted`] is emitted here, once every admission has
//!    succeeded, so a call that is announced always runs.
//! 3. **Fold** (always serial, in original call order): the lifecycle
//!    `after_tool` hooks, accounting, the [`AgentEvent::ToolCompleted`]
//!    emission, and the transcript append. Results are attached to their
//!    original `tool_call_id` in the calls' original order regardless of
//!    completion order.
//!
//! ## Started/terminal pairing (the invariant this module maintains)
//!
//! Every [`AgentEvent::ToolStarted`] is followed by exactly one terminal
//! partner — [`AgentEvent::ToolCompleted`] when the call produced a result
//! (successful *or* error-carrying) or [`AgentEvent::ToolFailed`] when the run
//! itself is aborting because of it. `status.active_tool_calls` is cleared on
//! both paths, and by *position*, so two calls sharing one id (a real provider
//! defect) cannot clear each other's entry.
//!
//! Recovery paths — unknown tool, schema-invalid arguments, provider-unparseable
//! arguments — are not exceptions: they are folded through the same
//! [`AgentHarness::finish_tool_call`] pipeline, so they emit the same
//! started/completed pair, run `after_tool`, and account identically. The only
//! difference is that no tool ran.
//!
//! ## Canonical tool failures
//!
//! TinyTools makes the contract explicit: `Err` aborts the run; a tool that
//! can recover must return `Ok(ToolResult::error(..))`. Dispatch failures are
//! sanitized before leaving this boundary; cancellation and timeout retain
//! their typed classifications.
//!
//! ## Tool-wrap middleware runs inside each concurrent call
//!
//! [`crate::middleware::ToolMiddleware::wrap_tool`] receives a shared
//! `&RunContext` (it may read it, emit events, request control, or use
//! interior-mutable handles). The concurrent path therefore runs the whole
//! wrap onion — wrap layers around the real tool — inside each call's own
//! future, so wrapped calls overlap exactly like unwrapped ones. A wrap that
//! cannot tolerate overlap opts out with
//! [`crate::middleware::ToolMiddleware::concurrent_safe`]` -> false`, and any
//! such wrap keeps every batch on the serial path.
//! Lifecycle `before_tool`/`after_tool` hooks never forced serial execution:
//! they run in the admission/fold phases and still bracket each call.
//!
//! ## Nested calls
//!
//! A tool may call another tool through
//! [`ToolExecutionContext::call_tool`][crate::tool::ToolExecutionContext::call_tool];
//! those calls are admitted and executed by `super::nested`, share this
//! module's `max_tool_calls` budget, and never touch the transcript. See
//! `docs/modules/harness/nested-tool-calls.md`.
//!
//! ## Semantics preserved (and one deliberate difference)
//!
//! - **Event ordering**: every call's `ToolStarted` precedes its
//!   `ToolCompleted`, and `ToolCompleted` events are emitted in original call
//!   order in both modes. In concurrent mode all `ToolStarted` events precede
//!   the first `ToolCompleted` (they are emitted at admission).
//! - **Limits/deadline**: each call is admitted fail-closed against the
//!   tool-call cap and wall-clock deadline before it starts, and each
//!   execution is individually bounded by the run's remaining wall-clock
//!   budget, exactly as in serial mode.
//! - **Cancellation**: observed between admissions (before each call starts),
//!   matching the serial path, which also never interrupts a mid-flight tool.
//! - **Errors**: an `Err` fails the turn at the first call in original order.
//!   Difference: in serial mode later calls never start after a failure; in
//!   concurrent mode they were already in flight and run to completion, but
//!   their results are discarded — each already-started sibling still gets
//!   exactly one terminal event, [`AgentEvent::ToolFailed`] with
//!   `"aborted: sibling tool call failed"`, so the started/terminal invariant
//!   above holds even on this path. Tools that must not observe a sibling's
//!   failure should be run under a `concurrent_safe() == false` tool-wrap
//!   middleware (serial) or a harness without parallel-capable turns.
//!
use super::model_call::ToolCallBase;
use super::*;
use crate::tool::{
    CallGate, DeferredToolRequests, LedgerFailure, ToolDispatch, ToolEffectSettle, ToolEffectStart,
    ToolEffectStatus, ToolGate, ToolProgressGate, ToolProgressLimits, provider_schema,
};
use sha2::{Digest, Sha256};
use tinyinference_llm::message::ContentBlock;
use tinytools::{ToolCall as CanonicalToolCall, ToolCallId, ToolCallOptions};

/// How a single requested tool call was resolved during admission.
enum ResolvedToolCall<State: Send + Sync, Ctx: Send + Sync> {
    /// A registered tool (possibly after an unknown-tool rewrite).
    Tool {
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
    },
    /// No tool runs; this result is appended to the transcript at the call's
    /// original position. A tool-error result for the recovery paths (unknown
    /// tool, invalid arguments); a success result for an intrinsic answer
    /// (`tool_search`).
    Answered(tinytools::ToolResult),
    /// No tool runs *yet*: the call needs a human decision or host-side
    /// execution first (A2). The loop finishes the batch's other calls and
    /// then hands every deferred request to the caller (or the inline
    /// `DeferredToolHandler`).
    Deferred(DeferredRequest),
}

/// Which [`DeferredToolRequests`] list a deferred call belongs to.
#[derive(Clone, Copy, Debug)]
pub(super) enum DeferredKind {
    /// Needs an [`crate::tool::ToolApprovalDecision`]; the harness runs the tool
    /// on approval.
    Approval,
    /// Needs a [`crate::tool::DeferredCallResult`]; the host runs the tool.
    External,
}

/// One call the loop is handing back instead of answering.
#[derive(Clone, Debug)]
pub(super) struct DeferredRequest {
    /// The call as the model made it (original arguments, so an approver
    /// sees — and may edit — exactly what the model asked for).
    pub(super) call: ToolCall,
    pub(super) kind: DeferredKind,
    /// Stable label for the `ToolDeferred` event.
    pub(super) reason: &'static str,
    /// Host-only payload from `ApprovalRequired`/`CallDeferred`, if any.
    pub(super) metadata: Option<Value>,
}

impl DeferredRequest {
    fn approval(call: ToolCall, reason: &'static str, metadata: Option<Value>) -> Self {
        Self {
            call,
            kind: DeferredKind::Approval,
            reason,
            metadata,
        }
    }

    fn external(call: ToolCall, reason: &'static str, metadata: Option<Value>) -> Self {
        Self {
            call,
            kind: DeferredKind::External,
            reason,
            metadata,
        }
    }
}

/// One requested call after admission, in original order.
///
/// The concurrent path materialises the whole admitted batch before emitting a
/// single [`AgentEvent::ToolStarted`], so an admission failure part-way through
/// the batch cannot leave earlier calls announced-but-never-run (TOOL-3).
enum AdmittedCall<State: Send + Sync, Ctx: Send + Sync> {
    /// A registered tool to invoke, with its (validated) call.
    Execute {
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
        call: ToolCall,
    },
    /// A recovery or an intrinsic answer: no tool runs, but the call is still
    /// answered through the normal result pipeline so it emits the same
    /// started/completed pair and runs the same `after_tool` hooks (TOOL-11).
    Recovered {
        call: ToolCall,
        result: tinytools::ToolResult,
    },
    /// Deferred at admission: nothing runs, nothing is announced; folded into
    /// the batch's [`DeferredToolRequests`] in original order.
    Deferred(DeferredRequest),
}

/// One transcript slot per requested call, in original order, used by the
/// concurrent path to reassemble results deterministically.
///
/// `Recovered` carries a full `ToolResult` (now noticeably larger than
/// `Execute`'s no-op payload since the vendor `tinytools::ToolControl`/
/// `follow_up`/`metadata` fields landed); boxing it would touch every
/// construction and pattern-match site in this file for a one-shot,
/// short-lived per-call value, so the size difference is accepted here
/// rather than threaded through as indirection.
#[allow(clippy::large_enum_variant)]
enum ToolSlot {
    /// An executed call: consumes the next prepared/result pair in order.
    Execute,
    /// A recovery, folded in place through the normal result pipeline.
    Recovered {
        call: ToolCall,
        result: tinytools::ToolResult,
    },
    /// A call deferred at admission (see [`AdmittedCall::Deferred`]).
    Deferred(DeferredRequest),
}

/// Admission metadata for one executable call, paired 1:1 (in order) with its
/// execution future/result on the concurrent path.
pub(super) struct PreparedToolCall {
    pub(super) call_id: CallId,
    pub(super) tool_name: String,
    /// The admitted call, kept so an execution-time deferral
    /// (`ApprovalRequired`/`CallDeferred` raised by the tool) can hand the
    /// original request back through [`DeferredToolRequests`].
    pub(super) call: ToolCall,
    pub(super) options: ToolCallOptions,
    pub(super) captured_input: Option<Value>,
    pub(super) started_at_ms: u64,
    pub(super) executed: bool,
    pub(super) output_origin: crate::host::ContentOrigin,
}

/// Derives a best-effort deduplication key for one tool call from its name
/// and arguments (B5).
///
/// `tinytools::ToolPolicy` does not currently declare an explicit
/// idempotency key field, so this hashes `(tool name, arguments)` with
/// SHA-256: two calls to the same tool with identical arguments derive the
/// same key, which is exactly what a host wants to notice when deciding
/// whether an orphaned effect might have already landed. This is content
/// equality, not a cryptographic guarantee — a tool whose "same effect" notion
/// differs from "identical arguments" (e.g. one that reads a clock) should
/// not rely on this key alone.
fn tool_call_idempotency_key(tool_name: &str, arguments: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tool_name.as_bytes());
    hasher.update([0u8]);
    hasher.update(serde_json::to_vec(arguments).unwrap_or_default());
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Resolves the effective host tool allow-list for `ctx`, or `Ok(None)`
    /// when nothing should be restricted.
    ///
    /// Three cases:
    /// - Not a hosted run at all (`host_invocation_binding` returns `None`):
    ///   no host allow-list concept applies, so this returns `None` (allow
    ///   every registered tool, same as an explicit-model run).
    /// - Hosted, and the resolved [`crate::host::AgentDefinition`] declared a
    ///   non-empty tool list: returns that set. Only those names are
    ///   dispatchable, checked with plain set membership — no empty-set
    ///   bypass (that bypass was I-9: an empty `HashSet` used to mean
    ///   "unrestricted" instead of "nothing").
    /// - Hosted, but the definition declared no tools at all (an empty or
    ///   absent list): fails closed by default — returns `Some(HashSet::new())`,
    ///   which allows nothing — unless
    ///   [`crate::host::HostCapabilities::fail_closed_tool_allowlist`] was
    ///   explicitly turned off on this host, in which case it returns `None`
    ///   (legacy unrestricted behavior, opt-in only).
    pub(super) fn resolve_tool_allowlist(
        &self,
        ctx: &RunContext<Ctx>,
    ) -> Result<Option<std::collections::HashSet<String>>> {
        let Some(binding) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? else {
            return Ok(None);
        };
        Ok(match &binding.allowed_tools {
            Some(declared) => Some(declared.clone()),
            None if binding.host.fail_closed_tool_allowlist => {
                Some(std::collections::HashSet::new())
            }
            None => None,
        })
    }

    /// Resolves the run's [`ToolGate`]: the exact allowlist from
    /// [`Self::resolve_tool_allowlist`], the harness policy's
    /// [`tool_rules`](crate::runtime::RunPolicy::tool_rules) and, on a hosted
    /// run, the resolved definition's own rules. Every listing and admission
    /// site asks this one gate, so the catalogue, `tool_search` and dispatch
    /// cannot disagree about a tool.
    pub(super) fn resolve_tool_gate(&self, ctx: &RunContext<Ctx>) -> Result<ToolGate> {
        let allowed = self.resolve_tool_allowlist(ctx)?;
        let binding = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)?;
        let definition_rules = binding.as_ref().and_then(|b| b.tool_rules.as_ref());
        Ok(ToolGate::new(
            allowed,
            &self.policy.tool_rules,
            definition_rules,
        ))
    }

    /// The tool rules' answer for a model call of `name` with `args`.
    ///
    /// An unregistered or allowlist-excluded name is admitted here with no
    /// directive: it is not a tool the rules can describe, and the
    /// unknown-tool policy answers it below.
    fn rule_admission(&self, gate: &ToolGate, name: &str, args: &serde_json::Value) -> CallGate {
        match gate
            .allows_name(name)
            .then(|| self.tools.model_dispatch(name))
            .flatten()
        {
            Some(dispatch) => gate.admit_call(dispatch.tool().as_ref(), args),
            None => CallGate::Admit(tinytools::ApprovalDirective::Default),
        }
    }

    /// Builds the run's deferred-tool catalogue: every
    /// [`tinytools::ToolExposure::Deferred`] registration the gate lets the
    /// model search for, or an empty catalogue when discovery is disabled.
    pub(super) fn deferred_catalog(
        &self,
        gate: &ToolGate,
    ) -> crate::tool::discover::DeferredCatalog {
        if !self.policy.discovery.enabled {
            return crate::tool::discover::DeferredCatalog::default();
        }
        let mut entries = self
            .tools
            .deferred_schemas_with_families()
            .into_iter()
            .filter(|(schema, _)| {
                gate.lists(
                    &schema.name,
                    self.tools.get(&schema.name).as_deref(),
                    tinytools::Surface::Search,
                )
            })
            .collect::<Vec<_>>();
        if let Some(preparation) = &self.policy.tool_schemas {
            let families: Vec<Option<String>> =
                entries.iter().map(|(_, family)| family.clone()).collect();
            let schemas: Vec<_> = entries.into_iter().map(|(schema, _)| schema).collect();
            entries = crate::tool::prepare_tool_schemas(&schemas, preparation)
                .into_iter()
                .zip(families)
                .collect();
        }
        crate::tool::discover::DeferredCatalog::build_with_families(entries)
    }

    /// Resolves the discovery bridge for one call, when it is one.
    ///
    /// Returns `Some` with the answer for a `tool_search` call (no tool runs)
    /// and `None` untouched for any other name. A deferred tool is not
    /// unwrapped from anything: the model calls it by its own name and
    /// admission handles it like any registered tool.
    async fn answer_discovery_bridge(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call: &mut ToolCall,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<Option<ResolvedToolCall<State, Ctx>>> {
        use crate::tool::discover::TOOL_SEARCH_NAME;
        if !self.policy.discovery.enabled || call.name != TOOL_SEARCH_NAME {
            return Ok(None);
        }
        // Reuses `resolve_tool_allowlist` (I-9's fail-closed allow-list
        // resolution) rather than reading `binding.allowed_tools` directly,
        // so the discovery bridge honors the same
        // `fail_closed_tool_allowlist` policy as the direct tool set built in
        // `run_loop_body` — an empty declared list never falls back to
        // "unrestricted" here either.
        let gate = self.resolve_tool_gate(ctx)?;
        let intrinsic = match gate.admits_intrinsic(TOOL_SEARCH_NAME, tinytools::Surface::Call) {
            // The bridge is answered in place and cannot be deferred to an
            // approver, so a rule requiring approval for discovery refuses it.
            CallGate::Admit(tinytools::ApprovalDirective::Required) => CallGate::Refuse(format!(
                "Tool '{TOOL_SEARCH_NAME}' requires approval by tool rules; discovery cannot be deferred."
            )),
            other => other,
        };
        if let CallGate::Refuse(message) = intrinsic {
            // Discovery itself is ruled out: refuse rather than answer, so the
            // bridge cannot reveal what the rules withhold from the model. Like
            // the other answered recoveries, the call keeps its budget slot.
            return Ok(Some(ResolvedToolCall::Answered(
                tinytools::ToolResult::error(message),
            )));
        }
        let catalog = self.deferred_catalog(&gate);
        if catalog.is_empty() {
            // Nothing was deferred, so the bridge was never advertised; let
            // the call fall through to the unknown-tool policy.
            return Ok(None);
        }
        {
            let answer = crate::tool::discover::answer_tool_search(
                &catalog,
                &self.policy.discovery,
                &call.arguments,
            )
            .await;
            promoted_names.extend(answer.matched_names.iter().cloned());
            let ranking = answer.ranking;
            // `query` is model-supplied tool-call content, same privacy
            // class as a normal tool call's arguments, so it honors the same
            // `RunPolicy::capture.tool_io` gate (default `false`, payload
            // free) instead of always recording potentially user/tenant
            // sensitive search text regardless of the run's capture policy.
            let record = ctx.emit(AgentEvent::ToolSearched {
                call_id: CallId::new(call.id.clone()),
                query: if self.policy.capture.tool_io {
                    call.arguments
                        .get("query")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    String::new()
                },
                matched: answer.matched,
                ranker: ranking
                    .as_ref()
                    .map(|r| r.ranker.to_string())
                    .unwrap_or_default(),
                top_confidence: ranking.as_ref().and_then(|r| r.top_confidence),
                fallback: ranking.as_ref().and_then(|r| r.fallback.clone()),
                shadow_matched: ranking.as_ref().and_then(|r| r.shadow_names.clone()),
                latency_ms: ranking.as_ref().map_or(0, |r| r.latency_ms),
            });
            status.set_last_event(record.id);
            Ok(Some(ResolvedToolCall::Answered(answer.result)))
        }
    }

    /// Executes one assistant turn's requested tool calls, appending each
    /// result to `messages` in the calls' original order.
    ///
    /// Dispatches to the concurrent path when it is safe (see the module docs
    /// for the exact conditions and preserved semantics); otherwise runs the
    /// historical serial path.
    ///
    /// Returns the calls the batch **deferred** (A2) — empty for the common
    /// case. A deferred call gets no tool-result row; every other call in the
    /// batch is still executed and answered, so the caller only has to decide
    /// what to do with the pending ones (exit, or resolve inline).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tools_with_promotions(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<DeferredToolRequests> {
        // Injection and argument normalization change the model payload before
        // execution. Until admission has produced those authoritative values,
        // a declaration cannot safely make a parallel decision from raw model
        // input, so such batches deliberately retain serial semantics.
        let canonical_parallel_safe =
            !matches!(
                self.policy.invalid_args,
                InvalidArgsPolicy::NormalizeThenReturnToolError
            ) && batch_is_canonical_parallel_safe(&self.tools, &tool_calls);
        // A fresh batch: votes left by an aborted earlier batch must not count,
        // and admission positions restart at the batch's first call.
        ctx.terminate_votes.clear();
        ctx.batch_admissions = 0;
        let deferred = if should_execute_tools_concurrently(
            tool_calls.len(),
            canonical_parallel_safe,
            self.middleware.tool_middleware_concurrent_safe(),
        ) {
            self.execute_tools_concurrently(
                state,
                ctx,
                run,
                status,
                messages,
                tool_calls,
                promoted_names,
            )
            .await?
        } else {
            self.execute_tools_serially(
                state,
                ctx,
                run,
                status,
                messages,
                tool_calls,
                promoted_names,
            )
            .await?
        };
        self.settle_batch_termination(ctx, run, deferred.is_empty());
        // The positions belong to this batch only; a later admission (a
        // deferred call resumed inline) must not match them.
        ctx.truncated_call_positions.clear();
        Ok(deferred)
    }

    /// Ends the run when **every** call of the just-finished batch asked to
    /// terminate (`ToolControl::terminate`; pi's `shouldTerminateToolBatch`).
    ///
    /// A batch where only some calls asked is not terminal: the others
    /// returned results the model has yet to read, so the hint is dropped
    /// (logged) and the loop goes on. A batch that deferred a call
    /// (`all_answered == false`) is never terminal either — that call has no
    /// answer yet. When the batch does end the run, the final response is the
    /// **last** call's output in source order (results fold in call order in
    /// both serial and concurrent mode, so this is deterministic).
    ///
    /// Always drains the batch's votes.
    pub(super) fn settle_batch_termination(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        all_answered: bool,
    ) {
        let votes = std::mem::take(&mut ctx.terminate_votes);
        let asked = votes.iter().filter(|vote| vote.is_some()).count();
        if asked == 0 {
            return;
        }
        if !all_answered || asked != votes.len() {
            tracing::debug!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                terminating_calls = asked,
                batch_calls = votes.len(),
                all_answered,
                "[agent_loop] tool terminate hint ignored: not every call in the batch asked to terminate"
            );
            return;
        }
        let Some(output) = votes.into_iter().next_back().flatten() else {
            return;
        };
        tracing::debug!(
            target: "tinyagents::agent_loop",
            run_id = %ctx.run_id(),
            batch_calls = asked,
            "[agent_loop] every call in the batch asked to terminate; ending the run"
        );
        run.final_response = Some(ModelResponse::assistant(output));
        ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::End));
    }

    /// Serial admission for one call: cancellation/deadline/limit checks
    /// (fail-closed), the lifecycle `before_tool` hooks, unknown-tool policy
    /// resolution, and schema validation.
    ///
    /// Shared by the serial and concurrent paths so admission semantics cannot
    /// drift between them.
    async fn admit_tool_call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call: &mut ToolCall,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<ResolvedToolCall<State, Ctx>> {
        // Safe cancellation checkpoint: stop before invoking the next
        // (side-effecting) tool if cancellation was requested.
        if ctx.cancellation.is_cancelled() {
            return Err(TinyAgentsError::Cancelled);
        }
        if ctx.check_deadline().is_err() {
            ctx.emit(AgentEvent::LimitReached {
                kind: LimitKind::WallClock,
            });
            return Err(TinyAgentsError::Timeout(format!(
                "run `{}` exceeded its wall-clock deadline",
                ctx.run_id()
            )));
        }
        // A call a length stop may have cut off is answered, not run, and spends
        // no budget slot (see `RunPolicy::reject_truncated_tool_calls`).
        let position = ctx.batch_admissions;
        ctx.batch_admissions += 1;
        if ctx.truncated_call_positions.contains(&position) {
            tracing::debug!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                tool = %call.name,
                call_id = %call.id,
                "[agent_loop] answering a possibly-truncated tool call with an error"
            );
            return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                truncated_tool_call_message(&call.name),
            )));
        }
        // The context's `LimitTracker` (synced with `RunPolicy::limits` at run
        // start) is the single enforced source of truth for the tool-call cap,
        // so the reported limit always matches the one that trips.
        if let Err(err) = ctx.record_tool_call() {
            ctx.emit(AgentEvent::LimitReached {
                kind: LimitKind::ToolCalls,
            });
            return Err(TinyAgentsError::LimitExceeded(err.to_string()));
        }

        // Discovery bridge, resolved before any hook runs. `tool_search` is
        // answered from the run's catalogue without running a tool. A deferred
        // tool is not bridged at all: the model calls it by its own name, so
        // every `before_tool` hook, allow-list, and the host authorization gate
        // below see the real tool name and arguments with nothing to unwrap. A
        // host-registered tool under the `tool_search` name wins, and a call the
        // provider could not parse is left for the recovery below.
        if call.invalid.is_none()
            && self.tools.dispatch(&call.name).is_none()
            && let Some(answered) = self
                .answer_discovery_bridge(ctx, status, call, promoted_names)
                .await?
        {
            return Ok(answered);
        }
        // Preserve the exact attacker-controlled provider payload for host
        // authorization/audit. `call.arguments` is later canonicalized for
        // execution and must not overwrite what the gate evaluates.
        let model_arguments = call.arguments.clone();

        // Tool rules, before any hook runs, so an approval middleware never
        // asks a human about a call the rules refuse. Evaluated against the
        // registered tool (an unregistered name falls through to the
        // unknown-tool policy below) and any target it dispatches to, on the
        // raw provider arguments the host gate also sees.
        let gate = self.resolve_tool_gate(ctx)?;
        let mut rule_approval = match self.rule_admission(&gate, &call.name, &model_arguments) {
            CallGate::Admit(approval) => approval,
            CallGate::Refuse(message) => {
                ctx.limits.rollback_tool_calls(1);
                return Ok(ResolvedToolCall::Answered(with_refusal_metadata(
                    ctx,
                    &call.id,
                    tinytools::ToolResult::error(message),
                )));
            }
        };

        // The slot is *reserved* above (cap-first, so a middleware hook never
        // runs for a call the budget has already refused) and *released* here
        // when `before_tool` refuses the call — an approval denial or an
        // allowlist rejection must not spend budget on a call that never ran
        // (TOOL-12). The recovery paths below deliberately keep their slot:
        // they answer the model and let it try again, so counting them is what
        // bounds the correction loop.
        if let Err(err) = self.middleware.run_before_tool(ctx, state, call).await {
            tracing::debug!(
                "[agent_loop::tools] `before_tool` refused `{}` (call `{}`); \
                 releasing its tool-call slot: {err}",
                call.name,
                call.id
            );
            ctx.limits.rollback_tool_calls(1);
            // A2/A3 signals from a `before_tool` hook are decisions about
            // *this call*, not failures of the run: a deferral hands the
            // call back to the host, and the retry/failed vocabulary answers
            // the model without running the tool (the `HumanApprovalMiddleware`
            // `Deny` outcome, for one).
            return match err {
                TinyAgentsError::ApprovalRequired { metadata } => Ok(ResolvedToolCall::Deferred(
                    DeferredRequest::approval(call.clone(), "approval_required", Some(metadata)),
                )),
                TinyAgentsError::CallDeferred { metadata } => Ok(ResolvedToolCall::Deferred(
                    DeferredRequest::external(call.clone(), "call_deferred", Some(metadata)),
                )),
                TinyAgentsError::ToolFailed(message) => Ok(ResolvedToolCall::Answered(
                    with_refusal_metadata(ctx, &call.id, tinytools::ToolResult::failed(message)),
                )),
                TinyAgentsError::ModelRetry(message) => Ok(ResolvedToolCall::Answered(
                    with_refusal_metadata(ctx, &call.id, tinytools::ToolResult::retry(message)),
                )),
                other => Err(other),
            };
        }

        // Before giving up on provider-unparseable arguments (below), try the
        // conservative, meaning-preserving repairs in the protocol crate's
        // `tinytools_agent::repair::json` (unquoted keys, redundant wrapping
        // braces, leaked chat-template quote tokens — see that module's doc
        // comment for the exact defects it targets).
        // This is the one place I-13 asked for it applied: admission was
        // short-circuiting straight to a tool error without ever trying the
        // repair the module exists for. On success the call proceeds through
        // normal (schema) validation below as if the provider had sent it
        // clean, rather than round-tripping a "fix your JSON" error the model
        // often cannot actually act on.
        if call.invalid.is_some()
            && let Some(raw) = call.arguments.as_str()
            && let Some(repaired) = tinytools_agent::repair::json::recover_object(raw)
        {
            let call_id = CallId::new(call.id.clone());
            let record = ctx.emit(AgentEvent::InvalidToolArgs {
                call_id,
                tool_name: call.name.clone(),
                arguments: call.arguments.clone(),
                error: call.invalid.clone().unwrap_or_default(),
                recovery: "repaired".to_string(),
            });
            status.set_last_event(record.id);
            call.arguments = repaired;
            call.invalid = None;
        }

        // The provider marked this call's arguments unparseable (a small local
        // model emitted malformed JSON). Rather than fail the run, inject a
        // tool-error result carrying the parse detail and the raw arguments so
        // the model can retry with corrected JSON — mirroring the schema-invalid
        // recovery below and how mature frameworks (LangChain `invalid_tool_calls`,
        // the AI SDK's invalid dynamic tool parts) surface the failure to the
        // model. This consumed one tool-call budget slot above, bounding the loop,
        // and always resolves the call so a stalled/never-resolving tool cannot
        // hang the loop. Applied unconditionally (not gated by `InvalidArgsPolicy`,
        // which governs *schema* validation of well-formed args): a call the
        // provider could not even parse is a transport-level defect.
        if let Some(detail) = call.invalid.clone() {
            let call_id = CallId::new(call.id.clone());
            let record = ctx.emit(AgentEvent::InvalidToolArgs {
                call_id,
                tool_name: call.name.clone(),
                arguments: call.arguments.clone(),
                error: detail.clone(),
                recovery: "tool_error".to_string(),
            });
            status.set_last_event(record.id);
            return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                detail,
            )));
        }

        // Hosted turns carry an explicit definition allowlist. Do not merely
        // hide disallowed schemas: a model can still fabricate a name, so the
        // dispatch boundary must reject it too.
        let is_allowed = gate.allows_name(&call.name);
        let (dispatch, tool) = match is_allowed
            .then(|| self.tools.model_dispatch(&call.name))
            .flatten()
        {
            Some(dispatch) => {
                let tool = dispatch.tool();
                // A deferred tool called by its own name: the model found it
                // through `tool_search`, the manifest, or a replayed
                // declaration. Reported so an audit consumer can tell a
                // deferred call from a direct one; admission is unchanged.
                if self.policy.discovery.enabled
                    && tool.exposure() == tinytools::ToolExposure::Deferred
                {
                    let record = ctx.emit(AgentEvent::DeferredToolCall {
                        call_id: CallId::new(call.id.clone()),
                        tool_name: call.name.clone(),
                    });
                    status.set_last_event(record.id);
                }
                (dispatch, tool)
            }
            None => {
                // The model called an unregistered tool. Apply the run's
                // `UnknownToolPolicy` instead of unconditionally aborting.
                let requested = call.name.clone();
                let arguments = call.arguments.clone();
                let call_id = CallId::new(call.id.clone());

                // Rewrite mode: retarget to a fixed compatibility tool if
                // that tool exists, otherwise fall through to recovery.
                let rewrite_target = match &self.policy.unknown_tool {
                    UnknownToolPolicy::Rewrite { tool_name } => self
                        .tools
                        .dispatch(tool_name)
                        .filter(|_| gate.allows_name(tool_name))
                        .and_then(|dispatch| {
                            match gate.admit_call(dispatch.tool().as_ref(), &arguments) {
                                CallGate::Admit(approval) => {
                                    Some((tool_name.clone(), dispatch, approval))
                                }
                                CallGate::Refuse(_) => None,
                            }
                        }),
                    _ => None,
                };

                if let Some((tool_name, dispatch, approval)) = rewrite_target {
                    // The rewrite target's own rules decide its approval, not
                    // the unknown name's.
                    rule_approval = approval;
                    call.name = tool_name.clone();
                    let record = ctx.emit(AgentEvent::UnknownToolCall {
                        call_id,
                        requested_name: requested,
                        arguments,
                        recovery: format!("rewrite:{tool_name}"),
                    });
                    status.set_last_event(record.id);
                    let tool = dispatch.tool();
                    (dispatch, tool)
                } else if matches!(self.policy.unknown_tool, UnknownToolPolicy::Fail) {
                    return Err(TinyAgentsError::ToolNotFound(requested));
                } else {
                    // `ReturnToolError` (or a Rewrite whose target is also
                    // missing): inject a tool-error result naming the
                    // requested tool, then continue so the model can correct
                    // itself. The result points to `tool_search` when the
                    // discovery bridge is advertised and names a few close
                    // matches, instead of listing every callable tool (see
                    // `unknown_tool`). The attempted arguments are echoed in the
                    // message and kept on the `UnknownToolCall` event. This consumed one tool-call
                    // budget slot above, bounding the loop.
                    let available = self
                        .tools
                        .model_callable_names()
                        .into_iter()
                        .filter(|name| {
                            self.tools
                                .get(name)
                                .is_some_and(|tool| gate.lists_tool(tool.as_ref()))
                        })
                        .collect::<Vec<_>>();
                    // A host-registered `tool_search` takes precedence over the
                    // intrinsic bridge (see `admit_tool_call`), so only advertise
                    // discovery when the bridge would receive the call.
                    let tool_search_available = self
                        .tools
                        .dispatch(crate::tool::discover::TOOL_SEARCH_NAME)
                        .is_none()
                        && matches!(
                            gate.admits_intrinsic(
                                crate::tool::discover::TOOL_SEARCH_NAME,
                                tinytools::Surface::Call
                            ),
                            CallGate::Admit(tinytools::ApprovalDirective::Default)
                                | CallGate::Admit(tinytools::ApprovalDirective::Waived)
                        )
                        && !self.deferred_catalog(&gate).is_empty();
                    let message = super::unknown_tool::unknown_tool_message(
                        &requested,
                        &arguments,
                        &available,
                        tool_search_available,
                    );
                    let record = ctx.emit(AgentEvent::UnknownToolCall {
                        call_id,
                        requested_name: requested.clone(),
                        arguments,
                        recovery: "tool_error".to_string(),
                    });
                    status.set_last_event(record.id);
                    return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                        message,
                    )));
                }
            }
        };
        // Step 1 of the injected-argument ordering rule (see
        // `crate::tool::injected`): strip every host-injected key from
        // the model-supplied arguments *before* validating them, so a model
        // that names a hidden key it was never shown cannot forge it. Step 2
        // then validates against the **model-facing** projection of the schema
        // — the same one `ToolRegistry::schemas` advertises — because a key the
        // model never saw must not be `required` of it.
        let canonical_call = CanonicalToolCall::new(
            ToolCallId::new(call.id.clone()),
            call.name.clone(),
            call.arguments.clone(),
        );
        // Canonical preparation is security sensitive: it removes protected
        // model values, injects authoritative host/call-id values, and only
        // then returns the value we validate and execute.
        let injected_values = match dispatch.injected_arguments(&canonical_call) {
            Ok(values) => values,
            Err(_) => {
                return Err(TinyAgentsError::Validation(format!(
                    "failed to prepare injected arguments for tool `{}`",
                    call.name
                )));
            }
        };
        let injected_declarations = tool.injected_arguments();
        // `prepare_tool_arguments` deliberately requires an object because it
        // strips and inserts named keys. A declaration without injected keys
        // has no host authority to protect, so preserve its native JSON shape
        // for normalization and schema validation (for example a string- or
        // array-valued schema) instead of rejecting it as an injection error.
        let prepared_arguments = if injected_declarations.is_empty() {
            Ok(canonical_call.arguments.clone())
        } else {
            tinytools::prepare_tool_arguments(
                &canonical_call,
                &injected_declarations,
                &injected_values,
            )
        };
        let prepared_arguments = match prepared_arguments {
            Ok(arguments) => arguments,
            Err(error) => {
                if matches!(self.policy.invalid_args, InvalidArgsPolicy::Fail) {
                    return Err(TinyAgentsError::Validation(error.to_string()));
                }
                return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                    format!(
                        "invalid injected arguments for tool `{}`: {error}",
                        call.name
                    ),
                )));
            }
        };
        call.arguments = prepared_arguments;
        let schema = provider_schema(tool.as_ref());
        let raw_arguments = call.arguments.clone();
        if matches!(
            self.policy.invalid_args,
            InvalidArgsPolicy::NormalizeThenReturnToolError
        ) {
            normalize_tool_arguments(call, &schema);
        }
        if let Err(err) = schema.validate_call(call) {
            // The model called a registered tool with schema-invalid arguments.
            // Apply the run's `InvalidArgsPolicy` instead of unconditionally
            // aborting the turn (mirrors the unknown-tool recovery above).
            if matches!(self.policy.invalid_args, InvalidArgsPolicy::Fail) {
                return Err(err.into());
            }
            // `ReturnToolError`: inject a tool-error result carrying the
            // validation detail and a compact, TypeScript-style signature of
            // the expected arguments (`{skill: "a" | "b", tool?: string}`),
            // then continue so the model can correct itself. The full JSON
            // Schema used to be echoed here; with descriptions and scaffolding
            // it was most of the corrective's bytes and re-sent on every bad
            // call, while the names, types, optionality and enum values the
            // model actually needs survive in the signature. This consumed one
            // tool-call budget slot above, bounding the loop.
            let call_id = CallId::new(call.id.clone());
            let detail = err.to_string();
            let message = format!(
                "invalid arguments for tool `{}`: {detail}; expected arguments: {}",
                call.name,
                crate::tool::signature::type_signature(&schema.parameters)
            );
            let record = ctx.emit(AgentEvent::InvalidToolArgs {
                call_id,
                tool_name: call.name.clone(),
                arguments: raw_arguments,
                error: detail,
                recovery: "tool_error".to_string(),
            });
            status.set_last_event(record.id);
            return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                message,
            )));
        }
        // Tool rules once more, on the arguments that will actually run. The
        // early check saw the raw provider payload; repair, normalization
        // (a JSON-encoded object decoded) or preparation can change what an
        // indirect target or an argument condition reads, so the final
        // decision is made here, before approval and dispatch.
        match self.rule_admission(&gate, &call.name, &call.arguments) {
            CallGate::Admit(approval) => rule_approval = rule_approval.strictest(approval),
            CallGate::Refuse(message) => {
                ctx.limits.rollback_tool_calls(1);
                return Ok(ResolvedToolCall::Answered(with_refusal_metadata(
                    ctx,
                    &call.id,
                    tinytools::ToolResult::error(message),
                )));
            }
        }
        // Deferral (A2), after validation so an approver only ever sees a
        // call the tool would actually accept, and before host authorization
        // so a host's own gate is not consulted for a call a human has not
        // yet approved. A call the resume path already approved
        // (`RunContext::is_call_approved`) goes straight through.
        if !ctx.is_call_approved(&call.id) {
            let original =
                ToolCall::new(call.id.clone(), call.name.clone(), model_arguments.clone());
            if crate::tool::is_external_tool(tool.as_ref()) {
                ctx.limits.rollback_tool_calls(1);
                return Ok(ResolvedToolCall::Deferred(DeferredRequest::external(
                    original, "external", None,
                )));
            }
            let policy = tool.policy();
            // A `require_approval` rule defers like a declared
            // `approval_required`; an `auto_approve` rule waives the
            // declaration (a stricter rule elsewhere already won).
            let needs_approval = match rule_approval {
                tinytools::ApprovalDirective::Required => true,
                tinytools::ApprovalDirective::Waived => false,
                tinytools::ApprovalDirective::Default => policy.access.approval_required,
            };
            if needs_approval {
                ctx.limits.rollback_tool_calls(1);
                let metadata = serde_json::to_value(&policy.display)
                    .ok()
                    .filter(|value| value.as_object().is_some_and(|map| !map.is_empty()));
                return Ok(ResolvedToolCall::Deferred(DeferredRequest::approval(
                    original,
                    "approval_required",
                    metadata,
                )));
            }
        }
        // Host authorization is deliberately last in admission: the gate sees
        // the raw provider arguments (including any forged hidden fields),
        // while execution receives the prepared trusted arguments. A hosted
        // run is identified from its explicit RunContext binding; the
        // lower-level SDK path has no implicit host policy.
        if let Some(binding) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? {
            let request = crate::host::ToolCallRequest::new(
                call.name.clone(),
                model_arguments,
                binding.agent_id.clone(),
            )
            .with_call_id(CallId::new(call.id.clone()));
            let authorization = binding.host.security.authorize_tool(&request);
            let decision = ctx
                .bounded(self.call_budget(ctx), authorization, || {
                    format!(
                        "tool authorization for run `{}` exceeded its remaining wall-clock budget",
                        ctx.run_id()
                    )
                })
                .await?;
            if !decision.is_allowed() {
                let reason = decision
                    .denial_reason()
                    .unwrap_or("tool call was not approved")
                    .to_string();
                // Admission reserved a slot before consulting policy, but a
                // denied call never enters execution. Release it so repeated
                // approval denials cannot exhaust the tool budget and block a
                // later authorized call in the same turn.
                ctx.limits.rollback_tool_calls(1);
                return Ok(ResolvedToolCall::Answered(tinytools::ToolResult::error(
                    reason,
                )));
            }
        }
        Ok(ResolvedToolCall::Tool { dispatch })
    }

    /// Marks one call as started: status bookkeeping, the `ToolStarted`
    /// emission, and the capture-policy input snapshot. Returns the admission
    /// metadata consumed by the fold phase.
    fn start_tool_call(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call: &ToolCall,
        options: ToolCallOptions,
        executed: bool,
        output_origin: crate::host::ContentOrigin,
    ) -> PreparedToolCall {
        let call_id = CallId::new(call.id.clone());
        let tool_name = call.name.clone();
        status.active_tool_calls.push(call_id.clone());
        // Captured here (where the call actually starts) so the completed
        // event carries a real start time for duration-aware exporters.
        let started_at_ms = crate::ids::now_ms();
        // Snapshot the arguments for observability before `call` is moved
        // into execution, gated by the capture policy. Shared between the
        // `ToolStarted` event (so a host sees the arguments as soon as the
        // call starts) and the fold-phase `ToolCompleted` event.
        let captured_input = self.policy.capture.tool_io.then(|| call.arguments.clone());
        let record = ctx.emit(AgentEvent::ToolStarted {
            parent_call_id: None,
            call_id: call_id.clone(),
            tool_name: tool_name.clone(),
            input: captured_input.clone(),
        });
        crate::runtime::emit_host_progress::<State, Ctx>(
            ctx,
            crate::host::ProgressEvent::ToolCall {
                run: ctx.run_id().clone(),
                call: call_id.clone(),
                tool: tool_name.clone(),
            },
        );
        status.set_last_event(record.id);
        PreparedToolCall {
            call_id,
            tool_name,
            call: call.clone(),
            options,
            captured_input,
            started_at_ms,
            executed,
            output_origin,
        }
    }

    /// Records a tool-effect-ledger `started` row for `prepared` (B5), if a
    /// ledger is attached to `ctx`. A no-op when [`RunContext::tool_effect_ledger`]
    /// is `None`.
    ///
    /// Must be called *before* the tool actually executes, so a crash between
    /// this write and the call settling is observable on resume. When the
    /// write itself fails, [`RunContext::tool_effect_ledger_failure`] decides
    /// whether that is fatal ([`LedgerFailure::Abort`], the default — the
    /// caller must fail the call and propagate the error) or merely logged
    /// ([`LedgerFailure::Continue`] — the call proceeds unrecorded).
    pub(super) async fn record_tool_effect_started(
        &self,
        ctx: &RunContext<Ctx>,
        arguments: &Value,
        prepared: &PreparedToolCall,
    ) -> Result<()> {
        let Some(ledger) = ctx.tool_effect_ledger.clone() else {
            return Ok(());
        };
        let idempotency_key = tool_call_idempotency_key(&prepared.tool_name, arguments);
        let start = ToolEffectStart {
            run_id: ctx.run_id().clone(),
            call_id: prepared.call_id.clone(),
            tool: prepared.tool_name.clone(),
            idempotency_key,
            effect_summary: None,
        };
        if let Err(err) = ledger.started(start).await {
            return match ctx.tool_effect_ledger_failure {
                LedgerFailure::Abort => Err(err),
                LedgerFailure::Continue => {
                    tracing::warn!(
                        "[agent_loop::tools] tool-effect ledger `started` write failed for \
                         call `{}` (tool `{}`): {err} — continuing per \
                         LedgerFailure::Continue",
                        prepared.call_id.as_str(),
                        prepared.tool_name
                    );
                    Ok(())
                }
            };
        }
        Ok(())
    }

    /// Records a tool-effect-ledger terminal row for `prepared` (B5), if a
    /// ledger is attached to `ctx`. A no-op when [`RunContext::tool_effect_ledger`]
    /// is `None`.
    ///
    /// Deliberately best-effort and never fatal: by the time this is called
    /// the tool has already executed (or its execution future has already
    /// failed), so aborting the run over a *settle* write failure would
    /// discard a real result rather than merely skip recording one. A failed
    /// settle write is logged; the row stays `started` and will surface again
    /// from [`crate::tool::ToolEffectLedger::unresolved`] on the next resume.
    pub(super) async fn record_tool_effect_settled(
        &self,
        ctx: &RunContext<Ctx>,
        prepared: &PreparedToolCall,
        status: ToolEffectStatus,
    ) {
        let Some(ledger) = ctx.tool_effect_ledger.clone() else {
            return;
        };
        let settle = ToolEffectSettle {
            run_id: ctx.run_id().clone(),
            call_id: prepared.call_id.clone(),
            status,
            effect_summary: None,
        };
        if let Err(err) = ledger.settled(settle).await {
            tracing::warn!(
                "[agent_loop::tools] tool-effect ledger `settled` write failed for call `{}` \
                 (tool `{}`): {err}",
                prepared.call_id.as_str(),
                prepared.tool_name
            );
        }
    }

    /// Terminal partner of [`AgentEvent::ToolStarted`] on the abort path:
    /// emits [`AgentEvent::ToolFailed`] and closes the call's `active_tool_calls`
    /// entry.
    ///
    /// Without this, every `?` between `ToolStarted` and `ToolCompleted` left a
    /// dangling start — an exporter pairing the two by `call_id` silently drops
    /// the failed span, and the run keeps reporting a call that is no longer in
    /// flight (TOOL-6). Call it on **every** error path after
    /// [`Self::start_tool_call`].
    fn fail_tool_call(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call_id: &CallId,
        tool_name: &str,
        started_at_ms: u64,
        error: &TinyAgentsError,
    ) {
        release_active_tool_call(status, call_id);
        let duration_ms = crate::ids::now_ms().saturating_sub(started_at_ms);
        tracing::debug!(
            "[agent_loop::tools] tool `{tool_name}` call `{}` failed after {duration_ms} ms: \
             {error}",
            call_id.as_str()
        );
        let record = ctx.emit(AgentEvent::ToolFailed {
            parent_call_id: None,
            call_id: call_id.clone(),
            tool_name: tool_name.to_string(),
            started_at_ms: Some(started_at_ms),
            duration_ms: Some(duration_ms),
            error: error.to_string(),
        });
        status.set_last_event(record.id);
    }

    /// Fold phase for one completed call: the lifecycle `after_tool` hooks,
    /// accounting, the `ToolCompleted` emission, and the transcript append.
    ///
    /// Returns the result's `follow_up` content as a user message (B2), or
    /// `None` when there is none. It is **not** appended here: a provider
    /// requires every tool row of a batch to sit directly after the assistant
    /// row that requested it, so the batch driver appends the follow-ups
    /// only after its last tool row (see [`append_follow_ups`]).
    #[allow(clippy::too_many_arguments)]
    async fn finish_tool_call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        prepared: PreparedToolCall,
        mut result: tinytools::ToolResult,
    ) -> Result<Option<Message>> {
        // Canonical ToolResult is intentionally correlation-free. The harness
        // owns `PreparedToolCall` and uses it below for transcript pairing,
        // events, and elapsed time; a tool cannot forge any of those fields.

        if let Err(err) = self
            .middleware
            .run_after_tool(
                ctx,
                state,
                &crate::middleware::ToolInvocationIdentity::new(
                    prepared.call_id.clone(),
                    prepared.tool_name.clone(),
                ),
                &mut result,
            )
            .await
        {
            self.fail_tool_call(
                ctx,
                status,
                &prepared.call_id,
                &prepared.tool_name,
                prepared.started_at_ms,
                &err,
            );
            return Err(err);
        }

        // Tool output is untrusted input on its way back into the next model
        // request. Screen it after host/result middleware shaping but before a
        // transcript message exists, so neither the original nor a blocked
        // value can reach the provider.
        if let Some(binding) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? {
            let rendered = result.output_for_llm(prepared.options.prefer_markdown);
            let screening = binding
                .host
                .security
                .screen_input(&rendered, prepared.output_origin);
            let screened = ctx
                .bounded(self.call_budget(ctx), screening, || {
                    format!(
                        "tool-output screening for run `{}` exceeded its remaining wall-clock budget",
                        ctx.run_id()
                    )
                })
                .await;
            match screened {
                Ok(crate::host::ScreenOutcome::Pass) => {}
                Ok(crate::host::ScreenOutcome::Redacted(text)) => {
                    result.content = vec![tinytools::ToolContent::Text { text }];
                    result.markdown_formatted = None;
                }
                Ok(crate::host::ScreenOutcome::Block { reason }) => {
                    result = tinytools::ToolResult::error(reason);
                }
                Err(error) => {
                    self.fail_tool_call(
                        ctx,
                        status,
                        &prepared.call_id,
                        &prepared.tool_name,
                        prepared.started_at_ms,
                        &error,
                    );
                    return Err(error);
                }
            }
        }

        if let Some(binding) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)?
            && let Some(classifier) = &binding.host.tool_outcomes
        {
            let outcome = classifier.classify(&prepared.tool_name, &result);
            match outcome {
                crate::host::OutcomeClass::Success => result.is_error = false,
                crate::host::OutcomeClass::PermanentFailure => result.is_error = true,
                crate::host::OutcomeClass::RetryableFailure => {
                    result.is_error = true;
                    let detail = result.output_for_llm(prepared.options.prefer_markdown);
                    result.content = vec![tinytools::ToolContent::Text {
                        text: format!("retryable tool failure: {detail}"),
                    }];
                    result.markdown_formatted = None;
                }
            }
            tracing::debug!(
                tool = %prepared.tool_name,
                agent = %binding.agent_id,
                ?outcome,
                "[host] classified tool outcome"
            );
        }

        // A tool's own `ToolControl` (`return_direct`/`terminate`/`goto`/
        // `state_update`) is the tool-vocabulary half of A1: it is *data* the
        // tool returned, not a middleware decision, so it is translated into
        // the same `MiddlewareControl` request a `Middleware` would make
        // rather than a separate mechanism.
        //
        // `return_direct` means "the model never gets another turn": this
        // call's own output becomes the run's final response, which — unlike
        // `MiddlewareControl::StopWithFinal` — `JumpTo(End)` alone cannot
        // express (it falls back to the *last assistant message*, which is
        // one turn too early here), so the final response is set directly.
        //
        // `terminate` is a *batch* decision (pi's `shouldTerminateToolBatch`):
        // a call's own hint only records a vote here, and the batch driver
        // ends the run in `settle_batch_termination` once every call of the
        // batch has answered and **all** of them voted to terminate. A call
        // that did not terminate may have returned something the model still
        // has to read.
        let mut terminate_vote: Option<String> = None;
        if let Some(control) = result.control.clone() {
            // `return_direct` is now a per-call override (`Option<bool>`):
            // `None` means "no opinion", so it falls back to the tool's own
            // static `Tool::return_direct` default rather than being treated
            // as `false`.
            let return_direct = control.return_direct.unwrap_or_else(|| {
                self.tools
                    .dispatch(&prepared.tool_name)
                    .map(|dispatch| dispatch.tool().return_direct())
                    .unwrap_or(false)
            });
            if return_direct {
                let output = result.output_for_llm(prepared.options.prefer_markdown);
                run.final_response = Some(ModelResponse::assistant(output));
                ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::End));
                // No terminate vote: the pending `JumpTo(End)` already ends
                // the run, and a vote would let a later terminating sibling
                // make the batch look unanimous and replace this output as
                // the final response in `settle_batch_termination`.
            } else if control.terminate {
                terminate_vote = Some(result.output_for_llm(prepared.options.prefer_markdown));
            } else if let Some(goto) = &control.goto {
                match goto.as_str() {
                    "model" => ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::Model)),
                    "tools" => ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::Tools)),
                    "end" => ctx.request_control(MiddlewareControl::JumpTo(LoopTarget::End)),
                    other => tracing::debug!(
                        target: "tinyagents::agent_loop",
                        tool = %prepared.tool_name,
                        goto = other,
                        "[agent_loop] tool requested an unrecognized `goto` target; ignoring"
                    ),
                }
            }
            if let Some(update) = control.state_update.clone() {
                ctx.push_tool_state_update(update);
            }
        }

        ctx.terminate_votes.push(terminate_vote);

        run.tool_calls += 1;
        if prepared.executed {
            run.executed_tools.push(prepared.tool_name.clone());
        }
        // Host-only metadata (B2): recorded on the run and on the event
        // below, never rendered into the transcript row.
        if let Some(metadata) = result.metadata.clone() {
            run.tool_metadata
                .push(crate::middleware::ToolResultMetadata {
                    call_id: prepared.call_id.clone(),
                    tool_name: prepared.tool_name.clone(),
                    metadata,
                });
        }
        status.tool_calls = run.tool_calls;
        release_active_tool_call(status, &prepared.call_id);
        let model_output = result.output_for_llm(prepared.options.prefer_markdown);
        let captured_output = self
            .policy
            .capture
            .tool_io
            .then(|| Value::String(model_output.clone()));
        // Outcome fields carried on the event itself (not a side-channel) so
        // journal-backed exporters render duration/size/success without the
        // live run's state. Duration is wall-clock (completion minus start);
        // `is_error` is a reported tool failure, distinct from execution Err.
        let duration_ms = crate::ids::now_ms().saturating_sub(prepared.started_at_ms);
        let output_bytes = model_output.len() as u64;
        let error = result.is_error.then_some(model_output.clone());
        // The transcript and event both answer the admitted call, never a
        // tool-owned id. Clone before the event consumes its fields so the two
        // records cannot drift.
        let transcript_call_id = prepared.call_id.to_string();
        let event_call_id = prepared.call_id.clone();
        let event_tool_name = prepared.tool_name.clone();
        let record = ctx.emit(AgentEvent::ToolCompleted {
            parent_call_id: None,
            call_id: event_call_id,
            tool_name: event_tool_name,
            started_at_ms: Some(prepared.started_at_ms),
            input: prepared.captured_input,
            output: captured_output,
            duration_ms: Some(duration_ms),
            output_bytes: Some(output_bytes),
            error,
            metadata: result.metadata.clone(),
        });
        crate::runtime::emit_host_progress::<State, Ctx>(
            ctx,
            crate::host::ProgressEvent::ToolCallFinished {
                run: ctx.run_id().clone(),
                call: prepared.call_id.clone(),
                success: !result.is_error,
                output: if self.policy.capture.tool_io {
                    model_output
                } else {
                    String::new()
                },
            },
        );
        status.set_last_event(record.id);

        let mut tool_message =
            tool_message_from_result(transcript_call_id, &result, prepared.options);
        if self.policy.tool_result_durations && prepared.executed {
            super::tool_timing::append_duration(&mut tool_message, duration_ms);
        }
        messages.push(Message::Tool(tool_message));
        Ok(follow_up_message(&result.follow_up))
    }

    /// Executes requested tools one at a time (the historical semantics; used
    /// for single-call turns, non-parallel-safe batches, and when a registered wrap is not
    /// `concurrent_safe`).
    #[allow(clippy::too_many_arguments)]
    async fn execute_tools_serially(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<DeferredToolRequests> {
        let mut deferred = DeferredToolRequests::default();
        let mut follow_ups = Vec::new();
        for call in tool_calls {
            follow_ups.extend(
                self.execute_tool_serially(
                    state,
                    ctx,
                    run,
                    status,
                    messages,
                    call,
                    &mut deferred,
                    promoted_names,
                )
                .await?,
            );
        }
        append_follow_ups(messages, follow_ups);
        Ok(deferred)
    }

    /// Admits, executes, and folds **one** call on the serial path, recording
    /// a deferral into `deferred` instead of answering it.
    ///
    /// Shared by [`Self::execute_tools_serially`] and the resume path
    /// (`apply_deferred_results`), which re-runs an approved call through
    /// exactly this pipeline so admission, the wrap onion, and the fold are
    /// never duplicated.
    ///
    /// Returns the call's follow-up user message, if any, for the batch
    /// driver to append after its last tool row (see [`append_follow_ups`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_serially(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        mut call: ToolCall,
        deferred: &mut DeferredToolRequests,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<Option<Message>> {
        let dispatch = match self
            .admit_tool_call(state, ctx, status, &mut call, promoted_names)
            .await?
        {
            ResolvedToolCall::Tool { dispatch } => dispatch,
            ResolvedToolCall::Answered(result) => {
                return self
                    .recover_tool_call(state, ctx, run, status, messages, &call, result)
                    .await;
            }
            ResolvedToolCall::Deferred(request) => {
                self.defer_tool_call(ctx, status, request, deferred);
                return Ok(None);
            }
        };

        let options = dispatch.call_options(&call.arguments);
        let prepared =
            self.start_tool_call(ctx, status, &call, options, true, dispatch.output_origin());
        if let Err(err) = self
            .record_tool_effect_started(ctx, &call.arguments, &prepared)
            .await
        {
            self.fail_tool_call(
                ctx,
                status,
                &prepared.call_id,
                &prepared.tool_name,
                prepared.started_at_ms,
                &err,
            );
            return Err(err);
        }

        // The real tool call is the innermost base of the tool-wrap
        // onion (same before -> wrap -> after ordering as the model
        // path): lifecycle `before_tool` ran in admission, the wrap onion
        // runs here, and lifecycle `after_tool` runs in the fold. The
        // crate-owned tool policy returns a recoverable tool error; the
        // outer run budget still aborts when the whole run is exhausted.
        let run_budget = self.call_budget(ctx);
        let base = ToolCallBase {
            harness: self,
            dispatch,
            options,
            timeout_settings: self.tool_timeouts.clone(),
            level: 0,
            nested_state: Default::default(),
            gate_held: super::nested::GateHold::None,
        };
        let run_id = ctx.run_id().as_str().to_string();
        let gate = self.open_progress_gate(ctx, &prepared);
        let fut = self.middleware.run_wrapped_tool(ctx, state, call, &base);
        // TinyTools distinguishes a fatal execution `Err` from a
        // recoverable `ToolResult::error`; no harness error-policy facade
        // rewrites that canonical distinction.
        let guarded = futures::FutureExt::map(fut, |result| {
            result.map(|wrapped| wrapped.into_result_with_control())
        });
        let outcome = Self::with_call_budget(
            run_budget,
            &run_id,
            "tool call",
            super::model_call::RUN_BOUND_LABEL,
            gate.scope(guarded),
        )
        .await;
        // The call has settled (returned, failed, timed out): stop accepting
        // progress and replay what it reported to middleware, all before the
        // terminal event below.
        gate.close();
        self.replay_tool_progress(state, ctx, &gate).await;
        let (result, wrap_control) = match outcome {
            Ok(pair) => pair,
            Err(err) => {
                if let Some(request) = execution_deferral(&prepared.call, &err) {
                    // Settled `Deferred` in the ledger (not left `started`):
                    // the call is genuinely paused pending external
                    // resolution, and `resume_deferred` is what eventually
                    // answers it — see `defer_started_tool_call`'s doc
                    // comment.
                    self.defer_started_tool_call(ctx, status, &prepared, request, deferred)
                        .await;
                    return Ok(None);
                }
                self.record_tool_effect_settled(ctx, &prepared, ToolEffectStatus::Failed)
                    .await;
                self.fail_tool_call(
                    ctx,
                    status,
                    &prepared.call_id,
                    &prepared.tool_name,
                    prepared.started_at_ms,
                    &err,
                );
                return Err(err);
            }
        };
        // A `ToolMiddleware::wrap_tool` that short-circuited with
        // `MiddlewareToolOutcome::Command` carries no real result; queue
        // its control the same way `run_wrapped_model`'s call site does
        // (see the comment there).
        if let Some(control) = wrap_control {
            ctx.request_control(control);
        }

        self.record_tool_effect_settled(ctx, &prepared, ToolEffectStatus::Completed)
            .await;
        self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
            .await
    }

    /// Opens the progress gate for one executing call: the sink behind
    /// `ToolRunContext::report_progress`. See [`ToolProgressGate`] for the
    /// ordering guarantees the loop relies on.
    fn open_progress_gate(
        &self,
        ctx: &RunContext<Ctx>,
        prepared: &PreparedToolCall,
    ) -> Arc<ToolProgressGate> {
        ToolProgressGate::new(
            prepared.call_id.clone(),
            prepared.tool_name.clone(),
            ctx.events.clone(),
            ToolProgressLimits::default(),
            // Nothing reads the replay queue when no middleware is registered.
            !self.middleware.is_empty(),
        )
    }

    /// Replays the progress a settled call reported to the middleware stack's
    /// `on_tool_delta`, in order. Progress is advisory, so a failing hook is
    /// logged (the stack has already fanned it out to `on_error`) and does not
    /// fail the call.
    async fn replay_tool_progress(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        gate: &ToolProgressGate,
    ) {
        for mut delta in gate.take_pending() {
            if let Err(error) = self
                .middleware
                .run_on_tool_delta(ctx, state, &mut delta)
                .await
            {
                tracing::warn!(
                    target: "tinyagents::tool_progress",
                    call_id = %delta.call_id,
                    %error,
                    "[tool_progress] on_tool_delta middleware failed; continuing"
                );
            }
        }
    }

    /// Records a call deferred at admission (A2): emits `ToolDeferred` and
    /// files the request under the right [`DeferredToolRequests`] list. The
    /// admission slot was already released by `admit_tool_call`; the call is
    /// re-admitted (and re-counted) if it is later approved.
    fn defer_tool_call(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        request: DeferredRequest,
        deferred: &mut DeferredToolRequests,
    ) {
        let call_id = CallId::new(request.call.id.clone());
        tracing::debug!(
            "[agent_loop::tools] deferring call `{}` for `{}` ({})",
            request.call.id,
            request.call.name,
            request.reason
        );
        let record = ctx.emit(AgentEvent::ToolDeferred {
            call_id: call_id.clone(),
            reason: request.reason.to_string(),
        });
        status.set_last_event(record.id);
        if let Some(metadata) = request.metadata {
            deferred.metadata.insert(call_id, metadata);
        }
        match request.kind {
            DeferredKind::Approval => deferred.approvals.push(request.call),
            DeferredKind::External => deferred.calls.push(request.call),
        }
    }

    /// Terminal partner of [`AgentEvent::ToolStarted`] for a call the *tool
    /// itself* deferred mid-execution by raising `ApprovalRequired` /
    /// `CallDeferred`: closes the in-flight entry, releases the tool-call
    /// slot (the call never produced a result), settles its tool-effect-ledger
    /// row as [`ToolEffectStatus::Deferred`], and files the request.
    ///
    /// Settling to `Deferred` (rather than leaving the row `started`) is
    /// what keeps [`Self::reconcile_tool_effects`] — which only reconciles
    /// rows still `started` — from mistaking this deliberate pause for a
    /// crash artifact on a later resume. See that method's doc comment and
    /// [`AgentHarness::resume_deferred`][crate::agent_loop::AgentHarness::resume_deferred]
    /// for the full picture.
    async fn defer_started_tool_call(
        &self,
        ctx: &mut RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        prepared: &PreparedToolCall,
        request: DeferredRequest,
        deferred: &mut DeferredToolRequests,
    ) {
        release_active_tool_call(status, &prepared.call_id);
        ctx.limits.rollback_tool_calls(1);
        self.record_tool_effect_settled(ctx, prepared, ToolEffectStatus::Deferred)
            .await;
        self.defer_tool_call(ctx, status, request, deferred);
    }

    /// Answers a call that no tool ran — unknown tool, schema-invalid
    /// arguments, or arguments the provider could not parse — through the same
    /// pipeline a real result takes.
    ///
    /// Before this existed the recovery arms pushed a bare
    /// [`Message::tool`] and hand-incremented the counters, so no
    /// `ToolStarted`/`ToolCompleted` pair was emitted, `after_tool` middleware
    /// never saw the result, and accounting lived in three places (TOOL-11). The
    /// transcript content is unchanged: [`ToolResult::error`][err] puts the
    /// message verbatim in `content`.
    ///
    /// [err]: crate::tool::ToolResult::error
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn recover_tool_call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        call: &ToolCall,
        result: tinytools::ToolResult,
    ) -> Result<Option<Message>> {
        tracing::debug!(
            "[agent_loop::tools] answering call `{}` for `{}` without executing a tool",
            call.id,
            call.name
        );
        let prepared = self.start_tool_call(
            ctx,
            status,
            call,
            ToolCallOptions::default(),
            false,
            crate::host::ContentOrigin::Tool,
        );
        self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
            .await
    }

    /// Answers **every** call of a length-truncated turn with a synthetic error
    /// result instead of running it. Used when the structured-output call itself
    /// was cut off, which no other path can answer; ordinary truncated calls are
    /// answered per call at admission instead (see
    /// [`RunPolicy::reject_truncated_tool_calls`][crate::runtime::RunPolicy::reject_truncated_tool_calls]).
    ///
    /// Each call is folded through [`Self::recover_tool_call`], so the
    /// started/terminal pairing, `after_tool` hooks, and accounting match the
    /// other recovery paths; no tool runs and no tool-call budget slot is
    /// spent (the retry loop is bounded by `RunPolicy::truncated_tool_call_retries`).
    pub(super) async fn fail_truncated_tool_calls(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        calls: &[ToolCall],
    ) -> Result<()> {
        let mut follow_ups = Vec::new();
        for call in calls {
            let result = tinytools::ToolResult::error(truncated_tool_call_message(&call.name));
            follow_ups.extend(
                self.recover_tool_call(state, ctx, run, status, messages, call, result)
                    .await?,
            );
        }
        append_follow_ups(messages, follow_ups);
        // Synthetic errors never terminate; drop the votes they recorded.
        ctx.terminate_votes.clear();
        Ok(())
    }

    /// Executes a multi-call turn concurrently, so turn latency is the
    /// slowest tool instead of the sum. Each call's future runs the tool-wrap
    /// onion around the real tool on a shared `&RunContext` (see the module
    /// docs); only reachable when every wrap is `concurrent_safe`.
    #[allow(clippy::too_many_arguments)]
    async fn execute_tools_concurrently(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
        promoted_names: &mut std::collections::BTreeSet<String>,
    ) -> Result<DeferredToolRequests> {
        let mut deferred = DeferredToolRequests::default();
        // Phase 1 — admission, serial, in call order. Nothing is announced and
        // nothing is queued here: an admission failure at call *k* must not
        // leave calls `0..k` with a `ToolStarted` they will never answer, nor
        // an `active_tool_calls` entry for work that is dropped unpolled
        // (TOOL-3). The serial path would have executed those calls; the
        // concurrent path now agrees with it by executing none of them.
        let mut admitted: Vec<AdmittedCall<State, Ctx>> = Vec::with_capacity(tool_calls.len());
        for mut call in tool_calls {
            match self
                .admit_tool_call(state, ctx, status, &mut call, promoted_names)
                .await?
            {
                ResolvedToolCall::Tool { dispatch } => {
                    admitted.push(AdmittedCall::Execute { dispatch, call })
                }
                ResolvedToolCall::Answered(result) => {
                    admitted.push(AdmittedCall::Recovered { call, result })
                }
                ResolvedToolCall::Deferred(request) => {
                    admitted.push(AdmittedCall::Deferred(request))
                }
            }
        }

        // Phase 2 — announce and queue. Every admission succeeded, so every
        // `ToolStarted` emitted here is matched by a terminal event below.
        let mut slots: Vec<ToolSlot> = Vec::with_capacity(admitted.len());
        let mut prepared: Vec<PreparedToolCall> = Vec::new();
        let mut futures: Vec<_> = Vec::new();
        // One progress gate per executing call, aligned with `prepared`.
        let mut gates: Vec<Arc<ToolProgressGate>> = Vec::new();
        // Concurrent dispatch receives a shared parent snapshot; the mutable
        // run context stays with the fold phase after all futures complete.
        let parent_ctx: &RunContext<Ctx> = ctx;
        for entry in admitted {
            let (dispatch, call) = match entry {
                AdmittedCall::Execute { dispatch, call } => (dispatch, call),
                AdmittedCall::Recovered { call, result } => {
                    slots.push(ToolSlot::Recovered { call, result });
                    continue;
                }
                AdmittedCall::Deferred(request) => {
                    slots.push(ToolSlot::Deferred(request));
                    continue;
                }
            };

            let options = dispatch.call_options(&call.arguments);
            prepared.push(self.start_tool_call(
                ctx,
                status,
                &call,
                options,
                true,
                dispatch.output_origin(),
            ));
            let just_prepared = prepared.last().expect("just pushed");
            if let Err(err) = self
                .record_tool_effect_started(ctx, &call.arguments, just_prepared)
                .await
            {
                // Every call in `prepared` so far (including this one) already
                // emitted `ToolStarted`; give each one a terminal event before
                // bailing, mirroring the sibling-abort handling in phase 4.
                for sibling in &prepared {
                    self.fail_tool_call(
                        ctx,
                        status,
                        &sibling.call_id,
                        &sibling.tool_name,
                        sibling.started_at_ms,
                        &err,
                    );
                }
                return Err(err);
            }
            slots.push(ToolSlot::Execute);
            let gate = self.open_progress_gate(ctx, prepared.last().expect("just pushed"));
            gates.push(Arc::clone(&gate));

            // Each call runs the full wrap onion around the real tool, exactly
            // as the serial path does: the recoverable per-tool timeout lives
            // in `ToolCallBase`, the run's hard remaining wall-clock budget
            // bounds the whole wrapped call. The future owns its call and base
            // and shares only `&RunContext`, so it runs alongside its siblings.
            let base = ToolCallBase {
                harness: self,
                dispatch,
                options,
                timeout_settings: self.tool_timeouts.clone(),
                level: 0,
                nested_state: Default::default(),
                gate_held: super::nested::GateHold::None,
            };
            let run_budget = self.call_budget(ctx);
            let run_id = ctx.run_id().as_str().to_string();
            futures.push(async move {
                let body = async move {
                    let wrapped = self
                        .middleware
                        .run_wrapped_tool(parent_ctx, state, call, &base);
                    // As in serial mode, canonical execution errors remain
                    // fatal; reported tool errors travel in
                    // `ToolResult::is_error`.
                    let guarded = futures::FutureExt::map(wrapped, |result| {
                        result.map(|outcome| outcome.into_result_with_control())
                    });
                    Self::with_call_budget(
                        run_budget,
                        &run_id,
                        "tool call",
                        super::model_call::RUN_BOUND_LABEL,
                        guarded,
                    )
                    .await
                };
                let outcome = gate.scope(body).await;
                // Close the moment *this* call settles, not when the whole
                // batch has: a late update must not slip in while siblings
                // are still running.
                gate.close();
                outcome
            });
        }

        // Phase 3 — run all admitted calls concurrently, bounded by
        // `RunLimits::max_tool_concurrency` when set (I-8). `buffered(n)`
        // polls up to `n` futures at once and yields them **in input order**
        // (unlike `buffer_unordered`), so results still pair 1:1 with
        // `prepared` exactly as `join_all` (the unbounded case) did.
        let concurrency = self
            .policy
            .limits
            .max_tool_concurrency
            .unwrap_or(futures.len().max(1));
        let results: Vec<_> = futures::stream::iter(futures)
            .buffered(concurrency)
            .collect()
            .await;

        // Phase 4 — fold in original call order: the first call whose policy
        // kept its failure fatal (in that order) fails the turn; siblings
        // already ran to completion.
        let mut executed = prepared.into_iter().zip(gates).zip(results);
        let mut follow_ups = Vec::new();
        for slot in slots {
            match slot {
                ToolSlot::Recovered { call, result } => {
                    follow_ups.extend(
                        self.recover_tool_call(state, ctx, run, status, messages, &call, result)
                            .await?,
                    );
                }
                ToolSlot::Deferred(request) => {
                    self.defer_tool_call(ctx, status, request, &mut deferred);
                }
                ToolSlot::Execute => {
                    let ((prepared, gate), result) = executed
                        .next()
                        .expect("every Execute slot has a prepared/result pair");
                    // Replay this call's progress to middleware before any
                    // terminal event for it (its live events are already out).
                    self.replay_tool_progress(state, ctx, &gate).await;
                    let (result, wrap_control) = match result {
                        Ok(pair) => pair,
                        Err(err) => {
                            if let Some(request) = execution_deferral(&prepared.call, &err) {
                                self.defer_started_tool_call(
                                    ctx,
                                    status,
                                    &prepared,
                                    request,
                                    &mut deferred,
                                )
                                .await;
                                continue;
                            }
                            self.record_tool_effect_settled(
                                ctx,
                                &prepared,
                                ToolEffectStatus::Failed,
                            )
                            .await;
                            self.fail_tool_call(
                                ctx,
                                status,
                                &prepared.call_id,
                                &prepared.tool_name,
                                prepared.started_at_ms,
                                &err,
                            );
                            // Every remaining `Execute` slot already emitted
                            // `ToolStarted` (phase 2) and is registered in
                            // `status.active_tool_calls`, but its future
                            // already resolved (phase 3 ran every future to
                            // completion via `join_all`) without ever getting
                            // a terminal event, because this fold stopped
                            // here. Give each of them one now so every
                            // `ToolStarted` still has exactly one terminal
                            // partner and no tool call is reported in-flight
                            // after the run has already failed.
                            let aborted = TinyAgentsError::Tool(
                                "aborted: sibling tool call failed".to_string(),
                            );
                            for ((sibling_prepared, sibling_gate), _) in executed {
                                self.replay_tool_progress(state, ctx, &sibling_gate).await;
                                self.record_tool_effect_settled(
                                    ctx,
                                    &sibling_prepared,
                                    ToolEffectStatus::Failed,
                                )
                                .await;
                                self.fail_tool_call(
                                    ctx,
                                    status,
                                    &sibling_prepared.call_id,
                                    &sibling_prepared.tool_name,
                                    sibling_prepared.started_at_ms,
                                    &aborted,
                                );
                            }
                            return Err(err);
                        }
                    };
                    // A wrap that short-circuited with `Command` carries no
                    // real result; queue its control as the serial path does.
                    if let Some(control) = wrap_control {
                        ctx.request_control(control);
                    }
                    self.record_tool_effect_settled(ctx, &prepared, ToolEffectStatus::Completed)
                        .await;
                    follow_ups.extend(
                        self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
                            .await?,
                    );
                }
            }
        }
        append_follow_ups(messages, follow_ups);
        Ok(deferred)
    }
}

/// The error a call from a length-truncated response is answered with.
fn truncated_tool_call_message(tool_name: &str) -> String {
    format!(
        "Tool call `{tool_name}` was not executed: your response hit the output token limit \
         mid-turn, so its arguments may be truncated. Re-issue the tool call with complete \
         arguments (split large content into smaller calls if needed)."
    )
}

/// Appends a batch's follow-up user messages (B2) after its last tool row,
/// in the calls' original order.
///
/// One user message per call that returned `follow_up` content, rather than
/// one merged message: each keeps its own block list, and a provider that
/// merges adjacent user turns does so on the wire anyway.
pub(super) fn append_follow_ups(messages: &mut Vec<Message>, follow_ups: Vec<Message>) {
    messages.extend(follow_ups);
}

/// Builds the user message a result's `follow_up` blocks become (B2), or
/// `None` when the result has none.
///
/// Block mapping — the same as the tool row's, except an image gets a real
/// [`ContentBlock::Image`] because a *user* message may carry one:
/// - `Text` → `Text`; `Json` → `Json`.
/// - `Image` → `Image(ImageRef)`: a URL as-is, inline bytes as a
///   `data:<media_type>;base64,<bytes>` URI; `mime_type` set from the block.
/// - `File` → `Text("[file <name> (<media_type>)]")` (the vendor
///   `ToolContent::render` placeholder), because the message model has no
///   file block yet; the host still has the full block on the event side if
///   it needs the bytes.
fn follow_up_message(follow_up: &[tinytools::ToolContent]) -> Option<Message> {
    if follow_up.is_empty() {
        return None;
    }
    let content = follow_up
        .iter()
        .map(|block| match block {
            tinytools::ToolContent::Text { text } => ContentBlock::Text(text.clone()),
            tinytools::ToolContent::Json { data } => ContentBlock::Json(data.clone()),
            tinytools::ToolContent::Image { media_type, data } => {
                let url = match data {
                    tinytools::ImageData::Url(url) => url.clone(),
                    tinytools::ImageData::Base64(bytes) => {
                        format!("data:{media_type};base64,{bytes}")
                    }
                };
                ContentBlock::Image(tinyinference_llm::message::ImageRef {
                    url,
                    mime_type: Some(media_type.clone()),
                })
            }
            file @ tinytools::ToolContent::File { .. } => ContentBlock::Text(file.render()),
        })
        .collect();
    Some(Message::User(tinyinference_llm::message::UserMessage {
        content,
    }))
}

/// Turns an execution-time `ApprovalRequired`/`CallDeferred` (raised by the
/// tool through `Err`, and passed through [`execute_tool_recovering_model_retry`]
/// untouched) into the request the loop hands back, or `None` for any other
/// error.
fn execution_deferral(call: &ToolCall, error: &TinyAgentsError) -> Option<DeferredRequest> {
    match error {
        TinyAgentsError::ApprovalRequired { metadata } => Some(DeferredRequest::approval(
            call.clone(),
            "approval_required",
            Some(metadata.clone()),
        )),
        TinyAgentsError::CallDeferred { metadata } => Some(DeferredRequest::external(
            call.clone(),
            "call_deferred",
            Some(metadata.clone()),
        )),
        _ => None,
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Reconciles unresolved tool-effect-ledger rows (B5) before resuming a
    /// run from durable transcript `messages`.
    ///
    /// A crash between [`Self::record_tool_effect_started`] and the matching
    /// settle write (or between the settle write and the tool result being
    /// folded into `messages`) leaves a `started` row a resumed run must
    /// resolve one way or another before it can safely feed `messages` back
    /// into the loop: the last assistant turn may still carry a tool call
    /// with no matching [`Message::Tool`] answer.
    ///
    /// For every tool call on the *last* assistant message that has no
    /// [`Message::Tool`] answer yet **and** an unresolved (`started`) ledger
    /// row, this consults the tool's declared
    /// [`tinytools::ToolReplay`][tinytools::ToolPolicy::runtime]:
    ///
    /// - [`tinytools::ToolReplay::Safe`]: the call is left unanswered.
    ///   `messages` is not appended to for that call, so the normal loop
    ///   re-executes it exactly as it would a fresh call — the tool declared
    ///   this safe.
    /// - [`tinytools::ToolReplay::Never`] (the default): a synthesized
    ///   tool-error result ("interrupted before settlement") is appended in
    ///   place of a real answer, the ledger row is settled as
    ///   [`crate::tool::ToolEffectStatus::Interrupted`], and the loop never
    ///   re-attempts the call.
    ///
    /// A call whose tool is no longer registered on this harness (renamed,
    /// removed since the interrupted run) is treated as [`tinytools::ToolReplay::Never`]
    /// — fail closed rather than blindly re-run an unknown effect.
    ///
    /// Only ledger rows still in [`crate::tool::ToolEffectStatus::Started`]
    /// are candidates: a call deferred mid-execution
    /// (`ApprovalRequired`/`CallDeferred`) is settled as
    /// [`crate::tool::ToolEffectStatus::Deferred`] by `defer_started_tool_call`
    /// the moment it pauses, so [`crate::tool::ToolEffectLedger::unresolved`]
    /// — which lists only `started` rows — never surfaces it here; a `Deferred`
    /// row is exactly what [`AgentHarness::resume_deferred`] settles to
    /// `Completed`/`Failed` once its answer runs.
    ///
    /// `excluded` is a second, defense-in-depth guard against the same
    /// mistake: any call id in it is skipped even if its ledger row is
    /// (unexpectedly) still `started` — e.g. the `Deferred` settle write
    /// above failed and was only logged (settle writes are best-effort, see
    /// [`Self::record_tool_effect_settled`]). [`AgentHarness::resume_deferred`]
    /// passes the ids `results` is about to answer; any other caller — a host
    /// reconciling a genuine crash, where no `results` exists at all — passes
    /// an empty set.
    ///
    /// Returns the messages synthesized for `Never`-classified calls (already
    /// appended to `messages` as well), so a caller that journals messages
    /// separately from the in-memory transcript knows what changed. Returns
    /// an empty `Vec` immediately, without any ledger I/O, when `ctx` has no
    /// [`crate::tool::ToolEffectLedger`] attached or the transcript has no
    /// pending tool calls.
    pub async fn reconcile_tool_effects(
        &self,
        ctx: &RunContext<Ctx>,
        run_id: &str,
        messages: &mut Vec<Message>,
        excluded: &std::collections::HashSet<CallId>,
    ) -> Result<Vec<Message>> {
        let mut synthesized = Vec::new();
        let Some(ledger) = ctx.tool_effect_ledger.clone() else {
            return Ok(synthesized);
        };

        // The calls a resumed run must judge are exactly the tool calls on
        // the *last* assistant turn — any earlier assistant tool-call turn
        // already has its answers folded in by definition, since the loop
        // never advances past an unanswered turn.
        let Some(pending_calls) = messages.iter().rev().find_map(|message| match message {
            Message::Assistant(assistant) if !assistant.tool_calls.is_empty() => {
                Some(assistant.tool_calls.clone())
            }
            _ => None,
        }) else {
            return Ok(synthesized);
        };
        let already_answered: std::collections::HashSet<&str> = messages
            .iter()
            .filter_map(|message| match message {
                Message::Tool(tool_message) => Some(tool_message.tool_call_id.as_str()),
                _ => None,
            })
            .collect();
        let unanswered: Vec<&ToolCall> = pending_calls
            .iter()
            .filter(|call| {
                !already_answered.contains(call.id.as_str())
                    && !excluded.contains(&CallId::new(call.id.clone()))
            })
            .collect();
        if unanswered.is_empty() {
            return Ok(synthesized);
        }

        let unresolved = ledger.unresolved(run_id).await?;
        for call in unanswered {
            let Some(effect) = unresolved.iter().find(|effect| effect.call_id == call.id) else {
                // No ledger row for this call: nothing was ever journaled as
                // started for it (e.g. a ledger was attached only after the
                // interrupted attempt began), so there is nothing to
                // reconcile — leave it for the loop to handle as it always
                // has.
                continue;
            };
            let replay = self
                .tools
                .dispatch(&call.name)
                .map(|dispatch| dispatch.tool().policy().runtime.replay)
                .unwrap_or(tinytools::ToolReplay::Never);
            match replay {
                tinytools::ToolReplay::Safe => {
                    ctx.emit(AgentEvent::ToolEffectReconciled {
                        call_id: CallId::new(call.id.clone()),
                        action: "re_execute".to_string(),
                    });
                    tracing::info!(
                        "[agent_loop::tools] reconciling unresolved tool effect for call `{}` \
                         (tool `{}`) as ToolReplay::Safe — leaving unanswered for re-execution",
                        call.id,
                        call.name
                    );
                }
                tinytools::ToolReplay::Never => {
                    let result = tinytools::ToolResult::error("interrupted before settlement");
                    let tool_message = tool_message_from_result(
                        call.id.clone(),
                        &result,
                        ToolCallOptions::default(),
                    );
                    messages.push(Message::Tool(tool_message.clone()));
                    synthesized.push(Message::Tool(tool_message));
                    if let Err(err) = ledger
                        .settled(ToolEffectSettle {
                            run_id: crate::ids::RunId::new(run_id),
                            call_id: CallId::new(call.id.clone()),
                            status: ToolEffectStatus::Interrupted,
                            effect_summary: Some(effect.tool.clone()),
                        })
                        .await
                    {
                        tracing::warn!(
                            "[agent_loop::tools] failed to settle interrupted tool effect for \
                             call `{}` (tool `{}`): {err}",
                            call.id,
                            call.name
                        );
                    }
                    ctx.emit(AgentEvent::ToolEffectReconciled {
                        call_id: CallId::new(call.id.clone()),
                        action: "interrupted".to_string(),
                    });
                    tracing::info!(
                        "[agent_loop::tools] reconciled unresolved tool effect for call `{}` \
                         (tool `{}`) as ToolReplay::Never — synthesized an interrupted result",
                        call.id,
                        call.name
                    );
                }
            }
        }
        Ok(synthesized)
    }
}

/// Decides whether a batch may leave the serial path.
///
/// Lifecycle middleware never forced serial execution: `before_tool` hooks
/// that can rewrite a call run during **admission** (`admit_tool_call`, phase 1
/// of [`AgentHarness::execute_tools_concurrently`]), which is serial and
/// completes in full — for every call in the batch — before any concurrent
/// future is built (I-8). Tool-*wrap* middleware used to force it because
/// `wrap_tool` held `&mut RunContext`; it now takes `&RunContext` and runs
/// inside each concurrent future, so only a wrap that opts out through
/// [`crate::middleware::ToolMiddleware::concurrent_safe`] keeps the serial
/// route (`tool_wraps_concurrent_safe == false`).
fn should_execute_tools_concurrently(
    calls: usize,
    canonical_parallel_safe: bool,
    tool_wraps_concurrent_safe: bool,
) -> bool {
    calls > 1 && canonical_parallel_safe && tool_wraps_concurrent_safe
}

/// A batch may leave the serial path only when every registered declaration
/// opts in for raw arguments that need no host-owned preparation. Unknown
/// calls and injected arguments remain serial: injection is performed at
/// admission, and an untrusted model value must not select concurrency before
/// its authoritative replacement exists.
fn batch_is_canonical_parallel_safe<State: Send + Sync, Ctx: Send + Sync>(
    tools: &crate::tool::ToolRegistry<State, Ctx>,
    calls: &[ToolCall],
) -> bool {
    calls.iter().all(|call| {
        tools.get(&call.name).is_some_and(|tool| {
            tool.injected_arguments().is_empty() && tool.is_concurrency_safe(&call.arguments)
        })
    })
}

/// Removes **one** occurrence of `call_id` from the in-flight list.
///
/// Positional, not `retain`: a provider can emit two calls in one turn that
/// share a `tool_call_id`, and a predicate-based removal would clear both
/// entries when the first completes — leaving the second call in flight with no
/// entry to close, and the run reporting an empty in-flight list while a tool is
/// still running (TOOL-10).
fn release_active_tool_call(status: &mut HarnessRunStatus, call_id: &CallId) {
    if let Some(position) = status
        .active_tool_calls
        .iter()
        .position(|active| active == call_id)
    {
        status.active_tool_calls.remove(position);
    }
}

/// Converts the canonical block result into the provider-neutral transcript
/// shape without discarding its richer host-side representation.
///
/// A preferred markdown rendering is model-facing, so it replaces the provider
/// message body only when it is non-blank. The complete ordered TinyTools block
/// list, optional markdown, and reported-error bit remain in the artifact for
/// host consumers and transcript persistence.
fn tool_message_from_result(
    tool_call_id: String,
    result: &tinytools::ToolResult,
    options: ToolCallOptions,
) -> tinyinference_llm::message::ToolMessage {
    let markdown_selected = options.prefer_markdown
        && result
            .markdown_formatted
            .as_deref()
            .is_some_and(|markdown| !markdown.trim().is_empty());
    let content = if markdown_selected {
        vec![ContentBlock::Text(result.output_for_llm(true))]
    } else {
        result
            .content
            .iter()
            .map(|block| match block {
                tinytools::ToolContent::Text { text } => ContentBlock::Text(text.clone()),
                tinytools::ToolContent::Json { data } => ContentBlock::Json(data.clone()),
                // Image/File blocks have no provider-neutral `ContentBlock`
                // representation yet (see `docs/sdk-gaps/tools.md`); render the same
                // short placeholder `ToolContent::render()` uses so a model
                // still sees *something* rather than the block vanishing.
                other @ (tinytools::ToolContent::Image { .. }
                | tinytools::ToolContent::File { .. }) => ContentBlock::Text(other.render()),
            })
            .collect()
    };
    let artifact = serde_json::json!({
        "tinytools_content": result.content,
        "markdown_formatted": result.markdown_formatted,
        "is_error": result.is_error,
        // A canonical tool result never gets to declare that its own output
        // bypasses host framing. Trust is host/dispatch policy; until that
        // policy explicitly marks a delivery, the safe default is false.
        "trusted_verbatim": false,
    });

    tinyinference_llm::message::ToolMessage {
        tool_call_id,
        content,
        trusted_verbatim: false,
        artifact: Some(artifact),
    }
}

/// Maps a canonical-dispatch failure back to the harness error surface.
///
/// Cancellation, timeout, and the structural errors that can escape a nested
/// sub-agent call ([`TinyAgentsError::SubAgentDepth`],
/// [`TinyAgentsError::LimitExceeded`]) keep their own typed classification.
/// Every other typed or foreign error is collapsed to a generic
/// [`TinyAgentsError::Tool`] because message-bearing errors from arbitrary
/// tool code can include credentials or user data exposed to model/event
/// consumers.
///
/// Preserving the structural variants matters for retry correctness, not just
/// diagnostics: [`crate::retry::is_retryable`] treats every
/// [`TinyAgentsError::Tool`] as unconditionally retryable (arbitrary
/// tool-authored text has no shared vocabulary to classify against), but a
/// depth cap or run-limit violation is deterministic and will never succeed
/// on retry. Flattening `SubAgentDepth`/`LimitExceeded` into `Tool` made a
/// `RetryMiddleware` around tools re-run a permanently failing sub-agent call
/// until its attempt budget was exhausted (M-3).
/// Runs a dispatch call, folding a [`TinyAgentsError::ModelRetry`]/
/// [`TinyAgentsError::ToolFailed`] the tool raised as `Err` into a
/// recoverable [`tinytools::ToolResult`] instead of aborting the run.
///
/// This is A3's unified retry/failure vocabulary for tool errors: a tool that
/// wants "ask the model to try again" (the common case — a transient or
/// correctable failure) returns `Err(TinyAgentsError::ModelRetry(..).into())`
/// instead of `Ok(ToolResult::error(..))`, so it reads the same as any other
/// `?`-propagated failure in the tool's implementation while the harness
/// still folds it into the ordinary "tool ran, told the model to fix it"
/// transcript path (via [`tinytools::ToolResult::retry`]) rather than ending
/// the run. `ToolFailed` is the permanent counterpart
/// ([`tinytools::ToolResult::failed`]); every other error still maps through
/// [`map_tool_dispatch_error`] unchanged, preserving TinyTools' "`Err` aborts
/// the run" contract for genuine dispatch failures.
pub(super) async fn execute_tool_recovering_model_retry<Fut>(
    fut: Fut,
) -> Result<tinytools::ToolResult>
where
    Fut: std::future::Future<Output = anyhow::Result<tinytools::ToolResult>>,
{
    match fut.await {
        Ok(result) => Ok(result),
        Err(error) => match error.downcast::<TinyAgentsError>() {
            Ok(TinyAgentsError::ModelRetry(message)) => Ok(tinytools::ToolResult::retry(message)),
            Ok(TinyAgentsError::ToolFailed(message)) => Ok(tinytools::ToolResult::failed(message)),
            // A2: a deferral is a typed signal for the loop, not a failure to
            // redact. The metadata is host-only (never model-visible), so it
            // is safe to carry through the wrap onion to the fold.
            Ok(
                deferral @ (TinyAgentsError::ApprovalRequired { .. }
                | TinyAgentsError::CallDeferred { .. }),
            ) => Err(deferral),
            Ok(other) => Err(map_tool_dispatch_error(anyhow::Error::from(other))),
            Err(error) => Err(map_tool_dispatch_error(error)),
        },
    }
}

pub(super) fn map_tool_dispatch_error(error: anyhow::Error) -> TinyAgentsError {
    match error.downcast::<TinyAgentsError>() {
        Ok(TinyAgentsError::Cancelled) => TinyAgentsError::Cancelled,
        Ok(TinyAgentsError::Timeout(message)) => TinyAgentsError::Timeout(message),
        Ok(TinyAgentsError::CallTimeout(message)) => TinyAgentsError::CallTimeout(message),
        // `usize` carries no free-form content, so it is always safe to keep.
        Ok(TinyAgentsError::SubAgentDepth(depth)) => TinyAgentsError::SubAgentDepth(depth),
        // The message is harness-generated (a limit description), not
        // attacker/tool-controlled, but is redacted anyway for the same
        // "never assume a message is safe" posture as every other variant
        // here; only the *classification* needs to survive for retry
        // purposes.
        Ok(TinyAgentsError::LimitExceeded(_)) => {
            TinyAgentsError::LimitExceeded("tool dispatch hit a run limit".to_string())
        }
        Ok(_) => TinyAgentsError::Tool("tool dispatch failed".to_string()),
        Err(_) => TinyAgentsError::Tool("tool dispatch failed".to_string()),
    }
}

/// Repairs provider-neutral argument shape defects before schema validation.
///
/// Schema-valid arguments are already canonical. Otherwise the protocol
/// crate's argument repairs ([`tinytools_agent::repair::args`]) are applied in
/// order — a stringified (possibly fenced, possibly relaxed) JSON document is
/// decoded; an object buried one level down in an envelope the model invented
/// is unwrapped; string scalars are coerced to the types the schema declares —
/// and each rewrite is kept only when the result validates, or (for the
/// decode) when it is at least the object the model meant, so the validation
/// error the model sees stays precise. Undecodable or non-object values become
/// an empty object only for object-capable schemas that declare no required
/// fields; required-field schemas retain the original value.
///
/// This is host policy, not parsing: it runs only under a recovering
/// [`InvalidArgsPolicy`](crate::runtime::InvalidArgsPolicy), and the schema
/// validator that gates every rewrite is the harness's.
pub(super) fn normalize_tool_arguments(call: &mut ToolCall, schema: &ToolSchema) {
    use tinytools_agent::repair::args;

    // Never rewrite a value the declared schema already accepts. In
    // particular, an object-capable union may validly accept a primitive too.
    if schema.validate_call(call).is_ok() {
        return;
    }

    let parameters = &schema.parameters;
    if !args::accepts_object(parameters) {
        return;
    }

    let template = ToolCall::new(call.id.clone(), call.name.clone(), Value::Null);
    let validates = |arguments: &Value| {
        let mut candidate = template.clone();
        candidate.arguments = arguments.clone();
        schema.validate_call(&candidate).is_ok()
    };

    if let Some(raw) = call.arguments.as_str() {
        let candidate = tinytools_agent::repair::json::strip_code_fence(raw);
        let decoded = serde_json::from_str::<Value>(candidate)
            .ok()
            .or_else(|| tinytools_agent::repair::json::recover_object(candidate));
        if let Some(value) = decoded {
            // Decoding must be lossless even when the decoded value is still
            // schema-invalid. Preserve it so the validation below reports the
            // actual bad field/type instead of silently replacing it with `{}`.
            call.arguments = value;
            if validates(&call.arguments) {
                return;
            }
            // A successfully decoded non-object value (e.g. the stringified
            // JSON `true`) is not an object, so it never reaches the
            // `is_object` repair branch below — it would otherwise fall
            // through to the has-no-required-fields fallback further down
            // and get silently replaced with `{}`, discarding the decoded
            // scalar the model actually sent and making an invalid-typed
            // call quietly "succeed" with fabricated empty arguments instead
            // of surfacing its real validation error. Only a value that
            // never decoded at all should reach that fallback.
            if !call.arguments.is_object() {
                return;
            }
        }
    }

    // A provider-native object is already the shape normalization is trying to
    // recover. If its contents violate the schema, try the envelope unwrap and
    // the scalar coercion, each kept only when it validates; otherwise
    // preserve them so the model sees the real validation error instead of
    // executing with an empty object.
    if call.arguments.is_object() {
        if let Some(inner) = args::unwrap_envelope(&call.arguments, parameters, &validates) {
            call.arguments = inner;
            return;
        }
        let coerced = args::coerce_to_schema(call.arguments.clone(), parameters);
        if coerced != call.arguments && validates(&coerced) {
            call.arguments = coerced;
        }
        return;
    }

    let has_required_fields = parameters
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| required.iter().any(Value::is_string));
    if !has_required_fields {
        call.arguments = serde_json::json!({});
    }
}

pub(super) fn timeout_result(
    call: &ToolCall,
    timeout: Option<crate::tool::ResolvedToolTimeout>,
) -> tinytools::ToolResult {
    let budget_ms = timeout.map_or(0, |resolved| resolved.budget_ms);
    tinytools::ToolResult::error(format!(
        "tool `{}` timed out after {budget_ms} ms",
        call.name
    ))
}

#[cfg(test)]
#[path = "tools_canonical_result_tests.rs"]
mod canonical_result_tests;

#[cfg(test)]
#[path = "tools_progress_tests.rs"]
mod progress_tests;

/// Stamps the metadata a refusing `before_tool` hook queued for `call_id`
/// ([`RunContext::set_refusal_metadata`]) onto the result that answers it.
fn with_refusal_metadata<Ctx: Send + Sync>(
    ctx: &mut RunContext<Ctx>,
    call_id: &str,
    mut result: tinytools::ToolResult,
) -> tinytools::ToolResult {
    if let Some(metadata) = ctx.take_refusal_metadata(call_id) {
        result.metadata = Some(metadata);
    }
    result
}
