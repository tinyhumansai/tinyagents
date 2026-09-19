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
//! ## Why tool-wrap middleware forces serial execution
//!
//! [`crate::middleware::ToolMiddleware::wrap_tool`] holds
//! `&mut RunContext` across the entire wrapped call — that exclusive borrow is
//! part of its public contract (it may mutate limits, request control, etc.).
//! Two wrapped calls therefore cannot be in flight at once without changing
//! the trait, so a harness with tool-wrap middleware keeps the serial path.
//! Lifecycle `before_tool`/`after_tool` hooks do *not* force serial execution:
//! they run in the admission/fold phases and still bracket each call.
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
//!   failure should be run under a tool-wrap middleware (serial) or a harness
//!   without parallel-capable turns.
//!
use super::model_call::ToolCallBase;
use super::*;
use crate::tool::{ToolDispatch, provider_schema};
use tinyinference_llm::message::ContentBlock;
use tinytools::{ToolCall as CanonicalToolCall, ToolCallId, ToolCallOptions};

/// How a single requested tool call was resolved during admission.
enum ResolvedToolCall<State: Send + Sync, Ctx: Send + Sync> {
    /// A registered tool (possibly after an unknown-tool rewrite).
    Tool {
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
        tool: Arc<dyn tinytools::Tool>,
    },
    /// No tool runs; this result is appended to the transcript at the call's
    /// original position. A tool-error result for the recovery paths (unknown
    /// tool, invalid arguments); a success result for an intrinsic answer
    /// (`tool_search`).
    Answered(tinytools::ToolResult),
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
        tool: Arc<dyn tinytools::Tool>,
        call: ToolCall,
    },
    /// A recovery or an intrinsic answer: no tool runs, but the call is still
    /// answered through the normal result pipeline so it emits the same
    /// started/completed pair and runs the same `after_tool` hooks (TOOL-11).
    Recovered {
        call: ToolCall,
        result: tinytools::ToolResult,
    },
}

/// One transcript slot per requested call, in original order, used by the
/// concurrent path to reassemble results deterministically.
enum ToolSlot {
    /// An executed call: consumes the next prepared/result pair in order.
    Execute,
    /// A recovery, folded in place through the normal result pipeline.
    Recovered {
        call: ToolCall,
        result: tinytools::ToolResult,
    },
}

/// Admission metadata for one executable call, paired 1:1 (in order) with its
/// execution future/result on the concurrent path.
struct PreparedToolCall {
    call_id: CallId,
    tool_name: String,
    options: ToolCallOptions,
    captured_input: Option<Value>,
    started_at_ms: u64,
    executed: bool,
    output_origin: crate::host::ContentOrigin,
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

    /// Builds the run's deferred-tool catalogue: every
    /// [`tinytools::ToolExposure::Deferred`] registration the host allow-list
    /// admits, or an empty catalogue when discovery is disabled.
    pub(super) fn deferred_catalog(
        &self,
        host_allows: &dyn Fn(&str) -> bool,
    ) -> crate::tool::discover::DeferredCatalog {
        if !self.policy.discovery.enabled {
            return crate::tool::discover::DeferredCatalog::default();
        }
        let mut schemas = self
            .tools
            .deferred_schemas()
            .into_iter()
            .filter(|schema| host_allows(&schema.name))
            .collect::<Vec<_>>();
        if let Some(preparation) = &self.policy.tool_schemas {
            schemas = crate::tool::prepare_tool_schemas(&schemas, preparation);
        }
        crate::tool::discover::DeferredCatalog::build(schemas)
    }

    /// Resolves the discovery bridge for one call, when it is one.
    ///
    /// Returns `Some` with the answer for a `tool_search` call (no tool runs),
    /// `None` after rewriting a `tool_call` in place to the real tool so
    /// admission continues with it, and `None` untouched for any other name.
    /// A malformed `tool_call` payload is answered with a tool error rather
    /// than passed on, so the model can correct it.
    fn answer_discovery_bridge(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call: &mut ToolCall,
    ) -> Result<Option<ResolvedToolCall<State, Ctx>>> {
        use crate::tool::discover::{TOOL_CALL_NAME, TOOL_SEARCH_NAME};
        if !self.policy.discovery.enabled
            || (call.name != TOOL_SEARCH_NAME && call.name != TOOL_CALL_NAME)
        {
            return Ok(None);
        }
        // Reuses `resolve_tool_allowlist` (I-9's fail-closed allow-list
        // resolution) rather than reading `binding.allowed_tools` directly,
        // so the discovery bridge honors the same
        // `fail_closed_tool_allowlist` policy as the direct tool set built in
        // `run_loop_body` — an empty declared list never falls back to
        // "unrestricted" here either.
        let allowed_tools = self.resolve_tool_allowlist(ctx)?;
        let host_allows = |name: &str| {
            allowed_tools
                .as_ref()
                .is_none_or(|allowed| allowed.contains(name))
        };
        let catalog = self.deferred_catalog(&host_allows);
        if catalog.is_empty() {
            // Nothing was deferred, so the bridge was never advertised; let
            // the call fall through to the unknown-tool policy.
            return Ok(None);
        }
        if call.name == TOOL_SEARCH_NAME {
            let (result, matched) = crate::tool::discover::answer_tool_search(
                &catalog,
                &self.policy.discovery,
                &call.arguments,
            );
            let record = ctx.emit(AgentEvent::ToolSearched {
                call_id: CallId::new(call.id.clone()),
                query: call
                    .arguments
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                matched,
            });
            status.set_last_event(record.id);
            return Ok(Some(ResolvedToolCall::Answered(result)));
        }
        match crate::tool::discover::unwrap_tool_call(&call.arguments) {
            Ok((name, arguments)) => {
                // `unwrap_tool_call` accepts any non-empty `name` — it only
                // validates the wrapper's shape, not that `name` is actually
                // in the deferred catalogue. A model can wrap a direct,
                // hidden, or entirely fabricated name in a `tool_call`
                // payload just as validly, and admission (via
                // `model_dispatch`/the unknown-tool policy below) decides
                // what happens to it next. Emitting `DeferredToolCall`
                // unconditionally would misrepresent that outcome to an
                // audit consumer — recording "a deferred call happened" for
                // a call that admission is about to execute as a direct
                // tool or reject as unknown/hidden. Only emit it when the
                // target is actually in the catalogue this bridge searched.
                if catalog.get(&name).is_some() {
                    let record = ctx.emit(AgentEvent::DeferredToolCall {
                        call_id: CallId::new(call.id.clone()),
                        tool_name: name.clone(),
                    });
                    status.set_last_event(record.id);
                }
                call.name = name;
                call.arguments = arguments;
                Ok(None)
            }
            Err(message) => Ok(Some(ResolvedToolCall::Answered(
                tinytools::ToolResult::error(message),
            ))),
        }
    }

    /// Resolves this tool's own timeout policy. The separate run wall-clock
    /// budget remains the outer hard deadline: a per-tool timeout becomes a
    /// recoverable tool-error result, while exhausting the run budget aborts.
    fn resolved_tool_timeout(
        &self,
        tool: &dyn tinytools::Tool,
        call: &ToolCall,
    ) -> Option<crate::tool::ResolvedToolTimeout> {
        self.tool_timeouts
            .as_ref()
            .map(|settings| settings.resolve(tool.timeout_policy(&call.arguments)))
    }

    /// Races `fut` against `timeout`'s deadline (if any), returning
    /// `timeout_value` instead of an error when the deadline elapses first.
    ///
    /// This is the crate-owned tool-timeout policy: unlike
    /// [`Self::with_call_budget`], which surfaces a run-level
    /// [`TinyAgentsError::Timeout`] and aborts the run, an elapsed per-tool
    /// deadline here becomes a *recoverable* `Ok(timeout_value)` (a
    /// `ToolResult::error`) so the run continues and the model can react.
    async fn with_tool_policy_timeout<T, F>(
        timeout: Option<crate::tool::ResolvedToolTimeout>,
        timeout_value: T,
        fut: F,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match timeout.and_then(|resolved| resolved.deadline) {
            Some(deadline) => match tokio::time::timeout(deadline, fut).await {
                Ok(result) => result,
                Err(_) => Ok(timeout_value),
            },
            None => fut.await,
        }
    }

    /// Executes one assistant turn's requested tool calls, appending each
    /// result to `messages` in the calls' original order.
    ///
    /// Dispatches to the concurrent path when it is safe (see the module docs
    /// for the exact conditions and preserved semantics); otherwise runs the
    /// historical serial path.
    pub(super) async fn execute_tools(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<()> {
        // Injection and argument normalization change the model payload before
        // execution. Until admission has produced those authoritative values,
        // a declaration cannot safely make a parallel decision from raw model
        // input, so such batches deliberately retain serial semantics.
        let canonical_parallel_safe =
            !matches!(
                self.policy.invalid_args,
                InvalidArgsPolicy::NormalizeThenReturnToolError
            ) && batch_is_canonical_parallel_safe(&self.tools, &tool_calls);
        if should_execute_tools_concurrently(
            tool_calls.len(),
            canonical_parallel_safe,
            self.middleware.tool_middleware_len(),
        ) {
            self.execute_tools_concurrently(state, ctx, run, status, messages, tool_calls)
                .await
        } else {
            self.execute_tools_serially(state, ctx, run, status, messages, tool_calls)
                .await
        }
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
        // The context's `LimitTracker` (synced with `RunPolicy::limits` at run
        // start) is the single enforced source of truth for the tool-call cap,
        // so the reported limit always matches the one that trips.
        if let Err(err) = ctx.record_tool_call() {
            ctx.emit(AgentEvent::LimitReached {
                kind: LimitKind::ToolCalls,
            });
            return Err(TinyAgentsError::LimitExceeded(err.to_string()));
        }

        // Discovery bridge, resolved before any hook runs. `tool_call` is
        // unwrapped here so every `before_tool` hook, allow-list, and the host
        // authorization gate below see the *real* tool name and arguments —
        // a deferred tool is admitted exactly as if the model had called it
        // directly. `tool_search` is answered from the run's catalogue without
        // running a tool. A host-registered tool under either name wins, and
        // a call the provider could not parse is left for the recovery below.
        if call.invalid.is_none()
            && self.tools.dispatch(&call.name).is_none()
            && let Some(answered) = self.answer_discovery_bridge(ctx, status, call)?
        {
            return Ok(answered);
        }
        // Preserve the exact attacker-controlled provider payload for host
        // authorization/audit, taken *after* discovery-bridge resolution:
        // `answer_discovery_bridge` rewrites `call.name`/`call.arguments` in
        // place when the call was a `tool_call` bridge wrapper (returning
        // `None` so admission continues with the unwrapped call), so the
        // snapshot here already reflects the real tool payload — not the
        // stale `{"name", "arguments"}` wrapper the model actually sent.
        // `call.arguments` is later canonicalized for execution and must not
        // overwrite what the gate evaluates.
        let model_arguments = call.arguments.clone();

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
            return Err(err);
        }

        // Before giving up on provider-unparseable arguments (below), try the
        // conservative, meaning-preserving repairs in `relaxed_json` (unquoted
        // keys, redundant wrapping braces, leaked chat-template quote tokens —
        // see that module's doc comment for the exact defects it targets).
        // This is the one place I-13 asked for it applied: admission was
        // short-circuiting straight to a tool error without ever trying the
        // repair the module exists for. On success the call proceeds through
        // normal (schema) validation below as if the provider had sent it
        // clean, rather than round-tripping a "fix your JSON" error the model
        // often cannot actually act on.
        if call.invalid.is_some()
            && let Some(raw) = call.arguments.as_str()
            && let Some(repaired) = crate::relaxed_json::recover_relaxed_object(raw)
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
        let allowed_tools = self.resolve_tool_allowlist(ctx)?;
        let is_allowed = allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.contains(&call.name));
        let (dispatch, tool) = match is_allowed
            .then(|| self.tools.model_dispatch(&call.name))
            .flatten()
        {
            Some(dispatch) => {
                let tool = dispatch.tool();
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
                        .filter(|_| {
                            allowed_tools
                                .as_ref()
                                .is_none_or(|allowed| allowed.contains(tool_name))
                        })
                        .map(|dispatch| (tool_name.clone(), dispatch)),
                    _ => None,
                };

                if let Some((tool_name, dispatch)) = rewrite_target {
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
                    // requested tool and the valid tools, then continue so
                    // the model can correct itself. This consumed one
                    // tool-call budget slot above, bounding the loop.
                    let valid = self
                        .tools
                        .model_callable_names()
                        .into_iter()
                        .filter(|name| {
                            allowed_tools
                                .as_ref()
                                .is_none_or(|allowed| allowed.contains(name))
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let args_repr = serde_json::to_string(&arguments)
                        .unwrap_or_else(|_| "<unserializable>".to_string());
                    let message = format!(
                        "unknown tool `{requested}` (arguments: {args_repr}); \
                         valid tools: [{valid}]"
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
            // validation detail and the tool's expected parameter schema, then
            // continue so the model can correct itself. This consumed one
            // tool-call budget slot above, bounding the loop.
            let call_id = CallId::new(call.id.clone());
            let detail = err.to_string();
            let schema_repr = serde_json::to_string(&schema.parameters)
                .unwrap_or_else(|_| "<unserializable>".to_string());
            let message = format!(
                "invalid arguments for tool `{}`: {detail}; expected schema: {schema_repr}",
                call.name
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
            let cancellation = ctx.cancellation.clone();
            let authorization = binding.host.security.authorize_tool(&request);
            let decision = match self.call_budget(ctx) {
                Some(remaining) => tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return Err(TinyAgentsError::Cancelled),
                    result = tokio::time::timeout(remaining, authorization) => result.map_err(|_| TinyAgentsError::Timeout(format!(
                        "tool authorization for run `{}` exceeded its remaining wall-clock budget",
                        ctx.run_id()
                    )))?,
                },
                None => tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return Err(TinyAgentsError::Cancelled),
                    result = authorization => result,
                },
            }?;
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
        Ok(ResolvedToolCall::Tool { dispatch, tool })
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
        let record = ctx.emit(AgentEvent::ToolStarted {
            call_id: call_id.clone(),
            tool_name: tool_name.clone(),
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
        // Snapshot the arguments for observability before `call` is moved
        // into execution, gated by the capture policy.
        let captured_input = self.policy.capture.tool_io.then(|| call.arguments.clone());
        PreparedToolCall {
            call_id,
            tool_name,
            options,
            captured_input,
            started_at_ms,
            executed,
            output_origin,
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
    ) -> Result<()> {
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
            let cancellation = ctx.cancellation.clone();
            let screening = binding
                .host
                .security
                .screen_input(&rendered, prepared.output_origin);
            let screened = match self.call_budget(ctx) {
                Some(remaining) => tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return Err(TinyAgentsError::Cancelled),
                    result = tokio::time::timeout(remaining, screening) => result.map_err(|_| TinyAgentsError::Timeout(format!(
                        "tool-output screening for run `{}` exceeded its remaining wall-clock budget", ctx.run_id()
                    )))?,
                },
                None => tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return Err(TinyAgentsError::Cancelled),
                    result = screening => result,
                },
            };
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

        run.tool_calls += 1;
        if prepared.executed {
            run.executed_tools.push(prepared.tool_name.clone());
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
            call_id: event_call_id,
            tool_name: event_tool_name,
            started_at_ms: Some(prepared.started_at_ms),
            input: prepared.captured_input,
            output: captured_output,
            duration_ms: Some(duration_ms),
            output_bytes: Some(output_bytes),
            error,
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

        messages.push(Message::Tool(tool_message_from_result(
            transcript_call_id,
            &result,
            prepared.options,
        )));
        Ok(())
    }

    /// Executes requested tools one at a time (the historical semantics; used
    /// for single-call turns and whenever tool-wrap middleware is registered).
    async fn execute_tools_serially(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<()> {
        for mut call in tool_calls {
            let dispatch = match self.admit_tool_call(state, ctx, status, &mut call).await? {
                ResolvedToolCall::Tool { dispatch, .. } => dispatch,
                ResolvedToolCall::Answered(result) => {
                    self.recover_tool_call(state, ctx, run, status, messages, &call, result)
                        .await?;
                    continue;
                }
            };

            let options = dispatch.call_options(&call.arguments);
            let prepared =
                self.start_tool_call(ctx, status, &call, options, true, dispatch.output_origin());

            // The real tool call is the innermost base of the tool-wrap
            // onion (same before -> wrap -> after ordering as the model
            // path): lifecycle `before_tool` ran in admission, the wrap onion
            // runs here, and lifecycle `after_tool` runs in the fold. The
            // crate-owned tool policy returns a recoverable tool error; the
            // outer run budget still aborts when the whole run is exhausted.
            let run_budget = self.call_budget(ctx);
            let base = ToolCallBase {
                dispatch,
                options,
                timeout_settings: self.tool_timeouts.clone(),
            };
            let run_id = ctx.run_id().as_str().to_string();
            let fut = self.middleware.run_wrapped_tool(ctx, state, call, &base);
            // TinyTools distinguishes a fatal execution `Err` from a
            // recoverable `ToolResult::error`; no harness error-policy facade
            // rewrites that canonical distinction.
            let guarded =
                futures::FutureExt::map(fut, |result| result.map(|wrapped| wrapped.into_result()));
            let outcome = Self::with_call_budget(
                run_budget,
                &run_id,
                "tool call",
                super::model_call::RUN_BOUND_LABEL,
                guarded,
            )
            .await;
            let result = match outcome {
                Ok(result) => result,
                Err(err) => {
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

            self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
                .await?;
        }
        Ok(())
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
    async fn recover_tool_call(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        call: &ToolCall,
        result: tinytools::ToolResult,
    ) -> Result<()> {
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

    /// Executes a multi-call turn concurrently (`join_all`), so turn latency
    /// is the slowest tool instead of the sum. Only reachable when no
    /// tool-wrap middleware is registered (see the module docs); execution
    /// therefore drives each tool directly — exactly what the empty wrap
    /// onion would have done — via a future that borrows no `RunContext`.
    async fn execute_tools_concurrently(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<()> {
        // Phase 1 — admission, serial, in call order. Nothing is announced and
        // nothing is queued here: an admission failure at call *k* must not
        // leave calls `0..k` with a `ToolStarted` they will never answer, nor
        // an `active_tool_calls` entry for work that is dropped unpolled
        // (TOOL-3). The serial path would have executed those calls; the
        // concurrent path now agrees with it by executing none of them.
        let mut admitted: Vec<AdmittedCall<State, Ctx>> = Vec::with_capacity(tool_calls.len());
        for mut call in tool_calls {
            match self.admit_tool_call(state, ctx, status, &mut call).await? {
                ResolvedToolCall::Tool { dispatch, tool } => admitted.push(AdmittedCall::Execute {
                    dispatch,
                    tool,
                    call,
                }),
                ResolvedToolCall::Answered(result) => {
                    admitted.push(AdmittedCall::Recovered { call, result })
                }
            }
        }

        // Phase 2 — announce and queue. Every admission succeeded, so every
        // `ToolStarted` emitted here is matched by a terminal event below.
        let mut slots: Vec<ToolSlot> = Vec::with_capacity(admitted.len());
        let mut prepared: Vec<PreparedToolCall> = Vec::new();
        let mut futures: Vec<_> = Vec::new();
        // Concurrent dispatch receives a shared parent snapshot; the mutable
        // run context stays with the fold phase after all futures complete.
        let parent_ctx: &RunContext<Ctx> = ctx;
        for entry in admitted {
            let (dispatch, tool, call) = match entry {
                AdmittedCall::Execute {
                    dispatch,
                    tool,
                    call,
                } => (dispatch, tool, call),
                AdmittedCall::Recovered { call, result } => {
                    slots.push(ToolSlot::Recovered { call, result });
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
            slots.push(ToolSlot::Execute);

            // Each call is bounded by its recoverable tool policy inside the
            // run's hard remaining wall-clock budget, mirroring serial mode.
            // The future owns everything it needs (tool Arc, call, a
            // non-generic `ToolExecutionContext` snapshot), so it does not
            // borrow the `RunContext` and can run alongside its siblings.
            let tool_timeout = self.resolved_tool_timeout(tool.as_ref(), &call);
            let timeout_result = timeout_result(&call, tool_timeout);
            let run_budget = self.call_budget(ctx);
            let run_id = ctx.run_id().as_str().to_string();
            futures.push(async move {
                let fut = async move {
                    dispatch
                        .execute(state, call.arguments, options, parent_ctx)
                        .await
                        .map_err(map_tool_dispatch_error)
                };
                let fut = Self::with_tool_policy_timeout(tool_timeout, timeout_result, fut);
                // As in serial mode, canonical execution errors remain fatal;
                // reported tool errors travel in `ToolResult::is_error`.
                Self::with_call_budget(
                    run_budget,
                    &run_id,
                    "tool call",
                    super::model_call::RUN_BOUND_LABEL,
                    fut,
                )
                .await
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
        let mut executed = prepared.into_iter().zip(results);
        for slot in slots {
            match slot {
                ToolSlot::Recovered { call, result } => {
                    self.recover_tool_call(state, ctx, run, status, messages, &call, result)
                        .await?;
                }
                ToolSlot::Execute => {
                    let (prepared, result) = executed
                        .next()
                        .expect("every Execute slot has a prepared/result pair");
                    let result = match result {
                        Ok(result) => result,
                        Err(err) => {
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
                            for (sibling_prepared, _) in executed {
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
                    self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
                        .await?;
                }
            }
        }
        Ok(())
    }
}

/// Decides whether a batch may leave the serial path.
///
/// Lifecycle middleware used to force serial execution unconditionally
/// (`lifecycle_middleware == 0`), but that precondition never actually
/// applied: lifecycle `before_tool` hooks that can rewrite a call's name or
/// arguments run during **admission** (`admit_tool_call`, phase 1 of
/// [`AgentHarness::execute_tools_concurrently`]), which is already serial and
/// completes in full — for every call in the batch — before any concurrent
/// future is built. By the time phase 3 runs the futures, every call has its
/// final, lifecycle-rewritten name and arguments; there is nothing left for a
/// lifecycle middleware to still mutate concurrently (I-8). Tool-*wrap*
/// middleware (`tool_wrap_middleware`) is a separate concern: the concurrent
/// path drives each tool directly, bypassing the wrap onion entirely (see
/// that method's docs), so a registered `ToolMiddleware` still forces serial
/// execution — dropping it silently would skip the middleware.
fn should_execute_tools_concurrently(
    calls: usize,
    canonical_parallel_safe: bool,
    tool_wrap_middleware: usize,
) -> bool {
    calls > 1 && canonical_parallel_safe && tool_wrap_middleware == 0
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
/// Schema-valid arguments are already canonical. A string containing valid
/// JSON is decoded, optionally through a markdown code fence, and the decoded
/// value is preserved for validation even when it remains invalid. Undecodable
/// or non-string values become an empty object only for object-capable schemas
/// that declare no required fields; required-field schemas retain the original
/// value so the validation error remains precise and model-visible.
fn normalize_tool_arguments(call: &mut ToolCall, schema: &ToolSchema) {
    // Never rewrite a value the declared schema already accepts. In
    // particular, an object-capable union may validly accept a primitive too.
    if schema.validate_call(call).is_ok() {
        return;
    }

    let parameters = &schema.parameters;
    let accepts_object = parameters.get("type").is_some_and(|kind| {
        kind.as_str() == Some("object")
            || kind
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("object")))
    }) || parameters.get("properties").is_some()
        || parameters.get("required").is_some()
        || parameters
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(Value::is_object));
    if !accepts_object {
        return;
    }

    if let Some(raw) = call.arguments.as_str() {
        let candidate = strip_markdown_code_fence(raw);
        if let Ok(value) = serde_json::from_str::<Value>(candidate) {
            let mut normalized = call.clone();
            normalized.arguments = value;
            // Decoding must be lossless even when the decoded value is still
            // schema-invalid. Preserve it so the validation below reports the
            // actual bad field/type instead of silently replacing it with `{}`.
            call.arguments = normalized.arguments;
            return;
        }
    }

    // A provider-native object is already the shape normalization is trying to
    // recover. If its contents violate the schema, preserve them so the model
    // sees the real validation error instead of executing with an empty object.
    if call.arguments.is_object() {
        unwrap_wrapped_arguments(call, schema);
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

/// Keys under which a model commonly buries the real arguments object.
///
/// `properties` is the JSON-Schema echo; the rest are the wrapper names small
/// models invent when they confuse the *call* envelope with its payload. All of
/// them were observed on local runtimes — see [`unwrap_wrapped_arguments`].
const ARGUMENT_WRAPPER_KEYS: [&str; 7] = [
    "properties",
    "arguments",
    "args",
    "parameters",
    "params",
    "param",
    "input",
];

/// Recovers arguments a model buried one level deep inside an envelope.
///
/// Small local models (observed on `llama3.2:3b` via Ollama) routinely send
/// something other than a bare arguments object. All three of these are real
/// captures for a tool declaring one required `city` string:
///
/// ```text
/// {"type":"object","required":["city"],"properties":{"city":"Paris"}}
/// {"properties":{...},"required":[...],"arguments":{"city":"Paris"}}
/// {"param":{"city":"Paris"}}
/// ```
///
/// In each case the intended `{"city":"Paris"}` is present, one level down.
/// Without this the call fails validation, costs a repair round trip, and on
/// the default [`InvalidArgsPolicy::Fail`] aborts the run outright.
///
/// The rewrite is deliberately conservative and cannot corrupt a legitimate
/// call. For each candidate key it applies only when the outer object is
/// already schema-invalid, when the tool does not itself declare an argument of
/// that name (so the key is not meaningfully the model's own data), and when
/// the unwrapped value *does* validate. If no candidate satisfies all three the
/// original arguments are left untouched, so the model still sees a precise
/// validation error rather than a rewritten one.
///
/// [`InvalidArgsPolicy::Fail`]: crate::runtime::InvalidArgsPolicy::Fail
fn unwrap_wrapped_arguments(call: &mut ToolCall, schema: &ToolSchema) {
    let declared = schema
        .parameters
        .get("properties")
        .and_then(Value::as_object);

    for key in ARGUMENT_WRAPPER_KEYS {
        // A tool that genuinely takes an argument of this name must never have
        // it unwrapped — for such a tool the key is data, not an envelope.
        if declared.is_some_and(|declared| declared.contains_key(key)) {
            continue;
        }
        let Some(inner) = call
            .arguments
            .get(key)
            .filter(|inner| inner.is_object())
            .cloned()
        else {
            continue;
        };

        let mut candidate = call.clone();
        candidate.arguments = inner;
        if schema.validate_call(&candidate).is_ok() {
            call.arguments = candidate.arguments;
            return;
        }
    }
}

fn strip_markdown_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let body = match after_open.find('\n') {
        Some(newline)
            if after_open[..newline]
                .chars()
                .all(|character| character.is_ascii_alphanumeric()) =>
        {
            &after_open[newline + 1..]
        }
        _ => after_open,
    };
    body.trim().strip_suffix("```").unwrap_or(body).trim()
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
mod canonical_result_tests {
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::{
        batch_is_canonical_parallel_safe, map_tool_dispatch_error,
        should_execute_tools_concurrently, tool_message_from_result,
    };
    use crate::error::TinyAgentsError;
    use tinyinference_llm::message::ContentBlock;
    use tinytools::{ToolCallOptions, ToolContent, ToolResult};

    struct DeclaredParallelTool {
        parallel: bool,
        injected_risk: bool,
    }

    #[async_trait]
    impl tinytools::Tool for DeclaredParallelTool {
        fn name(&self) -> &str {
            "parallel"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult::success("ok"))
        }

        fn is_concurrency_safe(&self, arguments: &serde_json::Value) -> bool {
            self.parallel && arguments["risk"].as_str().is_none_or(|risk| risk == "safe")
        }

        fn injected_arguments(&self) -> Vec<tinytools::ToolInjectedArgument> {
            if self.injected_risk {
                vec![tinytools::ToolInjectedArgument::host("risk")]
            } else {
                Vec::new()
            }
        }
    }

    fn markdown_result() -> ToolResult {
        ToolResult {
            content: vec![
                ToolContent::Text {
                    text: "plain summary".to_string(),
                },
                ToolContent::Json {
                    data: serde_json::json!({"ordered": 2}),
                },
            ],
            is_error: true,
            markdown_formatted: Some("## compact failure".to_string()),
        }
    }

    #[test]
    fn serial_and_concurrent_folds_select_the_same_markdown_and_preserve_blocks() {
        let result = markdown_result();
        let options = ToolCallOptions::prefer_markdown();

        // Both execution modes converge through this fold helper.
        let serial = tool_message_from_result("serial-call".to_string(), &result, options);
        let concurrent = tool_message_from_result("concurrent-call".to_string(), &result, options);

        for message in [&serial, &concurrent] {
            assert_eq!(
                message.content,
                vec![ContentBlock::Text("## compact failure".to_string())]
            );
            assert_eq!(message.artifact.as_ref().unwrap()["is_error"], true);
            assert!(!message.trusted_verbatim);
            assert_eq!(
                message.artifact.as_ref().unwrap()["trusted_verbatim"],
                false,
                "a tool-controlled canonical result cannot opt out of host framing"
            );
            assert_eq!(
                message.artifact.as_ref().unwrap()["tinytools_content"],
                serde_json::json!([
                    {"type":"text", "text":"plain summary"},
                    {"type":"json", "data":{"ordered":2}}
                ])
            );
        }
    }

    #[test]
    fn ordinary_result_keeps_ordered_blocks_when_markdown_is_not_preferred() {
        let message = tool_message_from_result(
            "call".to_string(),
            &markdown_result(),
            ToolCallOptions::default(),
        );
        assert_eq!(
            message.content,
            vec![
                ContentBlock::Text("plain summary".to_string()),
                ContentBlock::Json(serde_json::json!({"ordered": 2})),
            ]
        );
    }

    #[test]
    fn dispatch_error_mapping_preserves_harness_classification_without_leaking_foreign_detail() {
        let cancelled = map_tool_dispatch_error(anyhow::Error::new(TinyAgentsError::Cancelled));
        assert!(matches!(cancelled, TinyAgentsError::Cancelled));

        let foreign = map_tool_dispatch_error(anyhow::anyhow!("outer: {}", "root cause"));
        assert!(
            matches!(foreign, TinyAgentsError::Tool(message) if message == "tool dispatch failed")
        );
    }

    #[test]
    fn canonical_concurrency_declaration_gates_the_parallel_path() {
        let call =
            tinyinference_llm::tool::ToolCall::new("call", "parallel", serde_json::json!({}));

        let mut serial: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
        serial.register(Arc::new(DeclaredParallelTool {
            parallel: false,
            injected_risk: false,
        }));
        assert!(!batch_is_canonical_parallel_safe(
            &serial,
            std::slice::from_ref(&call)
        ));

        let mut concurrent: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
        concurrent.register(Arc::new(DeclaredParallelTool {
            parallel: true,
            injected_risk: false,
        }));
        assert!(batch_is_canonical_parallel_safe(&concurrent, &[call]));
    }

    #[test]
    fn forged_safe_injected_value_cannot_select_parallel_execution() {
        let mut registry: crate::tool::ToolRegistry<(), ()> = crate::tool::ToolRegistry::new();
        registry.register(Arc::new(DeclaredParallelTool {
            parallel: true,
            injected_risk: true,
        }));

        // A model can claim `safe`; admission will strip this and inject the
        // host's real (potentially unsafe) value. The raw value must therefore
        // never be considered a parallelization proof.
        let forged_safe = tinyinference_llm::tool::ToolCall::new(
            "call",
            "parallel",
            serde_json::json!({"risk": "safe"}),
        );
        let canonical = tinytools::ToolCall::new(
            tinytools::ToolCallId::new("call"),
            "parallel",
            forged_safe.arguments.clone(),
        );
        let mut host_values = tinytools::InjectedToolArguments::new();
        host_values.insert("risk", serde_json::json!("unsafe"));
        let authoritative = tinytools::prepare_tool_arguments(
            &canonical,
            &registry.get("parallel").unwrap().injected_arguments(),
            &host_values,
        )
        .unwrap();
        assert!(
            !registry
                .get("parallel")
                .unwrap()
                .is_concurrency_safe(&authoritative),
            "the host value is the one that makes this call unsafe"
        );
        assert!(!batch_is_canonical_parallel_safe(&registry, &[forged_safe]));
    }

    #[test]
    fn lifecycle_middleware_no_longer_forces_the_serial_route() {
        // Regression test (I-8): lifecycle middleware used to force the
        // serial path unconditionally, on the theory that `before_tool` can
        // rewrite a call (`&mut ToolCall`) while execution is concurrently in
        // flight. That never actually applied: admission (including every
        // `before_tool` hook) is serial and completes in full, for every call
        // in the batch, before any concurrent future is built — so a
        // lifecycle middleware has nothing left to mutate once execution
        // starts. Only tool-*wrap* middleware (bypassed entirely by the
        // concurrent path) still forces serial execution.
        assert!(should_execute_tools_concurrently(2, true, 0));
    }

    #[test]
    fn tool_wrap_middleware_still_forces_the_serial_route() {
        // The concurrent path drives each tool directly, skipping the
        // tool-wrap onion; a registered `ToolMiddleware` must still force
        // serial execution or it would silently never run.
        assert!(!should_execute_tools_concurrently(2, true, 1));
    }

    #[test]
    fn map_tool_dispatch_error_preserves_sub_agent_depth_and_limit_exceeded() {
        // M-3 regression: every non-cancel/timeout error used to collapse to
        // a generic `Tool("tool dispatch failed")`, which `is_retryable`
        // treats as unconditionally retryable. A `SubAgentDepth`/
        // `LimitExceeded` escaping a nested sub-agent tool call is
        // deterministic and will never succeed on retry, so it must keep its
        // own classification instead of masquerading as a retryable tool
        // error.
        let depth_err = anyhow::Error::from(TinyAgentsError::SubAgentDepth(4));
        assert!(matches!(
            map_tool_dispatch_error(depth_err),
            TinyAgentsError::SubAgentDepth(4)
        ));

        let limit_err = anyhow::Error::from(TinyAgentsError::LimitExceeded(
            "some sensitive detail".to_string(),
        ));
        match map_tool_dispatch_error(limit_err) {
            TinyAgentsError::LimitExceeded(message) => {
                assert!(
                    !message.contains("sensitive"),
                    "the original message must still be redacted: {message}"
                );
            }
            other => panic!("expected LimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn map_tool_dispatch_error_still_redacts_a_genuine_tool_error() {
        // An ordinary tool-authored error (arbitrary text, possibly carrying
        // secrets or user data) must still be collapsed to a generic message,
        // unlike the structural errors above.
        let tool_err = anyhow::Error::from(TinyAgentsError::Model(
            "leaked api key sk-secret".to_string(),
        ));
        match map_tool_dispatch_error(tool_err) {
            TinyAgentsError::Tool(message) => {
                assert!(!message.contains("sk-secret"));
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }
}
