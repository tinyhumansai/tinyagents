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
//!    unknown-tool policy resolution, schema validation, and the
//!    [`AgentEvent::ToolStarted`] emission.
//! 2. **Execution**: when the turn requests **two or more** tools and **no
//!    tool-wrap middleware** ([`crate::harness::middleware::ToolMiddleware`])
//!    is registered, the admitted calls run **concurrently**
//!    (`join_all`), so turn latency is the slowest tool instead of the sum.
//!    Otherwise execution is serial, preserving the historical semantics.
//! 3. **Fold** (always serial, in original call order): the lifecycle
//!    `after_tool` hooks, accounting, the [`AgentEvent::ToolCompleted`]
//!    emission, and the transcript append. Results are attached to their
//!    original `tool_call_id` in the calls' original order regardless of
//!    completion order.
//!
//! ## Why tool-wrap middleware forces serial execution
//!
//! [`crate::harness::middleware::ToolMiddleware::wrap_tool`] holds
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
//! - **Errors**: the first failing call *in original call order* fails the
//!   turn. Difference: in serial mode later calls never start after a
//!   failure; in concurrent mode they were already in flight and run to
//!   completion (their results are discarded). Tools that must not observe a
//!   sibling's failure should be run under a tool-wrap middleware (serial) or
//!   a harness without parallel-capable turns.

use super::model_call::ToolCallBase;
use super::*;
use crate::harness::tool::ToolExecutionContext;

/// How a single requested tool call was resolved during admission.
enum ResolvedToolCall<State: Send + Sync> {
    /// A registered tool (possibly after an unknown-tool rewrite).
    Tool(Arc<dyn Tool<State>>),
    /// Unknown-tool recovery: no tool runs; this tool-error message is
    /// appended to the transcript at the call's original position.
    ErrorMessage(String),
}

/// One transcript slot per requested call, in original order, used by the
/// concurrent path to reassemble results deterministically.
enum ToolSlot {
    /// An executed call: consumes the next prepared/future pair in order.
    Execute,
    /// An unknown-tool recovery message, appended verbatim.
    Immediate { call_id: String, message: String },
}

/// Admission metadata for one executable call, paired 1:1 (in order) with its
/// execution future/result on the concurrent path.
struct PreparedToolCall {
    call_id: CallId,
    tool_name: String,
    captured_input: Option<Value>,
    started_at_ms: u64,
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
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
        if tool_calls.len() > 1 && self.middleware.tool_middleware_len() == 0 {
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
    ) -> Result<ResolvedToolCall<State>> {
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

        self.middleware.run_before_tool(ctx, state, call).await?;

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
            return Ok(ResolvedToolCall::ErrorMessage(detail));
        }

        let tool = match self.tools.get(&call.name) {
            Some(tool) => tool,
            None => {
                // The model called an unregistered tool. Apply the run's
                // `UnknownToolPolicy` instead of unconditionally aborting.
                let requested = call.name.clone();
                let arguments = call.arguments.clone();
                let call_id = CallId::new(call.id.clone());

                // Rewrite mode: retarget to a fixed compatibility tool if
                // that tool exists, otherwise fall through to recovery.
                let rewrite_target = match &self.policy.unknown_tool {
                    UnknownToolPolicy::Rewrite { tool_name } => {
                        self.tools.get(tool_name).map(|t| (tool_name.clone(), t))
                    }
                    _ => None,
                };

                if let Some((tool_name, tool)) = rewrite_target {
                    call.name = tool_name.clone();
                    let record = ctx.emit(AgentEvent::UnknownToolCall {
                        call_id,
                        requested_name: requested,
                        arguments,
                        recovery: format!("rewrite:{tool_name}"),
                    });
                    status.set_last_event(record.id);
                    tool
                } else if matches!(self.policy.unknown_tool, UnknownToolPolicy::Fail) {
                    return Err(TinyAgentsError::ToolNotFound(requested));
                } else {
                    // `ReturnToolError` (or a Rewrite whose target is also
                    // missing): inject a tool-error result naming the
                    // requested tool and the valid tools, then continue so
                    // the model can correct itself. This consumed one
                    // tool-call budget slot above, bounding the loop.
                    let valid = self.tools.names().join(", ");
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
                    return Ok(ResolvedToolCall::ErrorMessage(message));
                }
            }
        };
        let schema = tool.schema();
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
                return Err(err);
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
            return Ok(ResolvedToolCall::ErrorMessage(message));
        }
        Ok(ResolvedToolCall::Tool(tool))
    }

    /// Marks one call as started: status bookkeeping, the `ToolStarted`
    /// emission, and the capture-policy input snapshot. Returns the admission
    /// metadata consumed by the fold phase.
    fn start_tool_call(
        &self,
        ctx: &RunContext<Ctx>,
        status: &mut HarnessRunStatus,
        call: &ToolCall,
    ) -> PreparedToolCall {
        let call_id = CallId::new(call.id.clone());
        let tool_name = call.name.clone();
        status.active_tool_calls.push(call_id.clone());
        // Captured here (where the call actually starts) so the completed
        // event carries a real start time for duration-aware exporters.
        let started_at_ms = crate::harness::ids::now_ms();
        let record = ctx.emit(AgentEvent::ToolStarted {
            call_id: call_id.clone(),
            tool_name: tool_name.clone(),
        });
        status.set_last_event(record.id);
        // Snapshot the arguments for observability before `call` is moved
        // into execution, gated by the capture policy.
        let captured_input = self.policy.capture.tool_io.then(|| call.arguments.clone());
        PreparedToolCall {
            call_id,
            tool_name,
            captured_input,
            started_at_ms,
        }
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
        mut result: crate::harness::tool::ToolResult,
    ) -> Result<()> {
        self.middleware
            .run_after_tool(ctx, state, &mut result)
            .await?;

        run.tool_calls += 1;
        status.tool_calls = run.tool_calls;
        status.active_tool_calls.retain(|c| c != &prepared.call_id);
        let captured_output = self
            .policy
            .capture
            .tool_io
            .then(|| Value::String(result.content.clone()));
        // Outcome fields carried on the event itself (not a side-channel) so
        // journal-backed exporters render duration/size/success without the
        // live run's state. Duration is wall-clock (completion minus start);
        // `error` mirrors `ToolResult::error` (`None` == success).
        let duration_ms = crate::harness::ids::now_ms().saturating_sub(prepared.started_at_ms);
        let output_bytes = result.content.len() as u64;
        let error = result.error.clone();
        let record = ctx.emit(AgentEvent::ToolCompleted {
            call_id: prepared.call_id,
            tool_name: prepared.tool_name,
            started_at_ms: Some(prepared.started_at_ms),
            input: prepared.captured_input,
            output: captured_output,
            duration_ms: Some(duration_ms),
            output_bytes: Some(output_bytes),
            error,
        });
        status.set_last_event(record.id);

        messages.push(Message::tool(
            result.call_id.clone(),
            result.content.clone(),
        ));
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
            let tool = match self.admit_tool_call(state, ctx, status, &mut call).await? {
                ResolvedToolCall::Tool(tool) => tool,
                ResolvedToolCall::ErrorMessage(message) => {
                    run.tool_calls += 1;
                    status.tool_calls = run.tool_calls;
                    messages.push(Message::tool(call.id.clone(), message));
                    continue;
                }
            };

            let prepared = self.start_tool_call(ctx, status, &call);

            // The real tool call is the innermost base of the tool-wrap
            // onion (same before -> wrap -> after ordering as the model
            // path): lifecycle `before_tool` ran in admission, the wrap onion
            // runs here, and lifecycle `after_tool` runs in the fold. Bounded
            // by the same remaining wall-clock budget as a model call, so a
            // hanging tool cannot block the run past its deadline either.
            let base = ToolCallBase { tool };
            let remaining = self.call_budget(ctx);
            let run_id = ctx.run_id().as_str().to_string();
            let fut = self.middleware.run_wrapped_tool(ctx, state, call, &base);
            let result = Self::with_call_budget(remaining, &run_id, "tool call", fut)
                .await?
                .into_result();

            self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
                .await?;
        }
        Ok(())
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
        // Phase 1 — admission, serial, in call order.
        let mut slots: Vec<ToolSlot> = Vec::with_capacity(tool_calls.len());
        let mut prepared: Vec<PreparedToolCall> = Vec::new();
        let mut futures: Vec<_> = Vec::new();
        for mut call in tool_calls {
            let tool = match self.admit_tool_call(state, ctx, status, &mut call).await? {
                ResolvedToolCall::Tool(tool) => tool,
                ResolvedToolCall::ErrorMessage(message) => {
                    run.tool_calls += 1;
                    status.tool_calls = run.tool_calls;
                    slots.push(ToolSlot::Immediate {
                        call_id: call.id.clone(),
                        message,
                    });
                    continue;
                }
            };

            prepared.push(self.start_tool_call(ctx, status, &call));
            slots.push(ToolSlot::Execute);

            // Each call is individually bounded by the run's remaining
            // wall-clock budget *at admission*, mirroring the serial path.
            // The future owns everything it needs (tool Arc, call, a
            // non-generic `ToolExecutionContext` snapshot), so it does not
            // borrow the `RunContext` and can run alongside its siblings.
            let remaining = self.call_budget(ctx);
            let run_id = ctx.run_id().as_str().to_string();
            let exec_ctx = ToolExecutionContext::from_run_context(ctx);
            futures.push(async move {
                let fut = tool.call_with_context(state, call, exec_ctx);
                Self::with_call_budget(remaining, &run_id, "tool call", fut).await
            });
        }

        // Phase 2 — run all admitted calls concurrently. `join_all` preserves
        // input order, so results pair 1:1 with `prepared`.
        let results = futures::future::join_all(futures).await;

        // Phase 3 — fold in original call order: the first failing call (in
        // that order) fails the turn; siblings already ran to completion.
        let mut executed = prepared.into_iter().zip(results);
        for slot in slots {
            match slot {
                ToolSlot::Immediate { call_id, message } => {
                    messages.push(Message::tool(call_id, message));
                }
                ToolSlot::Execute => {
                    let (prepared, result) = executed
                        .next()
                        .expect("every Execute slot has a prepared/result pair");
                    let result = result?;
                    self.finish_tool_call(state, ctx, run, status, messages, prepared, result)
                        .await?;
                }
            }
        }
        Ok(())
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
