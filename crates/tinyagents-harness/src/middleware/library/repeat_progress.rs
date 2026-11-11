//! [`RepeatProgressMiddleware`]: host adapter for the crate successful-repeat
//! tracker — halts identical-output / identical-call loops that succeed but
//! make no progress (#4088 / #4095), including loops whose repeats are not
//! back to back (#6275).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::{Result as TaResult, TinyAgentsError};
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::no_progress::{
    CallGate, OutcomeFingerprinter, RepeatProgressConfig, SuccessfulRepeat, VolatileSpanNormalizer,
    fingerprint_arguments,
};
use crate::steering::{SteeringCommand, SteeringHandle};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult as TaToolResult;

use super::repeat_progress_state::{
    PendingCallBatch, REPEAT_GUARD_BLOCKED, REPEAT_GUARD_HALTED, RepeatState, append_note,
    assistant_visible_text, guard_metadata, lock, visible_tool_results,
};
use super::wrap_up::DEFAULT_CLEARED_PLACEHOLDER;

/// Shared slot a guard writes its root-cause halt summary into when it trips, so
/// the turn can surface the cause instead of an empty or last-model reply.
///
/// Several guards can share one slot, and a halt only pauses the run at the
/// top of the next iteration, so the rest of the batch still reaches every
/// guard. The first guard to stop the run is its cause: a guard must not
/// replace a summary that is already there.
pub type HaltSummarySlot = Arc<Mutex<Option<String>>>;

/// Whether a tool is contractually re-invoked with identical arguments (a
/// polling/wait tool), so an identical repeat is progress rather than a loop.
pub type RepeatExemption = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Whether a tool cannot change state (a pure read). Hosts supply it from the
/// tools' declared policy (`ToolPolicy::read_only`); see
/// [`RepeatProgressMiddleware::with_read_only`].
pub type ReadOnlyCheck = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Host adapter for the crate's successful-repeat tracker (#4088 / #4095).
/// [`SuccessfulRepeatTracker`](crate::no_progress::SuccessfulRepeatTracker) owns the generic streak accounting; this adapter
/// builds canonical tool signatures, applies the host's polling-tool
/// exemption ([`RepeatExemption`]), and maps a crate halt verdict into the shared halt summary and
/// steering pause:
///
/// - **Repeat-output** (`after_model`, checked before the tools run): halts when
///   the assistant's visible text + tool-call `(name, args)` batch is byte
///   identical [`DEFAULT_REPEAT_OUTPUT_THRESHOLD`] iterations in a row.
/// - **Repeat-call** (evaluated once the batch's tool results are all back, gated
///   on every call succeeding): halts when the `(tool, args)` batch alone repeats
///   [`DEFAULT_REPEAT_CALL_THRESHOLD`] times — catching successful no-op loops
///   that vary only their narration.
/// - **Recurrence** (each successful `after_tool`, #6275): halts when one call
///   returns the identical result [`DEFAULT_REPEAT_CALL_THRESHOLD`] times in the
///   run, adjacent or not — catching a model cycling A, B, A, B through steps
///   whose results it already has. The result is the content after the earlier
///   `after_tool` layers (caps, summarizer), i.e. what the model saw, so a
///   re-read whose output changed does not count. A result that compaction
///   later evicts stops counting; see [`RepeatEvictionObserver`].
///
/// Escalation is **staged** by default (see [`RepeatProgressConfig`]): each
/// first threshold only *warns*, appending a `[repeat notice]` to the tool
/// result; an identical call that keeps repeating is then **blocked** in
/// `before_tool` (answered with an error result, never executed); a second
/// block *of the same call* halts as described below. Ping-pong and
/// argument-churn patterns only warn, as does a repeat of calls that were
/// already repeating right before a context compaction. At most one warning
/// lands on a result; the rest wait for the next one.
/// [`RepeatProgressConfig::immediate_halt`] restores halting at the first
/// threshold.
///
/// A block is a *prediction* that the call would return what it returned last
/// time. Without [`with_read_only`](Self::with_read_only) every tool is treated
/// as possibly state-changing, so any other successful call discards the
/// prediction and a repeat only blocks while it is the most recent call
/// (A, A, A, ...). With it, reads do not discard predictions, so A, B, A, B
/// cycles of reads block too. The results the guard answers itself carry
/// `REPEAT_GUARD_METADATA_KEY` in their metadata so a host can keep them out
/// of failure accounting.
///
/// Polling/wait tools (per the [`RepeatExemption`]) are exempt from all three:
/// their contract is to be re-invoked identically, so an all-poll batch resets
/// the streaks instead of recording. On a trip it writes the legacy root-cause
/// summary into the shared [`HaltSummarySlot`] and pauses
/// the run through the shared steering handle — the same halt mechanism as the
/// repeated-failure breaker.
///
/// [`DEFAULT_REPEAT_OUTPUT_THRESHOLD`]: crate::no_progress::DEFAULT_REPEAT_OUTPUT_THRESHOLD
/// [`DEFAULT_REPEAT_CALL_THRESHOLD`]: crate::no_progress::DEFAULT_REPEAT_CALL_THRESHOLD
pub struct RepeatProgressMiddleware {
    handle: SteeringHandle,
    halt_summary: HaltSummarySlot,
    exempt: RepeatExemption,
    /// Tools known not to change state; see [`Self::with_read_only`].
    read_only: ReadOnlyCheck,
    pub(super) state: Arc<RepeatState>,
    /// Reduces a tool result to the identity the recurrence ledger keys on.
    fingerprinter: Arc<dyn OutcomeFingerprinter>,
    /// Batch bookkeeping bridging `after_model` → `after_tool` for the call guard.
    pending: Mutex<HashMap<u64, PendingCallBatch>>,
}

impl RepeatProgressMiddleware {
    /// Build the guard. `exempt` names the polling/wait tools that are exempt
    /// from all three checks; the cleared-result placeholder defaults to
    /// [`DEFAULT_CLEARED_PLACEHOLDER`].
    pub fn new(
        handle: SteeringHandle,
        halt_summary: HaltSummarySlot,
        exempt: RepeatExemption,
    ) -> Self {
        Self {
            handle,
            halt_summary,
            exempt,
            read_only: Arc::new(|_| false),
            state: Arc::new(RepeatState::new(
                DEFAULT_CLEARED_PLACEHOLDER,
                RepeatProgressConfig::default(),
            )),
            fingerprinter: Arc::new(VolatileSpanNormalizer),
            pending: Mutex::default(),
        }
    }

    /// Replaces the fingerprinter the recurrence ledger uses to compare tool
    /// results. The default ignores volatile spans (timestamps, durations,
    /// request ids), so a result that differs only by those still counts as
    /// the same result.
    pub fn with_fingerprinter(mut self, fingerprinter: Arc<dyn OutcomeFingerprinter>) -> Self {
        self.fingerprinter = fingerprinter;
        self
    }

    /// Override the placeholder body treated as an evicted tool result. Must be
    /// called before [`eviction_observer`](Self::eviction_observer).
    pub fn with_cleared_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        let config = self.state.config.clone();
        self.state = Arc::new(RepeatState::new(placeholder, config));
        self
    }

    /// Names the tools that cannot change state (pure reads), typically from
    /// their declared `ToolPolicy::read_only`. A successful call to any other
    /// tool discards the guard's prediction of what *other* calls would return,
    /// so a read repeated after an edit is never blocked. Defaults to "no tool
    /// is read-only", the conservative reading.
    pub fn with_read_only(mut self, read_only: ReadOnlyCheck) -> Self {
        self.read_only = read_only;
        self
    }

    /// Replaces the thresholds and escalation settings (see
    /// [`RepeatProgressConfig`]). The default stages escalation (warn, block,
    /// halt); pass [`RepeatProgressConfig::immediate_halt`] for the historical
    /// halt at the first threshold. Must be called before
    /// [`eviction_observer`](Self::eviction_observer).
    pub fn with_config(mut self, config: RepeatProgressConfig) -> Self {
        let placeholder = self.state.cleared_placeholder.clone();
        self.state = Arc::new(RepeatState::new(placeholder, config));
        self
    }

    /// The companion that resets the recurrence ledger when a recorded result is
    /// evicted from context. It must be registered after every reduction
    /// middleware, and this guard before them.
    pub fn eviction_observer(&self) -> RepeatEvictionObserver {
        RepeatEvictionObserver {
            state: Arc::clone(&self.state),
        }
    }

    /// Latch a root-cause halt: record the summary the turn surfaces instead of an
    /// empty/last-model reply, and pause at the top of the next iteration (before
    /// the next model call), matching the repeated-failure breaker's halt path.
    ///
    /// When another guard sharing the slot already stopped the run earlier in
    /// the batch, its summary is kept: replacing it named the last guard to
    /// trip as the cause instead of the first. The pause is sent either way.
    /// Halt with a classifier-owned cause so terminal summaries can use typed
    /// class and operation identifiers without parsing this note.
    pub fn halt_classified<C>(
        &self,
        ctx: &mut RunContext<C>,
        cause: crate::no_progress::ClassifiedStopCause,
    ) {
        self.halt_inner(ctx, cause.summary.clone(), Some(cause));
    }

    fn halt_inner<C>(
        &self,
        ctx: &mut RunContext<C>,
        summary: String,
        cause: Option<crate::no_progress::ClassifiedStopCause>,
    ) {
        let first = {
            let mut slot = lock(&self.halt_summary);
            if slot.is_some() {
                tracing::debug!(
                    "[tinyagents::mw] run already stopped by another guard; keeping its halt summary"
                );
            }
            slot.get_or_insert(summary).clone()
        };
        // Mark the run so the loop reports `TerminalReason::Halted` for the
        // pause this causes, not a plain steering pause, with the same first
        // cause the slot keeps.
        if ctx.halted_by_guard.is_none() {
            ctx.halted_by_guard = Some(first.clone());
            ctx.halted_by_guard_cause = cause.filter(|cause| cause.summary == first);
        }
        ctx.guard_pauses += 1;
        self.handle.send(SteeringCommand::Pause);
    }
}

fn repeat_stop_cause(summary: String, operation: &str) -> crate::no_progress::ClassifiedStopCause {
    crate::no_progress::ClassifiedStopCause {
        class: "repeating_step".into(),
        operation: operation.to_owned(),
        summary,
    }
}

#[async_trait]
impl<C: Send + Sync> Middleware<(), C> for RepeatProgressMiddleware {
    fn name(&self) -> &str {
        "repeat_progress"
    }

    async fn after_agent(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        _run: &mut crate::middleware::AgentRun,
    ) -> TaResult<()> {
        let run_id = ctx.instance_id();
        self.state.forget_run(run_id);
        lock(&self.pending).remove(&run_id);
        Ok(())
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> TaResult<()> {
        // Registered ahead of the reduction steps, so this is the request as the
        // loop built it. The observer compares the same ids after they ran.
        let run_id = ctx.instance_id();
        let visible = {
            let recorded = lock(&self.state.recorded);
            match recorded.get(&run_id) {
                Some(ids) if !ids.is_empty() => {
                    visible_tool_results(request, ids, &self.state.cleared_placeholder)
                }
                _ => HashSet::new(),
            }
        };
        lock(&self.state.visible_before_reduction).insert(run_id, visible);
        Ok(())
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        response: &mut ModelResponse,
    ) -> TaResult<()> {
        let tool_calls = &response.message.tool_calls;
        if tool_calls.is_empty() {
            // A final answer (no tool calls) ends the loop; nothing to guard, and
            // there is no batch to track for the call guard.
            lock(&self.pending).remove(&ctx.instance_id());
            return Ok(());
        }

        // Polling/wait tools are contractually re-invoked with identical args +
        // narration each timeout while the work is still running, so an all-poll
        // batch is legitimate progress, not a no-progress repeat.
        let all_exempt = tool_calls.iter().all(|c| (self.exempt)(&c.name));

        // Canonical `(tool, args)` batch signature (call guard) and the broader
        // narration+call signature (output guard). Both fold each call in order
        // with a `\u{1}` separator, matching the legacy signatures.
        let mut call_sig = String::new();
        for call in tool_calls {
            call_sig.push('\u{1}');
            call_sig.push_str(&call.name);
            call_sig.push('\u{1}');
            call_sig.push_str(&call.arguments.to_string());
        }
        let output_sig = format!(
            "{}{}",
            assistant_visible_text(&response.message).trim(),
            call_sig
        );
        // Per-call signatures for the recurrence ledger. The invocation
        // identity joins each post-tool result to the provider call id.
        let call_sigs = tool_calls
            .iter()
            .filter(|call| !(self.exempt)(&call.name))
            .map(|call| {
                (
                    call.id.clone(),
                    (call.name.clone(), fingerprint_arguments(&call.arguments)),
                )
            })
            .fold(
                HashMap::<String, VecDeque<(String, String)>>::new(),
                |mut calls, (id, sig)| {
                    calls.entry(id).or_default().push_back(sig);
                    calls
                },
            );

        // Stage output with the crate tracker. Its halt verdict is intentionally
        // deferred until the matching tool batch is confirmed successful.
        self.state.with_monitor(ctx.instance_id(), |monitor| {
            monitor.record_output(&output_sig, all_exempt)
        });

        // Stage the batch for the repeat-CALL guard, evaluated once every result
        // is back (gated on success) in `after_tool`.
        {
            let mut pending = lock(&self.pending);
            pending.insert(
                ctx.instance_id(),
                PendingCallBatch {
                    call_sig,
                    remaining: tool_calls.len(),
                    all_ok: true,
                    exempt: all_exempt,
                    call_sigs,
                    halted: false,
                },
            );
        }
        Ok(())
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        call: &mut ToolCall,
    ) -> TaResult<()> {
        if (self.exempt)(&call.name) {
            return Ok(());
        }
        let arguments = fingerprint_arguments(&call.arguments);
        let run_id = ctx.instance_id();
        let gate = self
            .state
            .with_monitor(run_id, |monitor| monitor.pre_call(&call.name, &arguments));
        match gate {
            CallGate::Allow => Ok(()),
            // Refusing admission answers the call with this text as an error
            // result and never runs the tool (see `agent_loop::tools`).
            CallGate::Block(text) => {
                tracing::warn!(
                    tool = call.name,
                    "[tinyagents::mw] repeat-progress blocked a repeated call"
                );
                // Stamped now, so every `after_tool` in the stack sees it
                // (`after_tool` hooks run in reverse registration order).
                ctx.set_refusal_metadata(call.id.clone(), guard_metadata(REPEAT_GUARD_BLOCKED));
                Err(TinyAgentsError::ToolFailed(text))
            }
            CallGate::Halt(summary) => {
                tracing::warn!(
                    tool = call.name,
                    "[tinyagents::mw] repeat-progress halted the run after repeated blocks"
                );
                ctx.set_refusal_metadata(call.id.clone(), guard_metadata(REPEAT_GUARD_HALTED));
                // Later refusals and results in this batch must not pause the
                // run again.
                let first = lock(&self.pending)
                    .get_mut(&run_id)
                    .is_none_or(|batch| !std::mem::replace(&mut batch.halted, true));
                if first {
                    self.halt_classified(ctx, repeat_stop_cause(summary.clone(), &call.name));
                }
                Err(TinyAgentsError::ToolFailed(summary))
            }
        }
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        invocation: &ToolInvocationIdentity,
        result: &mut TaToolResult,
    ) -> TaResult<()> {
        let tool_name = invocation.tool_name();
        let call_id = invocation.call_id().to_string();
        let run_id = ctx.instance_id();
        // Fingerprint outside the mutexes below: it scans the whole result.
        let identity = (!result.is_error).then(|| self.fingerprinter.fingerprint(&result.output()));
        let mut notes = Vec::new();
        // Fold this result into the pending batch; the call guard only acts once
        // the batch is complete so it sees whole-batch success.
        let (already_halted, recurrence, completed) = {
            let mut pending = lock(&self.pending);
            let Some(batch) = pending.get_mut(&run_id) else {
                return Ok(());
            };
            let already_halted = batch.halted;
            let mut recurrence = SuccessfulRepeat::Continue;
            if result.is_error {
                batch.all_ok = false;
            } else if let Some((tool, arguments)) = batch
                .call_sigs
                .get_mut(&call_id)
                .and_then(VecDeque::pop_front)
            {
                let identity = identity.as_deref().unwrap_or_default();
                let observation = self.state.with_monitor(run_id, |monitor| {
                    monitor.record_call(&tool, &arguments, identity, (self.read_only)(&tool))
                });
                recurrence = observation.verdict;
                notes.extend(observation.notes);
                lock(&self.state.recorded)
                    .entry(run_id)
                    .or_default()
                    .insert(call_id);
            } else {
                // An exempt (polling/wait) call succeeded: it stays out of the
                // repeat accounting, but it may have changed state, so results
                // predicted before it are no longer safe to block on.
                let read_only = (self.read_only)(tool_name);
                self.state
                    .with_monitor(run_id, |monitor| monitor.note_untracked_success(read_only));
            }
            if matches!(recurrence, SuccessfulRepeat::Halt(_)) {
                batch.halted = true;
            }
            batch.remaining = batch.remaining.saturating_sub(1);
            let completed = if batch.remaining == 0 {
                pending.remove(&run_id)
            } else {
                None
            };
            (already_halted, recurrence, completed)
        };
        let batch_verdict = completed
            .map(|batch| {
                self.state.with_monitor(run_id, |monitor| {
                    monitor.record_call_batch(&batch.call_sig, batch.all_ok, batch.exempt)
                })
            })
            .unwrap_or(SuccessfulRepeat::Continue);
        // The per-call recurrence note is the more specific one: when it fires,
        // the batch streak that coincides with it adds nothing.
        let mut candidates = Vec::new();
        if let (SuccessfulRepeat::Warn(note), _) | (_, SuccessfulRepeat::Warn(note)) =
            (&recurrence, &batch_verdict)
        {
            candidates.push(note.clone());
        }
        candidates.extend(notes);
        if !result.is_error {
            // One warning per result: the rest wait for the next one.
            let note = self.state.take_one_note(run_id, candidates);
            if let Some(note) = note {
                tracing::debug!(
                    tool = tool_name,
                    "[tinyagents::mw] repeat-progress appended a warning to the tool result"
                );
                append_note(result, &note);
            }
        }
        if already_halted {
            // An earlier result in this batch paused the run; keep the streak
            // accounting current without pausing again.
            return Ok(());
        }
        // When both fire on the same result, the batch summary wins: it is the
        // more specific description of an adjacent repeat.
        let summary = match (batch_verdict, recurrence) {
            (SuccessfulRepeat::Halt(summary), _) => summary,
            (_, SuccessfulRepeat::Halt(summary)) => summary,
            _ => return Ok(()),
        };
        tracing::warn!(
            tool = tool_name,
            "[tinyagents::mw] crate successful-repeat tracker halted the run"
        );
        self.halt_classified(ctx, repeat_stop_cause(summary, tool_name));
        Ok(())
    }
}

/// Resets [`RepeatProgressMiddleware`]'s recurrence ledger when context
/// reduction evicts a recorded tool result (#6275).
///
/// Compression, microcompact and trim rewrite only the outgoing request, never
/// the loop's transcript, so the eviction is visible only inside the
/// `before_model` chain. The guard snapshots which recorded results are intact
/// before those steps; this observer, registered after them, checks the same
/// ids in the final request. A result that was blanked or dropped is no longer
/// in front of the model, so re-reading it is not a repeat the model can see,
/// and the tracker restarts. Keyed on what disappeared rather than on which
/// middleware removed it, so any reduction step is covered, and a dialect that
/// never carries tool results as tool messages is never snapshotted.
pub struct RepeatEvictionObserver {
    state: Arc<RepeatState>,
}

#[async_trait]
impl<C: Send + Sync> Middleware<(), C> for RepeatEvictionObserver {
    fn name(&self) -> &str {
        "repeat_progress_eviction"
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> TaResult<()> {
        let run_id = ctx.instance_id();
        let before = lock(&self.state.visible_before_reduction)
            .remove(&run_id)
            .unwrap_or_default();
        if before.is_empty() {
            return Ok(());
        }
        let evicted = before.len()
            - visible_tool_results(request, &before, &self.state.cleared_placeholder).len();
        if evicted == 0 {
            return Ok(());
        }
        tracing::debug!(
            evicted,
            "[tinyagents::mw] repeat-progress ledger reset: recorded tool results left the context"
        );
        if let Some(monitor) = lock(&self.state.monitors).get_mut(&run_id) {
            monitor.on_context_evicted();
        }
        lock(&self.state.recorded).remove(&run_id);
        // Warnings held back were about results the model no longer has.
        self.state.clear_deferred(run_id);
        Ok(())
    }
}
