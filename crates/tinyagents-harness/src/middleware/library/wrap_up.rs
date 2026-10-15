//! [`FinalCallWrapUpMiddleware`]: turn the last permitted model call of a
//! capped turn into the turn's conclusion, inside the loop (issue #6014).

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result as TaResult;
use crate::middleware::Middleware;
use tinyinference_llm::message::{ContentBlock, Message as TaMessage};
use tinyinference_llm::model::ModelRequest;

use super::ephemeral::push_ephemeral_instruction;
use super::image_trim::{estimate_message_tokens, estimate_text_tokens};

/// The body microcompact swaps in for a cleared tool result, and therefore the
/// only body this middleware treats as restorable. The default for
/// [`FinalCallWrapUpMiddleware::with_cleared_placeholder`]; it must match the
/// placeholder the run's `MicrocompactMiddleware` was built with.
pub const DEFAULT_CLEARED_PLACEHOLDER: &str = "[Old tool result content cleared]";

/// The captured outcome store could not be read (a poisoned lock, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutcomesUnavailable;

/// Where [`FinalCallWrapUpMiddleware`] finds each tool call's captured result,
/// so the concluding call can be given back what microcompact blanked.
///
/// The host owns the capture (it records each call as the result enters the
/// transcript, after any per-result byte cap); this is the read side.
pub trait CapturedOutcomes: Send + Sync {
    /// The captured result text for `call_id`, `Ok(None)` when nothing was
    /// captured for it, or `Err` when the store cannot be read at all (the
    /// middleware then concludes without restoring anything).
    fn content_for(&self, call_id: &str) -> Result<Option<String>, OutcomesUnavailable>;
}

/// Turns the **last permitted model call of a capped turn** into the turn's
/// conclusion, in the loop, instead of leaving the answer to an extra call
/// made after the loop has already exited.
///
/// # Why the loop and not afterwards
///
/// A capped turn used to end like this: the loop exits on the model-call cap,
/// and `OpenHumanSessionHost::summarize_turn_wrapup` then dispatches a second, out-of-band
/// request straight at the `ChatModel` asking for a checkpoint. Being outside
/// the harness, that request ran with **none** of the loop's context
/// management — no microcompact, no compression, no trim — while being built
/// from the largest transcript the turn would ever hold (every tool result a
/// full iteration budget produced). It was therefore the likeliest call of the
/// whole turn to overflow the window, and each of its failure paths returns
/// `("", None)` silently, so the answer degraded to a deterministic digest of
/// tool names exactly when the turn had the most to report. It also bypassed
/// usage accounting (folded back by hand at the call site) and the progress
/// bridge (re-implemented there as buffer-then-validate-then-forward).
///
/// Doing it here removes all of that rather than compensating for it: the
/// concluding call is an ordinary loop iteration, so it inherits the entire
/// middleware stack, its usage rides `UsageCarryMiddleware` like any other
/// call, and its text streams through the normal event bridge. It is also one
/// provider call cheaper — the wrap-up was an extra call *past* the cap the
/// operator configured.
///
/// # What "last permitted call" means, exactly
///
/// The loop records the model call **before** it builds the request
/// (`agent_loop::run_loop`), so by the time `before_model` runs for the Nth
/// call of an N-call budget, `remaining_model_calls()` is already `0`. That is
/// the trigger, and it needs no new plumbing or counter of its own.
///
/// The trade this makes is explicit: a 25-call budget becomes 24 tool rounds
/// plus a conclusion, rather than 25 tool rounds plus a 26th call nobody
/// budgeted for.
///
/// # Why the tools are cleared rather than merely discouraged
///
/// The instruction alone is a request the model may ignore — the out-of-band
/// wrap-up had to re-parse its own response through the dispatcher to catch a
/// model that emitted a tool call anyway. Removing the schemas from the
/// request makes it structural instead: there is nothing to call. `tool_choice`
/// is reset alongside them because a `Required` choice with an empty tool array
/// is a provider 400.
/// The middleware itself. Generic over the run-context payload: nothing here
/// reads it. Configure the deliverable-tool names with
/// [`with_deliverable_tools`](Self::with_deliverable_tools) — the tools left on
/// the belt for the **penultimate** call (see `reserve_final_write`).
///
/// The membership rule is "can only emit, never gather": a tool that writes a
/// file the caller already knows the contents of cannot be spent discovering
/// something the turn then has no room to report, which is what makes reserving
/// the call for them a safe trade rather than a gamble. A shell is deliberately
/// not one — it can equally run a crawler, so keeping it would leave the belt
/// effectively unnarrowed. With none configured the penultimate call is left
/// ordinary.
pub struct FinalCallWrapUpMiddleware {
    /// The synthetic user turn appended on the final call.
    instruction: String,
    /// The synthetic user turn appended on the call before it, when the belt is
    /// narrowed to the deliverable tools instead of cleared.
    final_write_instruction: String,
    /// Names of the tools the penultimate call keeps.
    deliverable_tools: Vec<String>,
    /// The body a cleared tool result carries (see
    /// [`DEFAULT_CLEARED_PLACEHOLDER`]).
    cleared_placeholder: String,
    /// Every tool call's captured outcome, so the concluding call can be given
    /// back the results microcompact blanked (see `before_model`).
    outcomes: Arc<dyn CapturedOutcomes>,
    /// The input-token allowance the trim downstream enforces, so restoration
    /// can stay under it rather than provoking an eviction. `0` disables the
    /// bound (a model advertising no context window).
    input_budget: u64,
    /// Set when the injection fires, so the caller can report the turn as
    /// capped. Necessary because this turn now ends *naturally* — the model
    /// returns text and requests no tools, which is the loop's ordinary
    /// terminal condition — so the old `final_response.is_none()` tell no
    /// longer distinguishes a capped turn from a finished one.
    fired: std::sync::Mutex<std::collections::HashSet<u64>>,
    /// Ascending fractions of the model-call budget at which a budget notice
    /// is appended (see [`with_budget_notice`](Self::with_budget_notice)).
    /// Empty disables the notice.
    budget_thresholds: Vec<f64>,
    /// Per run ([`RunContext::instance_id`]), how many of `budget_thresholds`
    /// have already been announced.
    budget_noticed: std::sync::Mutex<std::collections::HashMap<u64, usize>>,
}

impl FinalCallWrapUpMiddleware {
    /// Build the middleware. The cleared-result placeholder defaults to
    /// [`DEFAULT_CLEARED_PLACEHOLDER`] and no deliverable tools are configured.
    pub fn new(
        instruction: impl Into<String>,
        final_write_instruction: impl Into<String>,
        outcomes: Arc<dyn CapturedOutcomes>,
        input_budget: u64,
    ) -> Self {
        Self {
            instruction: instruction.into(),
            final_write_instruction: final_write_instruction.into(),
            deliverable_tools: Vec::new(),
            cleared_placeholder: DEFAULT_CLEARED_PLACEHOLDER.to_string(),
            outcomes,
            input_budget,
            fired: std::sync::Mutex::default(),
            budget_thresholds: Vec::new(),
            budget_noticed: std::sync::Mutex::default(),
        }
    }

    /// Tools the penultimate call keeps when the belt is narrowed.
    pub fn with_deliverable_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.deliverable_tools = tools.into_iter().map(Into::into).collect();
        self
    }

    /// Warn the model about its model-call budget partway through the turn.
    ///
    /// Each threshold is a fraction of `max_model_calls` (`0.5` = half the
    /// budget spent). The first call at or past a threshold gets a short note
    /// appended saying how many calls are left and that it should start, or
    /// keep, producing the deliverable instead of gathering. Each threshold
    /// fires once per run; thresholds crossed by the same call share one note.
    ///
    /// Without this the model learns about the budget only on the penultimate
    /// call, when the belt is already narrowed to the writers, which is too
    /// late to make a multi-file change (openhuman#6958). A note never lands on
    /// the penultimate or final call, which carry their own instruction.
    /// Thresholds outside `(0, 1)` (or NaN) are ignored.
    pub fn with_budget_notice<I>(mut self, thresholds: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        let mut thresholds: Vec<f64> = thresholds
            .into_iter()
            .filter(|t| t.is_finite() && *t > 0.0 && *t < 1.0)
            .collect();
        thresholds.sort_by(f64::total_cmp);
        thresholds.dedup();
        self.budget_thresholds = thresholds;
        self
    }

    /// Override the placeholder body treated as a cleared tool result.
    pub fn with_cleared_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.cleared_placeholder = placeholder.into();
        self
    }

    /// Whether a tool is one the penultimate call keeps.
    fn is_deliverable_tool(&self, name: &str) -> bool {
        self.deliverable_tools.iter().any(|tool| tool == name)
    }

    /// Whether a budget notice was announced in the run owning this context.
    /// Lets other middleware (for example `VerifyBeforeFinishMiddleware`)
    /// stay quiet once the model has been told how few calls remain.
    pub fn budget_notice_announced<C>(&self, ctx: &RunContext<C>) -> bool {
        self.budget_noticed
            .lock()
            .is_ok_and(|noticed| noticed.get(&ctx.instance_id()).is_some_and(|n| *n > 0))
    }

    /// Whether the wrap-up injection fired for this run context.
    pub fn fired<C>(&self, ctx: &RunContext<C>) -> bool {
        self.fired_for(ctx.instance_id())
    }

    /// Whether the wrap-up injection fired for the run whose context had this
    /// [`RunContext::instance_id`]. For callers that hand the context to the
    /// run (it is not `Clone`) and read the outcome afterwards.
    pub fn fired_for(&self, instance_id: u64) -> bool {
        self.fired
            .lock()
            .is_ok_and(|fired| fired.contains(&instance_id))
    }

    /// Give a concluding-or-persisting call back the tool results microcompact
    /// blanked, newest-first and only while the request still fits.
    ///
    /// Shared by both of this middleware's calls, because both need the same
    /// thing for the same reason: the content the turn gathered. The final call
    /// needs it to *report* findings, and the penultimate one needs it to
    /// *write them into a file* — and a turn asked to produce an artifact from
    /// nineteen rounds of "[Old tool result content cleared]" produces the same
    /// empty-handed result from either direction.
    ///
    /// `instruction` is the text the caller will append afterwards; it is
    /// seeded into the token accounting here rather than counted later, because
    /// restoration fills the budget to its boundary and an unaccounted fixed
    /// addition after it is exactly the overshoot the budget exists to prevent
    /// (CodeRabbit on #6068).
    fn restore_cleared_outcomes(&self, request: &mut ModelRequest, instruction: &str) -> usize {
        let budget = self.input_budget;
        // Seeded with the instruction this middleware appends unconditionally
        // below, not just with what the request already holds (CodeRabbit on
        // #6068). Restoration fills the budget to its boundary, so an
        // unaccounted fixed addition after it is exactly the overshoot the
        // budget exists to prevent.
        let mut used: u64 = request
            .messages
            .iter()
            .map(estimate_message_tokens)
            .sum::<u64>()
            .saturating_add(estimate_text_tokens(instruction));
        let restored = {
            let mut restored = 0usize;
            let mut skipped = 0usize;
            let mut unavailable = false;
            for message in request.messages.iter_mut().rev() {
                let TaMessage::Tool(tool) = message else {
                    continue;
                };
                if tool
                    .content
                    .iter()
                    .any(|block| !matches!(block, ContentBlock::Text(_)))
                {
                    continue;
                }
                let body: String = tool
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if body.trim() != self.cleared_placeholder {
                    continue;
                }
                let outcome = match self.outcomes.content_for(&tool.tool_call_id) {
                    Ok(Some(outcome)) => outcome,
                    Ok(None) => continue,
                    Err(OutcomesUnavailable) => {
                        unavailable = true;
                        break;
                    }
                };
                if outcome.trim().is_empty() {
                    continue;
                }
                // What restoring this body would add, against what the
                // placeholder already costs.
                let added = estimate_text_tokens(&outcome)
                    .saturating_sub(estimate_text_tokens(&self.cleared_placeholder));
                if budget > 0 && used.saturating_add(added) > budget {
                    // Everything older is at least as likely to overflow, but
                    // keep counting so the log reports the true shortfall
                    // rather than stopping at the first one that did not fit.
                    skipped += 1;
                    continue;
                }
                used = used.saturating_add(added);
                tool.content = vec![ContentBlock::Text(outcome)];
                restored += 1;
            }
            if unavailable {
                tracing::warn!(
                    "[tinyagents::mw] tool-outcome sink poisoned; concluding without restoring \
                         cleared tool results"
                );
                return 0;
            }
            if skipped > 0 {
                tracing::info!(
                    skipped,
                    restored,
                    budget,
                    used,
                    "[tinyagents::mw] left some cleared tool results cleared: restoring them \
                         would have pushed the concluding call past its input budget, and an \
                         eviction there costs whole messages rather than one body"
                );
            }
            restored
        };
        if restored > 0 {
            tracing::info!(
                restored,
                "[tinyagents::mw] restored cleared tool results for the concluding call"
            );
        }
        restored
    }
}

/// The note [`FinalCallWrapUpMiddleware::with_budget_notice`] appends.
///
/// It names no kind of task. The same harness runs agents that write code,
/// browse or shop, and an example of the wrong kind ("edit the files") steers
/// a model that has no files away from what it was asked to do.
fn budget_notice_text(remaining: usize, max: usize) -> String {
    format!(
        "Budget notice: {remaining} model calls left in this turn (of {max}). Stop gathering \
         and start (or continue) finishing the task now: take the steps that complete it \
         rather than looking around more. Near the end the tools are withdrawn, so anything \
         not done by then will not get done."
    )
}

impl FinalCallWrapUpMiddleware {
    /// Append a budget notice when this call is the first at or past one or
    /// more not-yet-announced thresholds. Returns `true` when it did.
    ///
    /// Only called while more than one call remains, so a notice never
    /// competes with the final-write or concluding instruction.
    fn maybe_notice_budget<C>(&self, ctx: &RunContext<C>, request: &mut ModelRequest) -> bool {
        if self.budget_thresholds.is_empty() {
            return false;
        }
        let max = ctx.limits.limits().max_model_calls;
        let used = ctx.limits.model_calls();
        let crossed = self
            .budget_thresholds
            .iter()
            // The epsilon keeps `0.7 * 10` (6.999…) landing on call 7.
            .take_while(|t| used as f64 + 1e-9 >= **t * max as f64)
            .count();
        let Ok(mut noticed) = self.budget_noticed.lock() else {
            return false;
        };
        let announced = noticed.entry(ctx.instance_id()).or_insert(0);
        if crossed <= *announced {
            return false;
        }
        *announced = crossed;
        drop(noticed);
        let remaining = ctx.limits.remaining_model_calls();
        tracing::info!(
            model_calls = used,
            max_model_calls = max,
            remaining,
            thresholds_crossed = crossed,
            "[tinyagents::mw] budget notice — telling the model how many calls are left"
        );
        push_ephemeral_instruction(
            request,
            budget_notice_text(remaining, max),
            ctx.model_profile.as_ref(),
        );
        true
    }

    /// The call *before* the conclusion: narrow the belt to
    /// the deliverable tools so a turn that owes a file can still write it.
    ///
    /// Clearing the belt one call later makes the conclusion structural, which
    /// is right — but for a turn whose product is an artifact rather than
    /// prose it makes *failure* structural too. The host's final-write
    /// instruction documents the case that motivated this and the trade it
    /// accepts.
    ///
    /// Returns `true` when the narrowing fired, so the caller can skip the
    /// instruction otherwise.
    fn reserve_final_write<C>(&self, ctx: &RunContext<C>, request: &mut ModelRequest) -> bool {
        // Below three, reserving would eat the turn rather than shape its end:
        // a two-call budget would be one write-only call plus the conclusion,
        // leaving no round in which anything could be gathered to write.
        if ctx.limits.limits().max_model_calls <= 2 {
            return false;
        }
        // Nothing to reserve the call *for*. A read-only or delegating agent
        // has no writer on its belt, and telling it "the only tools left are
        // the ones that write files" would be false — so leave the call as an
        // ordinary one and let the conclusion handle the cap.
        if !request
            .tools
            .iter()
            .any(|t| self.is_deliverable_tool(&t.name))
        {
            return false;
        }
        let before = request.tools.len();
        request.tools.retain(|t| self.is_deliverable_tool(&t.name));
        // `Auto`, never `Required`: a turn that has already written its file,
        // or was only ever asked for an answer, must be free to spend this call
        // on text instead. Forcing a call here would make it invent a write.
        request.tool_choice = tinyinference_llm::model::ToolChoice::Auto;
        tracing::info!(
            model_calls = ctx.limits.model_calls(),
            max_model_calls = ctx.limits.limits().max_model_calls,
            tools_withdrawn = before.saturating_sub(request.tools.len()),
            tools_kept = request.tools.len(),
            "[tinyagents::mw] penultimate model call — narrowing the belt to the tools that can \
             persist a deliverable"
        );
        // The same restoration the conclusion gets, and for a sharper reason:
        // this call is being asked to write the findings into a file, so it
        // needs to be able to read them.
        self.restore_cleared_outcomes(request, &self.final_write_instruction);
        push_ephemeral_instruction(
            request,
            self.final_write_instruction.clone(),
            ctx.model_profile.as_ref(),
        );
        true
    }
}

#[async_trait]
impl<C: Send + Sync> Middleware<(), C> for FinalCallWrapUpMiddleware {
    fn name(&self) -> &str {
        "final_call_wrap_up"
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> TaResult<()> {
        let remaining = ctx.limits.remaining_model_calls();
        if remaining > 1 {
            self.maybe_notice_budget(ctx, request);
            return Ok(());
        }
        // A budget of one call would make the very first call the concluding
        // one, so the turn could never run a tool at all. That is a
        // misconfiguration rather than a cap being reached, and silently
        // answering it with a "you have run out of tool calls" instruction
        // would misreport it — leave such a run alone.
        if ctx.limits.limits().max_model_calls <= 1 {
            return Ok(());
        }
        // One call before the conclusion: keep the writers rather than clear
        // the belt, so a turn whose deliverable is a file can still produce it.
        if remaining == 1 {
            self.reserve_final_write(ctx, request);
            return Ok(());
        }
        tracing::info!(
            model_calls = ctx.limits.model_calls(),
            max_model_calls = ctx.limits.limits().max_model_calls,
            tools_withdrawn = request.tools.len(),
            "[tinyagents::mw] final permitted model call — withdrawing tools and asking for the \
             turn's conclusion"
        );
        request.tools.clear();
        request.tool_choice = tinyinference_llm::model::ToolChoice::None;
        // Give the concluding call back the results microcompact blanked.
        //
        // `MicrocompactMiddleware` replaces every tool-result body past the
        // most recent `keep_recent` (5, by default) with `CLEARED_PLACEHOLDER`,
        // and — constructed without a token budget — it does so on every call,
        // not only under context pressure. That is right for an intermediate
        // call, which needs recent context to choose the next tool and nothing
        // more. It is exactly wrong for this one: a turn that spent 24 rounds
        // gathering would be asked to report its findings with 19 rounds of
        // them replaced by "[Old tool result content cleared]", which is the
        // same empty-handed answer this whole mechanism exists to prevent,
        // arrived at from the other direction.
        //
        // Restored from the captured outcomes rather than by exempting the
        // turn from microcompact, because the blanking has already happened by
        // the time this runs (registration order: microcompact is installed by
        // `context_mw.install`, this middleware immediately after it) and
        // because the sink is the honest source — it holds each result as it
        // entered the transcript, after the per-result byte cap.
        //
        // Only a body that IS the placeholder is replaced, so a result the
        // model legitimately saw in full is never rewritten.
        //
        // This deliberately runs BEFORE the compression and trim middlewares,
        // which are installed after it: restoring can make the request large,
        // and those two are what bound it. The resulting degradation ladder is
        // the one this call wants — everything when it fits, an LLM summary of
        // the older slice when it does not, and oldest-first eviction only in
        // extremis. What it never does again is silently blank the middle.
        // Restore newest-first, and only while the request still fits.
        //
        // CodeRabbit on #6068: restoration runs AFTER `ContextCompressionMiddleware`
        // and before `ImageAwareMessageTrimMiddleware`, so an unbounded restore can
        // push the request over the window and the trim then evicts whole
        // messages — which is strictly more destructive than the blanking being
        // undone, and can discard the very results just restored.
        //
        // The review suggested a compression phase after restoration. That is a
        // second summarizer model call on the one call already about to produce
        // the conclusion, and it is avoidable: the overflow is preventable
        // rather than repairable. Restoring under the same budget the trim
        // enforces means the trim never has cause to fire, so nothing is
        // evicted and nothing needs re-summarising.
        //
        // Newest-first because recency is relevance here: the last rounds'
        // results are the ones the model has not seen (the cap is checked
        // before the request is built), and the earliest ones are most likely
        // already reflected in the compression summary above.
        self.restore_cleared_outcomes(request, &self.instruction);
        push_ephemeral_instruction(
            request,
            self.instruction.clone(),
            ctx.model_profile.as_ref(),
        );
        if let Ok(mut fired) = self.fired.lock() {
            fired.insert(ctx.instance_id());
        }
        Ok(())
    }
}
