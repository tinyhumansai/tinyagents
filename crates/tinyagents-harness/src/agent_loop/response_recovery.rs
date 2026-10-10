//! Post-response recovery for the superstep loop: what to do with a model
//! response that cannot be taken at face value.
//!
//! Split out of `run_loop.rs`. Two stages run on every response, both bounded
//! by the counters in [`TurnRecovery`]:
//!
//! 1. [`AgentHarness::reject_truncated_tool_calls`] — a length stop that may
//!    have cut a tool call mid-arguments: answer the suspect calls with an
//!    error instead of running them and let the model retry.
//! 2. [`AgentHarness::recover_unusable_response`] — a response with no tool
//!    call that is not a usable answer (a withheld call, a truncated-empty or
//!    empty reply, a dropped or undecodable call): re-issue or re-prompt.

use super::run_loop::{
    DROPPED_TOOL_CALL_NUDGE, TRUNCATED_EMPTY_ANSWER_NUDGE, TRUNCATED_EMPTY_CARRY_PREFIX,
    TRUNCATED_EMPTY_CARRY_SUFFIX, TRUNCATED_EMPTY_REASONING_OFF_ANSWER_NOTE,
    TRUNCATED_EMPTY_REASONING_OFF_TOOL_NOTE, TRUNCATED_EMPTY_TOOL_NUDGE,
    UNDECODABLE_TOOL_CALL_NUDGE, WITHHELD_TOOL_CALL_NUDGE, truncated_call_positions,
};
use super::turn_recovery::TRUNCATED_CLOCK_NUDGE_LIMIT;
use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// A length stop means the output cap cut the reply off somewhere: the LAST
    /// call of the response may carry truncated (yet parseable) arguments —
    /// native or recovered from text alike, since a text grammar can close an
    /// open `{`/`[` or run a payload to end-of-text — as may any call the
    /// provider flagged invalid (a repair could make it look whole). Answer
    /// those with an error instead of running them and let the model retry;
    /// every earlier call was finished before the cut and runs normally. The
    /// call is chosen by position, not id, so duplicate or empty provider ids
    /// fail closed. Bounded per logical turn. A run resumed in a fresh context
    /// restarts this budget.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn reject_truncated_tool_calls(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        turn_recovery: &mut TurnRecovery,
        turn: &ResponseTurn<'_>,
    ) -> Result<TruncationOutcome> {
        let ResponseTurn {
            call_id,
            response,
            tool_calls,
            attempt_max_tokens,
            structured_call_names,
            ..
        } = *turn;
        ctx.truncated_call_positions.clear();
        ctx.truncated_repeat_positions.clear();
        if self.policy.reject_truncated_tool_calls
            && !tool_calls.is_empty()
            && crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
        {
            let truncated_positions = truncated_call_positions(tool_calls);
            if !truncated_positions.is_empty() {
                let truncated_names: std::collections::BTreeSet<String> = truncated_positions
                    .iter()
                    .map(|&index| tool_calls[index].name.clone())
                    .collect();
                // A tool already warned that its calls are too large, cut off
                // once more: the model is re-sending the same oversized shape
                // and the output cap is not going to move. Stop here instead
                // of spending the rest of the retry budget on it.
                if let Some(name) = truncated_names
                    .iter()
                    .find(|name| turn_recovery.truncated_repeat_names.contains(*name))
                {
                    tracing::warn!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        tool = %name,
                        "[agent_loop] the same tool keeps being cut off by the output token limit; stopping"
                    );
                    messages.pop();
                    ctx.retract_transcript(messages.len());
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "run `{}` stopped: tool `{name}` was cut off by the output token limit \
                         on consecutive turns after being told to split the call into smaller \
                         ones (RunPolicy::reject_truncated_tool_calls)",
                        ctx.run_id(),
                    )));
                }
                if turn_recovery.truncated_tool_call_retries_used
                    >= self.policy.truncated_tool_call_retries
                {
                    tracing::warn!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        retries = turn_recovery.truncated_tool_call_retries_used,
                        "[agent_loop] length-truncated tool calls keep recurring; truncated-tool-call retry budget exhausted"
                    );
                    messages.pop();
                    ctx.retract_transcript(messages.len());
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "run `{}` stopped: {} consecutive \
                             retries of a tool call truncated by the output token limit did not \
                             produce a complete call (RunPolicy::truncated_tool_call_retries)",
                        ctx.run_id(),
                        turn_recovery.truncated_tool_call_retries_used
                    )));
                }
                turn_recovery.truncated_tool_call_retries_used += 1;
                let repeated: std::collections::BTreeSet<String> = truncated_names
                    .intersection(&turn_recovery.truncated_tool_names)
                    .cloned()
                    .collect();
                // Give the retry room to finish the call.
                turn_recovery.boost_max_tokens(attempt_max_tokens);
                if !repeated.is_empty() {
                    tracing::info!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        tools = ?repeated,
                        "[agent_loop] same tool cut off by the output limit twice; answering with the stop-repeating corrective"
                    );
                }
                turn_recovery
                    .truncated_repeat_names
                    .extend(repeated.iter().cloned());
                turn_recovery.truncated_tool_names = truncated_names;
                tracing::info!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    call_id = %call_id,
                    calls = tool_calls.len(),
                    rejected = truncated_positions.len(),
                    attempt = turn_recovery.truncated_tool_call_retries_used,
                    max_tokens = ?turn_recovery.boosted_max_tokens,
                    finish_reason = ?response.finish_reason,
                    "[agent_loop] length-truncated response; failing its possibly-incomplete tool calls instead of running them"
                );
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "truncated_tool_calls".to_string(),
                    detail: format!(
                        "model call `{call_id}` hit its output limit mid-turn; {} of {} tool \
                             call(s) answered with an error, not run",
                        truncated_positions.len(),
                        tool_calls.len()
                    ),
                });
                status.set_last_event(record.id);
                if truncated_positions
                    .iter()
                    .any(|&index| structured_call_names.contains(&tool_calls[index].name))
                {
                    // The structured-output call itself was cut off: it
                    // cannot be extracted as the answer, and the turn has
                    // no other path that answers it. Fail the whole turn.
                    status.mark_running(HarnessPhase::Tools);
                    self.fail_truncated_tool_calls(state, ctx, run, status, messages, tool_calls)
                        .await?;
                    self.apply_queued_lane(
                        ctx,
                        status,
                        messages,
                        crate::run_queue::QueueLane::Steer,
                    )
                    .await;
                    return Ok(
                        match self.apply_pending_control(ctx, run, status, messages)? {
                            ControlEffect::Exit(exit) => TruncationOutcome::EndTurn(Some(exit)),
                            ControlEffect::None | ControlEffect::ContinueLoop => {
                                TruncationOutcome::EndTurn(None)
                            }
                        },
                    );
                }
                // Admission answers these calls with the error, in call
                // order, as the batch runs (see `admit_tool_call`). The
                // batch holds the non-structured calls only, so translate
                // each position into the batch's own index space.
                let mut batch_index = 0;
                for (index, call) in tool_calls.iter().enumerate() {
                    if structured_call_names.contains(&call.name) {
                        continue;
                    }
                    if truncated_positions.contains(&index) {
                        ctx.truncated_call_positions.insert(batch_index);
                        if repeated.contains(&call.name) {
                            ctx.truncated_repeat_positions.insert(batch_index);
                        }
                    }
                    batch_index += 1;
                }
                return Ok(TruncationOutcome::CallsRejected);
            }
        }
        Ok(TruncationOutcome::Clean)
    }

    /// Recovery for a response with no real tool call that is not yet a usable
    /// answer. Returns `true` when the loop must continue for recovery or its
    /// existing limit handling; `false` when the response stands and the caller
    /// goes on to resolve the turn.
    ///
    /// The checks run in a fixed order: a withheld call, a truncated-empty
    /// retry, a truncated-empty nudge, a non-truncated empty retry, and a
    /// dropped or undecodable call nudge.
    pub(super) fn recover_unusable_response(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        turn_recovery: &mut TurnRecovery,
        turn: &ResponseTurn<'_>,
    ) -> bool {
        let ResponseTurn {
            call_id,
            response,
            tool_calls,
            attempt_max_tokens,
            started_at_ms,
            recovery,
            tools_available: tools_available_this_turn,
            text_dialect_calls_recoverable,
            has_structured_plan,
            ..
        } = *turn;
        // A call written on a turn that could not take one (tools
        // withdrawn for a concluding answer, or `ToolChoice::None`).
        // It was scrubbed and not run; what is left is either nothing
        // or a lead-in to a step that never happened, so it is not the
        // answer the request asked for. Drop that row and ask once
        // more, telling the model plainly that tools are gone.
        // Replaying the bench request that leaked (DeepSeek V4, tools
        // withdrawn), the unchanged request leaked 6 times in 8; with
        // the row dropped and this re-prompt added it leaked 0 times
        // in 12. Runs before the empty-reply retries: a bare re-send
        // of the same transcript leaks the same way.
        let withheld_calls = recovery.dropped.withheld();
        if withheld_calls > 0
            && turn_recovery.withheld_call_nudges_used < self.policy.dropped_tool_call_nudges
            && ctx.limits.remaining_model_calls() > 0
        {
            turn_recovery.withheld_call_nudges_used += 1;
            messages.pop();
            ctx.retract_transcript(messages.len());
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                withheld_calls,
                attempt = turn_recovery.withheld_call_nudges_used,
                "[agent_loop] re-prompting after a tool call on a turn with no callable tools"
            );
            ctx.emit(AgentEvent::ControlApplied {
                control: "withheld_tool_call".to_string(),
                detail: format!(
                    "{withheld_calls} tool call(s) written while no tool was callable \
                             in model call `{call_id}`; scrubbed, not run, re-prompted"
                ),
            });
            messages.push(Message::user(WITHHELD_TOOL_CALL_NUDGE));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.withheld_call_nudges_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // Truncated-empty recovery (runs before structured extraction,
        // which would otherwise fail on the empty completion). A local
        // reasoning model can burn the whole token budget on its hidden
        // reasoning channel and return `finish_reason == "length"` with
        // no visible text, no tool calls, and no structured output — a
        // result useless to every caller. Retry the call (bumping the
        // token budget when one was set) instead of surfacing the blank.
        // A structured tool hit carries a real payload, so it is never
        // treated as truncated-empty.
        let truncated_empty = tool_calls.is_empty()
            && crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
            && response.text().trim().is_empty();
        if truncated_empty && self.policy.truncated_empty_reasoning_fallback {
            // Whatever follows (a retry, a nudge, or nothing), the next calls
            // go out without reasoning: the cap and the effort label have
            // both been shown not to stop this.
            let holdoff = turn_recovery.reasoning_fallback.on_dead_call();
            ctx.emit(AgentEvent::ControlApplied {
                control: "reasoning_fallback".to_string(),
                detail: format!(
                    "model call `{call_id}` died at its output cap with nothing to show; \
                     the next {holdoff} call(s) go out with reasoning switched off"
                ),
            });
        }
        let reasoning_off = self.policy.truncated_empty_reasoning_fallback
            && turn_recovery.reasoning_fallback.active();
        // What a retry would cost and whether it can change anything. The
        // dead call's own duration and output count give the rate this model
        // emits at here; the next cap divided by that rate is how long the
        // retry would run if it dies the same way.
        let truncated_retry = truncated_empty.then(|| {
            let dead_ms = crate::ids::now_ms().saturating_sub(started_at_ms);
            let dead_tokens = response
                .usage
                .as_ref()
                .map(|u| u.output_tokens)
                .unwrap_or(0);
            let remaining = crate::middleware::library::TurnClock::of(
                ctx,
                self.policy
                    .limits
                    .max_wall_clock_ms
                    .map(std::time::Duration::from_millis),
            )
            .map(|clock| clock.remaining());
            turn_recovery.truncated_retry_plan(
                attempt_max_tokens,
                dead_tokens,
                dead_ms,
                remaining,
                reasoning_off,
            )
        });
        if truncated_empty
            && turn_recovery.truncated_empty_retries_used < self.policy.truncated_empty_retries
            && ctx.limits.remaining_model_calls() > 0
            && truncated_retry.as_ref().is_some_and(|plan| plan.worth_it())
        {
            // Drop the useless empty assistant row appended above so the
            // retry re-sends the identical transcript, plus the dead call's
            // own working-out where it had any: that reasoning was real
            // progress (a correct derivation, cut off), and without it every
            // retry starts the same derivation over.
            messages.pop();
            ctx.retract_transcript(messages.len());
            self.carry_dead_call_reasoning(ctx, messages, call_id, response);
            let plan = truncated_retry.expect("a truncated-empty reply has a plan");
            turn_recovery.take_truncated_retry(attempt_max_tokens, &plan);
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.truncated_empty_retries_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }
        // A retry that cannot grow the cap re-sends the transcript that just
        // died at this cap, and one that would run past half the remaining
        // clock leaves no time to use its result either way (one 13-minute
        // run spent ten of them in four such calls, the last two at the
        // ceiling, and ended with nothing written). Say why, and let the
        // nudge below ask for a smaller step instead. The cap it runs under is
        // pinned to what the clock affords.
        if let Some(plan) = truncated_retry
            .as_ref()
            .filter(|plan| !plan.worth_it())
            .filter(|_| {
                turn_recovery.truncated_empty_retries_used < self.policy.truncated_empty_retries
            })
        {
            let reason = plan.skip_reason();
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                dead_ms = plan.dead_ms,
                dead_tokens = plan.dead_tokens,
                current_cap = ?plan.current,
                next_cap = ?plan.next,
                remaining_ms = ?plan.remaining.map(|d| d.as_millis()),
                "[agent_loop] truncated-empty retry skipped: {reason}"
            );
            ctx.emit(AgentEvent::ControlApplied {
                control: "truncated_empty_retry_skipped".to_string(),
                detail: format!("model call `{call_id}`: {reason}"),
            });
            if let Some(cap) = plan.affordable_cap() {
                turn_recovery.boosted_max_tokens = Some(cap);
            }
        }

        // The retries are spent and the model still deliberated past its
        // output budget. Re-sending the same transcript keeps failing the
        // same way (a high-effort reasoning model thinks as long as it is
        // allowed to), and finishing here hands the host a blank reply it
        // can only close as if the work were done. Say plainly what happened
        // and ask for the next step, then carry on with the loop.
        //
        // A nudged call that dies too used to end the turn: the loop
        // surfaced the blank and the host closed the run, in one case with
        // 21 of 30 minutes unused and the deliverable unwritten. While the
        // run has a clock with room for another bounded call, keep nudging,
        // and halve the output cap each time: the cap is the one limit on
        // deliberation these providers honour, and a smaller one makes the
        // model act sooner.
        // Zero explicitly disables nudges even when the run has clock room.
        let clock_allows_another_nudge = self.policy.truncated_empty_nudges > 0
            && truncated_retry
                .as_ref()
                .is_some_and(|plan| plan.another_nudge_fits())
            && turn_recovery.truncated_empty_nudges_used < TRUNCATED_CLOCK_NUDGE_LIMIT;
        let clock_only_nudge =
            turn_recovery.truncated_empty_nudges_used >= self.policy.truncated_empty_nudges;
        let policy_nudge_fits = truncated_retry
            .as_ref()
            .is_some_and(|plan| plan.remaining.is_none() || plan.another_nudge_fits());
        if truncated_empty
            && (turn_recovery.truncated_empty_nudges_used < self.policy.truncated_empty_nudges
                || clock_allows_another_nudge)
            && policy_nudge_fits
            && ctx.limits.remaining_model_calls() > 0
        {
            messages.pop();
            ctx.retract_transcript(messages.len());
            self.carry_dead_call_reasoning(ctx, messages, call_id, response);
            turn_recovery.truncated_empty_nudges_used += 1;
            let retry_was_skipped_for_clock = truncated_retry
                .as_ref()
                .is_some_and(|plan| !plan.fits_clock());
            let retry_was_skipped_at_ceiling = truncated_retry
                .as_ref()
                .is_some_and(|plan| plan.at_ceiling());
            let repeat_cap = (turn_recovery.truncated_empty_nudges_used > 1
                || clock_only_nudge
                || retry_was_skipped_for_clock
                || retry_was_skipped_at_ceiling)
                .then(|| truncated_retry.as_ref().and_then(|plan| plan.nudge_cap()))
                .flatten();
            if let Some(cap) = repeat_cap {
                turn_recovery.boosted_max_tokens = Some(cap);
            }
            let mut nudge: String = match repeat_cap {
                Some(cap) => format!(
                    "Your reply was cut off again before any tool call or answer. The output \
                     limit for the next call is {cap} tokens: reason in a few sentences at most, \
                     then act: {}.",
                    if tools_available_this_turn {
                        "make one tool call"
                    } else {
                        "write a short answer from what you already have"
                    }
                ),
                None if tools_available_this_turn => TRUNCATED_EMPTY_TOOL_NUDGE.to_string(),
                None => TRUNCATED_EMPTY_ANSWER_NUDGE.to_string(),
            };
            if reasoning_off {
                // The deliberation the model cannot finish in its head goes
                // into the workspace instead.
                nudge.push(' ');
                nudge.push_str(if tools_available_this_turn {
                    TRUNCATED_EMPTY_REASONING_OFF_TOOL_NOTE
                } else {
                    TRUNCATED_EMPTY_REASONING_OFF_ANSWER_NOTE
                });
            }
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                attempt = turn_recovery.truncated_empty_nudges_used,
                tools_available = tools_available_this_turn,
                max_tokens = ?turn_recovery.boosted_max_tokens.or(attempt_max_tokens),
                "[agent_loop] truncated-empty retries spent; nudging model to act"
            );
            ctx.emit(AgentEvent::ControlApplied {
                control: "truncated_empty_nudge".to_string(),
                detail: format!(
                    "model call `{call_id}` ran out of output tokens while reasoning \
                             after {} retry(ies); re-prompted to act",
                    turn_recovery.truncated_empty_retries_used
                ),
            });
            messages.push(Message::user(nudge));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: (turn_recovery.truncated_empty_retries_used
                    + turn_recovery.truncated_empty_nudges_used) as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // A provider can also finish normally after sending only a
        // reasoning side channel (or no content at all). Retrying that
        // unusable answer is opt-in because it incurs another provider
        // call. Unlike a length-truncated reply, keep the same token
        // cap: there is no evidence that output space ran out.
        let nontruncated_empty = tool_calls.is_empty()
            && response.text().trim().is_empty()
            && response.continue_turn.is_none()
            && !has_structured_plan
            && run.structured.is_none()
            && !crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
            && response.finish_reason.as_deref() != Some("tool_calls")
            && !response.served_from_cache;
        if nontruncated_empty
            && turn_recovery.empty_response_retries_used < self.policy.empty_response_retries
            && ctx.limits.remaining_model_calls() > 0
        {
            messages.pop();
            ctx.retract_transcript(messages.len());
            turn_recovery.empty_response_retries_used += 1;
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                attempt = turn_recovery.empty_response_retries_used,
                finish_reason = ?response.finish_reason,
                content_blocks = response.message.content.len(),
                "[agent_loop] retrying completion without visible answer"
            );
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.empty_response_retries_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // Dropped tool call: the provider says the model stopped to
        // call a tool, but nothing arrived — structured or in text.
        // A bounded re-prompt asks for the call itself. The assistant
        // row stays on the transcript so the model sees what it did.
        //
        // A text dialect always finishes with `stop`, so its dropped
        // call is a block a grammar recognised that became no call:
        // one whose body did not decode (scrubbed, only the lead-in
        // prose left), or one the model stopped inside without a
        // closer. A `length` stop inside a block is truncation, not a
        // forgotten closer, and is left to the truncation handling.
        // Native models with text recovery on parse the same grammars
        // out of their prose, so the same drop applies to them.
        let malformed_blocks = recovery.dropped.malformed();
        let unterminated_blocks =
            if crate::finish_reason::is_length_stop(response.finish_reason.as_deref()) {
                0
            } else {
                recovery.dropped.unterminated()
            };
        let undecodable_text_call =
            text_dialect_calls_recoverable && malformed_blocks + unterminated_blocks > 0;
        if tool_calls.is_empty()
            && (response.finish_reason.as_deref() == Some("tool_calls") || undecodable_text_call)
            && tools_available_this_turn
            && turn_recovery.dropped_tool_call_nudges_used < self.policy.dropped_tool_call_nudges
        {
            if ctx.limits.remaining_model_calls() == 0 {
                // Let the loop apply its limit policy without recording a retry
                // that cannot run. Returning false would accept this response
                // as final instead of preserving the limit stop.
                return true;
            }
            turn_recovery.dropped_tool_call_nudges_used += 1;
            let nudge = if undecodable_text_call {
                tracing::info!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    call_id = %call_id,
                    malformed_blocks,
                    unterminated_blocks,
                    attempt = turn_recovery.dropped_tool_call_nudges_used,
                    "[agent_loop] nudging after undecodable text-dialect tool call"
                );
                UNDECODABLE_TOOL_CALL_NUDGE
            } else {
                DROPPED_TOOL_CALL_NUDGE
            };
            messages.push(Message::user(nudge));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.dropped_tool_call_nudges_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        false
    }

    /// Appends the dead call's interrupted reasoning to the transcript as a
    /// user message ahead of the retry or nudged call (see
    /// [`dead_call_reasoning_carry`]), announcing it when it does.
    fn carry_dead_call_reasoning(
        &self,
        ctx: &mut RunContext<Ctx>,
        messages: &mut Vec<Message>,
        call_id: &CallId,
        response: &ModelResponse,
    ) {
        if let Some((carry, kept_chars)) =
            dead_call_reasoning_carry(response, self.policy.truncated_empty_carry_reasoning_chars)
        {
            ctx.emit(AgentEvent::ControlApplied {
                control: "truncated_empty_reasoning_carried".to_string(),
                detail: format!(
                    "model call `{call_id}`: {kept_chars} chars of its interrupted reasoning carried into the transcript"
                ),
            });
            messages.push(Message::user(carry));
        }
    }
}

/// The dead call's reasoning, framed for the transcript, when the response
/// carries any and the policy keeps it. The *last* `limit` characters are
/// kept: a derivation's state of play is at its end, and its start is the
/// part a fresh call re-derives fastest. `None` for a response with no
/// reasoning, a limit of zero, or reasoning too short to be worth a message.
/// Returns the framed message and the number of reasoning characters it
/// keeps (the framing excluded).
fn dead_call_reasoning_carry(response: &ModelResponse, limit: usize) -> Option<(String, usize)> {
    const MIN_CARRY_CHARS: usize = 200;
    if limit == 0 {
        return None;
    }
    let reasoning: String = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            tinyinference_llm::message::ContentBlock::Thinking { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let reasoning = reasoning.trim();
    // Both bounds are in characters, as the policy field is documented, not
    // in UTF-8 bytes.
    let total_chars = reasoning.chars().count();
    if total_chars < MIN_CARRY_CHARS {
        return None;
    }
    let tail = if total_chars > limit {
        // Keep the last `limit` characters, then cut on a line boundary where
        // one is near, so the excerpt does not open mid-word.
        let start = reasoning
            .char_indices()
            .nth(total_chars - limit)
            .map_or(0, |(index, _)| index);
        let excerpt = &reasoning[start..];
        match excerpt.find('\n') {
            Some(nl) if nl < 200 => &excerpt[nl + 1..],
            _ => excerpt,
        }
    } else {
        reasoning
    };
    let kept_chars = tail.chars().count();
    Some((
        format!("{TRUNCATED_EMPTY_CARRY_PREFIX}[…]\n{tail}{TRUNCATED_EMPTY_CARRY_SUFFIX}"),
        kept_chars,
    ))
}
