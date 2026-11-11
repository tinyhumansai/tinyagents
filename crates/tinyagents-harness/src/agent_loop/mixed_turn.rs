//! The mixed structured-output turn: one response that both answers (the
//! structured-output schema call) and requests further tools.
//!
//! Split out of `run_loop.rs`. `RunPolicy::end_strategy` (A6) decides what
//! happens to the two, through three named, documented outcomes
//! ([`EndStrategy`]).

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Settles a mixed turn per `RunPolicy::end_strategy`.
    ///
    /// `Early` finishes immediately and skips the tools; `Graceful` records the
    /// answer, runs the tools, then finishes; `Exhaustive` (and a turn whose
    /// call was cut off) ignores the structured call and runs another turn.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn finish_mixed_structured_turn(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        turn_recovery: &mut TurnRecovery,
        promoted_names: &mut std::collections::BTreeSet<String>,
        mixed: MixedStructuredTurn<'_>,
    ) -> Result<TurnFlow> {
        let MixedStructuredTurn {
            response,
            structured_plan,
            structured_hits,
            real_tool_calls,
            turn_had_truncated_calls,
        } = mixed;
        // A6: one turn asked to both answer (the structured-output
        // schema call) and run further tools. `RunPolicy::end_strategy`
        // decides what happens to the two, replacing the old
        // ad-hoc "record and keep going" behavior with three named,
        // documented outcomes (`EndStrategy`).
        let record = ctx.emit(AgentEvent::ControlApplied {
            control: "structured_with_tool_calls".to_string(),
            detail: format!(
                "{:?} end_strategy handling {} real tool call(s) alongside a \
                         structured-output call",
                self.policy.end_strategy,
                real_tool_calls.len()
            ),
        });
        status.set_last_event(record.id);

        if matches!(self.policy.end_strategy, EndStrategy::Early) {
            // Finish immediately: the structured answer wins outright,
            // and the accompanying tool calls never run. Every
            // requested `tool_call_id` — structured hits and the
            // skipped real calls alike — still needs an answer or the
            // transcript is malformed for a future replay.
            if let Some((strategy, name, schema)) = &structured_plan {
                let extractor = self.build_structured_extractor(strategy, name, schema);
                match extractor.extract(&response) {
                    Ok(output) => {
                        run.structured = Some(output.value);
                        run.structured_variant = output.variant;
                    }
                    Err(error) => tracing::debug!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        %error,
                        "[agent_loop] structured extraction failed on a mixed turn \
                         under EndStrategy::Early"
                    ),
                }
            }
            for call in &structured_hits {
                messages.push(Message::tool(
                    call.id.clone(),
                    "Structured output recorded.",
                ));
            }
            for call in &real_tool_calls {
                messages.push(Message::tool(
                    call.id.clone(),
                    "run stopped before this tool call was executed \
                             (EndStrategy::Early: the structured answer ends the run first)",
                ));
            }
            run.final_response = Some(response);
            ctx.close_turn(self.policy.capture, messages);
            return self.settle_natural_finish(ctx, status, messages).await;
        }

        // A real call that a length stop cut off was answered with a
        // "re-issue it" error, so finishing now would silently drop
        // the action the model asked for. Such a turn takes the
        // `Exhaustive` path below (answer not recorded, tools run,
        // another turn) whatever the strategy; `Early` still wins
        // outright because it discards the real calls by contract.
        if matches!(self.policy.end_strategy, EndStrategy::Graceful) && !turn_had_truncated_calls {
            // Record the answer now (it will not be asked for again),
            // but let the requested tools actually run before ending
            // the run — their side effects and results are not
            // silently dropped, unlike `Early`.
            if let Some((strategy, name, schema)) = &structured_plan {
                let extractor = self.build_structured_extractor(strategy, name, schema);
                match extractor.extract(&response) {
                    Ok(output) => {
                        run.structured = Some(output.value);
                        run.structured_variant = output.variant;
                    }
                    Err(error) => tracing::debug!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        %error,
                        "[agent_loop] structured extraction failed on a mixed turn \
                         under EndStrategy::Graceful"
                    ),
                }
            }
            for call in &structured_hits {
                messages.push(Message::tool(
                    call.id.clone(),
                    "Structured output recorded.",
                ));
            }
            status.mark_running(HarnessPhase::Tools);
            let deferred = self
                .execute_tools_with_promotions(
                    state,
                    ctx,
                    run,
                    status,
                    messages,
                    real_tool_calls,
                    promoted_names,
                )
                .await?;
            if let Some(exit) = self
                .settle_deferred(state, ctx, run, status, messages, deferred)
                .await?
            {
                return Ok(TurnFlow::Exit(exit));
            }
            ctx.close_turn(self.policy.capture, messages);
            // The tools ran, so this is a resolved tool turn: a later
            // dropped-call or truncation turn starts with its full budget.
            turn_recovery.reset_after_tool_turn(false);
            match self.apply_pending_control(ctx, run, status, messages)? {
                ControlEffect::None => {}
                ControlEffect::ContinueLoop => return Ok(TurnFlow::NextTurn),
                ControlEffect::Exit(exit) => return Ok(TurnFlow::Exit(exit)),
            }
            run.final_response = Some(response);
            return self.settle_natural_finish(ctx, status, messages).await;
        }

        // `EndStrategy::Exhaustive`: the output tool this turn is
        // ignored outright (never recorded) — the run keeps going
        // exactly as if only the real tool calls had been requested.
        // It only finishes once a later turn's output-tool call has
        // no accompanying function-tool calls.
        debug_assert!(
            matches!(self.policy.end_strategy, EndStrategy::Exhaustive) || turn_had_truncated_calls
        );
        let not_final_note = if matches!(self.policy.end_strategy, EndStrategy::Exhaustive) {
            "Structured output noted but not final yet; finish the remaining tool \
                     calls first (EndStrategy::Exhaustive)."
        } else {
            "Structured output noted but not final yet; a tool call in this turn was \
                     cut off by the output token limit, so re-issue it and answer again."
        };
        for call in &structured_hits {
            messages.push(Message::tool(call.id.clone(), not_final_note));
        }

        // A mixed turn (structured payload alongside real tool calls)
        // is a resolved turn exactly like an ordinary tool-calling one
        // (see the reset below at the non-mixed path): it must not
        // leave a spent `dropped_tool_call_nudges_used` counter to
        // leak into a later, unrelated dropped-call turn, which would
        // otherwise receive fewer than the policy's configured number
        // of consecutive re-prompts.
        // A turn whose call was cut off keeps its retry budget and
        // boosted output cap for the retry.
        turn_recovery.reset_after_tool_turn(turn_had_truncated_calls);

        status.mark_running(HarnessPhase::Tools);
        let deferred = self
            .execute_tools_with_promotions(
                state,
                ctx,
                run,
                status,
                messages,
                real_tool_calls,
                promoted_names,
            )
            .await?;
        if let Some(exit) = self
            .settle_deferred(state, ctx, run, status, messages, deferred)
            .await?
        {
            return Ok(TurnFlow::Exit(exit));
        }

        // Close the mixed turn before queued messages are drained, so they are
        // announced after its `TurnCompleted` like on the plain tool path.
        ctx.close_turn(self.policy.capture, messages);

        // Turn boundary (A4): same steer drain as the plain tool path.
        self.apply_queued_lane(ctx, status, messages, crate::run_queue::QueueLane::Steer)
            .await;

        // Safe checkpoint: a control requested from `after_tool` /
        // `wrap_tool` is honored here, at the edge it was raised on.
        match self.apply_pending_control(ctx, run, status, messages)? {
            ControlEffect::None => {}
            ControlEffect::ContinueLoop => return Ok(TurnFlow::NextTurn),
            ControlEffect::Exit(exit) => return Ok(TurnFlow::Exit(exit)),
        }
        Ok(TurnFlow::NextTurn)
    }
}
