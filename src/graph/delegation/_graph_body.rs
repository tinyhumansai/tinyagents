/// Build (but do not run) the delegation `CompiledGraph`. Shared by
/// [`run_delegation`] and [`delegation_graph_topology`] so the graph's structure
/// has one definition.
pub(super) fn build_delegation_graph<F, Fut>(
    max_revisions: usize,
    cancel: CancellationToken,
    require_review_approval: bool,
    run_stage: F,
) -> Result<CompiledGraph<DelegationState, DelegationUpdate>, String>
where
    F: Fn(DelegationStage, DelegationState) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<DelegationStageOutput, String>> + Send + 'static,
{
    let mut builder = GraphBuilder::<DelegationState, DelegationUpdate>::new().set_reducer(
        ClosureStateReducer::new(|mut s: DelegationState, u: DelegationUpdate| {
            match u {
                DelegationUpdate::Plan(p) => s.plan = Some(p),
                DelegationUpdate::Execution { prompt, result } => {
                    let index = s.executions.len();
                    s.executions.push(StepRecord {
                        index,
                        prompt,
                        result,
                    });
                }
                DelegationUpdate::Review { note, approved } => {
                    s.reviews.push(note);
                    s.approved = approved;
                    if !approved {
                        s.revisions += 1;
                    }
                }
                DelegationUpdate::HumanDecision { approved } => {
                    s.human_approved = Some(approved);
                    s.denied = !approved;
                    // A denial overrides the reviewer's in-graph approval: the
                    // human gate is the final authority on whether the result
                    // may finalize.
                    if !approved {
                        s.approved = false;
                    }
                }
                DelegationUpdate::Final(f) => s.final_output = Some(f),
                DelegationUpdate::Cancelled => s.cancelled = true,
            }
            Ok(s)
        }),
    );

    // plan: produce the plan, then route to execute (or finalize if cancelled).
    let run_plan = run_stage.clone();
    let cancel_plan = cancel.clone();
    builder = builder.add_node("plan", move |s: DelegationState, _c: NodeContext| {
        let run_plan = run_plan.clone();
        let cancel = cancel_plan.clone();
        async move {
            if cancel.is_cancelled() {
                return Ok(NodeResult::Command(
                    Command::default()
                        .with_update(DelegationUpdate::Cancelled)
                        .with_goto(["finalize"]),
                ));
            }
            let out = run_plan(DelegationStage::Plan, s)
                .await
                .map_err(to_node_err)?;
            Ok(NodeResult::Command(
                Command::default()
                    .with_update(DelegationUpdate::Plan(out.text))
                    .with_goto(["execute"]),
            ))
        }
    });

    // execute: run the plan; route to review.
    let run_exec = run_stage.clone();
    let cancel_exec = cancel.clone();
    builder = builder.add_node("execute", move |s: DelegationState, _c: NodeContext| {
        let run_exec = run_exec.clone();
        let cancel = cancel_exec.clone();
        async move {
            if cancel.is_cancelled() {
                return Ok(NodeResult::Command(
                    Command::default()
                        .with_update(DelegationUpdate::Cancelled)
                        .with_goto(["finalize"]),
                ));
            }
            let out = run_exec(DelegationStage::Execute, s)
                .await
                .map_err(to_node_err)?;
            Ok(NodeResult::Command(
                Command::default()
                    .with_update(DelegationUpdate::Execution {
                        prompt: out.prompt.unwrap_or_default(),
                        result: out.text,
                    })
                    .with_goto(["review"]),
            ))
        }
    });

    // review: approve (→ finalize) or request a revision (→ execute), bounded by
    // `max_revisions` so a never-approving reviewer still terminates.
    let run_review = run_stage.clone();
    let cancel_review = cancel.clone();
    builder = builder.add_node("review", move |s: DelegationState, _c: NodeContext| {
        let run_review = run_review.clone();
        let cancel = cancel_review.clone();
        async move {
            if cancel.is_cancelled() {
                return Ok(NodeResult::Command(
                    Command::default()
                        .with_update(DelegationUpdate::Cancelled)
                        .with_goto(["finalize"]),
                ));
            }
            let revisions = s.revisions;
            let out = run_review(DelegationStage::Review, s)
                .await
                .map_err(to_node_err)?;
            // Approve when the reviewer is satisfied OR the revision budget is spent.
            let approved = out.approved || revisions >= max_revisions;
            // An approved result routes to the durable human-approval gate when
            // the run is human-gated; otherwise it finalizes directly. A
            // not-approved result always loops back to `execute` for a revision.
            let next = if !approved {
                "execute"
            } else if require_review_approval {
                "approval"
            } else {
                "finalize"
            };
            Ok(NodeResult::Command(
                Command::default()
                    .with_update(DelegationUpdate::Review {
                        note: out.text,
                        approved,
                    })
                    .with_goto([next]),
            ))
        }
    });

    // approval (only when human-gated): a durable human-in-the-loop pause.
    //
    // First entry (`ctx.resume` is `None`): emit `NodeResult::Interrupt`. The
    // executor persists a boundary checkpoint (Sync durability) and returns
    // control to the caller — the pause now survives a process restart. Nothing
    // finalizes until a resume arrives.
    //
    // Re-entry (`ctx.resume` is `Some(decision)`): the approver's decision was
    // delivered via `Command { resume: .. }`. Apply approve/deny and route to
    // `finalize` (deny is honoured there as a blocked/denied result). This is a
    // durability mechanism for the PAUSE only — it grants no new approval
    // authority and never bypasses the security/approval boundary.
    //
    // Durable-vs-chat boundary: this pause is a *checkpointed graph interrupt*,
    // distinct from the interactive chat-turn approval gate (10-min TTL steering
    // pause via `ApprovalRequestCard`), which parks a live in-memory chat turn
    // and is deliberately left untouched.
    if require_review_approval {
        builder = builder.add_node("approval", move |s: DelegationState, ctx: NodeContext| {
            async move {
                match ctx.resume {
                    None => {
                        let payload = json!({
                            "kind": "delegation_review",
                            "reviews": s.reviews,
                            "executions": s.executions_texts(),
                            "revisions": s.revisions,
                        });
                        tracing::info!(
                            revisions = s.revisions,
                            "[interrupt] delegation review reached durable human-approval gate; pausing"
                        );
                        Ok(NodeResult::Interrupt(Interrupt::with_id(
                            "delegation-review-approval",
                            "approval",
                            payload,
                        )))
                    }
                    Some(decision) => {
                        let approved = decision_is_approve(&decision);
                        tracing::info!(
                            approved,
                            "[interrupt] delegation review resumed with human decision"
                        );
                        Ok(NodeResult::Command(
                            Command::default()
                                .with_update(DelegationUpdate::HumanDecision { approved })
                                .with_goto(["finalize"]),
                        ))
                    }
                }
            }
        });
    }

    // finalize: synthesize the final output from the accumulated state, then end.
    builder = builder.add_node(
        "finalize",
        move |s: DelegationState, _c: NodeContext| async move {
            let summary = s
                .executions
                .last()
                .map(|r| r.result.clone())
                .unwrap_or_else(|| "<no execution>".to_string());
            let final_text = if s.cancelled {
                format!("cancelled after {} execution(s)", s.executions.len())
            } else if s.denied {
                format!(
                    "denied by reviewer after {} execution(s)",
                    s.executions.len()
                )
            } else {
                summary
            };
            Ok(NodeResult::Command(
                Command::default()
                    .with_update(DelegationUpdate::Final(final_text))
                    .with_goto([END]),
            ))
        },
    );

    builder = builder
        .set_entry("plan")
        .mark_command_routing("plan")
        .mark_command_routing("execute")
        .mark_command_routing("review")
        .mark_command_routing("finalize");

    if require_review_approval {
        builder = builder
            .mark_command_routing("approval")
            .mark_interrupt("approval");
    }

    let graph = builder
        .compile()
        .map_err(|e| format!("delegation graph compile failed: {e}"))?
        // Bound the execute⇄review loop as a backstop to the in-state counter:
        // each of execute/review may be visited at most max_revisions + 1 times.
        .with_recursion_policy(RecursionPolicy {
            max_visits_per_node: Some(max_revisions + 2),
            max_total_steps: (max_revisions + 1) * 4 + 8,
            ..RecursionPolicy::default()
        })
        // Adapter-first landing of the crate-native per-node RetryPolicy
        // (tinyagents 1.5.0 `CompiledGraph::with_node_retry`). Conservative:
        // `max_attempts(1)` preserves today's single-attempt semantics exactly
        // (no bespoke retry glue existed here) and backoff sleeping stays off
        // (the default), so a transient node-handler failure surfaces as it does
        // today. This wires the seam so raising the attempt cap / enabling
        // backoff is a one-line, gated follow-up rather than a rewrite.
        .with_node_retry(RetryPolicy::default().with_max_attempts(1));

    Ok(graph)
}

/// Structure-only [`GraphTopology`] of the delegation graph for debug /
/// inspection (issue #4249, Phase 4). Built with a no-op stub stage worker —
/// the topology exposes only node names, edges, and routing, never closure
/// bodies.
pub fn delegation_graph_topology() -> Result<GraphTopology, String> {
    let graph = build_delegation_graph(
        DelegationConfig::default().max_revisions,
        CancellationToken::new(),
        // Topology export uses the non-gated shape (the four revision-loop
        // nodes); the durable `approval` interrupt node is additive and only
        // present when a run is human-gated.
        false,
        |_stage, _state| async { Ok(DelegationStageOutput::done("")) },
    )?;
    Ok(graph.topology())
}

/// Map an injected-stage error string into a graph node error.
fn to_node_err(e: String) -> crate::TinyAgentsError {
    crate::TinyAgentsError::Model(e)
}
