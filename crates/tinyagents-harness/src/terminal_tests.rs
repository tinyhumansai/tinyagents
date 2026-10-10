use super::*;
use crate::steering::PauseState;
use crate::tool::DeferredToolRequests;

fn out(reason: TerminalReason) -> TerminalOutcome {
    TerminalOutcome::new(reason, "m")
}

#[test]
fn class_follows_reason() {
    use TerminalReason::*;
    assert_eq!(Completed.class(), TerminalClass::Success);
    assert_eq!(Cancelled.class(), TerminalClass::Cancellation);
    assert_eq!(Timeout.class(), TerminalClass::Timeout);
    assert_eq!(
        ProviderFailed(Some(FailoverReason::Timeout)).class(),
        TerminalClass::Timeout
    );
    assert_eq!(
        ProviderFailed(Some(FailoverReason::RateLimit)).class(),
        TerminalClass::Failure
    );
    assert_eq!(ProviderFailed(None).class(), TerminalClass::Failure);
    assert_eq!(Paused.class(), TerminalClass::Suspended);
    assert_eq!(Deferred.class(), TerminalClass::Suspended);
    for failure in [LimitReached(None), Halted, ToolFailed, Internal] {
        assert_eq!(failure.class(), TerminalClass::Failure, "{failure:?}");
    }
    assert_eq!(out(Halted).class, TerminalClass::Failure);
}

#[test]
fn cancelled_error_maps_to_cancellation() {
    let o = TerminalOutcome::from_error(&TinyAgentsError::Cancelled, TimeoutPhase::AfterTurn);
    assert_eq!(o.reason, TerminalReason::Cancelled);
    assert_eq!(o.class, TerminalClass::Cancellation);
    assert_eq!(o.timeout_phase, None);
    assert!(o.provider_started);
    assert_eq!(o.message, "run cancelled");
}

#[test]
fn run_deadline_carries_phase_and_provider_started() {
    let error = TinyAgentsError::Timeout("deadline".into());
    let before = TerminalOutcome::from_error(&error, TimeoutPhase::BeforeProvider);
    assert_eq!(before.reason, TerminalReason::Timeout);
    assert_eq!(before.class, TerminalClass::Timeout);
    assert_eq!(before.timeout_phase, Some(TimeoutPhase::BeforeProvider));
    assert!(!before.provider_started);

    let during = TerminalOutcome::from_error(&error, TimeoutPhase::Provider);
    assert_eq!(during.timeout_phase, Some(TimeoutPhase::Provider));
    assert!(during.provider_started);

    let after = TerminalOutcome::from_error(&error, TimeoutPhase::AfterTurn);
    assert_eq!(after.timeout_phase, Some(TimeoutPhase::AfterTurn));
    assert!(after.provider_started);
}

#[test]
fn call_timeout_is_a_provider_timeout() {
    let o = TerminalOutcome::from_error(
        &TinyAgentsError::CallTimeout("wedged".into()),
        TimeoutPhase::Provider,
    );
    assert_eq!(
        o.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::Timeout))
    );
    assert_eq!(o.class, TerminalClass::Timeout);
    assert_eq!(o.timeout_phase, Some(TimeoutPhase::Provider));
    assert!(
        o.provider_started,
        "a call timeout inside a call means it started"
    );
}

#[test]
fn rate_limit_and_stream_idle_errors_keep_provider_classification() {
    let rate_limited = TerminalOutcome::from_error(
        &TinyAgentsError::RateLimited("bucket empty".into()),
        TimeoutPhase::Provider,
    );
    assert_eq!(
        rate_limited.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::RateLimit))
    );
    assert_eq!(rate_limited.class, TerminalClass::Failure);

    let idle = TerminalOutcome::from_error(
        &TinyAgentsError::StreamIdleTimeout("provider stalled".into()),
        TimeoutPhase::Provider,
    );
    assert_eq!(
        idle.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::StreamIdleTimeout))
    );
    assert_eq!(idle.class, TerminalClass::Timeout);
    assert_eq!(idle.timeout_phase, Some(TimeoutPhase::Provider));
}

#[test]
fn provider_errors_carry_the_failover_reason() {
    let o = TerminalOutcome::from_error(
        &TinyAgentsError::Model("HTTP 429 too many requests".into()),
        TimeoutPhase::Provider,
    );
    assert_eq!(
        o.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::RateLimit))
    );
    assert_eq!(o.class, TerminalClass::Failure);
    assert_eq!(o.timeout_phase, None);

    let empty =
        TerminalOutcome::from_error(&TinyAgentsError::EmptyResponse, TimeoutPhase::Provider);
    assert_eq!(
        empty.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::EmptyResponse))
    );
}

#[test]
fn tool_limit_and_internal_errors_map() {
    let site = TimeoutPhase::AfterTurn;
    let tool = TerminalOutcome::from_error(&TinyAgentsError::ToolFailed("x".into()), site);
    assert_eq!(tool.reason, TerminalReason::ToolFailed);
    let limit = TerminalOutcome::from_error(&TinyAgentsError::LimitExceeded("x".into()), site);
    assert_eq!(limit.reason, TerminalReason::LimitReached(None));
    let depth = TerminalOutcome::from_error(&TinyAgentsError::SubAgentDepth(3), site);
    assert_eq!(depth.reason, TerminalReason::LimitReached(None));
    let internal = TerminalOutcome::from_error(&TinyAgentsError::Middleware("x".into()), site);
    assert_eq!(internal.reason, TerminalReason::Internal);
    assert_eq!(internal.message, "middleware error: x");
}

#[test]
fn deferral_and_interrupt_errors_are_suspended() {
    let site = TimeoutPhase::AfterTurn;
    let deferred = TerminalOutcome::from_error(
        &TinyAgentsError::ApprovalRequired {
            metadata: serde_json::Value::Null,
        },
        site,
    );
    assert_eq!(deferred.class, TerminalClass::Suspended);
    let paused = TerminalOutcome::from_error(
        &TinyAgentsError::Interrupted {
            node: "steering-pause".into(),
            message: "m".into(),
        },
        site,
    );
    assert_eq!(paused.reason, TerminalReason::Paused);
    let other = TerminalOutcome::from_error(
        &TinyAgentsError::Interrupted {
            node: "n".into(),
            message: "m".into(),
        },
        site,
    );
    assert_eq!(other.reason, TerminalReason::Internal);
}

#[test]
fn loop_exits_map() {
    let finished = TerminalOutcome::from_loop_exit(&LoopExit::Finished, true);
    assert_eq!(finished.reason, TerminalReason::Completed);
    assert_eq!(finished.class, TerminalClass::Success);
    assert!(finished.provider_started);

    let limit = TerminalOutcome::from_loop_exit(&LoopExit::LimitStop(LimitKind::ToolCalls), true);
    assert_eq!(
        limit.reason,
        TerminalReason::LimitReached(Some(LimitKind::ToolCalls))
    );
    assert!(limit.message.contains("tool_calls"));

    let wall = TerminalOutcome::from_loop_exit(&LoopExit::LimitStop(LimitKind::WallClock), true);
    assert_eq!(
        wall.reason,
        TerminalReason::Timeout,
        "wall clock is the run deadline"
    );
    assert_eq!(wall.class, TerminalClass::Timeout);

    let paused = TerminalOutcome::from_loop_exit(
        &LoopExit::Paused(PauseState {
            reason: Some("operator".into()),
            paused_at_checkpoint: 2,
        }),
        false,
    );
    assert_eq!(paused.reason, TerminalReason::Paused);
    assert_eq!(paused.message, "operator");
    assert!(!paused.provider_started);

    let deferred =
        TerminalOutcome::from_loop_exit(&LoopExit::Deferred(DeferredToolRequests::default()), true);
    assert_eq!(deferred.reason, TerminalReason::Deferred);
    assert_eq!(deferred.class, TerminalClass::Suspended);
}

#[test]
fn merge_follows_the_documented_precedence() {
    use TerminalReason::*;
    // Highest first.
    let order = [
        Cancelled,
        Timeout,
        ProviderFailed(Some(FailoverReason::Timeout)),
        LimitReached(Some(LimitKind::ModelCalls)),
        Halted,
        ProviderFailed(Some(FailoverReason::RateLimit)),
        Paused,
        Completed,
    ];
    for (i, high) in order.iter().enumerate() {
        for low in &order[i + 1..] {
            let a = out(*high).merge(out(*low));
            let b = out(*low).merge(out(*high));
            assert_eq!(a.reason, *high, "{high:?} over {low:?}");
            assert_eq!(b.reason, *high, "{high:?} over {low:?} (swapped)");
        }
    }
    // Failures share a rank.
    for failure in [ProviderFailed(None), ToolFailed, Internal] {
        assert_eq!(out(Halted).merge(out(failure)).reason, Halted);
    }
}

#[test]
fn merge_ties_keep_the_earlier_and_or_provider_started() {
    let first = TerminalOutcome::new(TerminalReason::ToolFailed, "first");
    let second =
        TerminalOutcome::new(TerminalReason::Internal, "second").with_provider_started(true);
    let merged = first.merge(second);
    assert_eq!(merged.message, "first");
    assert!(merged.provider_started, "provider_started is OR-ed");

    let cancel = TerminalOutcome::new(TerminalReason::Cancelled, "c");
    let merged = cancel
        .merge(TerminalOutcome::new(TerminalReason::Completed, "").with_provider_started(true));
    assert_eq!(merged.reason, TerminalReason::Cancelled);
    assert!(merged.provider_started);
}

#[test]
fn serde_round_trip_and_wire_shape() {
    let o = TerminalOutcome::from_error(
        &TinyAgentsError::Timeout("t".into()),
        TimeoutPhase::BeforeProvider,
    );
    let json = serde_json::to_value(&o).unwrap();
    assert_eq!(json["reason"], "timeout");
    assert_eq!(json["class"], "timeout");
    assert_eq!(json["timeout_phase"], "before_provider");
    assert_eq!(json["provider_started"], false);
    assert_eq!(serde_json::from_value::<TerminalOutcome>(json).unwrap(), o);

    let limit = TerminalOutcome::limit_reached(Some(LimitKind::ModelCalls), "cap");
    let json = serde_json::to_value(&limit).unwrap();
    assert_eq!(
        json["reason"],
        serde_json::json!({"limit_reached": "model_calls"})
    );
    assert!(json.get("timeout_phase").is_none());
    assert_eq!(
        serde_json::from_value::<TerminalOutcome>(json).unwrap(),
        limit
    );

    let provider = out(TerminalReason::ProviderFailed(Some(
        FailoverReason::RateLimit,
    )));
    let json = serde_json::to_value(&provider).unwrap();
    assert_eq!(
        json["reason"],
        serde_json::json!({"provider_failed": "rate_limit"})
    );
}

#[test]
fn a_call_timeout_before_dispatch_keeps_the_pre_provider_phase() {
    let error = TinyAgentsError::CallTimeout("resolver".into());
    let before = TerminalOutcome::from_error(&error, TimeoutPhase::BeforeProvider);
    assert_eq!(before.timeout_phase, Some(TimeoutPhase::BeforeProvider));
    assert!(!before.provider_started);
    let during = TerminalOutcome::from_error(&error, TimeoutPhase::Provider);
    assert_eq!(during.timeout_phase, Some(TimeoutPhase::Provider));
    assert!(during.provider_started);
    let after = TerminalOutcome::from_error(&error, TimeoutPhase::AfterTurn);
    assert_eq!(after.timeout_phase, Some(TimeoutPhase::AfterTurn));
}

#[test]
fn a_summarizer_failure_is_classified_from_its_inner_error() {
    let error = TinyAgentsError::SummarizationUsage {
        error: Box::new(TinyAgentsError::Model("HTTP 429 too many requests".into())),
        usage: Default::default(),
    };
    let o = TerminalOutcome::from_error(&error, TimeoutPhase::AfterTurn);
    assert_eq!(
        o.reason,
        TerminalReason::ProviderFailed(Some(FailoverReason::RateLimit))
    );
}
