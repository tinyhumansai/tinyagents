use super::{
    HostedError, HostedErrorKind, hosted_error, sanitize_hosted_event, sanitize_hosted_stream_item,
};
use crate::agent_loop::AgentStreamItem;
use crate::error::TinyAgentsError;
use crate::events::{AgentEvent, EventRecord};
use crate::ids::{EventId, RunId};
use crate::middleware::AgentRun;
use crate::terminal::{TerminalOutcome, TerminalReason, TimeoutPhase};

const REFUSED: &str = "hosted agent invocation stopped: host policy refused a tool call";

fn refusal() -> TinyAgentsError {
    TinyAgentsError::ToolRefused("tool hook 'allow_list' denied secret: forbidden".into())
}

fn refused_run() -> AgentRun {
    let mut run = AgentRun::new();
    run.terminal = Some(TerminalOutcome::from_error(
        &refusal(),
        TimeoutPhase::AfterTurn,
    ));
    run
}

#[test]
fn tool_refusal_is_classified_as_its_own_terminal_reason() {
    let outcome = TerminalOutcome::from_error(&refusal(), TimeoutPhase::AfterTurn);
    assert_eq!(outcome.reason, TerminalReason::ToolRefused);
    assert_eq!(outcome.class, crate::terminal::TerminalClass::Failure);
}

#[test]
fn hosted_error_names_a_tool_refusal_without_its_detail() {
    let hosted = hosted_error(&refusal(), refused_run());

    assert_eq!(hosted.kind, HostedErrorKind::ToolRefused);
    assert_eq!(hosted.message, REFUSED);
    let outcome = hosted.run.as_ref().and_then(|run| run.terminal.as_ref());
    assert_eq!(
        outcome.map(|outcome| outcome.message.as_str()),
        Some(REFUSED)
    );
    assert_eq!(
        outcome.map(|outcome| outcome.reason),
        Some(TerminalReason::ToolRefused)
    );
}

#[test]
fn hosted_tool_refusal_round_trips_to_its_own_variant() {
    let hosted = HostedError {
        kind: HostedErrorKind::ToolRefused,
        message: REFUSED.to_owned(),
        timeout_bound: None,
        dispatch_refusal: None,
        run: None,
    };
    assert!(matches!(
        TinyAgentsError::from(hosted),
        TinyAgentsError::ToolRefused(message) if message == REFUSED
    ));
}

#[test]
fn tool_refusal_is_not_retryable() {
    assert!(!crate::retry::is_retryable(&refusal()));
}

#[test]
fn hosted_stream_failure_names_a_tool_refusal() {
    let item = sanitize_hosted_stream_item(AgentStreamItem::Failed {
        error: refusal().to_string(),
        run: Box::new(refused_run()),
    });
    let AgentStreamItem::Failed { error, run } = item else {
        panic!("still a failure");
    };
    assert_eq!(error, REFUSED);
    assert_eq!(
        run.terminal
            .as_ref()
            .map(|outcome| outcome.message.as_str()),
        Some(REFUSED)
    );
}

#[test]
fn hosted_stream_failure_stays_generic_for_other_errors() {
    let mut run = AgentRun::new();
    run.terminal = Some(TerminalOutcome::from_error(
        &TinyAgentsError::Tool("boom".into()),
        TimeoutPhase::AfterTurn,
    ));
    let item = sanitize_hosted_stream_item(AgentStreamItem::Failed {
        error: "tool error: boom".into(),
        run: Box::new(run),
    });
    assert!(matches!(
        item,
        AgentStreamItem::Failed { error, .. } if error == "hosted agent invocation failed"
    ));
}

#[test]
fn hosted_run_failed_event_names_a_tool_refusal() {
    let error = refusal();
    let mut record = EventRecord {
        id: EventId::new("evt-0"),
        offset: 0,
        event: AgentEvent::RunFailed {
            run_id: RunId::new("run"),
            error: error.to_string(),
            outcome: Some(TerminalOutcome::from_error(&error, TimeoutPhase::AfterTurn)),
        },
    };
    sanitize_hosted_event(&mut record);
    let AgentEvent::RunFailed { error, outcome, .. } = &record.event else {
        panic!("still RunFailed");
    };
    assert_eq!(error, REFUSED);
    assert_eq!(
        outcome.as_ref().map(|outcome| outcome.message.as_str()),
        Some(REFUSED)
    );
}
