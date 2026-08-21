//! Feature-level coverage for the SQLite session retention surface.
#![cfg(feature = "sqlite")]

use chrono::{Duration, Utc};
use serde_json::json;
use tinyagents::session;
use tinyagents::session::run_ledger::{
    self,
    types::{RunEventAppend, RunTelemetryUpsert},
};

fn start_session(workspace: &std::path::Path, id: &str) {
    session::record_session_start(
        workspace, id, "agent", "Agent", id, None, None, None, None, None,
    )
    .unwrap();
}

#[test]
fn trimming_keeps_the_newest_messages_and_reports_removed_rows() {
    let dir = tempfile::tempdir().unwrap();
    start_session(dir.path(), "trim");
    for content in ["oldest", "middle", "newest"] {
        session::record_message(dir.path(), "trim", "user", content, None, None, None, None)
            .unwrap();
    }

    assert_eq!(
        session::trim_session_messages(dir.path(), "trim", 2).unwrap(),
        1
    );
    let messages = session::list_messages(dir.path(), "trim", None).unwrap();
    assert_eq!(
        messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        ["middle", "newest"]
    );
    assert_eq!(
        session::trim_session_messages(dir.path(), "trim", 0).unwrap(),
        2
    );
    assert!(
        session::list_messages(dir.path(), "trim", None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn individual_age_pruners_remove_only_eligible_rows() {
    let dir = tempfile::tempdir().unwrap();
    start_session(dir.path(), "records");
    session::record_tool_call(
        dir.path(),
        "records",
        None,
        "shell",
        None,
        None,
        "completed",
        None,
    )
    .unwrap();
    run_ledger::append_run_event(
        dir.path(),
        RunEventAppend {
            run_id: "run".into(),
            event_type: "started".into(),
            payload: json!({}),
        },
    )
    .unwrap();
    run_ledger::upsert_run_telemetry(
        dir.path(),
        RunTelemetryUpsert {
            run_id: "run".into(),
            input_tokens: Some(3),
            ..Default::default()
        },
    )
    .unwrap();

    let future = Utc::now() + Duration::seconds(1);
    assert_eq!(
        session::prune_tool_calls_before(dir.path(), future).unwrap(),
        1
    );
    assert_eq!(
        session::prune_run_events_before(dir.path(), future).unwrap(),
        1
    );
    assert_eq!(
        session::prune_run_telemetry_before(dir.path(), future).unwrap(),
        1
    );
    assert!(
        session::list_tool_calls(dir.path(), "records", None)
            .unwrap()
            .is_empty()
    );

    let past = Utc::now() - Duration::days(1);
    assert_eq!(
        session::prune_tool_calls_before(dir.path(), past).unwrap(),
        0
    );
    assert_eq!(
        session::prune_run_events_before(dir.path(), past).unwrap(),
        0
    );
    assert_eq!(
        session::prune_run_telemetry_before(dir.path(), past).unwrap(),
        0
    );
}

#[test]
fn session_pruner_preserves_running_and_recent_sessions() {
    let dir = tempfile::tempdir().unwrap();
    start_session(dir.path(), "running");
    start_session(dir.path(), "finished");
    session::record_session_end(
        dir.path(),
        "finished",
        session::SessionStatus::Completed,
        1,
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    let past = Utc::now() - Duration::days(1);
    assert_eq!(session::prune_sessions_before(dir.path(), past).unwrap(), 0);
    let future = Utc::now() + Duration::seconds(1);
    assert_eq!(
        session::prune_sessions_before(dir.path(), future).unwrap(),
        1
    );
    assert!(session::get_session(dir.path(), "finished").is_err());
    assert_eq!(
        session::get_session(dir.path(), "running").unwrap().status,
        session::SessionStatus::Running
    );
}

#[test]
fn retention_report_total_counts_every_category() {
    let report = session::RetentionReport {
        sessions: 1,
        messages: 2,
        tool_calls: 3,
        run_events: 4,
        run_telemetry: 5,
    };
    assert_eq!(report.total(), 15);
    assert_eq!(
        serde_json::from_str::<session::RetentionReport>(&serde_json::to_string(&report).unwrap())
            .unwrap(),
        report
    );
}
