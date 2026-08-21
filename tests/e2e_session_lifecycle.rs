//! End-to-end coverage for a host-visible session lifecycle.
#![cfg(feature = "sqlite")]

use chrono::{Duration, Utc};
use tinyagents::session::{self, SessionSearchParams, SessionStatus};

#[test]
fn parent_and_child_sessions_are_recorded_searched_completed_and_retained() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    session::record_session_start(
        workspace,
        "parent",
        "orchestrator",
        "Orchestrator",
        "parent-key",
        None,
        Some("thread"),
        Some("test"),
        Some("mock"),
        None,
    )
    .unwrap();
    session::record_session_start(
        workspace,
        "child",
        "researcher",
        "Researcher",
        "child-key",
        Some("parent"),
        Some("thread"),
        Some("test"),
        Some("mock"),
        None,
    )
    .unwrap();
    let message_id = session::record_message(
        workspace,
        "child",
        "assistant",
        "the answer contains cobalt",
        Some("mock"),
        Some(8),
        Some(5),
        Some(0.01),
    )
    .unwrap();
    session::record_tool_call(
        workspace,
        "child",
        Some(message_id),
        "search",
        Some(r#"{"q":"cobalt"}"#),
        Some("one result"),
        "completed",
        Some(2),
    )
    .unwrap();

    let found = session::search_sessions(
        workspace,
        &SessionSearchParams {
            query: Some("cobalt".into()),
            tool_name: Some("search".into()),
            parent_session_id: Some("parent".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(found.total, 1);
    assert_eq!(found.sessions[0].id, "child");
    assert_eq!(
        session::list_children(workspace, "parent").unwrap().len(),
        1
    );

    for id in ["child", "parent"] {
        session::record_session_end(workspace, id, SessionStatus::Completed, 1, 8, 5, 0, 0.01)
            .unwrap();
    }
    let completed = session::list_sessions(workspace, None, None, Some("completed"), None).unwrap();
    assert_eq!(completed.total, 2);

    let report = session::apply_retention(workspace, Utc::now() + Duration::seconds(1)).unwrap();
    assert_eq!(report.sessions, 2);
    assert_eq!(report.total(), 2);
    assert_eq!(
        session::list_sessions(workspace, None, None, None, None)
            .unwrap()
            .total,
        0
    );
}
