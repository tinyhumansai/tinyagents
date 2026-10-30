use super::*;

#[test]
fn in_memory_transcript_starts_empty() {
    let locator = in_memory_transcript();
    let meta = crate::transcript::TranscriptMeta {
        agent_name: "fixture".into(),
        agent_id: None,
        agent_type: None,
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
        session_id: None,
        parent_session_id: None,
    };

    let transcript = locator.open_stem("fixture", meta).expect("bind transcript");
    assert!(
        transcript
            .read_session()
            .expect("read empty transcript")
            .is_none()
    );
}

#[test]
fn in_memory_session_adopts_imported_history_only_while_empty() {
    let locator = in_memory_transcript();
    let session = crate::transcript::SessionRef::scoped("import", "planner");
    let seed = crate::transcript::TranscriptMeta {
        agent_name: "planner".into(),
        agent_id: None,
        agent_type: None,
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("import".into()),
        task_id: None,
        session_id: Some(session.session_id()),
        parent_session_id: None,
    };
    let imported = vec![crate::transcript::TranscriptMessage::user("old history")];

    assert!(
        locator
            .adopt_messages_if_absent(&session, seed.clone(), &imported)
            .expect("first adoption succeeds")
    );
    assert!(
        !locator
            .adopt_messages_if_absent(
                &session,
                seed,
                &[crate::transcript::TranscriptMessage::user("new turn")],
            )
            .expect("later adoption is ignored")
    );
    let transcript = locator
        .read_session_transcript(&session)
        .expect("adopted session exists")
        .read_session()
        .expect("read session")
        .expect("session was written");
    assert_eq!(transcript.messages, imported);
}
