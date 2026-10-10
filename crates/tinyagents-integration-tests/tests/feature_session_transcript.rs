//! End-to-end contract coverage for append-only session transcripts.

use chrono::Local;
use serde_json::json;
use tinyagents_session::transcript::{
    DisplayRecord, FileTranscriptHistory, MessageUsage, TranscriptHistory, TranscriptMessage,
    TranscriptMeta, TranscriptRead, TranscriptToolCall, TurnUsage, append_interrupted_partial,
    append_transcript_turn, find_root_transcript_for_thread, read_thread_usage_summary,
    read_transcript, read_transcript_display,
};

fn meta(turn_count: usize, input_tokens: u64, output_tokens: u64) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "researcher".into(),
        agent_id: Some("researcher-v1".into()),
        agent_type: Some("root".into()),
        dispatcher: "offline-fixture".into(),
        provider: Some("fixture-provider".into()),
        model: Some("fixture-model".into()),
        created: "2026-09-19T10:00:00Z".into(),
        updated: format!("2026-09-19T10:00:0{turn_count}Z"),
        turn_count,
        prefix_message_count: None,
        input_tokens,
        output_tokens,
        cached_input_tokens: 3,
        charged_amount_usd: 0.125,
        thread_id: Some("thread-transcript-e2e".into()),
        task_id: Some("research-task".into()),
    }
}

fn usage(iteration: u32, input: u64, output: u64) -> TurnUsage {
    TurnUsage {
        provider: "fixture-provider".into(),
        model: "fixture-model".into(),
        usage: MessageUsage {
            input,
            output,
            cached_input: 3,
            context_window: 128,
            cost_usd: 0.125,
            ..Default::default()
        },
        ts: format!("2026-09-19T10:00:0{iteration}Z"),
        reasoning_content: Some("checked durable evidence".into()),
        tool_calls: vec![TranscriptToolCall {
            id: "call-search-1".into(),
            name: "search".into(),
            arguments: "{intentionally-not-json}".into(),
            extra_content: Some(json!({"provider": {"opaque": [1, 2]}})),
        }],
        iteration,
    }
}

#[test]
fn append_only_transcript_reopens_with_compacted_context_and_full_display_history() {
    let workspace = tempfile::tempdir().unwrap();
    let stem = "1700000000_researcher";
    let path = workspace
        .path()
        .join("session_raw")
        .join(format!("{stem}.jsonl"));
    let companion_date = Local::now().format("%Y_%m_%d").to_string();

    let initial = vec![
        TranscriptMessage {
            id: Some("system-1".into()),
            role: "system".into(),
            content: "Preserve source provenance.".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
            legacy: Default::default(),
            extra_metadata: Some(json!({"policy_version": 1})),
            cache_breakpoints: vec![8],
            turn_usage: None,
            request_id: None,
            preserve_request_id: false,
            interrupted: false,
            tool_failure: None,
        },
        TranscriptMessage::new("user", "Find the durable transcript contract."),
        TranscriptMessage::assistant("I will search the contract."),
    ];
    append_transcript_turn(
        &path,
        &[],
        &initial,
        &meta(1, 8, 3),
        Some(&usage(1, 8, 3)),
        Some("request-initial"),
    )
    .unwrap();

    let extended = [
        initial.clone(),
        vec![
            TranscriptMessage::new("tool", "search returned transcript docs"),
            TranscriptMessage::assistant("The transcript is append-only."),
        ],
    ]
    .concat();
    append_transcript_turn(
        &path,
        &initial,
        &extended,
        &meta(2, 15, 7),
        Some(&usage(2, 7, 4)),
        Some("request-extension"),
    )
    .unwrap();

    let compacted = vec![
        TranscriptMessage::new("system", "Preserve source provenance."),
        TranscriptMessage::assistant("Summary: transcript persistence is append-only."),
        TranscriptMessage::new("user", "Continue from the compacted context."),
    ];
    append_transcript_turn(
        &path,
        &extended,
        &compacted,
        &meta(3, 21, 9),
        Some(&usage(3, 6, 2)),
        Some("request-compaction"),
    )
    .unwrap();
    append_interrupted_partial(
        &path,
        "This partial must not reach a resumed model.",
        Some("request-interrupted"),
        Some(4),
        Some("stream was cancelled"),
    )
    .unwrap();

    let replayed = read_transcript(&path).unwrap();
    assert_eq!(replayed.meta.turn_count, 3);
    assert_eq!(replayed.meta.input_tokens, 21);
    assert_eq!(replayed.meta.output_tokens, 9);
    assert_eq!(
        replayed
            .messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect::<Vec<_>>(),
        compacted
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect::<Vec<_>>(),
    );
    assert!(
        replayed
            .messages
            .iter()
            .all(|message| !message.content.contains("partial must not"))
    );

    let display = read_transcript_display(&path).unwrap();
    assert_eq!(display.records.len(), 7);
    assert!(matches!(
        &display.records[0],
        DisplayRecord::Message(message)
            if message.message.id.as_deref() == Some("system-1")
                && message.message.extra_metadata == Some(json!({"policy_version": 1}))
                && message.message.cache_breakpoints == [8]
    ));
    assert!(matches!(
        &display.records[2],
        DisplayRecord::Message(message)
            if message.turn_usage.as_ref().is_some_and(|turn| {
                turn.tool_calls[0].arguments == "{intentionally-not-json}"
                    && turn.tool_calls[0].extra_content == Some(json!({"provider": {"opaque": [1, 2]}}))
            })
    ));
    assert!(matches!(
        &display.records[5],
        DisplayRecord::Compaction(marker)
            if marker.request_id.as_deref() == Some("request-compaction")
                && marker.replacement.len() == compacted.len()
    ));
    assert!(matches!(
        &display.records[6],
        DisplayRecord::Message(message)
            if message.interrupted
                && message.request_id.as_deref() == Some("request-interrupted")
                && message.message.content == "This partial must not reach a resumed model."
                && message.reasoning_content.as_deref() == Some("stream was cancelled")
    ));

    let markdown = workspace.path().join("sessions").join(companion_date);
    let markdown = markdown.join(format!("{stem}.md"));
    assert!(
        markdown.is_file(),
        "missing companion: {}",
        markdown.display()
    );
    assert!(
        std::fs::read_to_string(&markdown)
            .unwrap()
            .contains("Summary: transcript persistence is append-only.")
    );

    assert_eq!(
        find_root_transcript_for_thread(workspace.path(), "thread-transcript-e2e"),
        Some(path.clone())
    );
    let summary = read_thread_usage_summary(workspace.path(), "thread-transcript-e2e").unwrap();
    assert_eq!(summary.input_tokens, 21);
    assert_eq!(summary.output_tokens, 9);
    assert_eq!(summary.cached_input_tokens, 3);
    assert_eq!(summary.turn_count, 3);
    assert_eq!(summary.last_turn_input_tokens, 6);
    assert_eq!(summary.last_turn_output_tokens, 2);
    assert_eq!(summary.model.as_deref(), Some("fixture-model"));
    assert!(summary.subagents.is_empty());

    let reopened = FileTranscriptHistory::new(workspace.path(), stem, meta(0, 0, 0)).unwrap();
    assert_eq!(reopened.path(), path);
    let reopened_session = TranscriptRead::read_session(&reopened).unwrap().unwrap();
    assert_eq!(reopened_session.messages, replayed.messages);
    assert_eq!(
        TranscriptHistory::messages(&reopened).unwrap(),
        replayed.messages
    );
}
