//! Module-local unit tests for [`crate::transcript`]: full-rewrite and
//! append-only writers, model-context and display readers, compaction replay,
//! interrupted partials, path resolution/resume, thread-usage summaries, and
//! the [`super::history`] locator/handle seam.
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory.

use super::*;
use tempfile::tempdir;

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent-id".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: Some("provider".into()),
        model: Some("model".into()),
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 1,
        output_tokens: 2,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

#[test]
fn frozen_prefix_count_round_trips_in_meta() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "prefix-count").unwrap();
    let mut seed = meta();
    seed.prefix_message_count = Some(2);
    write_transcript(
        &path,
        &[TranscriptMessage::new("system", "stable")],
        &seed,
        None,
    )
    .unwrap();

    assert_eq!(
        read_transcript(&path).unwrap().meta.prefix_message_count,
        Some(2)
    );
}

#[test]
fn intentional_prefix_reset_is_distinct_from_legacy_missing_boundary() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "prefix-reset").unwrap();
    let mut seed = meta();
    seed.prefix_message_count = None;
    seed.prefix_reset = true;
    write_transcript(&path, &[], &seed, None).unwrap();

    let restored = read_transcript(&path).unwrap().meta;
    assert!(restored.prefix_reset);
    assert_eq!(restored.prefix_message_count, None);
}

#[test]
fn jsonl_round_trip_keeps_raw_tool_arguments_and_provider_extension() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let messages = vec![
        TranscriptMessage::new("user", "use a tool"),
        TranscriptMessage::assistant("working"),
    ];
    let usage = TurnUsage {
        provider: "provider".into(),
        model: "model".into(),
        usage: MessageUsage {
            input: 1,
            output: 2,
            cached_input: 0,
            context_window: 3,
            cost_usd: 0.0,
            ..Default::default()
        },
        ts: "2026-01-01T00:00:00Z".into(),
        reasoning_content: Some("reasoning".into()),
        tool_calls: vec![TranscriptToolCall {
            id: "call-1".into(),
            name: "tool".into(),
            arguments: "{not-json}".into(),
            extra_content: Some(serde_json::json!({"google": {"thought_signature": "opaque"}})),
        }],
        iteration: 1,
    };

    write_transcript(&path, &messages, &meta(), Some(&usage)).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert_eq!(loaded.messages[0].role, messages[0].role);
    assert_eq!(loaded.messages[0].content, messages[0].content);
    assert!(loaded.messages[0].preserve_request_id);
    let restored_usage = loaded.messages[1].turn_usage.as_ref().unwrap();
    assert_eq!(restored_usage.tool_calls[0].arguments, "{not-json}");
    assert_eq!(
        restored_usage.tool_calls[0].extra_content,
        Some(serde_json::json!({"google": {"thought_signature": "opaque"}}))
    );
    let raw = std::fs::read_to_string(path).unwrap();
    assert!(raw.contains("{not-json}"));
    assert!(raw.contains("thought_signature"));
}

#[test]
fn unknown_jsonl_records_are_ignored() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let header = serde_json::json!({"_meta": {"agent": "agent", "dispatcher": "native", "created": "", "updated": "", "turn_count": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "charged_amount_usd": 0.0}});
    std::fs::write(&path, format!("{header}\n{{\"kind\":\"future\",\"payload\":true}}\n{{\"role\":\"user\",\"content\":\"hello\"}}\n")).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert!(
        !loaded.meta.prefix_reset,
        "old metadata defaults to no reset"
    );
    assert_eq!(loaded.meta.prefix_message_count, None);
    assert_eq!(
        loaded.messages,
        vec![TranscriptMessage {
            preserve_request_id: true,
            ..TranscriptMessage::new("user", "hello")
        }]
    );
}

#[test]
fn malformed_required_message_record_is_skipped_without_losing_valid_rows() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let header = serde_json::json!({"_meta": {"agent": "agent", "dispatcher": "native", "created": "", "updated": "", "turn_count": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "charged_amount_usd": 0.0}});
    std::fs::write(&path, format!("{header}\n{{\"role\":\"user\"}}\n")).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert!(loaded.messages.is_empty());
}

#[test]
fn a_torn_append_does_not_swallow_the_next_turn() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "torn-tail").unwrap();
    let first = vec![TranscriptMessage::new("user", "before crash")];
    append_transcript_turn(&path, &[], &first, &meta(), None, Some("first")).unwrap();

    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"role\":\"assistant\",\"content\":\"cut")
        .unwrap();

    // Both readers skip malformed JSON records. The next append must start
    // on a separate line so only this cut record is skipped.
    let next = [
        first.clone(),
        vec![TranscriptMessage::new("user", "after crash")],
    ]
    .concat();
    append_transcript_turn(&path, &first, &next, &meta(), None, Some("second")).unwrap();

    let model = read_transcript(&path).unwrap();
    assert_eq!(model.messages.len(), 2);
    assert_eq!(model.messages[0].content, "before crash");
    assert_eq!(model.messages[1].content, "after crash");
    let display = read_transcript_display(&path).unwrap();
    assert_eq!(display.records.len(), 2);
}

#[test]
fn an_invalid_utf8_tail_does_not_block_cold_resume() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "cut-utf8").unwrap();
    let first = vec![TranscriptMessage::new("user", "before crash")];
    append_transcript_turn(&path, &[], &first, &meta(), None, Some("first")).unwrap();

    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[b'{', 0xc3])
        .unwrap();

    let next = [
        first.clone(),
        vec![TranscriptMessage::new("user", "after crash")],
    ]
    .concat();
    append_transcript_turn(&path, &first, &next, &meta(), None, Some("second")).unwrap();

    let model = read_transcript(&path).unwrap();
    assert_eq!(model.messages.len(), 2);
    assert_eq!(model.messages[1].content, "after crash");
    assert_eq!(read_transcript_display(&path).unwrap().records.len(), 2);
}

#[test]
fn append_replays_delta_compaction_request_ids_and_interrupted_partials() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let first = vec![
        TranscriptMessage::new("system", "system"),
        TranscriptMessage::new("user", "first"),
        TranscriptMessage::assistant("answer"),
    ];
    append_transcript_turn(&path, &[], &first, &meta(), None, Some("request-a")).unwrap();
    let extended = [
        first.clone(),
        vec![TranscriptMessage::new("user", "second")],
    ]
    .concat();
    append_transcript_turn(&path, &first, &extended, &meta(), None, Some("request-b")).unwrap();
    let compacted = vec![
        TranscriptMessage::new("system", "system"),
        TranscriptMessage::assistant("summary"),
        TranscriptMessage::new("user", "second"),
    ];
    append_transcript_turn(
        &path,
        &extended,
        &compacted,
        &meta(),
        None,
        Some("request-c"),
    )
    .unwrap();
    append_interrupted_partial(
        &path,
        "partial answer",
        Some("request-c"),
        Some(1),
        Some("partial reasoning"),
    )
    .unwrap();

    let replayed = read_transcript(&path).unwrap();
    assert_eq!(
        replayed
            .messages
            .iter()
            .map(|message| (&message.role, &message.content))
            .collect::<Vec<_>>(),
        compacted
            .iter()
            .map(|message| (&message.role, &message.content))
            .collect::<Vec<_>>()
    );
    let display = read_transcript_display(&path).unwrap();
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Compaction(_)))
    );
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Message(message) if message.interrupted))
    );
    let raw = std::fs::read_to_string(path).unwrap();
    assert!(raw.contains("request-a"));
    assert!(raw.contains("request-b"));
    assert!(raw.contains("request-c"));
}

// ── Session identity ──────────────────────────────────────────────────

#[test]
fn session_identity_round_trips_through_the_jsonl_meta() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "identity").unwrap();
    let mut written = meta();
    written.session_id = Some("thread-1.orchestrator.g1".into());
    written.parent_session_id = Some("thread-1.orchestrator".into());
    write_transcript(
        &path,
        &[TranscriptMessage::new("user", "hi")],
        &written,
        None,
    )
    .unwrap();

    let read = read_transcript(&path).unwrap();
    assert_eq!(
        read.meta.session_id.as_deref(),
        Some("thread-1.orchestrator.g1")
    );
    assert_eq!(
        read.meta.parent_session_id.as_deref(),
        Some("thread-1.orchestrator")
    );
}

/// [`write_transcript_if_absent`] is what keeps adoption from clobbering a
/// destination a concurrent normal turn created while adoption was still
/// scanning legacy roots (see `adoption::adopt_legacy_session_transcripts`).
/// The guarantee has to hold at the level of this primitive: it must publish
/// when nothing is there, and never overwrite when something already is —
/// regardless of *why* the destination already exists.
#[test]
fn write_transcript_if_absent_publishes_once_and_never_overwrites() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "identity").unwrap();

    let published = write_transcript_if_absent(
        &path,
        &[TranscriptMessage::new("user", "first writer")],
        &meta(),
    )
    .unwrap();
    assert!(published, "nothing was there yet");
    assert_eq!(
        read_transcript(&path).unwrap().messages[0].content,
        "first writer"
    );

    let published_again = write_transcript_if_absent(
        &path,
        &[TranscriptMessage::new(
            "user",
            "second writer, loses the race",
        )],
        &meta(),
    )
    .unwrap();
    assert!(!published_again, "the destination already exists");
    // The loser's content must never have touched disk.
    assert_eq!(
        read_transcript(&path).unwrap().messages[0].content,
        "first writer"
    );
}

#[test]
fn the_latest_tools_record_is_the_sessions_tools_and_never_a_message() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("agent.jsonl");
    let rows = vec![TranscriptMessage::new("user", "hi")];
    append_transcript_turn(&path, &[], &rows, &meta(), None, Some("r1")).unwrap();
    append_tools_record(&path, &serde_json::json!([{"name": "first"}])).unwrap();
    append_tools_record(&path, &serde_json::json!([{"name": "second"}])).unwrap();

    let transcript = read_transcript(&path).unwrap();
    assert_eq!(transcript.messages.len(), 1);
    assert_eq!(
        transcript.tools,
        Some(serde_json::json!([{"name": "second"}]))
    );
    assert_eq!(read_transcript_display(&path).unwrap().records.len(), 1);
}

#[test]
fn a_turn_and_its_tools_are_serialized_in_one_append_buffer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("agent.jsonl");
    let rows = vec![TranscriptMessage::new("user", "hi")];

    writer::append_transcript_turn_with_extras(
        &path,
        &[],
        &rows,
        &meta(),
        None,
        Some("r1"),
        writer::AppendTranscriptExtras {
            partial: None,
            tools: Some(&serde_json::json!([{"name": "search"}])),
        },
    )
    .unwrap();

    let transcript = read_transcript(&path).unwrap();
    assert_eq!(transcript.messages.len(), 1);
    assert_eq!(transcript.messages[0].content, "hi");
    assert_eq!(transcript.messages[0].request_id.as_deref(), Some("r1"));
    assert_eq!(
        transcript.tools,
        Some(serde_json::json!([{"name": "search"}]))
    );
}

#[test]
fn a_transcript_without_a_tools_record_reads_none() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("agent.jsonl");
    let rows = vec![TranscriptMessage::new("user", "hi")];
    append_transcript_turn(&path, &[], &rows, &meta(), None, None).unwrap();
    assert_eq!(read_transcript(&path).unwrap().tools, None);
}
/// Every model call of a multi-step turn is stamped with its own iteration,
/// not only the final answer that carries the turn's usage. Without it a
/// reader cannot tell which call a tool-calling row (or its reasoning)
/// belonged to.
#[test]
fn a_turn_stamps_iteration_and_ts_on_every_step_it_appends() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "steps").unwrap();
    let first = vec![
        TranscriptMessage::new("user", "earlier"),
        TranscriptMessage::assistant("earlier answer"),
    ];
    append_transcript_turn(&path, &[], &first, &meta(), None, Some("request-a")).unwrap();
    let mut prior = read_transcript(&path).unwrap().messages;
    // The replayed prefix is exactly what the runtime diffs against.
    let mut next = prior.clone();
    next.push(TranscriptMessage::new("user", "do it"));
    next.push(TranscriptMessage::assistant(
        r#"{"content":"step one","tool_calls":[{"id":"c1","name":"t","arguments":"{}"}]}"#,
    ));
    next.push(TranscriptMessage::new("tool", "r1"));
    next.push(TranscriptMessage::assistant(
        r#"{"content":"","tool_calls":[{"id":"c2","name":"t","arguments":"{}"}]}"#,
    ));
    next.push(TranscriptMessage::new("tool", "r2"));
    next.push(TranscriptMessage::assistant("done"));
    let usage = TurnUsage {
        provider: "provider".into(),
        model: "model".into(),
        usage: MessageUsage {
            input: 1,
            output: 1,
            cached_input: 0,
            context_window: 0,
            cost_usd: 0.0,
            ..Default::default()
        },
        ts: "2026-01-01T00:00:09Z".into(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        iteration: 3,
    };
    append_transcript_turn(
        &path,
        &prior,
        &next,
        &meta(),
        Some(&usage),
        Some("request-b"),
    )
    .unwrap();
    prior.clear();

    let display = read_transcript_display(&path).unwrap();
    let stamps: Vec<(String, Option<u32>, Option<String>)> = display
        .records
        .iter()
        .filter_map(|record| match record {
            DisplayRecord::Message(m) if m.message.role == "assistant" => Some((
                m.message.content.chars().take(16).collect(),
                m.iteration,
                m.ts.clone(),
            )),
            _ => None,
        })
        .collect();
    let ts = Some("2026-01-01T00:00:09Z".to_string());
    assert_eq!(
        stamps,
        vec![
            // An earlier turn written without usage stays unstamped.
            ("earlier answer".to_string(), None, None),
            ("step one".to_string(), Some(1), ts.clone()),
            (String::new(), Some(2), ts.clone()),
            ("done".to_string(), Some(3), ts),
        ]
    );
    // Only the final row is a usage record: the step stamps must not
    // fabricate usage on the intermediate rows.
    let with_usage = display
        .records
        .iter()
        .filter(|record| matches!(record, DisplayRecord::Message(m) if m.turn_usage.is_some()))
        .count();
    assert_eq!(with_usage, 1);
}

#[test]
fn role_constructors_build_bare_rows() {
    for (row, role) in [
        (TranscriptMessage::system("s"), "system"),
        (TranscriptMessage::user("u"), "user"),
        (TranscriptMessage::assistant("a"), "assistant"),
        (TranscriptMessage::tool("t"), "tool"),
    ] {
        assert_eq!(row, TranscriptMessage::new(role, row.content.clone()));
        assert!(row.id.is_none() && row.extra_metadata.is_none());
    }
    assert_eq!(
        TranscriptMessage::tool("ok")
            .with_id("call-1")
            .id
            .as_deref(),
        Some("call-1")
    );
}

/// A host that persisted only `{role, content}` rows (a checkpoint written
/// before the row type was shared) must still load as a bare row.
#[test]
fn role_content_only_json_deserializes_to_a_bare_row() {
    let row: TranscriptMessage =
        serde_json::from_str(r#"{"role":"assistant","content":"hello"}"#).unwrap();
    assert_eq!(row, TranscriptMessage::assistant("hello"));
}
