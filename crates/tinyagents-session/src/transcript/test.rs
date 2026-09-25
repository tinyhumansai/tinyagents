//! Module-local unit tests for [`crate::transcript`]: full-rewrite and
//! append-only writers, model-context and display readers, compaction replay,
//! interrupted partials, path resolution/resume, thread-usage summaries, and
//! the [`super::history`] locator/handle seam.
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory.

use super::*;
use std::sync::{Arc, Barrier};
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
    assert!(super::reader::read_transcript_meta_only(&path).is_some());
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

#[test]
fn file_history_never_converts_or_drops_durable_fields() {
    let dir = tempdir().unwrap();
    let history = FileTranscriptHistory::new(dir.path(), "session", meta()).unwrap();
    let assistant = TranscriptMessage {
        id: Some("assistant-id".into()),
        role: "assistant".into(),
        content: r#"{"tool_calls":[{"id":"call-1"}]}"#.into(),
        extra_metadata: Some(serde_json::json!({
            "trusted_verbatim": true,
            "artifacts": ["artifact-1"],
            "provider_extension": {"opaque": [1, 2]}
        })),
        cache_breakpoints: vec![4],
        turn_usage: None,
        request_id: None,
        preserve_request_id: false,
        interrupted: false,
        tool_failure: None,
    };
    TranscriptHistory::append(&history, assistant.clone()).unwrap();
    let replayed = TranscriptHistory::messages(&history).unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, assistant.id);
    assert_eq!(replayed[0].role, assistant.role);
    assert_eq!(replayed[0].content, assistant.content);
    assert_eq!(replayed[0].extra_metadata, assistant.extra_metadata);
    assert_eq!(replayed[0].cache_breakpoints, vec![4]);
    TranscriptHistory::replace(&history, std::slice::from_ref(&assistant)).unwrap();
    TranscriptHistory::clear(&history).unwrap();
    assert!(TranscriptHistory::messages(&history).unwrap().is_empty());
    let display = read_transcript_display(history.path()).unwrap();
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Compaction(_)))
    );
    let raw = std::fs::read_to_string(history.path()).unwrap();
    assert!(raw.contains("trusted_verbatim"));
    assert!(raw.contains("artifact-1"));
    assert!(raw.contains("\"cache_breakpoints\":[4]"));
}

#[test]
fn discovery_and_legacy_read_replay_the_canonical_format() {
    let dir = tempdir().unwrap();
    let root = resolve_keyed_transcript_path(dir.path(), "1714000000_agent").unwrap();
    let mut root_meta = meta();
    root_meta.thread_id = Some("thread-1".into());
    write_transcript(
        &root,
        &[TranscriptMessage::new("user", "hello")],
        &root_meta,
        None,
    )
    .unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let found = locator.root_for_thread("thread-1").unwrap();
    assert_eq!(
        found.read_session().unwrap().unwrap().messages[0].content,
        "hello"
    );

    let legacy = dir.path().join("legacy.md");
    std::fs::write(&legacy, "<!-- session_transcript\nagent: agent\ndispatcher: native\n-->\n<!--MSG role=\"user\"-->\nlegacy body\n<!--/MSG-->\n").unwrap();
    assert_eq!(
        read_transcript_legacy_md(&legacy).unwrap().messages[0].content,
        "legacy body"
    );
}

// ── Session identity ──────────────────────────────────────────────────

/// One conversation, two cold sessions: the second must find the first's file
/// rather than mint a second stem for the same thread. This is the regression
/// that cost a real user the opening turns of a thread.
#[test]
fn one_session_resolves_to_one_transcript_across_separate_bindings() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-9fa08", "orchestrator");

    let first = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    first
        .append(TranscriptMessage::new(
            "user",
            "i want to plan a trip to kashmir",
        ))
        .unwrap();

    // A brand-new locator and handle, as a restarted process would build.
    let second = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    second
        .append(TranscriptMessage::new("user", "hello?"))
        .unwrap();

    assert_eq!(first.path(), second.path());
    let messages = second.messages().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "i want to plan a trip to kashmir");
    assert_eq!(messages[1].content, "hello?");

    let roots: Vec<_> = std::fs::read_dir(dir.path().join("session_raw"))
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name())
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "one conversation must not sprawl: {roots:?}"
    );
}

#[test]
fn an_unwritten_session_reads_as_absent_rather_than_erroring() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-new", "orchestrator");

    assert!(locator.read_session_transcript(&session).is_none());
    assert!(!locator.session_exists(&session));
    assert_eq!(locator.head_generation(&session), session);
}

/// A directory occupying a session's canonical `.jsonl` path must never be
/// treated as an existing generation: reads/appends against it would fail
/// (or silently target the wrong thing), and `head_generation`'s chain walk
/// would otherwise stop at a phantom "generation" that was never written.
#[test]
fn a_directory_at_the_canonical_path_is_never_treated_as_an_existing_session() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let path = resolve_keyed_transcript_path(dir.path(), &session_stem(&session)).unwrap();
    std::fs::create_dir_all(&path).unwrap();

    assert!(!locator.session_exists(&session));
    assert!(locator.read_session_transcript(&session).is_none());
}

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

/// A compaction must never destroy what it replaces. It seals the current
/// generation and opens the next, so the replaced turns stay on disk.
#[test]
fn a_compaction_seals_a_generation_and_leaves_it_untouched() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let first = locator.open_session(&session, meta()).unwrap();
    for turn in ["one", "two", "three"] {
        first.append(TranscriptMessage::new("user", turn)).unwrap();
    }
    let sealed_path = first.path().to_path_buf();
    let sealed_bytes = std::fs::read(&sealed_path).unwrap();

    let (successor, handle) = locator.begin_generation(&session, meta()).unwrap();
    // The successor is bound but empty; the retained set is written through the
    // ordinary turn path so usage and request ids are recorded as usual.
    handle
        .replace(&[TranscriptMessage::new("user", "three")])
        .unwrap();

    assert_eq!(successor.generation, 1);
    assert_eq!(
        std::fs::read(&sealed_path).unwrap(),
        sealed_bytes,
        "the sealed generation must be byte-identical afterwards"
    );
    assert_ne!(handle.path(), sealed_path);

    let carried = handle.messages().unwrap();
    assert_eq!(carried.len(), 1);
    assert_eq!(carried[0].content, "three");

    let successor_meta = handle.read_session().unwrap().unwrap().meta;
    assert_eq!(
        successor_meta.session_id.as_deref(),
        Some(session_stem(&successor).as_str())
    );
    assert_eq!(
        successor_meta.parent_session_id.as_deref(),
        Some(session_stem(&session).as_str())
    );
}

/// A fork for edit/regenerate must give the same untouched-sealed-file
/// guarantee a compaction gives, and the chain must walk both generations.
#[test]
fn truncate_into_next_generation_seals_the_head_byte_identical_and_chain_walks_both() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let first = locator.open_session(&session, meta()).unwrap();
    for turn in ["one", "two", "three", "four"] {
        first.append(TranscriptMessage::new("user", turn)).unwrap();
    }
    let sealed_path = first.path().to_path_buf();
    let sealed_bytes = std::fs::read(&sealed_path).unwrap();

    let (successor, handle, truncated) = locator
        .truncate_into_next_generation(&session, TruncateCut::BeforeIndex(2), meta())
        .unwrap();

    assert_eq!(successor.generation, 1);
    assert_eq!(
        std::fs::read(&sealed_path).unwrap(),
        sealed_bytes,
        "the sealed head must be byte-identical afterwards"
    );
    assert_ne!(handle.path(), sealed_path);

    // The new head carries only the retained prefix.
    let contents: Vec<_> = truncated.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(contents, vec!["one", "two"]);
    assert_eq!(handle.messages().unwrap().len(), 2);

    // Parent recorded exactly as `begin_generation` records it.
    let successor_meta = handle.read_session().unwrap().unwrap().meta;
    assert_eq!(
        successor_meta.parent_session_id.as_deref(),
        Some(session_stem(&session).as_str())
    );

    // Both generations are reachable by walking the chain — nothing is lost.
    let chain = locator.session_chain(&session);
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0], session);
    assert_eq!(chain[1], successor);
    assert_eq!(locator.head_generation(&session), successor);
}

#[test]
fn truncate_into_next_generation_by_message_id_cuts_at_that_message() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let first = locator.open_session(&session, meta()).unwrap();
    let mut keep_a = TranscriptMessage::new("user", "keep-a");
    keep_a.id = Some("m1".into());
    let mut cut_here = TranscriptMessage::new("assistant", "cut-here");
    cut_here.id = Some("m2".into());
    let mut dropped = TranscriptMessage::new("user", "dropped");
    dropped.id = Some("m3".into());
    for message in [keep_a, cut_here, dropped] {
        first.append(message).unwrap();
    }

    let (_, handle, truncated) = locator
        .truncate_into_next_generation(&session, TruncateCut::BeforeMessageId("m2".into()), meta())
        .unwrap();

    assert_eq!(truncated.len(), 1);
    assert_eq!(truncated[0].id.as_deref(), Some("m1"));
    assert_eq!(handle.messages().unwrap().len(), 1);
}

#[test]
fn truncate_into_next_generation_by_unknown_message_id_fails() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");
    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();

    let err = locator.truncate_into_next_generation(
        &session,
        TruncateCut::BeforeMessageId("nonexistent".into()),
        meta(),
    );
    assert!(
        err.is_err(),
        "an unresolvable id must fail rather than silently keep everything"
    );
}

#[test]
fn truncate_into_next_generation_last_assistant_turn_drops_only_the_trailing_answer() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let first = locator.open_session(&session, meta()).unwrap();
    for (role, content) in [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer, to regenerate"),
    ] {
        first.append(TranscriptMessage::new(role, content)).unwrap();
    }

    let (_, handle, truncated) = locator
        .truncate_into_next_generation(&session, TruncateCut::LastAssistantTurn, meta())
        .unwrap();

    let contents: Vec<_> = truncated.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(
        contents,
        vec!["first question", "first answer", "second question"]
    );
    assert_eq!(handle.messages().unwrap().len(), 3);
}

#[test]
fn truncate_into_next_generation_operates_on_the_current_head_not_the_root() {
    // A fork after an earlier compaction must truncate the head generation's
    // messages, not the sealed root's.
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "root-only"))
        .unwrap();
    let (compacted, compacted_handle) = locator.begin_generation(&session, meta()).unwrap();
    for turn in ["alpha", "beta", "gamma"] {
        compacted_handle
            .append(TranscriptMessage::new("user", turn))
            .unwrap();
    }

    let (successor, handle, truncated) = locator
        .truncate_into_next_generation(&session, TruncateCut::BeforeIndex(1), meta())
        .unwrap();

    assert_eq!(successor.generation, 2);
    assert_eq!(successor.parent_session_id(), Some(compacted.session_id()));
    let contents: Vec<_> = truncated.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(contents, vec!["alpha"]);
    assert_eq!(handle.messages().unwrap().len(), 1);
}

/// After a compaction, a resume must land on the newest generation — the one
/// the model is actually continuing — not on the sealed original.
#[test]
fn head_generation_follows_the_compaction_chain() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session), session);

    let (first_successor, first_handle) = locator.begin_generation(&session, meta()).unwrap();
    first_handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session), first_successor);

    let (second_successor, second_handle) =
        locator.begin_generation(&first_successor, meta()).unwrap();
    second_handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session).generation, 2);
    assert_eq!(locator.head_generation(&session), second_successor);
}

#[test]
fn opening_a_generation_that_already_exists_is_refused() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let (_, handle) = locator.begin_generation(&session, meta()).unwrap();
    handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    let second = locator.begin_generation(&session, meta());

    assert!(
        second.is_err(),
        "sealing the same generation twice would overwrite durable history"
    );
}

/// Two racing `begin_generation` calls for the same session — two cores
/// compacting at once — must not both write their own first append into the
/// generation's still-nonexistent file: `begin_generation` itself performs no
/// I/O (`FileTranscriptHistory::new` only resolves a path), so without the
/// `path_lock` both handles' first `append_turn_with_partial` would
/// otherwise race on the writer's create-fresh branch and whichever `fs::write`
/// lands last would silently discard the other's retained set.
///
/// The threads rendezvous twice, and the second rendezvous is load-bearing.
/// `begin_generation` refuses to open a successor that already exists, so with
/// only the pre-`begin_generation` barrier a thread that got all the way
/// through its append before the other called `begin_generation` would make
/// that call fail on the existence check — a race in the test itself rather
/// than the contention it means to exercise. Holding both threads until each
/// owns its handle puts the contention where this test is aiming it: on the
/// two first appends.
#[test]
fn concurrent_begin_generation_handles_for_one_session_never_lose_either_append() {
    let dir = tempdir().unwrap();
    let session = FileTranscriptLocator::new(dir.path());
    let root = SessionRef::scoped("thread-1", "orchestrator");
    // Seal generation 0 so both racers are compacting into the same,
    // already-known successor generation 1.
    session
        .open_session(&root, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "sealed"))
        .unwrap();

    let locator = Arc::new(FileTranscriptLocator::new(dir.path()));
    let barrier = Arc::new(Barrier::new(2));

    let left_locator = Arc::clone(&locator);
    let left_root = root.clone();
    let left_barrier = Arc::clone(&barrier);
    let left = std::thread::spawn(move || {
        left_barrier.wait();
        let (_, handle) = left_locator.begin_generation(&left_root, meta()).unwrap();
        // Both handles exist before either append starts.
        left_barrier.wait();
        handle
            .append(TranscriptMessage::new("user", "from left"))
            .unwrap();
    });

    let right_locator = Arc::clone(&locator);
    let right_root = root.clone();
    let right_barrier = Arc::clone(&barrier);
    let right = std::thread::spawn(move || {
        right_barrier.wait();
        let (_, handle) = right_locator.begin_generation(&right_root, meta()).unwrap();
        // Both handles exist before either append starts.
        right_barrier.wait();
        handle
            .append(TranscriptMessage::new("user", "from right"))
            .unwrap();
    });

    left.join().unwrap();
    right.join().unwrap();

    let successor = root.next_generation();
    let handle = locator.open_session(&successor, meta()).unwrap();
    let contents: Vec<String> = handle
        .messages()
        .unwrap()
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(
        contents.len(),
        2,
        "both racing compactions' appends must survive: {contents:?}"
    );
}

/// Two handles on one session — the shape two cores over one workspace
/// produce — must both land in the same file, with neither losing the other's
/// turns.
#[test]
fn concurrent_handles_on_one_session_both_extend_it() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-1", "orchestrator");
    let left = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    let right = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();

    // Seed the file first, so both appends below exercise the append-only
    // path — `write_logical_set` re-reading `persisted` fresh immediately
    // before every write, which is the actual claim under test — rather than
    // racing on first-write file creation, a distinct, pre-existing concern
    // this test is not about.
    left.append(TranscriptMessage::new("user", "seed")).unwrap();

    // Genuinely overlapping, not merely interleaved: both handles race to
    // append from separate OS threads, released together by a barrier so
    // neither can start before the other is ready. A handle that cached its
    // own view of `persisted` instead of re-reading it fresh before every
    // write could lose whichever append the barrier let land second.
    let barrier = Arc::new(Barrier::new(2));
    let left_barrier = Arc::clone(&barrier);
    let left_thread = std::thread::spawn(move || {
        left_barrier.wait();
        left.append(TranscriptMessage::new("user", "from left"))
            .unwrap();
    });
    let right_barrier = Arc::clone(&barrier);
    let right_thread = std::thread::spawn(move || {
        right_barrier.wait();
        right
            .append(TranscriptMessage::new("user", "from right"))
            .unwrap();
    });
    left_thread.join().unwrap();
    right_thread.join().unwrap();

    let reread = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    let mut contents: Vec<String> = reread
        .messages()
        .unwrap()
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents.remove(0), "seed");
    contents.sort();
    assert_eq!(
        contents,
        ["from left", "from right"],
        "an overlapping append from either handle must not be lost"
    );
}

/// The model reads only the head generation, but a host rendering or
/// exporting the conversation needs every segment, in order.
#[test]
fn a_session_chain_lists_every_generation_oldest_first() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    assert!(locator.session_chain(&session).is_empty());

    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    let (successor, handle) = locator.begin_generation(&session, meta()).unwrap();
    handle
        .append(TranscriptMessage::new("user", "two"))
        .unwrap();

    let chain = locator.session_chain(&session);
    assert_eq!(chain, vec![session.clone(), successor]);
    // Asking from any generation returns the same whole chain.
    assert_eq!(locator.session_chain(&chain[1]), chain);
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
            (r#"{"content":"step"#.to_string(), Some(1), ts.clone()),
            (r#"{"content":"","t"#.to_string(), Some(2), ts.clone()),
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

/// A compaction successor inherits its predecessor's `created`, so the thread
/// scan has to order a generation chain by generation — by path alone
/// `X.g1` sorts before `X` and `X.g10` before `X.g2`, and the newest-wins
/// lookup resolved a sealed generation.
#[test]
fn thread_scan_orders_a_generation_chain_by_generation() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-gen", "orchestrator");
    let mut seed = meta();
    seed.thread_id = Some("thread-gen".into());
    locator
        .open_session(&session, seed.clone())
        .unwrap()
        .append(TranscriptMessage::new("user", "g0"))
        .unwrap();
    let mut head = session.clone();
    for n in 1..=10 {
        let (successor, handle) = locator.begin_generation(&head, seed.clone()).unwrap();
        handle
            .append(TranscriptMessage::new("user", format!("g{n}")))
            .unwrap();
        head = successor;
    }

    let roots = find_root_transcripts_for_thread(dir.path(), "thread-gen");
    let generations: Vec<u32> = roots
        .iter()
        .map(|path| super::thread_lookup::path_generation(path))
        .collect();
    assert_eq!(generations, (0..=10).collect::<Vec<_>>());
    let newest = find_root_transcript_for_thread(dir.path(), "thread-gen").unwrap();
    assert_eq!(
        read_transcript(&newest).unwrap().messages[0].content,
        "g10",
        "newest-wins resolves the head generation"
    );
}
