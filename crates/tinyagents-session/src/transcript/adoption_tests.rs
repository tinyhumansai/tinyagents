use super::*;
use crate::transcript::{
    FileTranscriptLocator, MessageUsage, TranscriptLocator, TranscriptToolCall, TurnUsage,
    append_tools_record, append_transcript_turn, read_transcript, write_transcript,
    write_transcript_if_absent,
};
use tempfile::tempdir;

fn legacy_meta(created: &str, updated: &str, thread_id: &str) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: Some("orchestrator".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: Some("model".into()),
        created: created.into(),
        updated: updated.into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 10,
        output_tokens: 5,
        cached_input_tokens: 2,
        charged_amount_usd: 0.5,
        thread_id: Some(thread_id.into()),
        task_id: None,
    }
}

fn write_legacy(dir: &Path, stem: &str, created: &str, body: &str, thread_id: &str) {
    let path = resolve_keyed_transcript_path(dir, stem).unwrap();
    write_transcript(
        &path,
        &[TranscriptMessage::new("user", body)],
        &legacy_meta(created, created, thread_id),
        None,
    )
    .unwrap();
}

/// The shape that cost a real user their thread: one conversation spread
/// across three timestamped stems, of which resume only ever saw the newest.
#[test]
fn legacy_roots_fold_into_one_session_in_created_order() {
    let dir = tempdir().unwrap();
    let thread = "thread-9fa08c44";
    write_legacy(
        dir.path(),
        "1790062247_orchestrator",
        "2026-09-22T07:30:47Z",
        "i want to plan a trip to kashmir",
        thread,
    );
    write_legacy(
        dir.path(),
        "1790100297_orchestrator",
        "2026-09-22T18:05:22Z",
        "hello?",
        thread,
    );
    write_legacy(
        dir.path(),
        "1790100352_orchestrator",
        "2026-09-22T18:06:27Z",
        "what did i ask here?",
        thread,
    );

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .expect("three legacy roots should be adopted");

    assert_eq!(adoption.adopted.len(), 3);
    assert_eq!(adoption.messages, 3);

    let adopted = read_transcript(&adoption.path).unwrap();
    let contents: Vec<&str> = adopted
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert_eq!(
        contents,
        [
            "i want to plan a trip to kashmir",
            "hello?",
            "what did i ask here?"
        ]
    );
    assert_eq!(adopted.meta.session_id, Some(session.session_id()));
    assert_eq!(adopted.meta.thread_id.as_deref(), Some(thread));
}

#[test]
fn adoption_preserves_the_first_sources_frozen_prefix_count() {
    let dir = tempdir().unwrap();
    let thread = "thread-prefix-adoption";
    let first = resolve_keyed_transcript_path(dir.path(), "1000_orchestrator").unwrap();
    let mut first_meta = legacy_meta("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", thread);
    first_meta.prefix_message_count = Some(1);
    write_transcript(
        &first,
        &[
            TranscriptMessage::new("system", "stable"),
            TranscriptMessage::new("system", "history summary"),
            TranscriptMessage::new("user", "first"),
        ],
        &first_meta,
        None,
    )
    .unwrap();
    let second = resolve_keyed_transcript_path(dir.path(), "2000_orchestrator").unwrap();
    let mut second_meta = legacy_meta("2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", thread);
    second_meta.prefix_message_count = Some(2);
    write_transcript(
        &second,
        &[TranscriptMessage::new("user", "later")],
        &second_meta,
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adopted = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    assert_eq!(
        read_transcript(&adopted.path)
            .unwrap()
            .meta
            .prefix_message_count,
        Some(1)
    );
}

#[test]
fn adoption_sums_usage_and_spans_the_whole_conversation() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    write_legacy(dir.path(), "2000_a", "2026-02-02T00:00:00Z", "two", thread);

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let meta = read_transcript(&adoption.path).unwrap().meta;
    assert_eq!(meta.turn_count, 2);
    assert_eq!(meta.input_tokens, 20);
    assert_eq!(meta.output_tokens, 10);
    assert_eq!(meta.cached_input_tokens, 4);
    assert_eq!(meta.charged_amount_usd, 1.0);
    assert_eq!(meta.created, "2026-01-01T00:00:00Z");
    assert_eq!(meta.updated, "2026-02-02T00:00:00Z");
}

#[test]
fn adoption_preserves_the_newest_legacy_tool_snapshot() {
    let dir = tempdir().unwrap();
    let thread = "thread-tools";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    write_legacy(dir.path(), "2000_a", "2026-01-02T00:00:00Z", "two", thread);
    let newer = resolve_keyed_transcript_path(dir.path(), "2000_a").unwrap();
    append_tools_record(&newer, &serde_json::json!([{"name": "newer"}])).unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .expect("legacy roots should be adopted");

    assert_eq!(
        read_transcript(&adoption.path).unwrap().tools,
        Some(serde_json::json!([{"name": "newer"}]))
    );
}

#[test]
fn adoption_never_touches_the_files_it_reads() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let legacy_path = resolve_keyed_transcript_path(dir.path(), "1000_a").unwrap();
    let before = std::fs::read(&legacy_path).unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    adopt_legacy_session_transcripts(dir.path(), &session, thread, &legacy_meta("", "", thread))
        .unwrap()
        .unwrap();

    assert_eq!(std::fs::read(&legacy_path).unwrap(), before);
}

#[test]
fn adoption_is_idempotent_and_never_re_folds() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let session = SessionRef::scoped(thread, "orchestrator");
    let seed = legacy_meta("", "", thread);

    let first = adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed)
        .unwrap()
        .unwrap();
    // The adopted transcript is itself a root matching this thread, so a
    // second pass must recognise the session as already backed rather than
    // folding the conversation into itself.
    let second = adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed).unwrap();

    assert!(second.is_none());
    assert_eq!(read_transcript(&first.path).unwrap().messages.len(), 1);
}

#[test]
fn a_thread_with_no_legacy_roots_adopts_nothing() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-fresh", "orchestrator");
    let result = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        "thread-fresh",
        &legacy_meta("", "", "thread-fresh"),
    )
    .unwrap();

    assert!(result.is_none());
}

#[test]
fn an_adopted_session_is_what_the_locator_then_resolves() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let session = SessionRef::scoped(thread, "orchestrator");
    adopt_legacy_session_transcripts(dir.path(), &session, thread, &legacy_meta("", "", thread))
        .unwrap()
        .unwrap();

    let locator = FileTranscriptLocator::new(dir.path());
    let read = locator
        .read_session_transcript(&session)
        .expect("the adopted transcript backs the session");
    assert_eq!(
        read.read_session().unwrap().unwrap().messages[0].content,
        "one"
    );
}

// ── Legacy OpenHuman layouts ──────────────────────────────────────────
//
// OpenHuman wrote transcripts several ways before session identity existed.
// Adoption has to recover a conversation from each of them, because these are
// exactly the files sitting in users' workspaces today.

/// A sub-agent transcript shares its parent's `thread_id`. Folding one into the
/// root conversation would splice a delegated worker's private reasoning into
/// the user's chat, so the `parent__child` stems must stay out.
#[test]
fn subagent_siblings_are_never_folded_into_the_root_conversation() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(
        dir.path(),
        "1000_orchestrator",
        "2026-01-01T00:00:00Z",
        "user ask",
        thread,
    );
    write_legacy(
        dir.path(),
        "1000_orchestrator__1001_researcher",
        "2026-01-01T00:00:01Z",
        "worker chatter",
        thread,
    );

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let contents: Vec<String> = read_transcript(&adoption.path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents, ["user ask"]);
    assert_eq!(adoption.adopted.len(), 1);
}

/// OpenHuman's pre-session-key naming was `{agent}_{index}` with no timestamp
/// at all. Those stems carry a `thread_id` in `_meta` just the same, so they
/// must adopt like any other root.
#[test]
fn legacy_indexed_openhuman_stems_are_adopted() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(
        dir.path(),
        "orchestrator_1",
        "2026-01-01T00:00:00Z",
        "first",
        thread,
    );
    write_legacy(
        dir.path(),
        "orchestrator_2",
        "2026-01-02T00:00:00Z",
        "second",
        thread,
    );

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let contents: Vec<String> = read_transcript(&adoption.path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents, ["first", "second"]);
}

/// The date-grouped `session_raw/DDMMYYYY/` layout is only reachable to the
/// thread scan after the layout migration has flattened it, so the two steps
/// have to compose: migrate, then adopt.
#[test]
fn date_grouped_openhuman_transcripts_adopt_after_the_layout_migration() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    let legacy_dir = dir.path().join("session_raw").join("01012026");
    std::fs::create_dir_all(&legacy_dir).unwrap();
    write_transcript(
        &legacy_dir.join("1000_orchestrator.jsonl"),
        &[TranscriptMessage::new("user", "from the dated layout")],
        &legacy_meta("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", thread),
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let seed = legacy_meta("", "", thread);
    // Before flattening, the dated file is invisible to the root scan.
    assert!(
        adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed)
            .unwrap()
            .is_none()
    );

    crate::transcript::migrate_layout_if_needed(dir.path()).unwrap();
    let adoption = adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed)
        .unwrap()
        .expect("the flattened transcript is adoptable");

    assert_eq!(
        read_transcript(&adoption.path).unwrap().messages[0].content,
        "from the dated layout"
    );
}

/// Transcripts written before `_meta.session_id` existed deserialize with it
/// absent. Adoption must not require it — it is the whole population being
/// migrated.
#[test]
fn transcripts_without_session_identity_still_adopt() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(
        dir.path(),
        "1000_a",
        "2026-01-01T00:00:00Z",
        "pre-identity",
        thread,
    );
    let legacy =
        read_transcript(&resolve_keyed_transcript_path(dir.path(), "1000_a").unwrap()).unwrap();
    assert_eq!(legacy.meta.session_id, None);
    assert_eq!(legacy.meta.parent_session_id, None);

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let adopted = read_transcript(&adoption.path).unwrap();
    assert_eq!(adopted.messages[0].content, "pre-identity");
    assert_eq!(adopted.meta.session_id, Some(session.session_id()));
}

/// Tool calls, tool results and usage are the reason the transcript is the
/// resume source rather than the prose conversation log. Adoption must carry
/// them across intact.
#[test]
fn adoption_preserves_tool_rounds_and_usage_of_legacy_transcripts() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    let mut assistant = TranscriptMessage::assistant("calling a tool");
    assistant.id = Some("call-1".into());
    assistant.turn_usage = Some(TurnUsage {
        provider: "openrouter".into(),
        model: "model".into(),
        usage: MessageUsage {
            input: 11,
            output: 7,
            cached_input: 3,
            context_window: 1000,
            cost_usd: 0.25,
            ..Default::default()
        },
        ts: "2026-01-01T00:00:00Z".into(),
        reasoning_content: Some("thinking".into()),
        tool_calls: vec![TranscriptToolCall {
            id: "call-1".into(),
            name: "web_search".into(),
            arguments: "{\"q\":\"kashmir\"".into(),
            extra_content: None,
        }],
        iteration: 2,
    });
    let mut tool_result = TranscriptMessage::new("tool", "search results");
    tool_result.id = Some("call-1".into());
    tool_result.cache_breakpoints = vec![1];

    write_transcript(
        &resolve_keyed_transcript_path(dir.path(), "1000_a").unwrap(),
        &[
            TranscriptMessage::new("user", "find me something"),
            assistant,
            tool_result,
        ],
        &legacy_meta("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", thread),
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let adopted = read_transcript(&adoption.path).unwrap();
    assert_eq!(adopted.messages.len(), 3);
    let usage = adopted.messages[1]
        .turn_usage
        .as_ref()
        .expect("assistant usage survives adoption");
    assert_eq!(usage.usage.input, 11);
    assert_eq!(usage.reasoning_content.as_deref(), Some("thinking"));
    assert_eq!(usage.tool_calls[0].name, "web_search");
    // Raw, unrepaired provider JSON is preserved verbatim.
    assert_eq!(usage.tool_calls[0].arguments, "{\"q\":\"kashmir\"");
    assert_eq!(adopted.messages[2].role, "tool");
    assert_eq!(adopted.messages[2].cache_breakpoints, vec![1]);
}

/// A different agent scoped to the same thread id writes its own root
/// transcript. Folding it into this agent's adoption would splice one
/// agent's private history into another's — the same mixing
/// `find_root_transcript_for_thread_scoped` exists to prevent for ordinary
/// resume.
#[test]
fn a_different_agents_root_on_the_same_thread_is_never_folded_in() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "mine", thread);
    let mut other_agent = legacy_meta("2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z", thread);
    other_agent.agent_id = Some("researcher".into());
    write_transcript(
        &resolve_keyed_transcript_path(dir.path(), "1000_researcher").unwrap(),
        &[TranscriptMessage::new("user", "not mine")],
        &other_agent,
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let contents: Vec<String> = read_transcript(&adoption.path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents, ["mine"]);
    assert_eq!(adoption.adopted.len(), 1);
}

/// A file that already carries session identity (already adopted, or one of
/// this session's own later generations sharing the thread id) is not
/// pre-identity legacy content. Folding it in would duplicate history that
/// adoption already recovered, or content this call has no business reading.
#[test]
fn a_session_identified_root_on_the_same_thread_is_never_folded_in() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(
        dir.path(),
        "1000_a",
        "2026-01-01T00:00:00Z",
        "legacy",
        thread,
    );
    let mut already_adopted = legacy_meta("2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z", thread);
    already_adopted.session_id = Some("thread-1~deadbeef.orchestrator~deadbeef".into());
    write_transcript(
        &resolve_keyed_transcript_path(dir.path(), "2000_a").unwrap(),
        &[TranscriptMessage::new("user", "already adopted elsewhere")],
        &already_adopted,
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let contents: Vec<String> = read_transcript(&adoption.path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents, ["legacy"]);
    assert_eq!(adoption.adopted.len(), 1);
}

/// The thread-lookup scan itself silently drops a candidate it cannot read
/// (logs a warning, excludes it) *before* adoption's own fold loop ever sees
/// it — so a fail-fast check inside that loop alone cannot catch this case.
/// The scan must surface "something in this workspace was unreadable" so
/// adoption can defer instead of finalizing on only the readable roots.
#[test]
fn an_unreadable_root_the_scan_itself_drops_defers_adoption_rather_than_finalizing() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(
        dir.path(),
        "1000_a",
        "2026-01-01T00:00:00Z",
        "readable",
        thread,
    );
    // A `.jsonl` file the scan cannot parse at all — `find_root_transcripts_for_thread`
    // would silently exclude this from its result and only log a warning.
    let corrupt_path = resolve_keyed_transcript_path(dir.path(), "2000_a").unwrap();
    std::fs::write(&corrupt_path, b"not a valid meta line at all\n").unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let result = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    );

    assert!(
        result.is_err(),
        "a workspace-wide unreadable candidate must defer adoption, not finalize on a partial fold"
    );
    let destination = resolve_keyed_transcript_path(dir.path(), &session_stem(&session)).unwrap();
    assert!(
        !destination.exists(),
        "a deferred adoption must not create the idempotency marker"
    );
}

/// Two independent adopters (simulating two racing processes) for the same
/// session must not both fold the same legacy roots: the lock in
/// [`adopt_legacy_session_transcripts`] serializes them, so the second call
/// backs off and sees the first call's result rather than re-folding (which
/// would duplicate messages) or overwriting it with a stale view.
#[test]
fn concurrent_adoption_attempts_do_not_duplicate_or_race() {
    use std::sync::Arc;
    use std::sync::Barrier;

    let dir = tempdir().unwrap();
    let dir_path: Arc<std::path::PathBuf> = Arc::new(dir.path().to_path_buf());
    let thread = "thread-1";
    write_legacy(
        dir_path.as_path(),
        "1000_a",
        "2026-01-01T00:00:00Z",
        "one",
        thread,
    );
    write_legacy(
        dir_path.as_path(),
        "2000_a",
        "2026-01-02T00:00:00Z",
        "two",
        thread,
    );

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let dir_path = Arc::clone(&dir_path);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let session = SessionRef::scoped(thread, "orchestrator");
                barrier.wait();
                adopt_legacy_session_transcripts(
                    dir_path.as_path(),
                    &session,
                    thread,
                    &legacy_meta("", "", thread),
                )
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap().unwrap())
        .collect();
    assert_eq!(
        results.iter().filter(|result| result.is_some()).count(),
        1,
        "exactly one racing adopter should perform the fold; results: {results:?}"
    );

    let session = SessionRef::scoped(thread, "orchestrator");
    let destination =
        resolve_keyed_transcript_path(dir_path.as_path(), &session_stem(&session)).unwrap();
    let adopted = read_transcript(&destination).unwrap();
    assert_eq!(adopted.messages.len(), 2, "no duplication across racers");
}

/// The lock above only serializes competing *adopters* — it says nothing
/// about a normal session turn independently creating this same session's
/// first transcript while adoption is mid-scan (the CodeRabbit-reported
/// gap). `write_transcript_if_absent`, not the lock, is what has to make
/// that safe: whichever of the two legitimately wins the race to publish
/// first must never be silently destroyed by the other.
#[test]
fn a_concurrent_normal_write_and_an_adoption_never_destroy_each_other() {
    use std::sync::{Arc, Barrier};

    for _ in 0..20 {
        let dir = tempdir().unwrap();
        let dir_path: Arc<std::path::PathBuf> = Arc::new(dir.path().to_path_buf());
        let thread = "thread-1";
        write_legacy(
            dir_path.as_path(),
            "1000_a",
            "2026-01-01T00:00:00Z",
            "legacy",
            thread,
        );

        let barrier = Arc::new(Barrier::new(2));

        let adopt_dir = Arc::clone(&dir_path);
        let adopt_barrier = Arc::clone(&barrier);
        let adopt_thread = std::thread::spawn(move || {
            let session = SessionRef::scoped(thread, "orchestrator");
            adopt_barrier.wait();
            adopt_legacy_session_transcripts(
                adopt_dir.as_path(),
                &session,
                thread,
                &legacy_meta("", "", thread),
            )
            .unwrap()
        });

        let write_dir = Arc::clone(&dir_path);
        let write_barrier = Arc::clone(&barrier);
        let write_thread = std::thread::spawn(move || {
            let session = SessionRef::scoped(thread, "orchestrator");
            let destination =
                resolve_keyed_transcript_path(write_dir.as_path(), &session_stem(&session))
                    .unwrap();
            let mut turn_meta = legacy_meta("2026-01-01T00:00:05Z", "2026-01-01T00:00:05Z", thread);
            turn_meta.session_id = Some(session.session_id());
            write_barrier.wait();
            write_transcript_if_absent(
                &destination,
                &[TranscriptMessage::new("user", "real first turn")],
                &turn_meta,
            )
            .unwrap()
        });

        let adopt_result = adopt_thread.join().unwrap();
        let write_result = write_thread.join().unwrap();

        let session = SessionRef::scoped(thread, "orchestrator");
        let destination =
            resolve_keyed_transcript_path(dir_path.as_path(), &session_stem(&session)).unwrap();
        assert!(destination.exists(), "one of the two must have published");
        let contents: Vec<String> = read_transcript(&destination)
            .unwrap()
            .messages
            .into_iter()
            .map(|message| message.content)
            .collect();

        // Exactly one side wins, and the file reflects that winner alone —
        // never a mix, and the loser's data never touched disk.
        if adopt_result.is_some() {
            assert_eq!(contents, ["legacy"]);
            assert!(
                !write_result,
                "adoption published first; the concurrent write must have lost"
            );
        } else {
            assert_eq!(contents, ["real first turn"]);
            assert!(
                write_result,
                "the concurrent write published first; adoption must have lost"
            );
        }
    }
}

/// A legacy transcript whose turns were compacted replays as its reduced set.
/// Adoption folds what the model would actually have seen, not the raw lines.
#[test]
fn adoption_folds_the_replayed_context_of_a_compacted_legacy_transcript() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    let path = resolve_keyed_transcript_path(dir.path(), "1000_a").unwrap();
    let meta = legacy_meta("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", thread);
    write_transcript(
        &path,
        &[
            TranscriptMessage::new("user", "one"),
            TranscriptMessage::new("user", "two"),
        ],
        &meta,
        None,
    )
    .unwrap();
    // A compaction record reduces the logical set, as OpenHuman's trim did.
    let retained = [TranscriptMessage::new("user", "two")];
    append_transcript_turn(
        &path,
        &[
            TranscriptMessage::new("user", "one"),
            TranscriptMessage::new("user", "two"),
        ],
        &retained,
        &meta,
        None,
        None,
    )
    .unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .unwrap();

    let contents: Vec<String> = read_transcript(&adoption.path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents, ["two"]);
}

/// A stale-lock reclaim (see `AdoptionLock::acquire`) means two
/// `AdoptionLock` values can briefly both believe they hold the same path.
/// The one that lost that race must never unlink the file out from under
/// the one that actually holds it now — that is exactly what its
/// ownership token (compared in `Drop`) exists to prevent.
#[test]
fn drop_never_removes_a_lock_another_owner_now_holds() {
    let dir = tempdir().unwrap();
    let lock_path = dir.path().join("thread-1.jsonl.adopting");

    let first = AdoptionLock::acquire(&lock_path).unwrap().unwrap();
    // Simulate a concurrent stale-lock reclaim by another process: it
    // removes the file and creates its own with a fresh token, exactly
    // what `AdoptionLock::acquire`'s reclaim branch does.
    std::fs::remove_file(&lock_path).unwrap();
    let second = AdoptionLock::acquire(&lock_path).unwrap().unwrap();

    drop(first);
    assert!(
        lock_path.exists(),
        "the first (now-stale) lock's Drop must not remove the second owner's live lock"
    );

    drop(second);
    assert!(
        !lock_path.exists(),
        "the still-legitimate second owner's Drop must remove its own lock"
    );
}
