//! Aggregation tests for the transcript spend totals.
//!
//! Every fixture is written with the session crate's own writer
//! (`append_transcript_turn`), never hand-rolled JSONL, so a wire-format change
//! upstream surfaces here as a failure instead of silently diverging from what
//! the core actually persists.

use super::*;
use crate::transcript::{
    MessageUsage, TranscriptMessage, TranscriptMeta, TurnUsage, append_transcript_turn,
};

const MODEL: &str = "openrouter/deepseek/deepseek-v4-flash";

fn meta(agent: &str, agent_type: &str, thread_id: Option<&str>) -> TranscriptMeta {
    TranscriptMeta {
        agent_name: agent.to_string(),
        agent_id: Some(agent.to_string()),
        agent_type: Some(agent_type.to_string()),
        dispatcher: "native".into(),
        provider: Some("openhuman".into()),
        model: Some(MODEL.into()),
        created: "2026-09-22T00:00:00Z".into(),
        updated: "2026-09-22T01:00:00Z".into(),
        turn_count: 0,
        prefix_message_count: None,
        // Every fixture leaves the denormalised `_meta` rollups at zero: that is
        // exactly what is on disk for every root transcript written since
        // `33566d382`, and the aggregate must not depend on them.
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: thread_id.map(str::to_owned),
        task_id: None,
        session_id: None,
        parent_session_id: None,
    }
}

fn turn_usage(input: u64, output: u64, cached: u64) -> TurnUsage {
    TurnUsage {
        provider: "openhuman".into(),
        model: MODEL.into(),
        usage: MessageUsage {
            input,
            output,
            cached_input: cached,
            context_window: 1_000_000,
            cost_usd: 0.0,
            ..Default::default()
        },
        ts: "2026-09-22T01:00:00Z".into(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        iteration: 1,
    }
}

/// Append `turns` to `session_raw/{stem}.jsonl`, one durable turn per entry,
/// growing the message set the way the runtime does.
fn write_transcript_turns(
    workspace: &Path,
    stem: &str,
    meta: &TranscriptMeta,
    turns: &[(u64, u64, u64)],
) {
    let dir = workspace.join("session_raw");
    std::fs::create_dir_all(&dir).expect("create session_raw");
    let path = dir.join(format!("{stem}.jsonl"));
    let mut persisted: Vec<TranscriptMessage> = Vec::new();
    for (i, (input, output, cached)) in turns.iter().enumerate() {
        let mut next = persisted.clone();
        next.push(TranscriptMessage::new("user", format!("q{i}")));
        next.push(TranscriptMessage::assistant(format!("a{i}")));
        let mut turn_meta = meta.clone();
        turn_meta.turn_count = i + 1;
        append_transcript_turn(
            &path,
            &persisted,
            &next,
            &turn_meta,
            Some(&turn_usage(*input, *output, *cached)),
            Some(&format!("req-{i}")),
        )
        .expect("append turn");
        persisted = next;
    }
}

/// A thread with no sub-agents: the total is the orchestrator's own spend, and
/// the zero `_meta` rollups are ignored.
#[test]
fn totals_a_thread_with_no_subagents_from_its_turn_records() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-solo";
    write_transcript_turns(
        tmp.path(),
        "1790000000_orchestrator_solo",
        &meta("orchestrator", "root", Some(thread)),
        &[(10_000, 100, 4_000), (20_000, 200, 8_000)],
    );

    let spend = thread_spend(tmp.path(), thread);

    assert_eq!(spend.root.input_tokens, 30_000);
    assert_eq!(spend.root.output_tokens, 300);
    assert_eq!(spend.root.cached_input_tokens, 12_000);
    assert_eq!(spend.root.turns, 2);
    assert!(
        spend.subagents.is_empty(),
        "no sub-agent transcripts exist, so none may be attributed"
    );
    // The newest turn owns the last-turn view, not the sum of turns.
    assert_eq!(spend.root.last_input_tokens, 20_000);
    assert_eq!(spend.root.last_output_tokens, 200);
}

/// The `thread-37dcc15f` shape from the bug report: a root whose children burned
/// far more than it did, and whose child transcripts carry a **worker** thread
/// id rather than the parent's.
///
/// This is the regression. Selecting children by `_meta.thread_id` — what
/// `read_thread_usage_summary` does — attributed 0 of the 397,423 child input
/// tokens to the thread, so ~88% of its spend was invisible (#6460).
#[test]
fn counts_subagents_whose_transcripts_carry_a_worker_thread_id() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-37dcc15f";
    let root_stem = "1789765020_orchestrator_thread-37dcc";
    write_transcript_turns(
        tmp.path(),
        root_stem,
        &meta("orchestrator", "root", Some(thread)),
        &[(16_884, 34, 0)],
    );
    // Both children name their own worker thread, never `thread`.
    write_transcript_turns(
        tmp.path(),
        &format!("{root_stem}__1789766110_researcher_sub-a"),
        &meta(
            "researcher",
            "subagent",
            Some("worker-0aa53709-436d-4566-8798-3629beb11723"),
        ),
        &[(180_295, 8_000, 0)],
    );
    write_transcript_turns(
        tmp.path(),
        &format!("{root_stem}__1789766852_researcher_sub-b"),
        &meta(
            "researcher",
            "subagent",
            Some("worker-6b2d242f-c1b1-49ef-8b74-0a7ebeda4afc"),
        ),
        &[(217_128, 8_873, 0)],
    );

    let spend = thread_spend(tmp.path(), thread);

    assert_eq!(spend.root.input_tokens, 16_884, "orchestrator's own spend");
    let (researcher, runs) = spend
        .subagents
        .get("researcher")
        .expect("both children group under their archetype despite the worker thread ids");
    assert_eq!(*runs, 2, "two separate delegations, two runs");
    assert_eq!(
        researcher.input_tokens, 397_423,
        "both children's input must be attributed to the parent thread"
    );
    assert_eq!(researcher.output_tokens, 16_873);

    // The thread total counts every token exactly ONCE. The pre-fix arithmetic
    // would reach 414,307 + 397,423 had the root record also carried its
    // children, which is the double count this design removes.
    let total = spend.root.input_tokens + researcher.input_tokens;
    assert_eq!(total, 414_307);
}

/// A child's spend is counted from the child's own transcript, so a delegation
/// whose usage never reached the parent's in-turn ledger (#6459, the default
/// async path) is still counted — and counted once, not twice.
#[test]
fn counts_each_transcript_once_regardless_of_the_parent_ledger() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-ledger";
    let root_stem = "1790000001_orchestrator_ledger";
    // The root records ONLY its own spend — the contract
    // `session_host::codec::turn_usage` now guarantees.
    write_transcript_turns(
        tmp.path(),
        root_stem,
        &meta("orchestrator", "root", Some(thread)),
        &[(1_000, 10, 0)],
    );
    write_transcript_turns(
        tmp.path(),
        &format!("{root_stem}__child_coder"),
        &meta("coding_agent", "subagent", Some("worker-xyz")),
        &[(5_000, 50, 0)],
    );

    let spend = thread_spend(tmp.path(), thread);
    let (coder, runs) = spend.subagents.get("coding_agent").expect("child counted");

    assert_eq!(spend.root.input_tokens, 1_000);
    assert_eq!(coder.input_tokens, 5_000);
    assert_eq!(*runs, 1);
    assert_eq!(spend.root.input_tokens + coder.input_tokens, 6_000);
}

/// A grandchild chains another `__`, and must be counted once at its own depth
/// rather than dropped or folded into its parent twice.
#[test]
fn counts_a_grandchild_delegation_once() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-deep";
    let root_stem = "1790000002_orchestrator_deep";
    write_transcript_turns(
        tmp.path(),
        root_stem,
        &meta("orchestrator", "root", Some(thread)),
        &[(1_000, 10, 0)],
    );
    write_transcript_turns(
        tmp.path(),
        &format!("{root_stem}__child"),
        &meta("researcher", "subagent", Some("worker-1")),
        &[(2_000, 20, 0)],
    );
    write_transcript_turns(
        tmp.path(),
        &format!("{root_stem}__child__grandchild"),
        &meta("coding_agent", "subagent", Some("worker-2")),
        &[(4_000, 40, 0)],
    );

    let spend = thread_spend(tmp.path(), thread);
    let total: u64 = spend.root.input_tokens
        + spend
            .subagents
            .values()
            .map(|(child, _)| child.input_tokens)
            .sum::<u64>();
    assert_eq!(
        spend.subagents.len(),
        2,
        "child and grandchild both counted"
    );
    assert_eq!(total, 7_000);
}

/// Legacy transcripts already on disk carry zero `_meta` rollups. They must keep
/// working, which they do because the aggregate never reads the header.
#[test]
fn ignores_the_dead_meta_rollup_on_legacy_transcripts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-legacy";
    let legacy = meta("orchestrator", "root", Some(thread));
    assert_eq!(
        legacy.input_tokens, 0,
        "fixture must reproduce the on-disk state: a zero header"
    );
    write_transcript_turns(
        tmp.path(),
        "1789983533_orchestrator_legacy",
        &legacy,
        &[(12_495, 23, 12_288)],
    );

    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(
        spend.root.input_tokens, 12_495,
        "the per-turn record is authoritative, the zero header is not"
    );
    assert_eq!(spend.root.cached_input_tokens, 12_288);
}

/// A thread with no transcripts at all reports nothing, and a thread whose
/// transcripts recorded no spend must not claim usage: the UI replaces its live
/// bucket with this payload, so a false `has_usage` zeroes a turn in flight.
#[test]
fn reports_no_usage_for_an_unknown_or_spendless_thread() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let absent = thread_spend(tmp.path(), "thread-nope");
    assert!(!absent.found_transcript);
    assert_eq!(absent.root.input_tokens, 0);

    // A transcript that exists but recorded a single all-zero turn. The codec
    // omits an all-zero usage record, so this is the shape on disk: rows, no
    // usage.
    let thread = "thread-empty";
    let dir = tmp.path().join("session_raw");
    std::fs::create_dir_all(&dir).expect("create session_raw");
    let path = dir.join("1790000003_orchestrator_empty.jsonl");
    append_transcript_turn(
        &path,
        &[],
        &[TranscriptMessage::new("user", "hi")],
        &meta("orchestrator", "root", Some(thread)),
        None,
        Some("req-0"),
    )
    .expect("append turn");

    let spend = thread_spend(tmp.path(), thread);
    assert!(spend.found_transcript, "the file is there");
    assert_eq!(spend.root.turns, 0, "but it recorded no spend");
    assert_eq!(spend.root.input_tokens, 0);
}

/// A text-dialect tool round's issuing row carries a provenance-only record —
/// its calls, zero spend — beside the turn's real record on the final row.
/// It is not a turn that spent and must not take over the last-turn view.
#[test]
fn provenance_only_tool_round_records_are_not_counted_as_turns() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-text-dialect";
    let path = tmp
        .path()
        .join("session_raw")
        .join("1790000001_orchestrator_text.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create session_raw");

    let mut issuing = TranscriptMessage::assistant("");
    issuing.turn_usage = Some(TurnUsage {
        tool_calls: vec![crate::transcript::TranscriptToolCall {
            id: "call-1".into(),
            name: "web_search_tool".into(),
            arguments: "{}".into(),
            extra_content: None,
        }],
        ..turn_usage(0, 0, 0)
    });
    let rows = vec![
        TranscriptMessage::new("user", "q"),
        issuing,
        TranscriptMessage::new(
            "user",
            "[Tool results]\n<tool_result id=\"call-1\">\nok\n</tool_result>\n",
        ),
        TranscriptMessage::assistant("a"),
    ];
    append_transcript_turn(
        &path,
        &[],
        &rows,
        &meta("orchestrator", "root", Some(thread)),
        Some(&turn_usage(5_000, 50, 1_000)),
        Some("req-0"),
    )
    .expect("append turn");

    let spend = thread_spend(tmp.path(), thread);

    assert_eq!(spend.root.turns, 1);
    assert_eq!(spend.root.input_tokens, 5_000);
    assert_eq!(spend.root.last_input_tokens, 5_000);
}

/// Compaction appends a record carrying the reduced context; the rows it
/// summarised away stay in the file. Their usage is still spend, and a usage
/// record the replacement re-carries must not count twice.
#[test]
fn compacted_transcript_keeps_pre_compaction_spend_and_counts_each_record_once() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-compacted";
    let stem = "1790000004_orchestrator_compacted";
    let root_meta = meta("orchestrator", "root", Some(thread));
    write_transcript_turns(
        tmp.path(),
        stem,
        &root_meta,
        &[(10_000, 100, 0), (20_000, 200, 0)],
    );
    let path = tmp.path().join("session_raw").join(format!("{stem}.jsonl"));
    let persisted = vec![
        TranscriptMessage::new("user", "q0"),
        TranscriptMessage::assistant("a0"),
        TranscriptMessage::new("user", "q1"),
        TranscriptMessage::assistant("a1"),
    ];
    // The reduced context keeps the second turn's answer, re-carrying its
    // usage, then adds the third turn.
    // A kept row was read back from the transcript, so it owns its original
    // correlation id.
    let mut kept = TranscriptMessage::assistant("a1");
    kept.turn_usage = Some(turn_usage(20_000, 200, 0));
    kept.request_id = Some("req-1".into());
    kept.preserve_request_id = true;
    let reduced = vec![
        TranscriptMessage::new("user", "summary of q0/a0"),
        kept,
        TranscriptMessage::new("user", "q2"),
        TranscriptMessage::assistant("a2"),
    ];
    append_transcript_turn(
        &path,
        &persisted,
        &reduced,
        &root_meta,
        Some(&turn_usage(30_000, 300, 0)),
        Some("req-2"),
    )
    .expect("append compaction turn");

    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(spend.root.input_tokens, 60_000);
    assert_eq!(spend.root.output_tokens, 600);
    assert_eq!(spend.root.turns, 3);
    assert_eq!(spend.root.last_input_tokens, 30_000);
}

/// Two independent turns can record identical usage (same numbers, same
/// timestamp); each appended line is its own spend and both count.
#[test]
fn identical_usage_on_separate_appends_counts_each_turn() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-twins";
    write_transcript_turns(
        tmp.path(),
        "1790000005_orchestrator_twins",
        &meta("orchestrator", "root", Some(thread)),
        &[(1_000, 10, 0), (1_000, 10, 0)],
    );
    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(spend.root.input_tokens, 2_000);
    assert_eq!(spend.root.turns, 2);
}

/// A compaction turn whose usage is value-identical to an earlier turn's is
/// still its own spend: identity is the turn's `request_id`, not the numbers.
#[test]
fn compaction_turn_with_usage_identical_to_history_still_counts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-compacted-twin";
    let stem = "1790000006_orchestrator_twin";
    let root_meta = meta("orchestrator", "root", Some(thread));
    write_transcript_turns(tmp.path(), stem, &root_meta, &[(5_000, 50, 0)]);
    let path = tmp.path().join("session_raw").join(format!("{stem}.jsonl"));
    let persisted = vec![
        TranscriptMessage::new("user", "q0"),
        TranscriptMessage::assistant("a0"),
    ];
    let reduced = vec![
        TranscriptMessage::new("user", "summary"),
        TranscriptMessage::new("user", "q1"),
        TranscriptMessage::assistant("a1"),
    ];
    append_transcript_turn(
        &path,
        &persisted,
        &reduced,
        &root_meta,
        Some(&turn_usage(5_000, 50, 0)),
        Some("req-1"),
    )
    .expect("append compaction turn");
    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(spend.root.input_tokens, 10_000);
    assert_eq!(spend.root.turns, 2);
}

/// A compaction written without a `request_id` still counts its turn's spend.
#[test]
fn request_less_compaction_turn_still_counts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-compacted-anon";
    let stem = "1790000007_orchestrator_anon";
    let root_meta = meta("orchestrator", "root", Some(thread));
    write_transcript_turns(tmp.path(), stem, &root_meta, &[(5_000, 50, 0)]);
    let path = tmp.path().join("session_raw").join(format!("{stem}.jsonl"));
    let persisted = vec![
        TranscriptMessage::new("user", "q0"),
        TranscriptMessage::assistant("a0"),
    ];
    let reduced = vec![
        TranscriptMessage::new("user", "summary"),
        TranscriptMessage::new("user", "q1"),
        TranscriptMessage::assistant("a1"),
    ];
    append_transcript_turn(
        &path,
        &persisted,
        &reduced,
        &root_meta,
        Some(&turn_usage(7_000, 70, 0)),
        None,
    )
    .expect("append compaction turn");
    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(spend.root.input_tokens, 12_000);
    assert_eq!(spend.root.turns, 2);
}

/// An anonymous compaction turn whose usage equals an earlier turn's is still
/// its own spend: its answer row was not carried over from the prior context.
#[test]
fn request_less_compaction_with_usage_identical_to_history_still_counts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-compacted-anon-twin";
    let stem = "1790000008_orchestrator_anon_twin";
    let root_meta = meta("orchestrator", "root", Some(thread));
    write_transcript_turns(tmp.path(), stem, &root_meta, &[(5_000, 50, 0)]);
    let path = tmp.path().join("session_raw").join(format!("{stem}.jsonl"));
    let persisted = vec![
        TranscriptMessage::new("user", "q0"),
        TranscriptMessage::assistant("a0"),
    ];
    let reduced = vec![
        TranscriptMessage::new("user", "summary"),
        TranscriptMessage::new("user", "q1"),
        TranscriptMessage::assistant("a1"),
    ];
    append_transcript_turn(
        &path,
        &persisted,
        &reduced,
        &root_meta,
        Some(&turn_usage(5_000, 50, 0)),
        None,
    )
    .expect("append compaction turn");
    let spend = thread_spend(tmp.path(), thread);
    assert_eq!(spend.root.input_tokens, 10_000);
    assert_eq!(spend.root.turns, 2);
}

/// A successor generation starts with the rows it retained from its parent.
/// Those rows keep their original `request_id`, and their usage was already
/// counted in the sealed generation.
#[test]
fn retained_rows_in_a_successor_generation_are_not_counted_twice() {
    use crate::transcript::{FileTranscriptLocator, SessionRef, TranscriptLocator, TruncateCut};
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-generations";
    let locator = FileTranscriptLocator::new(tmp.path());
    let session = SessionRef::scoped(thread, "orchestrator");
    let first = locator
        .open_session(&session, meta("orchestrator", "root", Some(thread)))
        .expect("open session");
    first
        .append(TranscriptMessage::new("user", "q0"))
        .expect("append user");
    let mut answer = TranscriptMessage::assistant("a0");
    answer.turn_usage = Some(turn_usage(4_000, 40, 0));
    answer.request_id = Some("req-0".into());
    answer.preserve_request_id = true;
    first.append(answer).expect("append answer");

    let before = thread_spend(tmp.path(), thread);
    assert_eq!(before.root.input_tokens, 4_000);

    locator
        .truncate_into_next_generation(
            &session,
            TruncateCut::BeforeIndex(2),
            meta("orchestrator", "root", Some(thread)),
        )
        .expect("next generation");
    assert_eq!(
        find_root_transcripts_for_thread(tmp.path(), thread).len(),
        2,
        "both generations are root transcripts of the thread"
    );
    let after = thread_spend(tmp.path(), thread);
    assert_eq!(after.root.input_tokens, 4_000, "retained row counted twice");
    assert_eq!(after.root.turns, 1);
}

#[test]
fn carried_from_compares_typed_structure() {
    use crate::transcript::TranscriptToolCall;
    let call = |id: &str| TranscriptToolCall {
        id: id.into(),
        name: "shell".into(),
        arguments: "{}".into(),
        extra_content: None,
    };
    let prior = TranscriptMessage::assistant_with_calls("on it", vec![call("c1")]);
    assert!(carried_from(&prior, &prior.clone()));
    // Same role, text and usage, different calls: not the carried-over row.
    let other_calls = TranscriptMessage::assistant_with_calls("on it", vec![call("c2")]);
    assert!(!carried_from(&prior, &other_calls));
    assert!(!carried_from(
        &prior,
        &TranscriptMessage::assistant("on it")
    ));
    let tool = TranscriptMessage::tool_result("c1", "ok");
    assert!(!carried_from(
        &tool,
        &TranscriptMessage::tool_result("c2", "ok")
    ));
    assert!(!carried_from(&tool, &TranscriptMessage::tool("ok")));
}

/// Write one durable turn whose usage record is `usage`.
fn write_single_turn(workspace: &Path, stem: &str, thread: &str, usage: &TurnUsage) {
    let dir = workspace.join("session_raw");
    std::fs::create_dir_all(&dir).expect("create session_raw");
    let next = vec![
        TranscriptMessage::new("user", "q0"),
        TranscriptMessage::assistant("a0"),
    ];
    let mut turn_meta = meta("orchestrator", "root", Some(thread));
    turn_meta.turn_count = 1;
    append_transcript_turn(
        &dir.join(format!("{stem}.jsonl")),
        &[],
        &next,
        &turn_meta,
        Some(usage),
        Some("req-0"),
    )
    .expect("append turn");
}

/// A turn of many tool rounds sums every request into its spend. The context
/// gauge must read the final request's size, not that sum: a 72-call turn
/// summed to 5.7M input tokens against a 1M window while no request exceeded
/// ~103k.
#[test]
fn last_context_is_the_final_calls_size_not_the_turns_summed_spend() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "thread-many-calls";
    let mut usage = turn_usage(5_710_657, 30_790, 5_398_592);
    usage.iteration = 72;
    usage.usage.last_call_input = 103_000;
    usage.usage.last_call_output = 900;
    write_single_turn(tmp.path(), "1790000000_orchestrator_many", thread, &usage);

    let spend = thread_spend(tmp.path(), thread);

    // Spend is still the full sum: that is what was billed.
    assert_eq!(spend.root.input_tokens, 5_710_657);
    assert_eq!(spend.root.last_input_tokens, 5_710_657);
    // The gauge reads one request.
    assert_eq!(spend.root.last_context_tokens, 103_900);
    assert!(spend.root.last_context_tokens < spend.root.context_window);
}

/// A record written before the per-call fields existed carries only the
/// summed spend. A one-call turn's sum is exact; a many-call turn's falls back
/// to the mean request size instead of the sum.
#[test]
fn last_context_of_a_legacy_record_is_exact_for_one_call_and_averaged_for_many() {
    let mut single = turn_usage(40_000, 1_000, 0);
    single.iteration = 1;
    assert_eq!(context_tokens_of(&single), 41_000);

    let mut many = turn_usage(5_710_657, 30_790, 0);
    many.iteration = 72;
    assert_eq!(context_tokens_of(&many), (5_710_657 + 30_790) / 72);

    // A record that never stamped its call count counts as one call.
    let mut unnumbered = turn_usage(40_000, 1_000, 0);
    unnumbered.iteration = 0;
    assert_eq!(context_tokens_of(&unnumbered), 41_000);
}

/// The per-call fields survive the JSONL round trip, and a line written
/// before they existed still parses with them at zero.
#[test]
fn last_call_fields_round_trip_and_default_on_old_lines() {
    let usage = MessageUsage {
        input: 10,
        output: 2,
        last_call_input: 7,
        last_call_output: 1,
        ..Default::default()
    };
    let json = serde_json::to_string(&usage).expect("serialize");
    let back: MessageUsage = serde_json::from_str(&json).expect("parse");
    assert_eq!(back, usage);

    let old: MessageUsage = serde_json::from_str(
        r#"{"input":10,"output":2,"cached_input":0,"context_window":0,"cost_usd":0.0}"#,
    )
    .expect("parse old line");
    assert_eq!(old.last_call_input, 0);
    assert_eq!(old.last_call_output, 0);
}
