//! Turn-ordering tests for the transcript view that go through the **real
//! writer** (`append_transcript_turn`) rather than hand-built JSONL: tool-call
//! de-duplication, tool-result unwrapping, per-step iterations, sub-agent
//! placement, and compaction-generation chains.

use super::project::{project_records, project_thread, resolve_files};
use super::types::{DisplayItem, ToolCallStatus, TranscriptSubagentStatus};
use crate::transcript::{
    self, SessionRef, TranscriptMessage, TranscriptMeta, TranscriptToolCall, TurnUsage,
    read_transcript, read_transcript_display,
};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn meta(thread_id: &str, session_id: Option<String>, parent: Option<String>) -> TranscriptMeta {
    TranscriptMeta {
        session_id,
        parent_session_id: parent,
        agent_name: "orchestrator".into(),
        agent_id: Some("orchestrator".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: Some("anthropic".into()),
        model: Some("claude-x".into()),
        created: "2023-11-14T22:00:00+00:00".into(),
        updated: "2023-11-14T22:00:00+00:00".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some(thread_id.into()),
        task_id: None,
    }
}

fn rfc3339(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .unwrap()
        .to_rfc3339()
}

fn usage(iteration: u32, unix: i64, tool_calls: Vec<TranscriptToolCall>) -> TurnUsage {
    TurnUsage {
        provider: "anthropic".into(),
        model: "claude-x".into(),
        usage: transcript::MessageUsage {
            input: 10,
            output: 5,
            cached_input: 0,
            context_window: 0,
            cost_usd: 0.001,
            ..Default::default()
        },
        ts: rfc3339(unix),
        reasoning_content: None,
        tool_calls,
        iteration,
    }
}

fn call(id: &str, name: &str, arguments: &str) -> TranscriptToolCall {
    TranscriptToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
        extra_content: None,
    }
}

/// An assistant row as the native dialect persists it: the provider replay
/// envelope, with its reasoning in `extra_metadata`.
fn envelope(content: &str, calls: &[(&str, &str, &str)], reasoning: &str) -> TranscriptMessage {
    let calls: Vec<serde_json::Value> = calls
        .iter()
        .map(|(id, name, args)| serde_json::json!({"id": id, "name": name, "arguments": args}))
        .collect();
    let mut row = TranscriptMessage::assistant(
        serde_json::json!({"content": content, "tool_calls": calls}).to_string(),
    );
    row.extra_metadata = Some(serde_json::json!({ "reasoning_content": reasoning }));
    row
}

/// A tool-result row as the native dialect persists it: wrapped.
fn tool_result(id: &str, output: &str) -> TranscriptMessage {
    let mut row = TranscriptMessage::new(
        "tool",
        serde_json::json!({"tool_call_id": id, "content": output}).to_string(),
    );
    row.id = Some(id.into());
    row
}

fn final_answer(text: &str, reasoning: &str) -> TranscriptMessage {
    let mut row = TranscriptMessage::assistant(text);
    row.extra_metadata = Some(serde_json::json!({ "reasoning_content": reasoning }));
    row
}

/// One two-step tool turn: step 1 narrates and calls `get_weather`, step 2
/// calls two tools in parallel with no narration, step 3 answers.
fn weather_turn() -> Vec<TranscriptMessage> {
    vec![
        TranscriptMessage::new("system", "policy"),
        TranscriptMessage::new("user", "Weather in NYC and SF?"),
        envelope(
            "Let me check.",
            &[("c1", "get_weather", r#"{"city":"NYC"}"#)],
            "think one",
        ),
        tool_result("c1", "72F"),
        envelope(
            "",
            &[
                ("c2", "get_weather", r#"{"city":"SF"}"#),
                ("c3", "web_search", r#"{"q":"sf fog"}"#),
            ],
            "think two",
        ),
        tool_result("c2", "60F"),
        tool_result("c3", "foggy"),
        final_answer("NYC 72F, SF 60F and foggy.", "think three"),
    ]
}

fn write_turn(path: &Path, rows: &[TranscriptMessage], turn_usage: &TurnUsage) {
    transcript::append_transcript_turn(
        path,
        &[],
        rows,
        &meta("thr_w", None, None),
        Some(turn_usage),
        Some("req-1"),
    )
    .unwrap();
}

fn assert_weather_projection(items: &[DisplayItem]) {
    let tools: Vec<(&str, u32, &str, ToolCallStatus)> = items
        .iter()
        .filter_map(|item| match item {
            DisplayItem::ToolCall {
                call_id,
                iteration,
                result,
                status,
                ..
            } => Some((
                call_id.as_str(),
                iteration.unwrap_or_default(),
                result.as_deref().unwrap_or_default(),
                *status,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        tools,
        vec![
            ("c1", 1, "72F", ToolCallStatus::Success),
            ("c2", 2, "60F", ToolCallStatus::Success),
            ("c3", 2, "foggy", ToolCallStatus::Success),
        ],
        "each call once, settled, with its unwrapped output: {items:#?}"
    );

    let answers: Vec<(&str, bool, Option<u32>)> = items
        .iter()
        .filter_map(|item| match item {
            DisplayItem::AssistantMessage {
                content,
                interim,
                iteration,
                ..
            } => Some((content.as_str(), *interim, *iteration)),
            _ => None,
        })
        .collect();
    assert_eq!(
        answers,
        vec![
            ("Let me check.", true, Some(1)),
            ("NYC 72F, SF 60F and foggy.", false, Some(3)),
        ]
    );

    // Each reasoning block carries the iteration of the step it precedes.
    let reasoning: Vec<(&str, Option<u32>)> = items
        .iter()
        .filter_map(|item| match item {
            DisplayItem::Reasoning { text, iteration } => Some((text.as_str(), *iteration)),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning,
        vec![
            ("think one", Some(1)),
            ("think two", Some(2)),
            ("think three", Some(3)),
        ]
    );
}

/// The shape every turn written before this fix has on disk: the turn's
/// aggregate tool outcomes were copied onto the usage of the final answer,
/// which the writer then stamped as that row's `tool_calls`.
#[test]
fn real_writer_turn_with_aggregate_usage_calls_projects_each_call_once() {
    let dir = TempDir::new().unwrap();
    let path = transcript::resolve_keyed_transcript_path(dir.path(), "1_orchestrator").unwrap();
    let aggregate = vec![
        call("c1", "get_weather", r#"{"city":"NYC"}"#),
        call("c2", "get_weather", r#"{"city":"SF"}"#),
        call("c3", "web_search", r#"{"q":"sf fog"}"#),
    ];
    write_turn(&path, &weather_turn(), &usage(3, 1_700_000_000, aggregate));

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.lines()
            .any(|line| line.contains("NYC 72F") && line.contains("\"tool_calls\"")),
        "precondition: the legacy final answer row carries the duplicated calls"
    );
    let display = read_transcript_display(&path).unwrap();
    assert_weather_projection(&project_records(&display.records));
}

/// The shape the codec writes now: usage without tool calls. Every step is
/// also stamped with its own iteration by the writer.
#[test]
fn real_writer_turn_projects_steps_in_order() {
    let dir = TempDir::new().unwrap();
    let path = transcript::resolve_keyed_transcript_path(dir.path(), "2_orchestrator").unwrap();
    write_turn(&path, &weather_turn(), &usage(3, 1_700_000_000, Vec::new()));

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.lines()
            .any(|line| line.contains("NYC 72F") && line.contains("\"tool_calls\"")),
        "the final answer row carries no tool calls"
    );
    let display = read_transcript_display(&path).unwrap();
    assert_weather_projection(&project_records(&display.records));
}

/// Only the native `{tool_call_id, content}` wrapper is unwrapped; a tool
/// whose real output is JSON with other keys is shown verbatim.
#[test]
fn tool_output_that_is_not_the_replay_wrapper_is_kept_verbatim() {
    let dir = TempDir::new().unwrap();
    let path = transcript::resolve_keyed_transcript_path(dir.path(), "3_orchestrator").unwrap();
    let mut rows = vec![
        TranscriptMessage::new("user", "go"),
        envelope("", &[("c1", "fetch_json", "{}")], ""),
    ];
    let mut own_json = TranscriptMessage::new("tool", r#"{"content":"x","status":200}"#);
    own_json.id = Some("c1".into());
    rows.push(own_json);
    rows.push(final_answer("done", ""));
    write_turn(&path, &rows, &usage(2, 1_700_000_000, Vec::new()));

    let display = read_transcript_display(&path).unwrap();
    let result = project_records(&display.records)
        .into_iter()
        .find_map(|item| match item {
            DisplayItem::ToolCall { result, .. } => result,
            _ => None,
        })
        .expect("tool result projected");
    assert_eq!(result, r#"{"content":"x","status":200}"#);
}

/// Write a session-identity root for `thread_id` (the stem the host binds via
/// `SessionRef::scoped`) and return its path.
fn session_root(workspace: &Path, thread_id: &str) -> (SessionRef, PathBuf) {
    let session = SessionRef::scoped(thread_id, "orchestrator");
    let path =
        transcript::resolve_keyed_transcript_path(workspace, &transcript::session_stem(&session))
            .unwrap();
    (session, path)
}

/// A session-identity root is named `{thread}.{agent}` (with digests) while a
/// sub-agent's file chains the parent's *session key* (`{unix}_{agent}…`), so
/// prefix discovery never found it. It is discovered by `_meta.thread_id`,
/// placed right after the call that spawned it, and keyed by its task id.
#[test]
fn subagent_of_a_session_root_is_discovered_and_placed_after_its_spawning_call() {
    let dir = TempDir::new().unwrap();
    let thread_id = "thr_sub_place";
    let (session, root) = session_root(dir.path(), thread_id);
    let root_meta = meta(thread_id, Some(session.session_id()), None);

    let turn1 = vec![
        TranscriptMessage::new("user", "hi"),
        final_answer("hello", ""),
    ];
    transcript::append_transcript_turn(
        &root,
        &[],
        &turn1,
        &root_meta,
        Some(&usage(1, 1_700_000_000, Vec::new())),
        Some("req-1"),
    )
    .unwrap();
    let persisted = read_transcript(&root).unwrap().messages;
    let mut turn2 = persisted.clone();
    turn2.extend([
        TranscriptMessage::new("user", "research bali"),
        envelope(
            "On it.",
            &[
                ("c-fetch", "web_fetch", r#"{"url":"https://x"}"#),
                ("c-research", "research", r#"{"prompt":"bali"}"#),
            ],
            "",
        ),
        tool_result("c-fetch", "page"),
        tool_result("c-research", "Bali is great."),
        final_answer("Here's the plan.", ""),
    ]);
    transcript::append_transcript_turn(
        &root,
        &persisted,
        &turn2,
        &root_meta,
        Some(&usage(2, 1_700_000_200, Vec::new())),
        Some("req-2"),
    )
    .unwrap();

    // Spawned during turn 2 (after turn 1 committed at …000, before …200).
    let child_stem = "1699999990_orchestrator_thread-x__1700000100_000000001_researcher_sub-abc";
    let child = transcript::resolve_keyed_transcript_path(dir.path(), child_stem).unwrap();
    let mut child_meta = meta(thread_id, None, None);
    child_meta.agent_name = "researcher".into();
    child_meta.agent_id = Some("researcher".into());
    child_meta.agent_type = Some("subagent".into());
    child_meta.task_id = Some("sub-abc-123".into());
    transcript::write_transcript(
        &child,
        &[
            TranscriptMessage::new("user", "bali"),
            TranscriptMessage::assistant("Bali is great."),
        ],
        &child_meta,
        None,
    )
    .unwrap();

    let (_, subs) = resolve_files(dir.path(), thread_id).expect("thread resolves");
    assert_eq!(subs, vec![child.clone()], "discovered by _meta.thread_id");

    let projected = project_thread(dir.path(), thread_id).expect("project");
    let items = &projected.items;
    let research_call = items
        .iter()
        .position(
            |item| matches!(item, DisplayItem::ToolCall { call_id, .. } if call_id == "c-research"),
        )
        .expect("research call projected");
    match &items[research_call + 1] {
        DisplayItem::Subagent {
            id,
            agent_id,
            task_id,
            call_id,
            status,
            request_id,
            items,
            ..
        } => {
            assert_eq!(id, "sub-abc-123");
            assert_eq!(agent_id.as_deref(), Some("researcher"));
            assert_eq!(task_id.as_deref(), Some("sub-abc-123"));
            assert_eq!(call_id.as_deref(), Some("c-research"));
            assert_eq!(*status, TranscriptSubagentStatus::Completed);
            assert_eq!(request_id.as_deref(), Some("req-2"));
            assert!(items.iter().any(|inner| matches!(
                inner,
                DisplayItem::AssistantMessage { content, .. } if content == "Bali is great."
            )));
        }
        other => panic!("expected the sub-agent right after its call, got {other:?}"),
    }
    // Not also appended after everything else.
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, DisplayItem::Subagent { .. }))
            .count(),
        1
    );
    assert!(
        matches!(items.last(), Some(DisplayItem::AssistantMessage { content, .. }) if content == "Here's the plan."),
        "the turn's final answer still closes the list"
    );
}

/// A delegation reported incomplete is an incomplete run, whatever the
/// child's last row says. The legacy text marker is still honoured.
#[test]
fn subagent_status_follows_a_legacy_incomplete_marker() {
    assert_eq!(
        status_for_delegation_result("[SUBAGENT_INCOMPLETE] gave up"),
        TranscriptSubagentStatus::Incomplete
    );
}

/// The typed `"status": "incomplete"` payload replaces the text marker.
#[test]
fn subagent_status_follows_a_typed_incomplete_result() {
    assert_eq!(
        status_for_delegation_result(
            r#"{"job_id":"j","status":"incomplete","incomplete_kind":"timeout"}"#
        ),
        TranscriptSubagentStatus::Incomplete
    );
    assert_ne!(
        status_for_delegation_result(r#"{"status":"incomplete"}"#),
        TranscriptSubagentStatus::Incomplete,
        "foreign JSON without the harness job keys is not trusted"
    );
    assert_eq!(
        status_for_delegation_result(r#"{"job_id":"j","status":"completed"}"#),
        TranscriptSubagentStatus::Completed
    );
}

fn status_for_delegation_result(result: &str) -> TranscriptSubagentStatus {
    let dir = TempDir::new().unwrap();
    let root_stem = "900_orchestrator";
    let thread_id = "thr_sub_fail";
    let root = transcript::resolve_keyed_transcript_path(dir.path(), root_stem).unwrap();
    transcript::append_transcript_turn(
        &root,
        &[],
        &[
            TranscriptMessage::new("user", "go"),
            envelope("", &[("c1", "delegate_coder", "{}")], ""),
            tool_result("c1", result),
            final_answer("Sorry.", ""),
        ],
        &meta(thread_id, None, None),
        Some(&usage(2, 1_700_000_000, Vec::new())),
        Some("req-1"),
    )
    .unwrap();
    let child = transcript::resolve_keyed_transcript_path(
        dir.path(),
        &format!("{root_stem}__1699999999_000000001_coder_sub-1"),
    )
    .unwrap();
    let mut child_meta = meta(thread_id, None, None);
    child_meta.agent_id = Some("coder".into());
    transcript::write_transcript(
        &child,
        &[
            TranscriptMessage::new("user", "task"),
            TranscriptMessage::assistant("partial thoughts"),
        ],
        &child_meta,
        None,
    )
    .unwrap();

    let projected = project_thread(dir.path(), thread_id).expect("project");
    let status = projected
        .items
        .iter()
        .find_map(|item| match item {
            DisplayItem::Subagent {
                status, call_id, ..
            } => Some((*status, call_id.clone())),
            _ => None,
        })
        .expect("sub-agent projected");
    assert_eq!(status.1, Some("c1".to_string()));
    status.0
}

/// A compaction opens `{stem}.g1`, which inherits `created` and starts with
/// the retained rows rewritten. The chain is projected oldest-first, the
/// retained rows once, with a compaction marker at the seam — and an adopted
/// legacy root of the same thread is not projected a second time.
#[test]
fn generation_chain_projects_in_order_without_duplicating_retained_rows() {
    let dir = TempDir::new().unwrap();
    let thread_id = "thr_generations";
    let (session, g0) = session_root(dir.path(), thread_id);
    let successor = session.next_generation();
    let g1 = transcript::resolve_keyed_transcript_path(
        dir.path(),
        &transcript::session_stem(&successor),
    )
    .unwrap();

    // A pre-identity root of the same thread, already adopted into g0.
    let legacy =
        transcript::resolve_keyed_transcript_path(dir.path(), "1600000000_orchestrator").unwrap();
    transcript::append_transcript_turn(
        &legacy,
        &[],
        &[
            TranscriptMessage::new("user", "one"),
            final_answer("answer one", ""),
        ],
        &meta(thread_id, None, None),
        None,
        Some("req-1"),
    )
    .unwrap();

    let g0_meta = meta(thread_id, Some(session.session_id()), None);
    transcript::append_transcript_turn(
        &g0,
        &[],
        &[
            TranscriptMessage::new("user", "one"),
            final_answer("answer one", ""),
        ],
        &g0_meta,
        None,
        Some("req-1"),
    )
    .unwrap();
    let after_one = read_transcript(&g0).unwrap().messages;
    let mut two = after_one.clone();
    two.extend([
        TranscriptMessage::new("user", "two"),
        final_answer("answer two", ""),
    ]);
    transcript::append_transcript_turn(&g0, &after_one, &two, &g0_meta, None, Some("req-2"))
        .unwrap();

    // Compaction during turn 3: turn 2 is retained, turn 1 summarised away.
    let persisted = read_transcript(&g0).unwrap().messages;
    let mut retained: Vec<TranscriptMessage> = persisted[2..].to_vec();
    retained.extend([
        TranscriptMessage::new("user", "three"),
        final_answer("answer three", ""),
    ]);
    let g1_meta = meta(
        thread_id,
        Some(successor.session_id()),
        Some(session.session_id()),
    );
    transcript::append_transcript_turn(&g1, &[], &retained, &g1_meta, None, Some("req-3")).unwrap();

    let (roots, _) = resolve_files(dir.path(), thread_id).expect("thread resolves");
    assert_eq!(
        roots,
        vec![g0.clone(), g1.clone()],
        "chain order, legacy dropped"
    );

    let items = project_thread(dir.path(), thread_id)
        .expect("project")
        .items;
    let users: Vec<&str> = items
        .iter()
        .filter_map(|item| match item {
            DisplayItem::UserMessage { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(users, vec!["one", "two", "three"], "{items:#?}");
    let answers = items
        .iter()
        .filter(|item| matches!(item, DisplayItem::AssistantMessage { content, .. } if content == "answer two"))
        .count();
    assert_eq!(answers, 1, "the retained answer renders once");
    assert!(items.iter().any(|item| matches!(
        item,
        DisplayItem::Compaction { kept_count, .. } if *kept_count == 2
    )));
}

#[test]
fn get_page_missing_thread_is_empty_not_error() {
    let dir = TempDir::new().unwrap();
    let page = super::get_page(
        dir.path(),
        "no_such_thread",
        None,
        Some(super::DEFAULT_LIMIT),
    );
    assert!(!page.has_transcript);
    assert_eq!(page.total, 0);
    assert!(page.items.is_empty());
}
