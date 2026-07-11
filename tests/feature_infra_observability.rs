//! Feature/integration tests for the harness durable-observability
//! infrastructure (`harness::observability`).
//!
//! Complements the existing `e2e_observability.rs` (which drives a full harness
//! run through the journal + Langfuse exporter) by exercising the durable
//! surfaces directly and offline: latency rollups derived from observation
//! timestamps, the in-memory journal's offset/retention/stale-offset contract,
//! the status store's lineage queries with terminal-vs-active eviction, and the
//! `FanOutSink` / `RedactingSink` listener sinks.
//!
//! Deterministic and offline — observations are hand-built with fixed
//! timestamps.

use std::sync::Arc;

use tinyagents::harness::events::{
    AgentEvent, EventListener, EventRecord, HarnessRunStatus, RecordingListener,
};
use tinyagents::harness::ids::{CallId, ComponentId, EventId, ExecutionStatus, RunId, ThreadId};
use tinyagents::harness::observability::{
    AgentLatencyMetrics, AgentObservation, FanOutSink, HarnessEventJournal, HarnessStatusStore,
    InMemoryEventJournal, InMemoryStatusStore, RedactingSink,
};

// ── Latency metrics from observations ───────────────────────────────────────

fn obs(run: &str, offset: u64, ts_ms: u64, event: AgentEvent) -> AgentObservation {
    let run_id = RunId::new(run);
    AgentObservation {
        event_id: EventId::new(format!("e-{offset}")),
        run_id: run_id.clone(),
        parent_run_id: None,
        root_run_id: run_id,
        offset,
        ts_ms,
        event,
    }
}

#[test]
fn latency_metrics_correlate_started_and_completed_by_call_id() {
    let observations = vec![
        obs(
            "r1",
            0,
            1_000,
            AgentEvent::RunStarted {
                run_id: RunId::new("r1"),
                thread_id: None,
            },
        ),
        obs(
            "r1",
            1,
            1_000,
            AgentEvent::ModelStarted {
                call_id: CallId::new("m1"),
                model: "test-model".into(),
            },
        ),
        obs(
            "r1",
            2,
            1_150,
            AgentEvent::ModelCompleted {
                call_id: CallId::new("m1"),
                started_at_ms: None,
                usage: None,
                input: None,
                output: None,
            },
        ),
        obs(
            "r1",
            3,
            1_150,
            AgentEvent::ToolStarted {
                call_id: CallId::new("t1"),
                tool_name: "search".into(),
            },
        ),
        obs(
            "r1",
            4,
            1_200,
            AgentEvent::ToolCompleted {
                call_id: CallId::new("t1"),
                tool_name: "search".into(),
                started_at_ms: None,
                input: None,
                output: None,
                duration_ms: None,
                output_bytes: None,
                error: None,
            },
        ),
        obs(
            "r1",
            5,
            1_500,
            AgentEvent::RunCompleted {
                run_id: RunId::new("r1"),
            },
        ),
    ];

    let metrics = AgentLatencyMetrics::from_observations(&observations);
    assert_eq!(metrics.run_elapsed_ms, Some(500)); // 1500 - 1000
    assert_eq!(metrics.model_calls.len(), 1);
    assert_eq!(metrics.model_calls[0].elapsed_ms, 150); // 1150 - 1000
    assert_eq!(metrics.model_calls[0].name, "test-model");
    assert_eq!(metrics.average_model_ms(), Some(150));
    assert_eq!(metrics.tool_calls.len(), 1);
    assert_eq!(metrics.tool_calls[0].elapsed_ms, 50); // 1200 - 1150
    assert_eq!(metrics.max_tool_ms, 50);
}

#[test]
fn latency_metrics_ignore_calls_without_a_completion() {
    // A ModelStarted with no matching ModelCompleted yields no model latency.
    let observations = vec![obs(
        "r2",
        0,
        0,
        AgentEvent::ModelStarted {
            call_id: CallId::new("m1"),
            model: "m".into(),
        },
    )];
    let metrics = AgentLatencyMetrics::from_observations(&observations);
    assert!(metrics.model_calls.is_empty());
    assert_eq!(metrics.total_model_ms, 0);
    assert_eq!(metrics.run_elapsed_ms, None);
}

// ── In-memory journal: offsets, retention, stale offsets ────────────────────

#[tokio::test]
async fn journal_appends_at_dense_offsets_and_replays_from_offset() {
    let journal = InMemoryEventJournal::new();
    for i in 0..3 {
        let offset = journal
            .append(obs(
                "run",
                i,
                i,
                AgentEvent::RunStarted {
                    run_id: RunId::new("run"),
                    thread_id: None,
                },
            ))
            .await
            .unwrap();
        assert_eq!(offset, i);
    }
    assert_eq!(journal.len("run"), 3);

    // Replay from 0 returns all; from a mid-offset returns the tail.
    assert_eq!(journal.read_from("run", 0).await.unwrap().len(), 3);
    assert_eq!(journal.read_from("run", 2).await.unwrap().len(), 1);
    // Unknown run replays empty rather than erroring.
    assert!(journal.read_from("nope", 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn journal_read_filtered_selects_by_event_kind() {
    let journal = InMemoryEventJournal::new();
    journal
        .append(obs(
            "run",
            0,
            0,
            AgentEvent::RunStarted {
                run_id: RunId::new("run"),
                thread_id: None,
            },
        ))
        .await
        .unwrap();
    journal
        .append(obs(
            "run",
            1,
            0,
            AgentEvent::ToolStarted {
                call_id: CallId::new("t1"),
                tool_name: "x".into(),
            },
        ))
        .await
        .unwrap();

    let only_tools = journal
        .read_filtered("run", 0, &["tool.started"])
        .await
        .unwrap();
    assert_eq!(only_tools.len(), 1);
    assert_eq!(only_tools[0].event.kind(), "tool.started");
    // A bounded replay window truncates.
    assert_eq!(journal.read_window("run", 0, 1).await.unwrap().len(), 1);
}

#[tokio::test]
async fn journal_evicts_oldest_run_and_flags_stale_offsets() {
    // Cap of one run: appending a second evicts the first.
    let journal = InMemoryEventJournal::with_max_runs(1);
    journal
        .append(obs(
            "old",
            0,
            0,
            AgentEvent::RunStarted {
                run_id: RunId::new("old"),
                thread_id: None,
            },
        ))
        .await
        .unwrap();
    journal
        .append(obs(
            "new",
            0,
            0,
            AgentEvent::RunStarted {
                run_id: RunId::new("new"),
                thread_id: None,
            },
        ))
        .await
        .unwrap();

    assert_eq!(journal.run_count(), 1);
    // A consumer resuming from a now-evicted run's saved offset is told the
    // offset is stale rather than silently getting an empty replay.
    assert!(journal.read_from("old", 0).await.is_err());
}

// ── Status store: lineage queries + eviction ────────────────────────────────

fn status(run: &str, root: &str, thread: Option<&str>, terminal: bool) -> HarnessRunStatus {
    let mut s = HarnessRunStatus::new(RunId::new(run), ComponentId::new("agent"));
    s.root_run_id = RunId::new(root);
    s.thread_id = thread.map(ThreadId::new);
    if terminal {
        s.mark_completed();
    }
    s
}

#[tokio::test]
async fn status_store_answers_thread_root_and_active_queries() {
    let store = InMemoryStatusStore::new();
    // Two children of one root; one still active, one terminal.
    store
        .put_status(status("child-a", "root", Some("thread-1"), false))
        .await
        .unwrap();
    store
        .put_status(status("child-b", "root", Some("thread-1"), true))
        .await
        .unwrap();
    store
        .put_status(status("other", "other-root", Some("thread-2"), true))
        .await
        .unwrap();

    assert_eq!(store.list_by_root("root").await.unwrap().len(), 2);
    assert_eq!(store.list_by_thread("thread-1").await.unwrap().len(), 2);
    assert_eq!(store.list_by_thread("thread-2").await.unwrap().len(), 1);

    // Only the non-terminal child is active.
    let active = store.list_active().await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].run_id.as_str(), "child-a");

    let fetched = store.get_status("child-b").await.unwrap().unwrap();
    assert_eq!(fetched.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn status_store_evicts_terminal_runs_but_keeps_active_ones() {
    // Cap of one: an active run must never be evicted to make room.
    let store = InMemoryStatusStore::with_max_runs(1);
    store
        .put_status(status("active", "active", None, false))
        .await
        .unwrap();
    store
        .put_status(status("terminal", "terminal", None, true))
        .await
        .unwrap();

    // The active run survives even though the cap was exceeded.
    assert!(store.get_status("active").await.unwrap().is_some());
}

// ── Listener sinks: fan-out + redaction ─────────────────────────────────────

fn record(offset: u64, event: AgentEvent) -> EventRecord {
    EventRecord {
        id: EventId::new(format!("e-{offset}")),
        offset,
        event,
    }
}

#[test]
fn fan_out_sink_broadcasts_to_every_listener() {
    let a = Arc::new(RecordingListener::new());
    let b = Arc::new(RecordingListener::new());
    let sink = FanOutSink::new().with(a.clone()).with(b.clone());
    assert_eq!(sink.len(), 2);

    sink.on_event(&record(
        0,
        AgentEvent::RunStarted {
            run_id: RunId::new("r"),
            thread_id: None,
        },
    ));
    assert_eq!(a.events().len(), 1);
    assert_eq!(b.events().len(), 1);
}

#[test]
fn redacting_sink_masks_secrets_in_string_fields_before_forwarding() {
    let downstream = Arc::new(RecordingListener::new());
    let sink = RedactingSink::new(downstream.clone(), vec!["sk-secret".to_string()]);

    // The secret rides in the `model` field of ModelStarted.
    sink.on_event(&record(
        0,
        AgentEvent::ModelStarted {
            call_id: CallId::new("m1"),
            model: "provider/sk-secret".into(),
        },
    ));

    let seen = downstream.events();
    assert_eq!(seen.len(), 1);
    match &seen[0].event {
        AgentEvent::ModelStarted { model, .. } => {
            assert!(!model.contains("sk-secret"));
            assert!(model.contains("[REDACTED]"));
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn redacting_sink_with_no_secrets_is_pass_through() {
    let downstream = Arc::new(RecordingListener::new());
    let sink = RedactingSink::new(downstream.clone(), Vec::new());
    sink.on_event(&record(
        0,
        AgentEvent::RunCompleted {
            run_id: RunId::new("r"),
        },
    ));
    assert_eq!(downstream.events().len(), 1);
}
