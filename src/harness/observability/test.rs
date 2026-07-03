//! Unit tests for the durable observability layer: journal round-trips
//! (in-memory + store-backed), status store get/list, and the redacting and
//! fan-out sinks.

use std::sync::Arc;

use crate::harness::events::{AgentEvent, EventListener, EventRecord, HarnessRunStatus, LimitKind};
use crate::harness::ids::{CallId, ComponentId, EventId, RunId, ThreadId};
use crate::harness::observability::{
    AgentLatencyMetrics, AgentObservation, FanOutSink, HarnessEventJournal, HarnessStatusStore,
    InMemoryEventJournal, InMemoryStatusStore, JournalSink, RedactingSink, StoreEventJournal,
};
use crate::harness::store::InMemoryAppendStore;

fn obs(run: &str, offset: u64, event: AgentEvent) -> AgentObservation {
    let run_id = RunId::new(run);
    AgentObservation {
        event_id: EventId::new(format!("evt-{offset}")),
        run_id: run_id.clone(),
        parent_run_id: None,
        root_run_id: run_id,
        offset,
        ts_ms: 1_000 + offset,
        event,
    }
}

#[tokio::test]
async fn in_memory_journal_append_read_round_trip() {
    let journal = InMemoryEventJournal::new();

    let a = obs(
        "run-1",
        0,
        AgentEvent::RunStarted {
            run_id: RunId::new("run-1"),
            thread_id: None,
        },
    );
    let b = obs(
        "run-1",
        1,
        AgentEvent::RunCompleted {
            run_id: RunId::new("run-1"),
        },
    );

    assert_eq!(journal.append(a.clone()).await.unwrap(), 0);
    assert_eq!(journal.append(b.clone()).await.unwrap(), 1);

    // A different run is isolated.
    journal
        .append(obs("run-2", 0, AgentEvent::StateUpdate))
        .await
        .unwrap();

    let all = journal.read_from("run-1", 0).await.unwrap();
    assert_eq!(all, vec![a, b.clone()]);

    let tail = journal.read_from("run-1", 1).await.unwrap();
    assert_eq!(tail, vec![b]);

    assert_eq!(journal.read_from("run-1", 9).await.unwrap().len(), 0);
    assert_eq!(journal.read_from("missing", 0).await.unwrap().len(), 0);
}

#[tokio::test]
async fn store_backed_journal_append_read_round_trip() {
    let journal = StoreEventJournal::new(InMemoryAppendStore::new());

    let a = obs("run-x", 0, AgentEvent::StreamClosed);
    let b = obs(
        "run-x",
        1,
        AgentEvent::LimitReached {
            kind: LimitKind::ModelCalls,
        },
    );

    assert_eq!(journal.append(a.clone()).await.unwrap(), 0);
    assert_eq!(journal.append(b.clone()).await.unwrap(), 1);

    let all = journal.read_from("run-x", 0).await.unwrap();
    assert_eq!(all, vec![a, b.clone()]);

    let tail = journal.read_from("run-x", 1).await.unwrap();
    assert_eq!(tail, vec![b]);
    assert_eq!(journal.read_from("run-x", 5).await.unwrap().len(), 0);
}

#[test]
fn agent_latency_metrics_include_model_tool_and_run_elapsed() {
    let run_id = RunId::new("run-latency");
    let model_id = CallId::new("model-1");
    let tool_id = CallId::new("tool-1");

    let observations = vec![
        obs(
            "run-latency",
            0,
            AgentEvent::RunStarted {
                run_id: run_id.clone(),
                thread_id: None,
            },
        ),
        obs(
            "run-latency",
            10,
            AgentEvent::ModelStarted {
                call_id: model_id.clone(),
                model: "gpt-test".to_string(),
            },
        ),
        obs(
            "run-latency",
            40,
            AgentEvent::ModelCompleted {
                call_id: model_id.clone(),
                usage: None,
                input: None,
                output: None,
            },
        ),
        obs(
            "run-latency",
            50,
            AgentEvent::ToolStarted {
                call_id: tool_id.clone(),
                tool_name: "lookup".to_string(),
            },
        ),
        obs(
            "run-latency",
            75,
            AgentEvent::ToolCompleted {
                call_id: tool_id.clone(),
                tool_name: "lookup".to_string(),
                input: None,
                output: None,
            },
        ),
        obs(
            "run-latency",
            90,
            AgentEvent::RunCompleted {
                run_id: run_id.clone(),
            },
        ),
    ];

    let metrics = AgentLatencyMetrics::from_observations(&observations);
    assert_eq!(metrics.run_elapsed_ms, Some(90));
    assert_eq!(metrics.model_calls.len(), 1);
    assert_eq!(metrics.model_calls[0].call_id, model_id);
    assert_eq!(metrics.model_calls[0].elapsed_ms, 30);
    assert_eq!(metrics.total_model_ms, 30);
    assert_eq!(metrics.average_model_ms(), Some(30));

    assert_eq!(metrics.tool_calls.len(), 1);
    assert_eq!(metrics.tool_calls[0].call_id, tool_id);
    assert_eq!(metrics.tool_calls[0].kind, "tool");
    assert_eq!(metrics.tool_calls[0].name, "lookup");
    assert_eq!(metrics.tool_calls[0].elapsed_ms, 25);
    assert_eq!(metrics.total_tool_ms, 25);
    assert_eq!(metrics.max_tool_ms, 25);
    assert_eq!(metrics.average_tool_ms(), Some(25));
}

#[tokio::test]
async fn status_store_put_get_list_by_thread() {
    let store = InMemoryStatusStore::new();

    let mut s1 = HarnessRunStatus::new(RunId::new("run-a"), ComponentId::new("agent"))
        .with_thread(ThreadId::new("thread-1"));
    s1.model_calls = 1;
    let s2 = HarnessRunStatus::new(RunId::new("run-b"), ComponentId::new("agent"))
        .with_thread(ThreadId::new("thread-1"));
    let s3 = HarnessRunStatus::new(RunId::new("run-c"), ComponentId::new("agent"))
        .with_thread(ThreadId::new("thread-2"));

    store.put_status(s1).await.unwrap();
    store.put_status(s2).await.unwrap();
    store.put_status(s3).await.unwrap();

    let got = store.get_status("run-a").await.unwrap().unwrap();
    assert_eq!(got.run_id, RunId::new("run-a"));
    assert_eq!(got.model_calls, 1);
    assert!(store.get_status("missing").await.unwrap().is_none());

    let mut thread1 = store.list_by_thread("thread-1").await.unwrap();
    thread1.sort_by(|a, b| a.run_id.as_str().cmp(b.run_id.as_str()));
    assert_eq!(thread1.len(), 2);
    assert_eq!(thread1[0].run_id, RunId::new("run-a"));
    assert_eq!(thread1[1].run_id, RunId::new("run-b"));

    assert_eq!(store.list_by_thread("thread-2").await.unwrap().len(), 1);
    assert_eq!(store.list_by_thread("none").await.unwrap().len(), 0);

    // put_status overwrites by run id.
    let updated = HarnessRunStatus::new(RunId::new("run-a"), ComponentId::new("agent"))
        .with_thread(ThreadId::new("thread-9"));
    store.put_status(updated).await.unwrap();
    assert_eq!(store.list_by_thread("thread-1").await.unwrap().len(), 1);
    assert_eq!(store.list_by_thread("thread-9").await.unwrap().len(), 1);
}

#[tokio::test]
async fn journal_window_and_filter_reads() {
    let journal = InMemoryEventJournal::new();
    journal
        .append(obs(
            "run-1",
            0,
            AgentEvent::RunStarted {
                run_id: RunId::new("run-1"),
                thread_id: None,
            },
        ))
        .await
        .unwrap();
    journal
        .append(obs("run-1", 1, AgentEvent::StateUpdate))
        .await
        .unwrap();
    journal
        .append(obs(
            "run-1",
            2,
            AgentEvent::RunCompleted {
                run_id: RunId::new("run-1"),
            },
        ))
        .await
        .unwrap();

    // Bounded window: at most 2 from offset 0.
    let window = journal.read_window("run-1", 0, 2).await.unwrap();
    assert_eq!(window.len(), 2);

    // Kind filter: only run.* lifecycle events.
    let filtered = journal
        .read_filtered("run-1", 0, &["run.started", "run.completed"])
        .await
        .unwrap();
    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().all(|o| o.event.kind().starts_with("run.")));

    // Empty kinds matches everything.
    let all = journal.read_filtered("run-1", 0, &[]).await.unwrap();
    assert_eq!(all.len(), 3);
}

#[tokio::test]
async fn status_store_lists_by_root_and_active() {
    let store = InMemoryStatusStore::new();
    let root = RunId::new("root");

    // Parent (root) running, child running, sibling completed under same root.
    let mut parent = HarnessRunStatus::new(root.clone(), ComponentId::new("agent"));
    parent.mark_running(crate::harness::ids::HarnessPhase::Model);
    let mut child = HarnessRunStatus::new(RunId::new("child"), ComponentId::new("agent"))
        .with_parent(root.clone(), root.clone());
    child.mark_running(crate::harness::ids::HarnessPhase::Tools);
    let mut done = HarnessRunStatus::new(RunId::new("done"), ComponentId::new("agent"))
        .with_parent(root.clone(), root.clone());
    done.mark_completed();
    // An unrelated run under a different root.
    let other = HarnessRunStatus::new(RunId::new("other"), ComponentId::new("agent"));

    store.put_status(parent).await.unwrap();
    store.put_status(child).await.unwrap();
    store.put_status(done).await.unwrap();
    store.put_status(other).await.unwrap();

    // Every descendant of the root run tree (parent + child + done).
    assert_eq!(store.list_by_root("root").await.unwrap().len(), 3);

    // Only non-terminal runs are active (parent + child; done is completed,
    // other is pending/... actually `other` is freshly-new = Pending -> active).
    let active = store.list_active().await.unwrap();
    assert!(active.iter().all(|s| s.run_id.as_str() != "done"));
}

#[tokio::test]
async fn journal_sink_persists_observations() {
    let journal = Arc::new(InMemoryEventJournal::new());
    let sink = JournalSink::new(journal.clone(), RunId::new("run-sink"));

    sink.on_event(&EventRecord {
        id: EventId::new("evt-0"),
        offset: 0,
        event: AgentEvent::RunStarted {
            run_id: RunId::new("run-sink"),
            thread_id: None,
        },
    });
    sink.on_event(&EventRecord {
        id: EventId::new("evt-1"),
        offset: 1,
        event: AgentEvent::StreamClosed,
    });

    let stored = journal.read_from("run-sink", 0).await.unwrap();
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[0].event.kind(), "run.started");
    assert_eq!(stored[0].offset, 0);
    assert_eq!(stored[1].event.kind(), "stream.closed");
    assert_eq!(stored[1].run_id, RunId::new("run-sink"));
    assert_eq!(stored[1].root_run_id, RunId::new("run-sink"));
}

/// Collects forwarded records for assertions.
struct Collector {
    records: std::sync::Mutex<Vec<EventRecord>>,
}

impl Collector {
    fn new() -> Self {
        Self {
            records: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn events(&self) -> Vec<EventRecord> {
        self.records.lock().unwrap().clone()
    }
}

impl EventListener for Collector {
    fn on_event(&self, record: &EventRecord) {
        self.records.lock().unwrap().push(record.clone());
    }
}

#[test]
fn redacting_sink_masks_secret_substrings() {
    let collector = Arc::new(Collector::new());
    let sink = RedactingSink::new(
        collector.clone(),
        vec!["sk-SUPERSECRET".to_string(), "hunter2".to_string()],
    );

    // The secret appears inside a string field of the event.
    sink.on_event(&EventRecord {
        id: EventId::new("evt-0"),
        offset: 0,
        event: AgentEvent::RunFailed {
            run_id: RunId::new("run-r"),
            error: "auth failed with key sk-SUPERSECRET and pw hunter2".to_string(),
        },
    });

    let events = collector.events();
    assert_eq!(events.len(), 1);
    match &events[0].event {
        AgentEvent::RunFailed { error, .. } => {
            assert!(!error.contains("sk-SUPERSECRET"));
            assert!(!error.contains("hunter2"));
            assert!(error.contains("[REDACTED]"));
        }
        other => panic!("unexpected event: {other:?}"),
    }
    // Structural fields are preserved.
    assert_eq!(events[0].id, EventId::new("evt-0"));
    assert_eq!(events[0].offset, 0);
}

#[test]
fn redacting_sink_custom_mask_and_passthrough() {
    let collector = Arc::new(Collector::new());
    let sink =
        RedactingSink::new(collector.clone(), vec!["topsecret".to_string()]).with_mask("***");

    // No secret present: forwarded unchanged.
    sink.on_event(&EventRecord {
        id: EventId::new("evt-0"),
        offset: 0,
        event: AgentEvent::MiddlewareFailed {
            name: "auth".to_string(),
            error: "boom topsecret boom".to_string(),
        },
    });

    match &collector.events()[0].event {
        AgentEvent::MiddlewareFailed { error, name } => {
            assert_eq!(name, "auth");
            assert_eq!(error, "boom *** boom");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn redacting_sink_empty_secrets_forwards_unchanged() {
    // With no secrets configured the sink takes the fast path and forwards the
    // original record untouched.
    let collector = Arc::new(Collector::new());
    let sink = RedactingSink::new(collector.clone(), Vec::new());

    sink.on_event(&EventRecord {
        id: EventId::new("evt-0"),
        offset: 7,
        event: AgentEvent::RunFailed {
            run_id: RunId::new("run-r"),
            error: "nothing to redact here".to_string(),
        },
    });

    let events = collector.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].id, EventId::new("evt-0"));
    assert_eq!(events[0].offset, 7);
    match &events[0].event {
        AgentEvent::RunFailed { error, .. } => {
            assert_eq!(error, "nothing to redact here");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn fan_out_sink_reaches_all_listeners() {
    let a = Arc::new(Collector::new());
    let b = Arc::new(Collector::new());
    let c = Arc::new(Collector::new());

    let sink = FanOutSink::new()
        .with(a.clone())
        .with(b.clone())
        .with(c.clone());
    assert_eq!(sink.len(), 3);

    sink.on_event(&EventRecord {
        id: EventId::new("evt-0"),
        offset: 0,
        event: AgentEvent::StateUpdate,
    });

    assert_eq!(a.events().len(), 1);
    assert_eq!(b.events().len(), 1);
    assert_eq!(c.events().len(), 1);
    assert_eq!(a.events()[0].event.kind(), "state.update");
}
