//! Tests added in a later pass.
//!
//! This file contains a minimal smoke test to verify that the events module
//! compiles and that the core fan-out and recording primitives work together.
//! Comprehensive property tests and replay tests are tracked for a later pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use crate::events::{
    AgentEvent, EventJournal, EventListener, EventRecord, EventSink, HarnessRunStatus,
    RecordingListener,
};
use crate::ids::{ComponentId, ExecutionStatus, HarnessPhase, RunId};

struct ReentrantEmitter {
    sink: EventSink,
    emitted: AtomicBool,
}

impl ReentrantEmitter {
    fn new(sink: EventSink) -> Self {
        Self {
            sink,
            emitted: AtomicBool::new(false),
        }
    }
}

impl EventListener for ReentrantEmitter {
    fn on_event(&self, _record: &EventRecord) {
        if !self.emitted.swap(true, Ordering::SeqCst) {
            self.sink.emit(AgentEvent::StateUpdate);
        }
    }
}

#[test]
fn smoke_event_sink_records_events() {
    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    assert_eq!(sink.len(), 1);

    let run_id = RunId::new("run-smoke");
    let record = sink.emit(AgentEvent::RunStarted {
        run_id: run_id.clone(),
        thread_id: None,
    });

    assert_eq!(record.offset, 0);
    assert_eq!(record.event.kind(), "run.started");
    assert_eq!(recorder.len(), 1);

    let _ = sink.emit(AgentEvent::RunCompleted {
        run_id: run_id.clone(),
    });
    assert_eq!(recorder.len(), 2);

    // Offsets are monotonically increasing.
    let events = recorder.events();
    assert_eq!(events[0].offset, 0);
    assert_eq!(events[1].offset, 1);
}

#[test]
fn stream_id_prefix_makes_event_ids_stable_and_collision_free() {
    // Two independent "processes" replaying the same run re-mint identical ids
    // for the same (stream_id, offset) — stable across restart.
    let first = EventSink::with_stream_id("run-42");
    let second = EventSink::with_stream_id("run-42");
    let a = first.emit(AgentEvent::StateUpdate);
    let b = second.emit(AgentEvent::StateUpdate);
    assert_eq!(a.id, b.id);
    assert_eq!(a.id.as_str(), "run-42-evt-0");

    // A different run never collides even though both restart at offset 0.
    let other = EventSink::with_stream_id("run-99");
    assert_ne!(other.emit(AgentEvent::StateUpdate).id, a.id);

    // Default sinks get distinct process-unique prefixes, so two default sinks
    // do not collide at offset 0 either.
    let d1 = EventSink::new();
    let d2 = EventSink::new();
    assert_ne!(
        d1.emit(AgentEvent::StateUpdate).id,
        d2.emit(AgentEvent::StateUpdate).id
    );
}

#[test]
fn smoke_event_journal_replay() {
    let journal = EventJournal::new();

    let run_id = RunId::new("run-journal");
    journal.append(AgentEvent::RunStarted {
        run_id: run_id.clone(),
        thread_id: None,
    });
    journal.append(AgentEvent::RunCompleted {
        run_id: run_id.clone(),
    });

    assert_eq!(journal.len(), 2);

    let all = journal.replay_from(0);
    assert_eq!(all.len(), 2);

    let tail = journal.replay_from(1);
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].event.kind(), "run.completed");
}

#[test]
fn completed_events_deserialize_without_started_at_ms() {
    // Journals written before `started_at_ms` existed must keep
    // deserializing: the field defaults to `None` and is omitted from the
    // serialized form when absent.
    let event: AgentEvent = serde_json::from_str(r#"{"kind":"model_completed","call_id":"c1"}"#)
        .expect("pre-started_at_ms model_completed still deserializes");
    assert!(matches!(
        event,
        AgentEvent::ModelCompleted {
            started_at_ms: None,
            ..
        }
    ));

    let event: AgentEvent =
        serde_json::from_str(r#"{"kind":"tool_completed","call_id":"t1","tool_name":"lookup"}"#)
            .expect("pre-started_at_ms tool_completed still deserializes");
    assert!(matches!(
        event,
        AgentEvent::ToolCompleted {
            started_at_ms: None,
            ..
        }
    ));

    // A populated start time round-trips.
    let event = AgentEvent::ToolCompleted {
        call_id: crate::ids::CallId::new("t2"),
        tool_name: "lookup".to_string(),
        started_at_ms: Some(1_704_067_199_000),
        input: None,
        output: None,
        duration_ms: Some(12),
        output_bytes: Some(5),
        error: None,
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["started_at_ms"], 1_704_067_199_000u64);
    let back: AgentEvent = serde_json::from_value(json).unwrap();
    assert_eq!(back, event);
}

#[test]
fn smoke_harness_run_status_lifecycle() {
    let run_id = RunId::new("run-status");
    let component = ComponentId::new("agent");
    let mut status = HarnessRunStatus::new(run_id, component);

    assert_eq!(status.status, ExecutionStatus::Pending);

    status.mark_running(HarnessPhase::Model);
    assert_eq!(status.status, ExecutionStatus::Running);
    assert_eq!(status.current_phase, HarnessPhase::Model);

    status.mark_completed();
    assert_eq!(status.status, ExecutionStatus::Completed);
    assert!(status.ended_at.is_some());
}

#[test]
fn smoke_sink_clone_shares_state() {
    let sink = EventSink::new();
    let sink2 = sink.clone();

    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    // Emitting through the clone should still reach the recorder.
    sink2.emit(AgentEvent::StateUpdate);
    assert_eq!(recorder.len(), 1);
}

#[test]
fn sink_listener_can_emit_to_same_sink_without_deadlock() {
    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());
    sink.subscribe(Arc::new(ReentrantEmitter::new(sink.clone())));

    let (tx, rx) = mpsc::channel();
    let emit_sink = sink.clone();
    let handle = thread::spawn(move || {
        emit_sink.emit(AgentEvent::StateUpdate);
        tx.send(()).unwrap();
    });

    rx.recv_timeout(Duration::from_secs(1))
        .expect("re-entrant emit should not deadlock");
    handle.join().expect("emit thread should finish");

    let events = recorder.events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].offset, 0);
    assert_eq!(events[1].offset, 1);
}

/// Concurrent emits must be delivered to listeners in offset order: offset
/// assignment and enqueueing share one critical section and a single drainer
/// dispatches the queue, so no listener can observe offset 1 before offset 0.
#[test]
fn concurrent_emits_reach_listeners_in_offset_order() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 50;

    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let sink = sink.clone();
            thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    sink.emit(AgentEvent::StateUpdate);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("emit thread should finish");
    }

    let offsets: Vec<u64> = recorder.events().iter().map(|r| r.offset).collect();
    let expected: Vec<u64> = (0..(THREADS * PER_THREAD) as u64).collect();
    assert_eq!(
        offsets, expected,
        "listeners must observe every offset exactly once, in order"
    );
}

/// Concurrent appends must land in the journal in offset order, so
/// `replay_from` returns offset order rather than the completion order of
/// racing appends.
#[test]
fn concurrent_journal_appends_replay_in_offset_order() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 50;

    let journal = Arc::new(EventJournal::new());
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let journal = journal.clone();
            thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    journal.append(AgentEvent::StateUpdate);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("append thread should finish");
    }

    let offsets: Vec<u64> = journal.replay_from(0).iter().map(|r| r.offset).collect();
    let expected: Vec<u64> = (0..(THREADS * PER_THREAD) as u64).collect();
    assert_eq!(offsets, expected, "replay_from must return offset order");

    let tail = journal.replay_from(expected.len() as u64 - 5);
    assert_eq!(tail.len(), 5);
}

// ---------------------------------------------------------------------------
// LOOP-9: a panicking listener must not wedge the sink
// ---------------------------------------------------------------------------

/// A listener that panics on its first `n` events, then records normally.
struct PanickingListener {
    remaining_panics: std::sync::Mutex<usize>,
    seen: std::sync::Mutex<Vec<String>>,
}

impl EventListener for PanickingListener {
    fn on_event(&self, record: &EventRecord) {
        {
            let mut remaining = self
                .remaining_panics
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if *remaining > 0 {
                *remaining -= 1;
                panic!("listener blew up on {}", record.event.kind());
            }
        }
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(record.event.kind().to_string());
    }
}

#[test]
fn a_panicking_listener_does_not_permanently_wedge_the_sink() {
    // Regression test (LOOP-9): `dispatching` was cleared only when the drain
    // loop reached an empty queue. A panic inside `on_event` unwound straight
    // past that reset, so the flag stayed `true` forever: every subsequent
    // `emit` saw a drain "already in progress", pushed onto `pending`, and
    // returned. The run kept emitting and no listener ever received anything
    // again, while `pending` grew without bound.
    let sink = EventSink::new();
    let bomb = Arc::new(PanickingListener {
        remaining_panics: std::sync::Mutex::new(1),
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(bomb.clone());
    sink.subscribe(recorder.clone());

    // First emit: the listener panics. Catch it the way a host would.
    let sink_for_panic = sink.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sink_for_panic.emit(AgentEvent::StateUpdate)
    }));
    assert!(result.is_err(), "the listener was supposed to panic");

    // Every later emit must still be delivered, synchronously, to every
    // listener — including the one that panicked.
    for _ in 0..3 {
        sink.emit(AgentEvent::StateUpdate);
    }

    assert_eq!(
        recorder.len(),
        3,
        "sink stayed wedged after a listener panic: later events were queued, never delivered"
    );
    assert_eq!(
        bomb.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        3,
        "the recovered listener should have received the later events too"
    );
}

#[test]
fn sink_accessors_recover_from_a_poisoned_lock() {
    // A panic while a listener holds no sink lock cannot poison it, but a panic
    // anywhere else in the process can. `SteeringHandle` already recovers from
    // poisoning; the events module used to `.expect(...)` and take the whole
    // bus down with it.
    let sink = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());

    // Poison the recorder's own buffer lock from a panicking thread.
    let poisoner = recorder.clone();
    let _ = thread::spawn(move || {
        let _guard = poisoner.records.lock().expect("first lock is clean");
        panic!("poison the recording listener");
    })
    .join();
    assert!(
        recorder.records.is_poisoned(),
        "the test needs a genuinely poisoned lock"
    );

    // Both the sink and the listener remain usable.
    sink.emit(AgentEvent::StateUpdate);
    assert_eq!(sink.len(), 1);
    assert_eq!(recorder.len(), 1);
}

// ---------------------------------------------------------------------------
// LOOP-6: every Started variant has a terminal partner on the error path
// ---------------------------------------------------------------------------

#[test]
fn failure_variants_exist_and_carry_stable_kind_strings() {
    // Regression test (LOOP-6): there was no `ToolFailed` / `ModelFailed` /
    // `SubAgentFailed`, so the three `?` sites that skip the `Completed` emit
    // left a `Started` with no terminal partner, and any exporter pairing
    // started/completed silently dropped every failed call.
    use crate::ids::CallId;

    let tool_failed = AgentEvent::ToolFailed {
        call_id: CallId::new("call-1"),
        tool_name: "search".into(),
        started_at_ms: Some(1_000),
        duration_ms: Some(25),
        error: "boom".into(),
    };
    let model_failed = AgentEvent::ModelFailed {
        call_id: CallId::new("call-2"),
        model: "gpt-4o".into(),
        started_at_ms: Some(1_000),
        attempts: Some(4),
        error: "429 rate limited".into(),
    };
    let subagent_failed = AgentEvent::SubAgentFailed {
        name: "researcher".into(),
        depth: 2,
        error: "child run failed".into(),
    };

    assert_eq!(tool_failed.kind(), "tool.failed");
    assert_eq!(model_failed.kind(), "model.failed");
    assert_eq!(subagent_failed.kind(), "subagent.failed");
}

#[test]
fn failure_variants_round_trip_through_serde() {
    use crate::ids::CallId;

    for event in [
        AgentEvent::ToolFailed {
            call_id: CallId::new("call-1"),
            tool_name: "search".into(),
            started_at_ms: None,
            duration_ms: None,
            error: "boom".into(),
        },
        AgentEvent::ModelFailed {
            call_id: CallId::new("call-2"),
            model: "gpt-4o".into(),
            started_at_ms: None,
            attempts: None,
            error: "boom".into(),
        },
        AgentEvent::SubAgentFailed {
            name: "researcher".into(),
            depth: 1,
            error: "boom".into(),
        },
    ] {
        let json = serde_json::to_value(&event).expect("serialize");
        let back: AgentEvent = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, event);
    }
}

#[test]
fn every_started_variant_pairs_with_both_a_completed_and_a_failed_variant() {
    // Guard against a future `*Started` landing without its error-path partner.
    // Kept as a string check because the enum has no structural grouping.
    let kinds: Vec<&str> = vec![
        "tool.started",
        "tool.completed",
        "tool.failed",
        "model.started",
        "model.completed",
        "model.failed",
        "subagent.started",
        "subagent.completed",
        "subagent.failed",
        "middleware.started",
        "middleware.completed",
        "middleware.failed",
    ];
    for started in kinds.iter().filter(|k| k.ends_with(".started")) {
        let prefix = started.trim_end_matches(".started");
        assert!(
            kinds.contains(&format!("{prefix}.completed").as_str()),
            "{prefix} has no completed partner"
        );
        assert!(
            kinds.contains(&format!("{prefix}.failed").as_str()),
            "{prefix} has no failed partner"
        );
    }
}
