//! Tests added in a later pass.
//!
//! This file contains a minimal smoke test to verify that the events module
//! compiles and that the core fan-out and recording primitives work together.
//! Comprehensive property tests and replay tests are tracked for a later pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use crate::harness::events::{
    AgentEvent, EventJournal, EventListener, EventRecord, EventSink, HarnessRunStatus,
    RecordingListener,
};
use crate::harness::ids::{ComponentId, ExecutionStatus, HarnessPhase, RunId};

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
