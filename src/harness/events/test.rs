//! Tests added in a later pass.
//!
//! This file contains a minimal smoke test to verify that the events module
//! compiles and that the core fan-out and recording primitives work together.
//! Comprehensive property tests and replay tests are tracked for a later pass.

use std::sync::Arc;

use crate::harness::events::{
    AgentEvent, EventJournal, EventSink, HarnessRunStatus, RecordingListener,
};
use crate::harness::ids::{ComponentId, ExecutionStatus, HarnessPhase, RunId};

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
