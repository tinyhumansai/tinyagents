use super::*;
use crate::harness::ids::NodeId;

#[test]
fn collecting_sink_records_events() {
    let sink = CollectingSink::new();
    assert!(sink.is_empty());
    sink.emit(GraphEvent::StepStarted {
        step: 1,
        active: vec![NodeId::from("a")],
    });
    sink.emit(GraphEvent::StepCompleted { step: 1 });
    assert_eq!(sink.len(), 2);
    assert!(matches!(
        sink.events()[0],
        GraphEvent::StepStarted { step: 1, .. }
    ));
}

#[test]
fn noop_sink_drops_events() {
    let sink = NoopSink;
    sink.emit(GraphEvent::StepCompleted { step: 1 });
}
