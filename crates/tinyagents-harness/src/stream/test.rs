//! Tests for the harness streaming projections.
//!
//! Cover [`StreamSink`] mode filtering, push/drain/peek, enable/disable of
//! active modes, the all-modes and empty-modes sinks, [`StreamChunk::mode`]
//! variant mapping, and the standalone [`stream`] filter helper.

use serde_json::json;

use crate::stream::{StreamChunk, StreamMode, StreamSink, stream};
use tinyinference::message::MessageDelta;

#[test]
fn smoke_sink_filters_by_mode() {
    let sink = StreamSink::new([StreamMode::Messages]);

    sink.push(StreamChunk::Message(MessageDelta {
        text: "hello".into(),
        reasoning: String::new(),
        tool_call: None,
    }));
    // Debug chunk should be discarded (mode not active).
    sink.push(StreamChunk::Debug("internal note".into()));

    assert_eq!(sink.len(), 1);

    let chunks = sink.drain();
    assert_eq!(chunks.len(), 1);
    assert!(matches!(chunks[0], StreamChunk::Message(_)));

    // Buffer is cleared after drain.
    assert!(sink.is_empty());
}

#[test]
fn smoke_sink_all_accepts_every_mode() {
    let sink = StreamSink::all();

    sink.push(StreamChunk::Values(json!({"state": 1})));
    sink.push(StreamChunk::Updates(json!({"delta": 2})));
    sink.push(StreamChunk::Message(MessageDelta::default()));
    sink.push(StreamChunk::Debug("trace".into()));
    sink.push(StreamChunk::Interrupt(json!({"kind": "approval"})));
    sink.push(StreamChunk::Custom(json!({"ext": true})));

    assert_eq!(sink.drain().len(), 6);
}

#[test]
fn smoke_stream_helper_filters() {
    let chunks = vec![
        StreamChunk::Message(MessageDelta {
            text: "tok".into(),
            reasoning: String::new(),
            tool_call: None,
        }),
        StreamChunk::Debug("trace".into()),
        StreamChunk::Values(json!(null)),
    ];

    let msgs = stream(&chunks, &[StreamMode::Messages]);
    assert_eq!(msgs.len(), 1);

    let two = stream(&chunks, &[StreamMode::Messages, StreamMode::Debug]);
    assert_eq!(two.len(), 2);

    let none = stream(&chunks, &[]);
    assert!(none.is_empty());
}

#[test]
fn smoke_chunk_mode_matches_variant() {
    assert_eq!(StreamChunk::Values(json!(null)).mode(), StreamMode::Values);
    assert_eq!(
        StreamChunk::Updates(json!(null)).mode(),
        StreamMode::Updates
    );
    assert_eq!(
        StreamChunk::Message(MessageDelta::default()).mode(),
        StreamMode::Messages
    );
    assert_eq!(StreamChunk::Debug("x".into()).mode(), StreamMode::Debug);
    assert_eq!(
        StreamChunk::Interrupt(json!(null)).mode(),
        StreamMode::Interrupts
    );
    assert_eq!(StreamChunk::Custom(json!(null)).mode(), StreamMode::Custom);
}

#[test]
fn sink_enable_disable_and_is_active() {
    let mut sink = StreamSink::new([StreamMode::Messages]);
    assert!(sink.is_active(StreamMode::Messages));
    assert!(!sink.is_active(StreamMode::Debug));

    // Enabling Debug starts accepting debug chunks.
    sink.enable(StreamMode::Debug);
    assert!(sink.is_active(StreamMode::Debug));
    sink.push(StreamChunk::Debug("now kept".into()));
    assert_eq!(sink.len(), 1);

    // Disabling Messages discards subsequent message chunks but keeps buffered.
    sink.disable(StreamMode::Messages);
    assert!(!sink.is_active(StreamMode::Messages));
    sink.push(StreamChunk::Message(MessageDelta::default()));
    assert_eq!(sink.len(), 1);
}

#[test]
fn sink_active_modes_returns_set() {
    let sink = StreamSink::new([StreamMode::Values, StreamMode::Custom]);
    let modes = sink.active_modes();
    assert_eq!(modes.len(), 2);
    assert!(modes.contains(&StreamMode::Values));
    assert!(modes.contains(&StreamMode::Custom));
}

#[test]
fn sink_empty_modes_discards_everything() {
    let sink = StreamSink::new([]);
    sink.push(StreamChunk::Debug("x".into()));
    sink.push(StreamChunk::Values(json!(1)));
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
}

#[test]
fn sink_peek_does_not_consume() {
    let sink = StreamSink::all();
    sink.push(StreamChunk::Debug("a".into()));
    sink.push(StreamChunk::Debug("b".into()));

    let peeked = sink.peek();
    assert_eq!(peeked.len(), 2);
    // Peek leaves the buffer intact.
    assert_eq!(sink.len(), 2);
    // Drain still returns the same chunks.
    assert_eq!(sink.drain().len(), 2);
    assert!(sink.is_empty());
}

/// Round-trips a [`StreamChunk`] through JSON and asserts it deserializes back
/// to an equal value, proving every variant survives serde (including the
/// scalar `Value` payloads an internally tagged enum could not encode).
fn roundtrip_chunk(chunk: StreamChunk) {
    let value = serde_json::to_value(&chunk).expect("serialize StreamChunk");
    let back: StreamChunk = serde_json::from_value(value).expect("deserialize StreamChunk");
    assert_eq!(chunk, back);
}

#[test]
fn stream_chunk_roundtrips_every_variant() {
    // Scalar / null / array Values are exactly what internal tagging corrupted.
    roundtrip_chunk(StreamChunk::Values(json!(null)));
    roundtrip_chunk(StreamChunk::Values(json!(42)));
    roundtrip_chunk(StreamChunk::Values(json!({ "state": 1 })));
    roundtrip_chunk(StreamChunk::Updates(json!([1, 2, 3])));
    roundtrip_chunk(StreamChunk::Message(MessageDelta::text("hi")));
    roundtrip_chunk(StreamChunk::Debug("trace".into()));
    roundtrip_chunk(StreamChunk::Interrupt(json!({ "kind": "approval" })));
    roundtrip_chunk(StreamChunk::Custom(json!(true)));
}

#[test]
fn stream_chunk_null_values_does_not_corrupt_to_empty_object() {
    let value = serde_json::to_value(StreamChunk::Values(json!(null))).unwrap();
    assert_eq!(value["type"], json!("values"));
    assert_eq!(value["content"], json!(null));
}

// ---------------------------------------------------------------------------
// C6: AgentEvent -> StreamChunk projection
// ---------------------------------------------------------------------------

mod project {
    use crate::events::AgentEvent;
    use crate::ids::{CallId, RunId};
    use crate::stream::{
        StreamChunk, StreamMode, StreamSink, project_event, project_event_for_modes, projected_mode,
    };
    use tinyinference::message::MessageDelta;

    fn delta_event() -> AgentEvent {
        AgentEvent::ModelDelta {
            run_id: RunId::new("r1"),
            call_id: CallId::new("c1"),
            delta: MessageDelta::text("hello"),
        }
    }

    #[test]
    fn model_deltas_project_onto_messages_mode() {
        // Before this projection existed, `StreamMode` / `StreamChunk` /
        // `StreamSink` were referenced NOWHERE outside their own module, and
        // every caller re-implemented delta reassembly against raw events.
        let chunk = project_event(&delta_event()).expect("a delta must project");
        assert_eq!(chunk.mode(), StreamMode::Messages);
        assert_eq!(chunk, StreamChunk::Message(MessageDelta::text("hello")));
    }

    #[test]
    fn an_interrupting_control_is_the_producer_of_stream_chunk_interrupt() {
        // `StreamChunk::Interrupt` was defined but never constructed anywhere
        // in the crate.
        let event = AgentEvent::ControlApplied {
            control: "interrupt".into(),
            detail: "approval_node: needs sign-off".into(),
        };
        let chunk = project_event(&event).expect("must project");
        assert_eq!(chunk.mode(), StreamMode::Interrupts);
        match chunk {
            StreamChunk::Interrupt(value) => {
                assert_eq!(value["control"], "interrupt");
                assert_eq!(value["detail"], "approval_node: needs sign-off");
            }
            other => panic!("expected an Interrupt chunk, got {other:?}"),
        }
    }

    #[test]
    fn a_non_interrupting_control_stays_in_the_debug_channel() {
        let event = AgentEvent::ControlApplied {
            control: "stop_with_final".into(),
            detail: "done".into(),
        };
        assert_eq!(projected_mode(&event), Some(StreamMode::Debug));
    }

    #[test]
    fn state_updates_project_onto_updates_mode() {
        let chunk = project_event(&AgentEvent::StateUpdate).expect("must project");
        assert_eq!(chunk.mode(), StreamMode::Updates);
    }

    #[test]
    fn every_other_event_falls_through_to_debug_and_keeps_its_payload() {
        let event = AgentEvent::ToolStarted {
            call_id: CallId::new("c9"),
            tool_name: "search".into(),
        };
        let chunk = project_event(&event).expect("must project");
        assert_eq!(chunk.mode(), StreamMode::Debug);
        match chunk {
            // The typed payload survives, not just the kind string.
            StreamChunk::Debug(text) => {
                assert!(text.contains("tool_started"), "got {text}");
                assert!(text.contains("search"), "got {text}");
            }
            other => panic!("expected a Debug chunk, got {other:?}"),
        }
    }

    #[test]
    fn stream_closed_is_a_terminator_not_content() {
        assert!(project_event(&AgentEvent::StreamClosed).is_none());
        assert!(projected_mode(&AgentEvent::StreamClosed).is_none());
        assert!(project_event_for_modes(&AgentEvent::StreamClosed, &[StreamMode::Debug]).is_none());
    }

    #[test]
    fn one_event_never_projects_into_two_modes() {
        // A consumer subscribed to several modes must not see the same event
        // twice in two shapes.
        for event in [
            delta_event(),
            AgentEvent::StateUpdate,
            AgentEvent::ControlApplied {
                control: "interrupt".into(),
                detail: "d".into(),
            },
            AgentEvent::MemoryLoaded,
        ] {
            let all = [
                StreamMode::Values,
                StreamMode::Updates,
                StreamMode::Messages,
                StreamMode::Debug,
                StreamMode::Interrupts,
                StreamMode::Custom,
            ];
            let hits = all
                .iter()
                .filter(|mode| project_event_for_modes(&event, &[**mode]).is_some())
                .count();
            assert_eq!(hits, 1, "{} projected into {hits} modes", event.kind());
        }
    }

    #[test]
    fn mode_filtering_happens_producer_side() {
        let event = delta_event();
        assert!(project_event_for_modes(&event, &[StreamMode::Messages]).is_some());
        assert!(project_event_for_modes(&event, &[StreamMode::Debug]).is_none());
        assert!(project_event_for_modes(&event, &[]).is_none());
        // Multiplexed mode sets work the way LangGraph's do.
        assert!(
            project_event_for_modes(&event, &[StreamMode::Debug, StreamMode::Messages]).is_some()
        );
    }

    #[test]
    fn push_event_bridges_the_event_bus_into_a_sink() {
        let sink = StreamSink::new([StreamMode::Messages]);
        assert!(sink.push_event(&delta_event()));
        assert!(!sink.push_event(&AgentEvent::StateUpdate));
        assert!(!sink.push_event(&AgentEvent::StreamClosed));

        let chunks = sink.drain();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].mode(), StreamMode::Messages);
    }

    #[test]
    fn values_and_custom_are_never_produced_by_the_projection() {
        // Documented gap: a full state snapshot is graph state, which the event
        // stream does not carry, and Custom is the caller's own channel.
        for event in [
            delta_event(),
            AgentEvent::StateUpdate,
            AgentEvent::MemorySaved,
            AgentEvent::RunCompleted {
                run_id: RunId::new("r1"),
            },
        ] {
            let mode = projected_mode(&event);
            assert_ne!(mode, Some(StreamMode::Values));
            assert_ne!(mode, Some(StreamMode::Custom));
        }
    }
}
