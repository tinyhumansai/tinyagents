//! Tests for the harness streaming projections.
//!
//! Cover [`StreamSink`] mode filtering, push/drain/peek, enable/disable of
//! active modes, the all-modes and empty-modes sinks, [`StreamChunk::mode`]
//! variant mapping, and the standalone [`stream`] filter helper.

use serde_json::json;

use crate::harness::message::MessageDelta;
use crate::harness::stream::{StreamChunk, StreamMode, StreamSink, stream};

#[test]
fn smoke_sink_filters_by_mode() {
    let sink = StreamSink::new([StreamMode::Messages]);

    sink.push(StreamChunk::Message(MessageDelta {
        text: "hello".into(),
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
