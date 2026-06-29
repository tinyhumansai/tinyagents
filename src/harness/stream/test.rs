//! Tests added in a later pass.
//!
//! This file contains minimal smoke tests to verify that the stream module
//! compiles and that mode filtering, push/drain, and the `stream()` helper
//! all behave correctly. Comprehensive integration tests are tracked for a
//! later pass.

use serde_json::json;

use crate::harness::message::MessageDelta;
use crate::harness::stream::{stream, StreamChunk, StreamMode, StreamSink};

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
    assert_eq!(
        StreamChunk::Values(json!(null)).mode(),
        StreamMode::Values
    );
    assert_eq!(
        StreamChunk::Updates(json!(null)).mode(),
        StreamMode::Updates
    );
    assert_eq!(
        StreamChunk::Message(MessageDelta::default()).mode(),
        StreamMode::Messages
    );
    assert_eq!(
        StreamChunk::Debug("x".into()).mode(),
        StreamMode::Debug
    );
    assert_eq!(
        StreamChunk::Interrupt(json!(null)).mode(),
        StreamMode::Interrupts
    );
    assert_eq!(
        StreamChunk::Custom(json!(null)).mode(),
        StreamMode::Custom
    );
}
