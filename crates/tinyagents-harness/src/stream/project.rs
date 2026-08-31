//! Projection from the raw [`AgentEvent`] stream onto typed [`StreamChunk`]s.
//!
//! This is the missing half of the streaming surface. [`StreamMode`],
//! [`StreamChunk`] and [`StreamSink`] were fully defined and tested but used
//! nowhere else in the crate — callers got raw `AgentEvent`s and each
//! re-implemented delta reassembly and filtering. This module supplies the one
//! function that turns an event into the chunk a consumer actually asked for.
//!
//! # Shape (ported from LangGraph)
//!
//! LangGraph multiplexes several *stream modes* over one producer: the producer
//! filters by the requested mode set (`_loop.py`'s `stream_modes` check) and the
//! consumer shapes the result (`main.py`'s `stream()` fan-out). The same split
//! applies here:
//!
//! - **Producer-side filtering** — [`project_event_for_modes`] returns `None`
//!   for an event whose chunk is not in the requested mode set, so nothing is
//!   allocated or cloned for a mode nobody is listening to.
//! - **Consumer-side shaping** — [`project_event`] gives the full projection
//!   when the caller wants to route chunks itself.
//!
//! # One event, at most one chunk
//!
//! Each [`AgentEvent`] projects to **at most one** [`StreamChunk`], so a
//! consumer subscribed to several modes never receives the same event twice in
//! two shapes. The routing table:
//!
//! | Event | Chunk | Mode |
//! |---|---|---|
//! | [`AgentEvent::ModelDelta`] | [`StreamChunk::Message`] | [`StreamMode::Messages`] |
//! | [`AgentEvent::StateUpdate`] | [`StreamChunk::Updates`] | [`StreamMode::Updates`] |
//! | [`AgentEvent::ControlApplied`] with an interrupting control | [`StreamChunk::Interrupt`] | [`StreamMode::Interrupts`] |
//! | [`AgentEvent::StreamClosed`] | — (not projected) | — |
//! | everything else | [`StreamChunk::Debug`] | [`StreamMode::Debug`] |
//!
//! [`StreamMode::Values`] is deliberately **not** produced here: a full state
//! snapshot is graph state, which the event stream does not carry. The graph
//! runtime pushes [`StreamChunk::Values`] itself.
//!
//! [`StreamMode::Custom`] is likewise never produced — by definition it is the
//! caller's own extension channel.

use crate::events::AgentEvent;

use super::{StreamChunk, StreamMode};

/// Control kinds from
/// [`MiddlewareControl::kind`][crate::context::MiddlewareControl::kind]
/// that mean "the run paused and is waiting for something external", and so
/// belong on [`StreamMode::Interrupts`] rather than in the debug firehose.
const INTERRUPTING_CONTROLS: [&str; 1] = ["interrupt"];

/// Projects one [`AgentEvent`] onto the [`StreamChunk`] a consumer sees.
///
/// Returns `None` for an event with no meaningful chunk representation — today
/// only [`AgentEvent::StreamClosed`], which is a stream terminator rather than
/// content (consumers detect end-of-stream by the stream ending).
///
/// # Example
///
/// ```
/// use tinyagents_harness::events::AgentEvent;
/// use tinyagents_harness::ids::{CallId, RunId};
/// use tinyinference::message::MessageDelta;
/// use tinyagents_harness::stream::{project_event, StreamChunk};
///
/// let event = AgentEvent::ModelDelta {
///     run_id: RunId::new("r1"),
///     call_id: CallId::new("c1"),
///     delta: MessageDelta::text("hi"),
/// };
/// assert!(matches!(project_event(&event), Some(StreamChunk::Message(_))));
/// ```
pub fn project_event(event: &AgentEvent) -> Option<StreamChunk> {
    match event {
        // The whole point of `messages` mode: raw token/tool-call fragments,
        // already reassembled by the provider adapter.
        AgentEvent::ModelDelta { delta, .. } => Some(StreamChunk::Message(delta.clone())),

        // `StateUpdate` is payload-free by design (see its variant docs), so the
        // chunk carries the fact of the update and its provenance rather than a
        // diff the event never had.
        AgentEvent::StateUpdate => Some(StreamChunk::Updates(serde_json::json!({
            "kind": event.kind(),
        }))),

        // A human-in-the-loop pause. This is the only producer of
        // `StreamChunk::Interrupt` — the variant existed but nothing ever
        // constructed it.
        AgentEvent::ControlApplied { control, detail }
            if INTERRUPTING_CONTROLS.contains(&control.as_str()) =>
        {
            Some(StreamChunk::Interrupt(serde_json::json!({
                "control": control,
                "detail": detail,
            })))
        }

        // A terminator, not content.
        AgentEvent::StreamClosed => None,

        // Everything else is diagnostic. Rendered as the event's own JSON so a
        // debug consumer keeps the full typed payload rather than a lossy
        // summary; falls back to the stable kind string if serialization ever
        // fails (it cannot for the current variants, but a `Debug` chunk must
        // never be the thing that panics a run).
        other => Some(StreamChunk::Debug(
            serde_json::to_string(other).unwrap_or_else(|_| other.kind().to_string()),
        )),
    }
}

/// Producer-side filtered projection: like [`project_event`], but returns
/// `None` when the resulting chunk's mode is not in `modes`.
///
/// This is the function a streaming run loop should call per event — it skips
/// the clone/serialization for any mode nobody subscribed to, which matters
/// because the [`StreamMode::Debug`] catch-all serializes every event.
///
/// # Example
///
/// ```
/// use tinyagents_harness::events::AgentEvent;
/// use tinyagents_harness::stream::{project_event_for_modes, StreamMode};
///
/// let event = AgentEvent::StateUpdate;
/// assert!(project_event_for_modes(&event, &[StreamMode::Updates]).is_some());
/// assert!(project_event_for_modes(&event, &[StreamMode::Messages]).is_none());
/// ```
pub fn project_event_for_modes(event: &AgentEvent, modes: &[StreamMode]) -> Option<StreamChunk> {
    let mode = projected_mode(event)?;
    if !modes.contains(&mode) {
        return None;
    }
    project_event(event)
}

/// Returns the [`StreamMode`] an event projects onto, without building the
/// chunk.
///
/// Cheap enough to call in a hot loop, and it is what makes
/// [`project_event_for_modes`] able to skip work rather than build-then-discard.
/// `None` mirrors [`project_event`] returning `None`.
pub fn projected_mode(event: &AgentEvent) -> Option<StreamMode> {
    match event {
        AgentEvent::ModelDelta { .. } => Some(StreamMode::Messages),
        AgentEvent::StateUpdate => Some(StreamMode::Updates),
        AgentEvent::ControlApplied { control, .. }
            if INTERRUPTING_CONTROLS.contains(&control.as_str()) =>
        {
            Some(StreamMode::Interrupts)
        }
        AgentEvent::StreamClosed => None,
        _ => Some(StreamMode::Debug),
    }
}
