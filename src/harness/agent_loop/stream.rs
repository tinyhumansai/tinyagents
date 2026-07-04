//! Caller-consumable streaming entry point ([`AgentHarness::invoke_stream`]).
//!
//! [`AgentHarness::invoke_streaming`] drives each model call through
//! [`ChatModel::stream`][crate::harness::model::ChatModel::stream] but only
//! returns the fully-accumulated [`AgentRun`] once the run is over — a caller
//! that wants to *watch* the run unfold has to attach an
//! [`EventListener`][crate::harness::events::EventListener] or middleware.
//!
//! `invoke_stream` closes that gap: it returns an async
//! [`Stream`][futures::Stream] of [`AgentStreamItem`]s that yields every
//! [`AgentEvent`][crate::harness::events::AgentEvent] emitted during the run —
//! model/reasoning deltas, tool-call `ToolStarted`/`ToolCompleted` lifecycle
//! events, and sub-agent `SubAgentStarted`/`SubAgentCompleted` events (which
//! reach the stream because sub-agents share the parent's
//! [`EventSink`][crate::harness::events::EventSink]) — and finishes with a
//! terminal item carrying the completed [`AgentRun`] (or the failure).
//!
//! It is a thin projection over the same ordered
//! [`EventSink::emit`][crate::harness::events::EventSink::emit] fan-out the
//! `EventListener` path already uses, not a parallel event system: a listener
//! forwards each [`EventRecord`] into an unbounded channel that the returned
//! stream drains. Lineage is carried by the events themselves (`ModelDelta`
//! stamps `run_id`; sub-agent events stamp `depth`). Streaming a child agent's
//! own model deltas into the parent requires routing the streaming flag through
//! the sub-agent runner and is tracked as follow-up work.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{EventListener, EventRecord};
use crate::harness::message::Message;
use crate::harness::middleware::AgentRun;
use crate::harness::runtime::AgentHarness;

use super::AgentLoopResult;

/// One item yielded by [`AgentHarness::invoke_stream`].
///
/// The stream yields zero or more [`AgentStreamItem::Event`]s in emission
/// order, followed by exactly one terminal item — either
/// [`AgentStreamItem::Completed`] carrying the final [`AgentRun`], or
/// [`AgentStreamItem::Failed`] carrying the error string. No items are produced
/// after the terminal item.
#[derive(Clone, Debug)]
pub enum AgentStreamItem {
    /// A live event emitted during the run. Carries the full
    /// [`EventRecord`] (id + offset + [`AgentEvent`][crate::harness::events::AgentEvent])
    /// so consumers keep ordering and lineage fields (`run_id`, `depth`).
    Event(EventRecord),
    /// Terminal: the run completed successfully. Boxed because [`AgentRun`] is
    /// large relative to the event variant.
    Completed(Box<AgentRun>),
    /// Terminal: the run failed; carries the error rendered as a string.
    Failed(String),
}

/// An [`EventListener`] that forwards every [`EventRecord`] into an unbounded
/// channel. `on_event` is synchronous and must not block, so the channel is
/// unbounded; in practice its depth is bounded by run progress (the loop awaits
/// the network between emits). Send failures (receiver dropped) are ignored so
/// a dropped stream never disturbs the run.
struct ChannelListener {
    tx: tokio::sync::mpsc::UnboundedSender<EventRecord>,
}

impl EventListener for ChannelListener {
    fn on_event(&self, record: &EventRecord) {
        let _ = self.tx.send(record.clone());
    }
}

/// Maps a finished run result onto its terminal [`AgentStreamItem`].
fn terminal_item(result: Result<AgentLoopResult>) -> AgentStreamItem {
    match result {
        Ok(loop_result) => AgentStreamItem::Completed(Box::new(loop_result.run)),
        Err(error) => AgentStreamItem::Failed(error.to_string()),
    }
}

/// Drive-phase for the streaming state machine.
enum Phase<'a> {
    /// The run future is still executing.
    Running(Pin<Box<dyn Future<Output = Result<AgentLoopResult>> + 'a>>),
    /// The run has finished; drain any buffered events, then emit `terminal`.
    Draining(AgentStreamItem),
    /// Terminal item already emitted; the stream is exhausted.
    Done,
}

impl<State: Send + Sync, Ctx: Send + Sync + 'static> AgentHarness<State, Ctx> {
    /// Runs the agent loop while streaming every emitted event to the caller.
    ///
    /// Returns a [`Stream`][futures::Stream] of [`AgentStreamItem`]s: live
    /// [`AgentStreamItem::Event`]s in emission order (model/reasoning deltas,
    /// tool-call lifecycle, sub-agent lifecycle, usage, …) followed by a single
    /// terminal [`AgentStreamItem::Completed`] / [`AgentStreamItem::Failed`].
    /// Driving the loop and consuming the stream are the same task: the run
    /// only makes progress while the stream is polled, so a caller that stops
    /// polling pauses the run (and dropping the stream cancels it).
    ///
    /// This is the streaming counterpart of
    /// [`AgentHarness::invoke_streaming`]; the run itself is identical (each
    /// model call is driven through
    /// [`ChatModel::stream`][crate::harness::model::ChatModel::stream]).
    pub fn invoke_stream<'a>(
        &'a self,
        state: &'a State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> impl futures::Stream<Item = AgentStreamItem> + 'a {
        let ctx = RunContext::new(config, ctx_data);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        // Subscribe before driving so no event (starting with `RunStarted`) is
        // missed. The listener rides the run's `EventSink`, which sub-agents
        // clone, so their lifecycle events reach this stream too.
        ctx.events.subscribe(Arc::new(ChannelListener { tx }));

        // `invoke_streaming_in_context_with_status` is the public wrapper over
        // the shared `drive(.., streaming = true)` path; it drives *our* `ctx`
        // (with the listener already attached) and hands back the terminal
        // `AgentLoopResult`.
        let run_fut: Pin<Box<dyn Future<Output = Result<AgentLoopResult>> + 'a>> =
            Box::pin(self.invoke_streaming_in_context_with_status(state, ctx, input));

        futures::stream::unfold(
            (Phase::Running(run_fut), rx),
            |(phase, mut rx)| async move {
                match phase {
                    Phase::Running(mut run_fut) => {
                        tokio::select! {
                            biased;
                            // Prefer draining ready events so the consumer sees
                            // fine-grained progress rather than a late burst.
                            maybe = rx.recv() => match maybe {
                                Some(record) => {
                                    Some((AgentStreamItem::Event(record), (Phase::Running(run_fut), rx)))
                                }
                                None => {
                                    // All senders dropped (the run's context —
                                    // and every sub-agent clone of the sink —
                                    // is gone): the run is finishing. Await it
                                    // for the terminal item.
                                    let terminal = terminal_item(run_fut.await);
                                    Some((terminal, (Phase::Done, rx)))
                                }
                            },
                            result = &mut run_fut => {
                                // The run finished. Events emitted during this
                                // final poll may still be buffered; drain them
                                // ahead of the terminal item.
                                let terminal = terminal_item(result);
                                match rx.try_recv() {
                                    Ok(record) => Some((
                                        AgentStreamItem::Event(record),
                                        (Phase::Draining(terminal), rx),
                                    )),
                                    Err(_) => Some((terminal, (Phase::Done, rx))),
                                }
                            }
                        }
                    }
                    Phase::Draining(terminal) => match rx.try_recv() {
                        Ok(record) => Some((
                            AgentStreamItem::Event(record),
                            (Phase::Draining(terminal), rx),
                        )),
                        Err(_) => Some((terminal, (Phase::Done, rx))),
                    },
                    Phase::Done => None,
                }
            },
        )
    }
}
