//! Caller-consumable streaming entry point ([`AgentHarness::invoke_stream`]).
//!
//! [`AgentHarness::invoke_streaming`] drives each model call through
//! [`ChatModel::stream`][tinyinference_llm::model::ChatModel::stream] but only
//! returns the fully-accumulated [`AgentRun`] once the run is over — a caller
//! that wants to *watch* the run unfold has to attach an
//! [`EventListener`][crate::events::EventListener] or middleware.
//!
//! `invoke_stream` closes that gap: it returns an async
//! [`Stream`][futures::Stream] of [`AgentStreamItem`]s that yields every
//! [`AgentEvent`][crate::events::AgentEvent] emitted during the run —
//! model/reasoning deltas, tool-call `ToolStarted`/`ToolCompleted` lifecycle
//! events, and sub-agent `SubAgentStarted`/`SubAgentCompleted` events (which
//! reach the stream because sub-agents share the parent's
//! [`EventSink`][crate::events::EventSink]) — and finishes with a
//! terminal item carrying the completed [`AgentRun`] (or the failure).
//!
//! It is a thin projection over the same ordered
//! [`EventSink::emit`][crate::events::EventSink::emit] fan-out the
//! `EventListener` path already uses, not a parallel event system: a listener
//! forwards each [`EventRecord`] into an unbounded channel that the returned
//! stream drains. Lineage is carried by the events themselves (`ModelDelta`
//! stamps `run_id`; sub-agent events stamp `depth`). Streaming a child agent's
//! own model deltas into the parent requires routing the streaming flag through
//! the sub-agent runner and is tracked as follow-up work.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::context::{RunConfig, RunContext};
use crate::events::{EventListener, EventRecord, EventSink};
use crate::middleware::AgentRun;
use crate::runtime::{AgentHarness, InvocationRuntime};
use tinyinference_llm::message::Message;

use super::PartialRunOutcome;

/// The concrete driver behind a caller-consumable stream: either the
/// durable harness borrowed for the caller's lifetime (the ordinary SDK
/// path), or an invocation-local runtime owned outright (the hosted path,
/// where the runtime is only alive as a local variable at the call site).
///
/// Moving the `Owned` variant into the driving future (see
/// [`invoke_stream_with_runner`]) is what lets the hosted stream avoid both
/// an unsound lifetime extension and depending on field drop order: the
/// runtime's lifetime becomes exactly the future's, which the stream already
/// owns.
pub(crate) enum StreamRunner<'a, State: Send + Sync, Ctx: Send + Sync> {
    Borrowed(&'a AgentHarness<State, Ctx>),
    Owned(Arc<InvocationRuntime<State, Ctx>>),
}

impl<State: Send + Sync, Ctx: Send + Sync> std::ops::Deref for StreamRunner<'_, State, Ctx> {
    type Target = AgentHarness<State, Ctx>;

    fn deref(&self) -> &AgentHarness<State, Ctx> {
        match self {
            StreamRunner::Borrowed(harness) => harness,
            StreamRunner::Owned(runtime) => runtime.harness(),
        }
    }
}

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
    /// [`EventRecord`] (id + offset + [`AgentEvent`][crate::events::AgentEvent])
    /// so consumers keep ordering and lineage fields (`run_id`, `depth`).
    Event(EventRecord),
    /// Terminal: the run completed successfully. Boxed because [`AgentRun`] is
    /// large relative to the event variant.
    Completed(Box<AgentRun>),
    /// Terminal: the run failed, with the partial run retained for accurate
    /// host-side accounting and terminal learning.
    Failed {
        /// Stable display form of the failure.
        error: String,
        /// Work completed before the failure.
        run: Box<AgentRun>,
    },
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

/// Removes the one-shot channel listener from the run's event sink when the
/// returned stream is exhausted or dropped early.
struct ChannelListenerGuard {
    events: EventSink,
    listener: Arc<dyn EventListener>,
}

impl Drop for ChannelListenerGuard {
    fn drop(&mut self) {
        let _ = self.events.unsubscribe(&self.listener);
    }
}

/// Maps a finished run result onto its terminal [`AgentStreamItem`].
fn terminal_item(outcome: PartialRunOutcome) -> AgentStreamItem {
    match outcome.error {
        None => AgentStreamItem::Completed(Box::new(outcome.run)),
        Some(error) => AgentStreamItem::Failed {
            error: error.to_string(),
            run: Box::new(outcome.run),
        },
    }
}

/// Drive-phase for the streaming state machine.
enum Phase<'a> {
    /// The run future is still executing.
    Running {
        run_fut: Pin<Box<dyn Future<Output = PartialRunOutcome> + Send + 'a>>,
        listener_guard: ChannelListenerGuard,
    },
    /// The run has finished; drain any buffered events, then emit `terminal`.
    Draining {
        terminal: Box<AgentStreamItem>,
        listener_guard: ChannelListenerGuard,
    },
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
    /// [`ChatModel::stream`][tinyinference_llm::model::ChatModel::stream]).
    pub fn invoke_stream<'a>(
        &'a self,
        state: &'a State,
        ctx_data: Ctx,
        config: RunConfig,
        input: Vec<Message>,
    ) -> impl futures::Stream<Item = AgentStreamItem> + Send + 'a {
        let ctx = RunContext::new(config, ctx_data);
        self.invoke_stream_in_context(state, ctx, input)
    }

    /// Runs the agent loop while streaming every emitted event to the caller,
    /// using a caller-supplied [`RunContext`].
    ///
    /// This is the context-preserving counterpart of
    /// [`AgentHarness::invoke_stream`]. Use it when the caller has already
    /// attached cancellation, events, stores, workspace metadata, or steering to
    /// the run context and still wants caller-consumable stream items.
    pub fn invoke_stream_in_context<'a>(
        &'a self,
        state: &'a State,
        ctx: RunContext<Ctx>,
        input: Vec<Message>,
    ) -> impl futures::Stream<Item = AgentStreamItem> + Send + 'a {
        invoke_stream_with_runner(StreamRunner::Borrowed(self), state, ctx, input)
    }
}

/// Builds the caller-consumable event stream for either an ordinary
/// (borrowed-harness) or a hosted (owned-runtime) invocation.
///
/// `runner` is moved into the driving future itself rather than dereferenced
/// up front, so an owned [`InvocationRuntime`] carried by `runner` lives
/// exactly as long as the future that needs it — no separate field, drop
/// order, or lifetime extension required on the caller's stream wrapper.
pub(crate) fn invoke_stream_with_runner<'a, State, Ctx>(
    runner: StreamRunner<'a, State, Ctx>,
    state: &'a State,
    ctx: RunContext<Ctx>,
    input: Vec<Message>,
) -> impl futures::Stream<Item = AgentStreamItem> + Send + 'a
where
    State: Send + Sync,
    Ctx: Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    // Subscribe before driving so no event (starting with `RunStarted`) is
    // missed. The listener rides the run's `EventSink`, which sub-agents
    // clone, so their lifecycle events reach this stream too.
    let listener: Arc<dyn EventListener> = Arc::new(ChannelListener { tx });
    ctx.events.subscribe(listener.clone());
    let listener_guard = ChannelListenerGuard {
        events: ctx.events.clone(),
        listener,
    };

    // Preserve partial work for a failed streamed run. The event stream
    // remains unchanged, but terminal host capabilities need honest usage
    // and executed-tool summaries for error and cancellation paths too.
    //
    // `runner` is moved into this async block rather than dereferenced
    // beforehand: the generated state machine owns it (and, for the hosted
    // `Owned` variant, the `Arc<InvocationRuntime>` inside it) for exactly as
    // long as the future borrows from it across the `.await` below, which is
    // what async/await's normal self-referential generator lowering makes
    // sound without any unsafe code.
    let run_fut: Pin<Box<dyn Future<Output = PartialRunOutcome> + Send + 'a>> =
        Box::pin(async move {
            let runner = runner;
            runner
                .invoke_streaming_in_context_collecting_partial(state, ctx, input)
                .await
        });

    futures::stream::unfold(
        (
            Phase::Running {
                run_fut,
                listener_guard,
            },
            rx,
        ),
        |(phase, mut rx)| async move {
            match phase {
                Phase::Running {
                    mut run_fut,
                    listener_guard,
                } => {
                    tokio::select! {
                        biased;
                        // Prefer draining ready events so the consumer sees
                        // fine-grained progress rather than a late burst.
                        maybe = rx.recv() => match maybe {
                            Some(record) => {
                                Some((
                                    AgentStreamItem::Event(record),
                                    (
                                        Phase::Running {
                                            run_fut,
                                            listener_guard,
                                        },
                                        rx,
                                    ),
                                ))
                            }
                            None => {
                                // All senders dropped (the run's context —
                                // and every sub-agent clone of the sink —
                                // is gone): the run is finishing. Await it
                                // for the terminal item.
                                let terminal = terminal_item(run_fut.await);
                                drop(listener_guard);
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
                                    (
                                        Phase::Draining {
                                            terminal: Box::new(terminal),
                                            listener_guard,
                                        },
                                        rx,
                                    ),
                                )),
                                Err(_) => {
                                    drop(listener_guard);
                                    Some((terminal, (Phase::Done, rx)))
                                }
                            }
                        }
                    }
                }
                Phase::Draining {
                    terminal,
                    listener_guard,
                } => match rx.try_recv() {
                    Ok(record) => Some((
                        AgentStreamItem::Event(record),
                        (
                            Phase::Draining {
                                terminal,
                                listener_guard,
                            },
                            rx,
                        ),
                    )),
                    Err(_) => {
                        drop(listener_guard);
                        Some((*terminal, (Phase::Done, rx)))
                    }
                },
                Phase::Done => None,
            }
        },
    )
}
