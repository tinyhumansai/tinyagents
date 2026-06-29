//! Data types for the graph testkit.
//!
//! See the module [`mod`](super) docs for the building blocks these back: the
//! event recorder ([`GraphEventRecorder`]), the projection collector
//! ([`StreamCollector`]), the bundled run-under-test ([`GraphRun`]), the fluent
//! assertion builder ([`GraphAssertions`]), and the stateful retry-counting node
//! ([`RetryCountingNode`]). This file holds the plain definitions; the impls and
//! the node-helper free functions live in `mod.rs`.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::graph::compiled::{GraphExecution, StateSnapshot};
use crate::graph::stream::{CollectingSink, GraphEvent};

/// Captures the durable executor's [`GraphEvent`] stream for inspection.
///
/// Wraps a [`CollectingSink`]; hand its [`sink`](GraphEventRecorder::sink) to
/// [`CompiledGraph::with_event_sink`](crate::graph::CompiledGraph::with_event_sink)
/// (or use [`run_recorded`](super::run_recorded), which wires it for you) so a
/// run's events are recorded, then read them back as raw events, `kind` strings,
/// or a [`StreamCollector`] of higher-level projections.
#[derive(Clone, Default)]
pub struct GraphEventRecorder {
    pub(crate) sink: CollectingSink,
}

/// A higher-level projection over a recorded [`GraphEvent`] stream.
///
/// Where [`GraphEventRecorder`] hands back raw events, `StreamCollector` slices
/// them into the views a test usually wants: the executed-node order, the nodes
/// that wrote state, the `(from, to)` routes the executor selected, the emitted
/// interrupts, the persisted-checkpoint count, and any custom node writes.
#[derive(Clone, Default)]
pub struct StreamCollector {
    pub(crate) events: Vec<GraphEvent>,
}

/// A completed (or interrupted) run bundled with everything the assertions need.
///
/// [`assert_graph`](super::assert_graph) reads its truth from one `GraphRun`:
/// the [`GraphExecution`] (visited nodes, steps, status, interrupts), the
/// recorded [`GraphEvent`] stream (real route selections and checkpoint saves),
/// and the thread's checkpoint `history` (newest-first, as
/// [`get_state_history`](crate::graph::CompiledGraph::get_state_history) returns
/// it). Build one with [`run_recorded`](super::run_recorded), or assemble it by
/// hand via [`GraphRun::new`] and the `with_*` builders.
pub struct GraphRun<State> {
    /// The execution result (final state, visited nodes, steps, status).
    pub execution: GraphExecution<State>,
    /// The recorded low-level graph events, in emission order.
    pub events: Vec<GraphEvent>,
    /// The thread's checkpoint history, newest-first (empty for a non-durable
    /// run, or one executed without a thread id).
    pub history: Vec<StateSnapshot<State>>,
}

/// A fluent, panic-on-failure assertion builder over a [`GraphRun`].
///
/// Returned by [`assert_graph`](super::assert_graph). Every method asserts and
/// returns `&Self` so checks chain:
///
/// ```ignore
/// assert_graph(&run)
///     .visited(["agent", "tools", "agent"])
///     .routed("agent", "tools")
///     .checkpoint_count(3)
///     .completed();
/// ```
pub struct GraphAssertions<'a, State> {
    pub(crate) run: &'a GraphRun<State>,
}

/// A stateful node double that counts its activations and fails the first
/// `fail_times` of them.
///
/// Each [`handler`](RetryCountingNode::handler) it produces shares the same
/// activation counter, so a test can run the graph and then read
/// [`attempts`](RetryCountingNode::attempts) to assert how many times the node
/// ran. The first `fail_times` activations return
/// [`TinyAgentsError::Graph`](crate::TinyAgentsError::Graph); subsequent ones
/// succeed with the configured update.
#[derive(Clone)]
pub struct RetryCountingNode {
    pub(crate) attempts: Arc<AtomicUsize>,
    pub(crate) fail_times: usize,
}
