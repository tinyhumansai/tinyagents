//! Legacy (milestone-1) sequential state-graph types.
//!
//! These types are the original `src/graph.rs` public surface. They are kept
//! verbatim so existing downstream code, the `basic_graph` example, and the
//! serialization tests continue to compile and behave identically while the
//! durable execution model grows alongside them.

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use crate::Result;

/// Boxed future returned by a legacy node handler.
pub type BoxNodeFuture<State> = Pin<Box<dyn Future<Output = Result<NodeOutput<State>>> + Send>>;

/// Legacy node handler function type.
pub type NodeFn<State> = dyn Fn(State) -> BoxNodeFuture<State> + Send + Sync;

/// A named async unit of work in a legacy [`StateGraph`].
#[derive(Clone)]
pub struct Node<State> {
    pub(crate) name: String,
    pub(crate) handler: Arc<NodeFn<State>>,
}

/// Whole-state output returned by a legacy node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeOutput<State> {
    /// Continue to the node's direct successor with the new state.
    Continue(State),
    /// Take a named conditional route with the new state.
    Route {
        /// Updated state.
        state: State,
        /// Selected route label.
        route: String,
    },
    /// End the run with the final state.
    End(State),
}

/// Outgoing edge configuration for a legacy node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Edge {
    /// A single direct successor.
    Direct(String),
    /// A label-to-node routing table.
    Conditional(HashMap<String, String>),
    /// A terminal edge.
    End,
}

/// The result of running a legacy [`StateGraph`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphRun<State> {
    /// Final state.
    pub state: State,
    /// Ordered list of visited node names.
    pub visited: Vec<String>,
}

/// A sequential, whole-state graph executor (milestone-1 scaffold).
pub struct StateGraph<State> {
    pub(crate) nodes: HashMap<String, Node<State>>,
    pub(crate) edges: HashMap<String, Edge>,
    pub(crate) start: Option<String>,
    pub(crate) recursion_limit: usize,
}
