//! Durable graph builder and compile contract.
//!
//! See [`types`] for the builder data types. `compile` validates the topology
//! and freezes it into an immutable [`crate::graph::CompiledGraph`].

mod types;

pub(crate) use types::{Branch, BuilderNode};
pub use types::{END, GraphBuilder, NodeContext, NodeFuture, NodeHandler, RouterFn, START};

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use crate::graph::command::NodeResult;
use crate::graph::compiled::CompiledGraph;
use crate::graph::reducer::{OverwriteStateReducer, StateReducer};
use crate::harness::ids::{GraphId, NodeId};
use crate::{Result, RustAgentsError};

impl<State, Update> Default for GraphBuilder<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<State, Update> GraphBuilder<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Creates an empty builder with a generated graph id and a default
    /// recursion limit of 50. A reducer must be set before [`Self::compile`].
    pub fn new() -> Self {
        Self {
            graph_id: GraphId::new(format!("graph-{}", crate::graph::compiled::next_seq())),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            branches: HashMap::new(),
            command_nodes: HashSet::new(),
            reducer: None,
            recursion_limit: 50,
        }
    }

    /// Overrides the graph id.
    pub fn with_graph_id(mut self, id: impl Into<GraphId>) -> Self {
        self.graph_id = id.into();
        self
    }

    /// Overrides the recursion limit (max number of supersteps).
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Sets the state reducer used to merge partial updates at step boundaries.
    pub fn set_reducer<R>(mut self, reducer: R) -> Self
    where
        R: StateReducer<State, Update> + 'static,
    {
        self.reducer = Some(Arc::new(reducer));
        self
    }

    /// Adds an async node returning a [`NodeResult`].
    pub fn add_node<F, Fut>(mut self, id: impl Into<NodeId>, handler: F) -> Self
    where
        F: Fn(State, NodeContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeResult<Update>>> + Send + 'static,
    {
        let id = id.into();
        self.nodes.insert(
            id.clone(),
            BuilderNode {
                id,
                handler: Arc::new(move |state, ctx| Box::pin(handler(state, ctx))),
            },
        );
        self
    }

    /// Adds a direct edge `from -> to`. Use [`START`]/[`END`] for the virtual
    /// entry/terminal nodes.
    pub fn add_edge(mut self, from: impl Into<NodeId>, to: impl Into<NodeId>) -> Self {
        self.edges.insert(from.into(), to.into());
        self
    }

    /// Sets the entry node, i.e. `add_edge(START, node)`.
    pub fn set_entry(self, node: impl Into<NodeId>) -> Self {
        self.add_edge(START, node)
    }

    /// Marks `node` as terminal, i.e. `add_edge(node, END)`.
    pub fn set_finish(self, node: impl Into<NodeId>) -> Self {
        self.add_edge(node, END)
    }

    /// Adds conditional edges: a router closure mapped against a label table.
    pub fn add_conditional_edges<F, I, K, V>(
        mut self,
        from: impl Into<NodeId>,
        router: F,
        routes: I,
    ) -> Self
    where
        F: Fn(&State) -> String + Send + Sync + 'static,
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<NodeId>,
    {
        let routes = routes
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self.branches.insert(
            from.into(),
            Branch {
                router: Arc::new(router),
                routes,
            },
        );
        self
    }

    /// Declares that `node` routes exclusively via [`crate::graph::Command`]
    /// `goto` (not static or conditional edges). Compile rejects nodes that mix
    /// command routing with static/conditional edges.
    pub fn mark_command_routing(mut self, node: impl Into<NodeId>) -> Self {
        self.command_nodes.insert(node.into());
        self
    }

    /// Validates topology and freezes the graph into a [`CompiledGraph`].
    pub fn compile(self) -> Result<CompiledGraph<State, Update>> {
        if self.reducer.is_none() {
            return Err(RustAgentsError::Validation(
                "no state reducer set; call set_reducer (or GraphBuilder::overwrite)".to_string(),
            ));
        }

        // entry must exist
        let entry = self
            .edges
            .get(&NodeId::from(START))
            .cloned()
            .ok_or(RustAgentsError::MissingStart)?;
        if entry.as_str() == END {
            return Err(RustAgentsError::Validation(
                "START cannot route directly to END".to_string(),
            ));
        }
        self.require_node(&entry)?;

        // static edges
        for (from, to) in &self.edges {
            if from.as_str() != START {
                self.require_node(from)?;
            }
            if to.as_str() != END {
                self.require_node(to)?;
            }
            if to.as_str() == START {
                return Err(RustAgentsError::Validation(
                    "START cannot be an edge target".to_string(),
                ));
            }
            if from.as_str() == END {
                return Err(RustAgentsError::Validation(
                    "END cannot be an edge source".to_string(),
                ));
            }
        }

        // conditional edges
        for (from, branch) in &self.branches {
            self.require_node(from)?;
            if self.edges.contains_key(from) {
                return Err(RustAgentsError::Validation(format!(
                    "node `{from}` has both a static edge and conditional edges"
                )));
            }
            for target in branch.routes.values() {
                if target.as_str() != END {
                    self.require_node(target)?;
                }
            }
        }

        // command-routing nodes must not also have static/conditional edges
        for node in &self.command_nodes {
            self.require_node(node)?;
            if self.edges.contains_key(node) || self.branches.contains_key(node) {
                return Err(RustAgentsError::Validation(format!(
                    "node `{node}` declares command routing but also has static/conditional edges"
                )));
            }
        }

        let Self {
            graph_id,
            nodes,
            edges,
            branches,
            command_nodes,
            reducer,
            recursion_limit,
        } = self;

        Ok(CompiledGraph::from_parts(
            graph_id,
            nodes,
            edges,
            branches,
            command_nodes,
            entry,
            reducer.expect("reducer presence checked above"),
            recursion_limit,
        ))
    }

    fn require_node(&self, id: &NodeId) -> Result<()> {
        if self.nodes.contains_key(id) {
            Ok(())
        } else {
            Err(RustAgentsError::MissingNode(id.to_string()))
        }
    }
}

impl<State> GraphBuilder<State, State>
where
    State: Clone + Send + Sync + 'static,
{
    /// Creates a builder that uses whole-state overwrite updates — the
    /// milestone-1 default where each node returns the full next state.
    pub fn overwrite() -> Self {
        Self::new().set_reducer(OverwriteStateReducer)
    }
}

#[cfg(test)]
mod test;
