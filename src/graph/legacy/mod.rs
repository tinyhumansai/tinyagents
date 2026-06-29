//! Legacy sequential state-graph implementation.
//!
//! See [`types`] for the data definitions. This module provides the original
//! builder, validation, and sequential executor preserved unchanged from the
//! milestone-1 `src/graph.rs`.

mod types;

pub use types::{BoxNodeFuture, Edge, GraphRun, Node, NodeFn, NodeOutput, StateGraph};

use std::{collections::HashMap, future::Future, sync::Arc};

use crate::{Result, TinyAgentsError};

impl<State> Node<State>
where
    State: Send + 'static,
{
    /// Creates a node from an async handler.
    pub fn new<F, Fut>(name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(State) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeOutput<State>>> + Send + 'static,
    {
        Self {
            name: name.into(),
            handler: Arc::new(move |state| Box::pin(handler(state))),
        }
    }

    /// Returns the node name.
    pub fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, state: State) -> Result<NodeOutput<State>> {
        (self.handler)(state).await
    }
}

impl<State> NodeOutput<State> {
    /// Continues to the direct successor with `state`.
    pub fn continue_with(state: State) -> Self {
        Self::Continue(state)
    }

    /// Takes the named conditional route with `state`.
    pub fn route(state: State, route: impl Into<String>) -> Self {
        Self::Route {
            state,
            route: route.into(),
        }
    }

    /// Ends the run with `state`.
    pub fn end(state: State) -> Self {
        Self::End(state)
    }
}

impl<State> Default for StateGraph<State>
where
    State: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<State> StateGraph<State>
where
    State: Send + 'static,
{
    /// Creates an empty graph with a default recursion limit of 50.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            start: None,
            recursion_limit: 50,
        }
    }

    /// Overrides the recursion limit.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Adds a node.
    pub fn add_node(mut self, node: Node<State>) -> Self {
        self.nodes.insert(node.name().to_string(), node);
        self
    }

    /// Sets the start node.
    pub fn set_start(mut self, name: impl Into<String>) -> Self {
        self.start = Some(name.into());
        self
    }

    /// Adds a direct edge.
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.insert(from.into(), Edge::Direct(to.into()));
        self
    }

    /// Adds conditional edges from `from` keyed by route label.
    pub fn add_conditional_edges<I, K, V>(mut self, from: impl Into<String>, routes: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let routes = routes
            .into_iter()
            .map(|(route, target)| (route.into(), target.into()))
            .collect();
        self.edges.insert(from.into(), Edge::Conditional(routes));
        self
    }

    /// Marks `from` as a terminal node.
    pub fn add_end(mut self, from: impl Into<String>) -> Self {
        self.edges.insert(from.into(), Edge::End);
        self
    }

    /// Validates the topology before execution.
    pub fn validate(&self) -> Result<()> {
        let start = self.start.as_ref().ok_or(TinyAgentsError::MissingStart)?;
        self.require_node(start)?;

        for (from, edge) in &self.edges {
            self.require_node(from)?;
            match edge {
                Edge::Direct(to) => {
                    self.require_node(to)?;
                }
                Edge::Conditional(routes) => {
                    for to in routes.values() {
                        self.require_node(to)?;
                    }
                }
                Edge::End => {}
            }
        }

        Ok(())
    }

    /// Runs the graph sequentially from the start node.
    pub async fn run(&self, initial_state: State) -> Result<GraphRun<State>> {
        self.validate()?;

        let mut state = initial_state;
        let mut current = self.start.clone().ok_or(TinyAgentsError::MissingStart)?;
        let mut visited = Vec::new();

        for _ in 0..self.recursion_limit {
            let node = self.require_node(&current)?;
            visited.push(current.clone());

            match node.run(state).await? {
                NodeOutput::End(final_state) => {
                    return Ok(GraphRun {
                        state: final_state,
                        visited,
                    });
                }
                NodeOutput::Continue(next_state) => {
                    state = next_state;
                    match self.next_direct_node(&current)? {
                        Some(next) => current = next,
                        None => return Ok(GraphRun { state, visited }),
                    }
                }
                NodeOutput::Route {
                    state: next_state,
                    route,
                } => {
                    state = next_state;
                    current = self.next_routed_node(&current, &route)?;
                }
            }
        }

        Err(TinyAgentsError::RecursionLimit(self.recursion_limit))
    }

    fn next_direct_node(&self, node: &str) -> Result<Option<String>> {
        match self.edges.get(node) {
            Some(Edge::Direct(next)) => Ok(Some(next.clone())),
            Some(Edge::End) | None => Ok(None),
            Some(Edge::Conditional(_)) => Err(TinyAgentsError::MissingRoute {
                node: node.to_string(),
                route: "continue".to_string(),
            }),
        }
    }

    fn next_routed_node(&self, node: &str, route: &str) -> Result<String> {
        match self.edges.get(node) {
            Some(Edge::Conditional(routes)) => {
                routes
                    .get(route)
                    .cloned()
                    .ok_or_else(|| TinyAgentsError::MissingRoute {
                        node: node.to_string(),
                        route: route.to_string(),
                    })
            }
            _ => Err(TinyAgentsError::MissingRoute {
                node: node.to_string(),
                route: route.to_string(),
            }),
        }
    }

    fn require_node(&self, name: &str) -> Result<&Node<State>> {
        self.nodes
            .get(name)
            .ok_or_else(|| TinyAgentsError::MissingNode(name.to_string()))
    }
}

#[cfg(test)]
mod test;
