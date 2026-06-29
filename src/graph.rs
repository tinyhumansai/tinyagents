use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use crate::{Result, RustAgentsError};

pub type BoxNodeFuture<State> = Pin<Box<dyn Future<Output = Result<NodeOutput<State>>> + Send>>;
pub type NodeFn<State> = dyn Fn(State) -> BoxNodeFuture<State> + Send + Sync;

#[derive(Clone)]
pub struct Node<State> {
    name: String,
    handler: Arc<NodeFn<State>>,
}

impl<State> Node<State>
where
    State: Send + 'static,
{
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

    pub fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, state: State) -> Result<NodeOutput<State>> {
        (self.handler)(state).await
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeOutput<State> {
    Continue(State),
    Route { state: State, route: String },
    End(State),
}

impl<State> NodeOutput<State> {
    pub fn continue_with(state: State) -> Self {
        Self::Continue(state)
    }

    pub fn route(state: State, route: impl Into<String>) -> Self {
        Self::Route {
            state,
            route: route.into(),
        }
    }

    pub fn end(state: State) -> Self {
        Self::End(state)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Edge {
    Direct(String),
    Conditional(HashMap<String, String>),
    End,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphRun<State> {
    pub state: State,
    pub visited: Vec<String>,
}

pub struct StateGraph<State> {
    nodes: HashMap<String, Node<State>>,
    edges: HashMap<String, Edge>,
    start: Option<String>,
    recursion_limit: usize,
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
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            start: None,
            recursion_limit: 50,
        }
    }

    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    pub fn add_node(mut self, node: Node<State>) -> Self {
        self.nodes.insert(node.name().to_string(), node);
        self
    }

    pub fn set_start(mut self, name: impl Into<String>) -> Self {
        self.start = Some(name.into());
        self
    }

    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.insert(from.into(), Edge::Direct(to.into()));
        self
    }

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

    pub fn add_end(mut self, from: impl Into<String>) -> Self {
        self.edges.insert(from.into(), Edge::End);
        self
    }

    pub fn validate(&self) -> Result<()> {
        let start = self.start.as_ref().ok_or(RustAgentsError::MissingStart)?;
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

    pub async fn run(&self, initial_state: State) -> Result<GraphRun<State>> {
        self.validate()?;

        let mut state = initial_state;
        let mut current = self.start.clone().ok_or(RustAgentsError::MissingStart)?;
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

        Err(RustAgentsError::RecursionLimit(self.recursion_limit))
    }

    fn next_direct_node(&self, node: &str) -> Result<Option<String>> {
        match self.edges.get(node) {
            Some(Edge::Direct(next)) => Ok(Some(next.clone())),
            Some(Edge::End) | None => Ok(None),
            Some(Edge::Conditional(_)) => Err(RustAgentsError::MissingRoute {
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
                    .ok_or_else(|| RustAgentsError::MissingRoute {
                        node: node.to_string(),
                        route: route.to_string(),
                    })
            }
            _ => Err(RustAgentsError::MissingRoute {
                node: node.to_string(),
                route: route.to_string(),
            }),
        }
    }

    fn require_node(&self, name: &str) -> Result<&Node<State>> {
        self.nodes
            .get(name)
            .ok_or_else(|| RustAgentsError::MissingNode(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestState {
        count: usize,
    }

    #[tokio::test]
    async fn runs_direct_graph() {
        let graph = StateGraph::new()
            .add_node(Node::new("increment", |mut state: TestState| async move {
                state.count += 1;
                Ok(NodeOutput::continue_with(state))
            }))
            .add_node(Node::new("finish", |state| async move {
                Ok(NodeOutput::end(state))
            }))
            .set_start("increment")
            .add_edge("increment", "finish");

        let run = graph.run(TestState { count: 0 }).await.unwrap();

        assert_eq!(run.state.count, 1);
        assert_eq!(run.visited, vec!["increment", "finish"]);
    }

    #[tokio::test]
    async fn runs_conditional_graph() {
        let graph = StateGraph::new()
            .add_node(Node::new("router", |state: TestState| async move {
                let route = if state.count == 0 { "empty" } else { "ready" };
                Ok(NodeOutput::route(state, route))
            }))
            .add_node(Node::new("empty", |mut state: TestState| async move {
                state.count = 1;
                Ok(NodeOutput::end(state))
            }))
            .add_node(Node::new("ready", |state| async move {
                Ok(NodeOutput::end(state))
            }))
            .set_start("router")
            .add_conditional_edges("router", [("empty", "empty"), ("ready", "ready")]);

        let run = graph.run(TestState { count: 0 }).await.unwrap();

        assert_eq!(run.state.count, 1);
        assert_eq!(run.visited, vec!["router", "empty"]);
    }
}
