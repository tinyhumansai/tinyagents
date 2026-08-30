//! Materialization of declarative language blueprints into executable graphs.

use std::sync::Arc;

use tinyagents_harness::error::Result;
use tinyagents_language::{Blueprint, NodeSpec, Routing};

use crate::{CompiledGraph, GraphBuilder, NodeHandler};

/// A durable node handler materialized from a declarative node specification.
pub type BoxedNode<State> = Arc<NodeHandler<State, State>>;

/// Builds runtime node handlers from declarative node specifications.
pub trait NodeFactory<State> {
    /// Materializes one executable handler.
    ///
    /// # Errors
    ///
    /// Returns an error when the node kind is unsupported or a required
    /// capability binding is unavailable.
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<State>>;
}

/// Wires a blueprint into a durable whole-state graph.
///
/// # Errors
///
/// Propagates factory errors and graph topology validation failures.
pub fn build_graph<State, F>(
    blueprint: &Blueprint,
    factory: &F,
) -> Result<CompiledGraph<State, State>>
where
    State: Clone + Send + Sync + 'static,
    F: NodeFactory<State>,
{
    let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());

    for spec in &blueprint.nodes {
        let handler = factory.make(spec)?;
        builder = builder.add_node(spec.name.as_str(), move |state, ctx| {
            (handler.clone())(state, ctx)
        });
        builder = match &spec.routing {
            Routing::Next(target) => builder.add_edge(spec.name.as_str(), target.as_str()),
            Routing::Conditional(_) => builder.mark_command_routing(spec.name.as_str()),
            Routing::Terminal => builder.set_finish(spec.name.as_str()),
        };
    }

    builder.compile()
}
