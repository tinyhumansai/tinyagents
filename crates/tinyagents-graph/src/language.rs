//! Materialization of declarative language blueprints into executable graphs.
//!
//! This is the bridge from `tinyagents-language`'s parsed [`Blueprint`] (a
//! `.rag` program) to this crate's [`GraphBuilder`]/[`CompiledGraph`]: a host
//! supplies a [`NodeFactory`] that turns each [`NodeSpec`] into a
//! [`BoxedNode`] handler, and [`build_graph`] wires those handlers into a
//! whole-state (`GraphBuilder::overwrite`) graph and compiles it. This module
//! knows nothing about what a node handler actually does — that is entirely
//! the factory's responsibility — only how to assemble the compiled topology
//! around it.
//!
//! [`build_graph`] only lowers a small slice of a [`Blueprint`]: the entry
//! node, each node's Rust-side handler (via [`NodeFactory`]), and its
//! [`Routing`] (a static edge, command-routing marker, or terminal). Every
//! other populated field — per-node fanout, joins, timeouts, retries,
//! metadata, and the graph-level input/output shape, checkpoint/interrupt
//! policy, and join barriers — is not read here. Silently dropping a
//! populated field is worse than refusing to build the graph: an operator
//! could deploy a blueprint believing a declared `timeout` or `retry` policy
//! is enforced when the runtime applies none. See
//! `docs/modules/expressive-language/implementation-status.md` for the exact
//! "lowered" vs "rejected" split and the fields intentionally left out of
//! this check (`channels`/`defaults` — see that doc for why).

use std::collections::BTreeSet;
use std::sync::Arc;

use tinyagents_harness::error::{Result, TinyAgentsError};
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

/// Returns every populated `Blueprint`/`NodeSpec` field that [`build_graph`]
/// does not lower, formatted as `"field (context)"` in a stable order.
///
/// `channels` and `defaults` are deliberately excluded: they are read by
/// [`crate::export`] already and rejecting them would be a breaking change
/// for existing blueprints that declare them purely for introspection (see
/// `docs/modules/expressive-language/implementation-status.md`).
fn ignored_populated_fields(blueprint: &Blueprint) -> Vec<String> {
    let mut ignored = Vec::new();

    if !blueprint.input.is_empty() {
        ignored.push("graph `input`".to_string());
    }
    if !blueprint.output.is_empty() {
        ignored.push("graph `output`".to_string());
    }
    if blueprint.checkpoint.is_some() {
        ignored.push("graph `checkpoint`".to_string());
    }
    if blueprint.interrupt.is_some() {
        ignored.push("graph `interrupt`".to_string());
    }
    if !blueprint.joins.is_empty() {
        ignored.push("graph `joins`".to_string());
    }

    for spec in &blueprint.nodes {
        if !spec.sends.is_empty() {
            ignored.push(format!("node `{}` `sends`", spec.name));
        }
        if !spec.join_sources.is_empty() {
            ignored.push(format!("node `{}` `join_sources`", spec.name));
        }
        if spec.command.as_ref().is_some_and(|c| !c.update.is_empty()) {
            ignored.push(format!("node `{}` `command.update`", spec.name));
        }
        if !spec.options.is_empty() {
            ignored.push(format!("node `{}` `options`", spec.name));
        }
        if spec.timeout.is_some() {
            ignored.push(format!("node `{}` `timeout`", spec.name));
        }
        if !spec.retry.is_empty() {
            ignored.push(format!("node `{}` `retry`", spec.name));
        }
        if !spec.metadata.is_empty() {
            ignored.push(format!("node `{}` `metadata`", spec.name));
        }
    }

    ignored
}

/// Wires a blueprint into a durable whole-state graph.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Compile`] naming every populated blueprint or
/// node field this function does not lower (see
/// `docs/modules/expressive-language/implementation-status.md`), before
/// touching the factory or the builder. Also propagates factory errors and
/// graph topology validation failures.
pub fn build_graph<State, F>(
    blueprint: &Blueprint,
    factory: &F,
) -> Result<CompiledGraph<State, State>>
where
    State: Clone + Send + Sync + 'static,
    F: NodeFactory<State>,
{
    let ignored = ignored_populated_fields(blueprint);
    if !ignored.is_empty() {
        return Err(TinyAgentsError::Compile(format!(
            "build_graph does not lower these populated blueprint fields yet (Phase 5): {}",
            ignored.join(", ")
        )));
    }

    let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());

    for spec in &blueprint.nodes {
        let handler = factory.make(spec)?;
        builder = builder.add_node(spec.name.as_str(), move |state, ctx| {
            (handler.clone())(state, ctx)
        });
        builder = match &spec.routing {
            Routing::Next(target) => builder.add_edge(spec.name.as_str(), target.as_str()),
            Routing::Conditional(routes) => {
                // Conditional routing is not lowered into
                // `add_conditional_edges` here: the node is marked
                // command-routing instead, so the materialized handler
                // itself must resolve `spec.routing`'s labeled targets and
                // return them via `Command::goto` at runtime. The route
                // table is not enforced against a handler's `Command::goto`
                // at compile time: `with_command_destinations` is advisory
                // only (used by `crate::export` to draw/validate the
                // declared destinations), because the runtime always
                // resolves the real successor from the `Command` a node
                // emits. Record it anyway so export/introspection sees the
                // declared labels instead of nothing.
                let destinations: BTreeSet<&str> =
                    routes.iter().map(|(_, target)| target.as_str()).collect();
                builder.with_command_destinations(spec.name.as_str(), destinations)
            }
            Routing::Terminal => builder.set_finish(spec.name.as_str()),
        };
    }

    builder.compile()
}

#[cfg(test)]
mod test {
    use super::*;
    use tinyagents_language::compiler::compile;
    use tinyagents_language::parser::parse_str;

    #[derive(Clone, Debug, Default, PartialEq)]
    struct S {
        trail: Vec<String>,
    }

    struct EchoFactory;

    impl NodeFactory<S> for EchoFactory {
        fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<S>> {
            let name = spec.name.clone();
            Ok(Arc::new(move |mut state: S, _ctx: crate::NodeContext| {
                let name = name.clone();
                Box::pin(async move {
                    state.trail.push(name);
                    Ok(crate::NodeResult::Update(state))
                }) as crate::NodeFuture<S>
            }))
        }
    }

    fn blueprint(src: &str) -> Blueprint {
        compile(&parse_str(src).unwrap()).unwrap().remove(0)
    }

    #[test]
    fn build_graph_rejects_a_populated_ignored_field() {
        let bp = blueprint(
            "graph g { start a node a { kind model next END options [\"yes\", \"no\"] } }",
        );
        assert!(matches!(bp.nodes[0].options.as_slice(), [_, _]));

        let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
        match err {
            TinyAgentsError::Compile(message) => {
                assert!(
                    message.contains("`options`"),
                    "expected the offending field named in the error, got: {message}"
                );
            }
            other => panic!("expected TinyAgentsError::Compile, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn build_graph_accepts_a_blueprint_with_no_ignored_fields() {
        let bp = blueprint(
            "graph g { start a node a { kind model next b } node b { kind model next END } }",
        );
        assert_eq!(bp.start, "a");

        let graph =
            build_graph::<S, _>(&bp, &EchoFactory).expect("no ignored fields, graph builds");
        let run = graph.run(S::default()).await.expect("graph runs to end");
        assert_eq!(run.state.trail, vec!["a".to_string(), "b".to_string()]);
    }
}
