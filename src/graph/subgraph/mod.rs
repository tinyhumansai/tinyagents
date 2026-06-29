//! Subgraph node adapters — the graph-level recursion surface where a graph
//! runs another graph.
//!
//! This is the structural counterpart to harness sub-agents (a model calling a
//! model): here an entire [`CompiledGraph`] is embedded *as a node* inside a
//! parent graph, so "graphs that run graphs" is just an ordinary node handler.
//! Each embedding extends the child's checkpoint namespace with the embedding
//! node id, which keeps every level of a recursively
//! nested run durable and collision-free, and the executor's recursion limit
//! bounds how deep that nesting can go.
//!
//! See [`types`] for the conceptual overview of the two embedding modes. The
//! functions here wrap a [`CompiledGraph`] into a node handler usable with
//! [`crate::graph::GraphBuilder::add_node`]:
//!
//! - [`shared_subgraph_node`] — parent and child share one state channel.
//! - [`adapter_subgraph_node`] — parent and child use different state shapes,
//!   bridged by `to_child` / `from_child` mappings.

mod types;

use std::future::Future;
use std::pin::Pin;

use crate::Result;
use crate::graph::builder::NodeContext;
use crate::graph::command::NodeResult;
use crate::graph::compiled::CompiledGraph;

type Handler<S, U> = Box<
    dyn Fn(S, NodeContext) -> Pin<Box<dyn Future<Output = Result<NodeResult<U>>> + Send>>
        + Send
        + Sync,
>;

/// Embeds `child` as a shared-state subgraph node.
///
/// The child uses the same `State` channel as the parent (so `Update == State`)
/// and runs over the parent state passed to the node. Its final state becomes
/// the parent update. The child's checkpoint namespace is extended with the
/// embedding node id.
pub fn shared_subgraph_node<State>(child: CompiledGraph<State, State>) -> Handler<State, State>
where
    State: Clone + Send + Sync + 'static,
{
    Box::new(move |state: State, ctx: NodeContext| {
        let child = namespaced(&child, &ctx);
        Box::pin(async move {
            let execution = child.run(state).await?;
            Ok(NodeResult::Update(execution.state))
        })
    })
}

/// Embeds `child` as an adapter subgraph node mapping between parent and child
/// state shapes.
///
/// `to_child` projects the parent state `P` into the child input `C`;
/// `from_child` folds the child's final state back into a parent update `PU`.
pub fn adapter_subgraph_node<P, PU, C, CU, ToChild, FromChild>(
    child: CompiledGraph<C, CU>,
    to_child: ToChild,
    from_child: FromChild,
) -> Handler<P, PU>
where
    P: Clone + Send + Sync + 'static,
    PU: Send + 'static,
    C: Clone + Send + Sync + 'static,
    CU: Send + 'static,
    ToChild: Fn(&P) -> C + Send + Sync + Clone + 'static,
    FromChild: Fn(&P, C) -> PU + Send + Sync + Clone + 'static,
{
    Box::new(move |state: P, ctx: NodeContext| {
        let child = namespaced(&child, &ctx);
        let to_child = to_child.clone();
        let from_child = from_child.clone();
        Box::pin(async move {
            let child_input = to_child(&state);
            let execution = child.run(child_input).await?;
            let update = from_child(&state, execution.state);
            Ok(NodeResult::Update(update))
        })
    })
}

/// Clones `child` and extends its checkpoint namespace with the embedding node
/// id, preventing parent/child checkpoint collisions.
fn namespaced<S, U>(child: &CompiledGraph<S, U>, ctx: &NodeContext) -> CompiledGraph<S, U> {
    let mut namespace = child.namespace().to_vec();
    namespace.push(ctx.node_id.to_string());
    child.clone().with_namespace(namespace)
}

#[cfg(test)]
mod test;
