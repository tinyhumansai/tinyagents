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
//! See `types` for the conceptual overview of the two embedding modes. The
//! functions here wrap a [`CompiledGraph`] into a node handler usable with
//! [`crate::GraphBuilder::add_node`]:
//!
//! - [`shared_subgraph_node`] — parent and child share one state channel.
//! - [`adapter_subgraph_node`] — parent and child use different state shapes,
//!   bridged by `to_child` / `from_child` mappings.

mod types;

use std::future::Future;
use std::pin::Pin;

use crate::Result;
use crate::builder::NodeContext;
use crate::command::{Command, NodeResult};
use crate::compiled::{CompiledGraph, GraphExecution};
use crate::recursion::ChildRun;

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
        let child = child_for(&child, &ctx);
        let thread_id = ctx.thread_id.clone();
        let resume = ctx.resume.clone();
        let binding = ctx.agent_binding.clone();
        let recorder = ChildRunRecorder::new(&ctx);
        Box::pin(async move {
            let execution = drive_child(child, thread_id, state, resume, binding).await?;
            recorder.record(&execution);
            // A child that paused on an interrupt must surface it to the parent
            // rather than have its partial state treated as a completed output.
            if execution.is_interrupted() {
                return Ok(NodeResult::Interrupt(child_interrupt(execution)));
            }
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
        let child = child_for(&child, &ctx);
        let thread_id = ctx.thread_id.clone();
        let resume = ctx.resume.clone();
        let binding = ctx.agent_binding.clone();
        let recorder = ChildRunRecorder::new(&ctx);
        let to_child = to_child.clone();
        let from_child = from_child.clone();
        Box::pin(async move {
            let child_input = to_child(&state);
            let execution = drive_child(child, thread_id, child_input, resume, binding).await?;
            recorder.record(&execution);
            // Propagate a child interrupt to the parent instead of folding a
            // paused child's partial state through `from_child`.
            if execution.is_interrupted() {
                return Ok(NodeResult::Interrupt(child_interrupt(execution)));
            }
            let update = from_child(&state, execution.state);
            Ok(NodeResult::Update(update))
        })
    })
}

/// Clones `child` and extends its checkpoint namespace with the embedding node
/// id, preventing parent/child checkpoint collisions.
///
/// I1: when this node is activated more than once in the same superstep (a
/// `Send` fan-out of a subgraph node — map-reduce over a subgraph), each
/// concurrent activation gets its own namespace (`[node_id, task_id]`)
/// instead of sharing one (`[node_id]`) across every fan-out branch. Sharing
/// one namespace is what let N concurrent activations write interleaved
/// lineages under the same key and made every fan-out branch's `resume`
/// non-deterministically pick up whichever child checkpoint was written
/// last. With exactly one activation (`ctx.siblings <= 1`, the overwhelmingly
/// common case) the namespace stays `[node_id]` so existing checkpoints
/// remain readable — this is purely additive.
fn namespaced<S, U>(child: &CompiledGraph<S, U>, ctx: &NodeContext) -> CompiledGraph<S, U> {
    let mut namespace = child.namespace().to_vec();
    namespace.push(ctx.node_id.to_string());
    if ctx.siblings > 1 {
        namespace.push(ctx.task_id().as_str().to_string());
    }
    child.clone().with_namespace(namespace)
}

/// Prepares an embedded `child` graph for a subgraph run: extends its checkpoint
/// namespace with the embedding node id (so nested checkpoints never collide),
/// seeds it with the enclosing run's live recursion frames (so the child run
/// extends the parent's recursion tree rather than starting a fresh one), and
/// records the embedding node so the child's root frame names it.
fn child_for<S, U>(child: &CompiledGraph<S, U>, ctx: &NodeContext) -> CompiledGraph<S, U> {
    namespaced(child, ctx)
        .with_recursion_frames(ctx.recursion_frames.clone())
        .with_recursion_node(ctx.node_id.clone())
}

/// What an existing child checkpoint (if any) says to do instead of running
/// fresh — the C4 fix.
enum ChildContinuation {
    /// A prior activation left a resumable failure-boundary checkpoint:
    /// `retry` it so the child's already-completed nodes (and their side
    /// effects) do not re-run.
    Retry,
    /// A prior activation left an interrupted checkpoint and the parent was
    /// handed a resume value for this activation: resume the child with it.
    Resume(serde_json::Value),
}

/// Checks the child's own checkpoint namespace for a record left by an
/// earlier activation of this same subgraph node, before a fresh run would
/// otherwise discard it (C4).
///
/// `child.run_with_thread(..)` on a thread that already has a child
/// checkpoint does not resume it — it starts a second, root-less lineage
/// under the same namespace, silently re-running the child's completed nodes
/// (and their side effects) and orphaning its partial progress. This is what
/// let a parent `retry()` after a subgraph node failure restart the child
/// from scratch instead of continuing it. Consulted by [`drive_child`]
/// before every fresh-run path (not only after a failure): a checkpoint
/// stamped `failed_node` always retries; one stamped `interrupted_nodes`
/// only resumes when the caller supplied a resume value — otherwise (no
/// checkpoint, or an interrupted one with no resume value) the caller's
/// original fresh/resume decision stands.
async fn child_continuation<S, U>(
    child: &CompiledGraph<S, U>,
    thread_id: &tinyagents_harness::ids::ThreadId,
    resume: Option<&serde_json::Value>,
) -> Result<Option<ChildContinuation>>
where
    S: Clone + Send + Sync + 'static,
    U: Send + 'static,
{
    let Some(checkpointer) = child.checkpointer.as_ref() else {
        return Ok(None);
    };
    let Some(checkpoint) = checkpointer
        .get_scoped(thread_id.as_str(), None, child.namespace())
        .await?
    else {
        return Ok(None);
    };
    let has_pending = checkpoint
        .pending_activations
        .as_ref()
        .map(|p| !p.is_empty())
        .unwrap_or(false)
        || !checkpoint.next_nodes.is_empty();
    if !has_pending {
        return Ok(None);
    }
    if checkpoint.metadata.get("failed_node").is_some() {
        return Ok(Some(ChildContinuation::Retry));
    }
    if checkpoint.metadata.get("interrupted_nodes").is_some()
        && let Some(value) = resume
    {
        return Ok(Some(ChildContinuation::Resume(value.clone())));
    }
    Ok(None)
}

/// Drives an embedded child graph for one parent-node activation.
///
/// On a fresh activation (`resume == None`) the child runs from `state`. On a
/// resumed activation (the parent was resumed and delivered a value to this
/// node) the child is *resumed* from its own checkpoint with that value, so a
/// subgraph interrupt is reachable through the parent's resume API rather than
/// re-running the child (which would just re-interrupt forever). Resuming
/// requires the child to have run under a thread; without one, a paused child
/// could not have persisted, so we fall back to a fresh run.
///
/// C4: on a threaded child, [`child_continuation`] is consulted first — a
/// failed child is retried (not restarted) and an interrupted child with a
/// resume value in hand is resumed, regardless of which of the branches below
/// the caller's own `(resume, binding)` shape would otherwise have taken.
async fn drive_child<S, U>(
    child: CompiledGraph<S, U>,
    thread_id: Option<tinyagents_harness::ids::ThreadId>,
    state: S,
    resume: Option<serde_json::Value>,
    binding: Option<crate::subagent_node::AgentInvocationBinding>,
) -> Result<GraphExecution<S>>
where
    S: Clone + Send + Sync + 'static,
    U: Send + 'static,
{
    if let Some(thread_id) = &thread_id
        && let Some(continuation) = child_continuation(&child, thread_id, resume.as_ref()).await?
    {
        let thread_id = thread_id.clone();
        return match (continuation, binding) {
            (ChildContinuation::Retry, Some(binding)) => {
                child.retry_with_agent_binding(thread_id, binding).await
            }
            (ChildContinuation::Retry, None) => child.retry(thread_id).await,
            (ChildContinuation::Resume(value), Some(binding)) => {
                child
                    .resume_with_agent_binding(thread_id, Command::resume(value), binding)
                    .await
            }
            (ChildContinuation::Resume(value), None) => {
                child.resume(thread_id, Command::resume(value)).await
            }
        };
    }
    match (thread_id, resume, binding) {
        (Some(thread_id), None, Some(binding)) => {
            child
                .run_with_thread_agent_binding(thread_id, state, binding)
                .await
        }
        (None, _, Some(binding)) => child.run_with_agent_binding(state, binding).await,
        (Some(thread_id), Some(value), Some(binding)) => {
            child
                .resume_with_agent_binding(thread_id, Command::resume(value), binding)
                .await
        }
        (Some(thread_id), Some(value), None) => {
            child.resume(thread_id, Command::resume(value)).await
        }
        (Some(thread_id), None, None) => child.run_with_thread(thread_id, state).await,
        (None, _, None) => child.run(state).await,
    }
}

/// Extracts the child's paused interrupt so the parent node can re-emit it.
fn child_interrupt<S>(mut execution: GraphExecution<S>) -> crate::command::Interrupt {
    execution
        .interrupts
        .drain(..)
        .next()
        .expect("an interrupted execution carries at least one interrupt")
}

/// Captures the enclosing run's child-run sink and lineage so a subgraph node can
/// report the child run it spawned (its distinct run id sharing the parent's
/// root) back to the executor after the embedded graph returns.
struct ChildRunRecorder {
    node: tinyagents_harness::ids::NodeId,
    sink: Option<crate::recursion::ChildRunSink>,
}

impl ChildRunRecorder {
    fn new(ctx: &NodeContext) -> Self {
        Self {
            node: ctx.node_id.clone(),
            sink: ctx.child_runs.clone(),
        }
    }

    /// Records the embedded run's id (keyed by the embedding node) into the
    /// enclosing run's sink, when one is attached.
    fn record<S>(&self, execution: &GraphExecution<S>) {
        if let Some(sink) = &self.sink {
            sink.record(ChildRun {
                node: self.node.clone(),
                graph_id: execution.graph_id.clone(),
                run_id: execution.run_id.clone(),
                root_run_id: execution.root_run_id.clone(),
                usage: tinyinference_llm::usage::UsageTotals::default(),
                // C4: the child's latest checkpoint, so the parent's own
                // checkpoint metadata (`child_runs`) carries an explicit
                // pointer to the exact record a later `retry`/`resume`
                // continuation would act on.
                checkpoint_id: execution.checkpoint_id.clone(),
            });
        }
    }
}

#[cfg(test)]
mod test;
