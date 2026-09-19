//! Unit tests for the subgraph adapters: shared-state embedding, adapter
//! embedding (including folding child output together with parent context), and
//! checkpoint-namespace isolation — verifying that recursively nested and
//! sibling embeddings accumulate distinct namespaces and never collide on
//! checkpoint ids when sharing one checkpointer and thread.

use std::collections::HashSet;
use std::sync::Arc;

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
use crate::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::command::NodeResult;
use crate::reducer::ClosureStateReducer;
use async_trait::async_trait;
use tinyagents_harness::ids::{NodeId, RunId};

#[derive(Default)]
struct NestedRecordingInvoker(std::sync::Mutex<Vec<crate::subagent_node::AgentInvocation>>);

#[async_trait]
impl crate::subagent_node::AgentInvoker for NestedRecordingInvoker {
    async fn invoke(
        &self,
        request: crate::subagent_node::AgentInvocation,
    ) -> crate::Result<crate::subagent_node::SubAgentOutput> {
        self.0.lock().unwrap().push(request.clone());
        request
            .events
            .emit(tinyagents_harness::events::AgentEvent::StateUpdate);
        Ok(crate::subagent_node::SubAgentOutput {
            text: request.input.prompt,
            ..Default::default()
        })
    }
}

/// Builds a minimal [`NodeContext`] standing in for the embedding node `id`.
fn ctx_for(id: &str) -> NodeContext {
    ctx_for_task(id, "task-test", 1)
}

/// Builds a minimal [`NodeContext`] standing in for a `Send` fan-out
/// activation of embedding node `id`: `task_id` names this activation and
/// `siblings` is the fan-out width (I1), which is what a subgraph node
/// consults to decide whether to namespace its child checkpoint by task id.
fn ctx_for_task(id: &str, task_id: &str, siblings: usize) -> NodeContext {
    NodeContext {
        graph_id: tinyagents_harness::ids::GraphId::new("graph-test"),
        node_id: NodeId::from(id),
        run_id: RunId::new("run-test"),
        thread_id: None,
        step: 1,
        resume: None,
        fork: None,
        send_arg: None,
        root_run_id: None,
        recursion_frames: Vec::new(),
        child_runs: None,
        agent_binding: None,
        task_id: tinyagents_harness::ids::TaskId::from(task_id),
        siblings,
    }
}

/// A small child graph (shared state `i32`) that adds 10.
fn child_add_ten() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("add", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 10))
        })
        .set_entry("add")
        .set_finish("add")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn shared_state_subgraph() {
    let child = child_add_ten();
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("pre", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("child", shared_subgraph_node(child))
        .set_entry("pre")
        .add_edge("pre", "child")
        .set_finish("child")
        .compile()
        .unwrap();

    // 0 -> pre(+1) -> child(+10) = 11
    let run = parent.run(0).await.unwrap();
    assert_eq!(run.state, 11);
}

#[tokio::test]
async fn embedded_graph_propagates_the_host_agent_invoker() {
    let invoker = Arc::new(NestedRecordingInvoker::default());
    let child = GraphBuilder::<String, String>::overwrite()
        .add_node(
            "delegate",
            crate::subagent_node::subagent_node(crate::subagent_node::SubAgentNode::from_fns(
                "researcher",
                |state: &String| crate::subagent_node::SubAgentInput::prompt(state.clone()),
                |out: crate::subagent_node::SubAgentOutput| out.text,
            )),
        )
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap();
    let parent = GraphBuilder::<String, String>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap();

    let run = parent
        .run_with_agent_binding(
            "nested".to_string(),
            crate::subagent_node::AgentInvocationBinding::new(
                invoker.clone(),
                tinyagents_harness::events::EventSink::new(),
                tinyagents_harness::cancel::CancellationToken::new(),
            ),
        )
        .await
        .unwrap();
    let requests = invoker.0.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].root_run_id, run.root_run_id);
    assert_ne!(requests[0].parent_run_id, run.run_id);
    assert_eq!(requests[0].input.prompt, "nested");
}

#[derive(Clone, Debug, PartialEq)]
struct ParentState {
    name: String,
    score: i32,
}

#[tokio::test]
async fn adapter_subgraph_maps_state() {
    // child works on a bare i32 score
    let child = child_add_ten();

    let parent = GraphBuilder::<ParentState, ParentState>::new()
        .set_reducer(ClosureStateReducer::new(|_old, new: ParentState| Ok(new)))
        .add_node(
            "score",
            adapter_subgraph_node(
                child,
                // project parent -> child input
                |p: &ParentState| p.score,
                // fold child output -> parent update
                |p: &ParentState, child_score: i32| ParentState {
                    name: p.name.clone(),
                    score: child_score,
                },
            ),
        )
        .set_entry("score")
        .set_finish("score")
        .compile()
        .unwrap();

    let run = parent
        .run(ParentState {
            name: "alice".to_string(),
            score: 5,
        })
        .await
        .unwrap();
    assert_eq!(run.state.name, "alice");
    assert_eq!(run.state.score, 15);
}

#[tokio::test]
async fn adapter_folds_child_output_with_parent_context() {
    // `from_child` receives BOTH the original parent state and the child output,
    // so it can combine them rather than just replacing parent fields.
    let child = child_add_ten();

    let parent = GraphBuilder::<ParentState, ParentState>::new()
        .set_reducer(ClosureStateReducer::new(|_old, new: ParentState| Ok(new)))
        .add_node(
            "score",
            adapter_subgraph_node(
                child,
                |p: &ParentState| p.score,
                |p: &ParentState, child_score: i32| ParentState {
                    name: format!("{}-scored", p.name),
                    // combine parent's own score (5) with the child output (15).
                    score: p.score + child_score,
                },
            ),
        )
        .set_entry("score")
        .set_finish("score")
        .compile()
        .unwrap();

    let run = parent
        .run(ParentState {
            name: "bob".to_string(),
            score: 5,
        })
        .await
        .unwrap();
    // child: 5 + 10 = 15; from_child uses both args: 5 + 15 = 20.
    assert_eq!(run.state.score, 20);
    assert_eq!(run.state.name, "bob-scored");
}

#[test]
fn namespaced_clone_appends_embedding_node_id() {
    let child = child_add_ten();
    assert!(child.namespace().is_empty());
    let scoped = namespaced(&child, &ctx_for("embed"));
    assert_eq!(scoped.namespace(), &["embed".to_string()]);
}

#[test]
fn nested_namespaces_accumulate_and_stay_distinct() {
    let child = child_add_ten();
    // Two levels of embedding accumulate node ids in order.
    let outer = namespaced(&child, &ctx_for("outer"));
    let inner = namespaced(&outer, &ctx_for("inner"));
    assert_eq!(
        inner.namespace(),
        &["outer".to_string(), "inner".to_string()]
    );
    // Siblings embedded under different node ids get distinct namespaces, so
    // their checkpoints can never collide.
    let sibling = namespaced(&child, &ctx_for("other"));
    assert_ne!(outer.namespace(), sibling.namespace());
}

#[tokio::test]
async fn namespaced_children_persist_under_isolated_namespaces() {
    // A single compiled child embedded under two different node ids, sharing one
    // checkpointer and thread: every checkpoint is tagged with its own namespace
    // and keeps a globally-unique id, so the two embeddings never collide.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let child = child_add_ten().with_checkpointer(ckpt.clone());

    let branch_a = namespaced(&child, &ctx_for("branch_a"));
    let branch_b = namespaced(&child, &ctx_for("branch_b"));

    branch_a.run_with_thread("t", 0).await.unwrap();
    branch_b.run_with_thread("t", 1).await.unwrap();

    let list = ckpt.list("t").await.unwrap();
    assert_eq!(list.len(), 2);

    // Checkpoint ids are unique (no collision).
    let ids: HashSet<&str> = list.iter().map(|m| m.checkpoint_id.as_str()).collect();
    assert_eq!(ids.len(), 2);

    // Each embedding's checkpoint carries its own namespace.
    assert!(
        list.iter()
            .any(|m| m.namespace == vec!["branch_a".to_string()])
    );
    assert!(
        list.iter()
            .any(|m| m.namespace == vec!["branch_b".to_string()])
    );
}

#[tokio::test]
async fn embedded_child_persists_under_parent_thread_and_child_namespace() {
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let child = child_add_ten().with_checkpointer(ckpt.clone());
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child.clone()))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let run = parent.run_with_thread("t", 0).await.unwrap();
    assert_eq!(run.state, 10);

    let list = ckpt.list("t").await.unwrap();
    assert_eq!(list.len(), 2);
    assert!(list.iter().any(|m| m.namespace.is_empty()));
    let child_meta = list
        .iter()
        .find(|m| m.namespace == vec!["child".to_string()])
        .expect("child checkpoint is stored under the embedding namespace");

    let child_scoped = namespaced(&child, &ctx_for("child"));
    let child_state = child_scoped
        .get_state("t", Some(&child_meta.checkpoint_id))
        .await
        .unwrap()
        .expect("child checkpoint can be loaded from the parent thread");
    assert_eq!(child_state.values, 10);
    assert_eq!(child_state.next_nodes, Vec::<NodeId>::new());
}

/// A child graph that pauses on an interrupt until resumed.
fn child_interrupts() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("gate", |s: i32, c: NodeContext| async move {
            match c.resume {
                Some(_) => Ok(NodeResult::Update(s + 10)),
                None => Ok(NodeResult::Interrupt(crate::command::Interrupt::new(
                    "gate",
                    serde_json::json!({ "ask": "ok?" }),
                ))),
            }
        })
        .set_entry("gate")
        .set_finish("gate")
        .compile()
        .unwrap()
}

#[tokio::test]
async fn subgraph_interrupt_propagates_to_parent() {
    // A child graph that interrupts must pause the *parent* run — not be
    // swallowed as a completed output with the child's partial state.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let child = child_interrupts().with_checkpointer(ckpt.clone());
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let run = parent.run_with_thread("t", 0).await.unwrap();
    assert!(
        run.is_interrupted(),
        "a child interrupt must pause the parent instead of being swallowed"
    );
    assert_eq!(run.interrupts.len(), 1, "the child's interrupt is surfaced");
    assert_eq!(
        run.interrupts[0].node.as_str(),
        "gate",
        "the propagated interrupt names the child's paused node"
    );
}

#[tokio::test]
async fn subgraph_interrupt_resumes_child_from_its_own_checkpoint() {
    // Resuming the parent must resume the *child* from its namespaced
    // checkpoint (not re-run it and not load the parent's checkpoint), so the
    // whole nested run completes. Exercises namespace-aware resume end to end.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let child = child_interrupts().with_checkpointer(ckpt.clone());
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let paused = parent.run_with_thread("t", 0).await.unwrap();
    assert!(paused.is_interrupted());

    let done = parent
        .resume(
            "t",
            crate::command::Command::resume(serde_json::json!("go")),
        )
        .await
        .unwrap();
    assert!(
        !done.is_interrupted(),
        "resuming the parent must drive the child to completion"
    );
    // Child added 10 to the initial 0.
    assert_eq!(done.state, 10);
}

#[tokio::test]
async fn resumed_subgraph_passes_the_supplied_binding_to_its_subagent() {
    // The child pauses before its SubAgentNode. The parent resume must route
    // the fresh binding into the child resume (rather than dropping it at the
    // subgraph boundary), where it reaches the durable continuation.
    let ckpt = Arc::new(InMemoryCheckpointer::<String>::new());
    let child = GraphBuilder::<String, String>::overwrite()
        .add_node("gate", |state: String, ctx: NodeContext| async move {
            if ctx.resume.is_some() {
                Ok(NodeResult::Update(state))
            } else {
                Ok(NodeResult::Interrupt(crate::command::Interrupt::new(
                    "gate",
                    serde_json::json!({ "ask": "continue?" }),
                )))
            }
        })
        .add_node(
            "delegate",
            crate::subagent_node::subagent_node(crate::subagent_node::SubAgentNode::from_fns(
                "researcher",
                |state: &String| crate::subagent_node::SubAgentInput::prompt(state.clone()),
                |output: crate::subagent_node::SubAgentOutput| output.text,
            )),
        )
        .set_entry("gate")
        .add_edge("gate", "delegate")
        .set_finish("delegate")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());
    let parent = GraphBuilder::<String, String>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt);

    let paused = parent
        .run_with_thread("nested-resume", "question".to_string())
        .await
        .unwrap();
    assert!(paused.is_interrupted());

    let invoker = Arc::new(NestedRecordingInvoker::default());
    let events = tinyagents_harness::events::EventSink::new();
    let listener = Arc::new(tinyagents_harness::events::RecordingListener::new());
    events.subscribe(listener.clone());
    let cancellation = tinyagents_harness::cancel::CancellationToken::new();
    cancellation.cancel();
    let resumed = parent
        .resume_with_agent_binding(
            "nested-resume",
            crate::command::Command::resume(serde_json::json!("go")),
            crate::subagent_node::AgentInvocationBinding::new(
                invoker.clone(),
                events,
                cancellation,
            ),
        )
        .await
        .unwrap();

    assert_eq!(resumed.state, "question");
    assert_eq!(resumed.child_runs.len(), 1);
    let request = invoker.0.lock().unwrap().pop().expect("child delegated");
    assert_eq!(request.parent_run_id, resumed.child_runs[0].run_id);
    assert_eq!(request.root_run_id, resumed.root_run_id);
    assert!(
        request
            .cancellation
            .expect("binding has cancellation")
            .is_cancelled()
    );
    assert_eq!(
        listener.len(),
        1,
        "sub-agent event used the resumed binding sink"
    );
}

#[tokio::test]
async fn subgraph_child_run_distinct_and_shares_root() {
    // A parent embedding one child: the parent run records exactly one child run
    // whose run id differs from the parent's, and whose root run id equals the
    // parent's (the child preserves the root of the recursion tree).
    let child = child_add_ten();
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap();

    let run = parent.run(0).await.unwrap();
    assert_eq!(run.state, 10);

    // Exactly one child run, keyed by the embedding node.
    assert_eq!(run.child_runs.len(), 1);
    let child_run = &run.child_runs[0];
    assert_eq!(child_run.node.as_str(), "child");
    // Distinct child run id, shared root.
    assert_ne!(child_run.run_id, run.run_id);
    assert_eq!(child_run.root_run_id, run.run_id);
    assert_eq!(run.root_run_id, run.run_id);
    assert!(run.parent_run_id.is_none());

    // The run tree mirrors the execution's lineage.
    let tree = run.run_tree();
    assert!(tree.is_root());
    assert_eq!(tree.children.len(), 1);
    assert_eq!(tree.children[0].run_id, child_run.run_id);
}

#[tokio::test]
async fn nested_subgraphs_produce_distinct_ids_sharing_one_root() {
    // grandchild adds 10; child embeds grandchild then is itself embedded in the
    // parent, so the run tree is three deep: parent -> child -> grandchild.
    let grandchild = child_add_ten();
    let child = GraphBuilder::<i32, i32>::overwrite()
        .add_node("grandchild", shared_subgraph_node(grandchild))
        .set_entry("grandchild")
        .set_finish("grandchild")
        .compile()
        .unwrap();
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap();

    let run = parent.run(0).await.unwrap();
    // 0 -> child -> grandchild(+10) = 10
    assert_eq!(run.state, 10);

    // Parent records the child run; that child's run shares the parent's root.
    assert_eq!(run.child_runs.len(), 1);
    let child_run = &run.child_runs[0];
    assert_eq!(child_run.node.as_str(), "child");
    assert_eq!(child_run.root_run_id, run.run_id);

    // All three run ids are distinct (parent, child, grandchild). The grandchild
    // is recorded on the child run's own child_runs, but the top-level parent
    // sees only its direct child; we assert direct-child distinctness here.
    assert_ne!(child_run.run_id, run.run_id);
}

#[tokio::test]
async fn parent_frames_balanced_after_subgraph_returns() {
    // Two sibling subgraph nodes run in sequence under one parent. Because each
    // child runs on its own seeded recursion stack, the parent's depth is never
    // mutated by a child returning, so both child runs see depth-consistent
    // lineage: each is a direct child of the parent and shares its root.
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", shared_subgraph_node(child_add_ten()))
        .add_node("b", shared_subgraph_node(child_add_ten()))
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .unwrap();

    let run = parent.run(0).await.unwrap();
    // 0 -> a(+10) -> b(+10) = 20
    assert_eq!(run.state, 20);

    assert_eq!(run.child_runs.len(), 2);
    let nodes: Vec<&str> = run.child_runs.iter().map(|c| c.node.as_str()).collect();
    assert_eq!(nodes, vec!["a", "b"]);
    // Both children share the parent's root and have distinct run ids.
    for c in &run.child_runs {
        assert_eq!(c.root_run_id, run.run_id);
        assert_ne!(c.run_id, run.run_id);
    }
    assert_ne!(run.child_runs[0].run_id, run.child_runs[1].run_id);
}

#[tokio::test]
async fn child_runs_recorded_in_checkpoint_metadata() {
    // With a checkpointer + thread, the boundary checkpoint that committed the
    // subgraph node carries a `child_runs` array in its metadata keyed by node.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child_add_ten()))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let run = parent.run_with_thread("t", 0).await.unwrap();
    assert_eq!(run.child_runs.len(), 1);

    // Walk every persisted checkpoint's raw metadata for the `child_runs` array
    // naming the embedding node.
    let list = ckpt.list("t").await.unwrap();
    let mut found = false;
    for meta in &list {
        let checkpoint = ckpt
            .get("t", Some(&meta.checkpoint_id))
            .await
            .unwrap()
            .unwrap();
        if checkpoint
            .metadata
            .get("child_runs")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter()
                    .any(|c| c.get("node").and_then(|n| n.as_str()) == Some("child"))
            })
        {
            found = true;
            break;
        }
    }
    assert!(found, "child_runs not found in any checkpoint metadata");
}

// ---- I1: Send fan-out of a subgraph node gets its own checkpoint namespace,
//      and R5: task-scoped interrupt/resume identity -----------------------

#[tokio::test]
async fn send_fanout_of_subgraph_node_gets_per_task_namespaces_and_resume() {
    // A `Send` fan-out of three activations of one subgraph node, each of
    // whose children interrupts: the parent must surface all three as
    // distinct task-scoped interrupts (not just the lowest-index one), their
    // children must persist under three distinct checkpoint namespaces (not
    // one shared `["child"]` namespace all three interleave into), and a
    // per-task resume map must deliver each activation its own value.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let seen = Arc::new(std::sync::Mutex::new(Vec::<i64>::new()));
    let seen_child = seen.clone();
    let child = GraphBuilder::<i32, i32>::overwrite()
        .add_node("gate", move |s: i32, c: NodeContext| {
            let seen_child = seen_child.clone();
            async move {
                match c.resume {
                    Some(v) => {
                        seen_child.lock().unwrap().push(v.as_i64().unwrap());
                        Ok(NodeResult::Update(s))
                    }
                    None => Ok(NodeResult::Interrupt(crate::command::Interrupt::new(
                        "gate",
                        serde_json::json!({ "ask": "ok?" }),
                    ))),
                }
            }
        })
        .set_entry("gate")
        .set_finish("gate")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let parent = GraphBuilder::<i32, i32>::overwrite()
        .with_parallel(true)
        .add_node("dispatch", |_s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Command(crate::command::Command::send([
                crate::command::Send::new("child", serde_json::json!(0)),
                crate::command::Send::new("child", serde_json::json!(1)),
                crate::command::Send::new("child", serde_json::json!(2)),
            ])))
        })
        .add_node("child", shared_subgraph_node(child))
        .set_entry("dispatch")
        .mark_command_routing("dispatch")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let paused = parent.run_with_thread("t", 0).await.unwrap();
    assert!(
        paused.is_interrupted(),
        "every fan-out branch's child interrupted"
    );
    assert_eq!(
        paused.interrupts.len(),
        3,
        "all three fan-out branches are surfaced, not only the lowest-index one"
    );
    let task_ids: HashSet<String> = paused
        .interrupts
        .iter()
        .map(|i| {
            i.task_id
                .clone()
                .expect("stamped with its own branch's task id (R5)")
                .as_str()
                .to_string()
        })
        .collect();
    assert_eq!(
        task_ids.len(),
        3,
        "each fan-out branch's interrupt carries a distinct task id"
    );

    // Each branch's child persisted under its own namespace (I1): three
    // distinct `["child", task_id]` namespaces, not one shared `["child"]`.
    let list = ckpt.list("t").await.unwrap();
    let child_namespaces: HashSet<Vec<String>> = list
        .iter()
        .filter(|m| m.namespace.first().map(String::as_str) == Some("child"))
        .map(|m| m.namespace.clone())
        .collect();
    assert_eq!(
        child_namespaces.len(),
        3,
        "each fan-out branch's child checkpoints live under a distinct namespace"
    );
    for ns in &child_namespaces {
        assert_eq!(
            ns.len(),
            2,
            "namespace is [node_id, task_id] once the node fans out (I1): {ns:?}"
        );
    }

    // Resume every branch with its own value in one call (I1).
    let pairs: Vec<(tinyagents_harness::ids::TaskId, serde_json::Value)> = paused
        .interrupts
        .iter()
        .enumerate()
        .map(|(i, interrupt)| {
            (
                interrupt.task_id.clone().unwrap(),
                serde_json::json!(100 + i as i64),
            )
        })
        .collect();
    let done = parent
        .resume("t", crate::command::Command::resume_tasks(pairs))
        .await
        .unwrap();
    assert!(!done.is_interrupted());

    let mut delivered = seen.lock().unwrap().clone();
    delivered.sort_unstable();
    assert_eq!(
        delivered,
        vec![100, 101, 102],
        "each activation's child received its own resume value"
    );
}

// ---- C4: a subgraph child failure is resumable through the parent --------

#[tokio::test]
async fn parent_retry_after_subgraph_child_failure_resumes_not_restarts() {
    // The child increments a shared side-effect counter on its first node,
    // then fails on its second. A parent `retry()` must continue the child
    // from its own resumable checkpoint (not restart it from scratch), so
    // the first node's side effect never runs twice.
    let ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let should_fail = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let counter_for_node = counter.clone();
    let should_fail_for_node = should_fail.clone();
    let child = GraphBuilder::<i32, i32>::overwrite()
        .add_node("bump", move |s: i32, _c: NodeContext| {
            let counter = counter_for_node.clone();
            async move {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(NodeResult::Update(s + 1))
            }
        })
        .add_node("maybe_fail", move |s: i32, _c: NodeContext| {
            let should_fail = should_fail_for_node.clone();
            async move {
                if should_fail.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    return Err(crate::TinyAgentsError::Graph("boom".to_string()));
                }
                Ok(NodeResult::Update(s + 1))
            }
        })
        .set_entry("bump")
        .add_edge("bump", "maybe_fail")
        .set_finish("maybe_fail")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("child", shared_subgraph_node(child))
        .set_entry("child")
        .set_finish("child")
        .compile()
        .unwrap()
        .with_checkpointer(ckpt.clone());

    let failed = parent.run_with_thread("t", 0).await;
    assert!(
        failed.is_err(),
        "the child's node failure aborts the parent run"
    );
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the child's first node ran exactly once before its second node failed"
    );

    let done = parent
        .retry("t")
        .await
        .expect("retry must continue the child from its resumable checkpoint");
    assert!(!done.is_interrupted());
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "C4: retry must not re-run the child's already-completed node"
    );
    // bump(+1) -> maybe_fail(+1) = 2, once retried past the failure.
    assert_eq!(done.state, 2);
}
