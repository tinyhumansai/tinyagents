//! Unit tests for the subgraph adapters: shared-state embedding, adapter
//! embedding (including folding child output together with parent context), and
//! checkpoint-namespace isolation — verifying that recursively nested and
//! sibling embeddings accumulate distinct namespaces and never collide on
//! checkpoint ids when sharing one checkpointer and thread.

use std::collections::HashSet;
use std::sync::Arc;

use super::*;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::graph::command::NodeResult;
use crate::graph::reducer::ClosureStateReducer;
use crate::harness::ids::{NodeId, RunId};

/// Builds a minimal [`NodeContext`] standing in for the embedding node `id`.
fn ctx_for(id: &str) -> NodeContext {
    NodeContext {
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
