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
