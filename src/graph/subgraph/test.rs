use super::*;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::command::NodeResult;
use crate::graph::reducer::ClosureStateReducer;

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
