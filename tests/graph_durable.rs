//! End-to-end coverage for the durable state-graph runtime.
//!
//! Covers reducer-applied partial updates, the deterministic recursion-limit
//! error, checkpoint put/get/list via [`InMemoryCheckpointer`], and an
//! interrupt-then-resume round trip.

use std::sync::Arc;

use serde_json::json;

use tinyagents::graph::ClosureStateReducer;
use tinyagents::{
    Checkpointer, Command, GraphBuilder, InMemoryCheckpointer, Interrupt, NodeContext, NodeResult,
    TinyAgentsError,
};

/// Running counter plus an audit log, used to prove partial updates are merged
/// by the reducer rather than overwriting whole state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Counter {
    value: i64,
    log: Vec<String>,
}

#[tokio::test]
async fn reducer_merges_partial_updates_into_final_state() {
    let graph = GraphBuilder::<Counter, i64>::new()
        .set_reducer(ClosureStateReducer::new(
            |mut state: Counter, update: i64| {
                state.value += update;
                state.log.push(format!("+{update}"));
                Ok(state)
            },
        ))
        .add_node("seed", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("grow", |s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(s.value * 10))
        })
        .add_node("finish", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(7))
        })
        .set_entry("seed")
        .add_edge("seed", "grow")
        .add_edge("grow", "finish")
        .set_finish("finish")
        .compile()
        .expect("graph compiles");

    let run = graph.run(Counter::default()).await.expect("run succeeds");

    // 0 +1 => 1, then +10 => 11, then +7 => 18.
    assert_eq!(run.state.value, 18);
    assert_eq!(run.state.log, vec!["+1", "+10", "+7"]);
    let visited: Vec<&str> = run.visited.iter().map(|n| n.as_str()).collect();
    assert_eq!(visited, vec!["seed", "grow", "finish"]);
    assert_eq!(run.steps, 3);
}

#[tokio::test]
async fn recursion_limit_is_deterministic() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .with_recursion_limit(3)
        .add_node("loop", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("loop")
        .add_edge("loop", "loop")
        .compile()
        .expect("graph compiles");

    let err = graph.run(0).await.expect_err("the loop never terminates");
    assert!(
        matches!(err, TinyAgentsError::RecursionLimit(3)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn checkpointer_put_get_list_round_trip() {
    let checkpointer = Arc::new(InMemoryCheckpointer::<i64>::new());

    let graph = GraphBuilder::<i64, i64>::overwrite()
        .add_node("a", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .expect("graph compiles")
        .with_checkpointer(checkpointer.clone());

    let run = graph
        .run_with_thread("t1", 0)
        .await
        .expect("threaded run succeeds");
    assert_eq!(run.state, 2);
    assert!(run.checkpoint_id.is_some());

    // One checkpoint per superstep boundary, chained parent -> child.
    let list = checkpointer.list("t1").await.expect("list succeeds");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].parent_checkpoint_id, None);
    assert_eq!(
        list[1].parent_checkpoint_id.as_deref(),
        Some(list[0].checkpoint_id.as_str())
    );

    // `get` with None returns the latest committed checkpoint.
    let latest = checkpointer
        .get("t1", None)
        .await
        .expect("get latest succeeds")
        .expect("a latest checkpoint exists");
    assert_eq!(latest.state, 2);
    assert_eq!(latest.thread_id, "t1");

    // `get` for an unknown thread is empty, not an error.
    let missing = checkpointer
        .get("does-not-exist", None)
        .await
        .expect("get is infallible");
    assert!(missing.is_none());
}

#[tokio::test]
async fn interrupt_then_resume_yields_resumed_result() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .add_node("approve", |s: i64, ctx: NodeContext| async move {
            match ctx.resume {
                Some(value) => {
                    let bump = value.get("bump").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(NodeResult::Update(s + bump))
                }
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    "approve",
                    json!({ "ask": "approve?" }),
                ))),
            }
        })
        .set_entry("approve")
        .set_finish("approve")
        .compile()
        .expect("graph compiles")
        .with_checkpointer(Arc::new(InMemoryCheckpointer::<i64>::new()));

    // First pass pauses at the interrupt.
    let paused = graph
        .run_with_thread("hitl", 10)
        .await
        .expect("first pass succeeds");
    assert!(paused.is_interrupted());
    assert_eq!(paused.interrupts.len(), 1);

    // Resuming injects the approval value and finishes: 10 + 5 = 15.
    let resumed = graph
        .resume("hitl", Command::resume(json!({ "bump": 5 })))
        .await
        .expect("resume succeeds");
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.state, 15);
}
