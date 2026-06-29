//! Durable state-graph execution with partial updates, a reducer, and
//! checkpointing.
//!
//! Builds a [`GraphBuilder`] over a `Counter` state. Each node returns a small
//! partial `i32` update; a [`ClosureStateReducer`] folds those updates into the
//! running counter and appends a log line, so the final state reflects every
//! superstep. The graph runs on a thread backed by an [`InMemoryCheckpointer`]
//! so we can list the checkpoints written at each superstep boundary.
//!
//! Run with:
//!
//! ```text
//! cargo run --example durable_graph
//! ```

use std::sync::Arc;

use rustagents::graph::ClosureStateReducer;
use rustagents::{
    Checkpointer, GraphBuilder, InMemoryCheckpointer, NodeContext, NodeResult, Result,
};

/// Running counter plus an append-only audit log of the updates applied.
#[derive(Clone, Debug, Default)]
struct Counter {
    value: i64,
    log: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let checkpointer = Arc::new(InMemoryCheckpointer::<Counter>::new());

    // State = Counter, Update = i64. The reducer merges each partial update into
    // the counter and records it in the log.
    let graph = GraphBuilder::<Counter, i64>::new()
        .set_reducer(ClosureStateReducer::new(|mut state: Counter, update: i64| {
            state.value += update;
            state.log.push(format!("+{update} => {}", state.value));
            Ok(state)
        }))
        .add_node("seed", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(1))
        })
        .add_node("grow", |s: Counter, _c: NodeContext| async move {
            // Partial update derived from the current state.
            Ok(NodeResult::Update(s.value * 10))
        })
        .add_node("finish", |_s: Counter, _c: NodeContext| async move {
            Ok(NodeResult::Update(7))
        })
        .set_entry("seed")
        .add_edge("seed", "grow")
        .add_edge("grow", "finish")
        .set_finish("finish")
        .compile()?
        .with_checkpointer(checkpointer.clone());

    let run = graph.run_with_thread("thread-1", Counter::default()).await?;

    println!("=== Durable graph run ===");
    println!("final value: {}", run.state.value);
    println!("update log : {:?}", run.state.log);
    println!("visited    : {:?}", run.visited);
    println!("supersteps : {}", run.steps);
    println!("status     : {:?}", run.status.status);

    let checkpoints = checkpointer.list("thread-1").await?;
    println!("checkpoints: {} written for thread-1", checkpoints.len());
    for meta in &checkpoints {
        println!(
            "  - {} (step {}, source {}, parent {:?})",
            meta.checkpoint_id, meta.step, meta.source, meta.parent_checkpoint_id
        );
    }

    Ok(())
}
