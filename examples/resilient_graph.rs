//! Surviving network blips and restarting a failed run.
//!
//! Two resilience primitives, on one durable graph:
//!
//! 1. **Node-level retry.** [`CompiledGraph::with_node_retry`] re-runs a node
//!    from its start when its handler fails with a transient
//!    ([retryable][tinyagents::harness::retry::is_retryable]) error — a model or
//!    tool/network error. A `fetch` node here fails its first two attempts (a
//!    simulated connectivity blip) and succeeds on the third, so the run
//!    completes without any operator involvement.
//!
//! 2. **Resumable failure + restart.** A `commit` node fails *harder* than the
//!    retry budget allows. On a checkpointed thread the run does not vanish: it
//!    persists a resumable failure-boundary checkpoint (the failed node is
//!    scheduled to re-run, earlier progress is preserved) and returns the error.
//!    The "outage" then clears and [`CompiledGraph::retry`] restarts the run
//!    from that checkpoint to completion. A run could equally be *continued on
//!    user feedback* by editing state with `update_state` before `retry`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example resilient_graph
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tinyagents::harness::ids::ExecutionStatus;
use tinyagents::harness::retry::RetryPolicy;
use tinyagents::{
    GraphBuilder, InMemoryCheckpointer, NodeContext, NodeResult, Result, TinyAgentsError,
};

/// A tiny pipeline state: how far we got, plus an audit log.
#[derive(Clone, Debug, Default)]
struct Pipeline {
    fetched: bool,
    committed: bool,
    log: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let checkpointer = Arc::new(InMemoryCheckpointer::<Pipeline>::new());

    // A flaky `fetch`: fails (retryable model error) its first two attempts,
    // then succeeds — a transient connectivity blip the node-retry policy
    // absorbs on its own.
    let fetch_attempts = Arc::new(AtomicUsize::new(0));
    // A `commit` that stays down until the "outage" clears between runs. The
    // outer restart, not the inner retry policy, is what recovers it.
    let outage = Arc::new(AtomicUsize::new(0));

    let fa = fetch_attempts.clone();
    let og = outage.clone();
    let graph = GraphBuilder::<Pipeline, Pipeline>::overwrite()
        .add_node("fetch", move |mut state: Pipeline, _c: NodeContext| {
            let fa = fa.clone();
            async move {
                let n = fa.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    return Err(TinyAgentsError::Model(format!(
                        "connection reset (fetch attempt {})",
                        n + 1
                    )));
                }
                state.fetched = true;
                state.log.push(format!("fetched on attempt {}", n + 1));
                Ok(NodeResult::Update(state))
            }
        })
        .add_node("commit", move |mut state: Pipeline, _c: NodeContext| {
            let og = og.clone();
            async move {
                // Down while the outage flag is 0; healthy once it is raised.
                if og.load(Ordering::SeqCst) == 0 {
                    return Err(TinyAgentsError::Tool(
                        "commit endpoint unreachable (outage)".into(),
                    ));
                }
                state.committed = true;
                state.log.push("committed".into());
                Ok(NodeResult::Update(state))
            }
        })
        .set_entry("fetch")
        .add_edge("fetch", "commit")
        .set_finish("commit")
        .compile()
        .unwrap()
        // 1 try + 3 retries per node, absorbing the fetch blips.
        .with_node_retry(RetryPolicy::default().with_max_attempts(4))
        .with_checkpointer(checkpointer.clone());

    // ---- First run: fetch self-heals, but commit is fully down. ------------
    let thread = "orders-4711";
    let err = graph
        .run_with_thread(thread, Pipeline::default())
        .await
        .expect_err("commit is down, so the first run fails");
    println!("first run failed as expected: {err}");
    println!(
        "  fetch took {} attempts",
        fetch_attempts.load(Ordering::SeqCst)
    );

    // The failure left a resumable checkpoint: `fetch`'s progress is committed
    // and `commit` is scheduled to re-run.
    let snapshot = graph
        .get_state(thread, None)
        .await?
        .expect("a failure-boundary checkpoint exists");
    println!(
        "  durable state: fetched={}, committed={}",
        snapshot.values.fetched, snapshot.values.committed
    );
    println!("  will re-run on resume: {:?}", snapshot.next_nodes);
    println!("  checkpoints on thread: {}", checkpointer.count(thread));

    // ---- The outage clears; restart the run from the checkpoint. -----------
    outage.store(1, Ordering::SeqCst);
    let resumed = graph.retry(thread).await?;
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    println!("\nrestarted to completion:");
    println!(
        "  fetched={}, committed={}",
        resumed.state.fetched, resumed.state.committed
    );
    for line in &resumed.state.log {
        println!("  - {line}");
    }

    Ok(())
}
