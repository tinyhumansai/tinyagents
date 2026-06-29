//! TRUE end-to-end: a durable [`CompiledGraph`] with reducer-merged partial
//! updates, an [`InMemoryCheckpointer`], and a human-in-the-loop
//! interrupt/resume round trip spanning a thread.
//!
//! This composes the **durable graph builder/executor**, the **reducer**
//! (partial-update merge), the **checkpointer** (durability), and the
//! **command/interrupt** model. An early node does work, a middle node
//! interrupts for approval, and a later node performs more work after resume.
//! We assert the run pauses, a checkpoint is persisted, and the resumed final
//! state reflects updates from *both* sides of the interrupt.

use std::sync::Arc;

use serde_json::json;

use rustagents::graph::ClosureStateReducer;
use rustagents::{
    Checkpointer, Command, GraphBuilder, InMemoryCheckpointer, Interrupt, NodeContext, NodeResult,
};

/// Audit log of the actions applied through the reducer. Each node contributes
/// a partial `String` update merged onto the running list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Workflow {
    actions: Vec<String>,
}

#[tokio::test]
async fn durable_graph_interrupts_then_resumes_with_full_state() {
    let checkpointer = Arc::new(InMemoryCheckpointer::<Workflow>::new());

    let graph = GraphBuilder::<Workflow, String>::new()
        .set_reducer(ClosureStateReducer::new(
            |mut state: Workflow, update: String| {
                state.actions.push(update);
                Ok(state)
            },
        ))
        // Early node: real pre-interrupt work.
        .add_node("prep", |_s: Workflow, _c: NodeContext| async move {
            Ok(NodeResult::Update("prep".to_string()))
        })
        // Middle node: pause for human approval, then apply the approval.
        .add_node("gate", |_s: Workflow, ctx: NodeContext| async move {
            match ctx.resume {
                Some(value) => {
                    let who = value
                        .get("approved_by")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    Ok(NodeResult::Update(format!("approved:{who}")))
                }
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    "gate",
                    json!({ "ask": "approve this workflow?" }),
                ))),
            }
        })
        // Later node: post-interrupt work.
        .add_node("finalize", |_s: Workflow, _c: NodeContext| async move {
            Ok(NodeResult::Update("finalize".to_string()))
        })
        .set_entry("prep")
        .add_edge("prep", "gate")
        .add_edge("gate", "finalize")
        .set_finish("finalize")
        .compile()
        .expect("graph compiles")
        .with_checkpointer(checkpointer.clone());

    // First pass: prep runs, gate interrupts before finalize.
    let paused = graph
        .run_with_thread("wf-1", Workflow::default())
        .await
        .expect("first pass succeeds");

    assert!(paused.is_interrupted(), "the run should pause at the gate");
    assert_eq!(paused.interrupts.len(), 1);
    assert_eq!(paused.interrupts[0].node.as_str(), "gate");
    // Pre-interrupt work is already reflected in the paused state.
    assert_eq!(paused.state.actions, vec!["prep".to_string()]);

    // Durability: at least one checkpoint was persisted for the thread.
    let checkpoints = checkpointer.list("wf-1").await.expect("list succeeds");
    assert!(
        !checkpoints.is_empty(),
        "an interrupt must persist a checkpoint"
    );

    // Resume: inject the approval, gate completes, finalize runs.
    let resumed = graph
        .resume("wf-1", Command::resume(json!({ "approved_by": "ada" })))
        .await
        .expect("resume succeeds");

    assert!(!resumed.is_interrupted(), "resumed run must finish");
    // Final state reflects BOTH pre- and post-interrupt updates.
    assert_eq!(
        resumed.state.actions,
        vec![
            "prep".to_string(),
            "approved:ada".to_string(),
            "finalize".to_string(),
        ]
    );
    assert!(resumed.steps >= 1, "the resumed pass advanced at least once");
}
