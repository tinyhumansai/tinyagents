//! LIVE end-to-end: a durable [`CompiledGraph`] whose single node drives a real
//! OpenAI [`AgentHarness`] — but only *after* a human-in-the-loop interrupt.
//!
//! This composes the **durable graph runtime** (builder/executor + reducer), the
//! **checkpointer** (durability), the **command/interrupt** model, and a real
//! network-backed **harness** run. The flow deliberately straddles the model
//! call with a checkpoint boundary:
//!
//! 1. First pass: the node sees no `resume` value and immediately interrupts
//!    *before* making any model call. The runtime persists a checkpoint.
//! 2. Resume: a [`Command::resume`] re-enters the node, which now performs the
//!    real OpenAI model call and folds the answer into state.
//!
//! We assert *structurally* — the run pauses, a checkpoint persists, the resumed
//! run completes, and a model call actually happened — never on LLM prose.
//!
//! # Skips gracefully
//!
//! The OpenAI-using code is gated behind `#[cfg(feature = "openai")]`, and the
//! test returns early (after an `eprintln!`) when `OPENAI_API_KEY` is unset, so
//! `cargo test --features openai` passes with no key configured. Without the
//! `openai` feature the test is a trivial no-op.

#[cfg(feature = "openai")]
#[tokio::test]
async fn live_durable_graph_checkpoints_then_resumes_across_model_call() {
    use std::sync::Arc;

    use serde_json::json;

    use rustagents::harness::message::Message;
    use rustagents::harness::providers::openai::OpenAiModel;
    use rustagents::harness::runtime::AgentHarness;
    use rustagents::{
        Checkpointer, Command, GraphBuilder, InMemoryCheckpointer, Interrupt, NodeContext,
        NodeResult,
    };

    // Load .env so `cargo test` picks up local credentials.
    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!(
            "skipping live_durable_graph_checkpoints_then_resumes_across_model_call: \
             OPENAI_API_KEY is not set"
        );
        return;
    }

    /// State threaded through the durable graph: the question to ask, plus the
    /// answer + model-call count once the node has driven the real harness.
    #[derive(Clone, Debug, Default)]
    struct QaState {
        question: String,
        answer: Option<String>,
        model_calls: usize,
    }

    // A real OpenAI-backed harness; the graph node owns a shared handle to it.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");
    let harness = Arc::new(harness);

    let checkpointer = Arc::new(InMemoryCheckpointer::<QaState>::new());

    let node_harness = harness.clone();
    let graph = GraphBuilder::<QaState, QaState>::overwrite()
        .with_graph_id("live-checkpoint-resume")
        .add_node("ask", move |mut state: QaState, ctx: NodeContext| {
            let harness = node_harness.clone();
            async move {
                match ctx.resume {
                    // First pass: pause *before* the model call so the checkpoint
                    // boundary lands squarely in front of the network request.
                    None => Ok(NodeResult::Interrupt(Interrupt::new(
                        "ask",
                        json!({ "ask": "approve calling the model?" }),
                    ))),
                    // Resume pass: perform the real OpenAI call and fold the
                    // result into state.
                    Some(_) => {
                        let run = harness
                            .invoke_default(&(), vec![Message::user(state.question.clone())])
                            .await?;
                        state.answer = run.text();
                        state.model_calls = run.model_calls;
                        Ok(NodeResult::Update(state))
                    }
                }
            }
        })
        .set_entry("ask")
        .set_finish("ask")
        .compile()
        .expect("durable graph compiles")
        .with_checkpointer(checkpointer.clone());

    // First pass: the node interrupts before touching the network.
    let paused = graph
        .run_with_thread(
            "ckpt-thread",
            QaState {
                question: "Reply with a single short greeting.".to_string(),
                answer: None,
                model_calls: 0,
            },
        )
        .await
        .expect("first pass succeeds");

    assert!(
        paused.is_interrupted(),
        "the run should pause before the model call"
    );
    assert_eq!(paused.interrupts.len(), 1);
    assert_eq!(paused.interrupts[0].node.as_str(), "ask");
    // No model call has happened yet on the paused side of the interrupt.
    assert!(
        paused.state.answer.is_none(),
        "no answer should exist before resume"
    );

    // Durability: the interrupt persisted at least one checkpoint for the thread.
    let checkpoints = checkpointer
        .list("ckpt-thread")
        .await
        .expect("list succeeds");
    assert!(
        !checkpoints.is_empty(),
        "an interrupt must persist a checkpoint"
    );

    // Resume: re-enter the node, which now makes the real OpenAI model call.
    let resumed = graph
        .resume("ckpt-thread", Command::resume(json!({ "approved": true })))
        .await
        .expect("resume succeeds");

    // Structural assertions only — never assert on exact LLM prose.
    assert!(!resumed.is_interrupted(), "resumed run must complete");
    assert!(
        resumed.state.model_calls >= 1,
        "the resumed pass should make at least one model call"
    );
    let answer = resumed.state.answer.unwrap_or_default();
    assert!(
        !answer.trim().is_empty(),
        "the resumed run should fold a non-empty model answer into state"
    );

    // The resumed run committed at least as many checkpoints as the paused one.
    let after_resume = checkpointer
        .list("ckpt-thread")
        .await
        .expect("list succeeds");
    assert!(
        after_resume.len() >= checkpoints.len(),
        "resuming should not drop persisted checkpoints"
    );
}
