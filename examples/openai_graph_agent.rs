//! A state graph whose node drives a real OpenAI-backed agent harness.
//!
//! Wraps an [`AgentHarness`] in an [`Arc`] and captures it in a graph node
//! closure. The graph runs START -> `agent` -> END: the `agent` node calls the
//! harness (which talks to OpenAI), stores the answer in the graph state, and
//! ends the run. This shows how the legacy [`StateGraph`] composes with a real
//! model behind a harness.
//!
//! Run with:
//!
//! ```text
//! cargo run --features openai --example openai_graph_agent
//! ```

use std::sync::Arc;

use tinyagents::harness::message::Message;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::{Node, NodeOutput, Result, StateGraph};

/// State threaded through the graph: the question to ask and the answer the
/// agent node fills in.
#[derive(Clone, Debug)]
struct ChatState {
    question: String,
    answer: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let model = OpenAiModel::from_env()?;
    println!("=== OpenAI-backed graph agent ===");
    println!("model: {}", model.model());

    // Build the harness once and share it into the node via an Arc.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("openai", Arc::new(model))
        .set_default_model("openai");
    let harness = Arc::new(harness);

    let graph = StateGraph::new()
        .add_node(Node::new("agent", move |mut state: ChatState| {
            let harness = harness.clone();
            async move {
                let run = harness
                    .invoke_default(&(), vec![Message::user(state.question.clone())])
                    .await?;
                state.answer = run.text();
                Ok(NodeOutput::end(state))
            }
        }))
        .set_start("agent")
        .add_end("agent");

    let question = "Name three popular Rust web frameworks, comma-separated.";
    println!("question: {question}\n");

    let run = graph
        .run(ChatState {
            question: question.to_string(),
            answer: None,
        })
        .await?;

    println!("visited: {:?}", run.visited);
    println!(
        "answer : {}",
        run.state.answer.unwrap_or_else(|| "<none>".to_string())
    );

    Ok(())
}
