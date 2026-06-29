//! One-shot chat against the real OpenAI Chat Completions API.
//!
//! Registers an [`OpenAiModel`] in an [`AgentHarness`] and runs a single
//! question through the default agent loop, then prints the answer and token
//! usage. This is the simplest happy-path integration.
//!
//! Run with (after copying `.env.example` to `.env` and setting your key):
//!
//! ```text
//! cargo run --features openai --example openai_chat
//! ```

use std::sync::Arc;

use tinyagents::Result;
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;

#[tokio::main]
async fn main() -> Result<()> {
    // Load OPENAI_API_KEY (and optional OPENAI_MODEL / OPENAI_BASE_URL) from a
    // local `.env` file if present; ignore the error when there is none.
    dotenvy::dotenv().ok();

    let model = OpenAiModel::from_env()?;
    println!("=== OpenAI chat ===");
    println!("model: {}", model.model());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("openai", Arc::new(model))
        .set_default_model("openai");

    let question = "In one sentence, what is a Rust trait?";
    println!("question: {question}\n");

    let run = harness
        .invoke_default(&(), vec![Message::user(question)])
        .await?;

    println!("answer: {}", run.text().unwrap_or_default());
    println!(
        "\nusage : {} input + {} output = {} total tokens ({} cached) over {} model call(s)",
        run.usage.usage.input_tokens,
        run.usage.usage.output_tokens,
        run.usage.usage.total_tokens,
        run.usage.usage.cache_read_tokens,
        run.usage.calls,
    );

    Ok(())
}
