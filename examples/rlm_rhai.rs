//! A live recursive-language-model run over the embedded Rhai sandbox.
//!
//! The classic RLM shape: a context too noisy to eyeball is injected into the
//! sandbox as the `context` variable, and the driver model must *probe it
//! with code* — slicing, filtering, and delegating fuzzy judgment on
//! individual entries to sub-LLM calls (`llm(...)`) — before answering with
//! `final_answer(...)`.
//!
//! Run with (needs `OPENAI_API_KEY`, optional `OPENAI_MODEL`):
//!
//! ```text
//! cargo run --features rlm --example rlm_rhai
//! ```

use std::sync::Arc;

use serde_json::json;

use tinyagents::Result;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{RlmConfig, RlmRunner};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let model = OpenAiModel::from_env()?;

    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry.register_model("openai", Arc::new(model))?;

    // The whole run is one JSON document — the same document an external
    // harness could load from disk.
    let config = RlmConfig::from_json(
        r#"{
            "interpreter": {"kind": "rhai"},
            "driver_model": "openai",
            "template": "context-explorer",
            "policy": {"max_cells": 8, "max_llm_calls": 16, "cell_timeout": 90000}
        }"#,
    )?;

    let mut runner = RlmRunner::from_config(config, Arc::new(registry), Arc::new(()))?;

    // A synthetic "too big to read" context: expense records with three
    // hidden anomalies among routine noise.
    let mut records = Vec::new();
    for i in 0..200usize {
        let team = ["platform", "growth", "ops"][i % 3];
        let amount = 40 + (i * 7) % 60;
        records.push(json!({
            "id": i,
            "team": team,
            "amount_usd": amount,
            "memo": format!("routine cloud spend, invoice {i}"),
        }));
    }
    records[57] = json!({"id": 57, "team": "growth", "amount_usd": 18400,
        "memo": "annual conference sponsorship paid twice, needs review"});
    records[121] = json!({"id": 121, "team": "ops", "amount_usd": 9750,
        "memo": "emergency hardware replacement after flood damage"});
    records[188] = json!({"id": 188, "team": "platform", "amount_usd": 12300,
        "memo": "contractor invoice with mismatched PO number"});
    runner.set_context(json!(records)).await?;

    println!("── system prompt ──\n{}\n", runner.system_prompt());

    let outcome = runner
        .run(
            "The `context` variable holds 200 expense records. Find every anomalous record \
             (unusual amount or memo), and summarize each anomaly in one line.",
        )
        .await?;

    for (i, step) in outcome.steps.iter().enumerate() {
        println!("── cell {} ──\n{}\n", i + 1, step.code);
        if !step.outcome.stdout.is_empty() {
            println!("stdout:\n{}", step.outcome.stdout);
        }
        if let Some(error) = &step.outcome.error {
            println!("error: {error}");
        }
    }
    println!("── answer ({:?}) ──", outcome.stop_reason);
    println!("{}", outcome.answer.as_deref().unwrap_or("(none)"));
    println!(
        "\ndriver calls: {}, sub-llm calls: {}, tool calls: {}, agent calls: {}",
        outcome.driver_calls, outcome.sub_llm_calls, outcome.tool_calls, outcome.agent_calls
    );
    Ok(())
}
