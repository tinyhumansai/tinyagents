//! A live RLM run over an **external Python interpreter**, with a real tool
//! and a real sub-agent registered.
//!
//! The driver model writes Python cells; the child `python3` process executes
//! them, and every capability call (`llm`, `tool`, `agent`, `final_answer`)
//! travels back over the wire protocol to the host, which enforces the policy
//! and lowers to the harness runtime. The `agent("summarizer", ...)` call
//! drives a *complete nested agent run* on its own OpenAI-backed harness —
//! scripts calling agents calling models.
//!
//! Run with (needs `OPENAI_API_KEY` and `python3` on PATH):
//!
//! ```text
//! cargo run --features rlm --example rlm_python
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};
use tinyagents::registry::CapabilityRegistry;
use tinyagents::rlm::{RlmConfig, RlmRunner};
use tinyagents::{HarnessSubAgent, Result, SubAgent};

/// A deterministic metrics tool the script can query.
struct MetricsTool;

#[async_trait]
impl Tool<()> for MetricsTool {
    fn name(&self) -> &str {
        "service_metrics"
    }

    fn description(&self) -> &str {
        "Returns latency and error-rate metrics for a named service."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "service_metrics",
            "Returns latency and error-rate metrics for a named service.",
            json!({
                "type": "object",
                "properties": { "service": { "type": "string" } },
                "required": ["service"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        let service = call
            .arguments
            .get("service")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let metrics = match service {
            "checkout" => json!({"p99_ms": 2140, "error_rate": 0.031, "deploys_today": 3}),
            "search" => json!({"p99_ms": 180, "error_rate": 0.002, "deploys_today": 0}),
            "auth" => json!({"p99_ms": 95, "error_rate": 0.001, "deploys_today": 1}),
            other => json!({"error": format!("unknown service `{other}`")}),
        };
        let mut result = ToolResult::text(call.id, call.name, metrics.to_string());
        result.raw = Some(metrics);
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry.register_model("openai", Arc::new(OpenAiModel::from_env()?))?;
    registry.register_tool(Arc::new(MetricsTool))?;

    // A real sub-agent on its own harness: scripts delegate to it by name.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model("openai", Arc::new(OpenAiModel::from_env()?))
        .set_default_model("openai");
    let summarizer = Arc::new(
        SubAgent::new(
            "summarizer",
            "Writes a crisp two-sentence incident summary from raw findings.",
            Arc::new(child),
        )
        .with_system_prompt(
            "You summarize incident findings for executives: two sentences, plain language, \
             lead with impact.",
        ),
    );
    registry.register_agent(Arc::new(HarnessSubAgent::new(summarizer)))?;

    let config = RlmConfig::from_json(
        r#"{
            "interpreter": {"kind": "python"},
            "driver_model": "openai",
            "template": "general",
            "policy": {"max_cells": 8, "cell_timeout": 90000}
        }"#,
    )?;

    let mut runner = RlmRunner::from_config(config, Arc::new(registry), Arc::new(()))?;
    let outcome = runner
        .run(
            "Check the service_metrics tool for the services checkout, search, and auth. \
             Identify which service looks unhealthy and why. Then delegate to the `summarizer` \
             agent to produce an executive summary of your findings, and return that summary \
             as the final answer.",
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
    runner.shutdown().await?;
    Ok(())
}
