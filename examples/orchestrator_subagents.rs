//! Flagship registry showcase: an orchestrator that *designs* which sub-agents
//! to call at runtime by resolving them **by name** from a
//! [`CapabilityRegistry`], then runs them and composes their results.
//!
//! The pattern this example demonstrates is the whole point of the named
//! capability registry:
//!
//! 1. **Register named capabilities.** Several specialized sub-agents
//!    (`researcher`, `coder`, `summarizer`), each a [`SubAgent`] over an
//!    [`OpenAiModel`] with a distinct system prompt, are wrapped as
//!    [`SubAgentTool`]s and registered in a [`CapabilityRegistry`] under their
//!    names.
//! 2. **Discover.** The set of available capabilities (names + descriptions) is
//!    read back out of the registry — nothing is hard-coded into the planner.
//! 3. **Design.** An *orchestrator* agent is given the task plus the discovered
//!    menu and decides, via structured output, **which** registered sub-agents
//!    to invoke. It returns a list of names.
//! 4. **Bind at runtime.** Each chosen name is resolved from the registry with
//!    [`CapabilityRegistry::tool`], run (in parallel via `join_all`), and the
//!    results are composed into a final answer.
//!
//! Capabilities are therefore *named, discovered, and bound at runtime* — the
//! orchestrator never holds a direct handle to any sub-agent; it only knows
//! their names and looks them up in the registry when it decides to use them.
//!
//! Run with:
//!
//! ```text
//! cargo run --features openai --example orchestrator_subagents
//! ```

use std::sync::Arc;

use futures::future::join_all;
use serde_json::{Value, json};

use tinyagents::harness::message::Message;
use tinyagents::harness::middleware::AgentRun;
use tinyagents::harness::model::{ChatModel, ResponseFormat};
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
use tinyagents::harness::tool::ToolCall;
use tinyagents::{CapabilityRegistry, ComponentKind, Result, SubAgent, SubAgentTool};

/// A specialized sub-agent's static identity.
struct AgentSpec {
    name: &'static str,
    description: &'static str,
    system_prompt: &'static str,
}

/// The three specialist sub-agents this orchestrator can choose from.
const SPECIALISTS: &[AgentSpec] = &[
    AgentSpec {
        name: "researcher",
        description: "Gathers and explains factual background on a topic.",
        system_prompt: "You are a meticulous researcher. Answer with a few concise, factual \
                        bullet points of background relevant to the user's task. No code.",
    },
    AgentSpec {
        name: "coder",
        description: "Writes small, focused code snippets and explains them.",
        system_prompt: "You are a senior Rust engineer. When asked, produce a short, correct \
                        code snippet with a one-line explanation. Keep it minimal.",
    },
    AgentSpec {
        name: "summarizer",
        description: "Condenses material into a short, plain-language summary.",
        system_prompt: "You are an editor. Produce a crisp, plain-language summary in 2-3 \
                        sentences. No jargon.",
    },
];

/// Builds a [`SubAgentTool`] wrapping a [`SubAgent`] over the shared OpenAI
/// model with the spec's distinct system prompt.
fn build_specialist(spec: &AgentSpec, model: Arc<dyn ChatModel<()>>) -> SubAgentTool<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("model", model)
        .set_default_model("model");
    let subagent = SubAgent::new(spec.name, spec.description, Arc::new(harness))
        .with_system_prompt(spec.system_prompt);
    SubAgentTool::new(Arc::new(subagent))
}

/// Reads the `{ "agents": [..] }` selection out of an [`AgentRun`], preferring
/// the extracted structured output and falling back to parsing the raw text.
fn parse_selection(run: &AgentRun) -> Vec<String> {
    let value: Value = run
        .structured
        .clone()
        .or_else(|| run.text().and_then(|t| serde_json::from_str(&t).ok()))
        .unwrap_or(Value::Null);
    value
        .get("agents")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let model: Arc<dyn ChatModel<()>> = Arc::new(OpenAiModel::from_env()?);
    println!("=== Orchestrator designs sub-agents via the registry ===");

    // 1. Register every specialist sub-agent by name in the capability registry.
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    for spec in SPECIALISTS {
        registry.register_tool(Arc::new(build_specialist(spec, model.clone())))?;
    }

    // 2. Discover the available capabilities straight out of the registry.
    let menu: Vec<(String, String)> = registry
        .names(ComponentKind::Tool)
        .into_iter()
        .map(|name| {
            let desc = registry
                .tool(&name)
                .map(|t| t.description().to_owned())
                .unwrap_or_default();
            (name, desc)
        })
        .collect();
    let menu_text = menu
        .iter()
        .map(|(name, desc)| format!("- {name}: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");
    let available: Vec<String> = menu.iter().map(|(n, _)| n.clone()).collect();
    println!("registered capabilities:\n{menu_text}\n");

    // 3. Design: the orchestrator picks which named sub-agents to use, returning
    //    a structured `{ "agents": [..] }` constrained to the registered names.
    let task = "Explain what a Rust trait object is and give a tiny code example, \
                then summarize it for a beginner.";
    println!("task: {task}\n");

    let selection_schema = ResponseFormat::json_schema(
        "agent_selection",
        json!({
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": { "type": "string", "enum": available },
                    "description": "Names of the sub-agents to invoke for this task."
                }
            },
            "required": ["agents"],
            "additionalProperties": false
        }),
    );

    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator
        .register_model("model", model.clone())
        .set_default_model("model")
        .with_policy(RunPolicy {
            default_response_format: Some(selection_schema),
            ..RunPolicy::default()
        });

    let plan = orchestrator
        .invoke_default(
            &(),
            vec![
                Message::system(format!(
                    "You are an orchestrator. You have these named sub-agents available:\n\
                     {menu_text}\n\n\
                     Choose the minimal subset whose combined skills solve the user's task. \
                     Respond ONLY with the requested JSON object listing the sub-agent names."
                )),
                Message::user(task),
            ],
        )
        .await?;

    let mut chosen = parse_selection(&plan);
    chosen.retain(|name| registry.has(ComponentKind::Tool, name));
    if chosen.is_empty() {
        // Defensive fallback so the showcase always does *something* useful.
        chosen = available.clone();
    }
    println!("orchestrator chose: {chosen:?}\n");

    // 4. Bind at runtime: resolve each chosen name from the registry and run the
    //    resolved sub-agents in parallel.
    let runs = chosen.iter().enumerate().map(|(i, name)| {
        let tool = registry
            .tool(name)
            .expect("chosen name resolves in the registry");
        let call = ToolCall::new(format!("c{i}"), name.clone(), json!({ "input": task }));
        async move {
            let result = tool.call(&(), call).await?;
            Ok::<(String, String), tinyagents::TinyAgentsError>((
                tool.name().to_owned(),
                result.content,
            ))
        }
    });
    let outputs: Vec<(String, String)> = join_all(runs).await.into_iter().collect::<Result<_>>()?;

    for (name, text) in &outputs {
        println!("── {name} ──\n{text}\n");
    }

    // 5. Compose: synthesize the sub-agent outputs into one final answer.
    let composed_context = outputs
        .iter()
        .map(|(name, text)| format!("[{name}]\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut composer: AgentHarness<()> = AgentHarness::new();
    composer
        .register_model("model", model)
        .set_default_model("model");
    let composed = composer
        .invoke_default(
            &(),
            vec![
                Message::system(
                    "Combine the labeled sub-agent outputs below into one coherent answer to \
                     the user's task. Preserve any code. Do not mention the sub-agents.",
                ),
                Message::user(format!(
                    "Task: {task}\n\nSub-agent outputs:\n{composed_context}"
                )),
            ],
        )
        .await?;

    println!("=== composed answer ===");
    println!("{}", composed.text().unwrap_or_default());

    Ok(())
}
