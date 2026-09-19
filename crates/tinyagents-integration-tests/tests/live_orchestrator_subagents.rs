//! LIVE end-to-end: a real OpenAI orchestrator designs which sub-agents to run
//! by resolving them **by name** from a [`CapabilityRegistry`], runs them, and
//! composes a final answer.
//!
//! This is the network-backed sibling of `e2e_orchestrator_subagents.rs`. The
//! orchestrator, every specialist sub-agent, and the composer are driven by a
//! real [`OpenAiModel`]. We assert *structurally* — the orchestrator selected at
//! least one registered capability, those capabilities were resolved out of the
//! registry and run, and a non-empty composed answer came back — never on the
//! exact prose.
//!
//! # Skips gracefully
//!
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so `cargo test` passes with no key
//! configured and never dials a real provider by accident.

mod common;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn live_openai_orchestrator_designs_subagents_via_registry() {
    use std::collections::HashMap;
    use std::sync::Arc;

    use futures::future::join_all;
    use serde_json::{Value, json};

    use tinyagents_graph::*;
    use tinyagents_harness::middleware::AgentRun;
    use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
    use tinyagents_harness::subagent::ChildDataPolicy;
    use tinyagents_harness::tool::ToolDispatch;
    use tinyagents_harness::*;
    use tinyagents_language::*;
    use tinyagents_registry::*;
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::{ChatModel, ResponseFormat};
    use tinyinference_llm::providers::openai::OpenAiModel;
    use tinyinference_llm::tool::ToolCall;

    if !common::live::require_live(&["OPENAI_API_KEY"]) {
        return;
    }

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

    let model: Arc<dyn ChatModel<()>> =
        Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present"));

    // Three named specialists, each a SubAgent over the real model.
    let specs = [
        (
            "researcher",
            "Gathers and explains factual background on a topic. No code.",
            "You are a meticulous researcher. Reply with a couple of factual bullet points.",
        ),
        (
            "coder",
            "Writes small, focused code snippets.",
            "You are a senior Rust engineer. Reply with a short, correct code snippet.",
        ),
        (
            "summarizer",
            "Condenses material into a short plain-language summary.",
            "You are an editor. Reply with a crisp 1-2 sentence summary.",
        ),
    ];

    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    let mut dispatches: HashMap<String, Arc<SubAgentTool<()>>> = HashMap::new();
    for (name, description, system_prompt) in specs {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness
            .register_model("model", model.clone())
            .set_default_model("model");
        let subagent =
            SubAgent::new(name, description, Arc::new(harness)).with_system_prompt(system_prompt);
        let dispatch = Arc::new(SubAgentTool::new(
            Arc::new(subagent),
            ChildDataPolicy::new(|parent: &()| *parent),
        ));
        registry
            .register_tool(dispatch.tool())
            .expect("unique specialist name");
        dispatches.insert(name.to_owned(), dispatch);
    }

    // Discover the menu from the registry.
    let available = registry.names(ComponentKind::Tool);
    let menu_text = available
        .iter()
        .map(|name| {
            let desc = registry
                .tool(name)
                .map(|t| t.description().to_owned())
                .unwrap_or_default();
            format!("- {name}: {desc}")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let task = "Summarize what a Rust trait object is for a beginner.";

    // The orchestrator designs the plan via structured output constrained to the
    // registered names.
    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator
        .register_model("model", model.clone())
        .set_default_model("model")
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema(
                "agent_selection",
                json!({
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "items": { "type": "string", "enum": available }
                        }
                    },
                    "required": ["agents"],
                    "additionalProperties": false
                }),
            )),
            ..RunPolicy::default()
        });

    let plan = orchestrator
        .invoke_default(
            &(),
            vec![
                Message::system(format!(
                    "You are an orchestrator with these named sub-agents available:\n{menu_text}\n\n\
                     Choose the minimal subset whose skills solve the task. Respond ONLY with the \
                     requested JSON object."
                )),
                Message::user(task),
            ],
        )
        .await
        .expect("orchestrator run succeeds");

    let mut chosen = parse_selection(&plan);
    chosen.retain(|name| registry.has(ComponentKind::Tool, name));
    assert!(
        !chosen.is_empty(),
        "the orchestrator selected at least one registered sub-agent (got {chosen:?})"
    );

    // Resolve each chosen name from the registry and run them in parallel.
    let dispatches = chosen.iter().enumerate().map(|(i, name)| {
        let name = name.clone();
        let dispatch = dispatches
            .get(&name)
            .cloned()
            .expect("a chosen name resolves in the registry");
        async move {
            let parent = tinyagents_harness::context::RunContext::new(
                tinyagents_harness::context::RunConfig::new(format!("dispatch-{i}")),
                (),
            );
            let result = dispatch
                .invoke_in_parent_context(
                    &(),
                    json!({ "input": task }),
                    tinytools::ToolCallOptions::default(),
                    &parent,
                )
                .await
                .expect("sub-agent run succeeds");
            (name, result.output())
        }
    });
    let outputs: Vec<(String, String)> = join_all(dispatches).await;

    assert!(
        outputs.iter().any(|(_, text)| !text.trim().is_empty()),
        "at least one resolved sub-agent returned non-empty text"
    );

    // Compose the resolved sub-agents' outputs into one final answer.
    let context = outputs
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
                    "Combine the labeled sub-agent outputs into one coherent answer. \
                     Do not mention the sub-agents.",
                ),
                Message::user(format!("Task: {task}\n\nOutputs:\n{context}")),
            ],
        )
        .await
        .expect("composer run succeeds");

    let final_text = composed.text().unwrap_or_default();
    assert!(
        !final_text.trim().is_empty(),
        "the composed answer is non-empty"
    );

    eprintln!("orchestrator chose {chosen:?}; composed answer: {final_text}");
}
