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
//! The test returns early (after an `eprintln!`) when `OPENAI_API_KEY` is
//! unset, so `cargo test` passes with no key configured.

#[tokio::test]
async fn live_openai_orchestrator_designs_subagents_via_registry() {
    use std::sync::Arc;

    use futures::future::join_all;
    use serde_json::{Value, json};

    use tinyagents::harness::message::Message;
    use tinyagents::harness::middleware::AgentRun;
    use tinyagents::harness::model::{ChatModel, ResponseFormat};
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::{AgentHarness, RunPolicy};
    use tinyagents::harness::tool::ToolCall;
    use tinyagents::{CapabilityRegistry, ComponentKind, SubAgent, SubAgentTool};

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!(
            "skipping live_openai_orchestrator_designs_subagents_via_registry: \
             OPENAI_API_KEY is not set"
        );
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
    for (name, description, system_prompt) in specs {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness
            .register_model("model", model.clone())
            .set_default_model("model");
        let subagent =
            SubAgent::new(name, description, Arc::new(harness)).with_system_prompt(system_prompt);
        registry
            .register_tool(Arc::new(SubAgentTool::new(Arc::new(subagent))))
            .expect("unique specialist name");
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
        let tool = registry
            .tool(&name)
            .expect("a chosen name resolves in the registry");
        let call = ToolCall::new(format!("c{i}"), name.clone(), json!({ "input": task }));
        async move {
            let result = tool.call(&(), call).await.expect("sub-agent run succeeds");
            (name, result.content)
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
