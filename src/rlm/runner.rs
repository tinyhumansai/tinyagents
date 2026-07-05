//! The model-driven RLM loop: a driver model writes code cells, the session
//! executes them, and the observations flow back until the script (or the
//! model) produces a final answer.
//!
//! The loop deliberately drives [`ChatModel::invoke`] directly instead of
//! going through `AgentHarness`: the "tool" here is the whole sandboxed
//! interpreter, whose feedback protocol (code fence in, observation out) is
//! the RLM contract rather than a JSON tool call.

use std::sync::Arc;

use serde_json::Value;

use super::host::{RlmHost, RlmHostApi};
use super::session::RlmSession;
use super::templates;
use super::types::{RlmConfig, RlmOutcome, RlmStep, RlmStopReason};
use crate::error::{Result, TinyAgentsError};
use crate::harness::message::Message;
use crate::harness::model::ModelRequest;
use crate::registry::CapabilityRegistry;

/// Extracts the first fenced code block from a model reply.
///
/// Accepts ```` ```<language> ````, a bare ```` ``` ````, or any other fence
/// info string (models occasionally mislabel the language); returns `None`
/// when the reply contains no complete fence.
pub fn extract_code_cell(reply: &str) -> Option<String> {
    let fence_start = reply.find("```")?;
    let after_fence = &reply[fence_start + 3..];
    let newline = after_fence.find('\n')?;
    let body = &after_fence[newline + 1..];
    let fence_end = body.find("```")?;
    let code = body[..fence_end].trim_end();
    (!code.trim().is_empty()).then(|| code.to_string())
}

/// Renders a cell outcome as the observation message fed back to the driver.
fn render_observation(outcome: &super::types::CellOutcome) -> String {
    let mut out = String::new();
    if !outcome.stdout.is_empty() {
        out.push_str("stdout:\n");
        out.push_str(&outcome.stdout);
        if !outcome.stdout.ends_with('\n') {
            out.push('\n');
        }
    }
    if let Some(value) = &outcome.value {
        out.push_str(&format!("value: {value}\n"));
    }
    if let Some(error) = &outcome.error {
        out.push_str(&format!("error: {error}\n"));
    }
    if out.is_empty() {
        out.push_str("(cell produced no output)\n");
    }
    out.push_str("Continue. Reply with the next code cell, or call final_answer(...) when done.");
    out
}

/// The model-driven RLM runner. Construct with [`RlmRunner::from_config`],
/// optionally inject a context, then [`run`](RlmRunner::run) a task.
pub struct RlmRunner<State: Send + Sync + 'static> {
    registry: Arc<CapabilityRegistry<State>>,
    config: RlmConfig,
    session: RlmSession<State>,
    driver_model: String,
    system_prompt: String,
}

impl<State: Send + Sync + 'static> RlmRunner<State> {
    /// Builds a runner from a config document, a capability registry, and the
    /// application state capability calls run against.
    pub fn from_config(
        config: RlmConfig,
        registry: Arc<CapabilityRegistry<State>>,
        state: Arc<State>,
    ) -> Result<Self> {
        let driver_model = config
            .driver_model
            .clone()
            .or_else(|| {
                registry
                    .names(crate::registry::ComponentKind::Model)
                    .into_iter()
                    .next()
            })
            .ok_or_else(|| {
                TinyAgentsError::Validation(
                    "rlm: no driver model configured and no model registered".to_string(),
                )
            })?;
        let sub_model = config
            .sub_model
            .clone()
            .unwrap_or_else(|| driver_model.clone());
        let host = Arc::new(
            RlmHost::new(registry.clone(), state)
                .with_policy(config.policy.clone())
                .with_default_model(sub_model),
        );
        let session = RlmSession::new(&config.interpreter, host)?;

        let template = templates::resolve(&config.template)?;
        let system_prompt = templates::render_system_prompt(
            &template,
            &session.language(),
            &session.usage_guide(),
            &session.host().capabilities(),
            &config.policy,
        );
        Ok(Self {
            registry,
            config,
            session,
            driver_model,
            system_prompt,
        })
    }

    /// The session, for injecting variables or inspecting call counts.
    pub fn session_mut(&mut self) -> &mut RlmSession<State> {
        &mut self.session
    }

    /// The rendered driver system prompt (for inspection/telemetry).
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Injects the task context as the `context` variable in the sandbox.
    pub async fn set_context(&mut self, context: Value) -> Result<()> {
        self.session.set_variable("context", context).await
    }

    /// Runs the loop for one task until a final answer or the cell budget.
    pub async fn run(&mut self, task: impl Into<String>) -> Result<RlmOutcome> {
        let driver = self
            .registry
            .model(&self.driver_model)
            .ok_or_else(|| TinyAgentsError::ModelNotFound(self.driver_model.clone()))?;
        let state = self.session.host().app_state();

        let mut messages = vec![
            Message::system(self.system_prompt.clone()),
            Message::user(task.into()),
        ];
        let mut steps: Vec<RlmStep> = Vec::new();
        let mut driver_calls = 0usize;
        // Set after a reply with no code fence: the driver gets one nudge to
        // produce a cell before its prose is accepted as the answer. Models
        // occasionally emit raw, unfenced code; without the nudge that code
        // would be mistaken for a final answer.
        let mut nudged = false;

        let outcome = loop {
            if steps.len() >= self.config.policy.max_cells {
                break RlmOutcome {
                    answer: None,
                    stop_reason: RlmStopReason::CellBudgetExhausted,
                    steps,
                    driver_calls,
                    sub_llm_calls: 0,
                    tool_calls: 0,
                    agent_calls: 0,
                };
            }

            // `driver_model` is a *registry* name; the resolved ChatModel
            // already knows its provider model id, so the request leaves
            // `model` unset rather than leaking the registry name upstream.
            let request = ModelRequest {
                messages: messages.clone(),
                ..Default::default()
            };
            driver_calls += 1;
            let response = driver.invoke(&state, request).await?;
            let reply_text = Message::Assistant(response.message.clone()).text();
            messages.push(Message::Assistant(response.message));

            let Some(code) = extract_code_cell(&reply_text) else {
                if !nudged {
                    nudged = true;
                    messages.push(Message::user(
                        "Your reply contained no fenced code block, so nothing was executed. \
                         Reply with exactly one fenced code block, or — if you are done — call \
                         final_answer(...) from code. If you truly have nothing to run, repeat \
                         your final answer in plain prose.",
                    ));
                    continue;
                }
                // Two fence-less replies in a row: accept the prose answer.
                break RlmOutcome {
                    answer: Some(reply_text.trim().to_string()),
                    stop_reason: RlmStopReason::ModelAnswered,
                    steps,
                    driver_calls,
                    sub_llm_calls: 0,
                    tool_calls: 0,
                    agent_calls: 0,
                };
            };
            nudged = false;

            let cell = self.session.eval(&code).await?;
            let answered = cell.final_answer.clone();
            let observation = render_observation(&cell);
            steps.push(RlmStep {
                code,
                outcome: cell,
            });

            if let Some(answer) = answered {
                break RlmOutcome {
                    answer: Some(answer),
                    stop_reason: RlmStopReason::Answered,
                    steps,
                    driver_calls,
                    sub_llm_calls: 0,
                    tool_calls: 0,
                    agent_calls: 0,
                };
            }
            messages.push(Message::user(observation));
        };

        let (llm, tool, agent) = self.session.host().call_counts();
        Ok(RlmOutcome {
            sub_llm_calls: llm,
            tool_calls: tool,
            agent_calls: agent,
            ..outcome
        })
    }

    /// Releases interpreter resources.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.session.shutdown().await
    }
}
