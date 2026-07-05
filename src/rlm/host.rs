//! The host side of the RLM capability boundary.
//!
//! Every interpreter backend — embedded or external — funnels script
//! capability calls through one object-safe seam, [`RlmHostApi::handle`].
//! [`RlmHost`] is the standard implementation: it resolves names through the
//! session's [`CapabilityRegistry`], enforces the [`RlmPolicy`] call and
//! recursion limits fail-closed, records an [`RlmCallRecord`] per call, and
//! lowers to the real harness runtime (`ChatModel::invoke`, `Tool::call`,
//! `HarnessAgent::run`).
//!
//! ## Fatal vs script-visible errors
//!
//! A capability call can fail two ways, and the distinction is the sandbox
//! contract:
//!
//! - **Script-visible** failures (unknown tool, tool returned an error, model
//!   provider error) surface *inside* the script as a raised
//!   exception/runtime error. The driving model observes them in the cell
//!   outcome and may adapt — that feedback loop is the whole point of an RLM.
//! - **Fatal** failures ([`TinyAgentsError::LimitExceeded`],
//!   [`Timeout`](TinyAgentsError::Timeout),
//!   [`Cancelled`](TinyAgentsError::Cancelled),
//!   [`SubAgentDepth`](TinyAgentsError::SubAgentDepth)) mean a policy bound
//!   tripped; the cell is aborted (and an external interpreter's child
//!   process killed) rather than letting the script observe and route around
//!   its own resource limits.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};

use super::types::{HostCall, RlmCallRecord, RlmCancelFlag, RlmPolicy};
use crate::error::{Result, TinyAgentsError};
use crate::graph::subagent_node::SubAgentInput;
use crate::harness::events::EventSink;
use crate::harness::message::Message;
use crate::harness::model::ModelRequest;
use crate::harness::tool::ToolCall;
use crate::registry::{CapabilityRegistry, ComponentKind};

/// Returns whether a capability error must abort the cell (policy bound
/// tripped) instead of surfacing inside the script.
pub fn is_fatal(err: &TinyAgentsError) -> bool {
    matches!(
        err,
        TinyAgentsError::LimitExceeded(_)
            | TinyAgentsError::Timeout(_)
            | TinyAgentsError::Cancelled
            | TinyAgentsError::SubAgentDepth(_)
    )
}

/// A snapshot of the registered capability names a session may reach,
/// rendered into the driver prompt so the model knows what it can call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityListing {
    /// Registered model names.
    pub models: Vec<String>,
    /// Registered tool names (with descriptions when available).
    pub tools: Vec<(String, String)>,
    /// Registered agent names.
    pub agents: Vec<String>,
}

/// The object-safe host seam every interpreter backend calls through.
///
/// Implementations must be cheap to share (`Arc<dyn RlmHostApi>`): the
/// embedded Rhai engine clones the handle into each capability closure, and
/// the external-interpreter driver holds it across protocol frames.
#[async_trait]
pub trait RlmHostApi: Send + Sync {
    /// Executes one capability call on behalf of a script.
    ///
    /// A `Ok(value)` is handed back to the script; an `Err` is surfaced into
    /// the script when recoverable (see [`is_fatal`]) or aborts the cell when
    /// fatal.
    async fn handle(&self, call: HostCall) -> Result<Value>;

    /// The capability names available to scripts (for prompt rendering).
    fn capabilities(&self) -> CapabilityListing;

    /// The wall-clock deadline of the cell currently being evaluated.
    fn deadline(&self) -> Option<Instant>;

    /// The session cancellation flag.
    fn cancel_flag(&self) -> RlmCancelFlag;
}

/// Session-cumulative counters enforced against [`RlmPolicy`].
#[derive(Debug, Default, Clone, Copy)]
struct CallCounters {
    llm: usize,
    tool: usize,
    agent: usize,
}

/// Per-cell mutable buffers the session arms before evaluating a cell and
/// drains after.
#[derive(Debug, Default)]
struct CellBuffers {
    deadline: Option<Instant>,
    calls: Vec<RlmCallRecord>,
    final_answer: Option<String>,
}

/// The standard capability host, bound to a [`CapabilityRegistry`].
pub struct RlmHost<State: Send + Sync> {
    registry: Arc<CapabilityRegistry<State>>,
    state: Arc<State>,
    policy: RlmPolicy,
    /// The model `llm(...)` reaches when the script names none.
    default_model: Option<String>,
    /// The session's run depth — the parent depth for sub-agent runs.
    run_depth: usize,
    events: EventSink,
    cancel: RlmCancelFlag,
    counters: Mutex<CallCounters>,
    cell: Mutex<CellBuffers>,
}

impl<State: Send + Sync + 'static> RlmHost<State> {
    /// Builds a host over a capability registry and application state.
    pub fn new(registry: Arc<CapabilityRegistry<State>>, state: Arc<State>) -> Self {
        Self {
            registry,
            state,
            policy: RlmPolicy::default(),
            default_model: None,
            run_depth: 0,
            events: EventSink::default(),
            cancel: RlmCancelFlag::new(),
            counters: Mutex::new(CallCounters::default()),
            cell: Mutex::new(CellBuffers::default()),
        }
    }

    /// Sets the session policy.
    pub fn with_policy(mut self, policy: RlmPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the default model `llm(...)` reaches when the script names none.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Sets the run depth sub-agent calls recurse below.
    pub fn with_run_depth(mut self, depth: usize) -> Self {
        self.run_depth = depth;
        self
    }

    /// Installs an event sink child agent runs fan onto.
    pub fn with_events(mut self, events: EventSink) -> Self {
        self.events = events;
        self
    }

    /// Installs an external cancellation flag.
    pub fn with_cancel_flag(mut self, cancel: RlmCancelFlag) -> Self {
        self.cancel = cancel;
        self
    }

    /// The session policy.
    pub fn policy(&self) -> &RlmPolicy {
        &self.policy
    }

    /// The application state capability calls run against.
    pub fn app_state(&self) -> Arc<State> {
        self.state.clone()
    }

    /// Arms the per-cell buffers before a cell is evaluated.
    pub(super) fn begin_cell(&self) {
        let mut cell = self.cell.lock().expect("cell buffers poisoned");
        cell.deadline = self.policy.cell_timeout.map(|t| Instant::now() + t);
        cell.calls.clear();
        cell.final_answer = None;
    }

    /// Drains the per-cell buffers after a cell finished: the recorded calls
    /// and the final answer, if `final_answer(...)` was called.
    pub(super) fn end_cell(&self) -> (Vec<RlmCallRecord>, Option<String>) {
        let mut cell = self.cell.lock().expect("cell buffers poisoned");
        cell.deadline = None;
        (std::mem::take(&mut cell.calls), cell.final_answer.take())
    }

    fn record(&self, record: RlmCallRecord) {
        self.cell
            .lock()
            .expect("cell buffers poisoned")
            .calls
            .push(record);
    }

    fn bump(&self, call: &HostCall) -> Result<()> {
        let mut counters = self.counters.lock().expect("counters poisoned");
        match call {
            HostCall::Llm { .. } => {
                if counters.llm >= self.policy.max_llm_calls {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "llm call limit ({}) exceeded",
                        self.policy.max_llm_calls
                    )));
                }
                counters.llm += 1;
            }
            HostCall::Tool { .. } => {
                if counters.tool >= self.policy.max_tool_calls {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "tool call limit ({}) exceeded",
                        self.policy.max_tool_calls
                    )));
                }
                counters.tool += 1;
            }
            HostCall::Agent { .. } => {
                if counters.agent >= self.policy.max_agent_calls {
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "agent call limit ({}) exceeded",
                        self.policy.max_agent_calls
                    )));
                }
                counters.agent += 1;
            }
            HostCall::FinalAnswer { .. } => {}
        }
        Ok(())
    }

    /// Session-cumulative `(llm, tool, agent)` call counts.
    pub fn call_counts(&self) -> (usize, usize, usize) {
        let counters = self.counters.lock().expect("counters poisoned");
        (counters.llm, counters.tool, counters.agent)
    }

    async fn handle_llm(
        &self,
        model: Option<String>,
        prompt: String,
        system: Option<String>,
    ) -> Result<Value> {
        let model_name = model
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| {
                TinyAgentsError::Validation(
                    "llm: no model named and the session has no default model".to_string(),
                )
            })?;
        let model = self
            .registry
            .model(&model_name)
            .ok_or_else(|| TinyAgentsError::ModelNotFound(model_name.clone()))?;
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(prompt));
        // `model_name` is the registry name, not a provider model id; the
        // resolved ChatModel carries its own provider configuration.
        let request = ModelRequest {
            messages,
            ..Default::default()
        };
        let start = Instant::now();
        let response = model.invoke(&self.state, request).await?;
        let text = Message::Assistant(response.message).text();
        self.record(RlmCallRecord {
            kind: super::types::RlmCallKind::Llm,
            name: model_name,
            detail: json!({ "chars": text.len() }),
            elapsed: start.elapsed(),
        });
        Ok(Value::String(text))
    }

    async fn handle_tool(&self, tool_name: String, arguments: Value) -> Result<Value> {
        let tool = self
            .registry
            .tool(&tool_name)
            .ok_or_else(|| TinyAgentsError::ToolNotFound(tool_name.clone()))?;
        let call = ToolCall::new(
            crate::harness::ids::new_call_id().as_str().to_string(),
            tool_name.clone(),
            arguments.clone(),
        );
        // Validate against the tool's schema up front so the script gets a
        // precise, catchable error instead of tool-dependent behavior.
        tool.schema().validate_call(&call)?;
        let start = Instant::now();
        let result = tool.call(&self.state, call).await?;
        self.record(RlmCallRecord {
            kind: super::types::RlmCallKind::Tool,
            name: tool_name,
            detail: json!({ "arguments": arguments }),
            elapsed: start.elapsed(),
        });
        if let Some(error) = result.error {
            return Err(TinyAgentsError::Tool(error));
        }
        match result.raw {
            Some(raw) => Ok(raw),
            None => Ok(Value::String(result.content)),
        }
    }

    async fn handle_agent(
        &self,
        agent_name: String,
        input: String,
        data: Option<Value>,
    ) -> Result<Value> {
        // Reuse the shared harness depth guard so RLM sub-runs stay in
        // lock-step with SubAgent / SubAgentTool / the REPL.
        crate::harness::context::RunConfig::checked_child_depth(
            self.run_depth,
            self.policy.max_depth,
        )?;
        let agent = self.registry.agent(&agent_name).ok_or_else(|| {
            TinyAgentsError::Capability(format!("agent `{agent_name}` is not registered"))
        })?;
        let mut sub_input = SubAgentInput::prompt(input);
        if let Some(data) = data {
            sub_input = sub_input.with_data(data);
        }
        let start = Instant::now();
        let output = agent.run(sub_input, self.events.clone()).await?;
        self.record(RlmCallRecord {
            kind: super::types::RlmCallKind::Agent,
            name: agent_name,
            detail: json!({
                "model_calls": output.model_calls,
                "tool_calls": output.tool_calls,
            }),
            elapsed: start.elapsed(),
        });
        Ok(Value::String(output.text))
    }
}

#[async_trait]
impl<State: Send + Sync + 'static> RlmHostApi for RlmHost<State> {
    async fn handle(&self, call: HostCall) -> Result<Value> {
        if self.cancel.is_cancelled() {
            return Err(TinyAgentsError::Cancelled);
        }
        if let Some(deadline) = self.deadline()
            && Instant::now() >= deadline
        {
            return Err(TinyAgentsError::Timeout(
                "rlm cell exceeded its wall-clock timeout".to_string(),
            ));
        }
        self.bump(&call)?;
        match call {
            HostCall::Llm {
                model,
                prompt,
                system,
            } => self.handle_llm(model, prompt, system).await,
            HostCall::Tool { tool, arguments } => self.handle_tool(tool, arguments).await,
            HostCall::Agent { agent, input, data } => self.handle_agent(agent, input, data).await,
            HostCall::FinalAnswer { answer } => {
                let mut cell = self.cell.lock().expect("cell buffers poisoned");
                cell.final_answer = Some(answer);
                cell.calls.push(RlmCallRecord {
                    kind: super::types::RlmCallKind::FinalAnswer,
                    name: "final_answer".to_string(),
                    detail: Value::Null,
                    elapsed: Duration::default(),
                });
                Ok(Value::Null)
            }
        }
    }

    fn capabilities(&self) -> CapabilityListing {
        let tools = self
            .registry
            .names(ComponentKind::Tool)
            .into_iter()
            .map(|name| {
                let description = self
                    .registry
                    .tool(&name)
                    .map(|tool| tool.description().to_string())
                    .unwrap_or_default();
                (name, description)
            })
            .collect();
        CapabilityListing {
            models: self.registry.names(ComponentKind::Model),
            tools,
            agents: self.registry.names(ComponentKind::Agent),
        }
    }

    fn deadline(&self) -> Option<Instant> {
        self.cell.lock().expect("cell buffers poisoned").deadline
    }

    fn cancel_flag(&self) -> RlmCancelFlag {
        self.cancel.clone()
    }
}
