//! Model-facing controls for the thread goal, exposed as ordinary harness
//! [`Tool`]s.
//!
//! Ownership is **asymmetric**: a model may read the goal (`goal_get`), create
//! or replace it (`goal_set`), and mark it done (`goal_complete`). Pause /
//! resume / clear are host-driven controls — constructible as tools for a host
//! that wants to expose them, but not part of the default model-facing set
//! returned by [`goal_tools`].
//!
//! The target thread is resolved from
//! [`ToolExecutionContext::thread_id`](crate::harness::tool::ToolExecutionContext),
//! the harness analogue of an ambient thread id: a tool never takes a
//! `thread_id` argument, so a model can't address another thread's goal. The
//! bare [`Tool::call`] entry point (no context) errors, matching the "tools
//! require an active thread" contract.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::store;
use super::types::ThreadGoal;
use crate::error::Result;
use crate::harness::store::Store;
use crate::harness::tool::{
    Tool, ToolCall, ToolExecutionContext, ToolPolicy, ToolRegistry, ToolResult, ToolSchema,
    ToolSideEffects,
};

/// Which thread-goal control a [`GoalTool`] implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalToolKind {
    /// Read the current thread goal.
    Get,
    /// Create or replace the current thread goal.
    Set,
    /// Mark the current thread goal complete.
    Complete,
    /// Pause the current thread goal (host control).
    Pause,
    /// Resume a paused thread goal (host control).
    Resume,
    /// Delete the current thread goal (host control).
    Clear,
}

impl GoalToolKind {
    /// The default model-facing controls (asymmetric ownership).
    pub const MODEL_FACING: [Self; 3] = [Self::Get, Self::Set, Self::Complete];

    /// Every control, including the host-driven pause/resume/clear.
    pub const ALL: [Self; 6] = [
        Self::Get,
        Self::Set,
        Self::Complete,
        Self::Pause,
        Self::Resume,
        Self::Clear,
    ];

    /// Stable model-visible tool name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Get => "goal_get",
            Self::Set => "goal_set",
            Self::Complete => "goal_complete",
            Self::Pause => "goal_pause",
            Self::Resume => "goal_resume",
            Self::Clear => "goal_clear",
        }
    }

    /// Short model-visible description.
    pub fn description(self) -> &'static str {
        match self {
            Self::Get => {
                "Read this thread's goal — the durable objective you're pursuing across \
                 turns — with its status (active/paused/budget_limited/complete) and token \
                 usage. Returns 'no goal set' when the thread has none."
            }
            Self::Set => {
                "Set (or replace) this thread's goal — the durable objective you should keep \
                 pursuing across turns until it's complete. Changing the objective resets \
                 usage counters. Optionally set a token_budget; when reached, work halts."
            }
            Self::Complete => {
                "Mark this thread's goal complete. Only call this when concrete evidence \
                 confirms the objective is satisfied — completing stops autonomous \
                 continuation."
            }
            Self::Pause => "Pause this thread's active goal (host control).",
            Self::Resume => "Resume this thread's paused goal (host control).",
            Self::Clear => "Delete this thread's goal (host control).",
        }
    }

    /// Whether the control only reads state.
    fn read_only(self) -> bool {
        matches!(self, Self::Get)
    }

    /// The model-visible JSON-schema parameters for the control.
    fn parameters(self) -> Value {
        match self {
            Self::Set => json!({
                "type": "object",
                "required": ["objective"],
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The durable objective — what 'done' looks like for this thread."
                    },
                    "token_budget": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional token ceiling for the goal. Omit for no limit."
                    }
                }
            }),
            _ => json!({ "type": "object", "properties": {} }),
        }
    }
}

/// A harness [`Tool`] for one thread-goal control, backed by a
/// [`Store`](crate::harness::store::Store).
pub struct GoalTool {
    kind: GoalToolKind,
    store: Arc<dyn Store>,
}

impl GoalTool {
    /// Creates one goal tool of `kind` backed by `store`.
    pub fn new(kind: GoalToolKind, store: Arc<dyn Store>) -> Self {
        Self { kind, store }
    }

    /// The control kind this tool implements.
    pub fn kind(&self) -> GoalToolKind {
        self.kind
    }

    /// Dispatches the control against `thread_id`, returning the model-facing
    /// content and a structured `raw` payload.
    async fn dispatch(&self, thread_id: &str, args: &Value) -> Result<(String, Option<Value>)> {
        match self.kind {
            GoalToolKind::Get => match store::get(&self.store, thread_id).await? {
                Some(goal) => Ok((render_goal(&goal), Some(serde_json::to_value(&goal)?))),
                None => Ok(("no goal set for this thread".to_string(), None)),
            },
            GoalToolKind::Set => {
                let Some(objective) = args.get("objective").and_then(Value::as_str) else {
                    return Ok(("error: missing 'objective' parameter".to_string(), None));
                };
                let token_budget = args.get("token_budget").and_then(Value::as_u64);
                let goal = store::set(&self.store, thread_id, objective, token_budget).await?;
                Ok((
                    format!("Goal set.\n{}", render_goal(&goal)),
                    Some(serde_json::to_value(&goal)?),
                ))
            }
            GoalToolKind::Complete => {
                let goal = store::complete(&self.store, thread_id).await?;
                Ok((
                    format!("Goal marked complete.\n{}", render_goal(&goal)),
                    Some(serde_json::to_value(&goal)?),
                ))
            }
            GoalToolKind::Pause => {
                let goal = store::pause(&self.store, thread_id).await?;
                Ok((render_goal(&goal), Some(serde_json::to_value(&goal)?)))
            }
            GoalToolKind::Resume => {
                let goal = store::resume(&self.store, thread_id).await?;
                Ok((render_goal(&goal), Some(serde_json::to_value(&goal)?)))
            }
            GoalToolKind::Clear => {
                let removed = store::clear(&self.store, thread_id).await?;
                Ok((
                    format!("Goal cleared (removed={removed})."),
                    Some(json!({ "removed": removed })),
                ))
            }
        }
    }
}

/// Renders a goal as a compact, model-readable block.
fn render_goal(goal: &ThreadGoal) -> String {
    let budget = match goal.token_budget {
        Some(b) => format!(
            "{} used / {b} budget ({} left)",
            goal.tokens_used,
            goal.budget_remaining().unwrap_or(0)
        ),
        None => format!("{} used / no budget", goal.tokens_used),
    };
    format!(
        "[thread_goal]\nstatus: {}\nobjective: {}\ntokens: {budget}\n[/thread_goal]",
        goal.status.as_str(),
        goal.objective
    )
}

fn error_result(call_id: String, name: &str, message: impl Into<String>) -> ToolResult {
    ToolResult {
        call_id,
        name: name.to_string(),
        content: String::new(),
        raw: None,
        error: Some(message.into()),
        elapsed_ms: 0,
    }
}

/// Builds the default **model-facing** goal controls (`goal_get`, `goal_set`,
/// `goal_complete`).
pub fn goal_tools(store: Arc<dyn Store>) -> Vec<Arc<GoalTool>> {
    GoalToolKind::MODEL_FACING
        .into_iter()
        .map(|kind| Arc::new(GoalTool::new(kind, store.clone())))
        .collect()
}

/// Registers the default model-facing goal controls into a tool registry.
pub fn register_goal_tools<State: Send + Sync>(
    registry: &mut ToolRegistry<State>,
    store: Arc<dyn Store>,
) -> &mut ToolRegistry<State> {
    for tool in goal_tools(store) {
        registry.register(tool);
    }
    registry
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for GoalTool {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn description(&self) -> &str {
        self.kind.description()
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.kind.name().to_string(),
            description: self.kind.description().to_string(),
            parameters: self.kind.parameters(),
            format: Default::default(),
        }
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy {
            classified: true,
            side_effects: ToolSideEffects {
                read_only: self.kind.read_only(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        Ok(error_result(
            call.id,
            self.kind.name(),
            "goal tools require an active thread (no thread_id in tool context)",
        ))
    }

    async fn call_with_context(
        &self,
        _state: &State,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult> {
        let Some(thread_id) = context.thread_id.as_ref() else {
            return Ok(error_result(
                call.id,
                self.kind.name(),
                "goal tools require an active thread (no thread_id in tool context)",
            ));
        };
        let (content, raw) = self.dispatch(thread_id.as_str(), &call.arguments).await?;
        Ok(ToolResult {
            call_id: call.id,
            name: self.kind.name().to_string(),
            content,
            raw,
            error: None,
            elapsed_ms: 0,
        })
    }
}
