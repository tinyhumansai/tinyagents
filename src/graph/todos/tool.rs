//! `todo` — a single multiplexer harness [`Tool`] over the per-thread task
//! board.
//!
//! Dispatches on the `op` field so one tool exposes `add` / `edit` /
//! `update_status` / `decide_plan` / `revise_plan` / `remove` / `replace` /
//! `clear` / `list`. The board is bound to the caller's
//! [`ToolExecutionContext::thread_id`] (never a tool argument), so a model can't
//! address another thread's board; the bare [`Tool::call`] entry point (no
//! context) errors. Returns the updated cards plus a markdown rendering.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::store;
use super::types::{CardPatch, TaskApprovalMode, TaskBoardCard, parse_status};
use crate::error::Result;
use crate::harness::store::Store;
use crate::harness::tool::{
    Tool, ToolCall, ToolExecutionContext, ToolPolicy, ToolRegistry, ToolResult, ToolSchema,
    ToolSideEffects,
};

const TODO_TOOL_NAME: &str = "todo";

const TODO_DESCRIPTION: &str = "Maintain a visible plan for THIS thread: an ordered kanban board of \
    task cards that survives across turns. Use it for any request with several distinct steps: at \
    the start, `add` one card per step; keep exactly ONE card `in_progress` at a time; mark a card \
    `done` the moment it finishes; if a step is blocked, set it `blocked` with a `blocker`. `list` \
    to re-read the plan. The board is bound automatically to the current thread — do not pass a \
    thread id. Dispatch via `op`: `add` (content, status?, objective?, plan?, assignedAgent?, \
    allowedTools?, approvalMode?, acceptanceCriteria?, evidence?, notes?, blocker?), `edit` (id, \
    same optional fields), `update_status` (id, status), `decide_plan` (id, approve), \
    `revise_plan`, `remove` (id), `replace` (cards), `clear`, or `list`. Returns the updated cards \
    plus a markdown rendering.";

/// A single harness [`Tool`] exposing the whole task-board CRUD surface, backed
/// by a [`Store`](crate::harness::store::Store).
pub struct TodoTool {
    store: Arc<dyn Store>,
}

impl TodoTool {
    /// Creates the `todo` tool backed by `store`.
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }

    async fn dispatch(&self, thread_id: &str, args: &Value) -> Result<TodoOutcome> {
        let Some(op) = args.get("op").and_then(Value::as_str).map(str::trim) else {
            return Ok(TodoOutcome::Error(
                "missing required field `op`".to_string(),
            ));
        };
        let s = &self.store;
        let snap = match op {
            "add" => {
                let Some(content) = required_str(args, "content") else {
                    return Ok(TodoOutcome::Error(
                        "missing required field `content`".to_string(),
                    ));
                };
                match patch_from_args(args) {
                    Ok(patch) => store::add(s, thread_id, &content, patch).await,
                    Err(e) => return Ok(TodoOutcome::Error(e)),
                }
            }
            "edit" => {
                let Some(id) = required_str(args, "id") else {
                    return Ok(TodoOutcome::Error(
                        "missing required field `id`".to_string(),
                    ));
                };
                match patch_from_args(args) {
                    Ok(mut patch) => {
                        patch.content = args
                            .get("content")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        store::edit(s, thread_id, &id, patch).await
                    }
                    Err(e) => return Ok(TodoOutcome::Error(e)),
                }
            }
            "update_status" => {
                let (Some(id), Some(status)) =
                    (required_str(args, "id"), required_str(args, "status"))
                else {
                    return Ok(TodoOutcome::Error(
                        "update_status requires `id` and `status`".to_string(),
                    ));
                };
                match parse_status(&status) {
                    Ok(status) => store::update_status(s, thread_id, &id, status).await,
                    Err(e) => return Ok(TodoOutcome::Error(e)),
                }
            }
            "decide_plan" => {
                let Some(id) = required_str(args, "id") else {
                    return Ok(TodoOutcome::Error(
                        "missing required field `id`".to_string(),
                    ));
                };
                let Some(approve) = args.get("approve").and_then(Value::as_bool) else {
                    return Ok(TodoOutcome::Error(
                        "decide_plan requires a boolean `approve`".to_string(),
                    ));
                };
                store::decide_plan(s, thread_id, &id, approve).await
            }
            "revise_plan" => store::revise_plan(s, thread_id).await,
            "remove" => {
                let Some(id) = required_str(args, "id") else {
                    return Ok(TodoOutcome::Error(
                        "missing required field `id`".to_string(),
                    ));
                };
                store::remove(s, thread_id, &id).await
            }
            "replace" => {
                let Some(cards_value) = args.get("cards") else {
                    return Ok(TodoOutcome::Error(
                        "missing `cards` for op=replace".to_string(),
                    ));
                };
                match serde_json::from_value::<Vec<TaskBoardCard>>(cards_value.clone()) {
                    Ok(cards) => store::replace(s, thread_id, cards).await,
                    Err(e) => return Ok(TodoOutcome::Error(format!("invalid `cards`: {e}"))),
                }
            }
            "clear" => store::clear(s, thread_id).await,
            "list" => store::list(s, thread_id).await,
            other => {
                return Ok(TodoOutcome::Error(format!(
                    "unknown op '{other}' (expected add|edit|update_status|decide_plan|revise_plan|remove|replace|clear|list)"
                )));
            }
        };
        // A domain error (unknown id, invariant violation) is surfaced to the
        // model rather than failing the whole run.
        match snap {
            Ok(snap) => Ok(TodoOutcome::Ok(json!({
                "threadId": snap.thread_id,
                "cards": snap.cards,
                "markdown": snap.markdown,
            }))),
            Err(e) => Ok(TodoOutcome::Error(e.to_string())),
        }
    }
}

/// Internal dispatch outcome: a structured payload or a model-facing error.
enum TodoOutcome {
    Ok(Value),
    Error(String),
}

fn required_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn optional_string(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn optional_string_array(
    args: &Value,
    key: &str,
) -> std::result::Result<Option<Vec<String>>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("`{key}` items must be strings"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(format!("`{key}` must be an array of strings")),
    }
}

fn patch_from_args(args: &Value) -> std::result::Result<CardPatch, String> {
    let status = match args.get("status").and_then(Value::as_str) {
        Some(s) => Some(parse_status(s)?),
        None => None,
    };
    let approval_mode = match args.get("approvalMode") {
        None => None,
        Some(Value::Null) => Some(None),
        Some(Value::String(s)) => match s.as_str() {
            "required" => Some(Some(TaskApprovalMode::Required)),
            "not_required" => Some(Some(TaskApprovalMode::NotRequired)),
            other => {
                return Err(format!(
                    "invalid approvalMode '{other}' (expected required|not_required|null)"
                ));
            }
        },
        Some(_) => {
            return Err(
                "invalid approvalMode type (expected required|not_required|null)".to_string(),
            );
        }
    };
    Ok(CardPatch {
        content: None,
        status,
        objective: optional_string(args, "objective"),
        plan: optional_string_array(args, "plan")?,
        assigned_agent: optional_string(args, "assignedAgent"),
        allowed_tools: optional_string_array(args, "allowedTools")?,
        approval_mode,
        acceptance_criteria: optional_string_array(args, "acceptanceCriteria")?,
        evidence: optional_string_array(args, "evidence")?,
        notes: optional_string(args, "notes"),
        blocker: optional_string(args, "blocker"),
        source_metadata: None,
    })
}

fn parameters_schema() -> Value {
    json!({
        "type": "object",
        "required": ["op"],
        "properties": {
            "op": {
                "type": "string",
                "enum": ["add", "edit", "update_status", "decide_plan", "revise_plan", "remove", "replace", "clear", "list"]
            },
            "id": { "type": "string", "description": "Card id (required for edit/update_status/decide_plan/remove)." },
            "content": { "type": "string", "description": "Card title (required for add; optional for edit)." },
            "status": {
                "type": "string",
                "enum": ["todo", "awaiting_approval", "ready", "in_progress", "blocked", "done", "rejected"]
            },
            "approve": { "type": "boolean", "description": "For op=decide_plan: approve (true) or reject (false)." },
            "notes": { "type": "string" },
            "blocker": { "type": "string" },
            "objective": { "type": "string", "description": "Desired outcome for this task." },
            "plan": { "type": "array", "items": { "type": "string" }, "description": "Ordered execution steps." },
            "assignedAgent": { "type": "string" },
            "allowedTools": { "type": "array", "items": { "type": "string" } },
            "approvalMode": { "type": ["string", "null"], "enum": ["required", "not_required", null] },
            "acceptanceCriteria": { "type": "array", "items": { "type": "string" } },
            "evidence": { "type": "array", "items": { "type": "string" } },
            "cards": { "type": "array", "items": { "type": "object" }, "description": "Full card list for op=replace." }
        }
    })
}

fn error_result(call_id: String, message: impl Into<String>) -> ToolResult {
    ToolResult {
        call_id,
        name: TODO_TOOL_NAME.to_string(),
        content: String::new(),
        raw: None,
        error: Some(message.into()),
        elapsed_ms: 0,
    }
}

/// Builds the `todo` tool backed by `store`.
pub fn todo_tools(store: Arc<dyn Store>) -> Vec<Arc<TodoTool>> {
    vec![Arc::new(TodoTool::new(store))]
}

/// Registers the `todo` tool into a tool registry.
pub fn register_todo_tools<State: Send + Sync>(
    registry: &mut ToolRegistry<State>,
    store: Arc<dyn Store>,
) -> &mut ToolRegistry<State> {
    registry.register(Arc::new(TodoTool::new(store)));
    registry
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for TodoTool {
    fn name(&self) -> &str {
        TODO_TOOL_NAME
    }

    fn description(&self) -> &str {
        TODO_DESCRIPTION
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: TODO_TOOL_NAME.to_string(),
            description: TODO_DESCRIPTION.to_string(),
            parameters: parameters_schema(),
            format: Default::default(),
        }
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy {
            classified: true,
            side_effects: ToolSideEffects {
                read_only: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        Ok(error_result(
            call.id,
            "todo tool requires an active thread (no thread_id in tool context)",
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
                "todo tool requires an active thread (no thread_id in tool context)",
            ));
        };
        match self.dispatch(thread_id.as_str(), &call.arguments).await? {
            TodoOutcome::Ok(payload) => Ok(ToolResult {
                call_id: call.id,
                name: TODO_TOOL_NAME.to_string(),
                content: payload.to_string(),
                raw: Some(payload),
                error: None,
                elapsed_ms: 0,
            }),
            TodoOutcome::Error(message) => Ok(error_result(call.id, message)),
        }
    }
}
