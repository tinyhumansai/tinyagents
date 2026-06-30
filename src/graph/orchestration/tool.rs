//! Ordinary harness tools for graph orchestration controls.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::harness::ids::{GraphId, TaskId, new_call_id, next_seq};
use crate::harness::tool::{Tool, ToolCall, ToolRegistry, ToolResult, ToolSchema};
use crate::{Result, TinyAgentsError};

use super::store::{TaskStore, orchestration_not_found};
use super::types::*;

/// A standard harness [`Tool`] for one orchestration control.
///
/// Instances are inserted into a [`ToolRegistry`] like any other tool. The
/// model sees a normal [`ToolSchema`], and execution returns a normal
/// [`ToolResult`]; the only shared dependency is the task store backing the
/// control.
pub struct OrchestrationTool {
    kind: OrchestrationToolKind,
    store: Arc<dyn TaskStore>,
}

impl OrchestrationTool {
    /// Creates one orchestration tool backed by `store`.
    pub fn new(kind: OrchestrationToolKind, store: Arc<dyn TaskStore>) -> Self {
        Self { kind, store }
    }

    /// The control kind this tool implements.
    pub fn kind(&self) -> OrchestrationToolKind {
        self.kind
    }
}

/// Builds every built-in orchestration control as ordinary harness tools.
pub fn orchestration_tools(store: Arc<dyn TaskStore>) -> Vec<Arc<OrchestrationTool>> {
    OrchestrationToolKind::ALL
        .into_iter()
        .map(|kind| Arc::new(OrchestrationTool::new(kind, store.clone())))
        .collect()
}

/// Registers every built-in orchestration control in a normal tool registry.
pub fn register_orchestration_tools<State: Send + Sync>(
    registry: &mut ToolRegistry<State>,
    store: Arc<dyn TaskStore>,
) -> &mut ToolRegistry<State> {
    for tool in orchestration_tools(store) {
        registry.register(tool);
    }
    registry
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for OrchestrationTool {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn description(&self) -> &str {
        self.kind.description()
    }

    fn schema(&self) -> ToolSchema {
        orchestration_tool_schema(self.kind)
    }

    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        orchestration_tool_schema(self.kind).validate_call(&call)?;
        let raw = match self.kind {
            OrchestrationToolKind::Spawn => self.call_spawn(&call.arguments)?,
            OrchestrationToolKind::Await => self.call_await(&call.arguments)?,
            OrchestrationToolKind::Cancel => {
                let outcome = self.store.request_cancel(&task_id_arg(&call.arguments)?)?;
                serde_json::to_value(outcome)?
            }
            OrchestrationToolKind::Kill => {
                let outcome = self.store.kill(&task_id_arg(&call.arguments)?)?;
                serde_json::to_value(outcome)?
            }
            OrchestrationToolKind::Status => {
                let task_id = task_id_arg(&call.arguments)?;
                let record = self
                    .store
                    .get(&task_id)
                    .ok_or_else(|| orchestration_not_found(&task_id))?;
                serde_json::to_value(record)?
            }
            OrchestrationToolKind::List => self.call_list(&call.arguments)?,
            OrchestrationToolKind::Timeout => self.call_timeout(&call.arguments)?,
            OrchestrationToolKind::Race => self.call_race(&call.arguments)?,
            OrchestrationToolKind::YieldInterrupt => self.call_yield(&call.arguments)?,
            OrchestrationToolKind::Steer => self.call_steer(&call.arguments)?,
        };

        let content = serde_json::to_string(&raw)?;
        Ok(ToolResult {
            call_id: call.id,
            name: self.kind.name().to_string(),
            content,
            raw: Some(raw),
            error: None,
            elapsed_ms: 0,
        })
    }
}

impl OrchestrationTool {
    fn call_spawn(&self, args: &Value) -> Result<Value> {
        let kind = required_str(args, "kind")?;
        let target = required_str(args, "target")?;
        let task_id = TaskId::new(format!("task-{}", next_seq()));
        let mut spec = OrchestrationTaskSpec::new(task_id, task_kind_from_args(kind, target)?);
        if let Some(input) = args.get("input") {
            spec = spec.with_input(input.clone());
        }
        if let Some(timeout_ms) = optional_u64(args, "timeout_ms")? {
            spec = spec.with_timeout_ms(timeout_ms);
        }
        serde_json::to_value(self.store.insert(spec)?).map_err(Into::into)
    }

    fn call_await(&self, args: &Value) -> Result<Value> {
        let records = task_ids_arg(args)?
            .into_iter()
            .map(|task_id| {
                self.store
                    .get(&task_id)
                    .ok_or_else(|| orchestration_not_found(&task_id))
            })
            .collect::<Result<Vec<_>>>()?;
        serde_json::to_value(records).map_err(Into::into)
    }

    fn call_list(&self, args: &Value) -> Result<Value> {
        let filter = OrchestrationTaskFilter {
            parent_run_id: optional_string(args, "parent_run_id")?.map(Into::into),
            root_run_id: optional_string(args, "root_run_id")?.map(Into::into),
            thread_id: optional_string(args, "thread_id")?.map(Into::into),
            node_id: optional_string(args, "node_id")?.map(Into::into),
            status: optional_status(args, "status")?,
        };
        serde_json::to_value(self.store.list(filter)).map_err(Into::into)
    }

    fn call_timeout(&self, args: &Value) -> Result<Value> {
        let record = self
            .store
            .set_timeout_ms(&task_id_arg(args)?, required_u64(args, "timeout_ms")?)?;
        serde_json::to_value(record).map_err(Into::into)
    }

    fn call_race(&self, args: &Value) -> Result<Value> {
        let cancel_losers = optional_bool(args, "cancel_losers")?.unwrap_or(false);
        let records = task_ids_arg(args)?
            .into_iter()
            .map(|task_id| {
                self.store
                    .get(&task_id)
                    .ok_or_else(|| orchestration_not_found(&task_id))
            })
            .collect::<Result<Vec<_>>>()?;
        let winner = records
            .iter()
            .find(|record| record.status == OrchestrationTaskStatus::Completed)
            .cloned();

        if cancel_losers && winner.is_some() {
            for record in records.iter().filter(|record| {
                record.status.is_live()
                    && Some(record.task_id()) != winner.as_ref().map(|winner| winner.task_id())
            }) {
                let _ = self.store.request_cancel(record.task_id());
            }
        }

        Ok(json!({
            "winner": winner,
            "tasks": records,
        }))
    }

    fn call_yield(&self, args: &Value) -> Result<Value> {
        Ok(json!({
            "status": "interrupt_requested",
            "message": required_str(args, "message")?,
            "resume_schema": args.get("resume_schema").cloned().unwrap_or(Value::Null),
        }))
    }

    fn call_steer(&self, args: &Value) -> Result<Value> {
        let task_id = task_id_arg(args)?;
        let record = self
            .store
            .get(&task_id)
            .ok_or_else(|| orchestration_not_found(&task_id))?;
        Ok(json!({
            "task_id": task_id,
            "status": record.status,
            "command": required_str(args, "command")?,
            "accepted": record.status.is_live(),
            "steering_id": new_call_id(),
        }))
    }
}

/// Returns model-visible schemas for all built-in orchestration tools.
pub fn orchestration_tool_schemas() -> Vec<ToolSchema> {
    OrchestrationToolKind::ALL
        .into_iter()
        .map(orchestration_tool_schema)
        .collect()
}

/// Returns the model-visible schema for one built-in orchestration tool.
pub fn orchestration_tool_schema(kind: OrchestrationToolKind) -> ToolSchema {
    ToolSchema::new(
        kind.name(),
        kind.description(),
        orchestration_parameters(kind),
    )
}

fn orchestration_parameters(kind: OrchestrationToolKind) -> Value {
    match kind {
        OrchestrationToolKind::Spawn => json!({
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["graph", "sub_agent", "tool", "external_process"]
                },
                "target": {
                    "type": "string",
                    "description": "Registered graph, agent, tool, or external-process label."
                },
                "input": { "description": "Structured task input." },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional task deadline in milliseconds."
                }
            },
            "required": ["kind", "target"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::Await => json!({
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "timeout_ms": { "type": "integer" },
                "mode": { "type": "string", "enum": ["all", "any"] }
            },
            "required": ["task_ids"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::Cancel
        | OrchestrationToolKind::Kill
        | OrchestrationToolKind::Status => json!({
            "type": "object",
            "properties": { "task_id": { "type": "string" } },
            "required": ["task_id"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::List => json!({
            "type": "object",
            "properties": {
                "parent_run_id": { "type": "string" },
                "root_run_id": { "type": "string" },
                "thread_id": { "type": "string" },
                "node_id": { "type": "string" },
                "status": {
                    "type": "string",
                    "enum": [
                        "pending",
                        "running",
                        "awaiting",
                        "completed",
                        "failed",
                        "cancel_requested",
                        "cancelled",
                        "timed_out",
                        "abandoned"
                    ]
                }
            },
            "additionalProperties": false
        }),
        OrchestrationToolKind::Timeout => json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string" },
                "timeout_ms": { "type": "integer" }
            },
            "required": ["task_id", "timeout_ms"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::Race => json!({
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "timeout_ms": { "type": "integer" },
                "cancel_losers": { "type": "boolean" }
            },
            "required": ["task_ids"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::YieldInterrupt => json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" },
                "resume_schema": {
                    "description": "Optional JSON schema describing the expected resume payload."
                }
            },
            "required": ["message"],
            "additionalProperties": false
        }),
        OrchestrationToolKind::Steer => json!({
            "type": "object",
            "properties": {
                "task_id": { "type": "string" },
                "command": {
                    "type": "string",
                    "enum": ["pause", "resume", "cancel", "inject_message", "redirect", "set_metadata"]
                },
                "payload": { "description": "Command-specific steering payload." }
            },
            "required": ["task_id", "command"],
            "additionalProperties": false
        }),
    }
}

fn task_kind_from_args(kind: &str, target: &str) -> Result<OrchestrationTaskKind> {
    match kind {
        "graph" => Ok(OrchestrationTaskKind::Graph {
            graph_id: GraphId::new(target),
        }),
        "sub_agent" => Ok(OrchestrationTaskKind::SubAgent {
            agent: target.to_string(),
        }),
        "tool" => Ok(OrchestrationTaskKind::Tool {
            tool: target.to_string(),
        }),
        "external_process" => Ok(OrchestrationTaskKind::ExternalProcess {
            label: target.to_string(),
        }),
        other => Err(TinyAgentsError::Validation(format!(
            "unsupported orchestration task kind `{other}`"
        ))),
    }
}

fn task_id_arg(args: &Value) -> Result<TaskId> {
    required_str(args, "task_id").map(TaskId::new)
}

fn task_ids_arg(args: &Value) -> Result<Vec<TaskId>> {
    let values = args
        .get("task_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| TinyAgentsError::Validation("task_ids must be an array".to_string()))?;
    if values.is_empty() {
        return Err(TinyAgentsError::Validation(
            "task_ids must contain at least one task id".to_string(),
        ));
    }
    values
        .iter()
        .map(|value| {
            value.as_str().map(TaskId::new).ok_or_else(|| {
                TinyAgentsError::Validation("task_ids entries must be strings".to_string())
            })
        })
        .collect()
}

fn required_str<'a>(args: &'a Value, field: &str) -> Result<&'a str> {
    args.get(field).and_then(Value::as_str).ok_or_else(|| {
        TinyAgentsError::Validation(format!("orchestration argument `{field}` must be a string"))
    })
}

fn required_u64(args: &Value, field: &str) -> Result<u64> {
    args.get(field).and_then(Value::as_u64).ok_or_else(|| {
        TinyAgentsError::Validation(format!(
            "orchestration argument `{field}` must be a non-negative integer"
        ))
    })
}

fn optional_u64(args: &Value, field: &str) -> Result<Option<u64>> {
    match args.get(field) {
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            TinyAgentsError::Validation(format!(
                "orchestration argument `{field}` must be a non-negative integer"
            ))
        }),
        None => Ok(None),
    }
}

fn optional_bool(args: &Value, field: &str) -> Result<Option<bool>> {
    match args.get(field) {
        Some(value) => value.as_bool().map(Some).ok_or_else(|| {
            TinyAgentsError::Validation(format!(
                "orchestration argument `{field}` must be a boolean"
            ))
        }),
        None => Ok(None),
    }
}

fn optional_string(args: &Value, field: &str) -> Result<Option<String>> {
    match args.get(field) {
        Some(value) => value.as_str().map(|s| Some(s.to_string())).ok_or_else(|| {
            TinyAgentsError::Validation(format!(
                "orchestration argument `{field}` must be a string"
            ))
        }),
        None => Ok(None),
    }
}

fn optional_status(args: &Value, field: &str) -> Result<Option<OrchestrationTaskStatus>> {
    optional_string(args, field)?
        .map(|status| parse_status(&status))
        .transpose()
}

fn parse_status(status: &str) -> Result<OrchestrationTaskStatus> {
    match status {
        "pending" => Ok(OrchestrationTaskStatus::Pending),
        "running" => Ok(OrchestrationTaskStatus::Running),
        "awaiting" => Ok(OrchestrationTaskStatus::Awaiting),
        "completed" => Ok(OrchestrationTaskStatus::Completed),
        "failed" => Ok(OrchestrationTaskStatus::Failed),
        "cancel_requested" => Ok(OrchestrationTaskStatus::CancelRequested),
        "cancelled" => Ok(OrchestrationTaskStatus::Cancelled),
        "timed_out" => Ok(OrchestrationTaskStatus::TimedOut),
        "abandoned" => Ok(OrchestrationTaskStatus::Abandoned),
        other => Err(TinyAgentsError::Validation(format!(
            "unsupported orchestration task status `{other}`"
        ))),
    }
}
