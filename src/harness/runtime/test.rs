//! Tests for the [`AgentHarness`] builder and [`RunPolicy`].

use std::sync::Arc;

use crate::Result;
use crate::harness::limits::RunLimits;
use crate::harness::middleware::LoggingMiddleware;
use crate::harness::providers::MockModel;
use crate::harness::retry::{FallbackPolicy, RetryPolicy};
use crate::harness::runtime::{AgentHarness, RunPolicy};
use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

use async_trait::async_trait;
use serde_json::json;

struct NoopTool;

#[async_trait]
impl Tool<()> for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("noop", "does nothing", json!({"type": "object"}))
    }
    async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::text(call.id, "noop", "ok"))
    }
}

#[test]
fn new_harness_is_empty_with_default_policy() {
    let harness: AgentHarness<()> = AgentHarness::new();
    assert!(harness.models().default_name().is_none());
    assert_eq!(harness.tools().names().len(), 0);
    assert!(harness.middleware().is_empty());
    assert_eq!(harness.policy(), &RunPolicy::default());
}

#[test]
fn register_first_model_becomes_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("a", Arc::new(MockModel::constant("a")))
        .register_model("b", Arc::new(MockModel::constant("b")));
    assert_eq!(harness.models().default_name(), Some("a"));
    harness.set_default_model("b");
    assert_eq!(harness.models().default_name(), Some("b"));
}

#[test]
fn register_tool_and_push_middleware() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.push_middleware(Arc::new(LoggingMiddleware::new()));
    assert_eq!(harness.tools().names(), vec!["noop".to_string()]);
    assert_eq!(harness.middleware().len(), 1);
}

#[test]
fn with_policy_replaces_policy() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let policy = RunPolicy {
        limits: RunLimits::default().with_max_model_calls(3),
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy::new(["a", "b"])),
        default_response_format: None,
        ..RunPolicy::default()
    };
    harness.with_policy(policy.clone());
    assert_eq!(harness.policy(), &policy);
    assert_eq!(harness.policy().limits.max_model_calls, 3);
}

#[test]
fn default_matches_new() {
    let harness: AgentHarness<()> = AgentHarness::default();
    assert!(harness.models().default_name().is_none());
}
