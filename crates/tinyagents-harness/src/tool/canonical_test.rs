//! Tests for [`ToolRegistry`]'s canonical (non-recursive) dispatch path:
//! registration, lookup, name listing, schema/spec/policy projection, and
//! execution through the internal `CanonicalDispatch` adapter.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::{RegisterOutcome, ToolRegistry};

struct Echo;

#[async_trait]
impl tinytools::Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns text."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"text": {"type": "string"}, "call_id": {"type": "string"}},
            "required": ["text", "call_id"]
        })
    }

    fn injected_arguments(&self) -> Vec<tinytools::ToolInjectedArgument> {
        vec![tinytools::ToolInjectedArgument::tool_call_id("call_id")]
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success(
            arguments["text"].as_str().unwrap_or_default(),
        ))
    }
}

#[test]
fn registry_accepts_the_canonical_trait_and_hides_injected_values() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(Echo));

    let schema = registry.schemas().pop().expect("registered schema");
    assert!(schema.parameters["properties"].get("call_id").is_none());
    assert_eq!(schema.parameters["required"], json!(["text"]));
}

/// M-5 regression: a second `Echo` registered under the same name used to
/// silently overwrite the first with no signal at all. `register` keeps its
/// `&mut Self`-chaining, non-breaking signature, but `try_register` now
/// reports the collision so a caller that wants to detect it can.
#[test]
fn try_register_reports_a_duplicate_name() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    assert_eq!(
        registry.try_register(Arc::new(Echo)),
        RegisterOutcome::Registered
    );
    assert_eq!(
        registry.try_register(Arc::new(Echo)),
        RegisterOutcome::Replaced("echo".to_string())
    );
    // Still only one entry under the name; the second registration replaced
    // the first rather than being rejected outright.
    assert_eq!(registry.names(), vec!["echo".to_string()]);
}

#[test]
fn register_still_replaces_silently_for_the_non_breaking_api() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    // `register` keeps its existing `&mut Self` chaining contract even on a
    // duplicate name; the collision is only surfaced through `try_register`
    // or the `tracing::warn!` diagnostic.
    registry.register(Arc::new(Echo)).register(Arc::new(Echo));
    assert_eq!(registry.names(), vec!["echo".to_string()]);
}

#[test]
fn preparation_discards_a_forged_call_id_before_validation() {
    let call = tinytools::ToolCall::new(
        tinytools::ToolCallId::new("real-call"),
        "echo",
        json!({"text": "hi", "call_id": "forged"}),
    );
    let prepared = tinytools::prepare_tool_arguments(
        &call,
        &[tinytools::ToolInjectedArgument::tool_call_id("call_id")],
        &tinytools::InjectedToolArguments::new(),
    )
    .expect("canonical preparation succeeds");

    assert_eq!(prepared["call_id"], "real-call");
}

struct Exposed {
    name: &'static str,
    exposure: tinytools::ToolExposure,
}

#[async_trait]
impl tinytools::Tool for Exposed {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "exposure fixture"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    fn exposure(&self) -> tinytools::ToolExposure {
        self.exposure
    }

    async fn execute(
        &self,
        _arguments: serde_json::Value,
    ) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success("ok"))
    }
}

fn exposure_registry() -> ToolRegistry<(), ()> {
    use tinytools::ToolExposure::{Deferred, Direct, Hidden};
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    for (name, exposure) in [
        ("zeta_direct", Direct),
        ("alpha_direct", Direct),
        ("mid_deferred", Deferred),
        ("beta_deferred", Deferred),
        ("hidden_step", Hidden),
    ] {
        registry.register(Arc::new(Exposed { name, exposure }));
    }
    registry
}

#[test]
fn schemas_split_by_exposure_and_stay_name_sorted() {
    let registry = exposure_registry();
    let direct: Vec<_> = registry.schemas().into_iter().map(|s| s.name).collect();
    assert_eq!(direct, vec!["alpha_direct", "zeta_direct"]);
    let deferred: Vec<_> = registry
        .deferred_schemas()
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(deferred, vec!["beta_deferred", "mid_deferred"]);
    // `names()` still lists everything the host registered, hidden included.
    assert_eq!(registry.names().len(), 5);
    assert_eq!(
        registry.model_callable_names(),
        vec![
            "alpha_direct",
            "beta_deferred",
            "mid_deferred",
            "zeta_direct"
        ]
    );
}

#[test]
fn hidden_tools_resolve_for_the_host_but_not_the_model() {
    let registry = exposure_registry();
    assert_eq!(
        registry.exposure("hidden_step"),
        Some(tinytools::ToolExposure::Hidden)
    );
    assert!(registry.dispatch("hidden_step").is_some());
    assert!(registry.model_dispatch("hidden_step").is_none());
    assert!(registry.model_dispatch("mid_deferred").is_some());
    assert!(registry.model_dispatch("alpha_direct").is_some());
    assert!(registry.model_dispatch("missing").is_none());
    assert_eq!(registry.exposure("missing"), None);
}

#[test]
fn tool_schema_bytes_matches_compact_wire_json() {
    let registry = exposure_registry();
    let schemas = registry.schemas();
    let expected: usize = schemas
        .iter()
        .map(|s| {
            serde_json::to_vec(&json!({
                "name": s.name, "description": s.description, "parameters": s.parameters
            }))
            .unwrap()
            .len()
        })
        .sum();
    assert_eq!(
        crate::token_estimation::tool_schema_bytes(&schemas),
        expected
    );
    assert_eq!(crate::token_estimation::tool_schema_bytes(&[]), 0);
}
