//! Unit tests for the [`CapabilityRegistry`](super::CapabilityRegistry):
//! registration and lookup of models/tools/graphs, kind-scoped namespacing,
//! duplicate rejection, `replace_*` overwrite semantics, alias resolution and
//! validation, and the harness/`.rag` resolver hand-off builders.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::component::ComponentKind;
use tinyagents_definition::AgentDefinition;
use tinyagents_language::Blueprint;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinytools::{Tool, ToolResult};

struct FakeModel(&'static str);

#[async_trait]
impl ChatModel<()> for FakeModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant(self.0))
    }
}

struct FakeTool(&'static str);

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        "fake tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

fn blueprint(id: &str) -> Blueprint {
    Blueprint {
        graph_id: id.to_owned(),
        start: "a".to_owned(),
        channels: Vec::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
        defaults: Vec::new(),
        ..Blueprint::default()
    }
}

#[test]
fn registers_and_looks_up_models_tools_graphs() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("hi")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_graph_blueprint("flow", blueprint("flow"))
        .unwrap();
    reg.register_router("classify").unwrap();
    reg.register_reducer("append").unwrap();

    assert!(reg.model("default").is_some());
    assert!(reg.tool("lookup_user").is_some());
    assert!(reg.graph_blueprint("flow").is_some());

    assert!(reg.has(ComponentKind::Model, "default"));
    assert!(reg.has(ComponentKind::Tool, "lookup_user"));
    assert!(reg.has(ComponentKind::Graph, "flow"));
    assert!(reg.has(ComponentKind::Router, "classify"));
    assert!(reg.has(ComponentKind::Reducer, "append"));

    assert!(!reg.has(ComponentKind::Model, "missing"));
    assert!(reg.model("missing").is_none());
}

#[test]
fn registers_declarative_agents_without_an_executable_graph_adapter() {
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_agent(
            AgentDefinition::new("planner", "Planner", "Plans work")
                .with_tools(["todo"])
                .with_subagents(["researcher"]),
        )
        .unwrap();
    registry
        .alias(ComponentKind::Agent, "default", "planner")
        .unwrap();

    let definition = registry.agent("default").unwrap();
    assert_eq!(definition.id, "planner");
    assert_eq!(definition.subagents, vec!["researcher"]);
    assert_eq!(registry.names(ComponentKind::Agent), vec!["planner"]);
}

#[test]
fn names_are_sorted_and_kind_scoped() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("b", Arc::new(FakeModel("x"))).unwrap();
    reg.register_model("a", Arc::new(FakeModel("y"))).unwrap();
    reg.register_tool(Arc::new(FakeTool("t"))).unwrap();

    assert_eq!(reg.names(ComponentKind::Model), vec!["a", "b"]);
    assert_eq!(reg.names(ComponentKind::Tool), vec!["t"]);
    assert!(reg.names(ComponentKind::Graph).is_empty());
}

#[test]
fn same_kind_name_namespaces_are_independent() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("shared", Arc::new(FakeModel("m")))
        .unwrap();
    // Same name under a different kind is allowed.
    reg.register_router("shared").unwrap();
    assert!(reg.has(ComponentKind::Model, "shared"));
    assert!(reg.has(ComponentKind::Router, "shared"));
}

#[test]
fn duplicate_registration_is_rejected() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("a")))
        .unwrap();
    let err = reg
        .register_model("default", Arc::new(FakeModel("b")))
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::DuplicateComponent(_)));

    reg.register_tool(Arc::new(FakeTool("t"))).unwrap();
    assert!(matches!(
        reg.register_tool(Arc::new(FakeTool("t"))).unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));

    reg.register_router("r").unwrap();
    assert!(matches!(
        reg.register_router("r").unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
}

#[test]
fn replace_overwrites_without_error_and_keeps_metadata() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("first")))
        .unwrap();
    reg.alias(ComponentKind::Model, "fast", "default").unwrap();

    // Replacing the value does not error and preserves the alias metadata.
    reg.replace_model("default", Arc::new(FakeModel("second")));
    assert!(reg.model("default").is_some());
    assert!(reg.model("fast").is_some());
    let meta = reg.metadata(ComponentKind::Model, "default").unwrap();
    assert_eq!(meta.aliases, vec!["fast".to_string()]);
}

#[test]
fn aliases_resolve_in_lookups() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_graph_blueprint("flow", blueprint("flow"))
        .unwrap();

    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    reg.alias(ComponentKind::Tool, "user", "lookup_user")
        .unwrap();
    reg.alias(ComponentKind::Graph, "main", "flow").unwrap();

    assert!(reg.model("default").is_some());
    assert!(reg.tool("user").is_some());
    assert!(reg.graph_blueprint("main").is_some());
    assert!(reg.has(ComponentKind::Model, "default"));
    assert_eq!(
        reg.resolve_name(ComponentKind::Model, "default").as_deref(),
        Some("gpt-4o")
    );

    // metadata via alias resolves to the canonical entry.
    let meta = reg.metadata(ComponentKind::Model, "default").unwrap();
    assert_eq!(meta.name(), "gpt-4o");
    assert!(meta.aliases.contains(&"default".to_string()));

    // Aliases are not listed in names().
    assert_eq!(reg.names(ComponentKind::Model), vec!["gpt-4o"]);
}

#[test]
fn alias_validation_rejects_unknown_target_and_duplicates() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();

    // Unknown target.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "x", "missing").unwrap_err(),
        TinyAgentsError::Capability(_)
    ));

    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    // Duplicate alias.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "default", "gpt-4o")
            .unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
    // Alias colliding with a registered component name.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "gpt-4o", "gpt-4o")
            .unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
}

#[tokio::test]
async fn builds_harness_registries_with_model_aliases() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("hello")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    let models = reg.to_model_registry();
    assert!(models.get("gpt-4o").is_some());
    assert!(models.get("default").is_some());

    let tools = reg.to_tool_registry::<()>();
    assert_eq!(tools.names(), vec!["lookup_user"]);
}

/// Builds a registry with `charlie`, `alpha`, `bravo` registered in that
/// exact order (deliberately not alphabetical, so a name-sorted iteration
/// would pick a different "first" model than registration order does).
fn registry_with_three_models_in_order() -> CapabilityRegistry<()> {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("charlie", Arc::new(FakeModel("c")))
        .unwrap();
    reg.register_model("alpha", Arc::new(FakeModel("a")))
        .unwrap();
    reg.register_model("bravo", Arc::new(FakeModel("b")))
        .unwrap();
    reg
}

#[test]
fn to_model_registry_default_is_the_first_registered_model_every_time() {
    // Build the same registry several times over; a `HashMap`-order default
    // would vary run to run (or construction to construction within a
    // process, depending on hash-seed timing), while first-registration
    // order should not.
    for _ in 0..5 {
        let reg = registry_with_three_models_in_order();
        let models = reg.to_model_registry();
        assert_eq!(
            models.default_name(),
            Some("charlie"),
            "default model should always be the first one registered"
        );
        assert!(models.get("charlie").is_some());
        assert!(models.get("alpha").is_some());
        assert!(models.get("bravo").is_some());
    }
}

#[test]
fn replace_model_does_not_move_an_existing_name_in_registration_order() {
    let mut reg = registry_with_three_models_in_order();
    // Re-registering "bravo" (already registered second) must not make it
    // the new first-registered name.
    reg.replace_model("bravo", Arc::new(FakeModel("b2")));
    reg.register_model("delta", Arc::new(FakeModel("d")))
        .unwrap();

    let models = reg.to_model_registry();
    assert_eq!(models.default_name(), Some("charlie"));
}

#[test]
fn to_model_registry_with_default_overrides_first_registered() {
    let reg = registry_with_three_models_in_order();

    let models = reg
        .to_model_registry_with_default("bravo")
        .expect("bravo is registered");
    assert_eq!(models.default_name(), Some("bravo"));
    assert!(models.get("charlie").is_some());

    let err = reg
        .to_model_registry_with_default("not-registered")
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::ModelNotFound(name) if name == "not-registered"));
}

#[test]
fn capability_resolver_includes_names_and_aliases() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    let resolver = reg.capability_resolver();
    assert!(resolver.model_allowed("gpt-4o"));
    assert!(resolver.model_allowed("default"));
    assert!(resolver.tool_allowed("lookup_user"));
    assert!(!resolver.tool_allowed("unknown"));
}

#[test]
fn snapshot_lists_components_sorted_by_kind_and_name() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_router("classify").unwrap();

    let snapshot = reg.snapshot();
    assert_eq!(snapshot.len(), 3);
    assert_eq!(snapshot.count(ComponentKind::Model), 1);
    assert_eq!(snapshot.by_kind(ComponentKind::Tool)[0].id.0, "lookup_user");
    // Round-trips for audit logs / UIs.
    let json = serde_json::to_string(&snapshot).unwrap();
    let back: crate::RegistrySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, snapshot);
    // DOT export clusters every registered kind.
    let dot = snapshot.to_dot();
    assert!(dot.contains("digraph registry"));
    assert!(dot.contains("cluster_model"));
}

#[test]
fn snapshot_enumerates_aliases() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    let snapshot = reg.snapshot();
    assert_eq!(snapshot.aliases.len(), 1);
    assert_eq!(snapshot.aliases[0].alias, "default");
    assert_eq!(snapshot.aliases[0].canonical, "gpt-4o");
    assert_eq!(snapshot.aliases[0].kind, ComponentKind::Model);
    // Aliases survive serialization round-trips for audit logs.
    let json = serde_json::to_string(&snapshot).unwrap();
    let back: crate::RegistrySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, snapshot);
}

#[test]
fn diagnostics_flag_name_reused_across_kinds() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("shared", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_router("shared").unwrap();

    let diags = reg.diagnostics();
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].name, "shared");
    assert!(diags[0].message.contains("multiple kinds"));
}

#[test]
fn diagnostics_are_clean_for_a_healthy_registry() {
    // `alias()` is fail-closed against shadowing and dangling targets, so a
    // registry built through the public API always passes the integrity check.
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    assert!(reg.diagnostics().is_empty());
}

#[test]
fn set_metadata_replaces_the_recorded_description_and_tags() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();

    // Before `set_metadata`, registration only ever attached the bare
    // default `ComponentMetadata::new` — no description or tags.
    let before = reg.metadata(ComponentKind::Model, "gpt-4o").unwrap();
    assert!(before.description.is_none());
    assert!(before.tags.is_empty());

    let richer = crate::component::ComponentMetadata::new("gpt-4o", ComponentKind::Model)
        .with_description("OpenAI's flagship chat model")
        .with_tag("openai");
    reg.set_metadata(ComponentKind::Model, "gpt-4o", richer)
        .unwrap();

    let after = reg.metadata(ComponentKind::Model, "gpt-4o").unwrap();
    assert_eq!(
        after.description.as_deref(),
        Some("OpenAI's flagship chat model")
    );
    assert_eq!(after.tags, vec!["openai".to_string()]);
}

#[test]
fn set_metadata_rejects_an_unregistered_component() {
    let mut reg = CapabilityRegistry::<()>::new();
    let meta = crate::component::ComponentMetadata::new("ghost", ComponentKind::Model);
    let err = reg
        .set_metadata(ComponentKind::Model, "ghost", meta)
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}

#[test]
fn register_model_with_and_register_tool_with_attach_metadata_atomically() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model_with(
        "gpt-4o",
        Arc::new(FakeModel("m")),
        crate::component::ComponentMetadata::new("gpt-4o", ComponentKind::Model)
            .with_description("flagship"),
    )
    .unwrap();
    reg.register_tool_with(
        Arc::new(FakeTool("lookup_user")),
        crate::component::ComponentMetadata::new("lookup_user", ComponentKind::Tool)
            .with_tag("crm"),
    )
    .unwrap();

    assert_eq!(
        reg.metadata(ComponentKind::Model, "gpt-4o")
            .unwrap()
            .description
            .as_deref(),
        Some("flagship")
    );
    assert_eq!(
        reg.metadata(ComponentKind::Tool, "lookup_user")
            .unwrap()
            .tags,
        vec!["crm".to_string()]
    );
    // Registering the same name again is still rejected, exactly like the
    // bare `register_model`/`register_tool`.
    assert!(matches!(
        reg.register_model_with(
            "gpt-4o",
            Arc::new(FakeModel("m2")),
            crate::component::ComponentMetadata::new("gpt-4o", ComponentKind::Model),
        )
        .unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
}

#[test]
fn remove_makes_alias_shadows_component_and_dangling_alias_reachable() {
    // Before `remove`, `alias()`'s fail-closed checks make these two
    // diagnostics unreachable through the public API (only
    // `name_reused_across_kinds` could fire) — see W-I8.
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    // Removing the alias's target leaves a dangling alias.
    assert!(reg.remove(ComponentKind::Model, "gpt-4o"));
    let diags = reg.diagnostics();
    assert!(
        diags
            .iter()
            .any(|d| d.name == "default" && d.message.contains("not a registered")),
        "{diags:#?}"
    );

    // Removing something never registered is a no-op, not an error.
    assert!(!reg.remove(ComponentKind::Tool, "never-registered"));
}

#[test]
fn remove_drops_the_component_and_its_metadata() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    assert!(reg.has(ComponentKind::Model, "gpt-4o"));

    assert!(reg.remove(ComponentKind::Model, "gpt-4o"));
    assert!(!reg.has(ComponentKind::Model, "gpt-4o"));
    assert!(reg.metadata(ComponentKind::Model, "gpt-4o").is_none());
    assert!(reg.model("gpt-4o").is_none());

    // The name can be freely re-registered afterward.
    reg.register_model("gpt-4o", Arc::new(FakeModel("m2")))
        .unwrap();
    assert!(reg.has(ComponentKind::Model, "gpt-4o"));
}

#[tokio::test]
async fn capability_registry_implements_definition_registry() {
    use tinyagents_definition::{AgentDefinition, DefinitionRegistry};

    let mut reg = CapabilityRegistry::<()>::new();
    let parent = AgentDefinition::new("parent", "Parent", "delegates work")
        .with_subagents(["researcher", "writer"]);
    reg.register_agent(parent).unwrap();
    reg.register_agent(AgentDefinition::new(
        "researcher",
        "Researcher",
        "looks things up",
    ))
    .unwrap();

    // `resolve`.
    let found = DefinitionRegistry::resolve(&reg, "parent").await.unwrap();
    assert_eq!(found.unwrap().id, "parent");
    assert!(
        DefinitionRegistry::resolve(&reg, "ghost")
            .await
            .unwrap()
            .is_none()
    );

    // `list`.
    let all = DefinitionRegistry::list(&reg).await.unwrap();
    assert_eq!(all.len(), 2);

    // `delegates_for`.
    let delegates = DefinitionRegistry::delegates_for(&reg, "parent")
        .await
        .unwrap();
    assert_eq!(
        delegates,
        vec!["researcher".to_string(), "writer".to_string()]
    );
    assert!(
        DefinitionRegistry::delegates_for(&reg, "researcher")
            .await
            .unwrap()
            .is_empty()
    );
}
