//! TRUE end-to-end: the registry → language binding gate.
//!
//! This composes the **registry** ([`CapabilityRegistry`]) with the
//! **language** compiler ([`compile_source`]). A `CapabilityRegistry` is built
//! up with a real model, real tools, a registered subgraph [`Blueprint`], and
//! named reducers; then a real `.rag` source is compiled *through the registry*
//! so that every model/tool/subgraph/reducer reference in the source is
//! validated against exactly what Rust registered.
//!
//! The test asserts the produced [`Blueprint`] (topology, node kinds, model and
//! tool bindings, channels) and that an unregistered capability reference is
//! rejected with [`TinyAgentsError::Capability`]. Everything is offline: the
//! registered model is never invoked — registration only proves the *name* is
//! bound, which is what the binding gate checks.

use std::sync::Arc;

use tinyagents::harness::testkit::{FakeTool, ScriptedModel};
use tinyagents::language::compiler::{compile, compile_source};
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::Routing;
use tinyagents::{CapabilityRegistry, ComponentKind, TinyAgentsError};

/// A retrieval subgraph the main flow references by name through a `subgraph`
/// node. Compiled independently and registered as a graph blueprint.
const RETRIEVAL: &str = r#"
graph retrieval {
  start search
  node search {
    kind model
    model "default"
    next END
  }
}
"#;

/// The main flow: a subgraph node, an agent tool loop, and two channels whose
/// reducers must be registered.
const SUPPORT_FLOW: &str = r#"
graph support_flow {
  start retrieve

  channel messages append
  channel summary last_write

  node retrieve {
    kind subgraph
    model "retrieval"
    next agent
  }

  node agent {
    kind agent
    model "default"
    tools ["lookup_user", "create_ticket"]
    routes {
      tool_call -> tools
      final -> END
    }
  }

  node tools {
    kind tool_executor
    next agent
  }
}
"#;

/// Builds a registry populated with a model, two tools, the retrieval subgraph
/// blueprint, and the two reducers referenced by `SUPPORT_FLOW`.
fn populated_registry() -> CapabilityRegistry<()> {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();

    // A real model handle (never invoked here — registration binds the name).
    registry
        .register_model("default", Arc::new(ScriptedModel::new(vec![])))
        .expect("model registers");

    // Real tools.
    registry
        .register_tool(Arc::new(FakeTool::returning("lookup_user", "Ada Lovelace")))
        .expect("lookup_user registers")
        .register_tool(Arc::new(FakeTool::returning("create_ticket", "TICKET-1")))
        .expect("create_ticket registers");

    // A registered subgraph blueprint, compiled from real source.
    let retrieval = compile(&parse_str(RETRIEVAL).expect("retrieval parses"))
        .expect("retrieval compiles")
        .remove(0);
    registry
        .register_graph_blueprint("retrieval", retrieval)
        .expect("subgraph registers");

    // Named reducers the channels bind to.
    registry
        .register_reducer("append")
        .expect("append reducer registers")
        .register_reducer("last_write")
        .expect("last_write reducer registers");

    registry
}

#[test]
fn compiles_rag_source_through_a_populated_registry() {
    let registry = populated_registry();

    // Presence checks across kinds, including the subgraph and reducers.
    assert!(registry.has(ComponentKind::Model, "default"));
    assert!(registry.has(ComponentKind::Tool, "lookup_user"));
    assert!(registry.has(ComponentKind::Graph, "retrieval"));
    assert!(registry.has(ComponentKind::Reducer, "append"));
    assert!(registry.has(ComponentKind::Reducer, "last_write"));

    // The registry-backed path: parse + compile + bind against the registry.
    let mut blueprints = compile_source(SUPPORT_FLOW, &registry).expect("source compiles + binds");
    assert_eq!(blueprints.len(), 1);
    let blueprint = blueprints.remove(0);

    // Topology.
    assert_eq!(blueprint.graph_id, "support_flow");
    assert_eq!(blueprint.start, "retrieve");
    assert_eq!(blueprint.nodes.len(), 3);

    // Channels bound their reducers.
    let channels: Vec<(&str, &str)> = blueprint
        .channels
        .iter()
        .map(|c| (c.name.as_str(), c.reducer.as_str()))
        .collect();
    assert_eq!(
        channels,
        vec![("messages", "append"), ("summary", "last_write")]
    );

    // The subgraph node references the registered blueprint and flows to `agent`.
    let retrieve = &blueprint.nodes[0];
    assert_eq!(retrieve.name, "retrieve");
    assert_eq!(retrieve.kind, "subgraph");
    assert_eq!(retrieve.model.as_deref(), Some("retrieval"));
    assert_eq!(retrieve.routing, Routing::Next("agent".to_string()));

    // The agent node binds the model + both tools and routes conditionally.
    let agent = &blueprint.nodes[1];
    assert_eq!(agent.name, "agent");
    assert_eq!(agent.kind, "agent");
    assert_eq!(agent.model.as_deref(), Some("default"));
    assert_eq!(agent.tools, vec!["lookup_user", "create_ticket"]);
    match &agent.routing {
        Routing::Conditional(routes) => {
            assert_eq!(routes.len(), 2);
            assert_eq!(routes[0], ("tool_call".to_string(), "tools".to_string()));
            assert_eq!(routes[1], ("final".to_string(), "END".to_string()));
        }
        other => panic!("expected conditional routing, got {other:?}"),
    }

    // The tool executor loops back to the agent.
    let tools = &blueprint.nodes[2];
    assert_eq!(tools.name, "tools");
    assert_eq!(tools.kind, "tool_executor");
    assert_eq!(tools.routing, Routing::Next("agent".to_string()));
}

#[test]
fn unregistered_capability_is_rejected() {
    let registry = populated_registry();

    // A source referencing a tool that was never registered.
    const UNREGISTERED_TOOL: &str = r#"
graph bad_flow {
  start agent
  node agent {
    kind agent
    model "default"
    tools ["delete_database"]
    next END
  }
}
"#;

    let err = compile_source(UNREGISTERED_TOOL, &registry)
        .expect_err("unregistered tool must be rejected");
    match err {
        TinyAgentsError::Capability(msg) => {
            assert!(
                msg.contains("delete_database"),
                "error should name the missing capability: {msg}"
            );
        }
        other => panic!("expected Capability error, got {other:?}"),
    }

    // And an unregistered model is rejected too.
    const UNREGISTERED_MODEL: &str = r#"
graph bad_model {
  start agent
  node agent {
    kind agent
    model "ghost_model"
    next END
  }
}
"#;

    let err = compile_source(UNREGISTERED_MODEL, &registry)
        .expect_err("unregistered model must be rejected");
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}
