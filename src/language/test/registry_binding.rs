//! Registry-backed capability-binding tests (registry -> language
//! binding via `bind_capabilities_with_registry`/`compile_source`).
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Registry-backed capability binding (registry → language binding)
// ---------------------------------------------------------------------------

use crate::language::capability_resolver::{DEFAULT_NODE_KINDS, bind_capabilities_with_registry};
use crate::language::compiler::compile_source;
use crate::registry::CapabilityRegistry;

/// A `.rag` graph that exercises every registry-backed reference kind: a model,
/// a tool, a subgraph reference (`kind subgraph` whose `model` names a
/// registered blueprint), a router reference (`kind router`), and a channel
/// reducer.
pub(super) const FULL_SOURCE: &str = r#"
graph main {
  start agent

  channel messages append

  node agent {
    kind agent
    model "default"
    tools ["lookup_user"]
    routes {
      retrieve -> sub
      classify -> route
      done -> END
    }
  }

  node sub {
    kind subgraph
    model "retrieval"
    next END
  }

  node route {
    kind router
    model "classify"
    next END
  }
}
"#;

/// Builds a registry that satisfies every reference in [`FULL_SOURCE`].
pub(super) fn full_registry() -> CapabilityRegistry<TestState> {
    let mut reg = CapabilityRegistry::<TestState>::new();
    reg.register_model(
        "default",
        std::sync::Arc::new(crate::language::test::provenance_diff_testkit::testkit::EchoModel),
    )
    .unwrap();
    reg.register_tool(std::sync::Arc::new(
        crate::language::test::provenance_diff_testkit::testkit::NoopTool,
    ))
    .unwrap();
    reg.register_graph_blueprint(
        "retrieval",
        compile(&parse_str("graph retrieval { start r node r { kind model next END } }").unwrap())
            .unwrap()
            .remove(0),
    )
    .unwrap();
    reg.register_router("classify").unwrap();
    reg.register_reducer("append").unwrap();
    reg
}

#[test]
fn compile_source_binds_against_registry() {
    let reg = full_registry();
    let blueprints = compile_source(FULL_SOURCE, &reg).unwrap();
    assert_eq!(blueprints.len(), 1);
    assert_eq!(blueprints[0].graph_id, "main");
}

#[test]
fn registry_resolver_allows_all_kinds() {
    let reg = full_registry();
    let resolver = reg.capability_resolver();
    assert!(resolver.model_allowed("default"));
    assert!(resolver.tool_allowed("lookup_user"));
    assert!(resolver.subgraph_allowed("retrieval"));
    assert!(resolver.router_allowed("classify"));
    assert!(resolver.reducer_allowed("append"));
    for kind in DEFAULT_NODE_KINDS {
        assert!(resolver.node_kind_allowed(kind));
    }
}

#[test]
fn registry_bind_rejects_unregistered_model() {
    let mut reg = full_registry();
    reg.replace_model(
        "other",
        std::sync::Arc::new(crate::language::test::provenance_diff_testkit::testkit::EchoModel),
    );
    // Source references `default`, which we did not register here.
    let mut bare = CapabilityRegistry::<TestState>::new();
    bare.register_tool(std::sync::Arc::new(
        crate::language::test::provenance_diff_testkit::testkit::NoopTool,
    ))
    .unwrap();
    bare.register_graph_blueprint(
        "retrieval",
        reg.graph_blueprint("retrieval").unwrap().clone(),
    )
    .unwrap();
    bare.register_router("classify").unwrap();
    bare.register_reducer("append").unwrap();
    let err = compile_source(FULL_SOURCE, &bare).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown model"), "{err}");
}

#[test]
fn registry_bind_rejects_unregistered_tool() {
    let src = r#"graph g { start a channel m append node a { kind agent model "default" tools ["missing"] next END } }"#;
    let reg = full_registry();
    let err = compile_source(src, &reg).unwrap_err();
    assert!(err.to_string().contains("unknown tool"), "{err}");
}

#[test]
fn registry_bind_rejects_unregistered_subgraph() {
    let src = r#"graph g { start s node s { kind subgraph model "ghost" next END } }"#;
    let reg = full_registry();
    let err = compile_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown subgraph"), "{err}");
}

#[test]
fn registry_bind_rejects_unregistered_router() {
    let src = r#"graph g { start r node r { kind router model "ghost" next END } }"#;
    let reg = full_registry();
    let err = compile_source(src, &reg).unwrap_err();
    assert!(err.to_string().contains("unknown router"), "{err}");
}

#[test]
fn registry_bind_rejects_unregistered_reducer() {
    let src = r#"graph g { start a channel messages ghost node a { kind model next END } }"#;
    let reg = full_registry();
    let err = compile_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown reducer"), "{err}");
}

#[test]
fn registry_bind_rejects_unknown_node_kind() {
    let src = r#"graph g { start a node a { kind wizard next END } }"#;
    let reg = full_registry();
    let err = compile_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
    assert!(err.to_string().contains("unknown kind"), "{err}");
}

#[test]
fn manual_bind_path_ignores_kinds_and_reducers() {
    // The legacy manual resolver must keep working: a non-empty node kind set is
    // never consulted, and reducers/subgraphs are not checked.
    let src = r#"graph g { start a channel messages ghost node a { kind wizard model "default" next END } }"#;
    let bp = compile(&parse_str(src).unwrap()).unwrap().remove(0);
    let resolver = CapabilityResolver::new().allow_model("default");
    // Manual gate only checks model + tool; passes despite the unknown kind,
    // unknown reducer, and exotic node kind.
    bind_capabilities(&bp, &resolver).unwrap();
}

#[test]
fn bind_capabilities_with_registry_matches_compile_source() {
    let reg = full_registry();
    let bp = compile(&parse_str(FULL_SOURCE).unwrap()).unwrap().remove(0);
    bind_capabilities_with_registry(&bp, &reg).unwrap();
}
