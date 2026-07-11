//! Feature tests for capability binding
//! ([`tinyagents::language::capability_resolver`]).
//!
//! The existing suite exercises the minimal model/tool gate and the
//! registry-backed happy/sad path. These add the strict [`bind_blueprint`] gate
//! for every reference class the language distinguishes — subgraph, router,
//! agent, script, and reducer references plus node-kind validation — the shared
//! [`classify_reference`] policy, and the deliberate blind spots of the minimal
//! [`bind_capabilities`] gate (it never inspects kinds, reducers, or subgraph
//! references).

use tinyagents::TinyAgentsError;
use tinyagents::language::capability_resolver::{
    CapabilityResolver, DEFAULT_NODE_KINDS, ReferenceClass, bind_capabilities,
};
use tinyagents::language::testkit;

/// A resolver with node-kind validation enabled (seeded with the default kinds).
fn strict_resolver() -> CapabilityResolver {
    CapabilityResolver::new().with_node_kinds(DEFAULT_NODE_KINDS.iter().copied())
}

/// Extracts a [`TinyAgentsError::Capability`] message.
fn capability_message(err: TinyAgentsError) -> String {
    match err {
        TinyAgentsError::Capability(message) => message,
        other => panic!("expected capability error, got {other:?}"),
    }
}

#[test]
fn classify_reference_maps_each_kind_to_its_reference_class() {
    let subgraph =
        CapabilityResolver::classify_reference("subgraph", None, Some("flow"), None, None)
            .expect("subgraph carries a reference");
    assert_eq!(subgraph.class, ReferenceClass::Subgraph);
    assert_eq!(subgraph.target, "flow");

    let router = CapabilityResolver::classify_reference("router", Some("r"), None, None, None)
        .expect("router carries a reference");
    assert_eq!(router.class, ReferenceClass::Router);

    let agent = CapabilityResolver::classify_reference("subagent", None, None, Some("a"), None)
        .expect("subagent carries a reference");
    assert_eq!(agent.class, ReferenceClass::Agent);

    let script = CapabilityResolver::classify_reference("repl_agent", None, None, None, Some("s"))
        .expect("repl_agent carries a reference");
    assert_eq!(script.class, ReferenceClass::Script);

    // An unknown kind falls through to a model check, mirroring the compiler
    // default of an unspecified kind being `model`.
    let fallthrough = CapabilityResolver::classify_reference("agent", Some("m"), None, None, None)
        .expect("agent falls through to a model reference");
    assert_eq!(fallthrough.class, ReferenceClass::Model);

    // A model kind with no model field carries no primary reference at all.
    assert!(CapabilityResolver::classify_reference("model", None, None, None, None).is_none());
}

#[test]
fn reference_class_words_are_stable() {
    assert_eq!(ReferenceClass::Model.word(), "model");
    assert_eq!(ReferenceClass::Subgraph.word(), "subgraph");
    assert_eq!(ReferenceClass::Router.word(), "router");
    assert_eq!(ReferenceClass::Agent.word(), "agent");
    assert_eq!(ReferenceClass::Script.word(), "script");
}

#[test]
fn node_kind_validation_is_disabled_when_the_allowlist_is_empty() {
    // A resolver with no node kinds allows any kind (the legacy manual gate).
    assert!(CapabilityResolver::new().node_kind_allowed("anything_at_all"));
    // A seeded resolver rejects unknown kinds.
    assert!(strict_resolver().node_kind_allowed("agent"));
    assert!(!strict_resolver().node_kind_allowed("wizard"));
}

#[test]
fn strict_binding_rejects_an_unknown_node_kind_as_a_compile_error() {
    let blueprint = testkit::blueprint("graph g { start a node a { kind wizard next END } }");
    let err = strict_resolver()
        .bind_blueprint(&blueprint)
        .expect_err("`wizard` is not a known kind");
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(message.contains("unknown kind"), "{message}");
            assert!(message.contains("wizard"), "{message}");
        }
        other => panic!("expected compile error, got {other:?}"),
    }
}

#[test]
fn strict_binding_resolves_a_subgraph_reference() {
    let blueprint =
        testkit::blueprint(r#"graph g { start a node a { kind subgraph graph "flow" next END } }"#);
    let resolver = strict_resolver().allow_subgraph("flow");
    resolver
        .bind_blueprint(&blueprint)
        .expect("registered subgraph resolves");

    let err = strict_resolver()
        .bind_blueprint(&blueprint)
        .expect_err("unregistered subgraph fails");
    let message = capability_message(err);
    assert!(message.contains("subgraph"), "{message}");
    assert!(message.contains("flow"), "{message}");
}

#[test]
fn strict_binding_resolves_a_router_reference() {
    let blueprint =
        testkit::blueprint(r#"graph g { start a node a { kind router model "pick" next END } }"#);
    strict_resolver()
        .allow_router("pick")
        .bind_blueprint(&blueprint)
        .expect("registered router resolves");

    let message = capability_message(
        strict_resolver()
            .bind_blueprint(&blueprint)
            .expect_err("unregistered router fails"),
    );
    assert!(message.contains("router"), "{message}");
}

#[test]
fn strict_binding_resolves_an_agent_reference() {
    let blueprint = testkit::blueprint(
        r#"graph g { start a node a { kind subagent agent "researcher" next END } }"#,
    );
    strict_resolver()
        .allow_agent("researcher")
        .bind_blueprint(&blueprint)
        .expect("registered agent resolves");

    let message = capability_message(
        strict_resolver()
            .bind_blueprint(&blueprint)
            .expect_err("unregistered agent fails"),
    );
    assert!(message.contains("agent"), "{message}");
    assert!(message.contains("researcher"), "{message}");
}

#[test]
fn strict_binding_resolves_a_repl_script_reference() {
    let blueprint = testkit::blueprint(
        r#"graph g { start a node a { kind repl_agent script "triage" next END } }"#,
    );
    strict_resolver()
        .allow_script("triage")
        .bind_blueprint(&blueprint)
        .expect("registered script resolves");

    let message = capability_message(
        strict_resolver()
            .bind_blueprint(&blueprint)
            .expect_err("unregistered script fails"),
    );
    assert!(message.contains("script"), "{message}");
}

#[test]
fn strict_binding_checks_channel_reducers() {
    let blueprint = testkit::blueprint("graph g { start a channel messages append node a {} }");
    strict_resolver()
        .allow_reducer("append")
        .bind_blueprint(&blueprint)
        .expect("registered reducer resolves");

    let message = capability_message(
        strict_resolver()
            .bind_blueprint(&blueprint)
            .expect_err("unregistered reducer fails"),
    );
    assert!(message.contains("reducer"), "{message}");
    assert!(message.contains("append"), "{message}");
}

#[test]
fn minimal_gate_checks_only_models_and_tools() {
    // A subagent node references an unknown agent and an unknown reducer, but the
    // minimal `bind_capabilities` gate inspects neither — it only checks the
    // `model`/`tool` fields, which this node does not populate — so it passes.
    let blueprint = testkit::blueprint(
        r#"graph g { start a channel messages append node a { kind subagent agent "ghost" next END } }"#,
    );
    bind_capabilities(&blueprint, &CapabilityResolver::new())
        .expect("minimal gate ignores agent and reducer references");
}

#[test]
fn minimal_gate_rejects_the_first_unknown_tool() {
    let blueprint = testkit::blueprint(
        r#"graph g { start a node a { model "m" tools ["known", "unknown"] next END } }"#,
    );
    let resolver = CapabilityResolver::new()
        .allow_model("m")
        .allow_tool("known");
    let message = capability_message(
        bind_capabilities(&blueprint, &resolver).expect_err("`unknown` tool is not allowed"),
    );
    assert!(message.contains("unknown"), "{message}");
}
