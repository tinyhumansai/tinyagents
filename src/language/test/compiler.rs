//! Compiler tests: AST -> Blueprint lowering.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Compiler: AST -> Blueprint
// ---------------------------------------------------------------------------

#[test]
fn compiles_support_agent_blueprint() {
    let program = parse_str(SUPPORT_AGENT).unwrap();
    let blueprints = compile(&program).unwrap();
    assert_eq!(blueprints.len(), 1);
    let bp = &blueprints[0];

    assert_eq!(bp.graph_id, "support_agent");
    assert_eq!(bp.start, "agent");
    assert_eq!(bp.channels.len(), 2);
    assert_eq!(bp.defaults.len(), 3);
    assert_eq!(bp.nodes.len(), 2);

    let agent = &bp.nodes[0];
    assert_eq!(agent.kind, "agent");
    assert_eq!(agent.tools, vec!["lookup_user", "create_ticket"]);
    match &agent.routing {
        Routing::Conditional(routes) => {
            assert_eq!(routes.len(), 2);
            assert_eq!(routes[0], ("tool_call".into(), "tools".into()));
            assert_eq!(routes[1], ("final".into(), "END".into()));
        }
        other => panic!("expected conditional routing, got {other:?}"),
    }

    let tools = &bp.nodes[1];
    assert_eq!(tools.routing, Routing::Next("agent".into()));
}

#[test]
fn next_end_lowers_to_terminal() {
    let src = "graph g { start a node a { kind model next END } }";
    let bp = &compile(&parse_str(src).unwrap()).unwrap()[0];
    assert_eq!(bp.nodes[0].routing, Routing::Terminal);
}

#[test]
fn blueprint_round_trips_through_serde() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let json = serde_json::to_string(&bp).unwrap();
    let back: crate::language::types::Blueprint = serde_json::from_str(&json).unwrap();
    assert_eq!(bp, back);
}

#[test]
fn missing_start_is_a_compile_error() {
    let src = "graph g { node a { kind model } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
    assert!(err.to_string().contains("no `start`"), "{err}");
}

#[test]
fn start_not_defined_is_a_compile_error() {
    let src = "graph g { start missing node a { kind model } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("is not defined"), "{err}");
}

#[test]
fn duplicate_node_is_a_compile_error() {
    let src = "graph g { start a node a { kind model } node a { kind model } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("duplicate node"), "{err}");
}

#[test]
fn unknown_route_target_is_a_compile_error() {
    let src = "graph g { start a node a { routes { go -> ghost } } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("route target"), "{err}");
}

#[test]
fn unknown_next_target_is_a_compile_error() {
    let src = "graph g { start a node a { next ghost } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("next target"), "{err}");
}

#[test]
fn mixing_next_and_routes_is_a_compile_error() {
    let src = "graph g { start a node a { next b routes { x -> b } } node b { } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("mixes static routing"), "{err}");
}

#[test]
fn mixing_edge_and_routes_is_a_compile_error() {
    let src = "graph g { start a node a { routes { x -> b } } node b { } a -> b }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("mixes static routing"), "{err}");
}

#[test]
fn duplicate_route_label_is_a_compile_error() {
    let src = "graph g { start a node a { routes { x -> b\n x -> b } } node b { } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("duplicate route label"), "{err}");
}

#[test]
fn duplicate_channel_is_a_compile_error() {
    let src = "graph g { start a channel messages append channel messages messages node a { } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("duplicate channel"), "{err}");
}

#[test]
fn duplicate_graph_id_is_a_compile_error() {
    let src = "graph g { start a node a { } } graph g { start b node b { } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("duplicate graph"), "{err}");
}

#[test]
fn next_and_command_goto_conflict_is_a_compile_error() {
    let src = "graph g { start a node a { next b command { goto c } } node b { } node c { } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(
        err.to_string().contains("conflicting routing sources"),
        "{err}"
    );
}

#[test]
fn command_goto_and_edge_conflict_is_a_compile_error() {
    let src = "graph g { start a node a { command { goto b } } node b { } a -> b }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(
        err.to_string().contains("conflicting routing sources"),
        "{err}"
    );
}

#[test]
fn multiple_top_level_edges_from_same_source_is_a_compile_error() {
    let src = "graph g { start a node a { } node b { } node c { } a -> b a -> c }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(
        err.to_string().contains("multiple top-level edges"),
        "{err}"
    );
}
