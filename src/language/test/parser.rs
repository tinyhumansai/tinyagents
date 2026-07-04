//! Parser tests.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[test]
fn parses_support_agent_into_ast() {
    let program = parse_str(SUPPORT_AGENT).unwrap();
    assert_eq!(program.graphs.len(), 1);
    let graph = &program.graphs[0];

    assert_eq!(graph.name, "support_agent");
    assert_eq!(graph.start.as_deref(), Some("agent"));
    assert_eq!(graph.channels.len(), 2);
    assert_eq!(graph.channels[0].name, "messages");
    assert_eq!(graph.channels[0].reducer, "messages");
    assert_eq!(graph.channels[1].reducer, "append");

    // Defaults preserve declared order and literal kinds.
    assert_eq!(graph.defaults.len(), 3);
    assert_eq!(graph.defaults[0].0, "recursion_limit");
    assert_eq!(graph.defaults[0].1, Literal::Num(50.0));
    assert_eq!(graph.defaults[1].1, Literal::Str("exponential".into()));
    assert_eq!(graph.defaults[2].1, Literal::Ident("inherit".into()));

    assert_eq!(graph.nodes.len(), 2);
    let agent = &graph.nodes[0];
    assert_eq!(agent.kind.as_deref(), Some("agent"));
    assert_eq!(agent.model.as_deref(), Some("default"));
    assert_eq!(agent.tools, vec!["lookup_user", "create_ticket"]);
    assert_eq!(agent.routes.len(), 2);
    assert_eq!(agent.routes[0].label, "tool_call");
    assert_eq!(agent.routes[0].target, "tools");
    assert_eq!(agent.routes[1].target, "END");

    let tools = &graph.nodes[1];
    assert_eq!(tools.kind.as_deref(), Some("tool_executor"));
    assert_eq!(tools.next.as_deref(), Some("agent"));
}

#[test]
fn parses_top_level_edge() {
    let src = "graph g { start a node a { } node b { } a -> b b -> END }";
    let program = parse_str(src).unwrap();
    let graph = &program.graphs[0];
    assert_eq!(graph.edges.len(), 2);
    assert_eq!(graph.edges[0].from, "a");
    assert_eq!(graph.edges[0].to, "b");
    assert_eq!(graph.edges[1].to, "END");
}

#[test]
fn parse_reports_unexpected_token() {
    // Missing graph name.
    let tokens = tokenize("graph { }").unwrap();
    let err = parse(&tokens).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("expected identifier"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_rejects_unknown_node_item() {
    let src = "graph g { start a node a { bogus x } }";
    let err = parse_str(src).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unknown node item"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}
