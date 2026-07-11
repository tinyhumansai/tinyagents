//! Feature tests for the compiler's semantic validation and lowering
//! ([`tinyagents::language::compiler`]).
//!
//! The existing suite covers missing `start`, a duplicate node, and one unknown
//! route target. These add the remaining semantic guards — duplicate graph and
//! channel names, an undefined `start`, unknown `command`/`send`/`join`
//! targets, duplicate route labels, and every conflicting-routing-source
//! combination — plus the lowering guarantees: the default `model` kind, and
//! that `next`/`command goto`/an edge to `END` each lower to a terminal.

use tinyagents::TinyAgentsError;
use tinyagents::language::compiler::compile;
use tinyagents::language::parser::parse_str;
use tinyagents::language::testkit;
use tinyagents::language::types::Routing;

/// Compiles `source` and returns the [`TinyAgentsError::Compile`] message,
/// panicking if the source unexpectedly compiles or fails another way.
fn compile_error(source: &str) -> String {
    let program = parse_str(source).expect("source should parse");
    match compile(&program).expect_err("source should not compile") {
        TinyAgentsError::Compile(message) => message,
        other => panic!("expected compile error, got {other:?}"),
    }
}

#[test]
fn duplicate_graph_names_are_rejected() {
    let message = compile_error("graph g { start a node a {} } graph g { start b node b {} }");
    assert!(message.contains("duplicate graph"), "{message}");
    assert!(message.contains('g'), "{message}");
}

#[test]
fn duplicate_channel_names_are_rejected() {
    let message = compile_error(
        "graph g { start a channel messages append channel messages overwrite node a {} }",
    );
    assert!(message.contains("duplicate channel"), "{message}");
    assert!(message.contains("messages"), "{message}");
}

#[test]
fn an_undefined_start_node_is_rejected() {
    let message = compile_error("graph g { start ghost node a {} }");
    assert!(message.contains("start node"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn a_duplicate_route_label_on_one_node_is_rejected() {
    let message = compile_error("graph g { start a node a { routes { ok -> END ok -> END } } }");
    assert!(message.contains("duplicate route label"), "{message}");
}

#[test]
fn an_unknown_command_goto_target_is_rejected() {
    let message = compile_error("graph g { start a node a { command { goto ghost } } }");
    assert!(message.contains("command goto target"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn an_unknown_send_target_is_rejected() {
    let message = compile_error("graph g { start a node a { sends [ send ghost ] } } ");
    assert!(message.contains("send target"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn an_unknown_top_level_join_source_is_rejected() {
    let message = compile_error("graph g { start a node a {} join [ghost] -> a }");
    assert!(message.contains("join source"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn an_unknown_top_level_join_target_is_rejected() {
    let message = compile_error("graph g { start a node a {} join [a] -> ghost }");
    assert!(message.contains("join target"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn an_unknown_join_node_source_is_rejected() {
    let message =
        compile_error("graph g { start a node a {} node j { kind join sources [ghost] } }");
    assert!(message.contains("join source"), "{message}");
    assert!(message.contains("ghost"), "{message}");
}

#[test]
fn multiple_top_level_edges_from_one_node_are_rejected() {
    let message = compile_error("graph g { start a node a {} node b {} node c {} a -> b a -> c }");
    assert!(message.contains("multiple top-level edges"), "{message}");
    assert!(message.contains('a'), "{message}");
}

#[test]
fn mixing_routes_with_next_is_rejected() {
    let message =
        compile_error("graph g { start a node a { routes { ok -> END } next b } node b {} }");
    assert!(message.contains("static routing"), "{message}");
}

#[test]
fn mixing_next_with_command_goto_is_rejected() {
    let message = compile_error(
        "graph g { start a node a { next b command { goto c } } node b {} node c {} }",
    );
    assert!(message.contains("conflicting routing sources"), "{message}");
}

#[test]
fn mixing_next_with_a_top_level_edge_is_rejected() {
    let message = compile_error("graph g { start a node a { next b } node b {} node c {} a -> c }");
    assert!(message.contains("conflicting routing sources"), "{message}");
}

#[test]
fn an_unspecified_node_kind_defaults_to_model() {
    let blueprint = testkit::blueprint("graph g { start a node a {} }");
    assert_eq!(testkit::node(&blueprint, "a").kind, "model");
}

#[test]
fn a_terminal_node_with_no_routing_source_routes_to_end() {
    let blueprint = testkit::blueprint("graph g { start a node a {} }");
    testkit::assert_terminal(&blueprint, "a");
}

#[test]
fn next_command_goto_and_edge_to_end_all_lower_to_terminal() {
    let via_next = testkit::blueprint("graph g { start a node a { next END } }");
    assert_eq!(testkit::node(&via_next, "a").routing, Routing::Terminal);

    let via_goto = testkit::blueprint("graph g { start a node a { command { goto END } } }");
    assert_eq!(testkit::node(&via_goto, "a").routing, Routing::Terminal);

    let via_edge = testkit::blueprint("graph g { start a node a {} a -> END }");
    assert_eq!(testkit::node(&via_edge, "a").routing, Routing::Terminal);
}

#[test]
fn a_command_goto_to_a_real_node_lowers_to_a_static_next() {
    let blueprint =
        testkit::blueprint("graph g { start a node a { command { goto b } } node b {} }");
    assert_eq!(
        testkit::node(&blueprint, "a").routing,
        Routing::Next("b".to_string())
    );
}

#[test]
fn two_graphs_in_one_program_each_compile_to_a_blueprint() {
    let blueprints = testkit::compile_all(
        "graph first { start a node a { next END } } graph second { start b node b { next END } }",
    );
    assert_eq!(blueprints.len(), 2);
    assert_eq!(blueprints[0].graph_id, "first");
    assert_eq!(blueprints[1].graph_id, "second");
}
