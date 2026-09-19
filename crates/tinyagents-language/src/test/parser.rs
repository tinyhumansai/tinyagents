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
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
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
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unknown node item"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_rejects_token_stream_missing_eof_sentinel_instead_of_hanging() {
    // The lexer always terminates a stream with `Eof`; the cursor helpers in
    // `Parser` rely on that sentinel to guarantee forward progress. A caller
    // that slices off the trailing `Eof` (e.g. after filtering/truncating a
    // token stream) must get a parse error, not an infinite loop.
    let tokens = tokenize("graph g { start").unwrap();
    assert!(matches!(tokens.last().unwrap().token, Token::Eof));
    let truncated = &tokens[..tokens.len() - 1];
    assert!(!matches!(truncated.last().unwrap().token, Token::Eof));

    let err = parse(truncated).unwrap_err();
    match err {
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("end-of-input sentinel"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn duplicate_node_item_is_a_parse_error() {
    // M1: duplicate single-value node items are a diagnostic, not last-wins.
    let src = r#"graph g { start a node a { model "one" model "two" next END } }"#;
    let err = parse_str(src).unwrap_err();
    match err {
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("duplicate `model`"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn duplicate_graph_start_is_a_parse_error() {
    let src = "graph g { start a start b node a { next END } }";
    let err = parse_str(src).unwrap_err();
    match err {
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("duplicate `start`"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn prompt_and_system_alias_still_count_as_the_same_duplicate_field() {
    let src = r#"graph g { start a node a { prompt "one" system "two" next END } }"#;
    let err = parse_str(src).unwrap_err();
    match err {
        tinyagents_harness::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("duplicate `prompt`"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn sends_block_requires_a_comma_between_entries() {
    // M2: `sends` now shares the "comma-separated, optional trailing comma"
    // rule with `parse_ident_list`/`parse_string_list`, instead of making the
    // separator optional even between entries.
    let src = r#"graph g { start a node a { sends [ send b "x" send c "y" ] next END } node b { next END } node c { next END } }"#;
    assert!(parse_str(src).is_err());

    let with_comma = r#"graph g { start a node a { sends [ send b "x", send c "y" ] next END } node b { next END } node c { next END } }"#;
    assert!(parse_str(with_comma).is_ok());

    // A trailing comma after the last entry is still allowed.
    let trailing =
        r#"graph g { start a node a { sends [ send b "x", ] next END } node b { next END } }"#;
    assert!(parse_str(trailing).is_ok());
}

#[test]
fn parses_a_dedicated_router_item() {
    // M4: `router "name"` is a dedicated item, parallel to `agent`/`graph`/`script`.
    let src = r#"graph g { start a node a { kind router router "classify" } }"#;
    let program = parse_str(src).unwrap();
    let node = &program.graphs[0].nodes[0];
    assert_eq!(node.router.as_deref(), Some("classify"));
    assert!(node.model.is_none());
}

#[test]
fn parses_true_and_false_as_boolean_literals() {
    // M9: `Literal::Bool`, not `Literal::Ident("true"/"false")`.
    let src = "graph g { start a defaults { streaming true retryable false } node a { next END } }";
    let program = parse_str(src).unwrap();
    let defaults = &program.graphs[0].defaults;
    assert_eq!(defaults[0], ("streaming".to_string(), Literal::Bool(true)));
    assert_eq!(defaults[1], ("retryable".to_string(), Literal::Bool(false)));
    assert_eq!(Literal::Bool(true).as_display(), "true");
}
