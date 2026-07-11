//! Feature tests for the `.rag` parser surface
//! ([`tinyagents::language::parser`]).
//!
//! These cover structural parsing behaviours the existing suite leaves thin:
//! an empty program, multiple top-level graphs, the `system`/`prompt` alias, the
//! back-compat `parse(&tokens)` entry point (including its empty-slice guard),
//! and the distinct structural errors — an unknown node-item keyword, a missing
//! brace, a truncated block, and a stray token where a graph item is expected.

use tinyagents::TinyAgentsError;
use tinyagents::language::lexer::tokenize;
use tinyagents::language::parser::{parse, parse_str};
use tinyagents::language::types::{SpannedToken, Token};

#[test]
fn an_empty_source_parses_to_a_program_with_no_graphs() {
    let program = parse_str("").expect("empty source parses");
    assert!(program.graphs.is_empty());
}

#[test]
fn a_comment_only_source_parses_to_no_graphs() {
    let program = parse_str("// just a comment\n").expect("comment-only source parses");
    assert!(program.graphs.is_empty());
}

#[test]
fn multiple_top_level_graphs_parse_in_source_order() {
    let program = parse_str("graph first { start a node a {} } graph second { start b node b {} }")
        .expect("two graphs parse");
    assert_eq!(program.graphs.len(), 2);
    assert_eq!(program.graphs[0].name, "first");
    assert_eq!(program.graphs[1].name, "second");
}

#[test]
fn system_is_accepted_as_an_alias_for_prompt() {
    let via_system =
        parse_str(r#"graph g { start a node a { system "hi" } }"#).expect("system parses");
    let via_prompt =
        parse_str(r#"graph g { start a node a { prompt "hi" } }"#).expect("prompt parses");
    assert_eq!(via_system.graphs[0].nodes[0].prompt.as_deref(), Some("hi"));
    assert_eq!(
        via_system.graphs[0].nodes[0].prompt,
        via_prompt.graphs[0].nodes[0].prompt
    );
}

#[test]
fn channel_reducer_arguments_are_only_string_or_number_literals() {
    // A trailing bare identifier is NOT consumed as a channel arg; it begins the
    // next declaration instead. Here `node` starts a node declaration.
    let program = parse_str(r#"graph g { start a channel facts aggregate "agg" 3 node a {} }"#)
        .expect("channel args parse");
    let channel = &program.graphs[0].channels[0];
    assert_eq!(channel.name, "facts");
    assert_eq!(channel.reducer, "aggregate");
    assert_eq!(channel.args.len(), 2);
    assert_eq!(program.graphs[0].nodes.len(), 1);
}

#[test]
fn parse_over_a_token_slice_matches_parse_str() {
    let source = "graph g { start a node a { next END } }";
    let toks = tokenize(source).expect("lexes");
    let from_tokens = parse(&toks).expect("token slice parses");
    let from_str = parse_str(source).expect("string parses");
    assert_eq!(from_tokens, from_str);
}

#[test]
fn parse_rejects_an_empty_token_slice_instead_of_panicking() {
    let empty: Vec<SpannedToken> = Vec::new();
    let err = parse(&empty).expect_err("an empty slice violates the Eof contract");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("empty token stream"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_accepts_a_lone_eof_token_as_an_empty_program() {
    let eof = vec![SpannedToken {
        token: Token::Eof,
        span: tinyagents::Span::new(1, 1),
    }];
    let program = parse(&eof).expect("a lone Eof is a valid empty program");
    assert!(program.graphs.is_empty());
}

#[test]
fn an_unknown_node_item_keyword_is_a_structural_error() {
    let err = parse_str("graph g { start a node a { frobnicate 3 } }")
        .expect_err("`frobnicate` is not a node item");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unknown node item"), "{message}");
            assert!(message.contains("frobnicate"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn a_missing_opening_brace_reports_the_expected_token() {
    let err = parse_str("graph g start a").expect_err("graph body needs a brace");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains('{'), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn a_truncated_node_body_reports_unexpected_end_of_input() {
    let err = parse_str("graph g { start a node a {").expect_err("node body is unterminated");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unexpected end of input"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn a_route_without_a_target_is_a_structural_error() {
    let err = parse_str("graph g { start a node a { routes { ok -> } } }")
        .expect_err("route target is missing");
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn deeply_repeated_route_labels_parse_without_recursion_limits() {
    // Structural parsing admits many routes; semantic checks (duplicate labels)
    // are the compiler's job, so a large routes block still parses.
    let mut source = String::from("graph g { start a node a { routes {");
    for i in 0..200 {
        source.push_str(&format!(" r{i} -> END"));
    }
    source.push_str(" } } }");
    let program = parse_str(&source).expect("many routes parse");
    assert_eq!(program.graphs[0].nodes[0].routes.len(), 200);
}
