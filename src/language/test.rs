//! Tests for the expressive language pipeline: lexer, parser, compiler,
//! capability binding, and graph materialisation.

use std::sync::Arc;

use crate::graph::{Command, NodeContext, NodeFuture, NodeResult};
use crate::language::compiler::{
    BoxedNode, CapabilityResolver, NodeFactory, bind_capabilities, build_graph, compile,
};
use crate::language::lexer::tokenize;
use crate::language::parser::{parse, parse_str};
use crate::language::types::{Literal, NodeSpec, Routing, Token};

/// The `support_agent` fixture from the module spec: an agent node with a tool
/// loop plus conditional routing to `END`.
const SUPPORT_AGENT: &str = r#"
// A support workflow with a tool loop.
graph support_agent {
  start agent

  defaults {
    recursion_limit 50
    backoff "exponential"
    checkpoint inherit
  }

  channel messages messages
  channel tool_calls append

  node agent {
    kind agent
    model "default"
    system "Resolve support requests using tools when useful."
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

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[test]
fn tokenizes_punctuation_and_arrow() {
    let tokens = tokenize("a -> b { } [ ] ,").unwrap();
    let kinds: Vec<_> = tokens.into_iter().map(|t| t.token).collect();
    assert_eq!(
        kinds,
        vec![
            Token::Ident("a".into()),
            Token::Arrow,
            Token::Ident("b".into()),
            Token::LBrace,
            Token::RBrace,
            Token::LBracket,
            Token::RBracket,
            Token::Comma,
            Token::Eof,
        ]
    );
}

#[test]
fn tokenizes_strings_numbers_and_comments() {
    let tokens = tokenize("// comment\n\"hi\\n\" 50 1.5 -3").unwrap();
    let kinds: Vec<_> = tokens.into_iter().map(|t| t.token).collect();
    assert_eq!(
        kinds,
        vec![
            Token::Str("hi\n".into()),
            Token::Num(50.0),
            Token::Num(1.5),
            Token::Num(-3.0),
            Token::Eof,
        ]
    );
}

#[test]
fn tracks_line_and_column_spans() {
    let tokens = tokenize("graph\n  foo").unwrap();
    assert_eq!(tokens[0].span.line, 1);
    assert_eq!(tokens[0].span.column, 1);
    assert_eq!(tokens[1].span.line, 2);
    assert_eq!(tokens[1].span.column, 3);
}

#[test]
fn unterminated_string_is_a_parse_error() {
    let err = tokenize("\"oops").unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Parse { .. }));
}

#[test]
fn invalid_escape_is_a_parse_error() {
    let err = tokenize("\"bad\\x\"").unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Parse { .. }));
}

#[test]
fn literal_as_display_does_not_saturate_huge_floats() {
    // A huge finite float must not be truncated to i64::MAX; it should render
    // using the float's own formatting instead.
    let huge = Literal::Num(1e30);
    assert_eq!(huge.as_display(), format!("{}", 1e30_f64));
    assert_ne!(huge.as_display(), format!("{}", i64::MAX));

    let nan = Literal::Num(f64::NAN);
    assert_eq!(nan.as_display(), "NaN");

    let inf = Literal::Num(f64::INFINITY);
    assert_eq!(inf.as_display(), "inf");
}

// ---------------------------------------------------------------------------
// Spans, source map, and diagnostics
// ---------------------------------------------------------------------------

use crate::language::diagnostic::{Diagnostic, Severity};
use crate::language::source::{SourceFile, SourceMap};
use crate::language::span::Span;

#[test]
fn span_merge_covers_both_inputs() {
    let a = Span::at(2, 5, 1, 3);
    let b = Span::at(10, 14, 2, 1);
    let merged = a.merge(b);
    assert_eq!(merged.start, 2);
    assert_eq!(merged.end, 14);
    // Anchor comes from the earlier-starting span.
    assert_eq!((merged.line, merged.column), (1, 3));
    // Merge is commutative over the covered range.
    assert_eq!(b.merge(a).start, 2);
    assert_eq!(b.merge(a).end, 14);
}

#[test]
fn span_len_and_is_empty() {
    assert!(Span::new(1, 1).is_empty());
    let s = Span::at(4, 9, 1, 5);
    assert_eq!(s.len(), 5);
    assert!(!s.is_empty());
}

#[test]
fn source_file_maps_offsets_to_line_and_column() {
    let file = SourceFile::new("demo.rag", "graph g\n  node a\n");
    // `g` is on line 1.
    assert_eq!(file.location(6), (1, 7));
    // The `node` keyword starts at byte 10 on line 2, column 3.
    let node_byte = file.text().find("node").unwrap();
    assert_eq!(file.location(node_byte), (2, 3));
    assert_eq!(file.line_text(2), Some("  node a"));
    assert_eq!(
        file.snippet(Span::at(node_byte, node_byte + 4, 2, 3)),
        "node"
    );
}

#[test]
fn source_map_assigns_ids_and_resolves_files() {
    let mut map = SourceMap::new();
    assert!(map.is_empty());
    let a = map.add("a.rag", "graph a {}");
    let b = map.add("b.rag", "graph b {}");
    assert_eq!(map.len(), 2);
    assert_ne!(a, b);
    assert_eq!(map.get(a).unwrap().name(), "a.rag");
    assert_eq!(map.get(b).unwrap().text(), "graph b {}");
}

#[test]
fn diagnostic_renders_caret_under_primary_span() {
    let source = "graph g {\n  tool_call -> toolz\n}\n";
    let file = SourceFile::new("support.rag", source);
    let target = source.find("toolz").unwrap();
    let span = Span::at(target, target + "toolz".len(), 2, 16);
    let rendered = Diagnostic::error("route target `toolz` does not exist", span)
        .with_code("E-rag-unknown-node")
        .with_primary_label("unknown node")
        .with_help("did you mean `tools`?")
        .render(&file);

    assert!(
        rendered.contains("error[E-rag-unknown-node]: route target `toolz` does not exist"),
        "{rendered}"
    );
    assert!(rendered.contains("--> support.rag:2:16"), "{rendered}");
    assert!(rendered.contains("tool_call -> toolz"), "{rendered}");
    // Five carets under the five characters of `toolz`, plus the label.
    assert!(rendered.contains("^^^^^ unknown node"), "{rendered}");
    assert!(
        rendered.contains("help: did you mean `tools`?"),
        "{rendered}"
    );
}

#[test]
fn diagnostic_renders_span_past_end_of_source_without_panic() {
    // A span whose bytes extend past (or start past) the end of the source must
    // not panic when rendered — the caret range is clamped into the line.
    let source = "graph g {}\n";
    let file = SourceFile::new("plan.rag", source);
    let past = source.len() + 50;
    let span = Span::at(past, past + 10, 99, 1);
    let rendered = Diagnostic::error("dangling span", span)
        .with_primary_label("here")
        .render(&file);
    assert!(rendered.contains("error: dangling span"), "{rendered}");
    // At least one caret is emitted even for an empty clamped range.
    assert!(rendered.contains('^'), "{rendered}");
}

#[test]
fn severity_labels_are_lowercase() {
    assert_eq!(Severity::Error.label(), "error");
    assert_eq!(Severity::Warning.label(), "warning");
    assert_eq!(Severity::Note.label(), "note");
}

#[test]
fn parse_error_carries_rendered_caret_for_source() {
    // `bogus` is not a valid node item; `parse_str` has the source so the error
    // message should render a caret beneath the offending token.
    let err = parse_str("graph g {\n  node a { bogus x }\n}\n").unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse {
            message,
            line,
            column,
        } => {
            assert!(message.contains("unknown node item `bogus`"), "{message}");
            assert!(message.contains('^'), "{message}");
            assert!(message.contains("--> <source>:2:12"), "{message}");
            assert_eq!((line, column), (2, 12));
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_error_without_source_renders_plain() {
    // The token-only `parse` entry point has no source text, so the rendered
    // message falls back to the source-free presentation (no caret).
    let tokens = tokenize("graph { }").unwrap();
    let err = parse(&tokens).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("expected identifier"), "{message}");
            assert!(!message.contains('^'), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_empty_token_slice_returns_error_not_panic() {
    // A well-formed token stream always ends with an `Eof` sentinel; an empty
    // slice violates that contract and previously underflowed `len() - 1`.
    // `parse` must return a parse error instead of panicking.
    let err = parse(&[]).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("empty token stream"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

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

// ---------------------------------------------------------------------------
// Extended grammar (H2): channels+policy, command, send/join, subgraph,
// subagent, repl_agent, interrupt, io shape, checkpoint/interrupt policy.
// ---------------------------------------------------------------------------

/// A graph exercising every H2 primitive in one declarative blueprint.
const EXTENDED: &str = r#"
graph orchestrator {
  start planner

  input {
    request string
    customer_id string
  }
  output {
    answer string
  }

  checkpoint inherit
  interrupt manual

  channel messages messages
  channel usage aggregate "usage_delta"
  channel arrivals barrier 2

  node planner {
    kind agent
    model "default"
    command {
      goto fanout
      update {
        status "planned"
      }
    }
  }

  node fanout {
    kind model
    sends [
      send worker_a "split_a"
      send worker_b "split_b"
    ]
    next worker_a
  }

  node worker_a {
    kind model
    next gather
  }

  node worker_b {
    kind model
    next gather
  }

  node gather {
    kind join
    sources [worker_a, worker_b]
    next research
  }

  node research {
    kind subagent
    agent "researcher"
    input "topic"
    timeout 30
    retry {
      max_attempts 3
      backoff "exponential"
    }
    next sub
  }

  node sub {
    kind subgraph
    graph "summarize"
    next triage
  }

  node triage {
    kind repl_agent
    model "default"
    script "triage_script"
    next review
  }

  node review {
    kind interrupt
    prompt "Approve?"
    options ["approve", "reject"]
    routes {
      approve -> END
      reject -> planner
    }
  }

  join [worker_a, worker_b] -> gather
}
"#;

#[test]
fn extended_grammar_parses_and_compiles_blueprint_shape() {
    let program = parse_str(EXTENDED).unwrap();
    let bp = compile(&program).unwrap().remove(0);

    assert_eq!(bp.graph_id, "orchestrator");
    assert_eq!(bp.start, "planner");
    assert_eq!(bp.checkpoint.as_deref(), Some("inherit"));
    assert_eq!(bp.interrupt.as_deref(), Some("manual"));

    // Input/output shape.
    assert_eq!(bp.input.len(), 2);
    assert_eq!(bp.input[0].name, "request");
    assert_eq!(bp.input[0].ty, "string");
    assert_eq!(bp.output.len(), 1);
    assert_eq!(bp.output[0].name, "answer");

    // Channels carry reducer + policy args.
    assert_eq!(bp.channels.len(), 3);
    let usage = bp.channels.iter().find(|c| c.name == "usage").unwrap();
    assert_eq!(usage.reducer, "aggregate");
    assert_eq!(usage.args, vec![Literal::Str("usage_delta".into())]);
    let arrivals = bp.channels.iter().find(|c| c.name == "arrivals").unwrap();
    assert_eq!(arrivals.reducer, "barrier");
    assert_eq!(arrivals.args, vec![Literal::Num(2.0)]);

    // Command lowering + routing precedence (goto becomes a static next).
    let planner = bp.nodes.iter().find(|n| n.name == "planner").unwrap();
    let cmd = planner.command.as_ref().unwrap();
    assert_eq!(cmd.goto.as_deref(), Some("fanout"));
    assert_eq!(
        cmd.update,
        vec![("status".into(), Literal::Str("planned".into()))]
    );
    assert_eq!(planner.routing, Routing::Next("fanout".into()));

    // Fanout sends.
    let fanout = bp.nodes.iter().find(|n| n.name == "fanout").unwrap();
    assert_eq!(fanout.sends.len(), 2);
    assert_eq!(fanout.sends[0].target, "worker_a");
    assert_eq!(fanout.sends[0].input.as_deref(), Some("split_a"));

    // Join node.
    let gather = bp.nodes.iter().find(|n| n.name == "gather").unwrap();
    assert_eq!(gather.kind, "join");
    assert_eq!(gather.join_sources, vec!["worker_a", "worker_b"]);

    // Sub-agent node with input mapping + policies.
    let research = bp.nodes.iter().find(|n| n.name == "research").unwrap();
    assert_eq!(research.kind, "subagent");
    assert_eq!(research.agent.as_deref(), Some("researcher"));
    assert_eq!(research.input.as_deref(), Some("topic"));
    assert_eq!(research.timeout.as_deref(), Some("30"));
    assert_eq!(
        research.retry,
        vec![
            ("max_attempts".into(), Literal::Num(3.0)),
            ("backoff".into(), Literal::Str("exponential".into())),
        ]
    );

    // Subgraph node references a registered graph by name.
    let sub = bp.nodes.iter().find(|n| n.name == "sub").unwrap();
    assert_eq!(sub.kind, "subgraph");
    assert_eq!(sub.subgraph.as_deref(), Some("summarize"));

    // REPL-backed node names a script capability (declaration only).
    let triage = bp.nodes.iter().find(|n| n.name == "triage").unwrap();
    assert_eq!(triage.kind, "repl_agent");
    assert_eq!(triage.script.as_deref(), Some("triage_script"));

    // Interrupt node with options + conditional routing.
    let review = bp.nodes.iter().find(|n| n.name == "review").unwrap();
    assert_eq!(review.kind, "interrupt");
    assert_eq!(review.options, vec!["approve", "reject"]);
    assert!(matches!(review.routing, Routing::Conditional(_)));

    // Top-level join declaration.
    assert_eq!(bp.joins.len(), 1);
    assert_eq!(bp.joins[0].target, "gather");
    assert_eq!(bp.joins[0].sources, vec!["worker_a", "worker_b"]);
}

#[test]
fn extended_blueprint_round_trips_through_serde() {
    let bp = compile(&parse_str(EXTENDED).unwrap()).unwrap().remove(0);
    let json = serde_json::to_string(&bp).unwrap();
    let back: crate::language::types::Blueprint = serde_json::from_str(&json).unwrap();
    assert_eq!(bp, back);
}

#[test]
fn command_goto_unknown_target_is_a_compile_error() {
    let src = "graph g { start a node a { command { goto ghost } } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("command goto target"), "{err}");
}

#[test]
fn send_unknown_target_is_a_compile_error() {
    let src = "graph g { start a node a { sends [ send ghost ] } }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("send target"), "{err}");
}

#[test]
fn join_unknown_source_is_a_compile_error() {
    let src = "graph g { start a node a { } join [ghost] -> a }";
    let err = compile(&parse_str(src).unwrap()).unwrap_err();
    assert!(err.to_string().contains("join source"), "{err}");
}

#[test]
fn extended_kinds_bind_against_a_resolver() {
    let bp = compile(&parse_str(EXTENDED).unwrap()).unwrap().remove(0);
    let resolver = CapabilityResolver::from_lists(["default".to_string()], std::iter::empty())
        .allow_subgraph("summarize")
        .allow_agent("researcher")
        .allow_script("triage_script")
        .allow_reducer("messages")
        .allow_reducer("aggregate")
        .allow_reducer("barrier")
        .with_node_kinds(
            crate::language::compiler::DEFAULT_NODE_KINDS
                .iter()
                .map(|k| k.to_string()),
        );
    resolver.bind_blueprint(&bp).unwrap();
}

#[test]
fn bind_blueprint_rejects_unregistered_subagent_and_script() {
    // The strict blueprint gate must validate `subagent` agent references and
    // `repl_agent` script references, not silently pass them through a model
    // check. Both were previously admitted, so this exercises the fail-closed
    // path the `Resolver` already covered but `bind_blueprint` did not.
    let node_kinds = || {
        crate::language::compiler::DEFAULT_NODE_KINDS
            .iter()
            .map(|k| k.to_string())
    };
    let bp = compile(&parse_str(EXTENDED).unwrap()).unwrap().remove(0);

    // Missing agent `researcher`: rejected with an unknown-agent capability error.
    let missing_agent = CapabilityResolver::from_lists(["default".to_string()], std::iter::empty())
        .allow_subgraph("summarize")
        .allow_script("triage_script")
        .allow_reducer("messages")
        .allow_reducer("aggregate")
        .allow_reducer("barrier")
        .with_node_kinds(node_kinds());
    let err = missing_agent.bind_blueprint(&bp).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown agent"), "{err}");
    assert!(err.to_string().contains("researcher"), "{err}");

    // Missing script `triage_script`: rejected with an unknown-script error.
    let missing_script =
        CapabilityResolver::from_lists(["default".to_string()], std::iter::empty())
            .allow_subgraph("summarize")
            .allow_agent("researcher")
            .allow_reducer("messages")
            .allow_reducer("aggregate")
            .allow_reducer("barrier")
            .with_node_kinds(node_kinds());
    let err = missing_script.bind_blueprint(&bp).unwrap_err();
    assert!(err.to_string().contains("unknown script"), "{err}");
    assert!(err.to_string().contains("triage_script"), "{err}");
}

// ---------------------------------------------------------------------------
// Capability binding
// ---------------------------------------------------------------------------

#[test]
fn bind_capabilities_accepts_allowed_references() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::from_lists(
        ["default".to_string()],
        ["lookup_user".to_string(), "create_ticket".to_string()],
    );
    bind_capabilities(&bp, &resolver).unwrap();
}

#[test]
fn bind_capabilities_rejects_unknown_model() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new()
        .allow_tool("lookup_user")
        .allow_tool("create_ticket");
    let err = bind_capabilities(&bp, &resolver).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown model"), "{err}");
}

#[test]
fn bind_capabilities_rejects_unknown_tool() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new().allow_model("default");
    let err = bind_capabilities(&bp, &resolver).unwrap_err();
    assert!(err.to_string().contains("unknown tool"), "{err}");
}

// ---------------------------------------------------------------------------
// Graph materialisation + execution
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestState {
    trail: Vec<String>,
    agent_visits: u32,
}

/// Resolves a conditional route `label` to a *durable* target node id from the
/// blueprint's `(label, target)` table, translating the language `END` sentinel
/// (`"END"`) to the durable graph terminal ([`crate::graph::END`]). Unknown
/// labels fall back to the durable `END`.
fn resolve_durable_target(routes: &[(String, String)], label: &str) -> String {
    let target = routes
        .iter()
        .find(|(l, _)| l == label)
        .map(|(_, t)| t.as_str());
    match target {
        Some(t) if t != crate::language::types::END => t.to_string(),
        _ => crate::graph::END.to_string(),
    }
}

/// A trivial factory that materialises echo/route/end nodes purely from the
/// declarative [`NodeSpec`]. It demonstrates that runnable behaviour comes from
/// Rust, not the source: each node records its name; terminal/`next` nodes
/// commit a whole-state update (static edges route them), and conditional nodes
/// loop once before terminating by emitting an explicit `goto` command.
struct TestFactory;

impl NodeFactory<TestState> for TestFactory {
    fn make(&self, spec: &NodeSpec) -> crate::error::Result<BoxedNode<TestState>> {
        let name = spec.name.clone();
        let routing = spec.routing.clone();
        Ok(Arc::new(
            move |mut state: TestState, _ctx: NodeContext| -> NodeFuture<TestState> {
                let name = name.clone();
                let routing = routing.clone();
                Box::pin(async move {
                    state.trail.push(name.clone());
                    let result = match &routing {
                        // Static edges (Next/Terminal) handle routing; just
                        // commit the whole-state update.
                        Routing::Terminal | Routing::Next(_) => NodeResult::Update(state),
                        Routing::Conditional(routes) => {
                            state.agent_visits += 1;
                            // Take the `tool_call -> tools` route until the
                            // second visit, then take `final -> END`.
                            let label = if state.agent_visits >= 2 {
                                "final"
                            } else {
                                "tool_call"
                            };
                            let target = resolve_durable_target(routes, label);
                            NodeResult::Command(Command::goto([target]).with_update(state))
                        }
                    };
                    Ok(result)
                })
            },
        ))
    }
}

/// Collects the visited node ids into owned strings for comparison.
fn visited_names(run: &crate::graph::GraphExecution<TestState>) -> Vec<String> {
    run.visited.iter().map(ToString::to_string).collect()
}

#[tokio::test]
async fn build_graph_runs_to_end() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let graph = build_graph(&bp, &TestFactory).unwrap();

    let run = graph
        .run(TestState {
            trail: Vec::new(),
            agent_visits: 0,
        })
        .await
        .unwrap();

    // agent -> tools -> agent (ends on second visit).
    assert_eq!(visited_names(&run), vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.trail, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.agent_visits, 2);
}

#[tokio::test]
async fn build_graph_handles_linear_terminal() {
    let src = "graph g { start a node a { kind model next b } node b { kind model next END } }";
    let bp = compile(&parse_str(src).unwrap()).unwrap().remove(0);
    let graph = build_graph(&bp, &TestFactory).unwrap();
    let run = graph
        .run(TestState {
            trail: Vec::new(),
            agent_visits: 0,
        })
        .await
        .unwrap();
    assert_eq!(visited_names(&run), vec!["a", "b"]);
}

// ---------------------------------------------------------------------------
// Registry-backed capability binding (registry → language binding)
// ---------------------------------------------------------------------------

use crate::language::compiler::{
    DEFAULT_NODE_KINDS, bind_capabilities_with_registry, compile_source,
};
use crate::registry::CapabilityRegistry;

/// A `.rag` graph that exercises every registry-backed reference kind: a model,
/// a tool, a subgraph reference (`kind subgraph` whose `model` names a
/// registered blueprint), a router reference (`kind router`), and a channel
/// reducer.
const FULL_SOURCE: &str = r#"
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
fn full_registry() -> CapabilityRegistry<TestState> {
    let mut reg = CapabilityRegistry::<TestState>::new();
    reg.register_model(
        "default",
        std::sync::Arc::new(crate::language::test::testkit::EchoModel),
    )
    .unwrap();
    reg.register_tool(std::sync::Arc::new(
        crate::language::test::testkit::NoopTool,
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
        std::sync::Arc::new(crate::language::test::testkit::EchoModel),
    );
    // Source references `default`, which we did not register here.
    let mut bare = CapabilityRegistry::<TestState>::new();
    bare.register_tool(std::sync::Arc::new(
        crate::language::test::testkit::NoopTool,
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

// ---------------------------------------------------------------------------
// Registry-backed Resolver (H3): spanned diagnostics, single binding gate
// ---------------------------------------------------------------------------

use crate::language::resolver::{Resolver, resolve_source};

#[test]
fn resolver_accepts_fully_registered_blueprint() {
    let reg = full_registry();
    let program = parse_str(FULL_SOURCE).unwrap();
    let resolver = Resolver::from_registry(&reg);
    // No diagnostics: every model/tool/subgraph/router/reducer is registered.
    assert!(resolver.resolve_program(&program).is_empty());
    // And the convenience façade lowers it to a blueprint.
    let blueprints = resolve_source(FULL_SOURCE, &reg).unwrap();
    assert_eq!(blueprints[0].graph_id, "main");
}

#[test]
fn resolver_reports_unregistered_tool_with_spanned_diagnostic() {
    // `missing` is not a registered tool.
    let src = r#"
graph g {
  start a
  channel m append
  node a {
    kind agent
    model "default"
    tools ["missing"]
    next END
  }
}
"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let file = SourceFile::new("plan.rag", src);
    let resolver = Resolver::from_registry(&reg);

    let diagnostics = resolver.resolve_program(&program);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    let diag = &diagnostics[0];
    assert_eq!(diag.code.as_deref(), Some("E-rag-unknown-tool"));
    let rendered = diag.render(&file);
    assert!(
        rendered.contains("node `a` references unknown tool `missing`"),
        "{rendered}"
    );
    // The diagnostic carries a caret pointing at the offending node span.
    assert!(rendered.contains('^'), "{rendered}");
    assert!(rendered.contains("--> plan.rag:"), "{rendered}");

    // `check_program` folds it into a Capability error with the rendered caret.
    let err = resolver.check_program(&program, Some(&file)).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown tool"), "{err}");
    assert!(err.to_string().contains('^'), "{err}");
}

#[test]
fn resolve_source_rejects_unregistered_tool() {
    let src = r#"graph g { start a channel m append node a { kind agent model "default" tools ["missing"] next END } }"#;
    let reg = full_registry();
    let err = resolve_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown tool"), "{err}");
}

#[test]
fn resolver_reports_unknown_node_kind_as_compile_error() {
    let src = r#"graph g { start a node a { kind wizard next END } }"#;
    let reg = full_registry();
    let err = resolve_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
    assert!(err.to_string().contains("unknown kind"), "{err}");
}

#[test]
fn resolver_reports_unregistered_agent() {
    // A `subagent` node binds its `agent "…"` reference through the registry's
    // Agent allowlist.
    let src = r#"graph g { start a node a { kind subagent agent "ghost" next END } }"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let diagnostics = Resolver::from_registry(&reg).resolve_program(&program);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    assert_eq!(diagnostics[0].code.as_deref(), Some("E-rag-unknown-agent"));
    assert!(
        diagnostics[0].message.contains("unknown agent `ghost`"),
        "{:?}",
        diagnostics[0]
    );
}

#[test]
fn resolver_collects_multiple_diagnostics() {
    // Two independent problems: an unregistered model and an unregistered
    // reducer. `resolve_program` reports both.
    let src = r#"graph g { start a channel m ghost node a { kind model model "nope" next END } }"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let diagnostics = Resolver::from_registry(&reg).resolve_program(&program);
    let codes: Vec<_> = diagnostics
        .iter()
        .filter_map(|d| d.code.as_deref())
        .collect();
    assert!(codes.contains(&"E-rag-unknown-model"), "{codes:?}");
    assert!(codes.contains(&"E-rag-unknown-reducer"), "{codes:?}");
}

#[test]
fn resolver_blueprint_path_matches_registry_binding() {
    // The span-less blueprint path mirrors the legacy gate's variants/messages.
    let reg = full_registry();
    let bp = compile(&parse_str(FULL_SOURCE).unwrap()).unwrap().remove(0);
    Resolver::from_registry(&reg)
        .resolve_blueprint(&bp)
        .unwrap();

    let bad = compile(
        &parse_str(r#"graph g { start a channel m append node a { kind subgraph model "ghost" next END } }"#)
            .unwrap(),
    )
    .unwrap()
    .remove(0);
    let err = Resolver::from_registry(&reg)
        .resolve_blueprint(&bad)
        .unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown subgraph"), "{err}");
}

// ---------------------------------------------------------------------------
// Provenance, diff, and language testkit (H4)
// ---------------------------------------------------------------------------

use crate::language::compiler::compile_with_provenance;
use crate::language::diff::{FieldChange, blueprint_diff};
use crate::language::testkit as lang_testkit;
use crate::language::types::Origin;

const DIFF_BASE: &str = r#"
graph flow {
  start plan

  channel messages append

  node plan {
    kind model
    model "default"
    routes {
      research -> work
      done -> END
    }
  }

  node work {
    kind tool_executor
    tools ["lookup_user"]
    next END
  }
}
"#;

/// Adds a node (`review`) and changes a route target on `plan`
/// (`research -> review` instead of `research -> work`).
const DIFF_NEW: &str = r#"
graph flow {
  start plan

  channel messages append

  node plan {
    kind model
    model "default"
    routes {
      research -> review
      done -> END
    }
  }

  node review {
    kind interrupt
    prompt "ok?"
    next work
  }

  node work {
    kind tool_executor
    tools ["lookup_user"]
    next END
  }
}
"#;

#[test]
fn blueprint_diff_reports_added_node_and_changed_route() {
    let old = lang_testkit::blueprint(DIFF_BASE);
    let new = lang_testkit::blueprint(DIFF_NEW);

    let diff = blueprint_diff(&old, &new);
    assert!(!diff.is_empty());

    // One node added: `review`.
    assert_eq!(diff.nodes_added, vec!["review".to_string()]);
    assert!(diff.nodes_removed.is_empty());

    // `plan`'s routing changed (research target work -> review).
    assert_eq!(diff.nodes_changed.len(), 1, "{diff:?}");
    let plan_change = &diff.nodes_changed[0];
    assert_eq!(plan_change.name, "plan");
    assert_eq!(
        plan_change.fields,
        vec![FieldChange {
            field: "routing".to_string(),
            old: "{ research -> work, done -> END }".to_string(),
            new: "{ research -> review, done -> END }".to_string(),
        }]
    );

    // The rendered summary names both the added node and the changed route.
    let rendered = diff.to_string();
    assert!(rendered.contains("+ node review"), "{rendered}");
    assert!(rendered.contains("~ node plan"), "{rendered}");
    assert!(rendered.contains("routing:"), "{rendered}");
}

#[test]
fn blueprint_diff_of_identical_blueprints_is_empty() {
    let a = lang_testkit::blueprint(DIFF_BASE);
    let b = lang_testkit::blueprint(DIFF_BASE);
    let diff = blueprint_diff(&a, &b);
    assert!(diff.is_empty(), "{diff:?}");
    assert_eq!(diff.to_string(), "no changes");
}

#[test]
fn blueprint_diff_serializes_round_trip() {
    let old = lang_testkit::blueprint(DIFF_BASE);
    let new = lang_testkit::blueprint(DIFF_NEW);
    let diff = blueprint_diff(&old, &new);
    let json = serde_json::to_string(&diff).unwrap();
    let back: crate::language::diff::BlueprintDiff = serde_json::from_str(&json).unwrap();
    assert_eq!(diff, back);
}

#[test]
fn provenance_points_each_node_at_its_span() {
    let program = parse_str(DIFF_NEW).unwrap();
    let bp = compile_with_provenance(&program, Origin::file("flow.rag"))
        .unwrap()
        .remove(0);

    let prov = bp.provenance().expect("provenance attached");
    assert_eq!(prov.origin, Origin::File("flow.rag".to_string()));

    // Every node in the blueprint has a recorded span anchored at the line of
    // its `node <name>` declaration, and the byte range slices back to the
    // `node` keyword that opens it.
    let file = SourceFile::new("flow.rag", DIFF_NEW);
    for node in &bp.nodes {
        let span = prov
            .node_span(&node.name)
            .unwrap_or_else(|| panic!("no span for node `{}`", node.name));
        // The span's byte range covers the opening `node` keyword.
        assert_eq!(&DIFF_NEW[span.start..span.end], "node");
        // The line the span anchors at is the node's declaration line.
        let line = file
            .line_text(span.line)
            .unwrap_or_else(|| panic!("no source line {}", span.line));
        assert!(
            line.contains(&format!("node {}", node.name)),
            "node `{}` span anchors at line `{line}`, not its declaration",
            node.name
        );
    }

    // The channel span is recorded too.
    assert!(prov.channel_span("messages").is_some());
}

#[test]
fn plain_compile_leaves_provenance_none() {
    let bp = lang_testkit::blueprint(DIFF_BASE);
    assert!(bp.provenance().is_none());
}

#[test]
fn generated_origin_renders_label() {
    let program = parse_str(DIFF_BASE).unwrap();
    let bp = compile_with_provenance(&program, Origin::generated_by("repl-7"))
        .unwrap()
        .remove(0);
    let prov = bp.provenance().unwrap();
    assert_eq!(prov.origin.as_display(), "generated by repl-7");
}

#[test]
fn testkit_assertions_inspect_lowered_topology() {
    let bp = lang_testkit::blueprint(DIFF_NEW);
    lang_testkit::assert_kind(&bp, "review", "interrupt");
    lang_testkit::assert_next(&bp, "review", "work");
    lang_testkit::assert_terminal(&bp, "work");
    lang_testkit::assert_route(&bp, "plan", "research", "review");
}

#[test]
fn testkit_try_compile_surfaces_errors() {
    // `start` references an undefined node, so compilation fails.
    let err = lang_testkit::try_compile("graph g { start missing node a { kind model next END } }")
        .unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
}

/// Minimal fake model/tool used to populate a [`CapabilityRegistry`] in tests.
mod testkit {
    use async_trait::async_trait;
    use serde_json::json;

    use crate::error::Result;
    use crate::harness::model::{ChatModel, ModelRequest, ModelResponse};
    use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

    use super::TestState;

    pub(super) struct EchoModel;

    #[async_trait]
    impl ChatModel<TestState> for EchoModel {
        async fn invoke(
            &self,
            _state: &TestState,
            _request: ModelRequest,
        ) -> Result<ModelResponse> {
            Ok(ModelResponse::assistant("echo"))
        }
    }

    pub(super) struct NoopTool;

    #[async_trait]
    impl Tool<TestState> for NoopTool {
        fn name(&self) -> &str {
            "lookup_user"
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("lookup_user", "noop", json!({"type": "object"}))
        }
        async fn call(&self, _state: &TestState, call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::text(call.id, call.name, "ok"))
        }
    }
}
