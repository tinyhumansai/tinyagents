//! Tests for the expressive language pipeline: lexer, parser, compiler,
//! capability binding, and graph materialisation.

use crate::graph::{Node, NodeOutput};
use crate::language::compiler::{
    CapabilityResolver, NodeFactory, bind_capabilities, build_graph, compile,
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

/// A trivial factory that materialises echo/route/end nodes purely from the
/// declarative [`NodeSpec`]. It demonstrates that runnable behaviour comes from
/// Rust, not the source: each node records its name; terminal nodes end, `next`
/// nodes continue, and conditional nodes loop once before terminating.
struct TestFactory;

impl NodeFactory<TestState> for TestFactory {
    fn make(&self, spec: &NodeSpec) -> crate::error::Result<Node<TestState>> {
        let name = spec.name.clone();
        let routing = spec.routing.clone();
        Ok(Node::new(name.clone(), move |mut state: TestState| {
            let name = name.clone();
            let routing = routing.clone();
            async move {
                state.trail.push(name.clone());
                let output = match &routing {
                    Routing::Terminal => NodeOutput::end(state),
                    Routing::Next(_) => NodeOutput::continue_with(state),
                    Routing::Conditional(_) => {
                        state.agent_visits += 1;
                        if state.agent_visits >= 2 {
                            // Take the `final -> END` route by ending.
                            NodeOutput::end(state)
                        } else {
                            NodeOutput::route(state, "tool_call")
                        }
                    }
                };
                Ok(output)
            }
        }))
    }
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
    assert_eq!(run.visited, vec!["agent", "tools", "agent"]);
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
    assert_eq!(run.visited, vec!["a", "b"]);
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
