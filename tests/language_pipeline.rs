//! End-to-end coverage for the expressive-language pipeline.
//!
//! Parses `.rag` source, compiles it into a [`Blueprint`], asserts the compiled
//! structure, exercises the parser/compiler error cases, and materialises a
//! durable [`CompiledGraph`] via a trivial [`NodeFactory`] before running it to
//! `END`.

use std::sync::Arc;

use tinyagents::graph::{END, NodeFuture};
use tinyagents::language::capability_resolver::{CapabilityResolver, bind_capabilities};
use tinyagents::language::compiler::{BoxedNode, NodeFactory, build_graph, compile};
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::{END as LANG_END, NodeSpec, Routing};
use tinyagents::{Command, NodeContext, NodeResult, Result, TinyAgentsError};

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

#[test]
fn compiles_support_agent_blueprint_structure() {
    let program = parse_str(SUPPORT_AGENT).expect("source parses");
    assert_eq!(program.graphs.len(), 1);

    let blueprint = compile(&program).expect("program compiles").remove(0);

    assert_eq!(blueprint.graph_id, "support_agent");
    assert_eq!(blueprint.start, "agent");
    assert_eq!(blueprint.nodes.len(), 2);

    let agent = &blueprint.nodes[0];
    assert_eq!(agent.name, "agent");
    assert_eq!(agent.kind, "agent");
    assert_eq!(agent.model.as_deref(), Some("default"));
    assert_eq!(agent.tools, vec!["lookup_user", "create_ticket"]);
    match &agent.routing {
        Routing::Conditional(routes) => {
            assert_eq!(routes.len(), 2);
            assert!(routes.contains(&("tool_call".to_string(), "tools".to_string())));
        }
        other => panic!("expected conditional routing on `agent`, got {other:?}"),
    }

    let tools = &blueprint.nodes[1];
    assert_eq!(tools.name, "tools");
    assert_eq!(tools.routing, Routing::Next("agent".to_string()));

    // Capabilities bind cleanly against an allowlist that names the model/tools.
    let resolver = CapabilityResolver::new()
        .allow_model("default")
        .allow_tool("lookup_user")
        .allow_tool("create_ticket");
    bind_capabilities(&blueprint, &resolver).expect("capabilities resolve");
}

#[test]
fn bind_capabilities_rejects_unknown_tool() {
    let blueprint = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);

    // Allow the model and only one of the two tools.
    let resolver = CapabilityResolver::new()
        .allow_model("default")
        .allow_tool("lookup_user");

    let err = bind_capabilities(&blueprint, &resolver).expect_err("create_ticket is not allowed");
    match err {
        TinyAgentsError::Capability(msg) => assert!(msg.contains("create_ticket"), "{msg}"),
        other => panic!("expected Capability error, got {other:?}"),
    }
}

#[test]
fn missing_start_is_a_compile_error() {
    let program = parse_str("graph no_start { node a { kind model } }").expect("parses");
    let err = compile(&program).expect_err("a graph without `start` cannot compile");
    assert!(matches!(err, TinyAgentsError::Compile(_)), "got {err:?}");
}

#[test]
fn duplicate_node_is_a_compile_error() {
    let src = "graph dupes { start a node a { kind model } node a { kind model } }";
    let program = parse_str(src).expect("parses");
    let err = compile(&program).expect_err("duplicate node names cannot compile");
    match err {
        TinyAgentsError::Compile(msg) => assert!(msg.contains("duplicate"), "{msg}"),
        other => panic!("expected Compile error, got {other:?}"),
    }
}

#[test]
fn unknown_route_target_is_a_compile_error() {
    let src = "graph bad_route { start a node a { routes { go -> ghost } } }";
    let program = parse_str(src).expect("parses");
    let err = compile(&program).expect_err("routing to a missing node cannot compile");
    assert!(matches!(err, TinyAgentsError::Compile(_)), "got {err:?}");
}

/// Application state used to trace the path taken through the materialised
/// graph.
#[derive(Clone, Debug, Default)]
struct TraceState {
    trail: Vec<String>,
    agent_visits: u32,
}

/// A trivial factory that turns each [`NodeSpec`] into a durable node handler
/// which records its name and follows the spec's routing. The conditional
/// `agent` node loops once through `tools` and then ends via an explicit
/// `goto`.
struct TraceFactory;

impl NodeFactory<TraceState> for TraceFactory {
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<TraceState>> {
        let name = spec.name.clone();
        let routing = spec.routing.clone();
        Ok(Arc::new(
            move |mut state: TraceState, _ctx: NodeContext| -> NodeFuture<TraceState> {
                let name = name.clone();
                let routing = routing.clone();
                Box::pin(async move {
                    state.trail.push(name.clone());
                    let result = match &routing {
                        // Static edges (Next/Terminal) route these.
                        Routing::Terminal | Routing::Next(_) => NodeResult::Update(state),
                        Routing::Conditional(routes) => {
                            state.agent_visits += 1;
                            let label = if state.agent_visits >= 2 {
                                "final"
                            } else {
                                "tool_call"
                            };
                            let target = match routes
                                .iter()
                                .find(|(l, _)| l == label)
                                .map(|(_, t)| t.as_str())
                            {
                                Some(t) if t != LANG_END => t.to_string(),
                                _ => END.to_string(),
                            };
                            NodeResult::Command(Command::goto([target]).with_update(state))
                        }
                    };
                    Ok(result)
                })
            },
        ))
    }
}

#[tokio::test]
async fn build_graph_runs_to_end() {
    let blueprint = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);

    let graph = build_graph(&blueprint, &TraceFactory).expect("graph builds");

    let run = graph.run(TraceState::default()).await.expect("graph runs");

    let visited: Vec<String> = run.visited.iter().map(ToString::to_string).collect();
    assert_eq!(visited, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.trail, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.agent_visits, 2);
}
