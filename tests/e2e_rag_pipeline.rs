//! TRUE end-to-end: declarative `.rag` source → parsed AST → compiled
//! [`Blueprint`] → capability-bound → materialised durable [`CompiledGraph`] →
//! executed to `END`.
//!
//! This composes the **language** subsystem (parser + compiler + capability
//! binding + topology materialisation) with the **graph** executor. A
//! `NodeFactory` materialises each compiled `NodeSpec` into a runnable durable
//! node whose behaviour distinguishes `agent` from `tool_executor` kinds and
//! records the path taken, proving source text drives a real running graph.

use std::sync::Arc;

use tinyagents::graph::{END, NodeFuture};
use tinyagents::language::compiler::{
    BoxedNode, CapabilityResolver, NodeFactory, bind_capabilities, build_graph, compile,
};
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::{END as LANG_END, NodeSpec, Routing};
use tinyagents::{Command, NodeContext, NodeResult, Result};

const SUPPORT_AGENT: &str = r#"
graph support_agent {
  start agent

  defaults {
    recursion_limit 50
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

/// State that records the execution trail plus how many times each kind of node
/// ran, so the test can assert structure without depending on any model output.
#[derive(Clone, Debug, Default)]
struct RagState {
    trail: Vec<String>,
    agent_turns: u32,
    tool_runs: u32,
}

/// Resolves a conditional route `label` to a *durable* target node id from the
/// spec's `(label, target)` table, translating the language `END` sentinel to
/// the durable graph terminal ([`END`]) and defaulting to `END` when the label
/// is absent.
fn resolve_route(routes: &[(String, String)], label: &str) -> String {
    match routes
        .iter()
        .find(|(l, _)| l == label)
        .map(|(_, t)| t.as_str())
    {
        Some(t) if t != LANG_END => t.to_string(),
        _ => END.to_string(),
    }
}

/// Materialises each compiled [`NodeSpec`] into a runnable durable node:
///
/// - `agent` (conditional routing): first visit routes `tool_call -> tools`;
///   the second visit takes the `final -> END` route via an explicit `goto`.
/// - `tool_executor` (static `next`): records a tool run and commits an update;
///   the static edge routes it.
struct RagFactory;

impl NodeFactory<RagState> for RagFactory {
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<RagState>> {
        let name = spec.name.clone();
        let kind = spec.kind.clone();
        let routing = spec.routing.clone();

        Ok(Arc::new(
            move |mut state: RagState, _ctx: NodeContext| -> NodeFuture<RagState> {
                let name = name.clone();
                let kind = kind.clone();
                let routing = routing.clone();
                Box::pin(async move {
                    state.trail.push(name.clone());

                    let result = match (kind.as_str(), &routing) {
                        // Agent node: loop once through the tool executor, then end.
                        ("agent", Routing::Conditional(routes)) => {
                            state.agent_turns += 1;
                            let label = if state.agent_turns >= 2 {
                                "final"
                            } else {
                                "tool_call"
                            };
                            let target = resolve_route(routes, label);
                            NodeResult::Command(Command::goto([target]).with_update(state))
                        }
                        // Tool executor: record a run; the static edge routes it.
                        ("tool_executor", Routing::Next(_)) => {
                            state.tool_runs += 1;
                            NodeResult::Update(state)
                        }
                        (_, Routing::Terminal | Routing::Next(_)) => NodeResult::Update(state),
                        (_, Routing::Conditional(routes)) => {
                            let target = resolve_route(routes, "final");
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
async fn rag_source_compiles_binds_and_runs_to_end() {
    // 1. Parse + compile.
    let program = parse_str(SUPPORT_AGENT).expect("source parses");
    let blueprint = compile(&program).expect("program compiles").remove(0);

    // 2. Assert the compiled structure.
    assert_eq!(blueprint.graph_id, "support_agent");
    assert_eq!(blueprint.start, "agent");
    assert_eq!(blueprint.nodes.len(), 2);
    assert_eq!(blueprint.nodes[0].kind, "agent");
    assert_eq!(blueprint.nodes[1].kind, "tool_executor");
    assert_eq!(
        blueprint.nodes[1].routing,
        Routing::Next("agent".to_string())
    );

    // 3. Bind capabilities against an allowlist that names the model + tools.
    let resolver = CapabilityResolver::new()
        .allow_model("default")
        .allow_tool("lookup_user")
        .allow_tool("create_ticket");
    bind_capabilities(&blueprint, &resolver).expect("capabilities resolve");

    // 4. Materialise the blueprint into a runnable graph and execute it.
    let graph = build_graph(&blueprint, &RagFactory).expect("graph builds");
    let run = graph
        .run(RagState::default())
        .await
        .expect("graph runs to END");

    // 5. Assert the visited path + final state structure.
    let visited: Vec<String> = run.visited.iter().map(ToString::to_string).collect();
    assert_eq!(visited, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.trail, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.agent_turns, 2);
    assert_eq!(run.state.tool_runs, 1);
}
