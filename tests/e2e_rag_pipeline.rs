//! TRUE end-to-end: declarative `.rag` source → parsed AST → compiled
//! [`Blueprint`] → capability-bound → materialised [`StateGraph`] → executed to
//! `END`.
//!
//! This composes the **language** subsystem (parser + compiler + capability
//! binding + topology materialisation) with the **graph** executor. A
//! `NodeFactory` materialises each compiled `NodeSpec` into a runnable node
//! whose behaviour distinguishes `agent` from `tool_executor` kinds and records
//! the path taken, proving source text drives a real running graph.

use tinyagents::language::compiler::{
    CapabilityResolver, NodeFactory, bind_capabilities, build_graph, compile,
};
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::{NodeSpec, Routing};
use tinyagents::{Node, NodeOutput, Result};

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

/// Materialises each compiled [`NodeSpec`] into a runnable node:
///
/// - `agent` (conditional routing): first visit routes `tool_call -> tools`;
///   the second visit takes the `final -> END` route by emitting `End`.
/// - `tool_executor` (static `next`): records a tool run and continues.
struct RagFactory;

impl NodeFactory<RagState> for RagFactory {
    fn make(&self, spec: &NodeSpec) -> Result<Node<RagState>> {
        let name = spec.name.clone();
        let kind = spec.kind.clone();
        let routing = spec.routing.clone();

        Ok(Node::new(name.clone(), move |mut state: RagState| {
            let name = name.clone();
            let kind = kind.clone();
            let routing = routing.clone();
            async move {
                state.trail.push(name.clone());

                let output = match (kind.as_str(), &routing) {
                    // Agent node: loop once through the tool executor, then end.
                    ("agent", Routing::Conditional(_)) => {
                        state.agent_turns += 1;
                        if state.agent_turns >= 2 {
                            NodeOutput::end(state)
                        } else {
                            NodeOutput::route(state, "tool_call")
                        }
                    }
                    // Tool executor: record a run and follow its static edge.
                    ("tool_executor", Routing::Next(_)) => {
                        state.tool_runs += 1;
                        NodeOutput::continue_with(state)
                    }
                    (_, Routing::Terminal) => NodeOutput::end(state),
                    (_, Routing::Next(_)) => NodeOutput::continue_with(state),
                    (_, Routing::Conditional(_)) => NodeOutput::end(state),
                };
                Ok(output)
            }
        }))
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
    assert_eq!(run.visited, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.trail, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.agent_turns, 2);
    assert_eq!(run.state.tool_runs, 1);
}
