//! TRUE end-to-end (offline): graph topology extraction → JSON round-trip →
//! Mermaid rendering, from BOTH a programmatically built [`GraphBuilder`] graph
//! and a compiled `.rag` [`Blueprint`].
//!
//! This composes the **graph builder/compiler**, the **export/visualization**
//! subsystem (`GraphTopology` / `to_json` / `from_json` / `to_mermaid`), and the
//! **language** subsystem (parser + compiler → blueprint → topology). Every
//! assertion targets structure (nodes, edges, conditional labels, finish nodes)
//! and serialization fidelity — never any runnable behavior or model prose.

use rustagents::graph::export::{
    blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
use rustagents::language::compiler::compile;
use rustagents::language::parser::parse_str;
use rustagents::{GraphBuilder, GraphTopology, NodeContext, NodeResult};

/// A `.rag` source describing a two-node support agent with one conditional
/// (router-driven) edge and one static edge back to the agent.
const SUPPORT_AGENT: &str = r#"
graph support_agent {
  start agent

  defaults {
    recursion_limit 42
  }

  channel messages messages
  channel tool_calls append

  node agent {
    kind agent
    model "default"
    system "Resolve support requests using tools when useful."
    tools ["lookup_user"]
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

/// Builds a small linear durable graph: START -> a -> b -> END. The handler
/// bodies are irrelevant to topology export; only the declared structure is.
fn linear_graph() -> rustagents::CompiledGraph<i64, i64> {
    GraphBuilder::<i64, i64>::overwrite()
        .with_graph_id("linear")
        .with_recursion_limit(10)
        .add_node("a", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile()
        .expect("linear graph compiles")
}

/// Builds a graph with a conditional edge so we can assert labeled routes
/// survive both topology extraction and Mermaid rendering.
fn branching_graph() -> rustagents::CompiledGraph<i64, i64> {
    GraphBuilder::<i64, i64>::overwrite()
        .with_graph_id("branching")
        .add_node("router", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("left", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s - 1))
        })
        .add_node("right", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("router")
        .add_conditional_edges(
            "router",
            |s: &i64| {
                if *s < 0 {
                    "neg".to_string()
                } else {
                    "pos".to_string()
                }
            },
            [("neg", "left"), ("pos", "right")],
        )
        .set_finish("left")
        .set_finish("right")
        .compile()
        .expect("branching graph compiles")
}

#[tokio::test]
async fn compiled_graph_topology_json_round_trips() {
    let graph = linear_graph();
    let topology = graph.topology();

    // Structure: id, entry, finish node, both declared nodes, the single
    // a->b edge, and the recursion limit carried over.
    assert_eq!(topology.graph_id, "linear");
    assert_eq!(topology.entry.as_deref(), Some("a"));
    assert_eq!(topology.recursion_limit, 10);
    assert_eq!(topology.finish_nodes, vec!["b".to_string()]);
    let node_ids: Vec<&str> = topology.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(node_ids, vec!["a", "b"]);
    assert_eq!(topology.edges.len(), 1);
    assert_eq!(topology.edges[0].from, "a");
    assert_eq!(topology.edges[0].to, "b");
    assert!(topology.conditional_edges.is_empty());

    // JSON round-trip is lossless: serialize → deserialize → identical topology.
    let json = to_json(&topology);
    let parsed: GraphTopology = from_json(&json).expect("json parses back");
    assert_eq!(parsed, topology);
}

#[tokio::test]
async fn compiled_graph_mermaid_contains_nodes_and_edges() {
    let graph = linear_graph();
    let topology = graph.topology();
    let mermaid = to_mermaid(&topology);

    // Boundaries + header.
    assert!(mermaid.starts_with("flowchart TD\n"));
    assert!(mermaid.contains("START([START])"));
    assert!(mermaid.contains("END([END])"));

    // Node declarations carry the original label.
    assert!(mermaid.contains("[\"a\"]"));
    assert!(mermaid.contains("[\"b\"]"));

    // Entry, direct, and finish edges are all rendered.
    assert!(mermaid.contains("START --> n_a"));
    assert!(mermaid.contains("n_a --> n_b"));
    assert!(mermaid.contains("n_b --> END"));
}

#[tokio::test]
async fn conditional_edges_render_labeled_routes() {
    let graph = branching_graph();
    let topology = graph.topology();

    // The router has exactly one conditional edge group with two labeled routes.
    assert_eq!(topology.conditional_edges.len(), 1);
    let cond = &topology.conditional_edges[0];
    assert_eq!(cond.from, "router");
    // Routes are sorted by label: "neg" before "pos".
    let routes: Vec<(&str, &str)> = cond
        .routes
        .iter()
        .map(|r| (r.label.as_str(), r.target.as_str()))
        .collect();
    assert_eq!(routes, vec![("neg", "left"), ("pos", "right")]);

    // Round-trips through JSON unchanged.
    let parsed: GraphTopology = from_json(&to_json(&topology)).expect("json parses");
    assert_eq!(parsed, topology);

    // Mermaid renders both labeled conditional edges.
    let mermaid = to_mermaid(&topology);
    assert!(mermaid.contains("n_router -- neg --> n_left"));
    assert!(mermaid.contains("n_router -- pos --> n_right"));
}

#[tokio::test]
async fn rag_blueprint_topology_and_mermaid() {
    // Parse + compile the .rag source into a Blueprint.
    let program = parse_str(SUPPORT_AGENT).expect("source parses");
    let blueprint = compile(&program).expect("program compiles").remove(0);

    let topology = blueprint_to_topology(&blueprint);

    // Structure carried from the blueprint.
    assert_eq!(topology.graph_id, "support_agent");
    assert_eq!(topology.entry.as_deref(), Some("agent"));
    assert_eq!(topology.recursion_limit, 42);

    // Both nodes present, with kinds preserved (sorted by id: agent, tools).
    let nodes: Vec<(&str, Option<&str>)> = topology
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.kind.as_deref()))
        .collect();
    assert_eq!(
        nodes,
        vec![("agent", Some("agent")), ("tools", Some("tool_executor"))]
    );

    // The agent's conditional routing: tool_call -> tools, final -> END.
    assert_eq!(topology.conditional_edges.len(), 1);
    let cond = &topology.conditional_edges[0];
    assert_eq!(cond.from, "agent");
    let routes: Vec<(&str, &str)> = cond
        .routes
        .iter()
        .map(|r| (r.label.as_str(), r.target.as_str()))
        .collect();
    // Sorted by label: "final" before "tool_call".
    assert_eq!(routes, vec![("final", "END"), ("tool_call", "tools")]);

    // The tool_executor's static `next agent` edge.
    assert_eq!(topology.edges.len(), 1);
    assert_eq!(topology.edges[0].from, "tools");
    assert_eq!(topology.edges[0].to, "agent");

    // Channels (with reducer names) carried from the blueprint.
    let channels: Vec<(&str, &str)> = topology
        .channels
        .iter()
        .map(|c| (c.name.as_str(), c.reducer.as_str()))
        .collect();
    assert!(channels.contains(&("messages", "messages")));
    assert!(channels.contains(&("tool_calls", "append")));

    // JSON round-trip preserves everything.
    let parsed: GraphTopology = from_json(&to_json(&topology)).expect("json parses");
    assert_eq!(parsed, topology);

    // Mermaid contains the nodes, the static edge, and the labeled routes.
    let mermaid = blueprint_to_mermaid(&blueprint);
    assert!(mermaid.contains("flowchart TD"));
    assert!(mermaid.contains("[\"agent\"]"));
    assert!(mermaid.contains("[\"tools\"]"));
    assert!(mermaid.contains("START --> n_agent"));
    assert!(mermaid.contains("n_tools --> n_agent"));
    assert!(mermaid.contains("n_agent -- tool_call --> n_tools"));
    assert!(mermaid.contains("n_agent -- final --> END"));
}
