//! Unit tests for graph export / visualization.

use crate::graph::builder::{END, GraphBuilder, START};
use crate::graph::command::NodeResult;
use crate::graph::export::{
    blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
use crate::language::{compiler, parser};

/// Builds a small branching graph: START -> route -> {even,odd} with `route`
/// conditionally selecting a successor, both successors finishing.
fn sample_graph() -> crate::graph::CompiledGraph<i64, i64> {
    GraphBuilder::<i64, i64>::overwrite()
        .with_graph_id("demo")
        .add_node("route", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("even", |s, _| async move { Ok(NodeResult::Update(s + 1)) })
        .add_node("odd", |s, _| async move { Ok(NodeResult::Update(s - 1)) })
        .add_edge(START, "route")
        .add_conditional_edges(
            "route",
            |s: &i64| {
                if s % 2 == 0 {
                    "even".to_string()
                } else {
                    "odd".to_string()
                }
            },
            [("even", "even"), ("odd", "odd")],
        )
        .add_edge("even", END)
        .add_edge("odd", END)
        .compile()
        .expect("compiles")
}

#[test]
fn extracts_topology_structure() {
    let topology = sample_graph().topology();

    assert_eq!(topology.graph_id, "demo");
    assert_eq!(topology.entry.as_deref(), Some("route"));

    // Nodes are sorted by id.
    let ids: Vec<&str> = topology.nodes.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, vec!["even", "odd", "route"]);

    // `route` has a conditional edge with both labeled routes (sorted).
    assert_eq!(topology.conditional_edges.len(), 1);
    let cond = &topology.conditional_edges[0];
    assert_eq!(cond.from, "route");
    let routes: Vec<(&str, &str)> = cond
        .routes
        .iter()
        .map(|r| (r.label.as_str(), r.target.as_str()))
        .collect();
    assert_eq!(routes, vec![("even", "even"), ("odd", "odd")]);

    // Both `even` and `odd` are finish nodes; START/END edges are lifted out.
    assert_eq!(topology.finish_nodes, vec!["even", "odd"]);
    assert!(topology.edges.is_empty());
}

#[test]
fn json_round_trips() {
    let topology = sample_graph().topology();
    let json = to_json(&topology);
    assert!(json.contains("\"graph_id\": \"demo\""));

    let restored = from_json(&json).expect("valid json");
    assert_eq!(restored, topology);
}

#[test]
fn from_json_rejects_malformed_input() {
    assert!(from_json("not json").is_err());
}

#[test]
fn mermaid_contains_nodes_edges_and_labels() {
    let topology = sample_graph().topology();
    let mermaid = to_mermaid(&topology);

    assert!(mermaid.starts_with("flowchart TD\n"));
    assert!(mermaid.contains("START([START])"));
    assert!(mermaid.contains("END([END])"));
    // Node declarations carry the original label.
    assert!(mermaid.contains("n_route[\"route\"]"));
    // Entry edge.
    assert!(mermaid.contains("START --> n_route"));
    // Conditional routes are labeled.
    assert!(mermaid.contains("n_route -- even --> n_even"));
    assert!(mermaid.contains("n_route -- odd --> n_odd"));
    // Finish edges.
    assert!(mermaid.contains("n_even --> END"));
    assert!(mermaid.contains("n_odd --> END"));
}

#[test]
fn mermaid_is_deterministic() {
    let a = to_mermaid(&sample_graph().topology());
    let b = to_mermaid(&sample_graph().topology());
    assert_eq!(a, b);
}

#[test]
fn direct_edges_are_captured() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .add_node("a", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("b", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
        .expect("compiles");

    let topology = graph.topology();
    assert_eq!(topology.edges.len(), 1);
    assert_eq!(topology.edges[0].from, "a");
    assert_eq!(topology.edges[0].to, "b");
    assert_eq!(topology.finish_nodes, vec!["b"]);
}

#[test]
fn builder_topology_works_before_compile() {
    let builder = GraphBuilder::<i64, i64>::overwrite()
        .with_graph_id("wip")
        .add_node("a", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_edge(START, "a");

    let topology = builder.topology();
    assert_eq!(topology.graph_id, "wip");
    assert_eq!(topology.entry.as_deref(), Some("a"));
    assert_eq!(topology.nodes.len(), 1);
}

#[test]
fn blueprint_exports_topology_and_mermaid() {
    let source = r#"
graph support {
  start triage
  channel messages append

  node triage {
    routes {
      urgent -> escalate
      normal -> respond
    }
  }
  node escalate { next respond }
  node respond { next END }
}
"#;

    let program = parser::parse_str(source).expect("parses");
    let blueprints = compiler::compile(&program).expect("compiles");
    let blueprint = &blueprints[0];

    let topology = blueprint_to_topology(blueprint);
    assert_eq!(topology.graph_id, "support");
    assert_eq!(topology.entry.as_deref(), Some("triage"));

    // Channel/reducer names are carried over from the blueprint.
    assert_eq!(topology.channels.len(), 1);
    assert_eq!(topology.channels[0].name, "messages");
    assert_eq!(topology.channels[0].reducer, "append");

    // `respond` is terminal.
    assert_eq!(topology.finish_nodes, vec!["respond"]);

    // JSON round-trips.
    let restored = from_json(&to_json(&topology)).expect("round trip");
    assert_eq!(restored, topology);

    // Mermaid carries the conditional labels.
    let mermaid = blueprint_to_mermaid(blueprint);
    assert!(mermaid.contains("START --> n_triage"));
    assert!(mermaid.contains("n_triage -- urgent --> n_escalate"));
    assert!(mermaid.contains("n_triage -- normal --> n_respond"));
    assert!(mermaid.contains("n_respond --> END"));
}

/// A graph exercising every marker kind: a conditional route, a barrier
/// (waiting/fan-in) edge, and a subgraph node, plus an interrupt and a deferred
/// marker. The structured export must surface them all, and Mermaid must render.
#[test]
fn exports_all_marker_kinds_and_renders_mermaid() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .with_graph_id("markers")
        .with_name("Markers Demo")
        .add_node("route", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("left", |s, _| async move { Ok(NodeResult::Update(s + 1)) })
        .add_node("right", |s, _| async move { Ok(NodeResult::Update(s - 1)) })
        .add_node("join", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_conditional_edges(
            "route",
            |s: &i64| {
                if *s >= 0 {
                    "left".to_string()
                } else {
                    "right".to_string()
                }
            },
            [("left", "left"), ("right", "right")],
        )
        .add_waiting_edge("left", "join")
        .add_waiting_edge("right", "join")
        .add_edge(START, "route")
        .add_edge("join", END)
        .mark_subgraph("left")
        .mark_interrupt("route")
        .mark_deferred("join")
        .with_node_kind("left", "subgraph")
        .with_node_metadata("route", "owner", "triage")
        .compile()
        .expect("compiles");

    let topology = graph.topology();

    // Graph id + name.
    assert_eq!(topology.graph_id, "markers");
    assert_eq!(topology.name.as_deref(), Some("Markers Demo"));

    // Graph-level policy summary.
    assert_eq!(topology.policy.recursion_limit, 50);
    assert!(!topology.policy.parallel);

    // Conditional route labels.
    assert_eq!(topology.conditional_edges.len(), 1);
    assert_eq!(topology.conditional_edges[0].from, "route");

    // Barrier / waiting edges: `join` waits on both branches; they are lifted
    // out of the direct edge set.
    assert_eq!(topology.waiting_edges.len(), 1);
    assert_eq!(topology.waiting_edges[0].target, "join");
    assert_eq!(
        topology.waiting_edges[0].predecessors,
        vec!["left", "right"]
    );
    assert!(topology.edges.is_empty());

    // Per-node markers + policy summaries.
    let route = topology.nodes.iter().find(|n| n.id == "route").unwrap();
    assert!(route.interrupt);
    assert_eq!(route.policy.routing, "conditional");
    assert!(route.policy.interrupt);
    assert_eq!(
        route.metadata.get("owner").map(String::as_str),
        Some("triage")
    );

    let left = topology.nodes.iter().find(|n| n.id == "left").unwrap();
    assert!(left.subgraph);
    assert!(left.policy.subgraph);
    assert_eq!(left.kind.as_deref(), Some("subgraph"));

    let join = topology.nodes.iter().find(|n| n.id == "join").unwrap();
    assert!(join.deferred);
    assert!(join.policy.barrier);
    assert!(join.policy.deferred);
    assert_eq!(join.policy.routing, "terminal");

    // Validation is clean for this well-formed graph.
    assert!(topology.validation.ok);
    assert!(topology.validation.errors.is_empty());

    // JSON round-trips with all the new fields.
    let restored = from_json(&to_json(&topology)).expect("round trip");
    assert_eq!(restored, topology);

    // Mermaid renders every marker kind.
    let mermaid = to_mermaid(&topology);
    assert!(mermaid.starts_with("flowchart TD\n"));
    // Subgraph node uses the subroutine shape and a class.
    assert!(mermaid.contains("n_left[[\"left\"]]"));
    assert!(mermaid.contains("class n_left subgraph"));
    // Interrupt + deferred classes.
    assert!(mermaid.contains("class n_route interrupt"));
    assert!(mermaid.contains("class n_join deferred"));
    // Conditional labels and barrier joins.
    assert!(mermaid.contains("n_route -- left --> n_left"));
    assert!(mermaid.contains("n_left -. barrier .-> n_join"));
    assert!(mermaid.contains("n_right -. barrier .-> n_join"));
    assert!(mermaid.contains("n_join --> END"));
}

/// Command-routing nodes can declare `goto` destination hints, which the export
/// surfaces structurally and as dotted Mermaid edges.
#[test]
fn command_destination_hints_are_exported() {
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .add_node("dispatch", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("a", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("b", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_edge(START, "dispatch")
        .with_command_destinations("dispatch", ["a", "b"])
        .add_edge("a", END)
        .add_edge("b", END)
        .compile()
        .expect("compiles");

    let topology = graph.topology();
    let dispatch = topology.nodes.iter().find(|n| n.id == "dispatch").unwrap();
    assert!(dispatch.command_routing);
    assert_eq!(dispatch.command_destinations, vec!["a", "b"]);
    assert_eq!(dispatch.policy.routing, "command");

    let mermaid = to_mermaid(&topology);
    assert!(mermaid.contains("n_dispatch -. goto .-> n_a"));
    assert!(mermaid.contains("n_dispatch -. goto .-> n_b"));
}

/// A builder-stage topology with a dangling conditional target and an
/// unreachable node surfaces a validation error and warning respectively.
#[test]
fn validation_flags_dangling_and_unreachable() {
    let builder = GraphBuilder::<i64, i64>::overwrite()
        .add_node("a", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_node("orphan", |s, _| async move { Ok(NodeResult::Update(s)) })
        .add_edge(START, "a")
        .add_conditional_edges("a", |_: &i64| "x".to_string(), [("x", "missing")]);

    let topology = builder.topology();
    assert!(!topology.validation.ok);
    assert!(
        topology
            .validation
            .errors
            .iter()
            .any(|e| e.contains("missing"))
    );
    assert!(
        topology
            .validation
            .warnings
            .iter()
            .any(|w| w.contains("orphan") && w.contains("unreachable"))
    );
}
