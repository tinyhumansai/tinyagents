use std::sync::Arc;
use std::time::Duration;

use tinyagents::graph::export::{
    blueprint_to_json, blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
use tinyagents::graph::{Command, GraphBuilder, GraphDefaults, NodeResult, Route, START};
use tinyagents::harness::ids::GraphId;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::testkit::FakeTool;
use tinyagents::language::compiler::CapabilityResolver;
use tinyagents::language::resolver::{Resolver, resolve_source};
use tinyagents::language::{
    Blueprint, ChannelSpec, EdgeSpec, Literal, NodeSpec, Routing, parser,
};
use tinyagents::registry::{CapabilityRegistry, ComponentKind};

#[tokio::test]
async fn graph_builder_validates_topology_and_exports_metadata() {
    let no_reducer = GraphBuilder::<i32, i32>::new()
        .add_node("a", |state, _| async move { Ok(NodeResult::Update(state + 1)) })
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap_err();
    assert!(no_reducer.to_string().contains("no state reducer"));

    let missing_start = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |state, _| async move { Ok(NodeResult::Update(state + 1)) })
        .compile()
        .unwrap_err();
    assert!(missing_start.to_string().to_ascii_lowercase().contains("start"));

    let start_to_end = GraphBuilder::<i32, i32>::overwrite()
        .add_edge(START, tinyagents::graph::END)
        .compile()
        .unwrap_err();
    assert!(start_to_end.to_string().contains("START cannot route directly to END"));

    let dangling = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |state, _| async move { Ok(NodeResult::Update(state + 1)) })
        .set_entry("a")
        .add_edge("a", "missing")
        .compile()
        .unwrap_err();
    assert!(dangling.to_string().contains("missing"));

    let mixed_routing = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |state, _| async move { Ok(NodeResult::Update(state + 1)) })
        .add_node("b", |state, _| async move { Ok(NodeResult::Update(state + 1)) })
        .set_entry("a")
        .add_edge("a", "b")
        .mark_command_routing("a")
        .compile()
        .unwrap_err();
    assert!(mixed_routing.to_string().contains("command routing"));

    let route = Route::new("ok");
    assert_eq!(route.as_str(), "ok");
    assert_eq!(route.to_string(), "ok");

    let builder = GraphBuilder::<i32, i32>::overwrite()
        .with_graph_id(GraphId::new("graph-contract"))
        .with_name("Contract Graph")
        .set_defaults(GraphDefaults {
            recursion_limit: Some(7),
            parallel: Some(true),
            max_concurrency: Some(2),
            node_timeout: Some(Duration::from_millis(250)),
        })
        .with_parallel(true)
        .with_max_concurrency(0)
        .with_max_concurrency(2)
        .with_node_timeout(Duration::from_millis(250))
        .add_node("start", |state, ctx| async move {
            assert_eq!(ctx.node_id.as_str(), "start");
            Ok(NodeResult::Update(state + 1))
        })
        .add_node("branch", |state, _| async move { Ok(NodeResult::Update(state + 10)) })
        .add_node("join", |state, _| async move { Ok(NodeResult::Update(state + 100)) })
        .add_node("cmd", |_state, _| async move {
            Ok(NodeResult::Command(Command::goto(["join"]).with_update(1)))
        })
        .add_node("orphan", |state, _| async move { Ok(NodeResult::Update(state)) })
        .set_entry("start")
        .add_conditional_edges(
            "start",
            |_state: &i32| Route::new("ok"),
            [(Route::new("ok"), "branch")],
        )
        .add_waiting_edge("branch", "join")
        .add_edge("join", "cmd")
        .with_command_destinations("cmd", ["join", tinyagents::graph::END])
        .with_node_kind("cmd", "command")
        .with_node_metadata("cmd", "owner", "test")
        .mark_subgraph("branch")
        .mark_interrupt("start")
        .mark_deferred("join");

    let builder_topology = builder.topology();
    assert_eq!(builder_topology.graph_id, "graph-contract");
    assert_eq!(builder_topology.name.as_deref(), Some("Contract Graph"));
    assert_eq!(builder_topology.entry.as_deref(), Some("start"));
    assert_eq!(builder_topology.policy.recursion_limit, 7);
    assert!(builder_topology.policy.parallel);
    assert_eq!(builder_topology.policy.max_concurrency, Some(2));
    assert_eq!(builder_topology.policy.node_timeout_ms, Some(250));
    assert!(builder_topology
        .validation
        .warnings
        .iter()
        .any(|w| w.contains("orphan")));

    let graph = builder.compile().unwrap();
    let execution = graph.run(0).await.unwrap();
    assert_eq!(execution.state, 1);
    let topology = graph.topology();
    assert_eq!(topology.nodes.iter().find(|n| n.id == "cmd").unwrap().kind.as_deref(), Some("command"));
    assert_eq!(topology.nodes.iter().find(|n| n.id == "cmd").unwrap().metadata["owner"], "test");
    assert!(topology.nodes.iter().find(|n| n.id == "branch").unwrap().subgraph);
    assert!(topology.nodes.iter().find(|n| n.id == "start").unwrap().interrupt);
    assert!(topology.nodes.iter().find(|n| n.id == "join").unwrap().deferred);
    assert_eq!(topology.waiting_edges[0].target, "join");
    assert!(topology.conditional_edges[0]
        .routes
        .iter()
        .any(|route| route.label == "ok" && route.target == "branch"));

    let json = to_json(&topology);
    let round_trip = from_json(&json).unwrap();
    assert_eq!(round_trip.graph_id, topology.graph_id);
    assert!(from_json("{not json").is_err());
    let mermaid = to_mermaid(&topology);
    assert!(mermaid.contains("flowchart TD"));
    assert!(mermaid.contains("goto"));
    assert!(mermaid.contains("barrier"));
}

#[test]
fn blueprint_export_helpers_preserve_channels_routing_and_validation() {
    let blueprint = Blueprint {
        graph_id: "bp".into(),
        start: "plan".into(),
        channels: vec![ChannelSpec {
            name: "messages".into(),
            reducer: "append".into(),
            args: vec![Literal::Str("arg".into())],
        }],
        nodes: vec![
            NodeSpec {
                name: "plan".into(),
                kind: "model".into(),
                model: Some("planner".into()),
                prompt: Some("plan".into()),
                tools: vec!["search".into()],
                routing: Routing::Conditional(vec![
                    ("ok".into(), "answer".into()),
                    ("retry".into(), "plan".into()),
                ]),
                agent: None,
                subgraph: None,
                script: None,
                input: None,
                command: None,
                sends: Vec::new(),
                join_sources: Vec::new(),
                options: Vec::new(),
                checkpoint: None,
                timeout: None,
                retry: Vec::new(),
                metadata: Vec::new(),
            },
            NodeSpec {
                name: "answer".into(),
                kind: "subgraph".into(),
                model: None,
                prompt: None,
                tools: Vec::new(),
                routing: Routing::Terminal,
                agent: None,
                subgraph: Some("child".into()),
                script: None,
                input: Some("question".into()),
                command: None,
                sends: Vec::new(),
                join_sources: Vec::new(),
                options: Vec::new(),
                checkpoint: Some("always".into()),
                timeout: Some("100".into()),
                retry: Vec::new(),
                metadata: Vec::new(),
            },
        ],
        edges: vec![EdgeSpec {
            from: "plan".into(),
            to: "answer".into(),
        }],
        defaults: vec![("recursion_limit".into(), Literal::Num(9.0))],
        ..Blueprint::default()
    };

    let topology = blueprint_to_topology(&blueprint);
    assert_eq!(topology.graph_id, "bp");
    assert_eq!(topology.entry.as_deref(), Some("plan"));
    assert_eq!(topology.recursion_limit, 9);
    assert_eq!(topology.channels[0].name, "messages");
    assert!(topology.nodes.iter().find(|n| n.id == "answer").unwrap().subgraph);
    assert!(topology.finish_nodes.contains(&"answer".to_string()));
    assert!(topology
        .conditional_edges
        .iter()
        .any(|edge| edge.from == "plan" && edge.routes.len() == 2));
    assert!(blueprint_to_json(&blueprint).contains("\"graph_id\": \"bp\""));
    assert!(blueprint_to_mermaid(&blueprint).contains("START --> n_plan"));
}

#[test]
fn resolver_reports_all_capability_kinds_and_resolve_source_binds_known_source() {
    let source = r#"
graph workflow {
  channel messages missing_reducer
  start plan
  node plan {
    kind model
    model "missing_model"
    tools ["missing_tool"]
    next route
  }
  node route {
    kind router
    model "missing_router"
    next child
  }
  node child {
    kind subgraph
    graph "missing_graph"
    next agent
  }
  node agent {
    kind subagent
    agent "missing_agent"
    next weird
  }
  node weird {
    kind made_up
    model "missing_again"
    next END
  }
}
"#;
    let program = parser::parse_str(source).unwrap();
    let resolver = Resolver::from_capabilities(
        CapabilityResolver::new().with_node_kinds(["model", "router", "subgraph", "subagent"]),
    );
    assert!(!resolver.agent_allowed("missing_agent"));
    let diagnostics = resolver.resolve_program(&program);
    let rendered: Vec<String> = diagnostics.iter().map(|d| d.render_plain()).collect();
    assert!(rendered.iter().any(|d| d.contains("unknown model `missing_model`")));
    assert!(rendered.iter().any(|d| d.contains("unknown tool `missing_tool`")));
    assert!(rendered.iter().any(|d| d.contains("unknown router `missing_router`")));
    assert!(rendered.iter().any(|d| d.contains("unknown subgraph `missing_graph`")));
    assert!(rendered.iter().any(|d| d.contains("unknown agent `missing_agent`")));
    assert!(rendered.iter().any(|d| d.contains("unknown reducer `missing_reducer`")));
    assert!(rendered.iter().any(|d| d.contains("unknown kind `made_up`")));

    let check_err = resolver
        .clone()
        .allow_agent("other")
        .check_program(&program, None)
        .unwrap_err();
    assert!(check_err.to_string().contains("unknown model"));

    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    registry
        .register_model("planner", Arc::new(MockModel::constant("ok")))
        .unwrap()
        .register_tool(Arc::new(FakeTool::returning("search", "ok")))
        .unwrap()
        .register_reducer("append")
        .unwrap()
        .register_router("route_by_score")
        .unwrap()
        .register_graph_blueprint("child", Blueprint::default())
        .unwrap()
        .register_descriptor(ComponentKind::Agent, "researcher")
        .unwrap();
    let ok_source = r#"
graph ok {
  channel messages append
  start plan
  node plan {
    kind model
    model "planner"
    tools ["search"]
    next route
  }
  node route {
    kind router
    model "route_by_score"
    next child
  }
  node child {
    kind subgraph
    graph "child"
    next agent
  }
  node agent {
    kind subagent
    agent "researcher"
    next END
  }
}
"#;
    let resolver = Resolver::from_registry(&registry);
    assert!(resolver.capabilities().model_allowed("planner"));
    assert!(resolver.agent_allowed("researcher"));
    let blueprints = resolve_source(ok_source, &registry).unwrap();
    assert_eq!(blueprints[0].graph_id, "ok");

    let bad_blueprint = Blueprint {
        graph_id: "bad".into(),
        start: "x".into(),
        channels: vec![ChannelSpec {
            name: "messages".into(),
            reducer: "missing".into(),
            args: Vec::new(),
        }],
        nodes: vec![NodeSpec {
            name: "x".into(),
            kind: "model".into(),
            model: Some("missing".into()),
            prompt: None,
            tools: Vec::new(),
            routing: Routing::Terminal,
            agent: None,
            subgraph: None,
            script: None,
            input: None,
            command: None,
            sends: Vec::new(),
            join_sources: Vec::new(),
            options: Vec::new(),
            checkpoint: None,
            timeout: None,
            retry: Vec::new(),
            metadata: Vec::new(),
        }],
        ..Blueprint::default()
    };
    assert!(resolver.resolve_blueprint(&bad_blueprint).is_err());
}
