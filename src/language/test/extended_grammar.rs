//! Extended-grammar tests (H2): channels+policy, command, send/join,
//! subgraph, subagent, repl_agent, interrupt, io shape,
//! checkpoint/interrupt policy.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

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
            crate::language::capability_resolver::DEFAULT_NODE_KINDS
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
        crate::language::capability_resolver::DEFAULT_NODE_KINDS
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

#[test]
fn bind_blueprint_rejects_unregistered_secondary_model_on_subagent_and_repl_agent() {
    // `subagent` and `repl_agent` nodes may carry a documented `model` field
    // alongside their primary `agent`/`script` reference. The strict gate
    // must validate it too, not silently let it through because only the
    // primary reference was classified.
    let node_kinds = || {
        crate::language::capability_resolver::DEFAULT_NODE_KINDS
            .iter()
            .map(|k| k.to_string())
    };

    let subagent_src = r#"
        graph g {
          start r
          node r {
            kind subagent
            agent "researcher"
            model "totally-unregistered"
            next END
          }
        }
    "#;
    let bp = compile(&parse_str(subagent_src).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new()
        .allow_agent("researcher")
        .with_node_kinds(node_kinds());
    let err = resolver.bind_blueprint(&bp).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown model"), "{err}");
    assert!(err.to_string().contains("totally-unregistered"), "{err}");

    let repl_agent_src = r#"
        graph g {
          start r
          node r {
            kind repl_agent
            script "triage"
            model "totally-unregistered"
            next END
          }
        }
    "#;
    let bp = compile(&parse_str(repl_agent_src).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new()
        .allow_script("triage")
        .with_node_kinds(node_kinds());
    let err = resolver.bind_blueprint(&bp).unwrap_err();
    assert!(err.to_string().contains("unknown model"), "{err}");
    assert!(err.to_string().contains("totally-unregistered"), "{err}");

    // A registered model passes, alongside the registered primary reference.
    let ok_resolver = CapabilityResolver::new()
        .allow_agent("researcher")
        .allow_model("totally-unregistered")
        .with_node_kinds(node_kinds());
    let bp = compile(&parse_str(subagent_src).unwrap())
        .unwrap()
        .remove(0);
    ok_resolver.bind_blueprint(&bp).unwrap();
}

#[test]
fn steering_block_on_subagent_node_parses() {
    // The reference doc's own worked example
    // (docs/modules/expressive-language/reference.md, `subagent` section) must
    // parse: it previously failed with "unknown node item `steering`" even
    // though `steering` is documented as a supported `subagent` field and
    // README.md's grammar declares `steering_decl = "steering" object`.
    let src = r#"
        graph g {
          start research
          node research {
            kind subagent
            agent "researcher"
            steering {
              parent allow ["add_instruction", "request_status", "cancel"]
              human allow ["add_instruction", "pause", "resume", "cancel"]
              delivery "safe_boundary"
            }
            next synthesize
          }
          node synthesize {
            kind model
            next END
          }
        }
    "#;
    let program = parse_str(src).unwrap();
    let node = &program.graphs[0].nodes[0];
    assert_eq!(node.name, "research");
    let steering = node.steering.as_ref().expect("steering block parsed");
    assert_eq!(
        steering.parent_allow,
        vec!["add_instruction", "request_status", "cancel"]
    );
    assert_eq!(
        steering.human_allow,
        vec!["add_instruction", "pause", "resume", "cancel"]
    );
    assert_eq!(steering.delivery.as_deref(), Some("safe_boundary"));

    // The declaration survives compilation (lowering into the blueprint stays
    // a no-op; the point is that documented source is no longer rejected).
    compile(&program).unwrap();
}

#[test]
fn steering_block_rejects_unknown_item() {
    let src =
        "graph g { start a node a { kind subagent agent \"x\" steering { bogus 1 } next END } }";
    let err = parse_str(src).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(
                message.contains("`parent`, `human`, or `delivery`"),
                "{message}"
            );
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}
