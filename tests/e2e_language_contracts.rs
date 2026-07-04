//! End-to-end language API contracts: source maps, diagnostics, provenance,
//! blueprint diffs, testkit assertions, and registry-backed resolution.

use std::sync::Arc;

use tinyagents::harness::providers::MockModel;
use tinyagents::harness::testkit::FakeTool;
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::{Origin, Routing, Token};
use tinyagents::{
    CapabilityRegistry, Diagnostic, Label, Severity, SourceFile, SourceId, SourceMap, Span,
    TinyAgentsError, blueprint_diff, compile_source, compile_with_provenance, resolve_source,
    testkit as language_testkit,
};

const BASE: &str = r#"
graph review_flow {
  start plan
  channel messages messages
  node plan {
    kind agent
    model "default"
    tools ["lookup"]
    routes {
      research -> review
      final -> END
    }
  }
  node review {
    kind interrupt
    next work
  }
  node work {
    kind tool_executor
  }
}
"#;

const CHANGED: &str = r#"
graph review_flow_v2 {
  start review
  channel messages append
  channel facts topic
  node plan {
    kind agent
    model "default"
    tools ["lookup", "ticket"]
    routes {
      research -> review
      final -> END
    }
  }
  node review {
    kind interrupt
    timeout "30s"
    next work
  }
  node work {
    kind tool_executor
  }
  node audit {
    kind model
    next END
  }
  work -> audit
}
"#;

fn registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();
    registry
        .register_model("default", Arc::new(MockModel::constant("ok")))
        .unwrap()
        .register_tool(Arc::new(FakeTool::returning("lookup", "found")))
        .unwrap()
        .register_tool(Arc::new(FakeTool::returning("ticket", "created")))
        .unwrap();
    registry.register_reducer("messages").unwrap();
    registry.register_reducer("append").unwrap();
    registry.register_reducer("topic").unwrap();
    registry
}

#[test]
fn source_maps_spans_and_diagnostics_render_actionable_errors() {
    let source = "graph g {\n  start missing\n  node a { kind model }\n}\n";
    let file = SourceFile::new("flow.rag", source);
    assert_eq!(file.id(), SourceId(0));
    assert_eq!(file.name(), "flow.rag");
    assert_eq!(file.line_count(), 5);
    assert_eq!(file.location(source.find("missing").unwrap()), (2, 9));
    assert_eq!(file.line_text(2), Some("  start missing"));

    let span = Span::at(
        source.find("missing").unwrap(),
        source.find("missing").unwrap() + "missing".len(),
        2,
        9,
    );
    assert_eq!(file.snippet(span), "missing");
    assert_eq!(span.len(), "missing".len());
    assert!(!span.is_empty());
    let merged = span.merge(Span::at(0, 5, 1, 1));
    assert_eq!(merged.start, 0);
    assert_eq!(merged.end, span.end);

    let diagnostic = Diagnostic::error("unknown start node", span)
        .with_code("E-rag-start")
        .with_primary_label("start target is not declared")
        .with_label(Span::at(0, 5, 1, 1), "graph begins here")
        .with_help("declare node `missing` or change the start target");
    let rendered = diagnostic.render(&file);
    assert!(rendered.contains("error[E-rag-start]"));
    assert!(rendered.contains("flow.rag:2:9"));
    assert!(rendered.contains("start target is not declared"));
    assert!(rendered.contains("help: declare node"));

    let plain = Diagnostic::warning("heads up", Span::new(4, 2))
        .with_help("plain help")
        .render_plain();
    assert!(plain.contains("warning: heads up"));
    assert!(plain.contains("--> 4:2"));
    assert_eq!(Severity::Note.label(), "note");
    assert_eq!(Label::new(span, "x"), Label::new(span, "x"));

    let err = diagnostic.clone().into_parse_error(Some(&file));
    match err {
        TinyAgentsError::Parse {
            message,
            line,
            column,
        } => {
            assert_eq!((line, column), (2, 9));
            assert!(message.contains("unknown start node"));
        }
        other => panic!("expected parse error, got {other:?}"),
    }

    let mut map = SourceMap::new();
    assert!(map.is_empty());
    let a = map.add("a.rag", "graph a {}");
    let b = map.add("b.rag", "graph b {}");
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(a).unwrap().name(), "a.rag");
    assert_eq!(map.get(b).unwrap().text(), "graph b {}");
    assert_eq!(
        map.files().map(SourceFile::name).collect::<Vec<_>>(),
        vec!["a.rag", "b.rag"]
    );

    assert_eq!(Token::Ident("node".into()).describe(), "identifier `node`");
    assert_eq!(Token::Str("x".into()).describe(), "string");
    assert_eq!(Token::Num(1.0).describe(), "number");
    assert_eq!(Token::Arrow.describe(), "`->`");
}

#[test]
fn blueprint_provenance_diff_and_testkit_contracts_are_stable() {
    let program = parse_str(BASE).expect("base parses");
    let bp = compile_with_provenance(&program, Origin::file("flow.rag"))
        .expect("base compiles")
        .remove(0);
    let provenance = bp.provenance().expect("provenance present");
    assert_eq!(provenance.origin.as_display(), "flow.rag");
    assert_eq!(provenance.nodes.len(), 3);
    assert!(provenance.nodes.iter().any(|n| n.name == "plan"));
    assert!(Origin::generated().as_display().contains("generated"));
    assert_eq!(
        Origin::generated_by("repl").as_display(),
        "generated by repl"
    );

    let plain = language_testkit::blueprint(BASE);
    assert!(plain.provenance().is_none());
    language_testkit::assert_kind(&plain, "plan", "agent");
    language_testkit::assert_route(&plain, "plan", "research", "review");
    language_testkit::assert_next(&plain, "review", "work");
    language_testkit::assert_terminal(&plain, "work");
    assert_eq!(language_testkit::node(&plain, "work").kind, "tool_executor");

    let with_provenance =
        language_testkit::blueprint_with_provenance(BASE, Origin::generated_by("model"));
    assert!(with_provenance.provenance().is_some());

    let new = language_testkit::blueprint(CHANGED);
    let diff = blueprint_diff(&plain, &new);
    assert!(!diff.is_empty());
    assert_eq!(
        diff.graph_id_changed,
        Some(("review_flow".to_string(), "review_flow_v2".to_string()))
    );
    assert_eq!(
        diff.start_changed,
        Some(("plan".to_string(), "review".to_string()))
    );
    assert!(diff.nodes_added.contains(&"audit".to_string()));
    assert!(diff.channels_added.contains(&"facts".to_string()));
    assert_eq!(diff.channels_changed[0].name, "messages");
    assert!(
        diff.edges_added
            .iter()
            .any(|e| e.from == "work" && e.to == "audit")
    );
    assert!(
        diff.nodes_changed
            .iter()
            .any(|n| n.name == "plan" && n.fields.iter().any(|f| f.field == "tools"))
    );
    let rendered = diff.to_string();
    assert!(rendered.contains("~ graph_id"));
    assert!(rendered.contains("+ node audit"));

    let round_trip: tinyagents::BlueprintDiff =
        serde_json::from_str(&serde_json::to_string(&diff).unwrap()).unwrap();
    assert_eq!(round_trip, diff);
    assert!(blueprint_diff(&plain, &plain).is_empty());
    assert_eq!(blueprint_diff(&plain, &plain).to_string(), "no changes");

    let err = language_testkit::try_compile(
        "graph broken { start ghost node a { kind model next END } }",
    )
    .expect_err("bad source fails");
    assert!(matches!(err, TinyAgentsError::Compile(_)));
}

#[test]
fn registry_bound_language_resolution_accepts_known_capabilities_and_rejects_unknown() {
    let registry = registry();
    let blueprints = compile_source(BASE, &registry).expect("compile source");
    assert_eq!(blueprints.len(), 1);
    assert_eq!(blueprints[0].graph_id, "review_flow");

    let resolved = resolve_source(BASE, &registry).expect("resolve source");
    assert_eq!(resolved[0].nodes.len(), 3);
    match &resolved[0].nodes[0].routing {
        Routing::Conditional(routes) => {
            assert!(
                routes
                    .iter()
                    .any(|(label, target)| { label == "research" && target == "review" })
            );
        }
        other => panic!("expected conditional routes, got {other:?}"),
    }

    let mut missing_tool_registry: CapabilityRegistry = CapabilityRegistry::new();
    missing_tool_registry
        .register_model("default", Arc::new(MockModel::constant("ok")))
        .unwrap();
    missing_tool_registry.register_reducer("messages").unwrap();
    let err =
        resolve_source(BASE, &missing_tool_registry).expect_err("unregistered lookup tool fails");
    match err {
        TinyAgentsError::Capability(message) => {
            assert!(message.contains("lookup"));
            assert!(message.contains("^"));
        }
        other => panic!("expected capability error, got {other:?}"),
    }
}
