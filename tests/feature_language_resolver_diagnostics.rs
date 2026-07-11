//! Feature tests for spanned reference resolution
//! ([`tinyagents::language::resolver::Resolver`]).
//!
//! `resolve_source` (covered elsewhere) fails fast on the first bad reference.
//! These exercise the richer [`Resolver::resolve_program`] path, which collects
//! a spanned [`Diagnostic`] for *every* offending reference at once with stable
//! diagnostic codes, plus the [`Resolver::check_program`] fold and the span-less
//! [`Resolver::resolve_blueprint`] gate.

use tinyagents::language::capability_resolver::{CapabilityResolver, DEFAULT_NODE_KINDS};
use tinyagents::language::parser::parse_str;
use tinyagents::language::resolver::Resolver;
use tinyagents::language::testkit;
use tinyagents::{Severity, TinyAgentsError};

/// A resolver that allows one model, one tool, and one reducer, with node-kind
/// validation enabled.
fn resolver() -> Resolver {
    let caps = CapabilityResolver::new()
        .with_node_kinds(DEFAULT_NODE_KINDS.iter().copied())
        .allow_model("good_model")
        .allow_tool("good_tool")
        .allow_reducer("append");
    Resolver::from_capabilities(caps)
}

const MANY_PROBLEMS: &str = r#"
graph g {
  start a
  channel facts madeup_reducer
  node a {
    model "ghost_model"
    tools ["ghost_tool"]
    next END
  }
  node b {
    kind wizard
    next END
  }
}
"#;

#[test]
fn resolve_program_collects_every_offending_reference_at_once() {
    let program = parse_str(MANY_PROBLEMS).expect("parses");
    let diagnostics = resolver().resolve_program(&program);

    // Unknown model, unknown tool, invalid node kind, and unknown reducer.
    assert_eq!(diagnostics.len(), 4, "{diagnostics:#?}");
    assert!(diagnostics.iter().all(|d| d.severity == Severity::Error));

    let codes: Vec<&str> = diagnostics
        .iter()
        .filter_map(|d| d.code.as_deref())
        .collect();
    assert!(codes.contains(&"E-rag-unknown-model"), "{codes:?}");
    assert!(codes.contains(&"E-rag-unknown-tool"), "{codes:?}");
    assert!(codes.contains(&"E-rag-invalid-node-kind"), "{codes:?}");
    assert!(codes.contains(&"E-rag-unknown-reducer"), "{codes:?}");
}

#[test]
fn each_diagnostic_carries_a_span_and_a_help_line() {
    let program = parse_str(MANY_PROBLEMS).expect("parses");
    let diagnostics = resolver().resolve_program(&program);
    for diagnostic in &diagnostics {
        assert!(diagnostic.primary.line >= 1, "{diagnostic:?}");
        assert!(diagnostic.help.is_some(), "{diagnostic:?}");
        assert!(diagnostic.primary_label.is_some(), "{diagnostic:?}");
    }
}

#[test]
fn a_fully_registered_program_produces_no_diagnostics() {
    let source = r#"
graph g {
  start a
  channel messages append
  node a {
    model "good_model"
    tools ["good_tool"]
    next END
  }
}
"#;
    let program = parse_str(source).expect("parses");
    assert!(resolver().resolve_program(&program).is_empty());
}

#[test]
fn check_program_folds_the_first_capability_failure() {
    let program = parse_str(MANY_PROBLEMS).expect("parses");
    let err = resolver()
        .check_program(&program, None)
        .expect_err("resolution fails");
    // The first offending reference is the unknown model, a capability failure.
    match err {
        TinyAgentsError::Capability(message) => {
            assert!(message.contains("ghost_model"), "{message}");
        }
        other => panic!("expected capability error, got {other:?}"),
    }
}

#[test]
fn check_program_folds_an_invalid_kind_into_a_compile_error() {
    // A program whose only problem is an unknown node kind folds into a compile
    // (not capability) error, mirroring the compiler's node-kind gate.
    let program = parse_str("graph g { start a node a { kind wizard next END } }").expect("parses");
    let err = resolver()
        .check_program(&program, None)
        .expect_err("unknown kind fails");
    match err {
        TinyAgentsError::Compile(message) => assert!(message.contains("wizard"), "{message}"),
        other => panic!("expected compile error, got {other:?}"),
    }
}

#[test]
fn resolve_blueprint_reports_an_unknown_model_as_a_capability_error() {
    let blueprint =
        testkit::blueprint(r#"graph g { start a node a { model "ghost_model" next END } }"#);
    let err = resolver()
        .resolve_blueprint(&blueprint)
        .expect_err("unknown model fails");
    match err {
        TinyAgentsError::Capability(message) => {
            assert!(message.contains("ghost_model"), "{message}")
        }
        other => panic!("expected capability error, got {other:?}"),
    }
}

#[test]
fn resolve_blueprint_reports_an_unknown_kind_as_a_compile_error() {
    let blueprint = testkit::blueprint("graph g { start a node a { kind wizard next END } }");
    let err = resolver()
        .resolve_blueprint(&blueprint)
        .expect_err("unknown kind fails");
    assert!(matches!(err, TinyAgentsError::Compile(_)), "{err:?}");
}
