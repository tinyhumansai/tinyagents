//! Registry-backed `Resolver` tests (H3): spanned diagnostics, the
//! single binding gate.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;
use crate::language::source::SourceFile;

// ---------------------------------------------------------------------------
// Registry-backed Resolver (H3): spanned diagnostics, single binding gate
// ---------------------------------------------------------------------------

use crate::language::resolver::{Resolver, resolve_source};

#[test]
fn resolver_accepts_fully_registered_blueprint() {
    let reg = full_registry();
    let program = parse_str(FULL_SOURCE).unwrap();
    let resolver = Resolver::from_registry(&reg);
    // No diagnostics: every model/tool/subgraph/router/reducer is registered.
    assert!(resolver.resolve_program(&program).is_empty());
    // And the convenience façade lowers it to a blueprint.
    let blueprints = resolve_source(FULL_SOURCE, &reg).unwrap();
    assert_eq!(blueprints[0].graph_id, "main");
}

#[test]
fn resolver_reports_unregistered_tool_with_spanned_diagnostic() {
    // `missing` is not a registered tool.
    let src = r#"
graph g {
  start a
  channel m append
  node a {
    kind agent
    model "default"
    tools ["missing"]
    next END
  }
}
"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let file = SourceFile::new("plan.rag", src);
    let resolver = Resolver::from_registry(&reg);

    let diagnostics = resolver.resolve_program(&program);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    let diag = &diagnostics[0];
    assert_eq!(diag.code.as_deref(), Some("E-rag-unknown-tool"));
    let rendered = diag.render(&file);
    assert!(
        rendered.contains("node `a` references unknown tool `missing`"),
        "{rendered}"
    );
    // The diagnostic carries a caret pointing at the offending node span.
    assert!(rendered.contains('^'), "{rendered}");
    assert!(rendered.contains("--> plan.rag:"), "{rendered}");

    // `check_program` folds it into a Capability error with the rendered caret.
    let err = resolver.check_program(&program, Some(&file)).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown tool"), "{err}");
    assert!(err.to_string().contains('^'), "{err}");
}

#[test]
fn resolve_source_rejects_unregistered_tool() {
    let src = r#"graph g { start a channel m append node a { kind agent model "default" tools ["missing"] next END } }"#;
    let reg = full_registry();
    let err = resolve_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown tool"), "{err}");
}

#[test]
fn resolver_reports_unknown_node_kind_as_compile_error() {
    let src = r#"graph g { start a node a { kind wizard next END } }"#;
    let reg = full_registry();
    let err = resolve_source(src, &reg).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
    assert!(err.to_string().contains("unknown kind"), "{err}");
}

#[test]
fn resolver_reports_unregistered_agent() {
    // A `subagent` node binds its `agent "…"` reference through the registry's
    // Agent allowlist.
    let src = r#"graph g { start a node a { kind subagent agent "ghost" next END } }"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let diagnostics = Resolver::from_registry(&reg).resolve_program(&program);
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    assert_eq!(diagnostics[0].code.as_deref(), Some("E-rag-unknown-agent"));
    assert!(
        diagnostics[0].message.contains("unknown agent `ghost`"),
        "{:?}",
        diagnostics[0]
    );
}

#[test]
fn resolver_collects_multiple_diagnostics() {
    // Two independent problems: an unregistered model and an unregistered
    // reducer. `resolve_program` reports both.
    let src = r#"graph g { start a channel m ghost node a { kind model model "nope" next END } }"#;
    let reg = full_registry();
    let program = parse_str(src).unwrap();
    let diagnostics = Resolver::from_registry(&reg).resolve_program(&program);
    let codes: Vec<_> = diagnostics
        .iter()
        .filter_map(|d| d.code.as_deref())
        .collect();
    assert!(codes.contains(&"E-rag-unknown-model"), "{codes:?}");
    assert!(codes.contains(&"E-rag-unknown-reducer"), "{codes:?}");
}

#[test]
fn resolver_blueprint_path_matches_registry_binding() {
    // The span-less blueprint path mirrors the legacy gate's variants/messages.
    let reg = full_registry();
    let bp = compile(&parse_str(FULL_SOURCE).unwrap()).unwrap().remove(0);
    Resolver::from_registry(&reg)
        .resolve_blueprint(&bp)
        .unwrap();

    let bad = compile(
        &parse_str(r#"graph g { start a channel m append node a { kind subgraph model "ghost" next END } }"#)
            .unwrap(),
    )
    .unwrap()
    .remove(0);
    let err = Resolver::from_registry(&reg)
        .resolve_blueprint(&bad)
        .unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown subgraph"), "{err}");
}
