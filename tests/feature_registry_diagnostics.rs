//! Feature tests for the registry component/diagnostics surface
//! (`src/registry/component/` and `src/registry/diagnostics.rs`).
//!
//! These target genuinely under-covered features of the named capability
//! catalog:
//!
//! * the name-only descriptor kinds (Store, Script, Middleware, Checkpointer,
//!   TaskStore, Listener) registering, aliasing, and resolving by name;
//! * the `alias_shadows_component` health diagnostic (a `Warning`) — reachable
//!   because registering a component whose name already exists only as an alias
//!   succeeds, leaving the alias shadowed and unreachable;
//! * the serializable [`RegistrySnapshot::aliases`] projection of
//!   [`AliasBinding`]s and its stable sort order.
//!
//! Everything is offline; registration only binds names, so no model or tool is
//! ever invoked.

use std::sync::Arc;

use tinyagents::harness::providers::MockModel;
use tinyagents::harness::testkit::FakeTool;
use tinyagents::registry::{AliasBinding, CapabilityRegistry, ComponentKind, DiagnosticSeverity};

#[test]
fn name_only_descriptor_kinds_register_alias_and_resolve() {
    let mut reg = CapabilityRegistry::<()>::new();

    // Every name-only kind that routes through `register_descriptor`.
    for (kind, name) in [
        (ComponentKind::Store, "kv"),
        (ComponentKind::Script, "triage.ragsh"),
        (ComponentKind::Middleware, "redact"),
        (ComponentKind::Checkpointer, "sqlite"),
        (ComponentKind::TaskStore, "jobs"),
        (ComponentKind::Listener, "audit"),
    ] {
        reg.register_descriptor(kind, name)
            .unwrap_or_else(|e| panic!("register {kind} `{name}` failed: {e}"));
        assert!(reg.has(kind, name), "{kind} `{name}` should be present");
    }

    // Aliases scoped per kind resolve back to their canonical target.
    reg.alias(ComponentKind::Store, "cache", "kv").unwrap();
    reg.alias(ComponentKind::TaskStore, "queue", "jobs")
        .unwrap();

    assert!(reg.has(ComponentKind::Store, "cache"));
    assert_eq!(
        reg.resolve_name(ComponentKind::Store, "cache").as_deref(),
        Some("kv")
    );
    // A store alias is scoped to Store only — it does not leak into TaskStore.
    assert!(!reg.has(ComponentKind::TaskStore, "cache"));

    // Discovery lists the canonical name plus its alias, sorted and deduped.
    assert_eq!(
        reg.names_including_aliases(ComponentKind::Store),
        vec!["cache", "kv"]
    );
    // `names` excludes aliases.
    assert_eq!(reg.names(ComponentKind::Store), vec!["kv"]);
}

#[test]
fn registering_over_an_existing_alias_name_is_diagnosed_as_shadowing() {
    let mut reg = CapabilityRegistry::<()>::new();

    // `gpt` is the canonical model; `fast` is an alias for it.
    reg.register_model("gpt", Arc::new(MockModel::constant("hi")))
        .unwrap();
    reg.alias(ComponentKind::Model, "fast", "gpt").unwrap();

    // A healthy registry so far: no diagnostics.
    assert!(reg.diagnostics().is_empty());

    // Now register a *distinct* model under the name `fast`. `ensure_absent`
    // only checks canonical component metadata (not the alias table), so this
    // is allowed — but it leaves the `fast` alias shadowed and unreachable.
    reg.register_model("fast", Arc::new(MockModel::constant("distinct")))
        .expect("registering over an alias name is permitted");

    // The component now takes precedence for `fast`.
    assert!(reg.has(ComponentKind::Model, "fast"));

    // Diagnostics surface the shadowing as a Warning that names the alias.
    let diagnostics = reg.diagnostics();
    let shadow = diagnostics
        .iter()
        .find(|d| d.name == "fast" && d.severity == DiagnosticSeverity::Warning)
        .expect("a Warning should flag the shadowed alias");
    assert_eq!(shadow.kind, ComponentKind::Model);
    assert!(
        shadow.message.contains("shadows") && shadow.message.contains("unreachable"),
        "message should explain the alias is shadowed and unreachable: {}",
        shadow.message
    );
}

#[test]
fn snapshot_projects_alias_bindings_sorted_by_kind_and_alias() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt", Arc::new(MockModel::constant("hi")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool::returning("lookup", "ok")))
        .unwrap();

    // Two model aliases and one tool alias, declared out of sorted order.
    reg.alias(ComponentKind::Model, "zeta", "gpt").unwrap();
    reg.alias(ComponentKind::Tool, "search", "lookup").unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt").unwrap();

    let snapshot = reg.snapshot();

    // The `aliases` projection carries every binding, sorted by (kind, alias).
    // ComponentKind orders Model before Tool, so both model aliases precede the
    // tool alias, and within Model `default` precedes `zeta`.
    assert_eq!(
        snapshot.aliases,
        vec![
            AliasBinding {
                kind: ComponentKind::Model,
                alias: "default".to_string(),
                canonical: "gpt".to_string(),
            },
            AliasBinding {
                kind: ComponentKind::Model,
                alias: "zeta".to_string(),
                canonical: "gpt".to_string(),
            },
            AliasBinding {
                kind: ComponentKind::Tool,
                alias: "search".to_string(),
                canonical: "lookup".to_string(),
            },
        ]
    );

    // And the canonical component records its aliases for discovery.
    let gpt = snapshot
        .by_kind(ComponentKind::Model)
        .into_iter()
        .find(|m| m.id.0 == "gpt")
        .expect("gpt present");
    assert!(gpt.aliases.contains(&"default".to_string()));
    assert!(gpt.aliases.contains(&"zeta".to_string()));
}

#[test]
fn aliasing_an_unregistered_target_is_rejected() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt", Arc::new(MockModel::constant("hi")))
        .unwrap();

    // Target must be a registered component of the same kind.
    let err = reg
        .alias(ComponentKind::Model, "ghost", "missing")
        .expect_err("aliasing an unregistered target must fail");
    assert!(
        err.to_string().contains("target is not registered"),
        "unexpected error: {err}"
    );

    // Re-declaring the same alias twice is a duplicate.
    reg.alias(ComponentKind::Model, "fast", "gpt").unwrap();
    let dup = reg
        .alias(ComponentKind::Model, "fast", "gpt")
        .expect_err("re-declaring an alias must fail");
    assert!(
        dup.to_string().contains("already defined"),
        "unexpected error: {dup}"
    );
}
