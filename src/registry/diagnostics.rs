//! Registry introspection: serializable snapshots and health diagnostics.
//!
//! A [`CapabilityRegistry`][crate::registry::CapabilityRegistry] owns live
//! handles that cannot be serialized, but its *presence* metadata can be. This
//! module projects the registry into a durable [`RegistrySnapshot`] (for CLIs,
//! UIs, and audit logs) and surfaces [`RegistryDiagnostic`]s for alias
//! collisions and dangling aliases that the registration-time duplicate check
//! cannot catch on its own.

use serde::{Deserialize, Serialize};

use crate::registry::component::{ComponentKind, ComponentMetadata};

/// A serializable, point-in-time view of every registered component.
///
/// Produced by
/// [`CapabilityRegistry::snapshot`][crate::registry::CapabilityRegistry::snapshot].
/// Components are sorted by `(kind, name)` for stable, diff-friendly output.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistrySnapshot {
    /// All registered components' metadata, sorted by `(kind, id)`.
    pub components: Vec<ComponentMetadata>,
    /// All registered aliases, sorted by `(kind, alias)`, so a CLI/UI can
    /// enumerate the alternate names that resolve to a canonical component.
    #[serde(default)]
    pub aliases: Vec<AliasBinding>,
}

/// One alias → canonical binding in a [`RegistrySnapshot`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasBinding {
    /// The kind the alias is scoped to.
    pub kind: ComponentKind,
    /// The alternate name.
    pub alias: String,
    /// The canonical component name the alias resolves to.
    pub canonical: String,
}

impl RegistrySnapshot {
    /// Total number of registered components across all kinds.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` when nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns the metadata for every component of one kind.
    pub fn by_kind(&self, kind: ComponentKind) -> Vec<&ComponentMetadata> {
        self.components
            .iter()
            .filter(|meta| meta.kind == kind)
            .collect()
    }

    /// Number of components registered under one kind.
    pub fn count(&self, kind: ComponentKind) -> usize {
        self.components.iter().filter(|m| m.kind == kind).count()
    }

    /// Renders a Graphviz DOT document clustering components by kind.
    ///
    /// The registry does not track inter-component dependency edges, so this is
    /// a node-only clustered view — useful for visualizing what is registered.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph registry {\n  rankdir=LR;\n");
        for kind in ComponentKind::ALL {
            let members = self.by_kind(kind);
            if members.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "  subgraph cluster_{} {{\n    label=\"{}\";\n",
                kind_label(kind),
                kind_label(kind)
            ));
            for meta in members {
                out.push_str(&format!(
                    "    \"{}:{}\" [label=\"{}\"];\n",
                    kind_label(kind),
                    meta.id.0,
                    meta.id.0
                ));
            }
            out.push_str("  }\n");
        }
        out.push_str("}\n");
        out
    }
}

/// Severity of a [`RegistryDiagnostic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// A likely-unintended condition that does not break resolution.
    Warning,
    /// A condition that breaks name resolution.
    Error,
}

/// One actionable registry health finding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryDiagnostic {
    /// How serious the finding is.
    pub severity: DiagnosticSeverity,
    /// The component kind the finding concerns.
    pub kind: ComponentKind,
    /// The offending name (alias or component id).
    pub name: String,
    /// Human-readable explanation.
    pub message: String,
}

fn kind_label(kind: ComponentKind) -> &'static str {
    kind.as_str()
}

pub(crate) fn alias_shadows_component(kind: ComponentKind, alias: &str) -> RegistryDiagnostic {
    RegistryDiagnostic {
        severity: DiagnosticSeverity::Warning,
        kind,
        name: alias.to_string(),
        message: format!(
            "alias `{alias}` shadows a registered {} of the same name; \
             the component takes precedence and the alias is unreachable",
            kind_label(kind)
        ),
    }
}

pub(crate) fn name_reused_across_kinds(name: &str, kinds: &[ComponentKind]) -> RegistryDiagnostic {
    let rendered = kinds
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    RegistryDiagnostic {
        severity: DiagnosticSeverity::Warning,
        // Attribute the finding to the first kind for stable sorting.
        kind: kinds[0],
        name: name.to_string(),
        message: format!(
            "name `{name}` is registered under multiple kinds ({rendered}); \
             ensure callers disambiguate by kind"
        ),
    }
}

pub(crate) fn dangling_alias(
    kind: ComponentKind,
    alias: &str,
    canonical: &str,
) -> RegistryDiagnostic {
    RegistryDiagnostic {
        severity: DiagnosticSeverity::Error,
        kind,
        name: alias.to_string(),
        message: format!(
            "alias `{alias}` resolves to `{canonical}`, which is not a registered {}",
            kind_label(kind)
        ),
    }
}
