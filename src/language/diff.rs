//! Structured, human-readable diffs of two compiled [`Blueprint`]s.
//!
//! [`blueprint_diff`] compares an `old` and a `new` [`Blueprint`] and reports
//! exactly what changed: nodes added, removed, or field-changed; channels added,
//! removed, or reducer-changed; static edges added or removed; and graph-level
//! identity (`graph_id`, `start`) changes. The result is both a serializable
//! data structure ([`BlueprintDiff`]) and a renderable summary (its [`Display`]).
//!
//! This backs generated-workflow review — comparing a model-authored plan
//! against the version it replaces — and the future REPL `graph_diff` builtin
//! (Cluster I). The diff is purely a function of the two blueprints' compiled
//! shape; it does not consult provenance, clocks, or randomness, so the same
//! pair always produces the same diff.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::language::types::{Blueprint, ChannelSpec, EdgeSpec, NodeSpec, Routing};

/// A field of a node whose value changed between two blueprints.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldChange {
    /// The field name (e.g. `kind`, `model`, `tools`, `routing`).
    pub field: String,
    /// The old rendered value.
    pub old: String,
    /// The new rendered value.
    pub new: String,
}

/// A node present in both blueprints whose specification changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDiff {
    /// The node name.
    pub name: String,
    /// The fields that changed, in a stable field order.
    pub fields: Vec<FieldChange>,
}

/// A channel present in both blueprints whose reducer binding changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelDiff {
    /// The channel name.
    pub name: String,
    /// The old reducer reference (with any args rendered).
    pub old: String,
    /// The new reducer reference (with any args rendered).
    pub new: String,
}

/// A structured diff between two [`Blueprint`]s.
///
/// Empty vectors and `None` graph-level fields mean "no change". Use
/// [`BlueprintDiff::is_empty`] to test whether the two blueprints are
/// equivalent in everything this diff tracks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BlueprintDiff {
    /// `(old, new)` when the graph identifier changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_id_changed: Option<(String, String)>,
    /// `(old, new)` when the start node changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_changed: Option<(String, String)>,
    /// Node names present only in the new blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes_added: Vec<String>,
    /// Node names present only in the old blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes_removed: Vec<String>,
    /// Nodes present in both blueprints whose specification changed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes_changed: Vec<NodeDiff>,
    /// Channel names present only in the new blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels_added: Vec<String>,
    /// Channel names present only in the old blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels_removed: Vec<String>,
    /// Channels present in both blueprints whose reducer binding changed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels_changed: Vec<ChannelDiff>,
    /// Static edges present only in the new blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges_added: Vec<EdgeSpec>,
    /// Static edges present only in the old blueprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges_removed: Vec<EdgeSpec>,
}

impl BlueprintDiff {
    /// Returns true when the two blueprints are equivalent in everything this
    /// diff tracks (no additions, removals, or changes).
    pub fn is_empty(&self) -> bool {
        self.graph_id_changed.is_none()
            && self.start_changed.is_none()
            && self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.nodes_changed.is_empty()
            && self.channels_added.is_empty()
            && self.channels_removed.is_empty()
            && self.channels_changed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
    }
}

impl fmt::Display for BlueprintDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "no changes");
        }
        if let Some((old, new)) = &self.graph_id_changed {
            writeln!(f, "~ graph_id: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.start_changed {
            writeln!(f, "~ start: {old} -> {new}")?;
        }
        for name in &self.nodes_added {
            writeln!(f, "+ node {name}")?;
        }
        for name in &self.nodes_removed {
            writeln!(f, "- node {name}")?;
        }
        for node in &self.nodes_changed {
            writeln!(f, "~ node {}", node.name)?;
            for change in &node.fields {
                writeln!(f, "    {}: {} -> {}", change.field, change.old, change.new)?;
            }
        }
        for name in &self.channels_added {
            writeln!(f, "+ channel {name}")?;
        }
        for name in &self.channels_removed {
            writeln!(f, "- channel {name}")?;
        }
        for channel in &self.channels_changed {
            writeln!(
                f,
                "~ channel {}: {} -> {}",
                channel.name, channel.old, channel.new
            )?;
        }
        for edge in &self.edges_added {
            writeln!(f, "+ edge {} -> {}", edge.from, edge.to)?;
        }
        for edge in &self.edges_removed {
            writeln!(f, "- edge {} -> {}", edge.from, edge.to)?;
        }
        Ok(())
    }
}

/// Computes the structured diff that turns `old` into `new`.
///
/// Node, channel, and edge ordering follows the new blueprint for additions and
/// changes, and the old blueprint for removals, so the diff is deterministic for
/// any given pair. Provenance is ignored — only compiled topology and bindings
/// are compared.
pub fn blueprint_diff(old: &Blueprint, new: &Blueprint) -> BlueprintDiff {
    let mut diff = BlueprintDiff::default();

    if old.graph_id != new.graph_id {
        diff.graph_id_changed = Some((old.graph_id.clone(), new.graph_id.clone()));
    }
    if old.start != new.start {
        diff.start_changed = Some((old.start.clone(), new.start.clone()));
    }

    // Nodes.
    for node in &new.nodes {
        match old.nodes.iter().find(|n| n.name == node.name) {
            None => diff.nodes_added.push(node.name.clone()),
            Some(prev) => {
                let fields = node_field_changes(prev, node);
                if !fields.is_empty() {
                    diff.nodes_changed.push(NodeDiff {
                        name: node.name.clone(),
                        fields,
                    });
                }
            }
        }
    }
    for node in &old.nodes {
        if !new.nodes.iter().any(|n| n.name == node.name) {
            diff.nodes_removed.push(node.name.clone());
        }
    }

    // Channels.
    for channel in &new.channels {
        match old.channels.iter().find(|c| c.name == channel.name) {
            None => diff.channels_added.push(channel.name.clone()),
            Some(prev) => {
                let old_render = render_channel(prev);
                let new_render = render_channel(channel);
                if old_render != new_render {
                    diff.channels_changed.push(ChannelDiff {
                        name: channel.name.clone(),
                        old: old_render,
                        new: new_render,
                    });
                }
            }
        }
    }
    for channel in &old.channels {
        if !new.channels.iter().any(|c| c.name == channel.name) {
            diff.channels_removed.push(channel.name.clone());
        }
    }

    // Static edges (compared as whole from/to pairs).
    for edge in &new.edges {
        if !old
            .edges
            .iter()
            .any(|e| e.from == edge.from && e.to == edge.to)
        {
            diff.edges_added.push(edge.clone());
        }
    }
    for edge in &old.edges {
        if !new
            .edges
            .iter()
            .any(|e| e.from == edge.from && e.to == edge.to)
        {
            diff.edges_removed.push(edge.clone());
        }
    }

    diff
}

/// Renders the field-level changes between two specifications of the same node.
fn node_field_changes(old: &NodeSpec, new: &NodeSpec) -> Vec<FieldChange> {
    let mut changes = Vec::new();
    let mut push = |field: &str, old_val: String, new_val: String| {
        if old_val != new_val {
            changes.push(FieldChange {
                field: field.to_string(),
                old: old_val,
                new: new_val,
            });
        }
    };

    push("kind", old.kind.clone(), new.kind.clone());
    push("model", render_opt(&old.model), render_opt(&new.model));
    push("prompt", render_opt(&old.prompt), render_opt(&new.prompt));
    push("tools", render_list(&old.tools), render_list(&new.tools));
    push(
        "routing",
        render_routing(&old.routing),
        render_routing(&new.routing),
    );
    push("agent", render_opt(&old.agent), render_opt(&new.agent));
    push(
        "subgraph",
        render_opt(&old.subgraph),
        render_opt(&new.subgraph),
    );
    push("script", render_opt(&old.script), render_opt(&new.script));
    push("input", render_opt(&old.input), render_opt(&new.input));
    push(
        "join_sources",
        render_list(&old.join_sources),
        render_list(&new.join_sources),
    );
    push(
        "options",
        render_list(&old.options),
        render_list(&new.options),
    );
    push(
        "checkpoint",
        render_opt(&old.checkpoint),
        render_opt(&new.checkpoint),
    );
    push(
        "timeout",
        render_opt(&old.timeout),
        render_opt(&new.timeout),
    );

    changes
}

/// Renders a `Routing` for stable comparison and display.
fn render_routing(routing: &Routing) -> String {
    match routing {
        Routing::Next(target) => format!("-> {target}"),
        Routing::Terminal => "-> END".to_string(),
        Routing::Conditional(routes) => {
            let body = routes
                .iter()
                .map(|(label, target)| format!("{label} -> {target}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{ {body} }}")
        }
    }
}

/// Renders a channel's reducer binding, including any args.
fn render_channel(channel: &ChannelSpec) -> String {
    if channel.args.is_empty() {
        channel.reducer.clone()
    } else {
        let args = channel
            .args
            .iter()
            .map(|a| a.as_display())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({args})", channel.reducer)
    }
}

/// Renders an optional string field (`"(none)"` when absent).
fn render_opt(value: &Option<String>) -> String {
    value.clone().unwrap_or_else(|| "(none)".to_string())
}

/// Renders a list field as `[a, b]`.
fn render_list(values: &[String]) -> String {
    format!("[{}]", values.join(", "))
}
