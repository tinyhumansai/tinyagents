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

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::language::types::{
    Blueprint, ChannelSpec, CommandSpec, EdgeSpec, IoFieldSpec, JoinSpec, Literal, NodeSpec,
    Routing, SendSpec,
};

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
    /// `(old, new)` when the graph-level `defaults` entries changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults_changed: Option<(String, String)>,
    /// `(old, new)` when the declared graph `input` shape changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_changed: Option<(String, String)>,
    /// `(old, new)` when the declared graph `output` shape changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_changed: Option<(String, String)>,
    /// `(old, new)` when the graph-level checkpoint policy changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_changed: Option<(String, String)>,
    /// `(old, new)` when the graph-level interrupt policy changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt_changed: Option<(String, String)>,
    /// `(old, new)` when the compiled join/barrier declarations changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joins_changed: Option<(String, String)>,
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
            && self.defaults_changed.is_none()
            && self.input_changed.is_none()
            && self.output_changed.is_none()
            && self.checkpoint_changed.is_none()
            && self.interrupt_changed.is_none()
            && self.joins_changed.is_none()
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
        if let Some((old, new)) = &self.defaults_changed {
            writeln!(f, "~ defaults: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.input_changed {
            writeln!(f, "~ input: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.output_changed {
            writeln!(f, "~ output: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.checkpoint_changed {
            writeln!(f, "~ checkpoint: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.interrupt_changed {
            writeln!(f, "~ interrupt: {old} -> {new}")?;
        }
        if let Some((old, new)) = &self.joins_changed {
            writeln!(f, "~ joins: {old} -> {new}")?;
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

    let old_defaults = render_kv_list(&old.defaults);
    let new_defaults = render_kv_list(&new.defaults);
    if old_defaults != new_defaults {
        diff.defaults_changed = Some((old_defaults, new_defaults));
    }

    let old_input = render_io_fields(&old.input);
    let new_input = render_io_fields(&new.input);
    if old_input != new_input {
        diff.input_changed = Some((old_input, new_input));
    }

    let old_output = render_io_fields(&old.output);
    let new_output = render_io_fields(&new.output);
    if old_output != new_output {
        diff.output_changed = Some((old_output, new_output));
    }

    let old_checkpoint = render_opt(&old.checkpoint);
    let new_checkpoint = render_opt(&new.checkpoint);
    if old_checkpoint != new_checkpoint {
        diff.checkpoint_changed = Some((old_checkpoint, new_checkpoint));
    }

    let old_interrupt = render_opt(&old.interrupt);
    let new_interrupt = render_opt(&new.interrupt);
    if old_interrupt != new_interrupt {
        diff.interrupt_changed = Some((old_interrupt, new_interrupt));
    }

    let old_joins = render_joins(&old.joins);
    let new_joins = render_joins(&new.joins);
    if old_joins != new_joins {
        diff.joins_changed = Some((old_joins, new_joins));
    }

    // Nodes. Both sides are indexed by name up front so matching is
    // O(old + new) instead of a nested scan; on a (hand-built) duplicate name
    // the first occurrence wins, matching the previous `find` semantics.
    // Output ordering is unchanged: additions and changes follow the new
    // blueprint's declaration order, removals follow the old blueprint's.
    let mut old_nodes: HashMap<&str, &NodeSpec> = HashMap::with_capacity(old.nodes.len());
    for node in &old.nodes {
        old_nodes.entry(node.name.as_str()).or_insert(node);
    }
    let new_node_names: HashSet<&str> = new.nodes.iter().map(|n| n.name.as_str()).collect();
    for node in &new.nodes {
        match old_nodes.get(node.name.as_str()) {
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
        if !new_node_names.contains(node.name.as_str()) {
            diff.nodes_removed.push(node.name.clone());
        }
    }

    // Channels — same indexing scheme as nodes.
    let mut old_channels: HashMap<&str, &ChannelSpec> = HashMap::with_capacity(old.channels.len());
    for channel in &old.channels {
        old_channels.entry(channel.name.as_str()).or_insert(channel);
    }
    let new_channel_names: HashSet<&str> = new.channels.iter().map(|c| c.name.as_str()).collect();
    for channel in &new.channels {
        match old_channels.get(channel.name.as_str()) {
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
        if !new_channel_names.contains(channel.name.as_str()) {
            diff.channels_removed.push(channel.name.clone());
        }
    }

    // Static edges (compared as whole from/to pairs, membership-tested via
    // per-side sets of pairs).
    let old_edge_pairs: HashSet<(&str, &str)> = old
        .edges
        .iter()
        .map(|e| (e.from.as_str(), e.to.as_str()))
        .collect();
    let new_edge_pairs: HashSet<(&str, &str)> = new
        .edges
        .iter()
        .map(|e| (e.from.as_str(), e.to.as_str()))
        .collect();
    for edge in &new.edges {
        if !old_edge_pairs.contains(&(edge.from.as_str(), edge.to.as_str())) {
            diff.edges_added.push(edge.clone());
        }
    }
    for edge in &old.edges {
        if !new_edge_pairs.contains(&(edge.from.as_str(), edge.to.as_str())) {
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
    push(
        "command",
        render_command(&old.command),
        render_command(&new.command),
    );
    push("sends", render_sends(&old.sends), render_sends(&new.sends));
    push(
        "retry",
        render_kv_list(&old.retry),
        render_kv_list(&new.retry),
    );
    push(
        "metadata",
        render_kv_list(&old.metadata),
        render_kv_list(&new.metadata),
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

/// Renders a `command { goto … update { … } }` declaration (`"(none)"` when absent).
fn render_command(command: &Option<CommandSpec>) -> String {
    match command {
        None => "(none)".to_string(),
        Some(cmd) => {
            let goto = cmd.goto.clone().unwrap_or_else(|| "(none)".to_string());
            let update = render_kv_list(&cmd.update);
            format!("{{ goto {goto}, update {update} }}")
        }
    }
}

/// Renders fanout `send` declarations.
fn render_sends(sends: &[SendSpec]) -> String {
    let body = sends
        .iter()
        .map(|s| match &s.input {
            Some(input) => format!("send {} {input}", s.target),
            None => format!("send {}", s.target),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

/// Renders a `(key, Literal)` list (used for `defaults`, `retry`, `metadata`).
fn render_kv_list(entries: &[(String, Literal)]) -> String {
    let body = entries
        .iter()
        .map(|(k, v)| format!("{k} {}", v.as_display()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {body} }}")
}

/// Renders a graph `input`/`output` shape.
fn render_io_fields(fields: &[IoFieldSpec]) -> String {
    let body = fields
        .iter()
        .map(|f| format!("{} {}", f.name, f.ty))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ {body} }}")
}

/// Renders compiled join/barrier declarations.
fn render_joins(joins: &[JoinSpec]) -> String {
    let body = joins
        .iter()
        .map(|j| format!("[{}] -> {}", j.sources.join(", "), j.target))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}
