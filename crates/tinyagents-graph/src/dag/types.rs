//! Types for structural validation of a dependency DAG.
//!
//! A [`DagNode`] is a borrowed view of one node — its id plus the ids it
//! declares a dependency on — so a host can validate its own node shape without
//! this crate learning anything about that shape.

/// One node of a dependency graph: an id and the ids it depends on.
///
/// Borrowed on purpose: callers hold their own node structs (workflow phases,
/// team tasks, plan steps) and project them into this view for the duration of
/// a validation call. Edges are read as `depends_on -> id`, i.e. a node runs
/// after everything it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagNode<'a> {
    /// This node's id. Ids are compared as strings and are expected to be
    /// unique; a repeat is reported as [`DagIssue::DuplicateNode`].
    pub id: &'a str,
    /// Ids this node depends on. An entry naming no declared node is reported
    /// as [`DagIssue::UnknownDependency`] and is ignored by the cycle check.
    pub depends_on: Vec<&'a str>,
}

impl<'a> DagNode<'a> {
    /// Builds a node view from an id and any iterator of dependency ids.
    ///
    /// ```
    /// use tinyagents_graph::dag::DagNode;
    ///
    /// let deps = vec!["a".to_string(), "b".to_string()];
    /// let node = DagNode::new("c", deps.iter().map(String::as_str));
    /// assert_eq!(node.depends_on, vec!["a", "b"]);
    /// ```
    pub fn new(id: &'a str, depends_on: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            id,
            depends_on: depends_on.into_iter().collect(),
        }
    }
}

/// A structural problem found in a dependency graph.
///
/// Deliberately small: these are the graph-shaped facts only. Domain rules a
/// host layers on top — "a phase must name at least one agent", "a task may not
/// depend on itself by name" — stay with the host, which knows what to call
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagIssue {
    /// Two nodes share the same id.
    DuplicateNode {
        /// The repeated id.
        id: String,
    },
    /// A node named a dependency that is not a declared node.
    UnknownDependency {
        /// The node declaring the edge.
        node: String,
        /// The id it named.
        depends_on: String,
    },
    /// The dependency edges contain at least one cycle.
    ///
    /// A self-edge (`a` depends on `a`) is a cycle and is reported this way; a
    /// host that wants to distinguish it should check for that case itself
    /// before calling.
    Cycle,
}
