//! Structural validation of a dependency DAG.
//!
//! Orchestration surfaces keep re-deriving the same graph question: given nodes
//! that each name the nodes they must run after, are the ids unique, do all the
//! edges land, and is the whole thing acyclic? Workflow phases, team task
//! boards and plan steps all ask it of different node shapes, so this module
//! takes a borrowed [`DagNode`] view rather than any one caller's struct.
//!
//! Two entry points:
//!
//! - [`has_cycle`] — the cycle question alone, for a caller that already
//!   validated ids and edges its own way (a task board admitting one new node
//!   at a time, say, where a dangling edge on an *existing* node is not the new
//!   node's fault).
//! - [`validate_dag`] — duplicates, dangling edges and cycles in one pass,
//!   returning every issue rather than the first, so a definition can be shown
//!   with all of its problems at once.
//!
//! # Semantics worth knowing
//!
//! - **Edges pointing at unknown ids are ignored by the cycle check.** They are
//!   reported as [`DagIssue::UnknownDependency`] instead. A dangling edge
//!   cannot close a loop, and letting it participate would turn one mistake
//!   into two errors.
//! - **Duplicate ids never trip a false cycle.** The graph is keyed by unique
//!   id, and reachability is compared against the unique-node count rather than
//!   the input length — otherwise a repeated id (already reported as
//!   [`DagIssue::DuplicateNode`]) would look like an unreachable node and read
//!   as a cycle.
//! - **A self-edge is a cycle.** `a` depending on `a` yields
//!   [`DagIssue::Cycle`]. Callers that surface self-dependency as its own
//!   diagnostic should test for it before calling.
//!
//! The algorithm is Kahn's: count indegrees, drain the zero-indegree frontier,
//! and compare the number of nodes drained against the number of nodes. Runs in
//! O(V + E), allocates only the working maps, and touches no graph runtime
//! state — this module is pure structure and is usable before anything is
//! compiled or executed.
//!
//! ```
//! use tinyagents::graph::dag::{DagIssue, DagNode, validate_dag};
//!
//! let nodes = vec![
//!     DagNode::new("plan", []),
//!     DagNode::new("build", ["plan"]),
//!     DagNode::new("ship", ["build"]),
//! ];
//! assert!(validate_dag(&nodes).is_empty());
//!
//! let looped = vec![DagNode::new("a", ["b"]), DagNode::new("b", ["a"])];
//! assert_eq!(validate_dag(&looped), vec![DagIssue::Cycle]);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

mod types;

#[cfg(test)]
mod test;

pub use types::{DagIssue, DagNode};

/// Reports whether the dependency edges contain a cycle.
///
/// Edges naming an id that is not a declared node are ignored (see the module
/// docs). Duplicate ids are collapsed to one node (first declaration wins),
/// so they cannot produce a false positive — but a caller that wants to add
/// edges to a node id that already exists must merge them into that one
/// declaration; a second `DagNode` with the same id has its edges silently
/// dropped instead of contributing to the cycle check (see the comment at
/// the dedupe below).
pub fn has_cycle(nodes: &[DagNode<'_>]) -> bool {
    let ids: HashSet<&str> = nodes.iter().map(|n| n.id).collect();
    let mut indegree: HashMap<&str, usize> = ids.iter().map(|&id| (id, 0)).collect();
    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

    // A repeated id is reported separately as `DagIssue::DuplicateNode`; the
    // cycle graph must still be built from exactly one declaration per id, or
    // two conflicting declarations of the same id (e.g. `a[]` and `a[b]`) get
    // merged into edges that no single declaration actually names, which can
    // fabricate a cycle that would not exist under either declaration alone.
    //
    // Consequence for callers: a caller that wants to add dependency edges to
    // a node that already exists in `nodes` must merge those edges into that
    // node's single `DagNode` declaration, not append a second `DagNode` with
    // the same id and the new edges — the second declaration's edges are
    // silently dropped here (first declaration wins), so a real cycle they
    // would have introduced goes undetected. This is unreachable in a caller
    // that only ever mints fresh, never-reused ids (e.g. a fresh UUID per
    // node), but is a live footgun for any caller that revisits an existing
    // id.
    let mut processed: HashSet<&str> = HashSet::with_capacity(ids.len());
    for node in nodes {
        if !processed.insert(node.id) {
            continue;
        }
        for &dep in &node.depends_on {
            if ids.contains(dep) {
                // Edge dep -> node: `node` runs after `dep`.
                adjacency.entry(dep).or_default().push(node.id);
                *indegree.entry(node.id).or_insert(0) += 1;
            }
        }
    }

    let mut queue: VecDeque<&str> = indegree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(children) = adjacency.get(node) {
            for &child in children {
                let entry = indegree.get_mut(child).expect("child in indegree");
                *entry -= 1;
                if *entry == 0 {
                    queue.push_back(child);
                }
            }
        }
    }

    // Compare against the unique-id count, not `nodes.len()`: the graph is keyed
    // by unique id, so duplicates (reported separately) must not read as
    // unreachable nodes.
    visited != indegree.len()
}

/// Validates ids, edges and acyclicity in one pass, returning every issue found.
///
/// Issues are ordered: duplicate ids in input order, then dangling edges in
/// input order, then a single [`DagIssue::Cycle`] if the resolvable edges close
/// a loop. An empty result means the input is a well-formed DAG.
pub fn validate_dag(nodes: &[DagNode<'_>]) -> Vec<DagIssue> {
    let mut issues = Vec::new();

    let mut seen: HashSet<&str> = HashSet::new();
    for node in nodes {
        if !seen.insert(node.id) {
            issues.push(DagIssue::DuplicateNode {
                id: node.id.to_string(),
            });
        }
    }

    for node in nodes {
        for &dep in &node.depends_on {
            if !seen.contains(dep) {
                issues.push(DagIssue::UnknownDependency {
                    node: node.id.to_string(),
                    depends_on: dep.to_string(),
                });
            }
        }
    }

    if has_cycle(nodes) {
        issues.push(DagIssue::Cycle);
    }

    issues
}
