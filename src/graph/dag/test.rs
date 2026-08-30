//! Unit tests for dependency-DAG validation: acyclic acceptance, cycle and
//! self-edge detection, dangling edges, and the duplicate-id false-cycle guard.

use super::*;

#[test]
fn linear_chain_is_acyclic() {
    let nodes = vec![
        DagNode::new("a", []),
        DagNode::new("b", ["a"]),
        DagNode::new("c", ["b"]),
    ];
    assert!(!has_cycle(&nodes));
    assert!(validate_dag(&nodes).is_empty());
}

#[test]
fn diamond_is_acyclic() {
    let nodes = vec![
        DagNode::new("root", []),
        DagNode::new("left", ["root"]),
        DagNode::new("right", ["root"]),
        DagNode::new("join", ["left", "right"]),
    ];
    assert!(!has_cycle(&nodes));
    assert!(validate_dag(&nodes).is_empty());
}

#[test]
fn empty_graph_is_acyclic() {
    assert!(!has_cycle(&[]));
    assert!(validate_dag(&[]).is_empty());
}

#[test]
fn two_node_cycle_is_detected() {
    let nodes = vec![DagNode::new("a", ["b"]), DagNode::new("b", ["a"])];
    assert!(has_cycle(&nodes));
    assert_eq!(validate_dag(&nodes), vec![DagIssue::Cycle]);
}

#[test]
fn longer_cycle_is_detected() {
    let nodes = vec![
        DagNode::new("a", ["c"]),
        DagNode::new("b", ["a"]),
        DagNode::new("c", ["b"]),
    ];
    assert!(has_cycle(&nodes));
}

#[test]
fn self_edge_is_a_cycle() {
    let nodes = vec![DagNode::new("a", ["a"])];
    assert!(has_cycle(&nodes));
    assert_eq!(validate_dag(&nodes), vec![DagIssue::Cycle]);
}

#[test]
fn cycle_off_to_the_side_is_still_detected() {
    let nodes = vec![
        DagNode::new("start", []),
        DagNode::new("x", ["y"]),
        DagNode::new("y", ["x"]),
    ];
    assert!(has_cycle(&nodes));
}

#[test]
fn dangling_edges_are_reported_and_ignored_by_the_cycle_check() {
    let nodes = vec![DagNode::new("only", ["ghost"])];
    assert!(!has_cycle(&nodes));
    assert_eq!(
        validate_dag(&nodes),
        vec![DagIssue::UnknownDependency {
            node: "only".to_string(),
            depends_on: "ghost".to_string(),
        }]
    );
}

#[test]
fn duplicate_ids_are_reported_without_a_false_cycle() {
    let nodes = vec![DagNode::new("a", []), DagNode::new("a", [])];
    assert!(!has_cycle(&nodes));
    assert_eq!(
        validate_dag(&nodes),
        vec![DagIssue::DuplicateNode {
            id: "a".to_string()
        }]
    );
}

#[test]
fn all_issue_kinds_are_reported_together() {
    let nodes = vec![
        DagNode::new("a", ["b"]),
        DagNode::new("b", ["a"]),
        DagNode::new("b", ["ghost"]),
    ];
    let issues = validate_dag(&nodes);
    assert!(issues.contains(&DagIssue::DuplicateNode {
        id: "b".to_string()
    }));
    assert!(issues.contains(&DagIssue::UnknownDependency {
        node: "b".to_string(),
        depends_on: "ghost".to_string(),
    }));
    assert!(issues.contains(&DagIssue::Cycle));
}

#[test]
fn node_view_borrows_owned_dependency_ids() {
    let deps = vec!["a".to_string(), "b".to_string()];
    let node = DagNode::new("c", deps.iter().map(String::as_str));
    assert_eq!(node.id, "c");
    assert_eq!(node.depends_on, vec!["a", "b"]);
}
