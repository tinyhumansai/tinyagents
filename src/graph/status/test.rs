//! Unit tests for [`GraphRunStatus`]: the constructor's defaults (root run id
//! mirrors the run id, zeroed progress) and terminal-state detection across the
//! lifecycle.

use super::*;
use crate::harness::ids::{ExecutionStatus, GraphId, RunId};

#[test]
fn new_status_defaults() {
    let s = GraphRunStatus::new(
        RunId::from("r1"),
        GraphId::from("g1"),
        ExecutionStatus::Pending,
    );
    assert_eq!(s.run_id, s.root_run_id);
    assert_eq!(s.current_step, 0);
    assert!(s.active_nodes.is_empty());
    assert!(!s.is_terminal());
}

#[test]
fn terminal_detection() {
    let mut s = GraphRunStatus::new(
        RunId::from("r1"),
        GraphId::from("g1"),
        ExecutionStatus::Running,
    );
    assert!(!s.is_terminal());
    s.status = ExecutionStatus::Completed;
    assert!(s.is_terminal());
}
