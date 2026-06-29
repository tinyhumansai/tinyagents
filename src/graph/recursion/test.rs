//! Unit tests for recursion policy, frames, and depth tracking — both the
//! standalone [`RecursionStack`] contract and its enforcement inside the
//! superstep executor.

use super::*;
use crate::TinyAgentsError;
use crate::graph::builder::{GraphBuilder, NodeContext};
use crate::graph::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::graph::command::NodeResult;
use crate::graph::stream::{CollectingSink, GraphEvent};
use crate::harness::ids::{GraphId, NodeId, RunId};
use std::collections::HashMap;
use std::sync::Arc;

fn frame(run: &str, depth: usize, parent: Option<&str>) -> RecursionFrame {
    RecursionFrame {
        graph_id: GraphId::new("g"),
        node_id: None,
        run_id: RunId::new(run),
        task_id: None,
        namespace: Vec::new(),
        depth,
        parent: parent.map(RunId::new),
    }
}

#[test]
fn default_policy_is_conservative() {
    let p = RecursionPolicy::default();
    assert_eq!(p.max_depth, 25);
    assert_eq!(p.max_visits_per_node, None);
    assert_eq!(p.max_total_steps, 1000);
}

#[test]
fn frames_push_and_pop_are_symmetric() {
    let mut stack = RecursionStack::new(RecursionPolicy::default());
    assert_eq!(stack.depth(), 0);
    stack.push(frame("a", 0, None)).unwrap();
    stack.push(frame("b", 1, Some("a"))).unwrap();
    stack.push(frame("c", 2, Some("b"))).unwrap();
    assert_eq!(stack.depth(), 3);
    assert_eq!(stack.frames().len(), 3);

    assert_eq!(stack.pop().unwrap().run_id, RunId::new("c"));
    assert_eq!(stack.pop().unwrap().run_id, RunId::new("b"));
    assert_eq!(stack.pop().unwrap().run_id, RunId::new("a"));
    assert_eq!(stack.depth(), 0);
    assert!(stack.pop().is_none());
}

#[test]
fn depth_limit_trips_on_push() {
    let policy = RecursionPolicy {
        max_depth: 2,
        max_visits_per_node: None,
        max_total_steps: 1000,
    };
    let mut stack = RecursionStack::new(policy);
    stack.push(frame("a", 0, None)).unwrap();
    stack.push(frame("b", 1, Some("a"))).unwrap();
    let err = stack.push(frame("c", 2, Some("b"))).unwrap_err();
    assert!(matches!(err, TinyAgentsError::SubAgentDepth(2)));
    // The rejected frame is not retained, so the stack stays consistent.
    assert_eq!(stack.depth(), 2);
}

#[test]
fn with_frames_seeds_inherited_depth() {
    let stack =
        RecursionStack::with_frames(vec![frame("root", 0, None)], RecursionPolicy::default());
    assert_eq!(stack.depth(), 1);
    assert_eq!(stack.frames()[0].run_id, RunId::new("root"));
}

#[test]
fn node_visit_limit_trips() {
    let policy = RecursionPolicy {
        max_depth: 25,
        max_visits_per_node: Some(2),
        max_total_steps: 1000,
    };
    let stack = RecursionStack::new(policy);
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let node = NodeId::new("loop");
    stack.record_node_visit(&mut counts, &node).unwrap();
    stack.record_node_visit(&mut counts, &node).unwrap();
    let err = stack.record_node_visit(&mut counts, &node).unwrap_err();
    assert!(matches!(
        err,
        TinyAgentsError::NodeVisitLimit { limit: 2, .. }
    ));
}

#[test]
fn node_visit_unbounded_when_unset() {
    let stack = RecursionStack::new(RecursionPolicy::default());
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let node = NodeId::new("loop");
    for _ in 0..100 {
        stack.record_node_visit(&mut counts, &node).unwrap();
    }
}

#[test]
fn total_steps_check() {
    let policy = RecursionPolicy {
        max_depth: 25,
        max_visits_per_node: None,
        max_total_steps: 3,
    };
    let stack = RecursionStack::new(policy);
    assert!(stack.check_total_steps(2).is_ok());
    let err = stack.check_total_steps(3).unwrap_err();
    assert!(matches!(err, TinyAgentsError::RecursionLimit(3)));
}

// ---- Executor wiring -------------------------------------------------------

#[tokio::test]
async fn policy_max_total_steps_trips_in_executor() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("loop", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("loop")
        .add_edge("loop", "loop")
        .compile()
        .unwrap()
        .with_recursion_policy(RecursionPolicy {
            max_depth: 25,
            max_visits_per_node: None,
            max_total_steps: 3,
        });

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::RecursionLimit(3)));
}

#[tokio::test]
async fn policy_node_visit_limit_trips_in_executor() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("loop", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("loop")
        .add_edge("loop", "loop")
        .compile()
        .unwrap()
        .with_recursion_policy(RecursionPolicy {
            max_depth: 25,
            max_visits_per_node: Some(2),
            max_total_steps: 1000,
        });

    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(
        err,
        TinyAgentsError::NodeVisitLimit { limit: 2, .. }
    ));
}

#[tokio::test]
async fn run_emits_recursion_depth_event() {
    let sink = Arc::new(CollectingSink::default());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("only", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("only")
        .set_finish("only")
        .compile()
        .unwrap()
        .with_event_sink(sink.clone());

    graph.run(0).await.unwrap();
    let saw_depth = sink
        .events()
        .iter()
        .any(|e| matches!(e, GraphEvent::RecursionDepthChanged { depth: 1 }));
    assert!(saw_depth, "expected a RecursionDepthChanged {{ depth: 1 }}");
}

#[tokio::test]
async fn checkpoint_metadata_records_recursion_stack() {
    let checkpointer = Arc::new(InMemoryCheckpointer::<i32>::new());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("a")
        .set_finish("a")
        .compile()
        .unwrap()
        .with_checkpointer(checkpointer.clone());

    graph.run_with_thread("t1", 0).await.unwrap();

    let checkpoint = checkpointer.get("t1", None).await.unwrap().unwrap();
    let recursion = checkpoint
        .metadata
        .get("recursion")
        .expect("metadata carries a recursion array");
    let frames = recursion.as_array().expect("recursion is an array");
    assert_eq!(frames.len(), 1, "one frame for the top-level run");
    assert_eq!(frames[0]["depth"], 0);
    assert!(frames[0]["run_id"].is_string());
    // The original `source` field is preserved alongside the new array.
    assert_eq!(checkpoint.metadata["source"], "loop");
}
