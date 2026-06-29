//! Unit tests for the in-memory checkpointer: `put`/`get`/`list` roundtrips
//! (including latest-vs-specific lookup and missing threads) and the shared
//! storage guarantee across cheap clones.

use super::*;
use crate::harness::ids::NodeId;
use serde_json::json;

fn checkpoint(thread: &str, id: &str, parent: Option<&str>, step: usize) -> Checkpoint<i32> {
    Checkpoint {
        thread_id: thread.to_string(),
        checkpoint_id: id.to_string(),
        parent_checkpoint_id: parent.map(|s| s.to_string()),
        namespace: vec![],
        state: step as i32,
        next_nodes: vec![NodeId::from("n")],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        metadata: json!({ "source": "loop", "step": step }),
    }
}

#[tokio::test]
async fn put_get_list_roundtrip() {
    let cp = InMemoryCheckpointer::<i32>::new();

    cp.put(checkpoint("t1", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t1", "c2", Some("c1"), 2)).await.unwrap();

    // latest
    let latest = cp.get("t1", None).await.unwrap().unwrap();
    assert_eq!(latest.checkpoint_id, "c2");
    assert_eq!(latest.state, 2);

    // specific
    let first = cp.get("t1", Some("c1")).await.unwrap().unwrap();
    assert_eq!(first.checkpoint_id, "c1");

    // missing thread
    assert!(cp.get("other", None).await.unwrap().is_none());

    // list
    let list = cp.list("t1").await.unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].checkpoint_id, "c1");
    assert_eq!(list[1].parent_checkpoint_id.as_deref(), Some("c1"));
    assert_eq!(list[1].step, 2);
}

#[tokio::test]
async fn clones_share_storage() {
    let cp = InMemoryCheckpointer::<i32>::new();
    let cp2 = cp.clone();
    cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
    assert_eq!(cp2.count("t"), 1);
}
