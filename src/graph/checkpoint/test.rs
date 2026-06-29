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
        run_id: None,
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

#[test]
fn checkpoint_source_roundtrips_string_and_display() {
    for src in [
        CheckpointSource::Input,
        CheckpointSource::Loop,
        CheckpointSource::Update,
        CheckpointSource::Fork,
    ] {
        let s = src.to_string();
        assert_eq!(s, src.as_str());
        assert_eq!(CheckpointSource::parse(&s), Some(src));
        // serde wire form matches the Display/string form.
        let json = serde_json::to_string(&src).unwrap();
        assert_eq!(json, format!("\"{s}\""));
    }
    assert_eq!(CheckpointSource::parse("nope"), None);
}

#[test]
fn durability_mode_defaults_to_sync() {
    assert_eq!(DurabilityMode::default(), DurabilityMode::Sync);
}

#[tokio::test]
async fn list_metadata_parses_source_enum() {
    let cp = InMemoryCheckpointer::<i32>::new();
    let mut c = checkpoint("t1", "c1", None, 0);
    c.metadata = json!({ "source": "input", "step": 0 });
    cp.put(c).await.unwrap();
    // Unknown/missing source falls back to `loop`.
    let mut c2 = checkpoint("t1", "c2", Some("c1"), 1);
    c2.metadata = json!({ "step": 1 });
    cp.put(c2).await.unwrap();

    let list = cp.list("t1").await.unwrap();
    assert_eq!(list[0].source, CheckpointSource::Input);
    assert_eq!(list[1].source, CheckpointSource::Loop);
}

#[tokio::test]
async fn get_tuple_composes_config_and_parent() {
    let cp = InMemoryCheckpointer::<i32>::new();
    cp.put(checkpoint("t1", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t1", "c2", Some("c1"), 2)).await.unwrap();

    // Latest tuple resolves the concrete id and its parent config.
    let tuple = cp
        .get_tuple(CheckpointConfig::latest("t1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(tuple.config.checkpoint_id.as_deref(), Some("c2"));
    assert_eq!(tuple.checkpoint.checkpoint_id, "c2");
    let parent = tuple.parent_config.unwrap();
    assert_eq!(parent.checkpoint_id.as_deref(), Some("c1"));
    assert_eq!(parent.thread_id, "t1");

    // The root checkpoint has no parent config.
    let root = cp
        .get_tuple(CheckpointConfig {
            thread_id: "t1".to_string(),
            checkpoint_id: Some("c1".to_string()),
            namespace: vec![],
        })
        .await
        .unwrap()
        .unwrap();
    assert!(root.parent_config.is_none());

    // Missing thread yields no tuple.
    assert!(
        cp.get_tuple(CheckpointConfig::latest("missing"))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn list_threads_and_delete_thread() {
    let cp = InMemoryCheckpointer::<i32>::new();
    cp.put(checkpoint("a", "a1", None, 1)).await.unwrap();
    cp.put(checkpoint("b", "b1", None, 1)).await.unwrap();

    let mut threads = cp.list_threads().await.unwrap();
    threads.sort();
    assert_eq!(threads, vec!["a".to_string(), "b".to_string()]);

    cp.delete_thread("a").await.unwrap();
    assert_eq!(cp.list_threads().await.unwrap(), vec!["b".to_string()]);
    assert!(cp.get("a", None).await.unwrap().is_none());
    // Deleting a missing thread is a no-op.
    cp.delete_thread("missing").await.unwrap();
}

#[tokio::test]
async fn delete_by_run_removes_only_matching_run() {
    let cp = InMemoryCheckpointer::<i32>::new();
    let mut c1 = checkpoint("t", "c1", None, 1);
    c1.run_id = Some("run-1".to_string());
    let mut c2 = checkpoint("t", "c2", Some("c1"), 2);
    c2.run_id = Some("run-2".to_string());
    let mut c3 = checkpoint("t", "c3", Some("c2"), 3);
    c3.run_id = Some("run-2".to_string());
    cp.put(c1).await.unwrap();
    cp.put(c2).await.unwrap();
    cp.put(c3).await.unwrap();

    let removed = cp.delete_by_run("t", "run-2").await.unwrap();
    assert_eq!(removed, 2);
    let remaining: Vec<String> = cp
        .list("t")
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.checkpoint_id)
        .collect();
    assert_eq!(remaining, vec!["c1".to_string()]);
    // Records with no run id are never matched.
    assert_eq!(cp.delete_by_run("t", "run-1").await.unwrap(), 1);
}

#[tokio::test]
async fn copy_thread_preserves_lineage() {
    let cp = InMemoryCheckpointer::<i32>::new();
    cp.put(checkpoint("src", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("src", "c2", Some("c1"), 2))
        .await
        .unwrap();
    cp.put(checkpoint("src", "c3", Some("c2"), 3))
        .await
        .unwrap();

    cp.copy_thread("src", "dst").await.unwrap();

    // The source is untouched.
    assert_eq!(cp.count("src"), 3);

    // The target carries the same records (ids + parent chain) under a new
    // thread id, so time-travel walks the copied thread identically.
    let copied = cp.list("dst").await.unwrap();
    assert_eq!(copied.len(), 3);
    assert!(copied.iter().all(|m| m.thread_id == "dst"));
    assert_eq!(copied[0].checkpoint_id, "c1");
    assert_eq!(copied[0].parent_checkpoint_id, None);
    assert_eq!(copied[2].checkpoint_id, "c3");
    assert_eq!(copied[2].parent_checkpoint_id.as_deref(), Some("c2"));

    // The copied checkpoint's state and addressing are intact.
    let tip = cp.get("dst", None).await.unwrap().unwrap();
    assert_eq!(tip.thread_id, "dst");
    assert_eq!(tip.state, 3);
}

#[tokio::test]
async fn prune_keeps_window_and_full_ancestor_chain() {
    let cp = InMemoryCheckpointer::<i32>::new();
    // Linear lineage c1 <- c2 <- c3 <- c4 <- c5.
    cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t", "c2", Some("c1"), 2)).await.unwrap();
    cp.put(checkpoint("t", "c3", Some("c2"), 3)).await.unwrap();
    cp.put(checkpoint("t", "c4", Some("c3"), 4)).await.unwrap();
    cp.put(checkpoint("t", "c5", Some("c4"), 5)).await.unwrap();

    // Keep the last 2 (c4, c5). Their ancestor chain (c3, c2, c1) must be
    // retained too — a linear lineage protects everything, deleting nothing.
    let removed = cp.prune("t", 2).await.unwrap();
    assert_eq!(removed, 0);
    assert_eq!(cp.count("t"), 5);
}

#[tokio::test]
async fn prune_drops_off_lineage_branches() {
    let cp = InMemoryCheckpointer::<i32>::new();
    // c1 is the shared root. A dead fork b2 branches off c1 and is never an
    // ancestor of the kept tip; the live spine is c1 <- m2 <- m3.
    cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t", "b2", Some("c1"), 2)).await.unwrap();
    cp.put(checkpoint("t", "m2", Some("c1"), 3)).await.unwrap();
    cp.put(checkpoint("t", "m3", Some("m2"), 4)).await.unwrap();

    // Keep the last 1 (m3). Protected = {m3} ∪ ancestors {m2, c1}. The dead
    // fork b2 is not an ancestor of anything kept, so it is pruned, but the
    // ancestor chain a kept delta depends on (m2, c1) survives.
    let removed = cp.prune("t", 1).await.unwrap();
    assert_eq!(removed, 1);
    let remaining: std::collections::HashSet<String> = cp
        .list("t")
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.checkpoint_id)
        .collect();
    assert_eq!(
        remaining,
        ["c1", "m2", "m3"].iter().map(|s| s.to_string()).collect()
    );
}

#[tokio::test]
async fn prune_zero_keeps_latest_and_its_chain() {
    let cp = InMemoryCheckpointer::<i32>::new();
    cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t", "c2", Some("c1"), 2)).await.unwrap();

    // keep_last == 0 is clamped to 1: the latest checkpoint (and its ancestor
    // chain) is always retained so the thread stays resumable.
    let removed = cp.prune("t", 0).await.unwrap();
    assert_eq!(removed, 0);
    assert_eq!(cp.count("t"), 2);
}

// ---- File-backed checkpointer ---------------------------------------------

mod file_backend {
    use super::checkpoint;
    use crate::graph::checkpoint::{CheckpointConfig, Checkpointer, FileCheckpointer};
    use std::path::PathBuf;

    /// A unique-per-test temp dir derived from the test name + pid (no clock).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(test_name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "tinyagents-ckpt-{}-{}",
                test_name,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn put_get_list_roundtrip_survives_a_fresh_handle() {
        let tmp = TempDir::new("roundtrip");
        let cp = FileCheckpointer::<i32>::new(tmp.path());

        cp.put(checkpoint("t1", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t1", "c2", Some("c1"), 2)).await.unwrap();

        // A brand-new handle over the same dir reads what was persisted —
        // proving the records hit disk rather than living in memory.
        let reopened = FileCheckpointer::<i32>::new(tmp.path());
        let latest = reopened.get("t1", None).await.unwrap().unwrap();
        assert_eq!(latest.checkpoint_id, "c2");
        assert_eq!(latest.state, 2);

        let first = reopened.get("t1", Some("c1")).await.unwrap().unwrap();
        assert_eq!(first.checkpoint_id, "c1");
        assert!(reopened.get("t1", Some("nope")).await.unwrap().is_none());
        assert!(reopened.get("missing", None).await.unwrap().is_none());

        let list = reopened.list("t1").await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].checkpoint_id, "c1");
        assert_eq!(list[1].parent_checkpoint_id.as_deref(), Some("c1"));
        assert_eq!(list[1].step, 2);

        // The tuple convenience composes config + parent from the persisted record.
        let tuple = reopened
            .get_tuple(CheckpointConfig::latest("t1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tuple.config.checkpoint_id.as_deref(), Some("c2"));
        assert_eq!(
            tuple.parent_config.unwrap().checkpoint_id.as_deref(),
            Some("c1")
        );
    }

    #[tokio::test]
    async fn list_threads_and_delete_thread_track_files() {
        let tmp = TempDir::new("threads");
        // A thread id with separators/spaces exercises filename escaping.
        let cp = FileCheckpointer::<i32>::new(tmp.path());
        cp.put(checkpoint("a/b c", "x1", None, 1)).await.unwrap();
        cp.put(checkpoint("b", "b1", None, 1)).await.unwrap();

        let mut threads = cp.list_threads().await.unwrap();
        threads.sort();
        assert_eq!(threads, vec!["a/b c".to_string(), "b".to_string()]);

        cp.delete_thread("a/b c").await.unwrap();
        assert_eq!(cp.list_threads().await.unwrap(), vec!["b".to_string()]);
        assert!(cp.get("a/b c", None).await.unwrap().is_none());
        // Deleting a missing thread is a no-op.
        cp.delete_thread("missing").await.unwrap();
    }

    #[tokio::test]
    async fn prune_rewrites_the_thread_file() {
        let tmp = TempDir::new("prune");
        let cp = FileCheckpointer::<i32>::new(tmp.path());
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t", "b2", Some("c1"), 2)).await.unwrap();
        cp.put(checkpoint("t", "m2", Some("c1"), 3)).await.unwrap();
        cp.put(checkpoint("t", "m3", Some("m2"), 4)).await.unwrap();

        // Keep last 1 (m3) + its ancestors (m2, c1); the dead fork b2 is pruned.
        let removed = cp.prune("t", 1).await.unwrap();
        assert_eq!(removed, 1);
        let remaining: std::collections::HashSet<String> = cp
            .list("t")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.checkpoint_id)
            .collect();
        assert_eq!(
            remaining,
            ["c1", "m2", "m3"].iter().map(|s| s.to_string()).collect()
        );

        // Deleting everything removes the underlying file, so the thread drops
        // out of the listing.
        cp.delete_checkpoints("t", &["c1".into(), "m2".into(), "m3".into()])
            .await
            .unwrap();
        assert!(cp.list_threads().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn copy_thread_rewrites_thread_ids_on_disk() {
        let tmp = TempDir::new("copy");
        let cp = FileCheckpointer::<i32>::new(tmp.path());
        cp.put(checkpoint("src", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("src", "c2", Some("c1"), 2))
            .await
            .unwrap();
        cp.put(checkpoint("src", "c3", Some("c2"), 3))
            .await
            .unwrap();

        cp.copy_thread("src", "dst").await.unwrap();

        // Source untouched.
        assert_eq!(cp.list("src").await.unwrap().len(), 3);

        // Target carries the same lineage under the new thread id.
        let copied = cp.list("dst").await.unwrap();
        assert_eq!(copied.len(), 3);
        assert!(copied.iter().all(|m| m.thread_id == "dst"));
        assert_eq!(copied[2].checkpoint_id, "c3");
        assert_eq!(copied[2].parent_checkpoint_id.as_deref(), Some("c2"));
        let tip = cp.get("dst", None).await.unwrap().unwrap();
        assert_eq!(tip.thread_id, "dst");
        assert_eq!(tip.state, 3);
    }
}

// ---- SQLite-backed checkpointer (feature = "sqlite") ----------------------

#[cfg(feature = "sqlite")]
mod sqlite_backend {
    use super::checkpoint;
    use crate::graph::checkpoint::{CheckpointConfig, Checkpointer, SqliteCheckpointer};

    #[tokio::test]
    async fn put_get_list_roundtrip_in_memory() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();

        cp.put(checkpoint("t1", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t1", "c2", Some("c1"), 2)).await.unwrap();

        // latest
        let latest = cp.get("t1", None).await.unwrap().unwrap();
        assert_eq!(latest.checkpoint_id, "c2");
        assert_eq!(latest.state, 2);

        // specific
        let first = cp.get("t1", Some("c1")).await.unwrap().unwrap();
        assert_eq!(first.checkpoint_id, "c1");

        // missing checkpoint + missing thread
        assert!(cp.get("t1", Some("nope")).await.unwrap().is_none());
        assert!(cp.get("missing", None).await.unwrap().is_none());

        // list preserves insertion order + projects metadata from columns
        let list = cp.list("t1").await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].checkpoint_id, "c1");
        assert_eq!(list[1].parent_checkpoint_id.as_deref(), Some("c1"));
        assert_eq!(list[1].step, 2);

        // get_tuple composes config + parent from the persisted record
        let tuple = cp
            .get_tuple(CheckpointConfig::latest("t1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tuple.config.checkpoint_id.as_deref(), Some("c2"));
        assert_eq!(
            tuple.parent_config.unwrap().checkpoint_id.as_deref(),
            Some("c1")
        );
    }

    #[tokio::test]
    async fn clones_share_the_in_memory_database() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        let cp2 = cp.clone();
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        // The clone observes the write because both share one connection.
        assert!(cp2.get("t", None).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn list_threads_and_delete_thread() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        cp.put(checkpoint("a", "a1", None, 1)).await.unwrap();
        cp.put(checkpoint("b", "b1", None, 1)).await.unwrap();

        let mut threads = cp.list_threads().await.unwrap();
        threads.sort();
        assert_eq!(threads, vec!["a".to_string(), "b".to_string()]);

        cp.delete_thread("a").await.unwrap();
        assert_eq!(cp.list_threads().await.unwrap(), vec!["b".to_string()]);
        assert!(cp.get("a", None).await.unwrap().is_none());
        // Deleting a missing thread is a no-op.
        cp.delete_thread("missing").await.unwrap();
    }

    #[tokio::test]
    async fn prune_keeps_window_and_ancestor_chain() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        // c1 shared root; b2 is a dead fork; live spine c1 <- m2 <- m3.
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t", "b2", Some("c1"), 2)).await.unwrap();
        cp.put(checkpoint("t", "m2", Some("c1"), 3)).await.unwrap();
        cp.put(checkpoint("t", "m3", Some("m2"), 4)).await.unwrap();

        let removed = cp.prune("t", 1).await.unwrap();
        assert_eq!(removed, 1);
        let remaining: std::collections::HashSet<String> = cp
            .list("t")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.checkpoint_id)
            .collect();
        assert_eq!(
            remaining,
            ["c1", "m2", "m3"].iter().map(|s| s.to_string()).collect()
        );
    }

    #[tokio::test]
    async fn delete_by_run_removes_only_matching_run() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        let mut c1 = checkpoint("t", "c1", None, 1);
        c1.run_id = Some("run-1".to_string());
        let mut c2 = checkpoint("t", "c2", Some("c1"), 2);
        c2.run_id = Some("run-2".to_string());
        cp.put(c1).await.unwrap();
        cp.put(c2).await.unwrap();

        assert_eq!(cp.delete_by_run("t", "run-2").await.unwrap(), 1);
        let remaining: Vec<String> = cp
            .list("t")
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.checkpoint_id)
            .collect();
        assert_eq!(remaining, vec!["c1".to_string()]);
    }

    #[tokio::test]
    async fn copy_thread_preserves_lineage() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        cp.put(checkpoint("src", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("src", "c2", Some("c1"), 2))
            .await
            .unwrap();
        cp.put(checkpoint("src", "c3", Some("c2"), 3))
            .await
            .unwrap();

        cp.copy_thread("src", "dst").await.unwrap();

        // Source untouched.
        assert_eq!(cp.list("src").await.unwrap().len(), 3);

        // Target carries the same lineage under the new thread id.
        let copied = cp.list("dst").await.unwrap();
        assert_eq!(copied.len(), 3);
        assert!(copied.iter().all(|m| m.thread_id == "dst"));
        assert_eq!(copied[2].checkpoint_id, "c3");
        assert_eq!(copied[2].parent_checkpoint_id.as_deref(), Some("c2"));
        let tip = cp.get("dst", None).await.unwrap().unwrap();
        assert_eq!(tip.thread_id, "dst");
        assert_eq!(tip.state, 3);
    }
}
