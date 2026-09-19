//! Unit tests for the in-memory checkpointer: `put`/`get`/`list` roundtrips
//! (including latest-vs-specific lookup and missing threads) and the shared
//! storage guarantee across cheap clones.

use super::*;
use serde_json::json;
use tinyagents_harness::ids::NodeId;

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
        completed_routes: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
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

#[test]
fn legacy_checkpoint_json_without_new_fields_still_loads() {
    // Back-compat: a checkpoint serialized before `pending_activations` /
    // `barrier_arrivals` existed must deserialize, defaulting those fields so
    // resume falls back to `next_nodes` and an empty barrier set.
    let legacy = json!({
        "thread_id": "t",
        "checkpoint_id": "c1",
        "run_id": null,
        "parent_checkpoint_id": null,
        "namespace": [],
        "state": 7,
        "next_nodes": ["a", "b"],
        "completed_tasks": [],
        "pending_writes": [],
        "interrupts": [],
        "metadata": { "source": "loop", "step": 1 }
    });
    let cp: Checkpoint<i32> = serde_json::from_value(legacy).unwrap();
    assert_eq!(cp.state, 7);
    assert_eq!(cp.next_nodes.len(), 2);
    assert!(cp.pending_activations.is_none());
    assert!(cp.barrier_arrivals.is_empty());
}

#[test]
fn pending_activation_send_arg_roundtrips() {
    let cp = Checkpoint {
        thread_id: "t".into(),
        checkpoint_id: "c1".into(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state: 1i32,
        next_nodes: vec![NodeId::from("w")],
        completed_tasks: vec![],
        completed_routes: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: Some(vec![super::PendingActivation {
            node: NodeId::from("w"),
            send_arg: Some(json!({ "item": 42 })),
            task_id: tinyagents_harness::ids::TaskId::from("1:0:w"),
        }]),
        barrier_arrivals: vec![super::BarrierArrivals {
            node: NodeId::from("join"),
            arrived: vec![NodeId::from("p1")],
        }],
        metadata: json!({ "source": "loop", "step": 1 }),
    };
    let round: Checkpoint<i32> =
        serde_json::from_str(&serde_json::to_string(&cp).unwrap()).unwrap();
    let pa = round.pending_activations.unwrap();
    assert_eq!(pa[0].send_arg, Some(json!({ "item": 42 })));
    assert_eq!(round.barrier_arrivals[0].arrived, vec![NodeId::from("p1")]);
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
async fn get_thread_returns_full_records_in_listing_order() {
    let cp = InMemoryCheckpointer::<i32>::new();
    cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
    cp.put(checkpoint("t", "c2", Some("c1"), 2)).await.unwrap();
    cp.put(checkpoint("t", "c3", Some("c2"), 3)).await.unwrap();

    let records = cp.get_thread("t").await.unwrap();
    assert_eq!(records.len(), 3);
    // Full checkpoints (state included), in insertion order.
    let ids: Vec<&str> = records.iter().map(|c| c.checkpoint_id.as_str()).collect();
    assert_eq!(ids, vec!["c1", "c2", "c3"]);
    assert_eq!(records[2].state, 3);
    assert_eq!(records[2].parent_checkpoint_id.as_deref(), Some("c2"));

    // Unknown threads read as empty.
    assert!(cp.get_thread("missing").await.unwrap().is_empty());
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

#[tokio::test]
async fn prune_keeps_a_window_per_namespace() {
    let cp = InMemoryCheckpointer::<i32>::new();
    // An embedded subgraph writes under the parent's thread but its own
    // namespace, interleaved with the parent's records. The two lineages are
    // disjoint — no parent record ever references a child id — so a
    // thread-wide recency window would delete the child lineage outright and
    // leave the thread unresumable.
    let sub = |id: &str, parent: Option<&str>, step: usize| {
        let mut c = checkpoint("t", id, parent, step);
        c.namespace = vec!["sub".to_string()];
        c
    };
    cp.put(checkpoint("t", "p1", None, 1)).await.unwrap();
    cp.put(sub("s1", None, 1)).await.unwrap();
    cp.put(sub("s2", Some("s1"), 2)).await.unwrap();
    cp.put(checkpoint("t", "p2", Some("p1"), 2)).await.unwrap();

    // Keep the last 1 of each namespace plus its ancestors: {p2, p1} and
    // {s2, s1} — nothing is deleted, and the subgraph stays resolvable.
    let removed = cp.prune("t", 1).await.unwrap();
    assert_eq!(removed, 0);
    let child = cp
        .get_scoped("t", None, &["sub".to_string()])
        .await
        .unwrap()
        .expect("the subgraph namespace must stay resumable after prune");
    assert_eq!(child.checkpoint_id, "s2");
}

// ---- File-backed checkpointer ---------------------------------------------

mod file_backend {
    use super::checkpoint;
    use crate::Checkpoint;
    use crate::checkpoint::{CheckpointConfig, Checkpointer, FileCheckpointer};
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
    async fn legacy_uppercase_thread_files_remain_readable_and_copyable() {
        let tmp = TempDir::new("legacy-uppercase");
        let cp = FileCheckpointer::<i32>::new(tmp.path());
        cp.put(checkpoint("Run", "c1", None, 1)).await.unwrap();

        // Simulate the pre-upgrade filename scheme, which kept uppercase
        // letters unescaped (`Run.jsonl` rather than `%52un.jsonl`).
        std::fs::rename(tmp.path().join("%52un.jsonl"), tmp.path().join("Run.jsonl")).unwrap();

        assert_eq!(cp.get("Run", None).await.unwrap().unwrap().state, 1);
        cp.copy_thread("Run", "copy").await.unwrap();
        assert_eq!(cp.get("copy", None).await.unwrap().unwrap().state, 1);
        cp.delete_thread("Run").await.unwrap();
        assert!(cp.get("Run", None).await.unwrap().is_none());
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

    #[tokio::test]
    async fn get_thread_reads_the_file_once_in_order() {
        let tmp = TempDir::new("get-thread");
        let cp = FileCheckpointer::<i32>::new(tmp.path());
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t", "c2", Some("c1"), 2)).await.unwrap();

        let records = cp.get_thread("t").await.unwrap();
        let ids: Vec<&str> = records.iter().map(|c| c.checkpoint_id.as_str()).collect();
        assert_eq!(ids, vec!["c1", "c2"]);
        assert_eq!(records[1].state, 2);
        assert!(cp.get_thread("missing").await.unwrap().is_empty());
    }

    // ---- I9 regression: `list` must not decode full `State` -----------------

    /// A `State` whose `Deserialize` impl counts every call it makes, so a
    /// test can assert *how many times* something deserialized it rather than
    /// just observing the (correct either way) return value.
    #[derive(Clone, serde::Serialize)]
    struct CountedState(i32);

    /// Process-wide count of `CountedState` deserializations. `CountedState`
    /// is private to this test module, so nothing outside these tests can
    /// bump it — safe to share across the (OS-threaded) test binary without a
    /// dedicated fixture.
    static STATE_DECODE_COUNT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    impl<'de> serde::Deserialize<'de> for CountedState {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            STATE_DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            i32::deserialize(deserializer).map(CountedState)
        }
    }

    fn counted_checkpoint(
        thread: &str,
        id: &str,
        parent: Option<&str>,
        step: usize,
    ) -> Checkpoint<CountedState> {
        Checkpoint {
            thread_id: thread.to_string(),
            checkpoint_id: id.to_string(),
            run_id: None,
            parent_checkpoint_id: parent.map(|s| s.to_string()),
            namespace: vec![],
            state: CountedState(step as i32),
            next_nodes: vec![tinyagents_harness::ids::NodeId::from("n")],
            completed_tasks: vec![],
            completed_routes: vec![],
            pending_writes: vec![],
            interrupts: vec![],
            pending_activations: None,
            barrier_arrivals: vec![],
            metadata: serde_json::json!({ "source": "loop", "step": step }),
        }
    }

    #[tokio::test]
    async fn list_on_a_large_thread_does_not_decode_full_state() {
        let tmp = TempDir::new("list-header-only");
        let cp = FileCheckpointer::<CountedState>::new(tmp.path());

        let mut parent: Option<String> = None;
        for step in 0..200usize {
            let id = format!("c{step}");
            cp.put(counted_checkpoint("t", &id, parent.as_deref(), step))
                .await
                .unwrap();
            parent = Some(id);
        }

        // `put` only serializes, so the counter should already read 0 here;
        // reset explicitly anyway so this assertion is about `list` alone,
        // not an assumption about what came before it.
        STATE_DECODE_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

        let list = cp.list("t").await.unwrap();
        assert_eq!(
            list.len(),
            200,
            "list still returns every record's metadata"
        );
        assert_eq!(list[0].checkpoint_id, "c0");
        assert_eq!(list[199].checkpoint_id, "c199");
        assert_eq!(
            STATE_DECODE_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "list on a 200-record thread must not deserialize any record's full State"
        );
    }
}

// ---- SQLite-backed checkpointer (feature = "sqlite") ----------------------

#[cfg(feature = "sqlite")]
mod sqlite_backend {
    use super::checkpoint;
    use crate::checkpoint::{CheckpointConfig, Checkpointer, SqliteCheckpointer};

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
    async fn from_caller_owned_connection_and_reusable_schema() {
        use rusqlite::Connection;

        // An application owns its own connection and creates the checkpoint
        // tables from the reusable, dependency-free DDL helper.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SqliteCheckpointer::<i32>::schema_sql())
            .unwrap();

        // The checkpointer can then wrap that same connection (idempotent
        // schema) and operate over the app-owned database.
        let cp = SqliteCheckpointer::<i32>::from_connection(conn).unwrap();
        cp.put(checkpoint("t1", "c1", None, 1)).await.unwrap();
        let got = cp.get("t1", None).await.unwrap().unwrap();
        assert_eq!(got.state, 1);
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

    #[tokio::test]
    async fn get_thread_reads_rows_once_in_order() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        cp.put(checkpoint("t", "c2", Some("c1"), 2)).await.unwrap();

        let records = cp.get_thread("t").await.unwrap();
        let ids: Vec<&str> = records.iter().map(|c| c.checkpoint_id.as_str()).collect();
        assert_eq!(ids, vec!["c1", "c2"]);
        assert_eq!(records[1].state, 2);
        assert!(cp.get_thread("missing").await.unwrap().is_empty());
    }

    // ---- C3/R4: durable per-thread execution lease -------------------------

    #[tokio::test]
    async fn a_live_lease_is_refused_to_a_different_owner() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        assert!(
            cp.try_claim("t", "owner-a", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
        // A different owner is refused while the lease is still live.
        assert!(
            !cp.try_claim("t", "owner-b", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
        // The same owner re-claiming (e.g. a renew-by-reclaim) succeeds.
        assert!(
            cp.try_claim("t", "owner-a", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn a_stale_lease_past_its_ttl_is_reclaimable() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        // Claim with a TTL of 0 - expires immediately (simulates a dead
        // owner's lease that has aged out).
        assert!(
            cp.try_claim("t", "dead-owner", std::time::Duration::from_millis(0))
                .await
                .unwrap()
        );
        // A short sleep guarantees `now` has moved past the zero-TTL expiry.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        assert!(
            cp.try_claim("t", "new-owner", std::time::Duration::from_secs(60))
                .await
                .unwrap(),
            "an expired lease must be reclaimable by a different owner"
        );
        // The reclaim actually transferred ownership: the dead owner can no
        // longer renew it.
        assert!(
            !cp.renew("t", "dead-owner", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
        assert!(
            cp.renew("t", "new-owner", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn release_frees_the_lease_for_another_owner() {
        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        assert!(
            cp.try_claim("t", "owner-a", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
        cp.release("t", "owner-a").await.unwrap();
        assert!(
            cp.try_claim("t", "owner-b", std::time::Duration::from_secs(60))
                .await
                .unwrap()
        );
    }

    // ---- I8: pragmas, spawn_blocking, LIMIT-driven state_history -----------

    /// `i32` wrapper whose [`serde::Deserialize`] impl counts every decode, so
    /// tests can assert *how many* checkpoint records were actually
    /// deserialized rather than just how many the call returned — the thing a
    /// truncate-in-Rust `state_history` and a LIMIT-in-SQL one cannot be told
    /// apart by from the returned `Vec`'s length alone.
    #[derive(Clone, serde::Serialize)]
    struct CountingState(i32);

    static DECODE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    impl<'de> serde::Deserialize<'de> for CountingState {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let value = i32::deserialize(deserializer)?;
            DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(CountingState(value))
        }
    }

    fn counting_checkpoint(
        id: &str,
        parent: Option<&str>,
        step: usize,
    ) -> crate::Checkpoint<CountingState> {
        crate::Checkpoint {
            thread_id: "t".to_string(),
            checkpoint_id: id.to_string(),
            run_id: None,
            parent_checkpoint_id: parent.map(|s| s.to_string()),
            namespace: vec![],
            state: CountingState(step as i32),
            next_nodes: vec![tinyagents_harness::ids::NodeId::from("n")],
            completed_tasks: vec![],
            completed_routes: vec![],
            pending_writes: vec![],
            interrupts: vec![],
            pending_activations: None,
            barrier_arrivals: vec![],
            metadata: serde_json::json!({ "source": "loop", "step": step }),
        }
    }

    #[tokio::test]
    async fn state_history_with_limit_decodes_only_that_many_records() {
        let cp = SqliteCheckpointer::<CountingState>::in_memory().unwrap();

        // A 40-checkpoint chain: if `state_history(Some(1))` decoded the whole
        // namespace and truncated in Rust (the pre-fix behavior), the decode
        // count below would be 40, not 1.
        let mut parent: Option<String> = None;
        for step in 0..40 {
            let id = format!("c{step}");
            cp.put(counting_checkpoint(&id, parent.as_deref(), step))
                .await
                .unwrap();
            parent = Some(id);
        }

        DECODE_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
        let history = cp.state_history("t", &[], Some(1)).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].checkpoint.checkpoint_id, "c39");
        assert_eq!(
            DECODE_COUNT.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "state_history(Some(1)) must decode exactly one record via a \
             LIMIT applied in SQL, not the whole namespace truncated in Rust"
        );

        // Sanity: an unlimited call still returns (and decodes) the whole
        // chain, newest first.
        DECODE_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
        let full = cp.state_history("t", &[], None).await.unwrap();
        assert_eq!(full.len(), 40);
        assert_eq!(full[0].checkpoint.checkpoint_id, "c39");
        assert_eq!(full[39].checkpoint.checkpoint_id, "c0");
        assert_eq!(DECODE_COUNT.load(std::sync::atomic::Ordering::SeqCst), 40);
    }

    #[tokio::test]
    async fn wal_and_synchronous_pragmas_are_set_on_open() {
        // `:memory:` databases always report `journal_mode = memory`
        // regardless of the pragma, so this needs a real file — WAL mode is
        // stored in the database file's header.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoints.db");
        let cp = SqliteCheckpointer::<i32>::open(&path).unwrap();

        let journal_mode = cp.journal_mode().unwrap();
        assert_eq!(journal_mode.to_lowercase(), "wal");

        // NORMAL == 1 (OFF = 0, FULL = 2, EXTRA = 3).
        assert_eq!(cp.synchronous().unwrap(), 1);

        // The pragmas don't just read back cleanly — the checkpointer still
        // works normally under them.
        cp.put(checkpoint("t", "c1", None, 1)).await.unwrap();
        assert!(cp.get("t", None).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn put_with_writes_persists_both_in_one_call() {
        use crate::checkpoint::PendingWrite;
        use tinyagents_harness::ids::{NodeId, TaskId};

        let cp = SqliteCheckpointer::<i32>::in_memory().unwrap();
        let cfg = CheckpointConfig {
            thread_id: "t".to_string(),
            checkpoint_id: Some("c1".to_string()),
            namespace: vec![],
        };
        let writes = vec![PendingWrite {
            node: NodeId::from("n"),
            task_id: TaskId::from("task-1"),
            idx: 0,
            channel: "out".to_string(),
            payload: serde_json::json!("hi"),
        }];

        let id = cp
            .put_with_writes(checkpoint("t", "c1", None, 1), &writes)
            .await
            .unwrap();
        assert_eq!(id.as_str(), "c1");

        assert!(cp.get("t", Some("c1")).await.unwrap().is_some());
        let stored = cp.get_writes(&cfg).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].channel, "out");
    }
}
