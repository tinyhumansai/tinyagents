//! Regression tests for the harness store/memory backends and the file
//! checkpointer's on-disk durability.

use std::sync::Arc;

use tinyagents_graph::checkpoint::{Checkpoint, Checkpointer, FileCheckpointer};
use tinyagents_harness::ids::NodeId;
use tinyagents_harness::memory::{ChatHistory, StoreChatHistory};
use tinyagents_harness::message::Message;
use tinyagents_harness::store::{AppendStore, FileStore, JsonlAppendStore};

fn checkpoint(thread: &str, id: &str) -> Checkpoint<i32> {
    Checkpoint {
        thread_id: thread.to_string(),
        checkpoint_id: id.to_string(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state: 1,
        next_nodes: vec![NodeId::from("n")],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: serde_json::json!({ "source": "loop", "step": 1 }),
    }
}

// ── SESS-5: thread-id escaping ───────────────────────────────────────────────

/// Thread ids that differ only by case must not share a file. On APFS/NTFS
/// `Alice.jsonl` and `alice.jsonl` are the same file, so the old
/// `[A-Za-z0-9._-]` safe set silently merged two unrelated lineages.
#[tokio::test]
async fn thread_ids_differing_only_by_case_do_not_share_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let cp = FileCheckpointer::<i32>::new(dir.path());
    cp.put(checkpoint("Alice", "upper")).await.unwrap();
    cp.put(checkpoint("alice", "lower")).await.unwrap();

    let upper = cp.list("Alice").await.unwrap();
    let lower = cp.list("alice").await.unwrap();
    assert_eq!(upper.len(), 1, "`Alice` holds only its own checkpoint");
    assert_eq!(lower.len(), 1, "`alice` holds only its own checkpoint");
    assert_eq!(upper[0].checkpoint_id, "upper");
    assert_eq!(lower[0].checkpoint_id, "lower");

    // And the filenames themselves differ case-insensitively.
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_lowercase())
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(
        names.len(),
        2,
        "two distinct case-folded filenames: {names:?}"
    );
}

/// The empty thread id escapes to `""`, so its file is the dotfile `.jsonl`,
/// whose `Path::extension()` is `None` — it used to be invisible to listing.
#[tokio::test]
async fn the_empty_thread_id_is_visible_to_list_threads() {
    let dir = tempfile::tempdir().unwrap();
    let cp = FileCheckpointer::<i32>::new(dir.path());
    cp.put(checkpoint("", "c1")).await.unwrap();
    let threads = cp.list_threads().await.unwrap();
    assert!(
        threads.iter().any(|t| t.is_empty()),
        "list_threads reports the empty thread id: {threads:?}"
    );
}

// ── SESS-4: torn writes ──────────────────────────────────────────────────────

/// A crash mid-append leaves a partial final line. It used to make the whole
/// thread permanently unreadable; now the torn tail is discarded and every
/// intact record in front of it still loads.
#[tokio::test]
async fn a_torn_trailing_line_does_not_destroy_the_thread() {
    let dir = tempfile::tempdir().unwrap();
    let cp = FileCheckpointer::<i32>::new(dir.path());
    cp.put(checkpoint("t", "c1")).await.unwrap();
    cp.put(checkpoint("t", "c2")).await.unwrap();

    let path = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with("t.jsonl"))
        .expect("thread file");
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("{\"thread_id\":\"t\",\"checkpoint");
    std::fs::write(&path, text).unwrap();

    let listed = cp.list("t").await.expect("a torn tail is survivable");
    assert_eq!(listed.len(), 2, "both intact records still load");
}

/// One poisoned file must not break listing for every other thread.
#[tokio::test]
async fn one_unreadable_thread_file_does_not_break_list_threads() {
    let dir = tempfile::tempdir().unwrap();
    let cp = FileCheckpointer::<i32>::new(dir.path());
    cp.put(checkpoint("good", "c1")).await.unwrap();
    std::fs::write(dir.path().join("poison.jsonl"), "not json at all\n").unwrap();

    let threads = cp
        .list_threads()
        .await
        .expect("listing survives one bad file");
    assert!(threads.iter().any(|t| t == "good"));
}

// ── SESS-9: StoreChatHistory::append ─────────────────────────────────────────

/// `append` is a read-modify-write over the store. Concurrent appends used to
/// drop messages, giving the two `ChatHistory` backends different guarantees
/// for one trait method.
///
/// The runtime flavour is load-bearing. `#[tokio::test]` defaults to a
/// **current-thread** runtime, and `FileStore`'s methods are `async fn`s with
/// no interior `.await`, so each read-modify-write runs to completion before
/// the next task is polled — the race cannot occur and the test passes even
/// against the unsynchronised version. Only a multi-threaded runtime actually
/// interleaves them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_store_appends_do_not_lose_messages() {
    let dir = tempfile::tempdir().unwrap();
    let history = Arc::new(StoreChatHistory::new(FileStore::new(dir.path())));

    const N: usize = 24;
    let mut handles = Vec::new();
    for i in 0..N {
        let history = history.clone();
        handles.push(tokio::spawn(async move {
            history
                .append("thread", Message::user(format!("m{i}")))
                .await
                .unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let messages = history.messages("thread").await.unwrap();
    assert_eq!(
        messages.len(),
        N,
        "every concurrent append must survive; got {} of {N}",
        messages.len()
    );
}

// ── SESS-10: JsonlAppendStore offsets ────────────────────────────────────────

/// Two store instances over one directory must keep handing out fresh offsets.
/// The old per-instance counter learned the length once, so a second instance
/// re-issued offsets the first had already used.
#[tokio::test]
async fn offsets_stay_unique_across_store_instances() {
    let dir = tempfile::tempdir().unwrap();
    let first = JsonlAppendStore::new(dir.path());
    let second = JsonlAppendStore::new(dir.path());

    let mut offsets = Vec::new();
    offsets.push(first.append("s", serde_json::json!(0)).await.unwrap());
    offsets.push(second.append("s", serde_json::json!(1)).await.unwrap());
    offsets.push(first.append("s", serde_json::json!(2)).await.unwrap());
    offsets.push(second.append("s", serde_json::json!(3)).await.unwrap());

    assert_eq!(
        offsets,
        vec![0, 1, 2, 3],
        "a second instance continues the stream instead of restarting it"
    );
    assert_eq!(second.len("s").await.unwrap(), 4);
}

/// `read_from` resolves by offset, not by position — the documented contract,
/// and what the in-memory backend already did.
#[tokio::test]
async fn read_from_resolves_by_offset_not_position() {
    let dir = tempfile::tempdir().unwrap();
    let store = JsonlAppendStore::new(dir.path());
    for i in 0..4 {
        store.append("s", serde_json::json!(i)).await.unwrap();
    }
    let window = store.read_from("s", 2).await.unwrap();
    assert_eq!(
        window.iter().map(|(o, _)| *o).collect::<Vec<_>>(),
        vec![2, 3],
        "the window and the offsets labelling it must agree"
    );
    assert_eq!(window[0].1, serde_json::json!(2));
}

/// A stream whose offsets are sparse (a hand-written or partially pruned log)
/// must still window correctly. Positional `.skip` gets this wrong.
#[tokio::test]
async fn read_from_handles_a_sparse_offset_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.jsonl");
    std::fs::create_dir_all(dir.path()).unwrap();
    std::fs::write(
        &path,
        "{\"offset\":10,\"value\":\"a\",\"created_at_ms\":0}\n\
         {\"offset\":11,\"value\":\"b\",\"created_at_ms\":0}\n",
    )
    .unwrap();
    let store = JsonlAppendStore::new(dir.path());
    let window = store.read_from("sparse", 11).await.unwrap();
    assert_eq!(
        window.iter().map(|(o, _)| *o).collect::<Vec<_>>(),
        vec![11],
        "offset 11 selects the entry labelled 11, not the entry at index 11"
    );
    // A new append continues from the file's own numbering.
    assert_eq!(
        store
            .append("sparse", serde_json::json!("c"))
            .await
            .unwrap(),
        12
    );
}
