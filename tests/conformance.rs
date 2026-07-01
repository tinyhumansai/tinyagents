//! Storage conformance: the same contract suite applied to every built-in
//! backend, proving they behave interchangeably (gap #17).

use tinyagents::graph::checkpoint::{FileCheckpointer, InMemoryCheckpointer};
use tinyagents::graph::orchestration::{InMemoryTaskStore, JsonlTaskStore};
use tinyagents::graph::testkit::conformance::{checkpointer_contract, taskstore_contract};

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("tinyagents-conformance-{tag}"))
}

#[tokio::test]
async fn in_memory_checkpointer_satisfies_contract() {
    checkpointer_contract(InMemoryCheckpointer::<i32>::new()).await;
}

#[tokio::test]
async fn file_checkpointer_satisfies_contract() {
    let dir = temp_path("file-checkpointer");
    let _ = std::fs::remove_dir_all(&dir);
    checkpointer_contract(FileCheckpointer::<i32>::new(&dir)).await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_checkpointer_satisfies_contract() {
    use tinyagents::graph::checkpoint::SqliteCheckpointer;
    checkpointer_contract(SqliteCheckpointer::<i32>::in_memory().unwrap()).await;
}

#[test]
fn in_memory_task_store_satisfies_contract() {
    taskstore_contract(InMemoryTaskStore::new());
}

#[test]
fn jsonl_task_store_satisfies_contract() {
    let path = temp_path("jsonl-taskstore.jsonl");
    let _ = std::fs::remove_file(&path);
    taskstore_contract(JsonlTaskStore::open(&path).unwrap());
    let _ = std::fs::remove_file(&path);
}
