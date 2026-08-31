//! Checkpointer conformance for the lineage and pending-writes contracts.
//!
//! The pre-existing `tests/conformance.rs` covers put/get/list/delete_thread/
//! prune. These two suites cover what it did not — `get_tuple`,
//! `state_history`, `copy_thread`, `delete_checkpoints` and the whole
//! pending-writes protocol — against all three bundled backends, so a defect in
//! any one of them cannot hide behind the other two.

use tinyagents_graph::checkpoint::{FileCheckpointer, InMemoryCheckpointer};
use tinyagents_graph::testkit::conformance::{
    checkpointer_lineage_contract, checkpointer_writes_contract,
};

#[tokio::test]
async fn in_memory_checkpointer_satisfies_the_writes_contract() {
    checkpointer_writes_contract(InMemoryCheckpointer::<i32>::new()).await;
}

#[tokio::test]
async fn in_memory_checkpointer_satisfies_the_lineage_contract() {
    checkpointer_lineage_contract(InMemoryCheckpointer::<i32>::new()).await;
}

#[tokio::test]
async fn file_checkpointer_satisfies_the_writes_contract() {
    let dir = tempfile::tempdir().unwrap();
    checkpointer_writes_contract(FileCheckpointer::<i32>::new(dir.path())).await;
}

#[tokio::test]
async fn file_checkpointer_satisfies_the_lineage_contract() {
    let dir = tempfile::tempdir().unwrap();
    checkpointer_lineage_contract(FileCheckpointer::<i32>::new(dir.path())).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_checkpointer_satisfies_the_writes_contract() {
    use tinyagents_graph::checkpoint::SqliteCheckpointer;
    checkpointer_writes_contract(SqliteCheckpointer::<i32>::in_memory().unwrap()).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_checkpointer_satisfies_the_lineage_contract() {
    use tinyagents_graph::checkpoint::SqliteCheckpointer;
    checkpointer_lineage_contract(SqliteCheckpointer::<i32>::in_memory().unwrap()).await;
}
