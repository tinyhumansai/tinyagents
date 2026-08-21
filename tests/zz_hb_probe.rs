use std::sync::Arc;
use std::time::Duration;
use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::task_run_store;

#[tokio::test(flavor = "multi_thread")]
async fn probe_real_time() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
    let run = task_run_store::create_run(&store, "t", None, "c", "w").await.unwrap();
    // age it
    let mut runs = task_run_store::list_runs(&store, "t", None).await.unwrap();
    runs[0].last_heartbeat_at = "0".into();
    let key: String = "t".as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    store.put(task_run_store::RUNS_NAMESPACE, &key, serde_json::to_value(&runs).unwrap()).await.unwrap();

    let (_tx, rx) = tokio::sync::watch::channel(false);
    task_run_store::spawn_heartbeat_task(store.clone(), "t".into(), run.run_id.clone(), rx, Duration::from_millis(20));
    tokio::time::sleep(Duration::from_millis(200)).await;
    let now = task_run_store::get_run(&store, "t", &run.run_id).await.unwrap().unwrap();
    println!("after={}", now.last_heartbeat_at);
    assert_ne!(now.last_heartbeat_at, "0");
}
