use std::sync::Arc;
use std::time::Duration;
use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::task_run_store;

#[tokio::test(start_paused = true)]
async fn probe() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::new());
    let run = task_run_store::create_run(&store, "t", None, "c", "w").await.unwrap();
    let before = run.last_heartbeat_at.clone();
    let (_tx, rx) = tokio::sync::watch::channel(false);
    task_run_store::spawn_heartbeat_task(store.clone(), "t".into(), run.run_id.clone(), rx, Duration::from_secs(30));
    for i in 0..5 {
        tokio::time::advance(Duration::from_secs(30)).await;
        for _ in 0..16 { tokio::task::yield_now().await; }
        let now = task_run_store::get_run(&store, "t", &run.run_id).await.unwrap().unwrap();
        println!("iter {i}: before={before} after={}", now.last_heartbeat_at);
    }
}
