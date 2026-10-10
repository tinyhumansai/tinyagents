//! Unit tests for the todo-list domain types.

use super::types::*;

fn item(content: &str, status: TodoStatus) -> TodoItem {
    TodoItem::with_status(content, status)
}

#[test]
fn status_strings_match_serialized() {
    assert_eq!(TodoStatus::Pending.as_str(), "pending");
    assert_eq!(TodoStatus::InProgress.as_str(), "in_progress");
    assert_eq!(TodoStatus::Completed.as_str(), "completed");
    for status in [
        TodoStatus::Pending,
        TodoStatus::InProgress,
        TodoStatus::Completed,
    ] {
        assert_eq!(
            serde_json::to_value(status).unwrap(),
            serde_json::Value::String(status.as_str().to_string())
        );
    }
}

#[test]
fn parse_status_accepts_aliases() {
    assert_eq!(parse_status("pending").unwrap(), TodoStatus::Pending);
    assert_eq!(parse_status("TODO").unwrap(), TodoStatus::Pending);
    assert_eq!(parse_status("in-progress").unwrap(), TodoStatus::InProgress);
    assert_eq!(
        parse_status(" in_progress ").unwrap(),
        TodoStatus::InProgress
    );
    assert_eq!(parse_status("done").unwrap(), TodoStatus::Completed);
    assert_eq!(parse_status("completed").unwrap(), TodoStatus::Completed);
    assert!(parse_status("blocked").is_err(), "kanban states are gone");
    assert!(parse_status("nope").is_err());
}

#[test]
fn item_and_list_round_trip_through_json() {
    let list = TodoList {
        thread_id: "t".into(),
        items: vec![
            item("Draft plan", TodoStatus::Completed),
            item("Write code", TodoStatus::InProgress),
        ],
        updated_at: "0".into(),
    };
    let json = serde_json::to_value(&list).unwrap();
    assert_eq!(json["threadId"], "t");
    assert_eq!(json["items"][0]["status"], "completed");
    assert_eq!(json["items"][1]["content"], "Write code");
    let back: TodoList = serde_json::from_value(json).unwrap();
    assert_eq!(back, list);
}

#[test]
fn item_status_defaults_to_pending_when_absent() {
    let back: TodoItem = serde_json::from_value(serde_json::json!({ "content": "x" })).unwrap();
    assert_eq!(back.status, TodoStatus::Pending);
}

#[test]
fn render_markdown_uses_status_markers() {
    let md = render_markdown(&[
        item("Ship it", TodoStatus::Completed),
        item("Write docs", TodoStatus::InProgress),
        item("Later", TodoStatus::Pending),
    ]);
    assert_eq!(md, "- [x] Ship it\n- [~] Write docs\n- [ ] Later");
}

#[test]
fn render_markdown_empty_is_placeholder() {
    assert_eq!(render_markdown(&[]), "_No todos yet._");
}

#[test]
fn normalise_trims_and_drops_blank_items() {
    let mut list = TodoList {
        thread_id: " t ".into(),
        items: vec![
            item("  keep  ", TodoStatus::Pending),
            item("   ", TodoStatus::Completed),
            item("also keep", TodoStatus::InProgress),
        ],
        updated_at: String::new(),
    };
    normalise_list(&mut list);
    assert_eq!(list.thread_id, "t");
    assert_eq!(list.items.len(), 2);
    assert_eq!(list.items[0].content, "keep");
    assert_eq!(list.items[1].content, "also keep");
    assert!(!list.updated_at.is_empty());
}

mod store_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;
    use serde_json::Value;
    use tokio::sync::Notify;

    use super::super::store;
    use super::super::types::{TodoItem, TodoStatus};
    use tinyagents_harness::store::{InMemoryStore, Store};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    fn items(specs: &[(&str, TodoStatus)]) -> Vec<TodoItem> {
        specs
            .iter()
            .map(|(content, status)| TodoItem::with_status(*content, *status))
            .collect()
    }

    #[tokio::test]
    async fn replace_list_clear_round_trip() {
        let s = store();
        assert!(store::list(&s, "t").await.unwrap().items.is_empty());

        let snap = store::replace(
            &s,
            "t",
            items(&[
                ("Write the RFC", TodoStatus::InProgress),
                ("Review it", TodoStatus::Pending),
            ]),
        )
        .await
        .unwrap();
        assert_eq!(snap.thread_id, "t");
        assert_eq!(snap.items.len(), 2);
        assert_eq!(snap.items[0].content, "Write the RFC");
        assert_eq!(snap.markdown, "- [~] Write the RFC\n- [ ] Review it");

        let listed = store::list(&s, "t").await.unwrap();
        assert_eq!(listed, snap);

        // A replace is wholesale: the old items are gone, not merged.
        let snap = store::replace(&s, "t", items(&[("Only this", TodoStatus::Completed)]))
            .await
            .unwrap();
        assert_eq!(snap.items.len(), 1);
        assert_eq!(snap.items[0].status, TodoStatus::Completed);

        let cleared = store::clear(&s, "t").await.unwrap();
        assert!(cleared.items.is_empty());
        assert_eq!(cleared.markdown, "_No todos yet._");
    }

    #[tokio::test]
    async fn threads_are_isolated() {
        let s = store();
        store::replace(&s, "a", items(&[("A's work", TodoStatus::Pending)]))
            .await
            .unwrap();
        assert!(store::list(&s, "b").await.unwrap().items.is_empty());
        store::clear(&s, "b").await.unwrap();
        assert_eq!(store::list(&s, "a").await.unwrap().items.len(), 1);
    }

    #[tokio::test]
    async fn get_and_delete_preserve_absent_vs_empty() {
        let s = store();
        assert!(store::get(&s, " t ").await.unwrap().is_none());

        store::clear(&s, "t").await.unwrap();
        let list = store::get(&s, "t").await.unwrap().expect("present list");
        assert!(list.items.is_empty());

        assert!(store::delete(&s, "t").await.unwrap());
        assert!(store::get(&s, "t").await.unwrap().is_none());
        assert!(!store::delete(&s, "t").await.unwrap());
    }

    #[tokio::test]
    async fn replace_rejects_two_in_progress_and_blank_thread() {
        let s = store();
        let err = store::replace(
            &s,
            "t",
            items(&[("A", TodoStatus::InProgress), ("B", TodoStatus::InProgress)]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("in_progress"), "{err}");
        assert!(
            store::list(&s, "t").await.unwrap().items.is_empty(),
            "a rejected replace leaves the list untouched"
        );
        assert!(store::replace(&s, "  ", Vec::new()).await.is_err());
        assert!(store::list(&s, "").await.is_err());
    }

    #[tokio::test]
    async fn replace_normalises_content() {
        let s = store();
        let snap = store::replace(
            &s,
            "t",
            items(&[
                ("  spaced  ", TodoStatus::Pending),
                ("", TodoStatus::Pending),
            ]),
        )
        .await
        .unwrap();
        assert_eq!(snap.items.len(), 1);
        assert_eq!(snap.items[0].content, "spaced");
    }

    /// A store whose first armed `put` parks until released, so a test can
    /// hold the thread lock inside a mutation and check that a concurrent
    /// `delete` waits at the lock instead of racing the store.
    struct BlockingPutStore {
        inner: InMemoryStore,
        armed: AtomicBool,
        first_put_started: Notify,
        release_first_put: Notify,
        first_put_released: AtomicBool,
        concurrent_access: AtomicBool,
    }

    impl BlockingPutStore {
        fn new() -> Self {
            Self {
                inner: InMemoryStore::default(),
                armed: AtomicBool::new(false),
                first_put_started: Notify::new(),
                release_first_put: Notify::new(),
                first_put_released: AtomicBool::new(false),
                concurrent_access: AtomicBool::new(false),
            }
        }

        fn note_access(&self) {
            if !self.first_put_released.load(Ordering::SeqCst) {
                self.concurrent_access.store(true, Ordering::SeqCst);
            }
        }
    }

    #[async_trait]
    impl Store for BlockingPutStore {
        async fn get(
            &self,
            namespace: &str,
            key: &str,
        ) -> tinyagents_harness::error::Result<Option<Value>> {
            self.note_access();
            self.inner.get(namespace, key).await
        }

        async fn put(
            &self,
            namespace: &str,
            key: &str,
            value: Value,
        ) -> tinyagents_harness::error::Result<()> {
            if self.armed.swap(false, Ordering::SeqCst) {
                self.first_put_started.notify_one();
                self.release_first_put.notified().await;
                self.first_put_released.store(true, Ordering::SeqCst);
            } else {
                self.note_access();
            }
            self.inner.put(namespace, key, value).await
        }

        async fn delete(
            &self,
            namespace: &str,
            key: &str,
        ) -> tinyagents_harness::error::Result<()> {
            self.note_access();
            self.inner.delete(namespace, key).await
        }

        async fn list(&self, namespace: &str) -> tinyagents_harness::error::Result<Vec<String>> {
            self.inner.list(namespace).await
        }
    }

    #[tokio::test]
    async fn delete_waits_for_an_in_flight_mutation() {
        let concrete = Arc::new(BlockingPutStore::new());
        let s: Arc<dyn Store> = concrete.clone();

        concrete.armed.store(true, Ordering::SeqCst);
        let replace_store = s.clone();
        let replace = tokio::spawn(async move {
            store::replace(&replace_store, "t", vec![TodoItem::new("original")]).await
        });
        concrete.first_put_started.notified().await;

        let delete_store = s.clone();
        let delete = tokio::spawn(async move { store::delete(&delete_store, "t").await });
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(
            !concrete.concurrent_access.load(Ordering::SeqCst),
            "delete must not enter the store while a mutation holds the thread lock"
        );

        concrete.release_first_put.notify_one();
        replace.await.unwrap().unwrap();
        assert!(delete.await.unwrap().unwrap());
        assert!(store::get(&s, "t").await.unwrap().is_none());
    }
}

mod tool_tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::super::tool::{TodoTool, todo_tools};
    use tinyagents_harness::store::{InMemoryStore, Store};
    use tinytools::{Tool, ToolContent, ToolResult, ToolRunContext};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    struct ThreadContext(Option<String>);

    impl ToolRunContext for ThreadContext {
        fn thread_id(&self) -> Option<&str> {
            self.0.as_deref()
        }
    }

    async fn run(tool: &TodoTool, thread: Option<&str>, args: serde_json::Value) -> ToolResult {
        let context = ThreadContext(thread.map(str::to_owned));
        tool.execute_with_context(args, Default::default(), Some(&context))
            .await
            .unwrap()
    }

    fn raw(result: &ToolResult) -> &serde_json::Value {
        result
            .content
            .iter()
            .find_map(|block| match block {
                ToolContent::Json { data } => Some(data),
                _ => None,
            })
            .expect("successful todo result has a JSON payload")
    }

    /// The description is what makes a model treat the list as bookkeeping
    /// rather than as the work itself: without the "same response carries the
    /// next call" rule, models write a list, stop, and wait to be prompted;
    /// without the "only after its work has run" rule they tick items off
    /// ahead of doing them.
    #[test]
    fn description_states_when_an_item_may_be_completed_and_that_writing_is_not_working() {
        let tool = TodoTool::new(store());
        let description = Tool::description(&tool);
        assert!(
            description.contains("only after its work has run"),
            "an item is completed after its result exists: {description}"
        );
        assert!(
            description.contains("bookkeeping, not work"),
            "writing the list is not the work: {description}"
        );
        assert!(
            description.contains("do the next step"),
            "the model advances after bookkeeping: {description}"
        );
        assert!(
            description.contains("unless the list is finished"),
            "a terminal completion does not require a nonexistent next step: {description}"
        );
        assert!(
            description.contains("next turn"),
            "sequential providers may advance on their next turn: {description}"
        );
        assert!(
            description.contains("at most once per assistant response"),
            "replacement calls are bounded per assistant response: {description}"
        );
    }

    #[test]
    fn todo_tools_builds_a_single_tool() {
        let tools = todo_tools(store());
        assert_eq!(tools.len(), 1);
        assert_eq!(Tool::name(tools[0].as_ref()), "todo");
    }

    #[test]
    fn schema_admits_the_documented_read_and_status_forms() {
        let tool = TodoTool::new(store());
        let schema = tool.parameters_schema();
        assert_eq!(
            schema["properties"]["todos"]["items"]["properties"]["status"]["enum"],
            json!([
                "pending",
                "todo",
                "open",
                "not_started",
                "in_progress",
                "in-progress",
                "inprogress",
                "started",
                "active",
                "completed",
                "complete",
                "done",
                "finished"
            ])
        );
        assert_eq!(
            schema["properties"]["todos"]["type"],
            json!(["array", "null"])
        );
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema["properties"].get("op").is_none());
    }

    #[tokio::test]
    async fn write_then_read_via_tool_persists_to_the_thread() {
        let tool = TodoTool::new(store());
        let res = run(
            &tool,
            Some("t"),
            json!({ "todos": [
                { "content": "Write tests", "status": "in_progress" },
                { "content": "Ship", "status": "pending" }
            ] }),
        )
        .await;
        assert!(!res.is_error, "{res:?}");
        assert_eq!(raw(&res)["threadId"], "t");
        assert_eq!(raw(&res)["todos"][0]["status"], "in_progress");
        assert_eq!(
            raw(&res)["markdown"].as_str().unwrap(),
            "- [~] Write tests\n- [ ] Ship"
        );

        // No `todos` reads the list back unchanged.
        let res = run(&tool, Some("t"), json!({})).await;
        assert_eq!(raw(&res)["todos"].as_array().unwrap().len(), 2);
        let res = run(&tool, Some("t"), json!({ "todos": null })).await;
        assert_eq!(raw(&res)["todos"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn status_aliases_and_default_are_accepted() {
        let tool = TodoTool::new(store());
        let res = run(
            &tool,
            Some("t"),
            json!({ "todos": [
                { "content": "A", "status": "done" },
                { "content": "B" }
            ] }),
        )
        .await;
        assert!(!res.is_error, "{res:?}");
        assert_eq!(raw(&res)["todos"][0]["status"], "completed");
        assert_eq!(raw(&res)["todos"][1]["status"], "pending");
    }

    #[tokio::test]
    async fn tool_requires_a_thread() {
        let tool = TodoTool::new(store());
        // Bare call (no context).
        let res = tool.execute(json!({})).await.unwrap();
        assert!(res.is_error && res.output().contains("active thread"));
        // Context without a thread id.
        let res = run(&tool, None, json!({})).await;
        assert!(res.is_error);
    }

    #[tokio::test]
    async fn malformed_items_are_soft_errors() {
        let tool = TodoTool::new(store());
        let res = run(&tool, Some("t"), json!({ "todos": "nope" })).await;
        assert!(res.is_error && res.output().contains("invalid `todos`"));
        let res = run(&tool, Some("t"), json!({ "todos": [{ "content": "  " }] })).await;
        assert!(res.is_error && res.output().contains("content"));
        let res = run(
            &tool,
            Some("t"),
            json!({ "todos": [{ "content": "x", "status": "blocked" }] }),
        )
        .await;
        assert!(res.is_error && res.output().contains("invalid status"));

        for args in [
            json!(null),
            json!([]),
            json!("not an object"),
            json!({ "op": "clear" }),
        ] {
            let res = run(&tool, Some("t"), args).await;
            assert!(res.is_error, "invalid arguments must be a tool error");
        }
    }

    /// A model that sends the retired `{"cards": ...}` shape gets a tool error
    /// naming the replacement, never a dispatch error that would end the run.
    #[tokio::test]
    async fn retired_cards_shape_is_a_soft_error_naming_todos() {
        let tool = TodoTool::new(store());
        let res = run(
            &tool,
            Some("t"),
            json!({ "cards": [{ "content": "x", "status": "todo" }] }),
        )
        .await;
        assert!(res.is_error);
        assert!(res.output().contains("pass `todos`"), "{}", res.output());
    }

    /// Production captures: models name an item's text `title`, `item`,
    /// `text`, `task` or `description` instead of `content`. Each is the
    /// content; `content` wins when several are present, and an unrelated
    /// `id` is ignored rather than rejected.
    #[tokio::test]
    async fn content_aliases_are_accepted() {
        let tool = TodoTool::new(store());
        let res = run(
            &tool,
            Some("t"),
            json!({ "todos": [
                { "id": 1, "title": "A", "status": "in_progress" },
                { "id": "2", "item": "B" },
                { "text": "C", "status": "pending" },
                { "task": "D" },
                { "description": "E" },
                { "content": "F", "title": "ignored", "description": "also ignored" },
                { "title": "  ", "task": "G" }
            ] }),
        )
        .await;
        assert!(!res.is_error, "{res:?}");
        let contents: Vec<&str> = raw(&res)["todos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["content"].as_str().unwrap())
            .collect();
        assert_eq!(contents, ["A", "B", "C", "D", "E", "F", "G"]);
    }

    /// The harness validates arguments against the schema before the tool
    /// runs, so an alias the tool accepts must not be a schema error.
    #[test]
    fn schema_admits_content_aliases_and_a_missing_status() {
        let tool = TodoTool::new(store());
        let schema = tool.parameters_schema();
        for item in [
            json!({ "title": "A", "status": "pending" }),
            json!({ "id": 1, "task": "B" }),
            json!({ "content": "C" }),
        ] {
            let args = json!({ "todos": [item] });
            tinyagents_harness::tool::validate_against_schema(&schema, &args)
                .unwrap_or_else(|err| panic!("{args}: {err}"));
        }
        assert_eq!(
            schema["properties"]["todos"]["items"]["properties"]["content"]["type"],
            "string",
            "`content` stays the canonical, advertised field"
        );
    }

    #[tokio::test]
    async fn invariant_violation_is_a_soft_error() {
        let tool = TodoTool::new(store());
        let res = run(
            &tool,
            Some("t"),
            json!({ "todos": [
                { "content": "A", "status": "in_progress" },
                { "content": "B", "status": "in_progress" }
            ] }),
        )
        .await;
        assert!(res.is_error && res.output().contains("in_progress"));
    }
}
