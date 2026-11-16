//! Tests for [`ToolExecutionContext`] (B1): the call id, the namespaced
//! store, the typed state view, and the `custom` event helper a tool reaches
//! through `ToolRunContext::host_extension`.

use std::sync::Arc;

use serde_json::json;

use super::ToolExecutionContext;
use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::store::namespaced::{InMemoryNamespacedStore, Namespace, NamespacedStore};
use crate::testkit::EventRecorder;

#[derive(Debug, PartialEq)]
struct AppState {
    user: &'static str,
}

#[test]
fn from_run_context_carries_the_call_id_it_was_built_for() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ());
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("call-42"));
    assert_eq!(tool_ctx.call_id, CallId::new("call-42"));
}

#[test]
fn store_is_absent_unless_the_run_attached_one() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ());
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("c"));
    assert!(tool_ctx.store.is_none());
}

#[tokio::test]
async fn store_is_the_run_store_and_a_tool_can_read_and_write_through_it() {
    let store = Arc::new(InMemoryNamespacedStore::new());
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ())
        .with_namespaced_store(Arc::clone(&store) as Arc<dyn NamespacedStore>);
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("c"));

    let ns = Namespace::from("scratch");
    let handle = tool_ctx.store.as_ref().expect("store attached");
    handle.put(&ns, "k", json!({"n": 1})).await.unwrap();

    // The write is visible on the run's own store, not a private copy.
    let item = store.get(&ns, "k").await.unwrap().expect("item written");
    assert_eq!(item.value, json!({"n": 1}));
}

#[test]
fn state_view_downcasts_to_the_attached_type_and_is_none_otherwise() {
    let bare: RunContext = RunContext::new(RunConfig::new("run-1"), ());
    let tool_ctx = ToolExecutionContext::from_run_context(&bare, CallId::new("c"));
    assert!(tool_ctx.state::<AppState>().is_none());

    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ())
        .with_state_view(Arc::new(AppState { user: "alice" }));
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("c"));
    assert_eq!(
        tool_ctx.state::<AppState>(),
        Some(&AppState { user: "alice" })
    );
    // A mismatched type is `None`, never a panic.
    assert!(tool_ctx.state::<String>().is_none());
}

#[test]
fn custom_emits_a_custom_event_correlated_to_the_call() {
    let recorder = EventRecorder::new();
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ()).with_events(recorder.sink());
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("call-7"));

    tool_ctx.custom(json!({"progress": 0.25}));

    assert_eq!(
        recorder.events(),
        vec![AgentEvent::Custom {
            call_id: Some(CallId::new("call-7")),
            payload: json!({"progress": 0.25}),
        }]
    );
}

#[test]
fn a_tool_reaches_the_harness_context_through_the_erased_host_extension() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ());
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("call-3"));
    let erased: &dyn tinytools::ToolRunContext = &tool_ctx;
    let recovered = erased
        .host_extension()
        .and_then(|any| any.downcast_ref::<ToolExecutionContext>())
        .expect("host extension is the harness context");
    assert_eq!(recovered.call_id, CallId::new("call-3"));
}

#[test]
fn a_child_context_inherits_the_store_and_state_view() {
    let store = Arc::new(InMemoryNamespacedStore::new());
    let parent: RunContext = RunContext::new(RunConfig::new("parent"), ())
        .with_namespaced_store(Arc::clone(&store) as Arc<dyn NamespacedStore>)
        .with_state_view(Arc::new(AppState { user: "bob" }));
    let child = parent.child(RunConfig::new("child"), ()).unwrap();
    let tool_ctx = ToolExecutionContext::from_run_context(&child, CallId::new("c"));
    assert!(tool_ctx.store.is_some());
    assert_eq!(
        tool_ctx.state::<AppState>(),
        Some(&AppState { user: "bob" })
    );
}

#[test]
fn of_recovers_the_harness_context_from_the_erased_run_context() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-1"), ());
    let tool_ctx = ToolExecutionContext::from_run_context(&ctx, CallId::new("call-7"));
    let erased: &dyn tinytools::ToolRunContext = &tool_ctx;
    let recovered = ToolExecutionContext::of(Some(erased)).expect("harness context");
    assert!(std::ptr::eq(recovered, &tool_ctx));
}

#[test]
fn of_is_none_without_a_context_or_for_a_foreign_context() {
    struct Foreign;
    impl tinytools::ToolRunContext for Foreign {}

    assert!(ToolExecutionContext::of(None).is_none());
    let foreign: &dyn tinytools::ToolRunContext = &Foreign;
    assert!(ToolExecutionContext::of(Some(foreign)).is_none());
}

#[test]
fn call_id_of_reads_the_harness_execution_context() {
    let run_ctx: RunContext = RunContext::new(RunConfig::new("test-run"), ());
    let exec_ctx = ToolExecutionContext::from_run_context(&run_ctx, CallId::new("call-42"));

    let erased: &dyn tinytools::ToolRunContext = &exec_ctx;
    assert_eq!(
        ToolExecutionContext::call_id_of(Some(erased)),
        Some("call-42".to_string())
    );
}

#[test]
fn call_id_of_is_none_without_a_context() {
    assert_eq!(ToolExecutionContext::call_id_of(None), None);
}
