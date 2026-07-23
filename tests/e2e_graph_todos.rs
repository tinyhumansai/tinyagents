//! End-to-end coverage for the per-thread task board (`graph::todos`) exercised
//! through the public crate surface: a `MockModel`-driven agent loop calls the
//! `todo` tool, and the board persists on a shared `Store` addressed by the
//! run's thread id.

use std::sync::Arc;

use serde_json::json;

use tinyagents::harness::context::RunConfig;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message};
use tinyagents::harness::model::ModelResponse;
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::store::{InMemoryStore, Store};
use tinyagents::harness::tool::ToolCall;
use tinyagents::harness::usage::Usage;
use tinyagents::{TodoTool, todo_store};

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(4, 2)),
        },
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
    }
}

#[tokio::test]
async fn model_drives_the_todo_tool_and_board_persists_to_the_thread() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                // First turn: add a card via the `todo` tool.
                tool_call_response(
                    "call-1",
                    "todo",
                    json!({ "op": "add", "content": "Write the integration test" }),
                ),
                // Second turn: mark it in progress.
                tool_call_response("call-2", "todo", json!({ "op": "list" })),
                text_response("done"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(TodoTool::new(store.clone())));

    let run = harness
        .invoke(
            &(),
            (),
            RunConfig::new("run").with_thread("thread-e2e"),
            vec![Message::user("track this work")],
        )
        .await
        .expect("agent run succeeds");

    assert_eq!(run.tool_calls, 2, "both todo tool calls executed");

    // The board persisted under the run's thread id, reachable via the public
    // programmatic surface.
    let snapshot = todo_store::list(&store, "thread-e2e")
        .await
        .expect("board lists");
    assert_eq!(snapshot.cards.len(), 1);
    assert_eq!(snapshot.cards[0].title, "Write the integration test");
    assert!(
        snapshot.markdown.contains("Write the integration test"),
        "markdown renders the card: {}",
        snapshot.markdown
    );

    // A different thread has its own (empty) board.
    let other = todo_store::list(&store, "other-thread")
        .await
        .expect("other board lists");
    assert!(other.cards.is_empty());
}
