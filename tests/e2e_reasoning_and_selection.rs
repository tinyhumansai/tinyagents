//! End-to-end coverage for two harness features:
//!
//! **Part A — reasoning streaming.** The [`MessageDelta`] reasoning channel and
//! [`StreamAccumulator`] keep thinking output on a side channel that never
//! leaks into the final visible text. These tests drive a [`StreamingMock`]
//! through the real [`ChatModel::stream`] path, fold the items with a
//! [`StreamAccumulator`], and assert the reasoning/text split, plus the serde
//! shape of [`MessageDelta`].
//!
//! **Part B — contextual tool selection.** [`ContextualToolSelectionMiddleware`]
//! filters the tools the model is shown, either from allow/deny lists or from a
//! context-aware predicate that can vary exposure by run depth. These tests run
//! a full harness against a [`ScriptedModel`] and inspect the recorded
//! request's tool list to observe exactly what the model saw after filtering.

use std::sync::Arc;

use futures::StreamExt;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use tinyagents::harness::middleware::{ContextualToolSelectionMiddleware, ToolSelectionContext};
use tinyagents::harness::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStreamItem, StreamAccumulator,
};
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{FakeTool, ScriptedModel, StreamingMock};
use tinyagents::harness::tool::ToolSchema;
use tinyagents::harness::usage::Usage;

/// Builds a plain-text [`ModelResponse`] so a [`ScriptedModel`] can answer a run
/// in a single model call (no tool execution needed).
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(3, 1)),
        },
        usage: Some(Usage::new(3, 1)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds the authoritative `Completed` response carrying only visible text.
fn completed_text(text: &str) -> ModelStreamItem {
    ModelStreamItem::Completed(text_response(text))
}

// ---------------------------------------------------------------------------
// Part A — reasoning streaming
// ---------------------------------------------------------------------------

/// Reasoning fragments accumulate on the side channel and stay out of the final
/// visible text, which comes from the authoritative `Completed` response.
#[tokio::test]
async fn streamed_reasoning_stays_out_of_final_text() {
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::reasoning("think ")),
        ModelStreamItem::MessageDelta(MessageDelta::text("Hello")),
        ModelStreamItem::MessageDelta(MessageDelta::reasoning("harder")),
        ModelStreamItem::MessageDelta(MessageDelta {
            text: ", world".into(),
            reasoning: "!".into(),
            tool_call: None,
        }),
        completed_text("Hello, world"),
    ];

    let model = StreamingMock::new(items);
    let request = ModelRequest::new(vec![Message::user("hi")]);

    let mut stream = ChatModel::<()>::stream(&model, &(), request)
        .await
        .expect("opening the mock stream succeeds");

    let mut accumulator = StreamAccumulator::new();
    let mut delta_count = 0usize;
    while let Some(item) = stream.next().await {
        if matches!(item, ModelStreamItem::MessageDelta(_)) {
            delta_count += 1;
        }
        accumulator.push(&item);
    }

    assert_eq!(delta_count, 4, "four message deltas were streamed");
    assert!(
        accumulator.is_terminal(),
        "the Completed item marks the accumulator terminal"
    );
    // Reasoning is the concatenation of every reasoning fragment, in order.
    assert_eq!(accumulator.reasoning(), "think harder!");

    let response = accumulator.finish().expect("stream merges into a response");
    // The visible text comes from the authoritative Completed response and does
    // NOT include any reasoning fragments.
    assert_eq!(response.text(), "Hello, world");
    assert!(
        !response.text().contains("think"),
        "reasoning must never leak into the final visible text"
    );
}

/// A `MessageDelta` round-trips through serde unchanged, and a text-only delta
/// omits the empty `reasoning` field entirely (`skip_serializing_if`).
#[test]
fn message_delta_serde_round_trip_and_field_skipping() {
    let delta = MessageDelta {
        text: "a".into(),
        reasoning: "b".into(),
        tool_call: None,
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    let back: MessageDelta = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(delta, back, "round-trip preserves every field");
    assert!(json.contains("\"reasoning\":\"b\""));

    // A text-only delta has an empty reasoning field, which is skipped in JSON.
    let text_only = MessageDelta::text("visible");
    let text_json = serde_json::to_string(&text_only).expect("serialize");
    assert!(
        !text_json.contains("reasoning"),
        "empty reasoning is skipped, got {text_json}"
    );
    assert!(text_json.contains("\"text\":\"visible\""));
}

/// The two named constructors populate exactly one channel each.
#[test]
fn message_delta_constructors_populate_one_channel() {
    let reasoning = MessageDelta::reasoning("x");
    assert_eq!(reasoning.reasoning, "x");
    assert!(reasoning.text.is_empty());
    assert!(reasoning.tool_call.is_none());

    let text = MessageDelta::text("y");
    assert_eq!(text.text, "y");
    assert!(text.reasoning.is_empty());
    assert!(text.tool_call.is_none());
}

// ---------------------------------------------------------------------------
// Part B — contextual tool selection
// ---------------------------------------------------------------------------

/// Runs a harness with the given middleware at the given depth and returns the
/// tool names the model was shown (after filtering), sorted for stable
/// assertions.
async fn exposed_tools_at_depth(
    middleware: Arc<ContextualToolSelectionMiddleware>,
    depth: usize,
) -> Vec<String> {
    let scripted = Arc::new(ScriptedModel::new(vec![text_response("done")]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", scripted.clone());
    harness.register_tool(Arc::new(FakeTool::returning("a", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("b", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("c", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("privileged", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("safe", "ok")));
    harness.push_middleware(middleware);

    let ctx = RunContext::new(RunConfig::new("run").with_depth(depth), ());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("the run completes with a single scripted text response");

    let requests = scripted.requests();
    assert_eq!(requests.len(), 1, "exactly one model call was recorded");
    let mut names: Vec<String> = requests[0].tools.iter().map(|t| t.name.clone()).collect();
    names.sort();
    names
}

/// `from_lists(Some([a, b]), [b])`: deny removes `b`, and the allow-list is
/// fail-closed so unknown tools (`c`, `privileged`, `safe`) are excluded too —
/// the model is shown only `a`.
#[tokio::test]
async fn from_lists_deny_wins_and_allow_is_fail_closed() {
    let mw = Arc::new(ContextualToolSelectionMiddleware::from_lists(
        Some(["a", "b"]),
        ["b"],
    ));
    let exposed = exposed_tools_at_depth(mw, 0).await;
    assert_eq!(exposed, vec!["a".to_string()]);
}

/// A context-aware predicate hides the `privileged` tool at sub-agent depth
/// (>0) but exposes it at the top level (depth 0). The exposed toolsets differ
/// across the two runs.
#[tokio::test]
async fn contextual_predicate_varies_exposure_by_depth() {
    fn build() -> Arc<ContextualToolSelectionMiddleware> {
        Arc::new(ContextualToolSelectionMiddleware::new(Arc::new(
            |schema: &ToolSchema, sel: &ToolSelectionContext| {
                schema.name != "privileged" || sel.depth == 0
            },
        )))
    }

    let deep = exposed_tools_at_depth(build(), 2).await;
    let top = exposed_tools_at_depth(build(), 0).await;

    assert!(
        !deep.contains(&"privileged".to_string()),
        "privileged is hidden at depth 2, got {deep:?}"
    );
    assert!(
        top.contains(&"privileged".to_string()),
        "privileged is shown at depth 0, got {top:?}"
    );
    assert_ne!(deep, top, "the exposed toolset differs across depths");
    // Non-privileged tools remain visible at both depths.
    assert!(deep.contains(&"safe".to_string()));
    assert!(top.contains(&"safe".to_string()));
}
