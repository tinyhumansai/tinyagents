//! Regression coverage for the two text-only token estimators and for
//! micro-compaction's `trusted_verbatim` violation.
//!
//! Both estimators summed `estimate_tokens(&m.text())`, and `text()` returns
//! only *textual* content blocks — so a transcript dominated by large JSON tool
//! results or image blocks estimated to nearly nothing and sailed past the very
//! gate that exists to catch it.

use serde_json::json;

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::middleware::{MicrocompactMiddleware, Middleware};
use tinyinference::message::{ContentBlock, Message, ToolMessage};
use tinyinference::model::ModelRequest;

const PLACEHOLDER: &str = "[elided]";

/// A tool result whose payload lives in a JSON content block, which
/// `Message::text()` does not see at all — the shape a tool returning
/// structured data produces.
fn json_tool_message(id: &str, trusted_verbatim: bool) -> Message {
    let payload = json!({
        "id": id,
        "rows": (0..80).map(|i| json!({"n": i, "label": format!("row-{i}-{id}")}))
            .collect::<Vec<_>>(),
    });
    Message::Tool(ToolMessage {
        tool_call_id: id.to_string(),
        content: vec![ContentBlock::Json(payload)],
        trusted_verbatim,
        artifact: None,
    })
}

/// Runs the middleware's `before_model` hook against a throwaway context.
async fn compact(middleware: &MicrocompactMiddleware, request: &mut ModelRequest) {
    let mut ctx: RunContext<()> = RunContext::new(RunConfig::new("estimator"), ());
    Middleware::<(), ()>::before_model(middleware, &mut ctx, &(), request)
        .await
        .expect("before_model succeeds");
}

/// REASON-3: the shared estimator charges non-textual content; the old
/// text-only sum scored the identical transcript at zero.
#[test]
fn the_shared_estimator_charges_non_textual_payloads() {
    let messages: Vec<Message> = (0..6)
        .map(|i| json_tool_message(&format!("c{i}"), false))
        .collect();

    let text_only: u64 = messages
        .iter()
        .map(|m| tinyagents_harness::summarization::estimate_tokens(&m.text()))
        .sum();
    let counted = tinyinference::message::count_tokens_approximately(&messages);

    assert_eq!(
        text_only, 0,
        "precondition: the old text-only estimator scores a JSON transcript at zero"
    );
    assert!(
        counted > 1_000,
        "the shared estimator must charge JSON tool results, got {counted}"
    );
}

/// REASON-3, micro-compaction: with a budget far below the transcript's real
/// weight but above its text-only estimate of zero, the gate must fire.
///
/// Before the switch `total_message_tokens` returned 0 here, so the gate
/// concluded the transcript still fit and micro-compaction never ran.
#[tokio::test]
async fn microcompaction_budget_gate_trips_on_a_non_textual_transcript() {
    let messages: Vec<Message> = (0..6)
        .map(|i| json_tool_message(&format!("c{i}"), false))
        .collect();

    let middleware = MicrocompactMiddleware::new(2, PLACEHOLDER).with_token_budget(100);
    let mut request = ModelRequest::new(messages.clone());
    compact(&middleware, &mut request).await;

    assert_ne!(
        request.messages, messages,
        "the budget gate must trip on a transcript whose weight is non-textual"
    );
    assert!(
        request.messages.iter().any(|m| m.text() == PLACEHOLDER),
        "older tool bodies should have been blanked"
    );
}

/// MICROCOMPACT: a tool result flagged `trusted_verbatim` asked to reach the
/// model byte-for-byte. Blanking it is exactly the rewrite the flag's contract
/// forbids — it produces content that reads fine and is wrong.
#[tokio::test]
async fn microcompaction_leaves_trusted_verbatim_tool_results_alone() {
    let messages = vec![
        // Oldest — the first one micro-compaction would blank.
        Message::Tool(ToolMessage {
            tool_call_id: "c0".to_string(),
            content: vec![ContentBlock::Text(
                "argument schema for `write`".to_string(),
            )],
            trusted_verbatim: true,
            artifact: None,
        }),
        Message::Tool(ToolMessage {
            tool_call_id: "c1".to_string(),
            content: vec![ContentBlock::Text("ordinary tool output".to_string())],
            trusted_verbatim: false,
            artifact: None,
        }),
        Message::tool("c2", "kept recent"),
    ];

    let middleware = MicrocompactMiddleware::new(1, PLACEHOLDER);
    let mut request = ModelRequest::new(messages);
    compact(&middleware, &mut request).await;

    assert_eq!(
        request.messages[0].text(),
        "argument schema for `write`",
        "a trusted_verbatim tool result must survive compaction verbatim"
    );
    assert_eq!(
        request.messages[1].text(),
        PLACEHOLDER,
        "untrusted tool results should still be blanked"
    );
}
