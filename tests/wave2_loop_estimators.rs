//! Regression coverage for the two text-only token estimators and for
//! micro-compaction's `trusted_verbatim` violation.
//!
//! Both estimators summed `estimate_tokens(&m.text())`, and `text()` returns
//! only *textual* content blocks — so a transcript dominated by large JSON tool
//! results or image blocks estimated to nearly nothing and sailed past the very
//! gate that exists to catch it.

use serde_json::json;

use tinyagents::harness::message::{ContentBlock, Message, ToolMessage};
use tinyagents::harness::middleware::MicrocompactMiddleware;
use tinyagents::harness::model::ModelRequest;

/// A tool message whose payload lives in a non-text content block, which
/// `Message::text()` does not see at all.
fn image_tool_message(id: &str) -> Message {
    Message::Tool(ToolMessage {
        tool_call_id: id.to_string(),
        content: vec![ContentBlock::Image {
            mime_type: "image/png".to_string(),
            data: "A".repeat(4_000),
        }],
        trusted_verbatim: false,
        artifact: None,
    })
}

/// REASON-3: the micro-compaction budget gate must see a transcript whose
/// weight is non-textual. Summing over `text()` scored these at zero, so the
/// gate never tripped.
#[tokio::test]
async fn microcompaction_budget_gate_sees_non_textual_payloads() {
    let messages: Vec<Message> = (0..6).map(|i| image_tool_message(&format!("c{i}"))).collect();
    let counted = tinyagents::harness::message::count_tokens_approximately(&messages);
    let text_only: u64 = messages
        .iter()
        .map(|m| tinyagents::harness::summarization::estimate_tokens(&m.text()))
        .sum();

    assert_eq!(
        text_only, 0,
        "precondition: the old text-only estimator scores these at zero"
    );
    assert!(
        counted > 1_000,
        "the shared estimator must charge non-textual blocks, got {counted}"
    );

    // With a budget well below the real weight but above the text-only estimate
    // of zero, the gate must fire and blank the older tool results.
    let middleware = MicrocompactMiddleware::new(2).with_token_budget(100);
    let mut request = ModelRequest::new(messages);
    let before = request.messages.clone();
    middleware.compact_for_test(&mut request);
    assert_ne!(
        request.messages, before,
        "the budget gate must trip on a non-textual transcript"
    );
}

/// MICROCOMPACT: a tool result flagged `trusted_verbatim` asked to reach the
/// model byte-for-byte. Blanking it is exactly the rewrite the flag's contract
/// forbids — it produces content that reads fine and is wrong.
#[tokio::test]
async fn microcompaction_leaves_trusted_verbatim_tool_results_alone() {
    let mut messages: Vec<Message> = Vec::new();
    for i in 0..4 {
        let mut msg = ToolMessage {
            tool_call_id: format!("c{i}"),
            content: vec![ContentBlock::Text(format!(
                "argument schema {i}: {}",
                json!({"required": ["path"]})
            ))],
            trusted_verbatim: false,
            artifact: None,
        };
        // The oldest result is the one micro-compaction would blank first.
        if i == 0 {
            msg.trusted_verbatim = true;
        }
        messages.push(Message::Tool(msg));
    }

    let middleware = MicrocompactMiddleware::new(1);
    let mut request = ModelRequest::new(messages);
    middleware.compact_for_test(&mut request);

    let Message::Tool(first) = &request.messages[0] else {
        panic!("expected a tool message");
    };
    assert!(
        first.text_content().contains("argument schema 0"),
        "a trusted_verbatim tool result must survive compaction verbatim, got {:?}",
        first.content
    );
    assert!(first.trusted_verbatim);

    // The untrusted sibling is still compacted, so the middleware still works.
    let Message::Tool(second) = &request.messages[1] else {
        panic!("expected a tool message");
    };
    assert!(
        !second.text_content().contains("argument schema 1"),
        "untrusted tool results should still be blanked"
    );
}
