use serde_json::json;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::tool::ToolCall;

use super::drop_past_turn_reasoning;

fn thinking(text: &str, signature: Option<&str>) -> ContentBlock {
    ContentBlock::Thinking {
        text: text.to_string(),
        signature: signature.map(str::to_string),
    }
}

fn assistant(content: Vec<ContentBlock>, tool_calls: Vec<ToolCall>) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content,
        tool_calls,
        usage: None,
        origin: None,
    })
}

/// Three turns, each with reasoning; turn 3 also runs a tool, so its reasoning
/// sits on a tool-calling assistant message.
fn three_turns() -> Vec<Message> {
    vec![
        Message::user("turn 1"),
        assistant(
            vec![
                thinking("reasoning-1", Some("sig-1")),
                ContentBlock::Text("answer-1".into()),
            ],
            Vec::new(),
        ),
        Message::user("turn 2"),
        assistant(
            vec![
                ContentBlock::RedactedThinking {
                    data: "opaque-2".into(),
                },
                thinking("reasoning-2", None),
                ContentBlock::Text("answer-2".into()),
            ],
            Vec::new(),
        ),
        Message::user("turn 3"),
        assistant(
            vec![thinking("reasoning-3", Some("sig-3"))],
            vec![ToolCall::new("call-3", "lookup", json!({}))],
        ),
        Message::tool("call-3", "result-3"),
    ]
}

fn assistant_blocks(messages: &[Message]) -> Vec<Vec<ContentBlock>> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant(a) => Some(a.content.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn reasoning_from_completed_turns_is_dropped_and_the_current_turns_is_kept() {
    let mut messages = three_turns();
    let current_turn = messages[5].clone();

    let dropped = drop_past_turn_reasoning(&mut messages);

    assert_eq!(dropped, 3);
    assert_eq!(
        assistant_blocks(&messages),
        vec![
            vec![ContentBlock::Text("answer-1".into())],
            vec![ContentBlock::Text("answer-2".into())],
            vec![thinking("reasoning-3", Some("sig-3"))],
        ]
    );
    // The current turn is byte for byte what it was, signature included.
    assert_eq!(messages[5], current_turn);
    assert_eq!(messages.len(), 7);
}

#[test]
fn only_reasoning_is_removed_from_a_past_turn() {
    let mut messages = vec![
        Message::user("first"),
        assistant(
            vec![thinking("why", None), ContentBlock::Text("shown".into())],
            vec![ToolCall::new("call-1", "lookup", json!({"q": 1}))],
        ),
        Message::tool("call-1", "result"),
        assistant(vec![ContentBlock::Text("answered".into())], Vec::new()),
        Message::user("second"),
    ];

    drop_past_turn_reasoning(&mut messages);

    let Message::Assistant(kept) = &messages[1] else {
        panic!("the assistant message stays");
    };
    assert_eq!(kept.content, vec![ContentBlock::Text("shown".into())]);
    assert_eq!(kept.tool_calls.len(), 1);
    assert_eq!(messages[2].text(), "result");
}

#[test]
fn a_past_message_that_only_held_reasoning_is_removed() {
    let mut messages = vec![
        Message::user("first"),
        assistant(vec![thinking("only thoughts", Some("sig"))], Vec::new()),
        Message::user("second"),
    ];

    assert_eq!(drop_past_turn_reasoning(&mut messages), 1);

    assert_eq!(messages.len(), 2);
    assert!(messages.iter().all(|m| matches!(m, Message::User(_))));
}

#[test]
fn a_history_with_no_user_message_is_all_current_turn() {
    let mut messages = vec![assistant(vec![thinking("r", Some("s"))], Vec::new())];
    let before = messages.clone();

    assert_eq!(drop_past_turn_reasoning(&mut messages), 0);
    assert_eq!(messages, before);
}

#[test]
fn a_history_without_reasoning_is_unchanged_and_the_rewrite_is_idempotent() {
    let mut plain = vec![
        Message::user("a"),
        assistant(vec![ContentBlock::Text("b".into())], Vec::new()),
        Message::user("c"),
    ];
    let before = plain.clone();
    assert_eq!(drop_past_turn_reasoning(&mut plain), 0);
    assert_eq!(plain, before);

    let mut messages = three_turns();
    drop_past_turn_reasoning(&mut messages);
    let once = messages.clone();
    assert_eq!(drop_past_turn_reasoning(&mut messages), 0);
    assert_eq!(messages, once);
}

#[test]
fn a_tool_follow_up_user_message_does_not_end_the_current_turn() {
    let mut messages = vec![
        Message::user("ask"),
        assistant(
            vec![thinking("why", Some("sig"))],
            vec![ToolCall::new("call-1", "lookup", json!({}))],
        ),
        Message::tool("call-1", "result"),
        // A tool's follow-up content lands as a user message after the rows.
        Message::user("follow-up content"),
    ];

    assert_eq!(drop_past_turn_reasoning(&mut messages), 0);

    assert_eq!(assistant_blocks(&messages)[0].len(), 1);
}

#[test]
fn every_follow_up_of_a_tool_batch_stays_in_the_current_turn() {
    let mut messages = vec![
        Message::user("ask"),
        assistant(
            vec![thinking("why", Some("sig"))],
            vec![
                ToolCall::new("a", "lookup", json!({})),
                ToolCall::new("b", "lookup", json!({})),
            ],
        ),
        Message::tool("a", "ra"),
        Message::tool("b", "rb"),
        Message::user("follow-up a"),
        Message::user("follow-up b"),
    ];

    assert_eq!(drop_past_turn_reasoning(&mut messages), 0);
}
