//! End-to-end regression tests for transcript compaction defects.
//!
//! Every test in this file is written **only against APIs that existed before
//! the fixes**, so each one compiles against the unfixed crate and fails there.
//! That is deliberate: a regression test for a structural bug is worth little if
//! it can only be expressed in terms of the fix's own new surface.
//!
//! The three defects covered:
//!
//! - **Orphaned tool pairing.** `SummarizationPolicy::plan` and all three
//!   `TrimStrategy` variants cut at a blind index, severing an assistant
//!   `tool_calls` turn from the `tool` messages answering it. The rebuilt
//!   request then carries a `role:"tool"` with no preceding `tool_calls`, which
//!   OpenAI rejects with a `400` and Anthropic rejects as a `tool_result` with
//!   no matching `tool_use`.
//! - **Token estimation blind to tool calls.** An assistant turn that only
//!   calls tools has empty `content`, so it estimated at zero and no compaction
//!   gate ever fired on the runs that needed it most.
//! - **History erased by the default summarizer.** `ConcatSummarizer` built its
//!   output from `Message::text()`, which renders tool calls and tool results as
//!   the empty string.

use tinyagents_harness::summarization::{
    ConcatSummarizer, SummarizationPolicy, Summarizer, TrimStrategy, trim_messages,
};
use tinyinference::message::{AssistantMessage, Message};
use tinyinference::tool::ToolCall;

/// The provider invariant, checked locally so this file depends on no new
/// crate surface: every `tool` message must be preceded by an assistant turn
/// declaring its id, and every declared call must be answered.
fn pairing_intact(messages: &[Message]) -> bool {
    let mut declared: Vec<&str> = Vec::new();
    let mut answered: Vec<&str> = Vec::new();

    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                declared.extend(assistant.tool_calls.iter().map(|call| call.id.as_str()));
            }
            Message::Tool(tool) => {
                if !declared.contains(&tool.tool_call_id.as_str()) {
                    return false;
                }
                answered.push(tool.tool_call_id.as_str());
            }
            _ => {}
        }
    }

    declared.iter().all(|id| answered.contains(id))
}

fn assistant_calling(id: &str) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            id,
            "get_weather",
            serde_json::json!({"city": "Paris"}),
        )],
        usage: None,
    })
}

/// `[system, user, assistant(tool_calls=[c1]), tool(c1), assistant]` — the
/// shape a `keep_last = 2` cut splits down the middle.
fn tool_transcript() -> Vec<Message> {
    vec![
        Message::system("you are helpful"),
        Message::user("weather in Paris?"),
        assistant_calling("c1"),
        Message::tool("c1", r#"{"temp_c":21}"#),
        Message::assistant("It is 21C."),
    ]
}

#[test]
fn summarization_plan_never_orphans_a_tool_result() {
    let policy = SummarizationPolicy {
        keep_last: 2,
        ..Default::default()
    };
    let (_to_summarize, to_keep) = policy.plan(&tool_transcript());

    assert!(
        pairing_intact(&to_keep),
        "plan kept a tool result whose assistant tool-call turn was summarized away: {to_keep:#?}"
    );
}

#[test]
fn keep_last_never_orphans_a_tool_result() {
    let trimmed = trim_messages(&tool_transcript(), &TrimStrategy::KeepLast(2));
    assert!(
        pairing_intact(&trimmed),
        "KeepLast produced a slice a provider would reject: {trimmed:#?}"
    );
}

#[test]
fn keep_first_and_last_never_orphans_either_end() {
    let messages = vec![
        Message::user("one"),
        assistant_calling("c1"),
        Message::tool("c1", "r1"),
        Message::user("two"),
        assistant_calling("c2"),
        Message::tool("c2", "r2"),
        Message::assistant("done"),
    ];
    let trimmed = trim_messages(
        &messages,
        &TrimStrategy::KeepFirstAndLast { first: 2, last: 2 },
    );
    assert!(
        pairing_intact(&trimmed),
        "KeepFirstAndLast left an unanswered call or an unpaired result: {trimmed:#?}"
    );
}

#[test]
fn max_tokens_never_orphans_a_tool_result() {
    // The assistant turn carries visible text as well as its tool call, so the
    // budget loop stops *between* the call and its result — the only way a
    // front-dropping token trim can orphan one.
    let narrating_call = Message::Assistant(AssistantMessage {
        id: None,
        content: vec![tinyinference::message::ContentBlock::Text(
            "Let me look that up. ".repeat(20),
        )],
        tool_calls: vec![ToolCall::new(
            "c1",
            "get_weather",
            serde_json::json!({"city": "Paris"}),
        )],
        usage: None,
    });
    let messages = vec![
        Message::user("x".repeat(40)),
        narrating_call,
        Message::tool("c1", "r1"),
        Message::assistant("done"),
    ];
    let trimmed = trim_messages(&messages, &TrimStrategy::MaxTokens(3));
    assert!(
        pairing_intact(&trimmed),
        "MaxTokens produced a slice a provider would reject: {trimmed:#?}"
    );
}

#[test]
fn a_tool_only_assistant_turn_trips_the_compaction_gate() {
    // A 50-turn run whose assistant messages are all tool calls with 2 KB
    // argument blobs used to estimate at zero tokens, so the window overflowed
    // uncompacted.
    let heavy = Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            "c1",
            "search",
            serde_json::json!({"query": "q".repeat(2_000)}),
        )],
        usage: None,
    });
    assert_eq!(heavy.text(), "", "precondition: no visible text");

    let policy = SummarizationPolicy {
        trigger_tokens: 100,
        ..Default::default()
    };
    assert!(
        policy.should_summarize(&[heavy]),
        "a 2 KB tool-call turn must trip a 100-token trigger"
    );
}

#[tokio::test]
async fn the_default_summarizer_preserves_tool_history() {
    let record = ConcatSummarizer
        .summarize(&tool_transcript())
        .await
        .expect("summarizing a non-empty transcript must succeed");
    let summary = record.summary.text();

    assert!(
        summary.contains("get_weather"),
        "the tool name was erased: {summary}"
    );
    assert!(
        summary.contains("Paris"),
        "the tool arguments were erased: {summary}"
    );
    assert!(
        summary.contains("temp_c"),
        "the tool result was erased: {summary}"
    );
}
