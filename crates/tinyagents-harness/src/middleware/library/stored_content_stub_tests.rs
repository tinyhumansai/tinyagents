//! Tests for [`StoredContentStubMiddleware`].

use std::sync::Arc;

use serde_json::{Value, json};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::ModelRequest;
use tinyinference_llm::tool::ToolCall;

use super::{StoredContent, StoredContentStubMiddleware, StubRenderer};
use crate::context::{RunConfig, RunContext};
use crate::middleware::Middleware;

const BODY: &str = "milk\neggs\n";

/// `[stored <n> bytes as <id>]` from a result naming an `id`; `None` for a
/// result with none, and for what it has already rendered.
fn renderer() -> StubRenderer {
    Arc::new(|stored: &StoredContent<'_>| {
        if stored.content.starts_with("[stored") {
            return None;
        }
        let result: Value = serde_json::from_str(stored.result).ok()?;
        let id = result.get("id")?.as_str()?;
        Some(format!("[stored {} bytes as {id}]", stored.content.len()))
    })
}

fn middleware() -> StoredContentStubMiddleware {
    StoredContentStubMiddleware::new(renderer())
        .with_argument("save", "body")
        .with_earlier_result("load")
}

fn call(id: &str, tool: &str, arguments: Value) -> Message {
    let Message::Assistant(assistant) = Message::assistant("") else {
        unreachable!("an assistant constructor");
    };
    Message::Assistant(AssistantMessage {
        tool_calls: vec![ToolCall::new(id, tool, arguments)],
        ..assistant
    })
}

fn save(id: &str) -> Message {
    call(id, "save", json!({"title": "notes", "body": BODY}))
}

fn stored(id: &str) -> Message {
    Message::tool(id, json!({"id": "a1"}).to_string())
}

fn load(id: &str) -> Message {
    call(id, "load", json!({"id": "a1"}))
}

/// `messages` as they are sent.
async fn sent_with(mw: &StoredContentStubMiddleware, messages: &[Message]) -> Vec<Message> {
    let mut ctx = RunContext::new(RunConfig::new("stub-test"), ());
    let mut request = ModelRequest {
        messages: messages.to_vec(),
        ..ModelRequest::default()
    };
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    request.messages
}

async fn sent(messages: &[Message]) -> Vec<Message> {
    sent_with(&middleware(), messages).await
}

fn wire(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| serde_json::to_string(message).unwrap())
        .collect()
}

fn argument(message: &Message, name: &str) -> Value {
    let Message::Assistant(assistant) = message else {
        panic!("not an assistant message: {message:?}");
    };
    assistant.tool_calls[0].arguments[name].clone()
}

#[tokio::test]
async fn an_argument_is_stubbed_once_the_call_has_run() {
    let messages = [Message::user("save it"), save("c1"), stored("c1")];

    let sent = sent(&messages).await;

    assert_eq!(argument(&sent[1], "body"), json!("[stored 10 bytes as a1]"));
    assert_eq!(argument(&sent[1], "title"), json!("notes"));
    let Message::Assistant(assistant) = &sent[1] else {
        panic!("not an assistant message");
    };
    assert_eq!(assistant.tool_calls[0].id, "c1");
    assert_eq!(assistant.tool_calls[0].name, "save");
}

#[tokio::test]
async fn a_call_with_no_result_keeps_its_argument() {
    let messages = [Message::user("save it"), save("c1")];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn a_failed_call_keeps_its_argument_for_the_retry() {
    let messages = [
        Message::user("save it"),
        save("c1"),
        Message::tool("c1", "disk full"),
    ];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn a_tool_nobody_named_is_left_alone() {
    let other = call("c1", "search", json!({"body": BODY}));
    let messages = [Message::user("find it"), other, stored("c1")];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn an_invalid_call_is_left_alone() {
    let Message::Assistant(mut invalid) = save("c1") else {
        panic!("not an assistant message");
    };
    invalid.tool_calls[0].invalid = Some("bad json".to_string());
    let messages = [
        Message::user("save it"),
        Message::Assistant(invalid),
        stored("c1"),
    ];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn the_request_it_was_given_is_the_only_thing_it_changes() {
    let transcript = vec![Message::user("save it"), save("c1"), stored("c1")];
    let before = wire(&transcript);

    let _sent = sent(&transcript).await;

    assert_eq!(wire(&transcript), before);
    assert_eq!(argument(&transcript[1], "body"), json!(BODY));
}

#[tokio::test]
async fn each_request_extends_the_last_one_byte_for_byte() {
    // What the model has seen: every send of the growing transcript. A call
    // goes out for the first time once it has its result.
    let mut transcript = vec![Message::user("save it"), save("c1"), stored("c1")];
    let first = wire(&sent(&transcript).await);

    transcript.push(Message::assistant("Saved."));
    transcript.push(Message::user("save another"));
    transcript.push(save("c2"));
    transcript.push(stored("c2"));
    let second = wire(&sent(&transcript).await);

    assert_eq!(second[..first.len()], first[..]);
}

#[tokio::test]
async fn a_resume_renders_the_same_bytes_as_the_process_that_wrote_them() {
    let transcript = [Message::user("save it"), save("c1"), stored("c1")];
    let live = wire(&sent_with(&middleware(), &transcript).await);
    let stored_copy: Vec<Message> = wire(&transcript)
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    let resumed = wire(&sent_with(&middleware(), &stored_copy).await);

    assert_eq!(resumed, live);
}

#[tokio::test]
async fn stubbing_a_stub_changes_nothing() {
    let transcript = [Message::user("save it"), save("c1"), stored("c1")];
    let once = sent(&transcript).await;
    assert_eq!(wire(&sent(&once).await), wire(&once));
}

#[tokio::test]
async fn a_result_stays_for_the_turn_that_read_it() {
    let messages = [
        Message::user("change it"),
        load("c1"),
        Message::tool("c1", json!({"id": "a1", "body": BODY}).to_string()),
        Message::assistant("Done."),
    ];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn a_user_message_the_model_has_not_answered_keeps_the_result() {
    // A message steered in mid-turn: the model has not seen the read yet.
    let messages = [
        Message::user("change it"),
        load("c1"),
        Message::tool("c1", json!({"id": "a1", "body": BODY}).to_string()),
        Message::user("and add eggs"),
    ];
    assert_eq!(wire(&sent(&messages).await), wire(&messages));
}

#[tokio::test]
async fn a_result_from_an_earlier_turn_is_stubbed() {
    let body = json!({"id": "a1", "body": BODY}).to_string();
    let messages = [
        Message::user("change it"),
        load("c1"),
        Message::tool("c1", body.clone()),
        Message::assistant("Done."),
        Message::user("thanks"),
    ];

    let sent = sent(&messages).await;

    assert_eq!(
        sent[2].text(),
        format!("[stored {} bytes as a1]", body.len())
    );
    assert_eq!(wire(&sent)[3..], wire(&messages)[3..]);
}

#[tokio::test]
async fn a_reused_call_id_still_pairs_each_call_with_its_own_result() {
    // Some providers number calls per response, so `c1` comes back every step.
    let messages = [
        Message::user("write it"),
        save("c1"),
        stored("c1"),
        call("c1", "send", json!({"to": "me"})),
        Message::tool("c1", "Sent."),
        call("c1", "send", json!({"to": "them"})),
        Message::tool("c1", "Sent."),
    ];

    let sent = sent(&messages).await;

    assert!(!format!("{:?}", wire(&sent)).contains("milk"));
    assert_eq!(wire(&sent)[3..], wire(&messages)[3..]);
}

#[tokio::test]
async fn a_reused_call_id_stubs_only_the_named_result() {
    let body = json!({"id": "a1", "body": BODY}).to_string();
    let messages = [
        Message::user("change it"),
        load("c1"),
        Message::tool("c1", body.clone()),
        call("c1", "send", json!({"to": "me"})),
        Message::tool("c1", json!({"id": "a1", "note": "kept"}).to_string()),
        Message::assistant("Done."),
        Message::user("thanks"),
    ];

    let sent = sent(&messages).await;

    assert_eq!(
        sent[2].text(),
        format!("[stored {} bytes as a1]", body.len())
    );
    assert_eq!(wire(&sent)[3..], wire(&messages)[3..]);
}

#[tokio::test]
async fn every_named_argument_of_a_tool_is_stubbed() {
    let messages = [
        Message::user("add to it"),
        call("c1", "save", json!({"append": BODY})),
        stored("c1"),
    ];
    let middleware = middleware().with_argument("save", "append");

    let sent = sent_with(&middleware, &messages).await;

    assert!(!format!("{:?}", wire(&sent)).contains("milk"));
}
