use serde_json::{Value, json};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, ToolMessage};
use tinyinference_llm::tool::{ToolCall, ToolSchema};

use super::*;
use crate::tool::discover::{
    DeferredCatalog, TOOL_SEARCH_NAME, ToolDiscoveryPolicy, answer_tool_search,
};

fn schema(name: &str, description: &str, property: &str) -> ToolSchema {
    ToolSchema::new(
        name,
        description,
        json!({"type": "object", "properties": {property: {"type": "string"}}}),
    )
}

fn catalog() -> DeferredCatalog {
    DeferredCatalog::build(vec![
        schema(
            "stock_quote",
            "Fetch the latest price for a ticker.",
            "symbol",
        ),
        schema("stock_history", "Fetch past prices for a ticker.", "symbol"),
        schema("stock_news", "Fetch headlines for a ticker.", "symbol"),
        schema("pdf_read", "Read the text of a PDF file.", "path"),
    ])
}

/// What the bridge would answer for `query`, as the loop stores it.
async fn search_result(query: &str) -> String {
    answer_tool_search(
        &catalog(),
        &ToolDiscoveryPolicy::default(),
        &json!({"query": query, "limit": 3}),
    )
    .await
    .result
    .text()
}

fn assistant_calling(calls: Vec<ToolCall>) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: calls,
        usage: None,
        origin: None,
    })
}

fn search_call(id: &str, query: &str) -> ToolCall {
    ToolCall::new(id, TOOL_SEARCH_NAME, json!({"query": query}))
}

/// One finished run: a user turn, a search, the answer, then each call in
/// `then_call` answered in turn, then a closing assistant message.
async fn run_that_searched(query: &str, then_call: &[&str]) -> Vec<Message> {
    let mut messages = vec![
        Message::user("start"),
        assistant_calling(vec![search_call("s1", query)]),
        Message::tool("s1", search_result(query).await),
    ];
    for (n, name) in then_call.iter().enumerate() {
        let id = format!("t{n}");
        messages.push(assistant_calling(vec![ToolCall::new(
            id.clone(),
            *name,
            json!({"symbol": "ACME"}),
        )]));
        messages.push(Message::tool(id, "ok"));
    }
    messages.push(Message::assistant("done"));
    messages
}

fn text_at(messages: &[Message], index: usize) -> String {
    messages[index].text()
}

fn hit_names(text: &str) -> Vec<String> {
    let (_, body) = text.split_once('\n').expect("header then payload");
    serde_json::from_str::<Vec<Value>>(body)
        .expect("payload is a JSON array")
        .iter()
        .map(|hit| hit["name"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn keeps_only_the_called_tools_schema_under_the_query_line() {
    let mut messages = run_that_searched("price of a ticker", &["stock_quote"]).await;
    let full = text_at(&messages, 2);
    assert_eq!(hit_names(&full).len(), 3, "the search matched three tools");

    let rewritten = collapse_search_results(&mut messages, &PriorSearchResults::default());

    assert_eq!(rewritten, 1);
    let collapsed = text_at(&messages, 2);
    assert_eq!(
        collapsed.lines().next().unwrap(),
        "tool_search \"price of a ticker\" \u{2192} used stock_quote"
    );
    assert_eq!(hit_names(&collapsed), vec!["stock_quote"]);
    assert!(!collapsed.contains("stock_news") && !collapsed.contains("pdf_read"));
    // The kept schema is the one the search returned, unchanged.
    let kept: Vec<Value> = serde_json::from_str(collapsed.split_once('\n').unwrap().1).unwrap();
    let original: Vec<Value> = serde_json::from_str(full.split_once('\n').unwrap().1).unwrap();
    assert_eq!(
        kept[0],
        *original
            .iter()
            .find(|h| h["name"] == "stock_quote")
            .unwrap()
    );
    assert!(collapsed.len() < full.len());
}

#[tokio::test]
async fn several_used_tools_are_all_kept_in_search_order() {
    let mut messages = run_that_searched("price of a ticker", &["stock_news", "stock_quote"]).await;
    collapse_search_results(&mut messages, &PriorSearchResults::default());
    let collapsed = text_at(&messages, 2);
    assert!(collapsed.starts_with("tool_search \"price of a ticker\" \u{2192} used "));
    let names = hit_names(&collapsed);
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"stock_quote".to_string()));
    assert!(names.contains(&"stock_news".to_string()));
    let header = collapsed.lines().next().unwrap();
    assert!(
        header.ends_with(&names.join(", ")),
        "header lists the same names: {header}"
    );
}

#[tokio::test]
async fn a_search_nothing_followed_collapses_to_one_line_of_matched_names() {
    let mut messages = run_that_searched("price of a ticker", &[]).await;
    let matched = hit_names(&text_at(&messages, 2));

    collapse_search_results(&mut messages, &PriorSearchResults::default());

    let collapsed = text_at(&messages, 2);
    assert_eq!(collapsed.lines().count(), 1);
    assert_eq!(
        collapsed,
        format!(
            "tool_search \"price of a ticker\" \u{2192} none used; matched {}",
            matched.join(", ")
        )
    );
}

#[tokio::test]
async fn calling_a_tool_the_search_did_not_match_does_not_count_as_use() {
    let mut messages = run_that_searched("price of a ticker", &["read_file"]).await;
    collapse_search_results(&mut messages, &PriorSearchResults::default());
    assert!(text_at(&messages, 2).contains("none used"));
}

#[tokio::test]
async fn a_call_made_before_the_search_does_not_count_as_use() {
    let mut messages = vec![
        assistant_calling(vec![ToolCall::new("e0", "stock_quote", json!({}))]),
        Message::tool("e0", "ok"),
        assistant_calling(vec![search_call("s1", "price of a ticker")]),
        Message::tool("s1", search_result("price of a ticker").await),
    ];
    collapse_search_results(&mut messages, &PriorSearchResults::default());
    assert!(text_at(&messages, 3).contains("none used"));
}

#[tokio::test]
async fn collapsing_twice_changes_nothing_the_second_time() {
    let mut messages = run_that_searched("price of a ticker", &["stock_quote"]).await;
    let none = PriorSearchResults::default();
    assert_eq!(collapse_search_results(&mut messages, &none), 1);
    let once = messages.clone();
    assert_eq!(collapse_search_results(&mut messages, &none), 0);
    assert_eq!(messages, once);

    let mut unused = run_that_searched("price of a ticker", &[]).await;
    collapse_search_results(&mut unused, &none);
    let once = unused.clone();
    assert_eq!(collapse_search_results(&mut unused, &none), 0);
    assert_eq!(unused, once);
}

#[tokio::test]
async fn everything_but_the_collapsed_message_is_byte_identical() {
    let before = run_that_searched("price of a ticker", &["stock_quote", "stock_news"]).await;
    let mut after = before.clone();
    collapse_search_results(&mut after, &PriorSearchResults::default());

    let bytes = |messages: &[Message]| -> Vec<String> {
        messages
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect()
    };
    let (b, a) = (bytes(&before), bytes(&after));
    // The prefix before the collapsed message, and the tail after it.
    assert_eq!(b[..2], a[..2]);
    assert_eq!(b[3..], a[3..]);
    assert_ne!(b[2], a[2]);
}

#[tokio::test]
async fn a_result_already_in_the_history_is_never_rewritten() {
    // Written by an earlier run, before the option was on.
    let old = run_that_searched("price of a ticker", &["stock_quote"]).await;
    let prior = PriorSearchResults::of(&old);

    // This run adds a second thread turn on top of that history.
    let mut messages = old.clone();
    messages.push(Message::user("again"));
    messages.push(assistant_calling(vec![search_call("s2", "read a pdf")]));
    messages.push(Message::tool("s2", search_result("read a pdf").await));
    messages.push(assistant_calling(vec![ToolCall::new(
        "t9",
        "pdf_read",
        json!({}),
    )]));
    messages.push(Message::tool("t9", "ok"));

    let rewritten = collapse_search_results(&mut messages, &prior);

    assert_eq!(rewritten, 1, "only the new result collapses");
    assert_eq!(
        messages[..old.len()],
        old[..],
        "old history is byte-identical"
    );
    assert!(
        text_at(&messages, old.len() + 2)
            .starts_with("tool_search \"read a pdf\" \u{2192} used pdf_read")
    );
}

#[tokio::test]
async fn a_new_result_that_reuses_an_old_call_id_still_collapses() {
    let old = run_that_searched("price of a ticker", &["stock_quote"]).await;
    let prior = PriorSearchResults::of(&old);

    let mut messages = old.clone();
    messages.push(assistant_calling(vec![search_call("s1", "read a pdf")]));
    messages.push(Message::tool("s1", search_result("read a pdf").await));
    collapse_search_results(&mut messages, &prior);

    assert_eq!(messages[..old.len()], old[..]);
    assert!(text_at(&messages, old.len() + 1).contains("none used"));
}

#[tokio::test]
async fn the_query_is_quoted_so_odd_text_cannot_break_the_line() {
    let query = "price of a \"ticker\"\nplease";
    let mut messages = run_that_searched(query, &[]).await;
    collapse_search_results(&mut messages, &PriorSearchResults::default());
    let collapsed = text_at(&messages, 2);
    assert_eq!(
        collapsed.lines().count(),
        1,
        "newline in the query is escaped"
    );
    assert!(collapsed.starts_with(&format!(
        "tool_search {} ",
        serde_json::to_string(query).unwrap()
    )));
}

#[tokio::test]
async fn results_that_are_not_a_bridge_answer_are_left_alone() {
    let query = "price of a ticker";
    let no_match = "No deferred tool matches \"zzz\". 4 tool(s) are searchable.".to_string();
    let host_tool = "a host tool answered with its own text".to_string();
    let mut messages = vec![
        assistant_calling(vec![
            search_call("a", "zzz"),
            search_call("b", query),
            search_call("c", query),
        ]),
        Message::tool("a", no_match),
        Message::tool("b", host_tool),
        // A search answer whose message a host marked verbatim.
        Message::Tool(ToolMessage {
            tool_call_id: "c".into(),
            content: vec![ContentBlock::Text(search_result(query).await)],
            trusted_verbatim: true,
            artifact: None,
        }),
        // An answer with no call to read the query from.
        Message::tool("orphan", search_result(query).await),
    ];
    let before = messages.clone();
    assert_eq!(
        collapse_search_results(&mut messages, &PriorSearchResults::default()),
        0
    );
    assert_eq!(messages, before);
}

#[tokio::test]
async fn a_result_for_another_tools_call_is_never_read_as_a_search() {
    // Same text, but the call it answers is not `tool_search`.
    let text = search_result("price of a ticker").await;
    let mut messages = vec![
        assistant_calling(vec![ToolCall::new("x", "my_search", json!({"query": "q"}))]),
        Message::tool("x", text),
    ];
    let before = messages.clone();
    assert_eq!(
        collapse_search_results(&mut messages, &PriorSearchResults::default()),
        0
    );
    assert_eq!(messages, before);
}

#[tokio::test]
async fn the_artifact_of_a_collapsed_result_is_preserved() {
    let mut messages = vec![
        assistant_calling(vec![search_call("s1", "price of a ticker")]),
        Message::Tool(ToolMessage {
            tool_call_id: "s1".into(),
            content: vec![ContentBlock::Text(search_result("price of a ticker").await)],
            trusted_verbatim: false,
            artifact: Some(json!({"k": 1})),
        }),
    ];
    collapse_search_results(&mut messages, &PriorSearchResults::default());
    assert_eq!(messages[1].artifact(), Some(&json!({"k": 1})));
    assert!(messages[1].text().contains("none used"));
}

#[tokio::test]
async fn a_repeat_of_an_old_result_with_the_same_id_and_text_is_the_runs_own() {
    // The same query asked again under the same call id: history is
    // append-only, so the earlier occurrence is prior and the last is new.
    let old = run_that_searched("price of a ticker", &[]).await;
    let prior = PriorSearchResults::of(&old);
    let mut messages = old.clone();
    messages.push(assistant_calling(vec![search_call(
        "s1",
        "price of a ticker",
    )]));
    messages.push(Message::tool(
        "s1",
        search_result("price of a ticker").await,
    ));

    assert_eq!(collapse_search_results(&mut messages, &prior), 1);
    assert_eq!(messages[..old.len()], old[..]);
    assert!(text_at(&messages, old.len() + 1).contains("none used"));
}
