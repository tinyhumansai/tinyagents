//! Tests for [`ModelSummarizer`], [`FaultTolerantCachingSummarizer`] and the
//! context-window-aware policy builders.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde_json::json;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, ToolMessage};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

use super::{
    DEFAULT_SUMMARIZE_KEEP_LAST, DEFAULT_SUMMARIZE_THRESHOLD_FRACTION,
    FaultTolerantCachingSummarizer, ModelSummarizer, SummarizationPolicy, Summarizer,
    SummaryRecord, SummaryRequest, ToolRendering, summarization_policy, summarization_policy_with,
};
use crate::error::{Result, TinyAgentsError};
use crate::testkit::ScriptedModel;

fn tool_call_messages(arguments: serde_json::Value) -> Vec<Message> {
    vec![
        Message::Assistant(AssistantMessage {
            id: None,
            content: vec![ContentBlock::Thinking {
                text: "I should inspect the matching records.".into(),
                signature: None,
            }],
            tool_calls: vec![ToolCall::new("lookup-1", "lookup", arguments)],
            usage: None,
            origin: None,
        }),
        Message::Tool(ToolMessage {
            tool_call_id: "lookup-1".into(),
            content: vec![ContentBlock::Json(json!({"matches": 2}))],
            trusted_verbatim: false,
            artifact: None,
        }),
    ]
}

#[test]
fn policy_is_context_window_aware_at_the_default_threshold() {
    let policy = summarization_policy(200_000);
    assert_eq!(policy.context_window, Some(200_000));
    assert_eq!(
        policy.threshold_fraction,
        DEFAULT_SUMMARIZE_THRESHOLD_FRACTION
    );
    assert_eq!(policy.keep_last, DEFAULT_SUMMARIZE_KEEP_LAST);
}

#[test]
fn default_threshold_leaves_headroom_below_the_window() {
    let policy = summarization_policy(100_000);
    assert_eq!(policy.trigger_budget(), 80_000);
}

#[test]
fn default_trigger_is_capped_for_large_windows() {
    // min(80% of the window, 350k): small windows compact at 80%, a 1M window
    // at 350k rather than ~840k.
    for (window, trigger) in [
        (32_768, 26_214),
        (128_000, 102_400),
        (200_000, 160_000),
        (437_500, 350_000),
        (1_048_576, 350_000),
        (2_000_000, 350_000),
    ] {
        let budget = summarization_policy(window).trigger_budget();
        assert!(
            budget.abs_diff(trigger) <= 1,
            "window {window}: trigger {budget}, want {trigger}"
        );
    }
    assert_eq!(
        super::default_threshold_fraction_for(0),
        DEFAULT_SUMMARIZE_THRESHOLD_FRACTION
    );
}

#[test]
fn explicit_threshold_and_tail_override_the_defaults() {
    let policy = summarization_policy_with(10_000, 0.5, 3);
    assert_eq!(policy.threshold_fraction, 0.5);
    assert_eq!(policy.keep_last, 3);
}

#[tokio::test]
async fn model_summarizer_wraps_the_reply_and_records_provenance() {
    let model = Arc::new(ScriptedModel::replies(vec!["  the gist  "]));
    let summarizer = ModelSummarizer::new(model.clone(), "m-1").with_threshold_fraction(0.8);
    let messages = vec![Message::user("hello there"), Message::assistant("hi back")];
    let record = summarizer.summarize(&messages).await.unwrap();

    assert!(
        record
            .summary
            .text()
            .contains("=== Conversation Summary (compacted) ===")
    );
    assert!(record.summary.text().contains("the gist"));
    assert_eq!(record.provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(record.provenance.reason.contains("80%"));
    assert!(record.provenance.reason.contains("m-1"));
    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].messages[1].text().contains("user: hello there"));
}

#[tokio::test]
async fn model_summarizer_rejects_empty_input_and_empty_replies() {
    let summarizer = ModelSummarizer::new(Arc::new(ScriptedModel::replies(vec!["   "])), "m");
    assert!(summarizer.summarize(&[]).await.is_err());
    let err = summarizer
        .summarize(&[Message::user("x")])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("empty response"));
}

#[tokio::test]
async fn model_summarizer_renders_structured_messages_and_prior_summary() {
    let model = Arc::new(ScriptedModel::replies(vec!["combined context"]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    let messages = tool_call_messages(json!({"query": "open issues"}));
    let request =
        SummaryRequest::new(messages.clone()).with_previous_summary("Earlier result: 4 issues");

    let record = summarizer.summarize_request(&request).await.unwrap();
    let transcript = model.requests()[0].messages[1].text();
    assert!(transcript.contains("Earlier result: 4 issues"));
    assert!(transcript.contains("<reasoning>I should inspect the matching records.</reasoning>"));
    assert!(transcript.contains("<tool_call id=\"lookup-1\" name=\"lookup\">"));
    assert!(transcript.contains("<json>{\"matches\":2}</json>"));
    assert_eq!(
        record.provenance.original_token_estimate,
        crate::token_estimation::estimate_slice_tokens(&messages)
    );
}

struct CountingFailing(Arc<AtomicUsize>);

#[async_trait]
impl Summarizer for CountingFailing {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::Model("boom".into()))
    }
}

fn long_slice() -> Vec<Message> {
    (0..6)
        .map(|i| Message::user(format!("message {i} {}", "word ".repeat(40))))
        .collect()
}

#[tokio::test]
async fn failure_falls_back_to_a_deterministic_trim_and_trips_the_breaker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(CountingFailing(calls.clone())), &policy);

    let first = guarded.summarize(&long_slice()).await.unwrap();
    assert!(first.summary.text().contains("deterministic trim"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // A different slice: the breaker is open, so the inner summarizer is skipped.
    let other = vec![Message::user("something else entirely")];
    let second = guarded.summarize(&other).await.unwrap();
    assert!(second.provenance.reason.contains("circuit breaker open"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_identical_slice_is_served_from_the_cache() {
    let model = Arc::new(ScriptedModel::replies(vec!["one summary"]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded = FaultTolerantCachingSummarizer::new(
        Box::new(ModelSummarizer::new(model.clone(), "m")),
        &policy,
    );
    let slice = long_slice();
    let a = guarded.summarize(&slice).await.unwrap();
    let b = guarded.summarize(&slice).await.unwrap();
    assert_eq!(a.summary.text(), b.summary.text());
    assert_eq!(
        model.requests().len(),
        1,
        "second call must not reach the model"
    );
}

#[tokio::test]
async fn cache_key_includes_structured_messages_and_previous_summary() {
    let model = Arc::new(ScriptedModel::replies(vec!["first", "second", "third"]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded = FaultTolerantCachingSummarizer::new(
        Box::new(ModelSummarizer::new(model.clone(), "m")),
        &policy,
    );
    let first_messages = tool_call_messages(json!({"query": "one"}));
    let second_messages = tool_call_messages(json!({"query": "two"}));

    guarded
        .summarize_request(&SummaryRequest::new(first_messages.clone()))
        .await
        .unwrap();
    guarded
        .summarize_request(&SummaryRequest::new(second_messages))
        .await
        .unwrap();
    guarded
        .summarize_request(
            &SummaryRequest::new(first_messages).with_previous_summary("prior checkpoint"),
        )
        .await
        .unwrap();

    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests[2].messages[1].text().contains("prior checkpoint"));
}

#[tokio::test]
async fn the_fallback_front_drops_oldest_messages_to_fit_its_budget() {
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(CountingFailing(Arc::default())), &policy);
    // A 1_000-token window gives a floor budget of 1_024 tokens; oversize the slice.
    let big: Vec<Message> = (0..40)
        .map(|i| Message::user(format!("m{i} {}", "x".repeat(400))))
        .collect();
    let record = guarded.summarize(&big).await.unwrap();
    assert!(record.summary.text().contains("older message(s) dropped"));
    assert!(record.summary.text().contains("m39"));
}

/// What DeepSeek V4 returned as a "summary" in the replayed bench captures: its
/// native tool-call markup, with no tools declared on the request.
const DSML_REPLY: &str = "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name=\"shell\">\n\
<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">cd /app && cat src/lib.rs</｜｜DSML｜｜ parameter>\n\
</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>";

#[tokio::test]
async fn the_transcript_is_fenced_as_data_with_the_instruction_last() {
    let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    let request = SummaryRequest::new(tool_call_messages(json!({"query": "q"})))
        .with_previous_summary("Earlier result: 4 issues");
    summarizer.summarize_request(&request).await.unwrap();

    let sent = model.requests()[0].messages[1].text();
    let opens = sent.find("<transcript>").expect("transcript is fenced");
    let closes = sent.find("</transcript>").expect("transcript fence closes");
    assert!(sent.find("<previous_summary>").unwrap() < opens);
    assert!(sent[opens..closes].contains("<tool_call id=\"lookup-1\" name=\"lookup\">"));
    // The last thing the model reads is the instruction, not a tool result.
    assert!(sent.trim_end().ends_with("output only the summary."));
}

#[tokio::test]
async fn a_tool_call_reply_is_retried_once_and_a_real_summary_kept() {
    let model = Arc::new(ScriptedModel::replies(vec![
        DSML_REPLY,
        "## Goal\nShip it.",
    ]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    let record = summarizer.summarize(&[Message::user("x")]).await.unwrap();

    assert_eq!(model.requests().len(), 2);
    assert!(record.summary.text().contains("Ship it."));
    assert!(!record.summary.text().contains("DSML"));
}

#[tokio::test]
async fn a_summarizer_that_keeps_calling_tools_fails_so_the_fallback_trims() {
    let model = Arc::new(ScriptedModel::replies(vec![DSML_REPLY, DSML_REPLY]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    let err = summarizer
        .summarize(&[Message::user("x")])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("tool-call markup"));
    assert_eq!(model.requests().len(), 2);

    // Wrapped as hosts do, the markup never becomes the summary.
    let model = Arc::new(ScriptedModel::replies(vec![DSML_REPLY, DSML_REPLY]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(ModelSummarizer::new(model, "m")), &policy);
    let record = guarded.summarize(&long_slice()).await.unwrap();
    assert!(record.summary.text().contains("deterministic trim"));
    assert!(!record.summary.text().contains("DSML"));
}

#[tokio::test]
async fn fallback_reports_usage_from_failed_markup_attempts() {
    let usage = Usage {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        ..Usage::default()
    };
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant(DSML_REPLY).with_usage(usage),
        ModelResponse::assistant(DSML_REPLY).with_usage(usage),
    ]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(ModelSummarizer::new(model, "m")), &policy);
    let record = guarded.summarize(&[Message::user("x")]).await.unwrap();
    assert_eq!(record.usage.unwrap().total_tokens, 30);
}

#[tokio::test]
async fn fallback_reports_usage_from_empty_summary() {
    let usage = Usage {
        total_tokens: 15,
        ..Usage::default()
    };
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant("").with_usage(usage),
    ]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(ModelSummarizer::new(model, "m")), &policy);
    let record = guarded.summarize(&[Message::user("x")]).await.unwrap();
    assert_eq!(record.usage, Some(usage));
}
/// Issue tinyhumansai/openhuman#6960: a compaction that fires mid-turn folds
/// work the agent is still doing. The summary's pending section must hand that
/// work on as in progress, not tell the model it is stale.
#[tokio::test]
async fn pending_work_is_handed_on_as_in_progress_not_stale() {
    let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    summarizer
        .summarize(&[
            Message::user("write the report"),
            Message::assistant("drafting"),
        ])
        .await
        .unwrap();

    let prompt = model.requests()[0].messages[0].text();
    assert!(
        !prompt.contains("STALE"),
        "pending work must not be labelled stale: {prompt}"
    );
    assert!(
        prompt
            .contains("In progress — continue these unless a later live message changes direction"),
        "{prompt}"
    );
    // The guard against redoing requests that were already answered stays.
    assert!(prompt.contains("already answered"), "{prompt}");
}

/// A compaction must not cost what the user told the assistant about
/// themselves: the checkpoint keeps those facts, and the commitments the
/// assistant made, in sections of their own.
#[tokio::test]
async fn the_checkpoint_keeps_facts_about_the_user_and_commitments() {
    let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");
    summarizer
        .summarize(&[Message::user("I live in Pune"), Message::assistant("noted")])
        .await
        .unwrap();

    let prompt = model.requests()[0].messages[0].text();
    assert!(prompt.contains("## About the User"), "{prompt}");
    assert!(prompt.contains("## Commitments"), "{prompt}");
}

#[tokio::test]
async fn a_turn_prefix_request_uses_its_own_prompt() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "Asked: fix the bug.",
        "## Goal\nx",
    ]));
    let summarizer = ModelSummarizer::new(model.clone(), "m");

    summarizer
        .summarize_request(&SummaryRequest::turn_prefix(vec![Message::user("fix it")]))
        .await
        .unwrap();
    summarizer
        .summarize_request(&SummaryRequest::new(vec![Message::user("fix it")]))
        .await
        .unwrap();

    let requests = model.requests();
    let prefix_system = requests[0].messages[0].text();
    let prefix_user = requests[0].messages[1].text();
    assert!(
        prefix_system.contains("beginning of a turn"),
        "{prefix_system}"
    );
    assert!(prefix_user.contains("<transcript>"));
    assert!(prefix_user.contains("beginning of a turn"));
    // The ordinary prompt is untouched.
    assert!(
        !requests[1].messages[0]
            .text()
            .contains("beginning of a turn")
    );
    assert!(
        !requests[1].messages[1]
            .text()
            .contains("beginning of a turn")
    );
}

/// A short exchange around one tool call: the user's ask, a call with a long
/// argument, its result carrying a sentinel, and the assistant's reply.
fn cron_exchange() -> Vec<Message> {
    vec![
        Message::user("remind me to water the plants"),
        Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(
                "cron-1",
                "cron",
                json!({"schedule": "0 9 * * *", "note": format!("water the plants {}", "z".repeat(300))}),
            )],
            usage: None,
            origin: None,
        }),
        Message::tool("cron-1", "SENTINEL-TOOL-RESULT job 42 created"),
        Message::assistant("Done, I will remind you at 9 every morning."),
    ]
}

#[tokio::test]
async fn the_default_rendering_is_unchanged() {
    let expected = "user: remind me to water the plants\n\
assistant: <reasoning>I should inspect the matching records.</reasoning> \
<tool_call id=\"lookup-1\" name=\"lookup\">{\"query\":\"open issues\"}</tool_call>\n\
tool: <tool_result id=\"lookup-1\"><json>{\"matches\":2}</json></tool_result>\n\
assistant: There are 2.";
    let mut messages = vec![Message::user("remind me to water the plants")];
    messages.extend(tool_call_messages(json!({"query": "open issues"})));
    messages.push(Message::assistant("There are 2."));

    for rendering in [None, Some(ToolRendering::Full)] {
        let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
        let mut summarizer = ModelSummarizer::new(model.clone(), "m");
        if let Some(rendering) = rendering {
            summarizer = summarizer.with_tool_rendering(rendering);
        }
        summarizer.summarize(&messages).await.unwrap();
        let request = &model.requests()[0];
        let sent = request.messages[1].text();
        assert!(
            sent.contains(&format!("<transcript>\n{expected}\n</transcript>")),
            "{sent}"
        );
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.temperature, None);
    }
}

#[tokio::test]
async fn markers_send_the_conversation_without_tool_output() {
    let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
    let summarizer =
        ModelSummarizer::new(model.clone(), "m").with_tool_rendering(ToolRendering::Markers);
    summarizer.summarize(&cron_exchange()).await.unwrap();

    let sent = model.requests()[0].messages[1].text();
    assert!(
        sent.contains("user: remind me to water the plants"),
        "{sent}"
    );
    assert!(
        sent.contains("assistant: Done, I will remind you at 9 every morning."),
        "{sent}"
    );
    assert!(sent.contains("[used cron: "), "{sent}");
    assert!(!sent.contains("SENTINEL-TOOL-RESULT"), "{sent}");
    assert!(!sent.contains("<tool_call"), "{sent}");
    assert!(!sent.contains("<tool_result"), "{sent}");
    assert!(!sent.contains("cron-1"), "call ids are dropped: {sent}");
    let marker = sent
        .lines()
        .find(|line| line.starts_with("[used cron: "))
        .expect("one marker line");
    let shown = &marker["[used cron: ".len()..marker.len() - 1];
    assert!(shown.chars().count() <= 120, "{marker}");
    assert!(!sent.contains(&"z".repeat(120)), "{sent}");
}

#[tokio::test]
async fn max_output_tokens_rides_the_request() {
    let model = Arc::new(ScriptedModel::replies(vec!["## Goal\nx"]));
    let summarizer = ModelSummarizer::new(model.clone(), "m").with_max_output_tokens(512);
    summarizer.summarize(&[Message::user("x")]).await.unwrap();

    let request = &model.requests()[0];
    assert_eq!(request.max_tokens, Some(512));
    assert_eq!(request.temperature, None);
}
