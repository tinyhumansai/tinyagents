//! The per-session tool-result expiry option: whether a run replaces old tool
//! results in the transcript it stores with a one-line note.

use std::sync::Arc;

use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::result_expiry::{ExpiryOverrides, ToolResultExpiryPolicy};
use crate::runtime::{AgentHarness, RunPolicy};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn answer(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(1, 1)),
            origin: None,
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn harness(policy: RunPolicy) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_policy(policy);
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(vec![answer("done")])),
    )));
    harness
}

/// A thread with three finished turns, each of which called `lookup`, ending
/// in the user message the run is about to answer.
fn history() -> Vec<Message> {
    let mut messages = Vec::new();
    for n in 1..=3 {
        let id = format!("call-{n}");
        messages.push(Message::user(format!("question {n}")));
        messages.push(Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id.clone(), "lookup", json!({}))],
            usage: None,
            origin: None,
        }));
        messages.push(Message::tool(id, format!("result-{n}")));
    }
    messages.push(Message::user("question 4"));
    messages
}

fn expiry_config(overrides: ExpiryOverrides) -> RunConfig {
    RunConfig::new("expiry").with_result_expiry_overrides(overrides)
}

/// Runs one turn over `history()` and returns the stored tool-result texts for
/// `call-1` .. `call-3`.
async fn stored_results(harness: &AgentHarness<()>, config: RunConfig) -> Vec<String> {
    let ctx = RunContext::new(config, ());
    let result = harness
        .invoke_in_context_with_status(&(), ctx, history())
        .await
        .expect("the run finishes");
    (1..=3)
        .map(|n| {
            let id = format!("call-{n}");
            result
                .run
                .messages
                .iter()
                .find_map(|m| match m {
                    Message::Tool(t) if t.tool_call_id == id => Some(m.text()),
                    _ => None,
                })
                .expect("the result is still in the transcript")
        })
        .collect()
}

fn unchanged() -> Vec<String> {
    vec!["result-1".into(), "result-2".into(), "result-3".into()]
}

#[tokio::test]
async fn a_session_that_opts_in_stores_old_results_as_notes() {
    let stored = stored_results(
        &harness(RunPolicy::default()),
        expiry_config(ExpiryOverrides {
            enabled: Some(true),
            max_age_turns: Some(2),
            ..ExpiryOverrides::default()
        }),
    )
    .await;
    assert_eq!(
        stored,
        vec!["[used lookup: result-1]", "result-2", "result-3"]
    );
}

#[tokio::test]
async fn with_the_rule_off_the_transcript_is_unchanged() {
    // No override at all, and an override that sets only the age.
    for overrides in [
        ExpiryOverrides::default(),
        ExpiryOverrides {
            max_age_turns: Some(0),
            ..ExpiryOverrides::default()
        },
    ] {
        let stored = stored_results(&harness(RunPolicy::default()), expiry_config(overrides)).await;
        assert_eq!(stored, unchanged());
    }
}

#[tokio::test]
async fn a_harness_wide_policy_applies_and_a_session_can_turn_it_off() {
    let policy = RunPolicy {
        result_expiry: ToolResultExpiryPolicy {
            enabled: true,
            max_age_turns: 1,
            exempt_tools: Vec::new(),
        },
        ..RunPolicy::default()
    };
    let on = stored_results(&harness(policy.clone()), RunConfig::new("on")).await;
    assert_eq!(
        on,
        vec![
            "[used lookup: result-1]",
            "[used lookup: result-2]",
            "result-3"
        ]
    );

    let off = stored_results(
        &harness(policy),
        expiry_config(ExpiryOverrides {
            enabled: Some(false),
            ..ExpiryOverrides::default()
        }),
    )
    .await;
    assert_eq!(off, unchanged());
}

#[tokio::test]
async fn a_session_exempt_list_keeps_that_tools_results() {
    let stored = stored_results(
        &harness(RunPolicy::default()),
        expiry_config(ExpiryOverrides {
            enabled: Some(true),
            max_age_turns: Some(0),
            exempt_tools: Some(vec!["lookup".to_string()]),
        }),
    )
    .await;
    assert_eq!(stored, unchanged());
}

#[tokio::test]
async fn a_rewritten_seed_tells_transcript_mirrors_to_resynchronise() {
    use crate::events::AgentEvent;
    let harness = harness(RunPolicy::default());
    let config = expiry_config(ExpiryOverrides {
        enabled: Some(true),
        max_age_turns: Some(0),
        exempt_tools: None,
    });
    let recorder = crate::testkit::EventRecorder::new();
    let ctx = RunContext::new(config, ()).with_events(recorder.sink());
    harness
        .invoke_in_context_with_status(&(), ctx, history())
        .await
        .expect("the run finishes");
    assert!(recorder.events().iter().any(|event| matches!(
        event,
        AgentEvent::TranscriptRewritten { reason, .. } if reason == "result_expiry"
    )));
}
