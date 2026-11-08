//! The per-session reasoning replay option: whether the requests of a run
//! carry earlier turns' reasoning. The stored transcript is never rewritten.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::reasoning_replay::ReasoningReplay;
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::ScriptedModel;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

struct LookupTool;

#[async_trait]
impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "test tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("tool-output"))
    }
}

fn signed(text: &str, signature: &str) -> ContentBlock {
    ContentBlock::Thinking {
        text: text.to_string(),
        signature: Some(signature.to_string()),
    }
}

/// Two finished turns, each with signed reasoning, ending in the user message
/// the run answers (turn 3).
fn history() -> Vec<Message> {
    let mut messages = Vec::new();
    for n in 1..=2 {
        messages.push(Message::user(format!("question {n}")));
        messages.push(Message::Assistant(AssistantMessage {
            id: None,
            content: vec![
                signed(&format!("reasoning-{n}"), &format!("sig-{n}")),
                ContentBlock::Text(format!("answer {n}")),
            ],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        }));
    }
    messages.push(Message::user("question 3"));
    messages
}

/// Turn 3 reasons, calls a tool, then answers: two requests.
fn scripted_model() -> Arc<ScriptedModel> {
    let mut call = ModelResponse::assistant("");
    call.message.content = vec![signed("reasoning-3", "sig-3")];
    call.message.tool_calls = vec![ToolCall::new("call-3", "lookup", json!({}))];
    call.finish_reason = Some("tool_calls".to_string());
    Arc::new(ScriptedModel::new(vec![
        call,
        ModelResponse::assistant("done"),
    ]))
}

fn reasoning_texts(request: &ModelRequest) -> Vec<ContentBlock> {
    request
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant(a) => Some(a.content.iter().cloned()),
            _ => None,
        })
        .flatten()
        .filter(|block| matches!(block, ContentBlock::Thinking { .. }))
        .collect()
}

/// Runs turn 3 and returns the requests the model saw and the stored messages.
async fn run_turn(policy: RunPolicy, config: RunConfig) -> (Vec<ModelRequest>, Vec<Message>) {
    let model = scripted_model();
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::clone(&model) as _,
    )));
    harness.register_tool(Arc::new(LookupTool));
    harness.with_policy(policy);
    let result = harness
        .invoke_in_context_with_status(&(), RunContext::new(config, ()), history())
        .await
        .expect("the run finishes");
    (model.requests(), result.run.messages)
}

fn current_turn_only() -> RunConfig {
    RunConfig::new("replay").with_replay_reasoning(ReasoningReplay::CurrentTurn)
}

#[tokio::test]
async fn current_turn_requests_carry_no_earlier_reasoning_and_keep_the_current_turns() {
    let (requests, _) = run_turn(RunPolicy::default(), current_turn_only()).await;
    assert_eq!(requests.len(), 2);

    // The first request is turn 3's first call: nothing to carry yet.
    assert_eq!(reasoning_texts(&requests[0]), Vec::new());
    // The second follows the tool call: turn 3's signed block is preserved
    // byte for byte, and turns 1 and 2 stay out.
    assert_eq!(
        reasoning_texts(&requests[1]),
        vec![signed("reasoning-3", "sig-3")]
    );
}

#[tokio::test]
async fn current_turn_keeps_everything_but_reasoning_in_the_request() {
    let (requests, _) = run_turn(RunPolicy::default(), current_turn_only()).await;
    let texts: Vec<String> = requests[0].messages.iter().map(Message::text).collect();
    assert_eq!(
        texts,
        vec![
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
            "question 3"
        ]
    );
}

#[tokio::test]
async fn the_stored_transcript_keeps_every_reasoning_block() {
    let (_, stored) = run_turn(RunPolicy::default(), current_turn_only()).await;
    let kept: Vec<ContentBlock> = stored
        .iter()
        .filter_map(|m| match m {
            Message::Assistant(a) => Some(a.content.iter().cloned()),
            _ => None,
        })
        .flatten()
        .filter(|block| matches!(block, ContentBlock::Thinking { .. }))
        .collect();
    assert_eq!(
        kept,
        vec![
            signed("reasoning-1", "sig-1"),
            signed("reasoning-2", "sig-2"),
            signed("reasoning-3", "sig-3"),
        ]
    );
}

#[tokio::test]
async fn by_default_requests_replay_all_reasoning() {
    let all_three = vec![
        signed("reasoning-1", "sig-1"),
        signed("reasoning-2", "sig-2"),
        signed("reasoning-3", "sig-3"),
    ];
    // No policy and no override, and an explicit `All`, send today's requests.
    for config in [
        RunConfig::new("default"),
        RunConfig::new("all").with_replay_reasoning(ReasoningReplay::All),
    ] {
        let (requests, _) = run_turn(RunPolicy::default(), config).await;
        assert_eq!(
            reasoning_texts(&requests[0]),
            all_three[..2].to_vec(),
            "the first request carries both finished turns"
        );
        assert_eq!(reasoning_texts(&requests[1]), all_three);
    }
}

#[tokio::test]
async fn a_harness_wide_policy_applies_and_a_session_can_override_it() {
    let policy = RunPolicy {
        replay_reasoning: ReasoningReplay::CurrentTurn,
        ..RunPolicy::default()
    };
    let (on, _) = run_turn(policy.clone(), RunConfig::new("on")).await;
    assert_eq!(reasoning_texts(&on[0]), Vec::new());

    let (off, _) = run_turn(
        policy,
        RunConfig::new("off").with_replay_reasoning(ReasoningReplay::All),
    )
    .await;
    assert_eq!(reasoning_texts(&off[0]).len(), 2);
}
