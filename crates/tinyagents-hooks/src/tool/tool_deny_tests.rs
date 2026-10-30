use super::*;
use crate::command::{HookOutput, HookPermission};
use std::sync::Arc;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::host::FixedModelResolver;
use tinyagents_harness::limits::RunLimits;
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::{FakeTool, ScriptedModel, text_response, tool_call_response};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;

struct Host;
impl HookIdentity for Host {
    fn session_id(&self) -> Option<String> {
        Some("session".into())
    }
    fn agent_id(&self) -> Option<String> {
        Some("agent".into())
    }
    fn cwd(&self) -> Option<std::path::PathBuf> {
        None
    }
}

struct AllowListHook {
    allowed: &'static [&'static str],
    refusal: HookOutput,
}

#[async_trait]
impl ToolHook<Host> for AllowListHook {
    fn name(&self) -> &str {
        "allow_list"
    }
    async fn before_tool_decision(
        &self,
        _host: &Host,
        context: &ToolHookContext,
    ) -> ToolHookDecision {
        if self.allowed.contains(&context.tool_name.as_str()) {
            ToolHookDecision::default()
        } else {
            self.refusal.clone()
        }
    }
}

struct Fixture {
    harness: AgentHarness<(), Host>,
    model: Arc<ScriptedModel>,
    secret: Arc<FakeTool>,
    lookup: Arc<FakeTool>,
}

fn fixture(refusal: HookOutput, responses: Vec<ModelResponse>) -> Fixture {
    let model = Arc::new(ScriptedModel::new(responses));
    let secret = Arc::new(FakeTool::returning("secret", "secret ran"));
    let lookup = Arc::new(FakeTool::returning("lookup", "found"));
    let mut harness: AgentHarness<(), Host> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(FixedModelResolver::named(
        "scripted",
        model.clone(),
    )));
    harness.register_tool(secret.clone());
    harness.register_tool(lookup.clone());
    harness.push_middleware(Arc::new(ToolHooksMiddleware::new(vec![Arc::new(
        AllowListHook {
            allowed: &["lookup"],
            refusal,
        },
    )])));
    Fixture {
        harness,
        model,
        secret,
        lookup,
    }
}

async fn run(fixture: &Fixture) -> tinyagents_harness::error::Result<AgentRun> {
    fixture
        .harness
        .invoke_in_context(
            &(),
            RunContext::new(RunConfig::new("deny"), Host),
            vec![Message::user("go")],
        )
        .await
}

fn secret_call(id: &str) -> ModelResponse {
    tool_call_response(ToolCall::new(
        id,
        "secret",
        serde_json::json!({"token": "arg-value-must-not-leak"}),
    ))
}

fn tool_result_text(messages: &[Message], call_id: &str) -> Option<String> {
    messages.iter().find_map(|message| match message {
        Message::Tool(tool) if tool.tool_call_id == call_id => Some(message.text()),
        _ => None,
    })
}

#[tokio::test]
async fn denied_call_is_answered_to_the_model_and_the_run_finishes() {
    let fixture = fixture(
        HookOutput::deny("secret is not available to this agent"),
        vec![secret_call("call-1"), text_response("done without it")],
    );

    let run = run(&fixture)
        .await
        .expect("a denial must not abort the run");

    assert!(
        fixture.secret.calls().is_empty(),
        "a denied tool never runs"
    );
    assert_eq!(
        run.final_response
            .as_ref()
            .map(ModelResponse::text)
            .as_deref(),
        Some("done without it")
    );
    let answer = tool_result_text(&run.messages, "call-1").expect("the call is answered");
    assert!(
        answer.contains("secret is not available to this agent"),
        "the model sees the deny reason: {answer}"
    );
    let second_request = &fixture.model.requests()[1];
    assert!(second_request.messages.iter().any(|message| matches!(
        message,
        Message::Tool(tool) if tool.tool_call_id == "call-1"
    )));
}

#[tokio::test]
async fn denied_call_with_continue_false_aborts_with_a_refusal_error() {
    let refusal = HookOutput {
        continue_: Some(false),
        ..HookOutput::deny("secret is forbidden")
    };
    let fixture = fixture(refusal, vec![secret_call("call-1"), text_response("never")]);

    let error = run(&fixture)
        .await
        .expect_err("continue_ = false stops the run");

    assert!(
        fixture.secret.calls().is_empty(),
        "a denied tool never runs"
    );
    let TinyAgentsError::ToolRefused(message) = &error else {
        panic!("expected ToolRefused, got {error:?}");
    };
    assert!(message.contains("secret is forbidden"), "{message}");
    assert!(message.contains("secret"), "{message}");
    assert!(
        !error.to_string().contains("arg-value-must-not-leak"),
        "the refusal never carries tool arguments: {error}"
    );
    assert_eq!(fixture.model.requests().len(), 1);
}

#[tokio::test]
async fn ask_still_refuses_the_run() {
    let ask = HookOutput {
        permission: Some(HookPermission::Ask),
        agent_message: Some("needs a human".into()),
        ..Default::default()
    };
    let fixture = fixture(ask, vec![secret_call("call-1"), text_response("never")]);

    let error = run(&fixture).await.expect_err("ask is unchanged");

    assert!(fixture.secret.calls().is_empty());
    assert!(
        matches!(&error, TinyAgentsError::Tool(message) if message.contains("requires approval for secret")),
        "{error:?}"
    );
}

#[tokio::test]
async fn blank_tool_name_denied_by_a_hook_is_answered_not_fatal() {
    let fixture = fixture(
        HookOutput::deny("that tool is not on the allow list"),
        vec![
            tool_call_response(ToolCall::new("call-blank", "", serde_json::json!({}))),
            tool_call_response(ToolCall::new(
                "call-lookup",
                "lookup",
                serde_json::json!({}),
            )),
            text_response("recovered"),
        ],
    );

    let run = run(&fixture)
        .await
        .expect("a blank denied name is answered");

    let answer = tool_result_text(&run.messages, "call-blank").expect("answered");
    assert!(answer.contains("not on the allow list"), "{answer}");
    assert_eq!(fixture.lookup.calls().len(), 1);
    assert_eq!(
        run.final_response
            .as_ref()
            .map(ModelResponse::text)
            .as_deref(),
        Some("recovered")
    );
}

#[tokio::test]
async fn a_model_that_keeps_calling_a_denied_tool_stops_at_the_model_call_cap() {
    let responses = (0..20).map(|i| secret_call(&format!("call-{i}"))).collect();
    let mut fixture = fixture(HookOutput::deny("never allowed"), responses);
    fixture.harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_calls(3),
        ..RunPolicy::default()
    });

    let error = run(&fixture).await.expect_err("the loop must terminate");

    assert!(
        matches!(error, TinyAgentsError::LimitExceeded(_)),
        "{error:?}"
    );
    assert_eq!(fixture.model.requests().len(), 3);
    assert!(
        fixture.secret.calls().is_empty(),
        "a denied tool never runs"
    );
}
