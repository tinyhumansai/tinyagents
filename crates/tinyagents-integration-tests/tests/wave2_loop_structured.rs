//! Regression coverage for structured output inside a *tool-using* loop.
//!
//! - **TOOL-7**: under the tool-call structured strategy the loop forced
//!   `tool_choice` onto the artificial schema tool on every turn, so the model
//!   had to emit the structured call immediately and no registered tool could
//!   ever run.
//! - **TOOL-8**: a structured hit terminated the turn even when the model asked
//!   for real tools alongside it, and the schema name was never checked against
//!   the registered tool names.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::FakeTool;
use tinyagents_harness::tool::ToolCall;
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelProfile, ModelRequest, ModelResponse, ToolChoice};

/// The schema the tests ask the model to fill in.
fn schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    })
}

/// A model that records every request it receives and replays a script.
///
/// A profile with `native_structured_output = false` is the whole point: it is
/// what selects `StructuredStrategy::ToolCall`, the path the bug lived on, and
/// it is what `ModelProfile::default()` yields for that field — so this is the
/// ordinary case for a tool-calling provider without native constrained JSON,
/// not an exotic corner.
struct RecordingModel {
    profile: ModelProfile,
    script: Mutex<Vec<ModelResponse>>,
    seen: Mutex<Vec<ModelRequest>>,
}

impl RecordingModel {
    fn new(script: Vec<ModelResponse>) -> Self {
        Self {
            profile: ModelProfile {
                tool_calling: true,
                parallel_tool_calls: true,
                native_structured_output: false,
                json_schema: false,
                ..ModelProfile::default()
            },
            script: Mutex::new(script),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.seen.lock().expect("poisoned").clone()
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference::Result<ModelResponse> {
        self.seen.lock().expect("poisoned").push(request);
        let mut script = self.script.lock().expect("poisoned");
        if script.len() > 1 {
            Ok(script.remove(0))
        } else {
            Ok(script[0].clone())
        }
    }
}

/// Builds a response carrying exactly the supplied tool calls.
fn tool_call_response(calls: Vec<ToolCall>) -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response.message.content.clear();
    response.message.tool_calls = calls;
    response.finish_reason = Some("tool_calls".to_string());
    response
}

/// TOOL-7: with real tools registered, the artificial schema tool is *offered*
/// but never forced, so the model is free to call a registered tool first.
///
/// Before the fix the loop set `tool_choice = Tool("result")` on every request,
/// which compels the provider to emit the structured call on turn 1 and ends
/// the loop before any registered tool can run. The symptom — "my agent never
/// uses its tools" — points nowhere near the structured-output code.
#[tokio::test]
async fn structured_tool_strategy_does_not_force_the_schema_tool_when_real_tools_exist() {
    let model = Arc::new(RecordingModel::new(vec![tool_call_response(vec![
        ToolCall::new("c1", "result", json!({"answer": "42"})),
    ])]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("rec", model.clone());
    harness.register_tool(Arc::new(FakeTool::returning("search", "hits")));
    harness.with_policy(RunPolicy {
        default_response_format: Some(tinyinference::model::ResponseFormat::auto(
            "result",
            schema(),
        )),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.structured, Some(json!({"answer": "42"})));

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        !matches!(requests[0].tool_choice, ToolChoice::Tool(_)),
        "the schema tool must not be forced while real tools are registered, got {:?}",
        requests[0].tool_choice
    );
    assert!(
        requests[0].tools.iter().any(|t| t.name == "search"),
        "the registered tool must still be offered"
    );
    assert!(
        requests[0].tools.iter().any(|t| t.name == "result"),
        "the schema tool must still be offered"
    );
}

/// The terminal case is unchanged: with **no** registered tools the schema tool
/// is the only thing the model can call, so forcing it is correct.
#[tokio::test]
async fn structured_tool_strategy_still_forces_the_schema_tool_with_no_real_tools() {
    let model = Arc::new(RecordingModel::new(vec![tool_call_response(vec![
        ToolCall::new("c1", "result", json!({"answer": "42"})),
    ])]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("rec", model.clone());
    harness.with_policy(RunPolicy {
        default_response_format: Some(tinyinference::model::ResponseFormat::auto(
            "result",
            schema(),
        )),
        ..RunPolicy::default()
    });

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let requests = model.requests();
    assert!(
        matches!(&requests[0].tool_choice, ToolChoice::Tool(name) if name == "result"),
        "with no registered tools the schema tool should be forced, got {:?}",
        requests[0].tool_choice
    );
}

/// TOOL-8a: a turn that returns the schema call **alongside** real tool calls
/// must still execute the real ones.
///
/// Before the fix `structured_tool_hit` was true if *any* returned call matched
/// the schema name, and the loop broke out regardless — `search` never ran and
/// no event said so.
#[tokio::test]
async fn a_structured_hit_does_not_discard_sibling_real_tool_calls() {
    let model = Arc::new(RecordingModel::new(vec![
        tool_call_response(vec![
            ToolCall::new("c1", "search", json!({})),
            ToolCall::new("c2", "result", json!({"answer": "42"})),
        ]),
        tool_call_response(vec![ToolCall::new(
            "c3",
            "result",
            json!({"answer": "final"}),
        )]),
    ]));

    let search = Arc::new(FakeTool::returning("search", "hits"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("rec", model.clone());
    harness.register_tool(search.clone());
    harness.with_policy(RunPolicy {
        default_response_format: Some(tinyinference::model::ResponseFormat::auto(
            "result",
            schema(),
        )),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        search.calls().len(),
        1,
        "the real tool requested in the same turn must still be executed"
    );
    assert_eq!(run.structured, Some(json!({"answer": "final"})));
}

/// TOOL-8b: a schema name that collides with a registered tool would put two
/// identically-named `function` entries in one request — which OpenAI rejects —
/// and makes every returned call ambiguous. Fail closed, up front.
#[tokio::test]
async fn a_schema_name_colliding_with_a_registered_tool_fails_closed() {
    let model = Arc::new(RecordingModel::new(vec![ModelResponse::assistant("hi")]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("rec", model);
    harness.register_tool(Arc::new(FakeTool::returning("result", "hits")));
    harness.with_policy(RunPolicy {
        default_response_format: Some(tinyinference::model::ResponseFormat::auto(
            "result",
            schema(),
        )),
        ..RunPolicy::default()
    });

    let err = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("the collision must be rejected");

    assert!(matches!(err, TinyAgentsError::Validation(_)), "got {err:?}");
    assert!(
        err.to_string().contains("collides"),
        "the error should name the collision, got: {err}"
    );
}
