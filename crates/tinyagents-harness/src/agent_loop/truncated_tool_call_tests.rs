//! A tool call cut off by the output token limit (production: a model writing
//! a whole workflow document into one call's arguments, cut at a 16,384-token
//! cap, then re-sending the same oversized call). The corrective must tell the
//! model not to repeat the call and to split it, and a second cut-off call of
//! the same tool must stop that retry shape instead of looping on it.

use std::sync::Arc;

use serde_json::json;

use crate::error::TinyAgentsError;
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{FakeTool, ScriptedModel, text_response, tool_call_response};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;

fn cut_off(id: &str, tool: &str) -> ModelResponse {
    let mut response = tool_call_response(ToolCall::new(
        id,
        tool,
        json!({ "workflow": "{\"nodes\": [ ... cut" }),
    ));
    response.finish_reason = Some("length".to_string());
    response
}

fn harness(responses: Vec<ModelResponse>, retries: u32) -> (AgentHarness<()>, Arc<FakeTool>) {
    let tool = Arc::new(FakeTool::returning("save_workflow", "saved"));
    let other = Arc::new(FakeTool::returning("write_file", "written"));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(ScriptedModel::new(responses)) as _)
        .register_tool(Arc::clone(&tool) as _)
        .register_tool(other as _);
    harness.with_policy(RunPolicy {
        truncated_tool_call_retries: retries,
        ..RunPolicy::default()
    });
    (harness, tool)
}

fn tool_rows(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .collect()
}

#[tokio::test]
async fn a_cut_off_call_is_answered_with_do_not_repeat_and_split() {
    let (harness, tool) = harness(vec![cut_off("c1", "save_workflow"), text_response("ok")], 2);
    let run = harness
        .invoke_default(&(), vec![Message::user("build it")])
        .await
        .expect("one truncation is recoverable");
    assert!(tool.calls().is_empty(), "the cut-off call must not run");
    let rows = tool_rows(&run.messages);
    let text = rows.first().expect("the call is answered");
    assert!(text.contains("cut off at the output token limit"), "{text}");
    assert!(text.contains("Do not repeat"), "{text}");
    assert!(text.contains("smaller"), "{text}");
    assert!(
        !text.contains("Re-issue the tool call with complete arguments"),
        "the corrective must not invite the same oversized call: {text}"
    );
}

#[tokio::test]
async fn a_second_cut_off_call_of_the_same_tool_stops_that_shape() {
    // A generous budget: the same-tool repeat, not the budget, is what stops it.
    let (harness, _tool) = harness(
        vec![
            cut_off("c1", "save_workflow"),
            cut_off("c2", "save_workflow"),
            cut_off("c3", "save_workflow"),
            text_response("never reached"),
        ],
        5,
    );
    let err = harness
        .invoke_default(&(), vec![Message::user("build it")])
        .await
        .expect_err("a third oversized call of the same tool ends the run");
    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");
}

#[tokio::test]
async fn the_repeat_corrective_says_to_stop_retrying() {
    let (harness, _tool) = harness(
        vec![
            cut_off("c1", "save_workflow"),
            cut_off("c2", "save_workflow"),
            text_response("ok"),
        ],
        5,
    );
    let run = harness
        .invoke_default(&(), vec![Message::user("build it")])
        .await
        .expect("the model may still recover after the second corrective");
    let rows = tool_rows(&run.messages);
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(rows[1].contains("again"), "{}", rows[1]);
    assert!(rows[1].contains("Stop sending"), "{}", rows[1]);
    assert!(!rows[0].contains("Stop sending"), "{}", rows[0]);
}

#[tokio::test]
async fn cut_off_calls_of_different_tools_keep_the_ordinary_budget() {
    let (harness, _tool) = harness(
        vec![
            cut_off("c1", "save_workflow"),
            cut_off("c2", "write_file"),
            cut_off("c3", "save_workflow"),
            text_response("ok"),
        ],
        5,
    );
    let run = harness
        .invoke_default(&(), vec![Message::user("build it")])
        .await
        .expect("alternating tools are not the same shape");
    assert_eq!(run.text(), Some("ok".to_string()));
}
