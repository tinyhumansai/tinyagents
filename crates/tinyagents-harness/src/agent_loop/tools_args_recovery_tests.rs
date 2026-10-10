//! End-to-end regressions for tool-call argument shapes seen from production
//! models (DeepSeek via OpenHuman 0.64.x): a real call wrapped in a pseudo
//! `tool_call` tool, `use_skill` with a stringified or `null` nested value,
//! and a validation corrective that used to echo the whole schema back.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::runtime::{AgentHarness, InvalidArgsPolicy, RunPolicy};
use crate::testkit::{EventRecorder, ScriptedModel, text_response, tool_call_response};
use tinyinference_llm::message::Message;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

/// A stand-in for the pack tool: the `use_skill` schema, recording what it
/// receives.
struct SkillTool {
    received: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "use_skill"
    }
    fn description(&self) -> &str {
        "Reach a skill's tools. A long description the corrective must not repeat."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": { "type": "string", "enum": ["email", "calendar"], "description": "Skill to read or run a tool from." },
                "tool": { "type": "string", "description": "Tool to run. Omit to list the skill's tools and their arguments instead." },
                "args": {
                    "type": "object",
                    "description": "The tool's own arguments, as documented in the listing.",
                    "additionalProperties": true
                }
            },
            "required": ["skill"]
        })
    }
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.received.lock().unwrap().push(args);
        Ok(ToolResult::success("ran"))
    }
}

struct Outcome {
    received: Vec<Value>,
    transcript: String,
    events: Vec<AgentEvent>,
}

async fn run_one(name: &str, arguments: Value) -> Outcome {
    let received = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(ToolCall::new("call-1", name, arguments)),
        text_response("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model as _);
    harness.register_tool(Arc::new(SkillTool {
        received: Arc::clone(&received),
    }));
    harness.with_policy(RunPolicy {
        invalid_args: InvalidArgsPolicy::NormalizeThenReturnToolError,
        ..RunPolicy::default()
    });
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("run-args"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("argument defects are recoverable, never fatal");
    let received = received.lock().unwrap().clone();
    Outcome {
        received,
        transcript: format!("{:?}", run.messages),
        events: recorder.events(),
    }
}

// ── use_skill nested-value defects ─────────────────────────────────────────

#[tokio::test]
async fn use_skill_args_sent_as_relaxed_json_string_runs_the_tool() {
    let out = run_one(
        "use_skill",
        json!({ "skill": "email", "tool": "send", "args": "{to: \"a@b.c\", \"subject\": \"hi\",}" }),
    )
    .await;
    assert_eq!(
        out.received,
        vec![
            json!({ "skill": "email", "tool": "send", "args": { "to": "a@b.c", "subject": "hi" } })
        ],
        "{}",
        out.transcript
    );
}

#[tokio::test]
async fn use_skill_with_null_tool_and_args_runs_the_listing_half() {
    let out = run_one(
        "use_skill",
        json!({ "skill": "email", "tool": null, "args": null }),
    )
    .await;
    assert_eq!(
        out.received,
        vec![json!({ "skill": "email" })],
        "{}",
        out.transcript
    );
}

#[tokio::test]
async fn validation_corrective_is_a_short_hint_not_the_whole_schema() {
    let out = run_one("use_skill", json!({ "skill": "mail" })).await;
    assert!(out.received.is_empty());
    assert!(
        out.transcript
            .contains("invalid arguments for tool `use_skill`"),
        "{}",
        out.transcript
    );
    assert!(
        out.transcript.contains("expected arguments:"),
        "{}",
        out.transcript
    );
    // The enum the model got wrong is named, so it can correct itself.
    assert!(out.transcript.contains("email"), "{}", out.transcript);
    // Property descriptions and JSON-Schema scaffolding are not echoed.
    assert!(
        !out.transcript.contains("Omit to list the skill's tools"),
        "{}",
        out.transcript
    );
    assert!(
        !out.transcript.contains("additionalProperties"),
        "{}",
        out.transcript
    );
}

// ── pseudo `tool_call` wrapper ──────────────────────────────────────────────

fn unwrapped_to(events: &[AgentEvent]) -> Option<String> {
    events.iter().find_map(|event| match event {
        AgentEvent::InvalidToolArgs { recovery, .. } if recovery.starts_with("unwrapped:") => {
            Some(recovery.clone())
        }
        _ => None,
    })
}

#[tokio::test]
async fn a_tool_call_wrapper_naming_a_tool_is_retargeted() {
    for (wrapper, arguments) in [
        (
            "tool_call",
            json!({ "name": "use_skill", "arguments": { "skill": "email" } }),
        ),
        (
            "tool_call",
            json!({ "name": "use_skill", "arguments": "{\"skill\": \"email\"}" }),
        ),
        ("call_tool", json!({ "use_skill": { "skill": "email" } })),
        (
            "function_call",
            json!({ "function": { "name": "use_skill", "arguments": "{\"skill\":\"email\"}" } }),
        ),
        ("tool_use", json!({ "tool": "use_skill", "skill": "email" })),
        ("invoke", json!({ "skill": "email" })),
    ] {
        let out = run_one(wrapper, arguments.clone()).await;
        assert_eq!(
            out.received,
            vec![json!({ "skill": "email" })],
            "{wrapper} {arguments}: {}",
            out.transcript
        );
        assert_eq!(
            unwrapped_to(&out.events).as_deref(),
            Some("unwrapped:use_skill"),
            "{wrapper} {arguments}"
        );
        assert!(
            !out.events
                .iter()
                .any(|event| matches!(event, AgentEvent::UnknownToolCall { .. })),
            "a recovered wrapper is not an unknown-tool failure"
        );
    }
}

#[tokio::test]
async fn the_skill_tool_args_shape_inside_a_wrapper_means_use_skill() {
    let out = run_one(
        "tool_call",
        json!({ "skill": "email", "tool": "send", "args": { "to": "a@b.c" } }),
    )
    .await;
    assert_eq!(
        out.received,
        vec![json!({ "skill": "email", "tool": "send", "args": { "to": "a@b.c" } })],
        "{}",
        out.transcript
    );
}

#[tokio::test]
async fn a_wrapper_naming_no_registered_tool_stays_an_unknown_tool() {
    let out = run_one(
        "tool_call",
        json!({ "name": "frobnicate", "arguments": {} }),
    )
    .await;
    assert!(out.received.is_empty());
    assert!(
        out.transcript.contains("unknown tool `tool_call`"),
        "{}",
        out.transcript
    );
    assert!(unwrapped_to(&out.events).is_none());
}
