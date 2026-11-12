//! The staged repeat escalation inside the real agent loop: a blocked call
//! must never reach its tool, and the run must pause on the second block,
//! after one deliver-first turn when the host opts in.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::error::Result as TaResult;
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::runtime::AgentHarness;
use crate::steering::SteeringHandle;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolPolicy, ToolResult};

struct CountingTool {
    runs: Mutex<u32>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "counting tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.runs.lock().unwrap() += 1;
        Ok(ToolResult::success("same answer"))
    }
}

fn repeat_call(turn: usize) -> ModelResponse {
    let mut response = ModelResponse::assistant(format!("attempt {turn}"));
    response.message = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text(format!("attempt {turn}"))],
        tool_calls: vec![ToolCall::new(
            format!("call-{turn}"),
            "lookup",
            json!({"id": 1}),
        )],
        usage: None,
        origin: None,
    };
    response
}

#[tokio::test]
async fn blocked_calls_never_execute_and_the_second_block_pauses_the_run() {
    let steering = SteeringHandle::allow_all();
    let summary: HaltSummarySlot = Arc::new(Mutex::new(None));
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    )));
    harness.register_tool(tool.clone());
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        summary.clone(),
        Arc::new(|_| false),
    )));

    let ctx = RunContext::new(RunConfig::new("repeat-loop"), ()).with_steering(steering.clone());
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a halt pauses the run rather than failing it")
        .run;

    assert_eq!(
        *tool.runs.lock().unwrap(),
        4,
        "attempts 1-4 execute; 5 and 6 are answered without running"
    );
    let texts: Vec<String> = run
        .messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .map(|m| m.text())
        .collect();
    assert!(texts[2].contains("[repeat notice]"), "{texts:?}");
    assert!(texts[4].contains("not executed"), "{texts:?}");
    assert!(
        summary.lock().unwrap().is_some(),
        "the halt names its cause"
    );
}

/// Registered after the guard, so its `after_tool` runs before the guard's.
struct MarkerObserver(Arc<Mutex<Vec<Option<String>>>>);

#[async_trait]
impl Middleware<(), ()> for MarkerObserver {
    fn name(&self) -> &str {
        "marker_observer"
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> TaResult<()> {
        self.0
            .lock()
            .unwrap()
            .push(repeat_guard_marker(result).map(str::to_string));
        Ok(())
    }
}

#[tokio::test]
async fn a_later_registered_after_tool_sees_the_marker_on_refused_calls() {
    let steering = SteeringHandle::allow_all();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    )));
    harness.register_tool(tool);
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        Arc::new(Mutex::new(None)),
        Arc::new(|_| false),
    )));
    harness.push_middleware(Arc::new(MarkerObserver(seen.clone())));

    let ctx = RunContext::new(RunConfig::new("marker"), ()).with_steering(steering);
    harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a halt pauses the run");

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.iter().map(Option::as_deref).collect::<Vec<_>>(),
        [None, None, None, None, Some("blocked"), Some("halted")],
        "executed results are unmarked; refused ones are marked for every hook"
    );
}

#[tokio::test]
async fn a_guard_halt_is_reported_as_a_halted_terminal_outcome() {
    use crate::terminal::{TerminalClass, TerminalReason};
    let steering = SteeringHandle::allow_all();
    let summary: HaltSummarySlot = Arc::new(Mutex::new(None));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    )));
    harness.register_tool(Arc::new(CountingTool {
        runs: Mutex::new(0),
    }));
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        summary.clone(),
        Arc::new(|_| false),
    )));
    let ctx = RunContext::new(RunConfig::new("repeat-halted"), ()).with_steering(steering);
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap()
        .run;

    assert!(run.paused.is_some(), "a halt still pauses the run");
    let outcome = run.terminal.expect("outcome");
    assert_eq!(outcome.reason, TerminalReason::Halted);
    assert_eq!(outcome.class, TerminalClass::Failure);
    assert_eq!(Some(outcome.message), summary.lock().unwrap().clone());
}

/// Sends a host pause at the guard's halting call, so both pauses reach the
/// same steering checkpoint.
struct HostPauseAtHalt(SteeringHandle);

#[async_trait]
impl Middleware<(), ()> for HostPauseAtHalt {
    fn name(&self) -> &str {
        "host_pause_at_halt"
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> TaResult<()> {
        if repeat_guard_marker(result) == Some("halted") {
            self.0.send(crate::steering::SteeringCommand::PauseWith {
                reason: "host review".into(),
            });
        }
        Ok(())
    }
}

/// Sends a host pause from inside the deliver-first turn, after its model call.
struct HostPauseInFinalTurn(SteeringHandle);

#[async_trait]
impl Middleware<(), ()> for HostPauseInFinalTurn {
    fn name(&self) -> &str {
        "host_pause_in_final_turn"
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> TaResult<()> {
        if ctx.deferred_halt.is_some() {
            self.0.send(crate::steering::SteeringCommand::PauseWith {
                reason: "host review".into(),
            });
        }
        Ok(())
    }
}

/// A second guard that halts on the same call as the repeat guard: it marks
/// the run and pauses the way the library guards do.
struct GuardPauseAtHalt(SteeringHandle);

#[async_trait]
impl Middleware<(), ()> for GuardPauseAtHalt {
    fn name(&self) -> &str {
        "guard_pause_at_halt"
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> TaResult<()> {
        if repeat_guard_marker(result) == Some("halted") && !ctx.deliver_first_attempted {
            ctx.halted_by_guard
                .get_or_insert_with(|| "second guard".into());
            self.0.note_guard_pause();
            self.0.send(crate::steering::SteeringCommand::Pause);
        }
        Ok(())
    }
}

/// What else a deliver-first run has registered next to its repeat guard.
#[derive(Clone, Copy, PartialEq)]
enum Alongside {
    Nothing,
    HostPauseAtHalt,
    HostPauseInFinalTurn,
    SecondGuard,
}

/// A run whose guard halts on attempt 6, with the deliver-first turn turned
/// on; `after_halt` are the model's responses from the deliver-first turn on.
async fn deliver_first_run(
    after_halt: Vec<ModelResponse>,
    alongside: Alongside,
) -> (crate::middleware::AgentRun, u32) {
    let steering = SteeringHandle::allow_all();
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..6)
                .map(repeat_call)
                .chain(after_halt)
                .collect::<Vec<_>>(),
        )),
    )));
    harness.register_tool(tool.clone());
    harness.with_policy(crate::runtime::RunPolicy {
        deliver_first_on_guard_halt: true,
        ..Default::default()
    });
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        Arc::new(Mutex::new(None)),
        Arc::new(|_| false),
    )));
    match alongside {
        Alongside::Nothing => {}
        Alongside::HostPauseAtHalt => {
            harness.push_middleware(Arc::new(HostPauseAtHalt(steering.clone())));
        }
        Alongside::HostPauseInFinalTurn => {
            harness.push_middleware(Arc::new(HostPauseInFinalTurn(steering.clone())));
        }
        Alongside::SecondGuard => {
            harness.push_middleware(Arc::new(GuardPauseAtHalt(steering.clone())));
        }
    }
    let ctx = RunContext::new(RunConfig::new("deliver-first"), ()).with_steering(steering);
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .unwrap()
        .run;
    let runs = *tool.runs.lock().unwrap();
    (run, runs)
}

fn nudges(run: &crate::middleware::AgentRun) -> usize {
    run.messages
        .iter()
        .filter(|m| matches!(m, Message::User(_)) && m.text().contains("This is your last turn"))
        .count()
}

#[tokio::test]
async fn an_opted_in_guard_halt_grants_one_turn_to_deliver() {
    let (run, _) = deliver_first_run(
        vec![ModelResponse::assistant("final answer")],
        Alongside::Nothing,
    )
    .await;

    assert_eq!(nudges(&run), 1);
    assert_eq!(run.messages.last().unwrap().text(), "final answer");
    assert!(
        run.paused.is_none(),
        "the model delivered, so the run finished"
    );
}

#[tokio::test]
async fn the_deliver_first_turn_is_the_last_one_even_when_it_calls_a_tool() {
    use crate::terminal::TerminalReason;
    let mut new_call = repeat_call(6);
    new_call.message.tool_calls = vec![ToolCall::new("call-6", "lookup", json!({"id": 2}))];
    let (run, runs) = deliver_first_run(
        vec![new_call, ModelResponse::assistant("never asked for")],
        Alongside::Nothing,
    )
    .await;

    assert_eq!(nudges(&run), 1);
    assert_eq!(
        runs, 5,
        "attempts 1-4 and the deliver-first turn's new call"
    );
    assert!(
        run.messages.iter().all(|m| m.text() != "never asked for"),
        "no model call after the deliver-first turn"
    );
    assert!(run.paused.is_some());
    assert_eq!(run.terminal.unwrap().reason, TerminalReason::Halted);
}

#[tokio::test]
async fn a_host_pause_at_the_halting_checkpoint_is_not_lifted() {
    let (run, runs) = deliver_first_run(
        vec![ModelResponse::assistant("never asked for")],
        Alongside::HostPauseAtHalt,
    )
    .await;

    assert_eq!(nudges(&run), 0, "no deliver-first turn over a host pause");
    assert_eq!(runs, 4);
    assert!(run.paused.is_some());
}

#[tokio::test]
async fn two_guards_halting_together_still_grant_the_deliver_first_turn() {
    let (run, _) = deliver_first_run(
        vec![ModelResponse::assistant("final answer")],
        Alongside::SecondGuard,
    )
    .await;

    assert_eq!(nudges(&run), 1, "two guard pauses are not a host pause");
    assert_eq!(run.messages.last().unwrap().text(), "final answer");
}

#[tokio::test]
async fn a_host_pause_in_the_deliver_first_turn_still_reports_the_guard_halt() {
    use crate::terminal::TerminalReason;
    let (run, _) = deliver_first_run(
        vec![ModelResponse::assistant("final answer")],
        Alongside::HostPauseInFinalTurn,
    )
    .await;

    assert_eq!(nudges(&run), 1);
    assert_eq!(
        run.paused
            .as_ref()
            .and_then(|pause| pause.reason.as_deref()),
        Some("host review"),
        "the host's pause is kept"
    );
    assert_eq!(run.terminal.unwrap().reason, TerminalReason::Halted);
    assert!(run.failure_cause.is_some(), "the guard's cause is reported");
}
