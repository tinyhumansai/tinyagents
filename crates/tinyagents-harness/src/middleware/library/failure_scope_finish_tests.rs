use super::*;

use crate::context::RunConfig;
use crate::middleware::AgentRun;
use crate::no_progress::ClassifiedStopCause;

#[derive(Default)]
struct LatchPolicy {
    cause: Mutex<Option<ClassifiedStopCause>>,
}

impl FailurePolicy for LatchPolicy {
    fn identical_halt_summary(&self, tool: &str, count: u32, _failure_text: &str) -> String {
        format!("{tool} failed {count} times")
    }

    fn no_progress_halt_summary(&self, count: u32, tool: &str, _failure_text: &str) -> String {
        format!("{tool} made no progress in {count} calls")
    }
}

impl FailureScopePolicy<()> for LatchPolicy {
    fn assess(
        &self,
        _ctx: &RunContext<()>,
        _invocation: &ToolInvocationIdentity,
        _call: &ToolCall,
        _scope: &FailureScope,
        _result: &ToolResult,
    ) -> FailureAssessment {
        FailureAssessment::Ignore
    }

    fn current_halt_cause(&self, _ctx: &RunContext<()>) -> Option<ClassifiedStopCause> {
        self.cause.lock().unwrap().clone()
    }

    fn finish_run(&self, _ctx: &RunContext<()>) {
        *self.cause.lock().unwrap() = None;
    }
}

fn cause(class: &str) -> ClassifiedStopCause {
    ClassifiedStopCause {
        class: class.into(),
        operation: "send_message".into(),
        summary: format!("Stopped after {class}"),
    }
}

async fn finish(
    latched: Option<ClassifiedStopCause>,
    recorded: Option<ClassifiedStopCause>,
) -> AgentRun {
    let policy = Arc::new(LatchPolicy {
        cause: Mutex::new(latched),
    });
    let middleware = FailureScopeMiddleware::new(
        SteeringHandle::allow_all(),
        Arc::new(Mutex::new(None)),
        policy.clone(),
    );
    let mut ctx = RunContext::new(RunConfig::new("finish-cause"), ());
    let mut run = AgentRun {
        failure_cause: recorded,
        ..AgentRun::default()
    };

    middleware
        .after_agent(&mut ctx, &(), &mut run)
        .await
        .unwrap();

    assert!(
        policy.cause.lock().unwrap().is_none(),
        "finish must still discard the host's per-run latch"
    );
    run
}

#[tokio::test]
async fn classified_cause_survives_host_latch_cleanup() {
    let expected = cause("uncertain_side_effect");
    let run = finish(Some(expected.clone()), None).await;
    assert_eq!(run.failure_cause, Some(expected));
}

#[tokio::test]
async fn finish_preserves_an_already_recorded_cause() {
    let recorded = cause("provider_out_of_budget");
    let run = finish(Some(cause("uncertain_side_effect")), Some(recorded.clone())).await;
    assert_eq!(run.failure_cause, Some(recorded));
}

#[tokio::test]
async fn unclassified_finish_does_not_invent_a_cause() {
    assert_eq!(finish(None, None).await.failure_cause, None);
}
