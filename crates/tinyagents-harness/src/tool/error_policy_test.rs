//! Tests for [`ToolErrorPolicy`].
//!
//! The load-bearing case is the last one: a cancellation or an interrupt must
//! bubble no matter what policy a tool declares. Converting one into a tool
//! result would tell the model "the tool failed", let the loop continue, and
//! silently defeat the cancel.

use super::*;
use crate::error::TinyAgentsError;
use serde_json::json;

fn call() -> ToolCall {
    ToolCall::new("c1", "lookup", json!({}))
}

#[test]
fn fail_is_the_default_and_propagates() {
    assert_eq!(ToolErrorPolicy::default(), ToolErrorPolicy::Fail);

    let outcome = ToolErrorPolicy::Fail.apply(&call(), Err(TinyAgentsError::Tool("boom".into())));
    assert!(matches!(outcome, Err(TinyAgentsError::Tool(_))));
}

#[test]
fn return_to_error_hands_the_message_to_the_model() {
    let outcome =
        ToolErrorPolicy::ReturnToError.apply(&call(), Err(TinyAgentsError::Tool("boom".into())));
    let result = outcome.expect("a handled error must not fail the run");

    assert!(result.is_error());
    assert_eq!(result.call_id, "c1");
    assert_eq!(result.name, "lookup");
    assert!(result.content.contains("boom"));
}

#[test]
fn a_fixed_message_masks_an_error_that_should_not_be_shown() {
    let outcome = ToolErrorPolicy::Message("the lookup service is unavailable".into()).apply(
        &call(),
        Err(TinyAgentsError::Tool(
            "postgres://user:pw@db/internal timed out".into(),
        )),
    );
    let result = outcome.unwrap();

    assert_eq!(result.content, "the lookup service is unavailable");
    assert!(!result.content.contains("postgres"));
}

#[test]
fn successful_results_pass_through_untouched() {
    // A tool that already chose `Ok(ToolResult::error(..))` has made its own
    // decision; the policy does not second-guess it.
    let declared = ToolResult::error("c1", "lookup", "not found");
    let outcome = ToolErrorPolicy::Fail.apply(&call(), Ok(declared.clone()));
    assert_eq!(outcome.unwrap(), declared);
}

#[test]
fn cancellation_and_interruption_always_bubble() {
    for policy in [
        ToolErrorPolicy::Fail,
        ToolErrorPolicy::ReturnToError,
        ToolErrorPolicy::Message("masked".into()),
    ] {
        let cancelled = policy.apply(&call(), Err(TinyAgentsError::Cancelled));
        assert!(
            matches!(cancelled, Err(TinyAgentsError::Cancelled)),
            "policy {policy:?} swallowed a cancellation"
        );

        let interrupted = policy.apply(
            &call(),
            Err(TinyAgentsError::Interrupted {
                node: "approval".into(),
                message: "waiting for a human".into(),
            }),
        );
        assert!(
            matches!(interrupted, Err(TinyAgentsError::Interrupted { .. })),
            "policy {policy:?} swallowed an interrupt"
        );
    }

    assert!(is_control_flow_error(&TinyAgentsError::Cancelled));
    assert!(!is_control_flow_error(&TinyAgentsError::Tool(
        "boom".into()
    )));
}

#[test]
fn registry_exposes_per_tool_error_policies() {
    use async_trait::async_trait;

    struct Flaky;

    #[async_trait]
    impl Tool<()> for Flaky {
        fn name(&self) -> &str {
            "flaky"
        }
        fn description(&self) -> &str {
            "sometimes fails"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("flaky", "sometimes fails", json!({"type": "object"}))
        }
        fn error_policy(&self) -> ToolErrorPolicy {
            ToolErrorPolicy::ReturnToError
        }
        async fn call(&self, _state: &(), call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::text(call.id, call.name, "ok"))
        }
    }

    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(std::sync::Arc::new(Flaky));
    assert_eq!(
        registry.error_policies().get("flaky"),
        Some(&ToolErrorPolicy::ReturnToError)
    );
}
