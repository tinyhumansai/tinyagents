use std::time::Duration;

use tinyagents_harness::retry::RetryPolicy;

use super::*;

fn transient() -> TinyAgentsError {
    TinyAgentsError::Tool("flaky".into())
}

fn retrying() -> SubAgentPolicy {
    SubAgentPolicy::default().with_retry(
        RetryPolicy::default()
            .with_max_attempts(3)
            .with_backoff_sleep(false),
    )
}

#[test]
fn default_policy_never_retries() {
    assert!(!may_retry(
        &SubAgentPolicy::default(),
        0,
        &transient(),
        false
    ));
}

#[test]
fn retries_a_retryable_failure_until_attempts_run_out() {
    let policy = retrying();
    assert!(may_retry(&policy, 0, &transient(), false));
    assert!(may_retry(&policy, 1, &transient(), false));
    assert!(!may_retry(&policy, 2, &transient(), false));
}

#[test]
fn never_retries_a_permanent_failure() {
    let permanent = TinyAgentsError::Validation("bad".into());
    assert!(!may_retry(&retrying(), 0, &permanent, false));
}

#[test]
fn terminal_budget_refusals_never_retry_even_with_a_custom_predicate() {
    let policy = retrying().with_retry(
        RetryPolicy::default()
            .with_max_attempts(3)
            .with_retry_on(std::sync::Arc::new(|_| true)),
    );
    let refusal = TinyAgentsError::LimitExceeded("budget exceeded: max tokens".into());

    assert!(!may_retry(&policy, 0, &refusal, false));
}

#[test]
fn does_not_retry_after_tools_ran_unless_the_policy_allows_it() {
    assert!(!may_retry(&retrying(), 0, &transient(), true));
    let allowed = retrying().with_retry_after_tool_calls(true);
    assert!(may_retry(&allowed, 0, &transient(), true));
}

#[test]
fn the_orchestration_path_names_the_same_policy_type_as_the_graph() {
    let policy: tinyagents_graph::SubAgentPolicy =
        SubAgentPolicy::default().with_timeout(Duration::from_secs(1));
    assert_eq!(policy.timeout, Some(Duration::from_secs(1)));
}
