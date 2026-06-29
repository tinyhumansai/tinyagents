//! Tests added in a later pass.

#[test]
fn smoke_retry_policy_compiles() {
    use super::{RetryPolicy, is_retryable};
    use crate::error::RustAgentsError;

    let policy = RetryPolicy::default();
    assert!(policy.should_retry(0));
    assert!(!policy.should_retry(3));

    assert!(is_retryable(&RustAgentsError::Model("timeout".into())));
    assert!(!is_retryable(&RustAgentsError::Validation(
        "bad input".into()
    )));
}
