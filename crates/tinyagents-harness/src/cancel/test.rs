//! Unit tests for [`CancellationToken`].

use super::CancellationToken;

#[test]
fn new_token_is_not_cancelled() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
}

#[test]
fn default_token_is_not_cancelled() {
    let token = CancellationToken::default();
    assert!(!token.is_cancelled());
}

#[test]
fn cancel_latches_and_is_idempotent() {
    let token = CancellationToken::new();
    token.cancel();
    assert!(token.is_cancelled());
    // Cancelling again is harmless.
    token.cancel();
    assert!(token.is_cancelled());
}

#[test]
fn clones_share_state() {
    let token = CancellationToken::new();
    let clone = token.clone();
    assert!(!clone.is_cancelled());
    clone.cancel();
    // Cancellation through one clone is visible through the original.
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn cancelled_resolves_immediately_when_already_cancelled() {
    let token = CancellationToken::new();
    token.cancel();
    // Must not hang.
    token.cancelled().await;
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn cancelled_resolves_after_concurrent_cancel() {
    let token = CancellationToken::new();
    let waiter = token.clone();

    let handle = tokio::spawn(async move {
        waiter.cancelled().await;
        true
    });

    // Give the waiter a chance to park, then cancel from this task.
    tokio::task::yield_now().await;
    token.cancel();

    let resolved = handle.await.expect("waiter task should not panic");
    assert!(resolved);
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn cancelled_is_select_safe() {
    let token = CancellationToken::new();
    // Racing `cancelled()` against a ready branch must not cancel anything or
    // hang; dropping the un-awaited future simply deregisters the waiter.
    tokio::select! {
        _ = token.cancelled() => panic!("token was never cancelled"),
        _ = async {} => {}
    }
    assert!(!token.is_cancelled());
}
