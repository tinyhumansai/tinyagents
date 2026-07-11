//! Feature/integration tests for the harness control-plane infrastructure
//! (`harness::cancel` + `harness::no_progress` + `harness::ids`).
//!
//! Covers cooperative cancellation (latching, clone-shared state, the
//! `cancelled()` future in `select!`), the no-progress escalation ladder
//! (nudge-then-halt on identical repeats, hard-reject fast path, varied-failure
//! backstop, reset on success), and the id newtypes / generators (uniqueness,
//! restart-safe run/checkpoint ids, `Display`/`as_str` surface).
//!
//! Deterministic and offline.

use std::time::Duration;

use tinyagents::harness::cancel::CancellationToken;
use tinyagents::harness::ids::{
    CallId, RunId, ThreadId, new_call_id, new_run_id, next_seq, now_ms,
};
use tinyagents::harness::no_progress::{
    DEFAULT_IDENTICAL_HALT_THRESHOLD, NoProgress, NoProgressTracker, ToolAttempt,
};

// ── Cancellation ────────────────────────────────────────────────────────────

#[test]
fn token_starts_live_and_latches_on_cancel() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
    token.cancel();
    assert!(token.is_cancelled());
    // Latching: cancelling again is idempotent, never un-cancels.
    token.cancel();
    assert!(token.is_cancelled());
}

#[test]
fn cancellation_is_visible_through_every_clone() {
    let token = CancellationToken::new();
    let clone = token.clone();
    clone.cancel();
    // A cancel through any clone is observed through the original.
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn cancelled_future_resolves_immediately_when_already_cancelled() {
    let token = CancellationToken::new();
    token.cancel();
    // Already cancelled → the future is ready without parking.
    token.cancelled().await;
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn cancelled_future_wakes_a_parked_waiter() {
    let token = CancellationToken::new();
    let waiter = token.clone();
    let handle = tokio::spawn(async move {
        waiter.cancelled().await;
        true
    });
    // Give the task a moment to park on the future, then cancel.
    tokio::task::yield_now().await;
    token.cancel();
    let woke = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("waiter should wake promptly")
        .expect("task joins");
    assert!(woke);
}

#[tokio::test]
async fn cancelled_is_selectable_without_firing_when_live() {
    let token = CancellationToken::new();
    // A live token never resolves `cancelled()`, so the other select arm wins.
    tokio::select! {
        _ = token.cancelled() => panic!("live token must not resolve"),
        _ = tokio::time::sleep(Duration::from_millis(10)) => {}
    }
    assert!(!token.is_cancelled());
}

// ── No-progress escalation ladder ───────────────────────────────────────────

fn failing(tool: &str, args: &str, err: &str) -> ToolAttempt<'static> {
    // Leak short static strings so the borrow outlives the call in these tests.
    ToolAttempt {
        tool: Box::leak(tool.to_string().into_boxed_str()),
        arg_fingerprint: Box::leak(args.to_string().into_boxed_str()),
        error: Some(Box::leak(err.to_string().into_boxed_str())),
        hard_reject: false,
        recoverable_miss: false,
    }
}

#[test]
fn identical_failures_nudge_then_halt() {
    let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let attempt = failing("search", "q=x", "not found");

    // First identical failure: still making progress room.
    assert_eq!(tracker.record(1, &attempt), NoProgress::Continue);
    // Second identical failure: a corrective nudge fires.
    assert!(matches!(tracker.record(2, &attempt), NoProgress::Nudge(_)));
    // Third identical failure: same-strategy retries exhausted → halt.
    assert!(matches!(tracker.record(3, &attempt), NoProgress::Halt(_)));
}

#[test]
fn success_resets_the_ladder() {
    let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let bad = failing("search", "q=x", "boom");
    assert_eq!(tracker.record(1, &bad), NoProgress::Continue);

    // A success clears every counter.
    let ok = ToolAttempt {
        tool: "search",
        arg_fingerprint: "q=x",
        error: None,
        hard_reject: false,
        recoverable_miss: false,
    };
    assert_eq!(tracker.record(2, &ok), NoProgress::Continue);

    // The next identical failure starts from scratch (Continue, not a nudge).
    assert_eq!(tracker.record(3, &bad), NoProgress::Continue);
}

#[test]
fn hard_rejection_halts_faster_than_ordinary_failures() {
    let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let blocked = ToolAttempt {
        tool: "shell",
        arg_fingerprint: "rm -rf /",
        error: Some("blocked by security policy"),
        hard_reject: true,
        recoverable_miss: false,
    };
    // First blocked call: not yet halted.
    assert!(!matches!(tracker.record(1, &blocked), NoProgress::Halt(_)));
    // Re-issued unchanged, a hard rejection halts on the second occurrence.
    assert!(matches!(tracker.record(2, &blocked), NoProgress::Halt(_)));
}

#[test]
fn varied_failures_hit_the_any_failure_backstop() {
    let tracker = NoProgressTracker::new(100); // huge identical cap: isolate the varied path
    let mut verdicts = Vec::new();
    for i in 0..6 {
        // Distinct args each time so the identical-repeat ladder never trips.
        let attempt = failing("tool", &format!("arg-{i}"), &format!("err-{i}"));
        verdicts.push(tracker.record(i, &attempt));
    }
    // A run of varied failures produces at least one nudge before halting.
    assert!(verdicts.iter().any(|v| matches!(v, NoProgress::Nudge(_))));
    assert!(matches!(verdicts.last(), Some(NoProgress::Halt(_))));
}

// ── Ids ─────────────────────────────────────────────────────────────────────

#[test]
fn id_newtypes_expose_display_and_as_str() {
    let run = RunId::new("run-42");
    assert_eq!(run.as_str(), "run-42");
    assert_eq!(run.to_string(), "run-42");
    // Distinct newtypes prevent mixing a thread id for a call id at the type
    // level while sharing the same cheap string surface.
    let thread: ThreadId = "t-1".into();
    let call: CallId = "c-1".into();
    assert_eq!(thread.as_str(), "t-1");
    assert_eq!(call.as_str(), "c-1");
}

#[test]
fn next_seq_is_strictly_increasing_within_the_process() {
    let a = next_seq();
    let b = next_seq();
    let c = next_seq();
    assert!(a < b && b < c);
}

#[test]
fn generated_run_and_call_ids_are_unique_and_prefixed() {
    let r1 = new_run_id();
    let r2 = new_run_id();
    assert_ne!(r1, r2);
    assert!(r1.as_str().starts_with("run-"));

    let c1 = new_call_id();
    let c2 = new_call_id();
    assert_ne!(c1, c2);
    assert!(c1.as_str().starts_with("call-"));
}

#[test]
fn now_ms_returns_a_recent_epoch_timestamp() {
    // Sanity floor: well after 2020-01-01 in epoch millis.
    assert!(now_ms() > 1_577_836_800_000);
}
