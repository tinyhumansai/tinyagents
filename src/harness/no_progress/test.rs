//! Unit tests for the no-progress escalation ladder.

use super::*;

fn fail<'a>(tool: &'a str, fp: &'a str, err: &'a str) -> ToolAttempt<'a> {
    ToolAttempt {
        tool,
        arg_fingerprint: fp,
        error: Some(err),
        hard_reject: false,
        recoverable_miss: false,
    }
}

fn ok<'a>(tool: &'a str, fp: &'a str) -> ToolAttempt<'a> {
    ToolAttempt {
        tool,
        arg_fingerprint: fp,
        error: None,
        hard_reject: false,
        recoverable_miss: false,
    }
}

#[test]
fn identical_failure_nudges_then_halts() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    // First identical failure: not enough repetition yet.
    assert_eq!(
        t.record(1, &fail("read_file", "a", "not found")),
        NoProgress::Continue
    );
    // Second: nudge the model to change strategy before the retry cap.
    match t.record(2, &fail("read_file", "a", "not found")) {
        NoProgress::Nudge(msg) => {
            assert!(msg.contains("no progress since step 2"));
            assert!(msg.contains("read_file"));
            assert!(msg.contains("Change strategy"));
        }
        other => panic!("expected a nudge on the second identical failure, got {other:?}"),
    }
    // Third: same-strategy retries exhausted → halt.
    match t.record(3, &fail("read_file", "a", "not found")) {
        NoProgress::Halt(msg) => assert!(msg.contains("retried 3 times")),
        other => panic!("expected a halt on the third identical failure, got {other:?}"),
    }
}

#[test]
fn a_success_resets_the_ladder() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let _ = t.record(1, &fail("t", "a", "boom"));
    let _ = t.record(2, &fail("t", "a", "boom")); // nudge
    assert_eq!(t.record(3, &ok("t", "a")), NoProgress::Continue);
    // After the reset, two more identical failures only re-nudge — no halt.
    assert_eq!(t.record(4, &fail("t", "a", "boom")), NoProgress::Continue);
    assert!(matches!(
        t.record(5, &fail("t", "a", "boom")),
        NoProgress::Nudge(_)
    ));
}

#[test]
fn changed_arguments_clear_the_identical_streak() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let _ = t.record(1, &fail("read_file", "a", "not found"));
    // The model heeded the nudge and changed args: a new signature, so the
    // identical-repeat counter starts over and never reaches the halt.
    assert_eq!(
        t.record(2, &fail("read_file", "b", "not found")),
        NoProgress::Continue
    );
    assert_eq!(
        t.record(3, &fail("list_dir", "c", "denied")),
        NoProgress::Continue
    );
}

#[test]
fn a_hard_rejection_halts_on_the_second_identical_repeat() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    let mut a = fail("send_email", "a", "[policy-blocked] denied");
    a.hard_reject = true;
    assert_eq!(t.record(1, &a), NoProgress::Continue);
    let mut b = fail("send_email", "a", "[policy-blocked] denied");
    b.hard_reject = true;
    assert!(matches!(t.record(2, &b), NoProgress::Halt(msg) if msg.contains("security policy")));
}

#[test]
fn varied_failures_nudge_then_hit_the_backstop() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    // Distinct signatures each time: the identical counter never climbs, but
    // the any-failure streak does.
    for i in 1..=3 {
        let (fp, err) = (format!("fp{i}"), format!("err{i}"));
        assert_eq!(t.record(i, &fail("t", &fp, &err)), NoProgress::Continue);
    }
    // Fourth consecutive varied failure → nudge to step back.
    assert!(matches!(
        t.record(4, &fail("t", "fp4", "err4")),
        NoProgress::Nudge(_)
    ));
    assert_eq!(t.record(5, &fail("t", "fp5", "err5")), NoProgress::Continue);
    // Sixth → the no-progress backstop halts.
    assert!(matches!(
        t.record(6, &fail("t", "fp6", "err6")),
        NoProgress::Halt(msg) if msg.contains("in a row failed")
    ));
}

#[test]
fn unknown_tool_recovery_does_not_feed_the_backstop() {
    let t = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
    // Six correctable misses with *distinct* signatures must not trip the
    // any-failure backstop (they don't count toward `consecutive`).
    for i in 0..6 {
        let (fp, err) = (format!("fp{i}"), format!("unknown tool foo{i}"));
        let mut a = fail("__unknown_tool__", &fp, &err);
        a.recoverable_miss = true;
        assert_eq!(t.record(i, &a), NoProgress::Continue);
    }
}

#[test]
fn identical_halt_threshold_is_clamped_above_the_nudge() {
    // A caller asking for a halt threshold of 1 or 2 still gets a nudge
    // before the halt (the nudge lands at 2, so the halt must be >= 3).
    let t = NoProgressTracker::new(1);
    assert_eq!(t.record(1, &fail("t", "a", "boom")), NoProgress::Continue);
    assert!(matches!(
        t.record(2, &fail("t", "a", "boom")),
        NoProgress::Nudge(_)
    ));
    assert!(matches!(
        t.record(3, &fail("t", "a", "boom")),
        NoProgress::Halt(_)
    ));
}
