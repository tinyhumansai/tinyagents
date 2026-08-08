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

#[test]
fn identical_output_halts_at_threshold_and_changes_reset() {
    let tracker = SuccessfulRepeatTracker::new(3, 10);
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("call-1", true, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("call-2", true, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue,
        "output cannot halt before its tool batch is classified"
    );
    assert!(matches!(
        tracker.record_call_batch("call-3", true, false),
        SuccessfulRepeat::Halt(message) if message.contains("3 iterations")
    ));
    assert_eq!(
        tracker.record_output("different", false),
        SuccessfulRepeat::Continue
    );
}

#[test]
fn successful_call_batches_halt_but_failures_reset() {
    let tracker = SuccessfulRepeatTracker::new(4, 2);
    assert_eq!(
        tracker.record_call_batch("tool:args", true, false),
        SuccessfulRepeat::Continue
    );
    assert!(matches!(
        tracker.record_call_batch("tool:args", true, false),
        SuccessfulRepeat::Halt(message) if message.contains("2 times")
    ));
    assert_eq!(
        tracker.record_call_batch("tool:args", false, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("tool:args", true, false),
        SuccessfulRepeat::Continue
    );
}

#[test]
fn failed_call_batches_reset_output_repeats() {
    let tracker = SuccessfulRepeatTracker::new(2, 2);
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("first-call", true, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue,
        "a threshold crossing is pending until batch success is known"
    );
    assert_eq!(
        tracker.record_call_batch("failed-call", false, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue,
        "a failed prior batch must not count toward a successful output loop"
    );
}

#[test]
fn exempt_batches_reset_both_streaks() {
    let tracker = SuccessfulRepeatTracker::new(2, 2);
    assert_eq!(
        tracker.record_output("poll", false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("poll", true, true),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_output("poll", false),
        SuccessfulRepeat::Continue,
        "an exempt completed batch must clear its already-recorded output"
    );

    assert_eq!(
        tracker.record_call_batch("poll", true, false),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("poll", true, true),
        SuccessfulRepeat::Continue
    );
    assert_eq!(
        tracker.record_call_batch("poll", true, false),
        SuccessfulRepeat::Continue
    );
}

#[test]
fn zero_thresholds_are_fail_safe() {
    let tracker = SuccessfulRepeatTracker::new(0, 0);
    assert_eq!(
        tracker.record_output("same", false),
        SuccessfulRepeat::Continue
    );
    assert!(matches!(
        tracker.record_call_batch("same", true, false),
        SuccessfulRepeat::Halt(_)
    ));
}

// ---------------------------------------------------------------------------
// C5: the API a wave-2 `after_tool` driver actually needs
// ---------------------------------------------------------------------------

mod drivable {
    use serde_json::json;

    use crate::harness::no_progress::{
        DEFAULT_IDENTICAL_HALT_THRESHOLD, NoProgress, NoProgressTracker, ToolAttempt,
        fingerprint_arguments,
    };

    #[test]
    fn argument_fingerprints_ignore_object_key_order() {
        // The identical-repeat rung compares fingerprints, so a driver that
        // rendered arguments with `Value::to_string` would see two logically
        // identical calls as different (insertion order is preserved) and let
        // the loop run to the tool-call cap instead of tripping the ladder.
        let a = fingerprint_arguments(&json!({"path": "/tmp", "depth": 2, "glob": "*.rs"}));
        let b = fingerprint_arguments(&json!({"glob": "*.rs", "depth": 2, "path": "/tmp"}));
        assert_eq!(a, b);

        // Nested objects too.
        let n1 = fingerprint_arguments(&json!({"o": {"x": 1, "y": 2}}));
        let n2 = fingerprint_arguments(&json!({"o": {"y": 2, "x": 1}}));
        assert_eq!(n1, n2);
    }

    #[test]
    fn argument_fingerprints_distinguish_real_differences() {
        let base = fingerprint_arguments(&json!({"path": "/tmp"}));
        assert_ne!(base, fingerprint_arguments(&json!({"path": "/var"})));
        assert_ne!(base, fingerprint_arguments(&json!({"pathx": "/tmp"})));
        // Array order is semantic and must NOT be canonicalised away.
        assert_ne!(
            fingerprint_arguments(&json!({"a": [1, 2]})),
            fingerprint_arguments(&json!({"a": [2, 1]}))
        );
        // Type differences matter.
        assert_ne!(
            fingerprint_arguments(&json!({"a": 1})),
            fingerprint_arguments(&json!({"a": "1"}))
        );
    }

    #[test]
    fn fingerprints_are_short_and_stable_across_calls() {
        let value = json!({"path": "/tmp"});
        let first = fingerprint_arguments(&value);
        assert_eq!(first.len(), 16);
        assert_eq!(first, fingerprint_arguments(&value));
    }

    #[test]
    fn tool_attempt_constructors_cover_every_driver_case() {
        let fp = fingerprint_arguments(&json!({"q": "x"}));

        let ok = ToolAttempt::success("search", &fp);
        assert!(ok.error.is_none() && !ok.hard_reject && !ok.recoverable_miss);

        let bad = ToolAttempt::failure("search", &fp, "boom");
        assert_eq!(bad.error, Some("boom"));

        let blocked = ToolAttempt::failure("shell", &fp, "denied").hard_reject();
        assert!(blocked.hard_reject);

        let missing = ToolAttempt::failure("nope", &fp, "unknown tool").recoverable_miss();
        assert!(missing.recoverable_miss);
    }

    #[test]
    fn a_driver_can_run_the_whole_ladder_through_the_public_api_only() {
        // The end-to-end shape of the middleware wave 2 has to write: a shared
        // `&self` tracker, a fingerprint, a constructor, and verdict accessors —
        // no field literals, no hand-rolled hashing, no enum matching.
        let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let args = json!({"path": "/missing"});
        let fp = fingerprint_arguments(&args);

        let verdicts: Vec<NoProgress> = (1..=3)
            .map(|step| tracker.record(step, &ToolAttempt::failure("read", &fp, "ENOENT")))
            .collect();

        assert!(matches!(verdicts[0], NoProgress::Continue));
        assert!(verdicts[1].is_nudge(), "expected a nudge on the repeat");
        assert!(
            verdicts[2].is_halt(),
            "expected a halt once retries ran out"
        );

        // The accessors give a driver everything it needs without a match.
        assert!(verdicts[0].message().is_none());
        assert!(verdicts[1].message().unwrap().contains("no progress since"));
        assert!(verdicts[2].message().unwrap().contains("Stopping"));
        assert_eq!(
            verdicts.iter().map(NoProgress::as_str).collect::<Vec<_>>(),
            vec!["continue", "nudge", "halt"]
        );
    }

    #[test]
    fn record_takes_a_shared_reference_so_an_after_tool_hook_needs_no_refcell() {
        // `after_tool` hooks receive `&self`; the tracker must be usable through
        // a shared reference or every driver would need interior mutability of
        // its own.
        fn drive(tracker: &NoProgressTracker, fp: &str) -> NoProgress {
            tracker.record(1, &ToolAttempt::failure("t", fp, "boom"))
        }
        let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let fp = fingerprint_arguments(&json!({}));
        assert_eq!(drive(&tracker, &fp), NoProgress::Continue);
        assert!(drive(&tracker, &fp).is_nudge());
    }

    #[test]
    fn a_success_between_failures_clears_the_ladder() {
        // The driver contract says a success resets progress tracking; pinned
        // here because a wave-2 middleware relies on it to avoid halting a run
        // that is genuinely making progress with an occasional failure.
        let tracker = NoProgressTracker::new(DEFAULT_IDENTICAL_HALT_THRESHOLD);
        let fp = fingerprint_arguments(&json!({"a": 1}));

        assert_eq!(
            tracker.record(1, &ToolAttempt::failure("t", &fp, "boom")),
            NoProgress::Continue
        );
        assert_eq!(
            tracker.record(2, &ToolAttempt::success("t", &fp)),
            NoProgress::Continue
        );
        // The repeat counter restarted, so this is a first failure again.
        assert_eq!(
            tracker.record(3, &ToolAttempt::failure("t", &fp, "boom")),
            NoProgress::Continue
        );
    }
}
