//! Unit tests for the harness id newtypes and lifecycle enums.
//!
//! Covers construction and `as_str`/`Display` round-tripping, `From<&str>` /
//! `From<String>` conversions, the transparent string serde representation of
//! the id newtypes, and the snake_case serde encoding of [`ExecutionStatus`]
//! and [`HarnessPhase`].

use super::*;

#[test]
fn constructs_and_reads_ids() {
    let run = RunId::new("run-1");
    assert_eq!(run.as_str(), "run-1");
    assert_eq!(run.to_string(), "run-1");
}

#[test]
fn converts_from_string_and_str() {
    let from_str: ThreadId = "t-1".into();
    let from_string: ThreadId = String::from("t-1").into();
    assert_eq!(from_str, from_string);
}

#[test]
fn ids_are_distinct_types_but_serialize_as_strings() {
    let call = CallId::new("c-1");
    let json = serde_json::to_string(&call).unwrap();
    assert_eq!(json, "\"c-1\"");
    let back: CallId = serde_json::from_str(&json).unwrap();
    assert_eq!(back, call);
}

#[test]
fn generated_run_and_checkpoint_ids_carry_restart_nonce() {
    // Regression for the cross-restart collision: ids must not be the bare
    // `run-0`/`ckpt-0` monotonic form (which repeats in every fresh process),
    // but `<prefix>-<nonce>-<seq>` so a restarted, resumed thread never re-mints
    // an id it already used.
    let ckpt = new_checkpoint_id();
    let parts: Vec<&str> = ckpt.as_str().split('-').collect();
    assert_eq!(parts.len(), 3, "checkpoint id has prefix-nonce-seq shape");
    assert_eq!(parts[0], "ckpt");
    assert_eq!(parts[1], process_nonce().to_string(), "middle is the nonce");

    let run = new_run_id();
    assert!(run.as_str().starts_with("run-"));
    assert!(
        run.as_str().contains(&format!("-{}-", process_nonce())),
        "run id embeds the process nonce"
    );

    // Ids are unique within a process.
    let a = new_checkpoint_id();
    let b = new_checkpoint_id();
    assert_ne!(a, b, "consecutive checkpoint ids differ");
}

#[test]
fn status_and_phase_use_snake_case() {
    assert_eq!(
        serde_json::to_string(&ExecutionStatus::Interrupted).unwrap(),
        "\"interrupted\""
    );
    assert_eq!(
        serde_json::to_string(&HarnessPhase::BuildingRequest).unwrap(),
        "\"building_request\""
    );
}

#[test]
fn now_ms_returns_a_recent_unix_millis_timestamp() {
    // The shared clock helper must return a plausible wall-clock timestamp: at
    // or after a fixed 2020-01-01 epoch anchor and monotonic non-decreasing
    // across two reads. (2020-01-01T00:00:00Z in ms.)
    const JAN_2020_MS: u64 = 1_577_836_800_000;
    let first = now_ms();
    assert!(first >= JAN_2020_MS, "timestamp {first} predates 2020");
    let second = now_ms();
    assert!(second >= first, "now_ms went backwards: {second} < {first}");
}
