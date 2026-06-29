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
