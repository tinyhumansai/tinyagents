//! Tests for builtin time/date tools.

use super::time;
use super::*;
use chrono::{SecondsFormat, Utc};
use chrono_tz::Tz;
use serde_json::json;

use crate::harness::tool::{Tool, ToolCall, ToolRegistry};

fn call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall::new(id, name, arguments)
}

#[test]
fn time_tools_register_expected_names() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    register_time_tools(&mut registry);

    assert_eq!(
        registry.names(),
        vec!["current_time".to_string(), "resolve_time".to_string()]
    );
    assert!(registry.policies()["current_time"].side_effects.read_only);
    assert!(registry.policies()["resolve_time"].side_effects.read_only);
}

#[tokio::test]
async fn current_time_returns_utc_local_and_unix_seconds() {
    let tool = CurrentTimeTool::new();
    let result = tool
        .call(&(), call("c1", "current_time", json!({})))
        .await
        .unwrap();

    assert!(!result.is_error());
    let payload: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert!(payload["utc"].is_string());
    assert!(payload["local"].is_string());
    assert!(payload["local_timezone"].is_string());
    assert!(payload["unix_seconds"].is_number());
}

#[tokio::test]
async fn current_time_converts_requested_timezone() {
    let tool = CurrentTimeTool::new();
    let result = tool
        .call(
            &(),
            call("c1", "current_time", json!({ "timezone": "Asia/Kolkata" })),
        )
        .await
        .unwrap();

    let payload: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(payload["requested_timezone"]["name"], "Asia/Kolkata");
    assert!(payload["requested_timezone"]["time"].is_string());
}

#[tokio::test]
async fn current_time_unknown_timezone_reports_error_field() {
    let tool = CurrentTimeTool::new();
    let result = tool
        .call(
            &(),
            call("c1", "current_time", json!({ "timezone": "Not/AZone" })),
        )
        .await
        .unwrap();

    let payload: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert!(payload["requested_timezone_error"].is_string());
}

#[test]
fn relative_past_variants_are_negative_offsets() {
    for expr in [
        "24h ago",
        "last 24 hours",
        "past 24 hours",
        "-24h",
        "24 hours ago",
        "24h",
    ] {
        let duration = time::parse_relative_duration(expr).unwrap();
        assert_eq!(duration.num_seconds(), -86_400, "{expr}");
    }
    assert_eq!(
        time::parse_relative_duration("2 weeks")
            .unwrap()
            .num_seconds(),
        -1_209_600
    );
}

#[test]
fn relative_future_variants_are_positive_offsets() {
    assert_eq!(
        time::parse_relative_duration("in 10 minutes")
            .unwrap()
            .num_seconds(),
        600
    );
    assert_eq!(
        time::parse_relative_duration("30m from now")
            .unwrap()
            .num_seconds(),
        1_800
    );
    assert_eq!(
        time::parse_relative_duration("+2h").unwrap().num_seconds(),
        7_200
    );
    assert_eq!(
        time::parse_relative_duration("next 7d")
            .unwrap()
            .num_seconds(),
        604_800
    );
}

#[test]
fn resolve_expr_handles_exact_rfc3339_and_explicit_zone_dates() {
    let dt = time::resolve_expr("2026-06-09T19:12:00Z", time::ResolveZone::Local).unwrap();
    assert_eq!(dt.timestamp(), 1_781_032_320);

    let tz: Tz = "Asia/Kolkata".parse().unwrap();
    let dt = time::resolve_expr("2026-06-09", time::ResolveZone::Iana(tz)).unwrap();
    assert_eq!(
        dt.to_rfc3339_opts(SecondsFormat::Secs, true),
        "2026-06-08T18:30:00Z"
    );
}

#[test]
fn relative_resolution_tracks_now_with_expected_sign() {
    let before = Utc::now().timestamp();
    let past = time::resolve_expr("24h ago", time::ResolveZone::Local).unwrap();
    let future = time::resolve_expr("in 10 minutes", time::ResolveZone::Local).unwrap();
    let after = Utc::now().timestamp();

    assert!(past.timestamp() >= before - 86_400 - 2);
    assert!(past.timestamp() <= after - 86_400 + 2);
    assert!(future.timestamp() >= before + 600 - 2);
    assert!(future.timestamp() <= after + 600 + 2);
}

#[tokio::test]
async fn resolve_time_returns_all_formats_and_selected_value() {
    let tool = ResolveTimeTool::new();
    let result = tool
        .call(
            &(),
            call(
                "c1",
                "resolve_time",
                json!({
                    "expr": "2026-06-09T19:12:00Z",
                    "format": "slack_ts"
                }),
            ),
        )
        .await
        .unwrap();

    assert!(!result.is_error());
    let payload: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(payload["unix_s"], 1_781_032_320_i64);
    assert_eq!(payload["unix_ms"], 1_781_032_320_000_i64);
    assert_eq!(payload["slack_ts"], "1781032320.000000");
    assert_eq!(payload["value"], "1781032320.000000");
}

#[tokio::test]
async fn resolve_time_errors_for_missing_expr_and_bad_timezone() {
    let tool = ResolveTimeTool::new();

    let missing = tool
        .call(&(), call("c1", "resolve_time", json!({})))
        .await
        .unwrap();
    assert!(missing.is_error());
    assert!(missing.content.contains("`expr` is required"));

    let bad_zone = tool
        .call(
            &(),
            call(
                "c2",
                "resolve_time",
                json!({ "expr": "today", "timezone": "Not/AZone" }),
            ),
        )
        .await
        .unwrap();
    assert!(bad_zone.is_error());
    assert!(bad_zone.content.contains("unknown IANA timezone"));
}
