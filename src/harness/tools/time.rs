//! Builtin time/date tools.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, SecondsFormat, Utc};
use chrono_tz::Tz;
use serde_json::json;

use crate::Result;
use crate::harness::tool::{Tool, ToolCall, ToolPolicy, ToolRegistry, ToolResult, ToolSchema};

const CURRENT_TIME_NAME: &str = "current_time";
const RESOLVE_TIME_NAME: &str = "resolve_time";

/// Tool that returns the current time in UTC and local time, optionally
/// converted to an IANA timezone.
pub struct CurrentTimeTool;

impl CurrentTimeTool {
    /// Creates a current-time tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CurrentTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for CurrentTimeTool {
    fn name(&self) -> &str {
        CURRENT_TIME_NAME
    }

    fn description(&self) -> &str {
        "Get the current date and time in UTC and the machine's local timezone. \
         Optionally convert to a specific IANA timezone such as \
         'America/Los_Angeles' or 'Asia/Kolkata'. Use before scheduling tasks or \
         when a user refers to relative times like 'in 10 minutes', 'tomorrow', \
         or 'tonight'."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            CURRENT_TIME_NAME,
            <Self as Tool<State>>::description(self),
            json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone name, for example 'Europe/London'."
                    }
                }
            }),
        )
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        let payload = current_time_payload(&call.arguments);
        Ok(ToolResult::text(
            call.id,
            CURRENT_TIME_NAME,
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string()),
        ))
    }
}

fn current_time_payload(args: &serde_json::Value) -> serde_json::Value {
    let now_utc = Utc::now();
    let now_local = Local::now();

    let mut payload = json!({
        "utc": now_utc.to_rfc3339_opts(SecondsFormat::Secs, true),
        "local": now_local.to_rfc3339_opts(SecondsFormat::Secs, true),
        "local_timezone": now_local.format("%Z").to_string(),
        "unix_seconds": now_utc.timestamp(),
        "weekday": now_local.format("%A").to_string(),
    });

    if let Some(tz_name) = args.get("timezone").and_then(|value| value.as_str()) {
        let trimmed = tz_name.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<Tz>() {
                Ok(tz) => {
                    let converted = now_utc.with_timezone(&tz);
                    payload["requested_timezone"] = json!({
                        "name": trimmed,
                        "time": converted.to_rfc3339_opts(SecondsFormat::Secs, true),
                        "weekday": converted.format("%A").to_string(),
                    });
                }
                Err(_) => {
                    payload["requested_timezone_error"] = json!(format!(
                        "Unknown IANA timezone '{trimmed}' - use names like 'America/Los_Angeles'."
                    ));
                }
            }
        }
    }

    payload
}

/// Tool that resolves relative or absolute time expressions to exact
/// timestamps.
pub struct ResolveTimeTool;

impl ResolveTimeTool {
    /// Creates a resolve-time tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ResolveTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for ResolveTimeTool {
    fn name(&self) -> &str {
        RESOLVE_TIME_NAME
    }

    fn description(&self) -> &str {
        "Resolve a relative or absolute time expression into an exact timestamp. \
         Use this to produce date/time arguments for other tools instead of \
         hand-computing Unix seconds. Accepted expressions include 'now', \
         '24h ago', '7d', '2 weeks ago', 'in 10 minutes', '30m from now', \
         'today', 'yesterday', 'tomorrow', RFC-3339 timestamps, bare dates, and \
         'YYYY-MM-DD HH:MM:SS'."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            RESOLVE_TIME_NAME,
            <Self as Tool<State>>::description(self),
            json!({
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Time expression to resolve."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["unix_s", "unix_ms", "slack_ts", "rfc3339"],
                        "description": "Representation to place in the top-level value field. Defaults to unix_s."
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone used to interpret offset-less inputs."
                    }
                },
                "required": ["expr"]
            }),
        )
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        let expr = match call.arguments.get("expr").and_then(|value| value.as_str()) {
            Some(expr) => expr,
            None => {
                return Ok(ToolResult::error(
                    call.id,
                    RESOLVE_TIME_NAME,
                    "resolve_time: `expr` is required",
                ));
            }
        };

        let zone = match call
            .arguments
            .get("timezone")
            .and_then(|value| value.as_str())
        {
            Some(tz_name) if !tz_name.trim().is_empty() => match tz_name.trim().parse::<Tz>() {
                Ok(tz) => ResolveZone::Iana(tz),
                Err(_) => {
                    return Ok(ToolResult::error(
                        call.id,
                        RESOLVE_TIME_NAME,
                        format!(
                            "resolve_time: unknown IANA timezone '{}' - use names like 'America/Los_Angeles'.",
                            tz_name.trim()
                        ),
                    ));
                }
            },
            _ => ResolveZone::Local,
        };

        let dt = match resolve_expr(expr, zone) {
            Ok(dt) => dt,
            Err(error) => {
                return Ok(ToolResult::error(
                    call.id,
                    RESOLVE_TIME_NAME,
                    format!("resolve_time: {error}"),
                ));
            }
        };

        let payload = resolve_time_payload(expr, &call.arguments, dt);
        Ok(ToolResult::text(
            call.id,
            RESOLVE_TIME_NAME,
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string()),
        ))
    }
}

fn resolve_time_payload(
    expr: &str,
    args: &serde_json::Value,
    dt: DateTime<Utc>,
) -> serde_json::Value {
    let unix_s = dt.timestamp();
    let unix_ms = dt.timestamp_millis();
    let slack_ts = format!("{unix_s}.000000");
    let rfc3339 = dt.to_rfc3339_opts(SecondsFormat::Secs, true);

    let format = args
        .get("format")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|format| !format.is_empty())
        .unwrap_or("unix_s");
    let value = match format {
        "unix_ms" => unix_ms.to_string(),
        "slack_ts" => slack_ts.clone(),
        "rfc3339" => rfc3339.clone(),
        _ => unix_s.to_string(),
    };

    json!({
        "interpreted": expr,
        "value": value,
        "unix_s": unix_s,
        "unix_ms": unix_ms,
        "slack_ts": slack_ts,
        "rfc3339": rfc3339,
    })
}

pub(crate) fn parse_relative_duration(raw: &str) -> Option<Duration> {
    let mut text = raw.trim().to_ascii_lowercase();
    let mut future = false;

    if let Some(rest) = text.strip_suffix(" ago") {
        text = rest.trim().to_string();
    } else if let Some(rest) = text.strip_suffix(" from now") {
        future = true;
        text = rest.trim().to_string();
    }

    for (prefix, is_future) in [
        ("in ", true),
        ("next ", true),
        ("last ", false),
        ("past ", false),
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            future = is_future;
            text = rest.trim().to_string();
            break;
        }
    }

    if let Some(rest) = text.strip_prefix('+') {
        future = true;
        text = rest.trim().to_string();
    } else if let Some(rest) = text.strip_prefix('-') {
        text = rest.trim().to_string();
    }

    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let split_at = text.find(|ch: char| !ch.is_ascii_digit())?;
    if split_at == 0 {
        return None;
    }

    let (num_str, unit_str) = text.split_at(split_at);
    let n: i64 = num_str.trim().parse().ok()?;
    let seconds_per = match unit_str.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3_600,
        "d" | "day" | "days" => 86_400,
        "w" | "wk" | "wks" | "week" | "weeks" => 604_800,
        _ => return None,
    };

    let magnitude = Duration::seconds(n.saturating_mul(seconds_per));
    Some(if future { magnitude } else { -magnitude })
}

pub(crate) fn resolve_expr(
    expr: &str,
    zone: ResolveZone,
) -> std::result::Result<DateTime<Utc>, String> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Err("`expr` is required".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();

    if lower == "now" {
        return Ok(Utc::now());
    }

    if let Some(duration) = parse_relative_duration(trimmed) {
        return Ok(Utc::now() + duration);
    }

    if lower == "today" || lower == "yesterday" || lower == "tomorrow" {
        let offset_days = match lower.as_str() {
            "yesterday" => -1,
            "tomorrow" => 1,
            _ => 0,
        };
        return zone.civil_midnight_to_utc(zone.now_civil_date() + Duration::days(offset_days));
    }

    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(dt.with_timezone(&Utc));
    }

    for format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return zone.naive_to_utc(naive);
        }
    }

    if let Ok(date) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return zone.civil_midnight_to_utc(date);
    }

    Err(format!("could not parse time expression {trimmed:?}"))
}

#[derive(Clone, Copy)]
pub(crate) enum ResolveZone {
    Local,
    Iana(Tz),
}

impl ResolveZone {
    fn now_civil_date(&self) -> NaiveDate {
        match self {
            ResolveZone::Local => Local::now().date_naive(),
            ResolveZone::Iana(tz) => Utc::now().with_timezone(tz).date_naive(),
        }
    }

    fn civil_midnight_to_utc(&self, date: NaiveDate) -> std::result::Result<DateTime<Utc>, String> {
        let naive = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| "invalid civil midnight".to_string())?;
        self.naive_to_utc(naive)
    }

    fn naive_to_utc(&self, naive: NaiveDateTime) -> std::result::Result<DateTime<Utc>, String> {
        use chrono::TimeZone;
        match self {
            ResolveZone::Local => Local
                .from_local_datetime(&naive)
                .single()
                .map(|dt| dt.with_timezone(&Utc))
                .ok_or_else(|| format!("ambiguous or invalid local time {naive}")),
            ResolveZone::Iana(tz) => tz
                .from_local_datetime(&naive)
                .single()
                .map(|dt| dt.with_timezone(&Utc))
                .ok_or_else(|| format!("ambiguous or invalid time {naive} in {tz:?}")),
        }
    }
}

/// Returns the builtin time tool set.
pub fn time_tools<State: Send + Sync + 'static>() -> Vec<Arc<dyn Tool<State>>> {
    vec![
        Arc::new(CurrentTimeTool::new()),
        Arc::new(ResolveTimeTool::new()),
    ]
}

/// Registers the builtin time tool set into an existing registry.
pub fn register_time_tools<State: Send + Sync + 'static>(registry: &mut ToolRegistry<State>) {
    for tool in time_tools() {
        registry.register(tool);
    }
}
