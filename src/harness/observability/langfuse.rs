//! Langfuse ingestion exporter for durable harness observations.
//!
//! The exporter targets Langfuse's `/api/public/ingestion` batch API. It can
//! send directly to a self-hosted Langfuse instance with public/secret keys, or
//! to the TinyHumans backend proxy with a bearer token.

use serde::Serialize;
use serde_json::{Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::harness::events::AgentEvent;
use crate::harness::observability::AgentObservation;
use crate::harness::usage::Usage;

/// Authentication mode for [`LangfuseClient`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LangfuseAuth {
    /// Send `Authorization: Basic base64(public_key:secret_key)`.
    Basic {
        /// Langfuse project public key.
        public_key: String,
        /// Langfuse project secret key.
        secret_key: String,
    },
    /// Send `Authorization: Bearer <token>`.
    ///
    /// Use this when targeting the TinyHumans backend proxy at
    /// `/telemetry/langfuse/ingestion`; the backend injects Langfuse Basic Auth.
    Bearer {
        /// Backend access token.
        token: String,
    },
}

/// Configuration for a Langfuse trace export.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LangfuseTraceConfig {
    /// Stable Langfuse trace id. Defaults to the first observation's root run id.
    pub trace_id: Option<String>,
    /// Human-readable trace name.
    pub name: Option<String>,
    /// End-user id to filter by in Langfuse.
    pub user_id: Option<String>,
    /// Session/thread id to group related traces.
    pub session_id: Option<String>,
    /// Langfuse environment name.
    pub environment: Option<String>,
    /// Release identifier.
    pub release: Option<String>,
    /// Version identifier.
    pub version: Option<String>,
    /// Tags attached to the trace.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Extra trace metadata.
    #[serde(default)]
    pub metadata: Value,
}

/// Async Langfuse ingestion client.
#[derive(Clone, Debug)]
pub struct LangfuseClient {
    endpoint: String,
    auth: LangfuseAuth,
    client: reqwest::Client,
}

impl LangfuseClient {
    /// Creates a client from a base URL and auth mode.
    ///
    /// `base_url` may be a Langfuse origin such as `https://langfuse.example`
    /// or a TinyHumans backend origin. Basic Auth appends Langfuse's
    /// `/api/public/ingestion` path; Bearer Auth appends the backend proxy path
    /// `/telemetry/langfuse/ingestion`.
    pub fn new(base_url: impl Into<String>, auth: LangfuseAuth) -> Result<Self> {
        let endpoint = match &auth {
            LangfuseAuth::Basic { .. } => normalize_langfuse_endpoint(base_url.into())?,
            LangfuseAuth::Bearer { .. } => normalize_proxy_endpoint(base_url.into())?,
        };
        Ok(Self {
            endpoint,
            auth,
            client: reqwest::Client::new(),
        })
    }

    /// Creates a direct-to-Langfuse client using Basic Auth.
    pub fn direct(
        base_url: impl Into<String>,
        public_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Result<Self> {
        Self::new(
            base_url,
            LangfuseAuth::Basic {
                public_key: public_key.into(),
                secret_key: secret_key.into(),
            },
        )
    }

    /// Creates a client for the TinyHumans backend proxy.
    pub fn proxy(base_url: impl Into<String>, token: impl Into<String>) -> Result<Self> {
        Self::new(
            base_url,
            LangfuseAuth::Bearer {
                token: token.into(),
            },
        )
    }

    /// Reads configuration from environment variables.
    ///
    /// Direct mode: `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`,
    /// `LANGFUSE_SECRET_KEY`.
    ///
    /// Proxy mode: `TINYHUMANS_LANGFUSE_PROXY_URL`, `TINYHUMANS_AUTH_TOKEN`.
    /// Proxy mode wins when `TINYHUMANS_LANGFUSE_PROXY_URL` is set.
    pub fn from_env() -> Result<Self> {
        if let Ok(url) = std::env::var("TINYHUMANS_LANGFUSE_PROXY_URL")
            && !url.trim().is_empty()
        {
            let token = std::env::var("TINYHUMANS_AUTH_TOKEN").map_err(|_| {
                TinyAgentsError::Validation(
                    "TINYHUMANS_AUTH_TOKEN is required for Langfuse proxy mode".to_string(),
                )
            })?;
            return Self::proxy(url, token);
        }

        let base_url = std::env::var("LANGFUSE_BASE_URL").map_err(|_| {
            TinyAgentsError::Validation("LANGFUSE_BASE_URL is required".to_string())
        })?;
        let public_key = std::env::var("LANGFUSE_PUBLIC_KEY").map_err(|_| {
            TinyAgentsError::Validation("LANGFUSE_PUBLIC_KEY is required".to_string())
        })?;
        let secret_key = std::env::var("LANGFUSE_SECRET_KEY").map_err(|_| {
            TinyAgentsError::Validation("LANGFUSE_SECRET_KEY is required".to_string())
        })?;
        Self::direct(base_url, public_key, secret_key)
    }

    /// Returns the normalized ingestion endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Builds a Langfuse ingestion payload without sending it.
    pub fn build_ingestion_batch(
        &self,
        trace: LangfuseTraceConfig,
        observations: &[AgentObservation],
    ) -> Result<Value> {
        let trace_id = resolve_trace_id(&trace, observations)?;
        let timestamp = observations
            .first()
            .map(|obs| iso_ms(obs.ts_ms))
            .unwrap_or_else(|| iso_ms(now_ms()));

        let mut batch = Vec::with_capacity(observations.len() + 1);
        batch.push(json!({
            "id": format!("{}:trace", trace_id),
            "timestamp": timestamp,
            "type": "trace-create",
            "body": clean_nulls(json!({
                "id": trace_id,
                "timestamp": timestamp,
                "name": trace.name,
                "userId": trace.user_id,
                "sessionId": trace.session_id,
                "environment": trace.environment,
                "release": trace.release,
                "version": trace.version,
                "tags": if trace.tags.is_empty() { Value::Null } else { json!(trace.tags) },
                "metadata": if trace.metadata.is_null() { Value::Null } else { trace.metadata },
            })),
        }));

        for obs in observations {
            batch.push(observation_event(&trace_id, obs));
        }

        Ok(json!({ "batch": batch }))
    }

    /// Sends observations as one Langfuse ingestion batch.
    pub async fn send_observations(
        &self,
        trace: LangfuseTraceConfig,
        observations: &[AgentObservation],
    ) -> Result<Value> {
        let payload = self.build_ingestion_batch(trace, observations)?;
        let mut req = self.client.post(&self.endpoint).json(&payload);
        req = match &self.auth {
            LangfuseAuth::Basic {
                public_key,
                secret_key,
            } => req.basic_auth(public_key, Some(secret_key)),
            LangfuseAuth::Bearer { token } => req.bearer_auth(token),
        };

        let response = req
            .send()
            .await
            .map_err(|e| TinyAgentsError::Model(format!("Langfuse request failed: {e}")))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| TinyAgentsError::Model(format!("Langfuse response read failed: {e}")))?;
        let parsed = serde_json::from_str(&body).unwrap_or_else(|_| json!({ "message": body }));
        if !status.is_success() && status.as_u16() != 207 {
            return Err(TinyAgentsError::Model(format!(
                "Langfuse ingestion returned {status}: {parsed}"
            )));
        }
        Ok(parsed)
    }
}

fn normalize_langfuse_endpoint(raw: String) -> Result<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "Langfuse URL must not be empty".to_string(),
        ));
    }
    if trimmed.ends_with("/api/public/ingestion")
        || trimmed.ends_with("/telemetry/langfuse/ingestion")
    {
        return Ok(trimmed.to_string());
    }
    Ok(format!("{trimmed}/api/public/ingestion"))
}

fn normalize_proxy_endpoint(raw: String) -> Result<String> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "Langfuse proxy URL must not be empty".to_string(),
        ));
    }
    if trimmed.ends_with("/api/public/ingestion")
        || trimmed.ends_with("/telemetry/langfuse/ingestion")
    {
        return Ok(trimmed.to_string());
    }
    Ok(format!("{trimmed}/telemetry/langfuse/ingestion"))
}

fn resolve_trace_id(
    trace: &LangfuseTraceConfig,
    observations: &[AgentObservation],
) -> Result<String> {
    if let Some(id) = &trace.trace_id
        && !id.trim().is_empty()
    {
        return Ok(id.clone());
    }
    observations
        .first()
        .map(|obs| obs.root_run_id.as_str().to_string())
        .ok_or_else(|| {
            TinyAgentsError::Validation("at least one observation is required".to_string())
        })
}

fn observation_event(trace_id: &str, obs: &AgentObservation) -> Value {
    let timestamp = iso_ms(obs.ts_ms);
    let metadata = json!({
        "run_id": obs.run_id.as_str(),
        "root_run_id": obs.root_run_id.as_str(),
        "parent_run_id": obs.parent_run_id.as_ref().map(|id| id.as_str()),
        "offset": obs.offset,
        "event": obs.event,
    });
    match &obs.event {
        AgentEvent::ModelCompleted { call_id, usage } => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "generation-create",
            "body": clean_nulls(json!({
                "id": call_id.as_str(),
                "traceId": trace_id,
                "name": "model",
                "startTime": timestamp,
                "endTime": timestamp,
                "usage": usage.map(langfuse_usage),
                "metadata": metadata,
            })),
        }),
        AgentEvent::ToolCompleted { call_id, tool_name } => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "tool-create",
            "body": clean_nulls(json!({
                "id": call_id.as_str(),
                "traceId": trace_id,
                "name": tool_name,
                "startTime": timestamp,
                "endTime": timestamp,
                "metadata": metadata,
            })),
        }),
        AgentEvent::RunFailed { error, .. } => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "event-create",
            "body": clean_nulls(json!({
                "id": obs.event_id.as_str(),
                "traceId": trace_id,
                "name": obs.event.kind(),
                "startTime": timestamp,
                "level": "ERROR",
                "statusMessage": error,
                "metadata": metadata,
            })),
        }),
        _ => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "event-create",
            "body": clean_nulls(json!({
                "id": obs.event_id.as_str(),
                "traceId": trace_id,
                "name": obs.event.kind(),
                "startTime": timestamp,
                "metadata": metadata,
            })),
        }),
    }
}

fn langfuse_usage(usage: Usage) -> Value {
    json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens,
        "unit": "TOKENS",
    })
}

fn clean_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

fn iso_ms(ms: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let system_time = UNIX_EPOCH + Duration::from_millis(ms);
    let duration = system_time
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    format_unix_iso(secs, millis)
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn format_unix_iso(secs: u64, millis: u32) -> String {
    // Howard Hinnant civil-date conversion for Unix days, dependency-free.
    let days = (secs / 86_400) as i64;
    let day_secs = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = day_secs / 3_600;
    let minute = (day_secs % 3_600) / 60;
    let second = day_secs % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::events::AgentEvent;
    use crate::harness::ids::{CallId, EventId, RunId};

    fn obs(offset: u64, event: AgentEvent) -> AgentObservation {
        AgentObservation {
            event_id: EventId::new(format!("evt-{offset}")),
            run_id: RunId::new("run-1"),
            parent_run_id: None,
            root_run_id: RunId::new("root-1"),
            offset,
            ts_ms: 1_704_067_200_000 + offset,
            event,
        }
    }

    #[test]
    fn normalizes_langfuse_endpoints() {
        let client = LangfuseClient::proxy("https://api.example.test", "token").unwrap();
        assert_eq!(
            client.endpoint(),
            "https://api.example.test/telemetry/langfuse/ingestion"
        );
        let client = LangfuseClient::proxy(
            "https://api.example.test/telemetry/langfuse/ingestion",
            "token",
        )
        .unwrap();
        assert_eq!(
            client.endpoint(),
            "https://api.example.test/telemetry/langfuse/ingestion"
        );
    }

    #[test]
    fn builds_trace_and_generation_batch() {
        let client =
            LangfuseClient::proxy("https://backend.test/telemetry/langfuse/ingestion", "t")
                .unwrap();
        let batch = client
            .build_ingestion_batch(
                LangfuseTraceConfig {
                    user_id: Some("user-1".to_string()),
                    session_id: Some("thread-1".to_string()),
                    ..Default::default()
                },
                &[
                    obs(
                        0,
                        AgentEvent::RunStarted {
                            run_id: RunId::new("run-1"),
                            thread_id: None,
                        },
                    ),
                    obs(
                        1,
                        AgentEvent::ModelCompleted {
                            call_id: CallId::new("model-call"),
                            usage: Some(Usage {
                                input_tokens: 3,
                                output_tokens: 4,
                                total_tokens: 7,
                                ..Default::default()
                            }),
                        },
                    ),
                ],
            )
            .unwrap();

        let events = batch["batch"].as_array().unwrap();
        assert_eq!(events[0]["type"], "trace-create");
        assert_eq!(events[0]["body"]["id"], "root-1");
        assert_eq!(events[0]["body"]["userId"], "user-1");
        assert_eq!(events[2]["type"], "generation-create");
        assert_eq!(events[2]["body"]["id"], "model-call");
        assert_eq!(events[2]["body"]["usage"]["input"], 3);
    }
}
