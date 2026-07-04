//! Langfuse ingestion exporter for durable harness observations.
//!
//! The exporter targets Langfuse's `/api/public/ingestion` batch API. It can
//! send directly to a self-hosted Langfuse instance with public/secret keys, or
//! to the TinyHumans backend proxy with a bearer token.

use serde_json::{Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::harness::events::AgentEvent;
use crate::harness::ids::now_ms;
use crate::harness::observability::AgentObservation;
use crate::harness::usage::Usage;

mod types;

pub use types::{LangfuseAuth, LangfuseClient, LangfuseTraceConfig};

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
        let metadata = trace_metadata(&trace, observations);

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
                "metadata": metadata,
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
        self.send_batch(payload).await
    }

    /// Posts a pre-built ingestion `payload` to the configured endpoint,
    /// attaching the right auth header and translating transport/HTTP errors.
    ///
    /// This is the shared transport used by [`Self::send_observations`]. Other
    /// exporters that build their own `{ "batch": [...] }` payload — such as the
    /// graph observability exporter — reuse this method so authentication,
    /// endpoint normalization, and the Langfuse `207 Multi-Status` handling live
    /// in one place.
    pub async fn send_batch(&self, payload: Value) -> Result<Value> {
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
        // A `207 Multi-Status` reports per-item outcomes: some events may have
        // been rejected while the request itself "succeeded". Surface those
        // partial failures instead of swallowing them and reporting success.
        if status.as_u16() == 207
            && let Some(errors) = parsed.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            return Err(TinyAgentsError::Model(format!(
                "Langfuse ingestion partially failed ({} rejected): {}",
                errors.len(),
                json!(errors)
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

/// Builds the trace-level metadata, defaulting useful run-lineage coordinates
/// (root/first run ids and thread, mirroring what the graph exporter folds in)
/// so a harness trace is correlatable even when the caller passes no metadata.
/// Any caller-supplied [`LangfuseTraceConfig::metadata`] keys are merged on top
/// and win on collision. Returns [`Value::Null`] only when there is nothing to
/// attach, so [`clean_nulls`] drops the field entirely.
fn trace_metadata(trace: &LangfuseTraceConfig, observations: &[AgentObservation]) -> Value {
    let mut metadata = serde_json::Map::new();
    if let Some(first) = observations.first() {
        metadata.insert("root_run_id".to_string(), json!(first.root_run_id.as_str()));
        metadata.insert("run_id".to_string(), json!(first.run_id.as_str()));
        if let Some(parent) = &first.parent_run_id {
            metadata.insert("parent_run_id".to_string(), json!(parent.as_str()));
        }
    }
    if let Value::Object(extra) = &trace.metadata {
        for (k, v) in extra {
            metadata.insert(k.clone(), v.clone());
        }
    }
    if metadata.is_empty() {
        Value::Null
    } else {
        Value::Object(metadata)
    }
}

fn observation_event(trace_id: &str, obs: &AgentObservation) -> Value {
    let timestamp = iso_ms(obs.ts_ms);
    // Attach only run lineage, offset, and the event *kind* — not the full event
    // payload. The event's meaningful fields (input/output/usage/name) are
    // already lifted into the observation `body`; embedding the whole event here
    // duplicated every payload, roughly doubling batch bytes and growing
    // O(turns^2) as a run accumulates, which can trip the ~3.5MB batch cap.
    let metadata = json!({
        "run_id": obs.run_id.as_str(),
        "root_run_id": obs.root_run_id.as_str(),
        "parent_run_id": obs.parent_run_id.as_ref().map(|id| id.as_str()),
        "offset": obs.offset,
        "event_kind": obs.event.kind(),
    });
    match &obs.event {
        AgentEvent::ModelCompleted {
            call_id,
            started_at_ms,
            usage,
            input,
            output,
        } => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "generation-create",
            "body": clean_nulls(json!({
                "id": call_id.as_str(),
                "traceId": trace_id,
                "name": "model",
                // Use the loop-captured start time so the generation has a
                // real duration; fall back to the completion timestamp (a
                // zero-width point) for events journaled before the field
                // existed.
                "startTime": started_at_ms.map(iso_ms).unwrap_or_else(|| timestamp.clone()),
                "endTime": timestamp,
                "usage": usage.map(langfuse_usage),
                "input": input,
                "output": output,
                "metadata": metadata,
            })),
        }),
        AgentEvent::ToolCompleted {
            call_id,
            tool_name,
            started_at_ms,
            input,
            output,
        } => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            // A tool call is modelled as a span. `tool-create` is not a valid
            // Langfuse ingestion observation type — older/self-hosted Langfuse
            // rejects it, silently dropping every tool observation.
            "type": "span-create",
            "body": clean_nulls(json!({
                "id": call_id.as_str(),
                "traceId": trace_id,
                "name": tool_name,
                // Loop-captured start time when available (see the
                // generation branch above).
                "startTime": started_at_ms.map(iso_ms).unwrap_or_else(|| timestamp.clone()),
                "endTime": timestamp,
                "input": input,
                "output": output,
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

pub(crate) fn clean_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

pub(crate) fn iso_ms(ms: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let system_time = UNIX_EPOCH + Duration::from_millis(ms);
    let duration = system_time
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0));
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    format_unix_iso(secs, millis)
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
mod test;
