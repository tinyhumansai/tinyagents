//! Langfuse ingestion exporter for durable harness observations.
//!
//! The exporter builds a Langfuse ingestion batch (`trace-create` plus one
//! event per observation) and sends it either over OTLP to Langfuse v4's
//! `/api/public/otel/v1/traces` (`send_observations_otlp`, `send_batch_otlp`,
//! `send_otlp`), or to the legacy `/api/public/ingestion` batch API, which
//! Langfuse v4 keeps only for scores. It can send directly to a self-hosted
//! Langfuse instance with public/secret keys, or to the TinyHumans backend
//! proxy with a bearer token.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::ids::now_ms;
use crate::observability::AgentObservation;
use tinyinference_llm::usage::Usage;

mod types;

pub use types::{
    LangfuseAuth, LangfuseClient, LangfuseScore, LangfuseScoreValue, LangfuseTraceConfig,
};

/// Header that selects Langfuse's ingestion write path for an OTLP request.
pub const LANGFUSE_INGESTION_VERSION_HEADER: &str = "x-langfuse-ingestion-version";
/// The Langfuse v4 (observations-first) ingestion version.
pub const LANGFUSE_INGESTION_VERSION: &str = "4";

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
            headers: reqwest::header::HeaderMap::new(),
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

    /// Creates a proxy client with an injected HTTP transport and extra
    /// request headers. The supplied headers are applied to OTLP and batch
    /// requests; the bearer token remains owned by this client.
    pub fn proxy_with_transport(
        base_url: impl Into<String>,
        token: impl Into<String>,
        transport: reqwest::Client,
        headers: reqwest::header::HeaderMap,
    ) -> Result<Self> {
        let mut langfuse = Self::proxy(base_url, token)?;
        langfuse.client = transport;
        langfuse.headers = headers;
        Ok(langfuse)
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

    /// Returns the OTLP/HTTP trace endpoint beside the ingestion endpoint:
    /// `/api/public/otel/v1/traces` on a Langfuse origin, or
    /// `/telemetry/langfuse/otel/v1/traces` through the TinyHumans proxy.
    ///
    /// Langfuse's OTLP endpoint is the supported path for trace ingestion;
    /// the legacy `/api/public/ingestion` batch API is sunset on Langfuse
    /// Cloud on 2026-11-16 and unavailable on self-hosted v4 in its default
    /// `events_only` mode. Scores may still use [`Self::send_batch`].
    pub fn otlp_endpoint(&self) -> String {
        for (legacy, otlp) in [
            ("/api/public/ingestion", "/api/public/otel/v1/traces"),
            (
                "/telemetry/langfuse/ingestion",
                "/telemetry/langfuse/otel/v1/traces",
            ),
        ] {
            if let Some(origin) = self.endpoint.strip_suffix(legacy) {
                return format!("{origin}{otlp}");
            }
        }
        self.endpoint.clone()
    }

    /// Posts one OTLP/HTTP JSON trace request (`{"resourceSpans": [...]}`),
    /// such as one built by `trace_export::otlp::otlp_requests`, to
    /// [`Self::otlp_endpoint`] with this client's authentication.
    ///
    /// The request carries `x-langfuse-ingestion-version: 4`, which selects
    /// Langfuse v4's observations-first write path (the header does not make a
    /// span v4-shaped; the `trace_export` converters do that).
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be sent, Langfuse answers
    /// with a non-success status, or a 2xx answer reports rejected spans in
    /// its OTLP `partialSuccess`.
    pub async fn send_otlp(&self, body: Value) -> Result<Value> {
        let req = self.authorized(
            self.client
                .post(self.otlp_endpoint())
                .header(
                    LANGFUSE_INGESTION_VERSION_HEADER,
                    LANGFUSE_INGESTION_VERSION,
                )
                .json(&body),
        );
        let response = req
            .send()
            .await
            .map_err(|e| TinyAgentsError::Model(format!("Langfuse OTLP request failed: {e}")))?;
        let status = response.status();
        let text = response.text().await.map_err(|e| {
            TinyAgentsError::Model(format!("Langfuse OTLP response read failed: {e}"))
        })?;
        let parsed = serde_json::from_str(&text).unwrap_or_else(|_| json!({ "message": text }));
        if !status.is_success() {
            return Err(TinyAgentsError::Model(format!(
                "Langfuse OTLP ingestion returned {status}: {parsed}"
            )));
        }
        let partial = &parsed["partialSuccess"];
        // OTLP/JSON encodes int64 as a string; accept either spelling.
        let rejected = partial["rejectedSpans"]
            .as_u64()
            .or_else(|| partial["rejectedSpans"].as_str()?.parse().ok())
            .unwrap_or(0);
        if rejected > 0 {
            let message = partial["errorMessage"].as_str().unwrap_or_default();
            return Err(TinyAgentsError::Model(format!(
                "Langfuse OTLP ingestion rejected {rejected} span(s): {message}"
            )));
        }
        Ok(parsed)
    }

    /// Attaches this client's credentials to `request`.
    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let request = request.headers(self.headers.clone());
        match &self.auth {
            LangfuseAuth::Basic {
                public_key,
                secret_key,
            } => request.basic_auth(public_key, Some(secret_key)),
            LangfuseAuth::Bearer { token } => request.bearer_auth(token),
        }
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

        // The per-call generation is projected from `ModelCompleted`, which does
        // not carry the model name — only `ModelStarted` does. Pre-scan the batch
        // to correlate each call's model by `call_id` so the generation can stamp
        // `body["model"]`; without it Langfuse can't map pricing and every
        // generation's cost is $0.
        let call_models = collect_call_models(observations);
        let first_deltas = collect_first_deltas(observations);

        let mut batch = Vec::with_capacity(observations.len() + 2);
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

        // Emit one span per run so the batch surfaces a **run tree** rather than
        // a flat list, mirroring how LangChain's callback handler nests LLM
        // generations and tool spans under the chain/agent run that produced
        // them. Each run span is parented (cross-batch, via a deterministic id)
        // to its `parent_run_id`'s span, so a sub-agent run exported in its own
        // batch nests under the parent agent's span in the same trace.
        for run in run_spans(&trace_id, observations) {
            batch.push(run);
        }

        for obs in observations {
            // Run-lifecycle events are consumed into the run span (its start,
            // end, and error/status); re-emitting them as standalone events
            // would duplicate the run boundary the span already represents.
            if is_run_lifecycle(&obs.event) {
                continue;
            }
            batch.push(observation_event(
                &trace_id,
                obs,
                &call_models,
                &first_deltas,
            ));
        }

        Ok(json!({ "batch": batch }))
    }

    /// Builds a `score-create` ingestion batch for `score` without sending it.
    ///
    /// The score id defaults to a deterministic value derived from the trace,
    /// observation, and name so re-scoring the same target is idempotent
    /// (Langfuse upserts scores by id). Pair with a `trace_id` that matches an
    /// exported trace — recall the harness/graph exporters default their trace
    /// id to the run's `root_run_id` — to attach an evaluation to a run.
    pub fn build_score_batch(&self, score: LangfuseScore) -> Value {
        let timestamp = iso_ms(now_ms());
        let score_id = score.id.clone().unwrap_or_else(|| default_score_id(&score));
        let event_id = format!("{score_id}:score");
        json!({
            "batch": [json!({
                "id": event_id,
                "timestamp": timestamp,
                "type": "score-create",
                "body": clean_nulls(json!({
                    "id": score_id,
                    "traceId": score.trace_id,
                    "observationId": score.observation_id,
                    "name": score.name,
                    "value": score.value.to_value(),
                    "dataType": score.value.data_type(),
                    "comment": score.comment,
                })),
            })]
        })
    }

    /// Attaches an evaluation `score` to an already-exported trace (or one of
    /// its observations) by sending a `score-create` ingestion batch. Mirrors
    /// Langfuse's `createScore` — the way LangChain/LangGraph traces are graded
    /// after the fact (human ratings, LLM-as-judge, regression metrics).
    pub async fn create_score(&self, score: LangfuseScore) -> Result<Value> {
        let payload = self.build_score_batch(score);
        self.send_batch(payload).await
    }

    /// Sends observations to Langfuse v4 over OTLP: builds the same trace as
    /// [`Self::build_ingestion_batch`] and exports it through
    /// [`Self::send_batch_otlp`]. Prefer this over [`Self::send_observations`],
    /// whose legacy batch path Langfuse Cloud removes on 2026-11-16.
    ///
    /// # Errors
    ///
    /// As [`Self::build_ingestion_batch`] and [`Self::send_batch_otlp`].
    pub async fn send_observations_otlp(
        &self,
        trace: LangfuseTraceConfig,
        observations: &[AgentObservation],
        service: &str,
    ) -> Result<Vec<Value>> {
        let payload = self.build_ingestion_batch(trace, observations)?;
        self.send_batch_otlp(&payload, service).await
    }

    /// Exports a locally built ingestion `payload` (one `trace-create` plus its
    /// completed observations) to Langfuse v4 over OTLP, converted by
    /// [`ingestion_batch_otlp_requests`](crate::observability::trace_export::ingestion_otlp::ingestion_batch_otlp_requests)
    /// under resource `service.name = service` and scope `<service>.trace`.
    /// Requests are sent in order and the first failure stops the export.
    /// Non-trace events (scores) in the payload are skipped; send those with
    /// [`Self::send_batch`].
    ///
    /// # Errors
    ///
    /// Returns an error when the payload has no trace, or any request fails
    /// as described on [`Self::send_otlp`].
    pub async fn send_batch_otlp(&self, payload: &Value, service: &str) -> Result<Vec<Value>> {
        let requests =
            crate::observability::trace_export::ingestion_otlp::ingestion_batch_otlp_requests(
                payload,
                service,
                &format!("{service}.trace"),
            )?;
        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            responses.push(self.send_otlp(request).await?);
        }
        Ok(responses)
    }

    /// Sends observations as one legacy Langfuse ingestion batch.
    ///
    /// Langfuse v4 no longer accepts trace payloads on this path (Langfuse
    /// Cloud removes it on 2026-11-16); use [`Self::send_observations_otlp`].
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
        let req = self.authorized(self.client.post(&self.endpoint).json(&payload));

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

/// Whether an event marks a run boundary that the run span already represents.
///
/// These events are folded into the per-run span (its start time, end time, and
/// error/status) rather than emitted as standalone observations.
fn is_run_lifecycle(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::RunStarted { .. }
            | AgentEvent::RunCompleted { .. }
            | AgentEvent::RunFailed { .. }
    )
}

/// The stable, cross-batch Langfuse observation id for a run's span. Prefixing
/// with the (globally-unique) `trace_id` keeps it unique across turns/threads
/// while staying deterministic, so a child run exported in a *separate* batch
/// can reference its parent run's span by the same id and nest under it.
fn run_span_id(trace_id: &str, run_id: &str) -> String {
    format!("{trace_id}:run:{run_id}")
}

/// Accumulates a single run's timing/lineage while scanning the batch so the
/// exporter can emit one span per run.
struct RunAcc<'a> {
    parent_run_id: Option<&'a str>,
    root_run_id: &'a str,
    first_ts: u64,
    last_ts: u64,
    /// Terminal timestamp from a `RunCompleted`/`RunFailed`, when the run ended
    /// within this batch. `None` leaves the span open (still running).
    end_ts: Option<u64>,
    error: Option<&'a str>,
}

impl<'a> RunAcc<'a> {
    fn new(obs: &'a AgentObservation) -> Self {
        Self {
            parent_run_id: obs.parent_run_id.as_ref().map(|id| id.as_str()),
            root_run_id: obs.root_run_id.as_str(),
            first_ts: obs.ts_ms,
            last_ts: obs.ts_ms,
            end_ts: None,
            error: None,
        }
    }

    fn observe(&mut self, obs: &'a AgentObservation) {
        self.first_ts = self.first_ts.min(obs.ts_ms);
        self.last_ts = self.last_ts.max(obs.ts_ms);
        match &obs.event {
            AgentEvent::RunCompleted { .. } => self.end_ts = Some(obs.ts_ms),
            AgentEvent::RunFailed { error, .. } => {
                self.end_ts = Some(obs.ts_ms);
                self.error = Some(error.as_str());
            }
            _ => {}
        }
    }
}

/// Builds a `span-create` per distinct run in the batch, in first-seen order.
///
/// A run whose `run_id == root_run_id` is the top-level agent (`"agent"`); any
/// other run was spawned by a parent and is labelled `"sub-agent"`. The span is
/// parented to its `parent_run_id`'s run span (which may live in another batch)
/// so the trace renders the full recursion tree.
fn run_spans(trace_id: &str, observations: &[AgentObservation]) -> Vec<Value> {
    let mut order: Vec<&str> = Vec::new();
    let mut runs: BTreeMap<&str, RunAcc> = BTreeMap::new();
    for obs in observations {
        let rid = obs.run_id.as_str();
        runs.entry(rid)
            .and_modify(|acc| acc.observe(obs))
            .or_insert_with(|| {
                order.push(rid);
                let mut acc = RunAcc::new(obs);
                acc.observe(obs);
                acc
            });
    }

    order
        .into_iter()
        .map(|rid| {
            let acc = &runs[rid];
            let span_id = run_span_id(trace_id, rid);
            let is_root = rid == acc.root_run_id;
            let name = if is_root { "agent" } else { "sub-agent" };
            // Close the span at the terminal event when present, else at the
            // last observation seen (a real window for a completed export).
            let end_iso = iso_ms(acc.end_ts.unwrap_or(acc.last_ts));
            json!({
                "id": span_id,
                "timestamp": end_iso,
                "type": "span-create",
                "body": clean_nulls(json!({
                    "id": span_id,
                    "traceId": trace_id,
                    "parentObservationId": acc.parent_run_id.map(|p| run_span_id(trace_id, p)),
                    "name": name,
                    "startTime": iso_ms(acc.first_ts),
                    "endTime": end_iso,
                    "level": acc.error.map(|_| "ERROR"),
                    "statusMessage": acc.error,
                    "metadata": json!({
                        "run_id": rid,
                        "root_run_id": acc.root_run_id,
                        "parent_run_id": acc.parent_run_id,
                    }),
                })),
            })
        })
        .collect()
}

/// Collect the model each `ModelStarted` announced, keyed by its `call_id`.
///
/// The per-call `generation-create` is built from `ModelCompleted`, which does
/// not carry the model name; this correlation lets the generation stamp
/// `body["model"]` so Langfuse can price the call. Borrows from `observations`,
/// which outlive the batch build.
fn collect_call_models(observations: &[AgentObservation]) -> BTreeMap<&str, &str> {
    let mut models = BTreeMap::new();
    for obs in observations {
        if let AgentEvent::ModelStarted { call_id, model } = &obs.event {
            models.insert(call_id.as_str(), model.as_str());
        }
    }
    models
}

/// First streamed output for each model call. This is the observable TTFT;
/// non-streaming calls have no first-delta timestamp to report.
fn collect_first_deltas(observations: &[AgentObservation]) -> BTreeMap<&str, u64> {
    let mut first = BTreeMap::new();
    for obs in observations {
        if let AgentEvent::ModelDelta { call_id, delta, .. } = &obs.event
            && (!delta.text.is_empty() || !delta.reasoning.is_empty() || delta.tool_call.is_some())
        {
            first.entry(call_id.as_str()).or_insert(obs.ts_ms);
        }
    }
    first
}

fn observation_event(
    trace_id: &str,
    obs: &AgentObservation,
    call_models: &BTreeMap<&str, &str>,
    first_deltas: &BTreeMap<&str, u64>,
) -> Value {
    let timestamp = iso_ms(obs.ts_ms);
    // Every per-call observation nests under its run's span so the trace renders
    // as a tree (agent → generation/tool), matching LangChain's run hierarchy.
    let parent = run_span_id(trace_id, obs.run_id.as_str());
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
        } => {
            // Langfuse upserts observations by `id` across the ENTIRE project, so
            // the observation id must be globally unique. `call_id` is only
            // unique within a single run (e.g. `agent_turn-model-1`), and the
            // logical run id is reused by every interactive turn — so a bare
            // call_id collides across turns and threads, and each new turn's
            // generation silently overwrites the previous one, stealing it onto
            // the newest trace (leaving every earlier trace with no model usage/
            // cost/content). Namespace the id with the globally-unique, per-trace
            // `trace_id`: it never collides across turns yet stays stable for
            // idempotent re-ingestion of the same run. The raw `call_id` rides in
            // metadata so in-run correlation (model.started ↔ this generation) is
            // preserved.
            let metadata = with_call_id(metadata, call_id.as_str());
            json!({
                "id": obs.event_id.as_str(),
                "timestamp": timestamp,
                "type": "generation-create",
                "body": clean_nulls(json!({
                    "id": scoped_observation_id(trace_id, call_id.as_str()),
                    "traceId": trace_id,
                    "parentObservationId": parent,
                    "name": "model",
                    // The model the matching `ModelStarted` announced for this
                    // call. Langfuse maps its pricing table off this field, so
                    // without it every generation's cost is $0. `clean_nulls`
                    // drops the key when the model can't be correlated.
                    "model": call_models.get(call_id.as_str()).copied(),
                    // Use the loop-captured start time so the generation has a
                    // real duration; fall back to the completion timestamp (a
                    // zero-width point) for events journaled before the field
                    // existed.
                    "startTime": started_at_ms.map(iso_ms).unwrap_or_else(|| timestamp.clone()),
                    "endTime": timestamp,
                    "completionStartTime": first_deltas.get(call_id.as_str()).map(|ms| iso_ms(*ms)),
                    "usage": usage.map(langfuse_usage),
                    "usageDetails": usage.map(langfuse_usage_details),
                    "costDetails": usage.and_then(langfuse_cost_details),
                    "input": input,
                    "output": output,
                    "metadata": metadata,
                })),
            })
        }
        AgentEvent::ToolCompleted {
            parent_call_id,
            call_id,
            tool_name,
            started_at_ms,
            input,
            output,
            duration_ms,
            input_bytes,
            output_bytes,
            error,
            // The tool's host-only result metadata (B2) is not exported: the
            // observation's `metadata` is the exporter's own correlation
            // record, and a tool payload of arbitrary size/shape belongs in a
            // deliberate mapping, not merged in by default.
            metadata: _,
        } => {
            // Prefer the loop-captured start + real duration for the end time;
            // fall back to the journal timestamp. A failed call is marked ERROR
            // with the failure reason, and result size rides metadata so the
            // span is informative even in payload-free mode.
            let end_time = match (started_at_ms, duration_ms) {
                (Some(start), Some(dur)) => iso_ms(start.saturating_add(*dur)),
                _ => timestamp.clone(),
            };
            // Same project-wide id-collision hazard as ModelCompleted above:
            // builtin tools take deterministic per-run call_ids (e.g.
            // `agent_turn-tool-1`) that repeat every turn, so a bare call_id
            // overwrites earlier tool spans onto the newest trace. Namespace with
            // the per-trace id; keep the raw call_id in metadata for correlation.
            let mut tool_metadata = with_call_id(metadata.clone(), call_id.as_str());
            if let Some(map) = tool_metadata.as_object_mut() {
                if let Some(bytes) = input_bytes {
                    map.insert("input_bytes".into(), json!(bytes));
                }
                if let Some(bytes) = output_bytes {
                    map.insert("output_bytes".into(), json!(bytes));
                }
            }
            json!({
                "id": obs.event_id.as_str(),
                "timestamp": timestamp,
                // A tool call is modelled as a span. `tool-create` is not a
                // valid Langfuse ingestion observation type — older/self-hosted
                // Langfuse rejects it, silently dropping every tool observation.
                "type": "span-create",
                "body": clean_nulls(json!({
                    "id": scoped_observation_id(trace_id, call_id.as_str()),
                    "traceId": trace_id,
                    // A nested call (C9) nests under its immediate parent's
                    // span; everything else hangs off the run span.
                    "parentObservationId": parent_call_id
                        .as_ref()
                        .map(|p| scoped_observation_id(trace_id, p.as_str()))
                        .unwrap_or_else(|| parent.clone()),
                    "name": tool_name,
                    "startTime": started_at_ms.map(iso_ms).unwrap_or_else(|| timestamp.clone()),
                    "endTime": end_time,
                    "input": input,
                    "output": output,
                    "level": error.as_ref().map(|_| "ERROR"),
                    "statusMessage": error,
                    "metadata": tool_metadata,
                })),
            })
        }
        // Run-lifecycle events (RunStarted/RunCompleted/RunFailed) never reach
        // here — they are consumed into the run span by `is_run_lifecycle`.
        _ => json!({
            "id": obs.event_id.as_str(),
            "timestamp": timestamp,
            "type": "event-create",
            "body": clean_nulls(json!({
                "id": obs.event_id.as_str(),
                "traceId": trace_id,
                "parentObservationId": parent,
                "name": obs.event.kind(),
                "startTime": timestamp,
                "metadata": metadata,
            })),
        }),
    }
}

/// Derives a stable score id from its target and name so re-scoring the same
/// (trace, observation, metric) upserts the existing score rather than
/// duplicating it. A trace-level score omits the observation segment.
fn default_score_id(score: &LangfuseScore) -> String {
    match &score.observation_id {
        Some(obs) => format!("{}:{}:score:{}", score.trace_id, obs, score.name),
        None => format!("{}:score:{}", score.trace_id, score.name),
    }
}

/// Build a globally-unique-yet-stable Langfuse observation id for a call-scoped
/// observation (model generation / tool span). Langfuse keys observations by
/// `id` across the whole project and upserts on collision, so we prefix the
/// (only run-unique) `call_id` with the globally-unique `trace_id`. The result
/// is deterministic for a given (trace, call), keeping re-ingestion idempotent
/// while never colliding across turns or threads.
fn scoped_observation_id(trace_id: &str, call_id: &str) -> String {
    format!("{trace_id}:{call_id}")
}

/// Fold the raw `call_id` into an observation's metadata so in-run correlation
/// (e.g. `model.started` ↔ its `model` generation) survives even though the
/// public observation id is now trace-namespaced. Returns the metadata
/// unchanged when it is not a JSON object.
fn with_call_id(mut metadata: Value, call_id: &str) -> Value {
    if let Some(map) = metadata.as_object_mut() {
        map.insert("call_id".into(), json!(call_id));
    }
    metadata
}

fn langfuse_usage(usage: Usage) -> Value {
    json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens,
        "unit": "TOKENS",
    })
}

fn langfuse_usage_details(usage: Usage) -> Value {
    json!({
        "input": usage.input_tokens.saturating_sub(usage.cache_read_tokens),
        "output": usage.output_tokens,
        "total": usage.total_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "cache_creation_input_tokens": usage.cache_creation_tokens,
        "reasoning_output_tokens": usage.reasoning_tokens,
    })
}

fn langfuse_cost_details(usage: Usage) -> Option<Value> {
    usage
        .charged_amount
        .map(|amount| json!({ "total": amount.micros as f64 / 1_000_000.0 }))
}

/// Drops every `null`-valued key from a top-level JSON object, in place at
/// one level (not recursive). Used before every ingestion body so Langfuse
/// never sees an explicit `null` for a field this exporter chose not to
/// populate, distinct from one it deliberately set to null.
pub fn clean_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

/// Renders a Unix epoch millisecond timestamp as the `YYYY-MM-DDTHH:MM:SS.sssZ`
/// form Langfuse's ingestion API expects.
///
/// Delegates to `chrono` (M-9): `chrono` is already a non-optional workspace
/// dependency of this crate (`tools/time.rs` uses it under the `tools`
/// feature), so the hand-rolled Howard Hinnant civil-date conversion this
/// module carried was duplicating logic the dependency graph already pays
/// for, unconditionally, elsewhere.
pub fn iso_ms(ms: u64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(i64::try_from(ms).unwrap_or(i64::MAX))
        .unwrap_or_default()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
