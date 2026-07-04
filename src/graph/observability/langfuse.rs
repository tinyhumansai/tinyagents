//! Langfuse ingestion exporter for durable graph observations.
//!
//! Where [`crate::harness::observability::LangfuseClient`] exports an *agent*
//! run's observations (model generations, tool calls), this exporter turns a
//! *graph* run's durable [`GraphObservation`] stream into a Langfuse trace: each
//! superstep and node handler becomes a timed span, node/subgraph failures are
//! promoted to `ERROR` level, and per-node **health telemetry** rides along on
//! the trace metadata.
//!
//! # Unified traces across the graph and its agents
//!
//! Graph nodes frequently delegate to agents and tools (see
//! [`crate::graph::SubAgentNode`]). Those child agent runs share the graph run's
//! `root_run_id`, and both exporters default their Langfuse `traceId` to that
//! root run id. Sending a graph run's observations with this exporter **and**
//! the child agent runs' observations with the harness
//! [`LangfuseClient`](crate::harness::observability::LangfuseClient) therefore
//! lands every graph step, node, model generation, and tool call under one
//! Langfuse trace — full end-to-end telemetry including tool health.
//!
//! The exporter is pull-based and best-effort by construction: build the batch
//! from a completed (or in-progress) observation slice read back from a
//! [`GraphEventJournal`](crate::graph::GraphEventJournal), then send it. It does
//! not sit on the hot path of graph execution.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::graph::observability::{GraphHealthSummary, GraphObservation};
use crate::graph::stream::GraphEvent;
use crate::harness::ids::NodeId;
use crate::harness::observability::{LangfuseClient, LangfuseTraceConfig, clean_nulls, iso_ms};

/// A FIFO queue of pending span starts, each `(observation index, ts_ms)`.
/// Duplicate span identities (a node re-run in a later step, say) queue in
/// arrival order and pair with their terminals FIFO.
type StartQueue = VecDeque<(usize, u64)>;

/// A host-supplied hook that contributes extra metadata to every span the
/// exporter emits. Called with the span's coordinate observation; returning
/// `None` (or an empty map) contributes nothing. Host keys are merged **over**
/// the exporter's built-in coordinate keys, so a host can also override them.
pub type SpanMetadataFn = dyn Fn(&GraphObservation) -> Option<Map<String, Value>> + Send + Sync;

/// Async Langfuse exporter for durable graph observations.
///
/// Wraps a shared [`LangfuseClient`] for transport (auth, endpoint
/// normalization, `207 Multi-Status` handling) and adds graph-aware payload
/// construction on top.
#[derive(Clone)]
pub struct GraphLangfuseExporter {
    client: LangfuseClient,
    /// Optional host metadata injector, merged into every span's metadata.
    span_metadata_fn: Option<Arc<SpanMetadataFn>>,
}

impl std::fmt::Debug for GraphLangfuseExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphLangfuseExporter")
            .field("client", &self.client)
            .field(
                "span_metadata_fn",
                &self.span_metadata_fn.as_ref().map(|_| "Fn(..)"),
            )
            .finish()
    }
}

impl GraphLangfuseExporter {
    /// Wraps an existing [`LangfuseClient`] as a graph exporter.
    pub fn new(client: LangfuseClient) -> Self {
        Self {
            client,
            span_metadata_fn: None,
        }
    }

    /// Builds an exporter from the same environment variables as
    /// [`LangfuseClient::from_env`].
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(LangfuseClient::from_env()?))
    }

    /// Installs a host metadata injector: `f` is called once per emitted span
    /// with the span's coordinate observation, and any returned keys are merged
    /// into that span's metadata (host keys win on collision). Default: none.
    pub fn with_span_metadata_fn(
        mut self,
        f: impl Fn(&GraphObservation) -> Option<Map<String, Value>> + Send + Sync + 'static,
    ) -> Self {
        self.span_metadata_fn = Some(Arc::new(f));
        self
    }

    /// Returns the underlying transport client.
    pub fn client(&self) -> &LangfuseClient {
        &self.client
    }

    /// Returns the normalized ingestion endpoint.
    pub fn endpoint(&self) -> &str {
        self.client.endpoint()
    }

    /// Builds a Langfuse ingestion payload for `observations` without sending it.
    ///
    /// The batch always begins with a `trace-create` event, followed by a
    /// timed `span-create` for every completed (or still-running) superstep,
    /// node, and subgraph, then an `event-create` for every remaining
    /// observation (routes, checkpoints, interrupts, custom writes, run
    /// lifecycle). Node health telemetry is attached to the trace metadata.
    pub fn build_ingestion_batch(
        &self,
        trace: LangfuseTraceConfig,
        observations: &[GraphObservation],
    ) -> Result<Value> {
        if observations.is_empty() {
            return Err(TinyAgentsError::Validation(
                "at least one observation is required".to_string(),
            ));
        }
        let trace_id = resolve_trace_id(&trace, observations);
        let first = &observations[0];
        let trace_ts = iso_ms(first.ts_ms);
        let health = GraphHealthSummary::from_observations(observations);

        let mut batch = Vec::with_capacity(observations.len() + 1);
        batch.push(trace_create(&trace_id, &trace_ts, &trace, first, &health));

        let mut consumed = vec![false; observations.len()];
        push_span_events(
            &trace_id,
            observations,
            &mut consumed,
            &mut batch,
            self.span_metadata_fn.as_deref(),
        );
        push_point_events(&trace_id, observations, &consumed, &mut batch);

        Ok(json!({ "batch": batch }))
    }

    /// Sends `observations` as one Langfuse ingestion batch.
    pub async fn send_observations(
        &self,
        trace: LangfuseTraceConfig,
        observations: &[GraphObservation],
    ) -> Result<Value> {
        let payload = self.build_ingestion_batch(trace, observations)?;
        self.client.send_batch(payload).await
    }
}

/// Resolves the Langfuse trace id: the configured id when set, else the first
/// observation's root run id so it aligns with the harness agent exporter.
fn resolve_trace_id(trace: &LangfuseTraceConfig, observations: &[GraphObservation]) -> String {
    if let Some(id) = &trace.trace_id
        && !id.trim().is_empty()
    {
        return id.clone();
    }
    observations[0].root_run_id.as_str().to_string()
}

/// Builds the `trace-create` batch event, defaulting the trace name to the
/// graph id and the session to the run's thread, and folding node-health
/// telemetry plus graph coordinates into the trace metadata.
fn trace_create(
    trace_id: &str,
    trace_ts: &str,
    trace: &LangfuseTraceConfig,
    first: &GraphObservation,
    health: &GraphHealthSummary,
) -> Value {
    let name = trace
        .name
        .clone()
        .unwrap_or_else(|| first.graph_id.as_str().to_string());
    let session_id = trace
        .session_id
        .clone()
        .or_else(|| first.thread_id.as_ref().map(|t| t.as_str().to_string()));
    let mut metadata = json!({
        "graph_id": first.graph_id.as_str(),
        "root_run_id": first.root_run_id.as_str(),
        "health": health,
    });
    if !first.namespace.is_empty()
        && let Value::Object(map) = &mut metadata
    {
        map.insert("namespace".to_string(), json!(first.namespace));
    }
    if let (Value::Object(dst), Value::Object(extra)) = (&mut metadata, &trace.metadata) {
        for (k, v) in extra {
            dst.insert(k.clone(), v.clone());
        }
    }

    json!({
        "id": format!("{trace_id}:trace"),
        "timestamp": trace_ts,
        "type": "trace-create",
        "body": clean_nulls(json!({
            "id": trace_id,
            "timestamp": trace_ts,
            "name": name,
            "userId": trace.user_id,
            "sessionId": session_id,
            "environment": trace.environment,
            "release": trace.release,
            "version": trace.version,
            "tags": if trace.tags.is_empty() { Value::Null } else { json!(trace.tags) },
            "metadata": metadata,
        })),
    })
}

/// Emits a timed `span-create` for every step, node, and subgraph by pairing
/// each start observation with its terminal one (FIFO for duplicate keys).
/// Both the start and terminal indices are marked `consumed` so they are not
/// re-emitted as point events. Unpaired starts become open spans (start only).
fn push_span_events(
    trace_id: &str,
    observations: &[GraphObservation],
    consumed: &mut [bool],
    batch: &mut Vec<Value>,
    injector: Option<&SpanMetadataFn>,
) {
    // Pending starts keyed by their span identity, each holding (index, ts).
    let mut step_starts: HashMap<usize, StartQueue> = HashMap::new();
    let mut node_starts: HashMap<(NodeId, usize), StartQueue> = HashMap::new();
    let mut subgraph_starts: HashMap<(NodeId, Vec<String>), StartQueue> = HashMap::new();

    for (idx, obs) in observations.iter().enumerate() {
        match &obs.event {
            GraphEvent::StepStarted { step, .. } => {
                step_starts
                    .entry(*step)
                    .or_default()
                    .push_back((idx, obs.ts_ms));
                consumed[idx] = true;
            }
            GraphEvent::StepCompleted { step } => {
                if let Some((start_idx, start_ts)) = pop(&mut step_starts, step) {
                    consumed[idx] = true;
                    batch.push(step_span(
                        trace_id,
                        *step,
                        start_ts,
                        Some(obs.ts_ms),
                        obs,
                        &observations[start_idx],
                        injector,
                    ));
                }
            }
            GraphEvent::NodeStarted { node, step } => {
                node_starts
                    .entry((node.clone(), *step))
                    .or_default()
                    .push_back((idx, obs.ts_ms));
                consumed[idx] = true;
            }
            GraphEvent::NodeCompleted { node, step } => {
                if let Some((_, start_ts)) = pop(&mut node_starts, &(node.clone(), *step)) {
                    consumed[idx] = true;
                    batch.push(node_span(
                        trace_id,
                        node,
                        *step,
                        start_ts,
                        Some(obs.ts_ms),
                        None,
                        obs,
                        injector,
                    ));
                }
            }
            GraphEvent::NodeFailed { node, step, error } => {
                if let Some((_, start_ts)) = pop(&mut node_starts, &(node.clone(), *step)) {
                    consumed[idx] = true;
                    batch.push(node_span(
                        trace_id,
                        node,
                        *step,
                        start_ts,
                        Some(obs.ts_ms),
                        Some(error.as_str()),
                        obs,
                        injector,
                    ));
                }
            }
            GraphEvent::SubgraphStarted { node, namespace } => {
                subgraph_starts
                    .entry((node.clone(), namespace.clone()))
                    .or_default()
                    .push_back((idx, obs.ts_ms));
                consumed[idx] = true;
            }
            GraphEvent::SubgraphCompleted { node, namespace } => {
                if let Some((_, start_ts)) =
                    pop(&mut subgraph_starts, &(node.clone(), namespace.clone()))
                {
                    consumed[idx] = true;
                    batch.push(subgraph_span(
                        trace_id,
                        node,
                        namespace,
                        start_ts,
                        Some(obs.ts_ms),
                        obs,
                        injector,
                    ));
                }
            }
            _ => {}
        }
    }

    // Any still-open starts become spans with a start time but no end time.
    for (step, mut queue) in step_starts {
        while let Some((start_idx, start_ts)) = queue.pop_front() {
            let obs = &observations[start_idx];
            batch.push(step_span(
                trace_id, step, start_ts, None, obs, obs, injector,
            ));
        }
    }
    for ((node, step), mut queue) in node_starts {
        while let Some((start_idx, start_ts)) = queue.pop_front() {
            batch.push(node_span(
                trace_id,
                &node,
                step,
                start_ts,
                None,
                None,
                &observations[start_idx],
                injector,
            ));
        }
    }
    for ((node, namespace), mut queue) in subgraph_starts {
        while let Some((start_idx, start_ts)) = queue.pop_front() {
            batch.push(subgraph_span(
                trace_id,
                &node,
                &namespace,
                start_ts,
                None,
                &observations[start_idx],
                injector,
            ));
        }
    }
}

/// Emits an `event-create` for every observation not already represented by a
/// span, mapping `run.failed` to `ERROR` level with the rendered error.
fn push_point_events(
    trace_id: &str,
    observations: &[GraphObservation],
    consumed: &[bool],
    batch: &mut Vec<Value>,
) {
    for (idx, obs) in observations.iter().enumerate() {
        if consumed[idx] {
            continue;
        }
        // The trace itself represents the run start; skip the duplicate event.
        if matches!(obs.event, GraphEvent::RunStarted { .. }) {
            continue;
        }
        let ts = iso_ms(obs.ts_ms);
        let (level, status) = match &obs.event {
            GraphEvent::RunFailed { error, .. } => (Some("ERROR"), Some(error.clone())),
            _ => (None, None),
        };
        batch.push(json!({
            "id": obs.event_id.as_str(),
            "timestamp": ts,
            "type": "event-create",
            "body": clean_nulls(json!({
                "id": obs.event_id.as_str(),
                "traceId": trace_id,
                "name": obs.event.kind(),
                "startTime": ts,
                "level": level,
                "statusMessage": status,
                "metadata": span_metadata(obs, None, None, None),
            })),
        }));
    }
}

/// Builds a `span-create` for a superstep, parented directly to the trace.
#[allow(clippy::too_many_arguments)]
fn step_span(
    trace_id: &str,
    step: usize,
    start_ts: u64,
    end_ts: Option<u64>,
    terminal: &GraphObservation,
    start: &GraphObservation,
    injector: Option<&SpanMetadataFn>,
) -> Value {
    let metadata = span_metadata(start, None, Some(step), injector);
    span_event(
        trace_id,
        &format!("{trace_id}:step:{step}"),
        None,
        &format!("step {step}"),
        start_ts,
        end_ts,
        None,
        terminal,
        metadata,
    )
}

/// Builds a `span-create` for a node handler, parented to its superstep span.
///
/// Node spans carry the Langfuse **Agent Graph view** keys `langgraph_node`
/// (the node id) and `langgraph_step` (the superstep index) in metadata, so
/// Langfuse can lay the trace out as a graph.
#[allow(clippy::too_many_arguments)]
fn node_span(
    trace_id: &str,
    node: &NodeId,
    step: usize,
    start_ts: u64,
    end_ts: Option<u64>,
    error: Option<&str>,
    terminal: &GraphObservation,
    injector: Option<&SpanMetadataFn>,
) -> Value {
    let metadata = span_metadata(terminal, Some(node.as_str()), Some(step), injector);
    span_event(
        trace_id,
        &format!("{trace_id}:node:{}:{step}", node.as_str()),
        Some(format!("{trace_id}:step:{step}")),
        node.as_str(),
        start_ts,
        end_ts,
        error,
        terminal,
        metadata,
    )
}

/// Builds a `span-create` for an embedded subgraph, parented to the trace.
fn subgraph_span(
    trace_id: &str,
    node: &NodeId,
    namespace: &[String],
    start_ts: u64,
    end_ts: Option<u64>,
    terminal: &GraphObservation,
    injector: Option<&SpanMetadataFn>,
) -> Value {
    let metadata = span_metadata(terminal, None, Some(terminal.step), injector);
    span_event(
        trace_id,
        &format!("{trace_id}:subgraph:{}", namespace.join("/")),
        None,
        &format!("subgraph {}", node.as_str()),
        start_ts,
        end_ts,
        None,
        terminal,
        metadata,
    )
}

/// Shared `span-create` builder. `error` promotes the span to `ERROR` level.
/// The batch item id comes from the terminal observation so it is unique; the
/// caller supplies the fully-built span `metadata`.
#[allow(clippy::too_many_arguments)]
fn span_event(
    trace_id: &str,
    span_id: &str,
    parent: Option<String>,
    name: &str,
    start_ts: u64,
    end_ts: Option<u64>,
    error: Option<&str>,
    terminal: &GraphObservation,
    metadata: Value,
) -> Value {
    let start_iso = iso_ms(start_ts);
    let end_iso = end_ts.map(iso_ms);
    let (level, status) = match error {
        Some(err) => (Some("ERROR"), Some(err.to_string())),
        None => (None, None),
    };
    json!({
        "id": terminal.event_id.as_str(),
        "timestamp": end_iso.clone().unwrap_or_else(|| start_iso.clone()),
        "type": "span-create",
        "body": clean_nulls(json!({
            "id": span_id,
            "traceId": trace_id,
            "parentObservationId": parent,
            "name": name,
            "startTime": start_iso,
            "endTime": end_iso,
            "level": level,
            "statusMessage": status,
            "metadata": metadata,
        })),
    })
}

/// Extracts the correlation coordinates every span/event carries in metadata,
/// stamping the Langfuse Agent-Graph-view keys (`langgraph_node`,
/// `langgraph_step`) when the caller supplies them and merging any
/// host-injected keys last (host keys win on collision).
fn span_metadata(
    obs: &GraphObservation,
    langgraph_node: Option<&str>,
    langgraph_step: Option<usize>,
    injector: Option<&SpanMetadataFn>,
) -> Value {
    let mut metadata = json!({
        "run_id": obs.run_id.as_str(),
        "root_run_id": obs.root_run_id.as_str(),
        "parent_run_id": obs.parent_run_id.as_ref().map(|id| id.as_str()),
        "graph_id": obs.graph_id.as_str(),
        "checkpoint_id": obs.checkpoint_id.as_ref().map(|id| id.as_str()),
        "namespace": if obs.namespace.is_empty() { Value::Null } else { json!(obs.namespace) },
        "step": obs.step,
        "offset": obs.offset,
        "event": obs.event,
    });
    if let Value::Object(map) = &mut metadata {
        if let Some(node) = langgraph_node {
            map.insert("langgraph_node".to_string(), json!(node));
        }
        if let Some(step) = langgraph_step {
            map.insert("langgraph_step".to_string(), json!(step));
        }
        if let Some(extra) = injector.and_then(|f| f(obs)) {
            for (k, v) in extra {
                map.insert(k, v);
            }
        }
    }
    metadata
}

/// Pops the FIFO-oldest pending start for `key`, cleaning up empty queues.
fn pop<K: std::hash::Hash + Eq>(
    starts: &mut HashMap<K, StartQueue>,
    key: &K,
) -> Option<(usize, u64)> {
    let queue = starts.get_mut(key)?;
    let popped = queue.pop_front();
    if queue.is_empty() {
        starts.remove(key);
    }
    popped
}

#[cfg(test)]
mod tests;
