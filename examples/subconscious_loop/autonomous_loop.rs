//! A fully offline, testable version of the autonomous closed-loop harness.
//!
//! The example maps the Python/LangGraph reference architecture onto
//! TinyAgents' typed graph runtime:
//!
//! - channel ingestion and front-end response management run as lightweight
//!   graph nodes;
//! - the reasoning node performs deterministic retrieval, sub-agent simulation,
//!   semantic extraction, sequential diff generation, and event escalation;
//! - the summarization gate compresses world-state diffs before the deep layer;
//! - the context manager evicts semantic history into a mock vector store when
//!   utilization crosses a threshold;
//! - the subconscious node consumes gated summaries and emits a short steering
//!   directive while resetting escalation state.
//!
//! The implementation uses deterministic functions instead of live LLM calls so
//! `cargo run --example subconscious_loop` and the integration tests stay
//! offline and reproducible.

use tinyagents::graph::ClosureStateReducer;
use tinyagents::{CompiledGraph, GraphBuilder, NodeContext, NodeResult, Result, TinyAgentsError};

const DIFF_GATE_MIN_SEQUENCE: usize = 3;
const CONTEXT_EVICTION_THRESHOLD: f32 = 0.85;

/// One visible channel message in the surface layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChannelMessage {
    pub role: String,
    pub content: String,
}

/// A compact world-state mutation produced by the reasoning layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldDiff {
    pub event: String,
    pub magnitude: i32,
    pub status: String,
}

impl WorldDiff {
    pub fn new(event: impl Into<String>, magnitude: i32, status: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            magnitude,
            status: status.into(),
        }
    }
}

/// Dense package forwarded to the subconscious layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatedWorldSummary {
    pub macro_trend: String,
    pub critical_events: usize,
    pub total_magnitude: i32,
}

/// A mock long-term memory record. In production this would be a vector DB row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRecord {
    pub key: String,
    pub text: String,
}

impl MemoryRecord {
    pub fn new(key: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            text: text.into(),
        }
    }
}

/// The state carried through the closed-loop graph.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemState {
    pub messages: Vec<ChannelMessage>,
    pub channel_source: String,
    pub raw_channel_payload: String,
    pub agent_instructions: String,
    pub agent_reply: String,
    pub channel_response: String,
    pub subconscious_steering: String,
    pub semantic_history: Vec<String>,
    pub sequential_diffs: Vec<WorldDiff>,
    pub gated_world_summary: Option<GatedWorldSummary>,
    pub context_utilization: f32,
    pub trigger_subconscious: bool,
    pub cron_due: bool,
    pub long_term_memory: Vec<MemoryRecord>,
    pub retrieved_context: Vec<String>,
    pub event_log: Vec<String>,
}

impl SystemState {
    /// Builds an initial channel packet with a seeded memory corpus.
    pub fn new(channel_source: impl Into<String>, raw_payload: impl Into<String>) -> Self {
        Self {
            messages: Vec::new(),
            channel_source: channel_source.into(),
            raw_channel_payload: raw_payload.into(),
            agent_instructions: String::new(),
            agent_reply: String::new(),
            channel_response: String::new(),
            subconscious_steering: String::new(),
            semantic_history: Vec::new(),
            sequential_diffs: Vec::new(),
            gated_world_summary: None,
            context_utilization: 0.1,
            trigger_subconscious: false,
            cron_due: false,
            long_term_memory: vec![
                MemoryRecord::new(
                    "resource",
                    "Historical trace: resource reallocation occurred in cycle 42.",
                ),
                MemoryRecord::new(
                    "subagent",
                    "Historical trace: repeated sub-agent failures respond well to lower temperature.",
                ),
            ],
            retrieved_context: Vec::new(),
            event_log: Vec::new(),
        }
    }
}

/// Partial updates merged at graph step boundaries.
#[derive(Clone, Debug, Default)]
pub struct StatePatch {
    pub messages: Vec<ChannelMessage>,
    pub agent_instructions: Option<String>,
    pub agent_reply: Option<String>,
    pub channel_response: Option<String>,
    pub subconscious_steering: Option<String>,
    pub append_semantic_history: Vec<String>,
    pub replace_semantic_history: Option<Vec<String>>,
    pub append_sequential_diffs: Vec<WorldDiff>,
    pub clear_sequential_diffs: bool,
    pub set_gated_world_summary: Option<GatedWorldSummary>,
    pub clear_gated_world_summary: bool,
    pub context_utilization: Option<f32>,
    pub context_utilization_delta: f32,
    pub trigger_subconscious: Option<bool>,
    pub cron_due: Option<bool>,
    pub append_long_term_memory: Vec<MemoryRecord>,
    pub append_retrieved_context: Vec<String>,
    pub append_event_log: Vec<String>,
}

impl StatePatch {
    fn event(event: impl Into<String>) -> Self {
        Self {
            append_event_log: vec![event.into()],
            ..Self::default()
        }
    }
}

/// Builds the closed-loop graph.
pub fn build_subconscious_loop_graph() -> Result<CompiledGraph<SystemState, StatePatch>> {
    GraphBuilder::<SystemState, StatePatch>::new()
        .set_reducer(ClosureStateReducer::new(apply_patch))
        .add_node("channel_ingestion", channel_ingestion_node)
        .add_node("frontend_agent", frontend_agent_node)
        .add_node("agent_execution", agent_execution_node)
        .add_node("summarization_gate", summarization_gate_node)
        .add_node("context_manager_hook", context_manager_hook_node)
        .add_node("subconscious_eval", subconscious_eval_node)
        .set_entry("channel_ingestion")
        .add_edge("channel_ingestion", "frontend_agent")
        .add_conditional_edges(
            "frontend_agent",
            automated_loop_router,
            [
                ("agent_execution", "agent_execution"),
                ("context_manager_hook", "context_manager_hook"),
            ],
        )
        .add_edge("agent_execution", "summarization_gate")
        .add_edge("summarization_gate", "frontend_agent")
        .add_conditional_edges(
            "context_manager_hook",
            subconscious_router,
            [
                ("subconscious_eval", "subconscious_eval"),
                ("END", "__end__"),
            ],
        )
        .set_finish("subconscious_eval")
        .with_node_kind("channel_ingestion", "channel")
        .with_node_kind("frontend_agent", "quick_llm")
        .with_node_kind("agent_execution", "reasoning_llm")
        .with_node_kind("summarization_gate", "compression_gate")
        .with_node_kind("context_manager_hook", "context_manager")
        .with_node_kind("subconscious_eval", "subconscious_llm")
        .compile()
}

/// Runs the example graph from an initial state.
pub async fn run_subconscious_loop(initial: SystemState) -> Result<SystemState> {
    let graph = build_subconscious_loop_graph()?;
    Ok(graph.run(initial).await?.state)
}

fn apply_patch(mut state: SystemState, patch: StatePatch) -> Result<SystemState> {
    state.messages.extend(patch.messages);

    if let Some(value) = patch.agent_instructions {
        state.agent_instructions = value;
    }
    if let Some(value) = patch.agent_reply {
        state.agent_reply = value;
    }
    if let Some(value) = patch.channel_response {
        state.channel_response = value;
    }
    if let Some(value) = patch.subconscious_steering {
        state.subconscious_steering = value;
    }

    if let Some(history) = patch.replace_semantic_history {
        state.semantic_history = history;
    }
    state.semantic_history.extend(patch.append_semantic_history);

    if patch.clear_sequential_diffs {
        state.sequential_diffs.clear();
    }
    state.sequential_diffs.extend(patch.append_sequential_diffs);

    if patch.clear_gated_world_summary {
        state.gated_world_summary = None;
    }
    if let Some(summary) = patch.set_gated_world_summary {
        state.gated_world_summary = Some(summary);
    }

    if let Some(value) = patch.context_utilization {
        state.context_utilization = value.clamp(0.0, 1.0);
    }
    if patch.context_utilization_delta != 0.0 {
        state.context_utilization =
            (state.context_utilization + patch.context_utilization_delta).clamp(0.0, 1.0);
    }

    if let Some(value) = patch.trigger_subconscious {
        state.trigger_subconscious = value;
    }
    if let Some(value) = patch.cron_due {
        state.cron_due = value;
    }

    state.long_term_memory.extend(patch.append_long_term_memory);
    state
        .retrieved_context
        .extend(patch.append_retrieved_context);
    state.event_log.extend(patch.append_event_log);

    Ok(state)
}

fn channel_ingestion_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        Ok(NodeResult::Update(StatePatch {
            messages: vec![ChannelMessage {
                role: "user".to_string(),
                content: state.raw_channel_payload.clone(),
            }],
            append_event_log: vec![format!(
                "channel_ingestion: packet from {}",
                state.channel_source
            )],
            ..StatePatch::default()
        }))
    }
}

fn frontend_agent_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        if state.agent_reply.is_empty() {
            Ok(NodeResult::Update(StatePatch {
                agent_instructions: Some(format!(
                    "AUTONOMOUS_EXECUTE: {}",
                    state.raw_channel_payload
                )),
                append_event_log: vec!["frontend_agent: deferred macro instruction".to_string()],
                ..StatePatch::default()
            }))
        } else {
            Ok(NodeResult::Update(StatePatch {
                channel_response: Some(format!("Completed. Output: {}", state.agent_reply)),
                append_event_log: vec!["frontend_agent: compiled final response".to_string()],
                ..StatePatch::default()
            }))
        }
    }
}

fn agent_execution_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        let instructions = state.agent_instructions.clone();
        let retrieved = retrieve_context(&state.long_term_memory, &instructions);
        let anomaly = contains_any(
            &instructions,
            &["critical", "cascade", "failure", "anomaly"],
        );
        let magnitude = if anomaly {
            91
        } else if contains_any(&instructions, &["resource", "matrix", "allocation"]) {
            14
        } else {
            4
        };
        let status = if anomaly { "degraded" } else { "stable" };
        let semantic_trace = format!(
            "[Semantic Extraction] entity=sub_agents event=resource_shift magnitude={magnitude}% status={status} error_code={}",
            if anomaly { "SUBAGENT_CASCADE" } else { "NONE" }
        );
        let force_subconscious = anomaly || state.context_utilization >= 0.9;

        Ok(NodeResult::Update(StatePatch {
            agent_reply: Some(format!(
                "Pipeline mutations executed via sub-agents; {} memory traces retrieved.",
                retrieved.len()
            )),
            append_semantic_history: vec![semantic_trace],
            append_sequential_diffs: vec![WorldDiff::new("resource_shift", magnitude, status)],
            trigger_subconscious: Some(force_subconscious),
            context_utilization_delta: if anomaly { 0.2 } else { 0.05 },
            append_retrieved_context: retrieved,
            append_event_log: vec![
                "agent_execution: retrieved long-term context".to_string(),
                format!("agent_execution: emitted sequential diff status={status}"),
            ],
            ..StatePatch::default()
        }))
    }
}

fn summarization_gate_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        let should_forward =
            state.sequential_diffs.len() >= DIFF_GATE_MIN_SEQUENCE || state.trigger_subconscious;

        if !should_forward {
            return Ok(NodeResult::Update(StatePatch::event(
                "summarization_gate: threshold not met; holding diffs",
            )));
        }

        Ok(NodeResult::Update(StatePatch {
            set_gated_world_summary: Some(summarize_diffs(&state.sequential_diffs)?),
            clear_sequential_diffs: true,
            append_event_log: vec![
                "summarization_gate: forwarded consolidated world summary".to_string(),
            ],
            ..StatePatch::default()
        }))
    }
}

fn context_manager_hook_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        if state.context_utilization < CONTEXT_EVICTION_THRESHOLD {
            return Ok(NodeResult::Update(StatePatch::event(
                "context_manager_hook: utilization below eviction threshold",
            )));
        }

        let records = state
            .semantic_history
            .iter()
            .enumerate()
            .map(|(idx, trace)| MemoryRecord::new(format!("evicted-trace-{idx}"), trace.clone()))
            .collect();

        Ok(NodeResult::Update(StatePatch {
            replace_semantic_history: Some(vec![
                "--- semantic history evicted to vector DB ---".to_string(),
            ]),
            append_long_term_memory: records,
            context_utilization: Some(0.2),
            append_event_log: vec![
                "context_manager_hook: evicted semantic history to long-term memory".to_string(),
            ],
            ..StatePatch::default()
        }))
    }
}

fn subconscious_eval_node(state: SystemState, _ctx: NodeContext) -> impl FutureNode {
    async move {
        let summary = state
            .gated_world_summary
            .clone()
            .unwrap_or(GatedWorldSummary {
                macro_trend: "Cron review found no gated world-state package.".to_string(),
                critical_events: 0,
                total_magnitude: 0,
            });

        let directive = if summary.critical_events > 0 || summary.total_magnitude >= 50 {
            "STEERING_DIRECTIVE: High mutability detected. Lower sub-agent temperature and require retrieval before spawning."
        } else {
            "STEERING_DIRECTIVE: System stable. Preserve current execution policy."
        };

        Ok(NodeResult::Update(StatePatch {
            subconscious_steering: Some(directive.to_string()),
            trigger_subconscious: Some(false),
            cron_due: Some(false),
            clear_gated_world_summary: true,
            append_event_log: vec![format!(
                "subconscious_eval: digested summary '{}'",
                summary.macro_trend
            )],
            ..StatePatch::default()
        }))
    }
}

fn automated_loop_router(state: &SystemState) -> &'static str {
    if state.channel_response.is_empty() {
        "agent_execution"
    } else {
        "context_manager_hook"
    }
}

fn subconscious_router(state: &SystemState) -> &'static str {
    if state.trigger_subconscious || state.cron_due || state.gated_world_summary.is_some() {
        "subconscious_eval"
    } else {
        "END"
    }
}

fn retrieve_context(memory: &[MemoryRecord], query: &str) -> Vec<String> {
    let query = query.to_ascii_lowercase();
    memory
        .iter()
        .filter(|record| {
            query.contains(&record.key.to_ascii_lowercase())
                || record
                    .key
                    .split(['-', '_'])
                    .any(|term| !term.is_empty() && query.contains(term))
        })
        .map(|record| record.text.clone())
        .collect()
}

fn summarize_diffs(diffs: &[WorldDiff]) -> Result<GatedWorldSummary> {
    if diffs.is_empty() {
        return Err(TinyAgentsError::Validation(
            "summarization gate received no diffs".to_string(),
        ));
    }

    let critical_events = diffs
        .iter()
        .filter(|diff| diff.status != "stable" || diff.magnitude >= 50)
        .count();
    let total_magnitude = diffs.iter().map(|diff| diff.magnitude).sum();

    Ok(GatedWorldSummary {
        macro_trend: format!(
            "Aggregated {} operational shifts; {} critical events; total magnitude {}%.",
            diffs.len(),
            critical_events,
            total_magnitude
        ),
        critical_events,
        total_magnitude,
    })
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    needles.iter().any(|needle| haystack.contains(needle))
}

trait FutureNode:
    std::future::Future<Output = Result<NodeResult<StatePatch>>> + Send + 'static
{
}

impl<T> FutureNode for T where
    T: std::future::Future<Output = Result<NodeResult<StatePatch>>> + Send + 'static
{
}
