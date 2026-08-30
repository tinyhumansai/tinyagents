//! # TinyAgents — a durable agent + graph harness for Rust
//!
//! TinyAgents is a typed, durable runtime where **language models call models,
//! agents call agents, and graphs run graphs** — and where a model can author,
//! compile, and run the very workflow it is standing inside, all as inspectable,
//! checkpointed, policy-checked Rust.
//!
//! The "recursive" framing is the through-line of the whole crate, not a
//! footnote: rather than stuffing everything into one context window, a model
//! treats long context as an external *environment* it decomposes, recursively
//! calling sub-models, sub-agents, and sub-graphs as functions. TinyAgents
//! brings that to Rust as a production-shaped harness (capability calls by
//! name, depth tracking, and trajectory/event logging). The *scripted* form of
//! that loop — an embedded interpreter running model-written code cells — is a
//! host concern built on these surfaces, not something this crate ships.
//!
//! ## The four surfaces
//!
//! 1. **Harness** ([`harness`]) — provider-neutral model calls, typed tools,
//!    middleware, structured output, streaming, usage/cost, retry/limits, cache,
//!    memory/embeddings, sub-agents, steering, and a testkit.
//! 2. **Graph runtime** ([`graph`]) — LangGraph-style durable typed state
//!    graphs: [`START`]/[`END`], nodes, conditional routing, [`Command`]s,
//!    fan-out, reducers/channels, [`Checkpoint`]s, [`Interrupt`]s, subgraphs,
//!    streaming, topology export, and per-thread productivity primitives — a
//!    durable [`ThreadGoal`] with graph-native continuation and a
//!    [`TaskBoard`] kanban — exposed as harness tools.
//! 3. **Registry** ([`registry`]) — a named capability catalog (models, tools,
//!    agents, graphs, stores, middleware, policy) that `.rag` binds by name.
//! 4. **Expressive language `.rag`** ([`language`]) — a declarative,
//!    side-effect-free blueprint format that compiles (lexer → parser →
//!    compiler) into the same graph/harness runtime; the safe boundary for
//!    agent-authored plans.
//!
//! ## The recursion story
//!
//! `.rag` lowers into the *same* [`graph`] + [`harness`] types
//! as hand-written Rust — a language whose programs are the runtime that
//! interprets them. A harness agent can be exposed *as a tool* to another agent
//! ([`SubAgent`], [`SubAgentTool`], [`SubAgentSession`]), so orchestration is
//! just a model calling a model; the runtime tracks parent/child run lineage and
//! enforces a recursion cap ([`TinyAgentsError::SubAgentDepth`]). At the deepest
//! level a model can emit a `.rag` blueprint that compiles through the same
//! registry-bound path as a human-authored file and runs on the same runtime the
//! model is already executing in (see `examples/openai_self_blueprint.rs`).
//!
//! ## Provider features
//!
//! Hosted and local providers (OpenAI plus the OpenAI-compatible endpoints for
//! Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together, and Mistral)
//! are compiled in unconditionally alongside the offline, deterministic
//! [`harness::providers::MockModel`]. Two Cargo features gate optional,
//! heavier dependencies instead: `sqlite` (embedded SQLite checkpointer,
//! [`graph::checkpoint::SqliteCheckpointer`]) and `tools` (the builtin generic
//! tool family, [`harness::tools`]).
//!
//! Scripted, imperative orchestration surfaces (an embedded interpreter driving
//! capability calls — the `.ragsh` REPL and the recursive-language-model
//! runtime that used to ship here) are deliberately *not* part of this crate:
//! they are host concerns, built on top of [`registry`] and [`harness`].
//!
//! ## Crate-root re-exports
//!
//! For discoverability the most-used types from each surface are re-exported at
//! the crate root, grouped below by surface ([`error`], [`registry`],
//! [`language`], [`harness`], and [`graph`]).

pub mod error;
pub mod graph;
pub mod harness;
pub mod language;
pub mod registry;
/// Durable session history and run ledger — a persistence domain in its own
/// right, not part of the agent-loop harness. Requires the `sqlite` feature.
#[cfg(feature = "sqlite")]
pub mod session;

// --- Session: durable session history + run ledger (feature `sqlite`) ---
// Centralized here per AGENTS.md so downstream users get a predictable surface
// rather than reaching through the module path. The record/query entry points
// and the ledger's coordination types are the surface a host actually binds
// against; the rest stays reachable via `session::` for callers that want it.
#[cfg(feature = "sqlite")]
pub use session::run_ledger::{
    AgentRun, AgentRunKind, AgentRunStatus, AgentRunUpsert, AgentTeam, AgentTeamMember,
    AgentTeamMemberStatus, AgentTeamStatus, AgentTeamTask, AgentTeamTaskStatus, ClaimOutcome,
    CompletionOutcome, RunEvent, RunEventAppend, RunTelemetry, RunTelemetryUpsert, WorkflowRun,
    WorkflowRunStatus, WorkflowRunUpsert,
};
#[cfg(feature = "sqlite")]
pub use session::{
    SessionMessage, SessionRecord, SessionSearchParams, SessionSearchResult, SessionStatus,
    SessionToolCall, get_session, list_sessions, record_message, record_session_end,
    record_session_start, record_tool_call, search_sessions,
};

// --- Error: the crate-wide error type and `Result` alias ---
pub use error::{Result, TinyAgentsError};

// --- Registry: named capability catalog (.rag binding by name) ---
pub use registry::{
    AliasBinding, CapabilityRegistry, ComponentId, ComponentKind, ComponentMetadata,
    DiagnosticSeverity, ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot,
    ModelCatalogSource, ModelPricing, ModelRouter, RegistryDiagnostic, RegistrySnapshot,
    WorkloadRoute,
};

// --- Language: registry → blueprint binding façade ---
// The strict, registry-backed entry points the REPL and orchestrators use to
// turn `.rag` source into validated blueprints. `compile_source` runs
// parse -> compile -> registry-bind in one call.
pub use language::capability_resolver::{
    CapabilityResolver, bind_capabilities, bind_capabilities_with_registry,
};
pub use language::compiler::{compile, compile_source, compile_with_provenance};
// `Resolver` is the registry-backed binding gate: it resolves every reference in
// a `.rag` plan (file-backed or model-generated) against the registry, producing
// spanned diagnostics for unknown/disallowed capabilities. `resolve_source` is
// the recommended parse -> resolve -> lower façade.
pub use language::resolver::{Resolver, resolve_source};
pub use language::types::{
    Blueprint, BlueprintProvenance, ChannelSpec, CommandSpec, EdgeSpan, EdgeSpec, IoFieldSpec,
    JoinSpec, NamedSpan, NodeSpec, Origin, Routing, SendSpec,
};
// `blueprint_diff` produces a structured, human-readable `BlueprintDiff` of two
// compiled blueprints — the basis for generated-workflow review and the REPL
// `graph_diff` builtin. `testkit` holds deterministic compile/assert helpers.
pub use language::diff::{BlueprintDiff, ChannelDiff, FieldChange, NodeDiff, blueprint_diff};
pub use language::testkit;

// --- Language: diagnostics, spans, and the source map ---
// Structured, source-aware errors for `.rag`: a `Diagnostic` (with `Severity`
// and labelled spans) rendered against a `SourceFile`/`SourceMap` with caret
// underlines.
pub use language::diagnostic::{Diagnostic, Label, Severity};
pub use language::source::{SourceFile, SourceId, SourceMap};
pub use language::span::Span;

// --- Harness: embeddings + retrieval ---
pub use harness::embeddings::{
    EmbeddingModel, InMemoryVectorStore, MockEmbeddingModel, Retriever, ScoredDoc, VectorStore,
    cosine_similarity,
};

// --- Harness: first-class sub-agents (agent-calling-agent composition) ---
pub use harness::subagent::{SubAgent, SubAgentSession, SubAgentTool};

// --- Harness: orchestrator → sub-agent steering ---
pub use harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
};

// --- Harness: active-run message queues ---
pub use harness::run_queue::{QueueLane, QueueStatus, RunQueue};

// --- Cooperative run cancellation ---
pub use harness::cancel::CancellationToken;

// --- Workspace isolation / sandbox hooks ---
pub use harness::workspace::{SharedRootWorkspace, WorkspaceDescriptor, WorkspaceIsolation};

// --- Harness: durable observability (journals, status stores, sinks) ---
pub use harness::observability::{
    AgentCallLatency, AgentLatencyMetrics, AgentObservation, FanOutSink, HarnessEventJournal,
    HarnessStatusStore, InMemoryEventJournal, InMemoryStatusStore, JournalSink, JsonlSink,
    RedactingSink, StoreEventJournal,
};
pub use harness::observability::{
    LangfuseAuth, LangfuseClient, LangfuseScore, LangfuseScoreValue, LangfuseTraceConfig,
};

// --- Graph: durable execution model (LangGraph-style) ---
// Re-exported with explicit names so the durable API is discoverable at the
// crate root. The `harness::stream::StreamMode` and `graph::stream::StreamMode`
// types intentionally stay behind their module paths to avoid a name clash.
#[cfg(feature = "sqlite")]
pub use graph::SqliteCheckpointer;
pub use graph::{
    BarrierArrivals, Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource,
    CheckpointTuple, Checkpointer, ChildRun, ChildRunSink, ClosureReducer, ClosureStateReducer,
    Command, CompiledGraph, DurabilityMode, END, FileCheckpointer, ForkId, GraphBuilder,
    GraphDefaults, GraphEvent, GraphExecution, GraphInput, GraphRunStatus, InMemoryCheckpointer,
    Interrupt, NodeContext, NodeResult, PendingActivation, RecursionFrame, RecursionPolicy,
    RecursionStack, Reducer, ResumeTarget, Route, RouteTarget, RunTree, START, StateReducer,
    StateSnapshot,
};

// --- Graph: sub-agent nodes (delegate a graph step to a registered agent) ---
pub use graph::{
    HarnessAgent, HarnessSubAgent, SubAgentBudget, SubAgentInput, SubAgentNode, SubAgentOutput,
    SubAgentPolicy, subagent_node,
};

// --- Graph: channel-per-field state model (additive; see state-channels.md) ---
// An opt-in alternative to the monolithic State + StateReducer path: state is
// split into independently-merged named channels.
pub use graph::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, Delta, Ephemeral,
    LastValue, Messages, NamedBarrier, Topic, Untracked,
};

// --- Graph: durable observability (journals, status stores, journaling sink) ---
// Names are graph-prefixed so they never collide with the harness observability
// re-exports above.
pub use graph::{
    GraphEventJournal, GraphHealthSummary, GraphLangfuseExporter, GraphLatencyMetrics,
    GraphNodeHealth, GraphNodeLatency, GraphObservation, GraphStatusStore, GraphStepLatency,
    InMemoryGraphEventJournal, InMemoryGraphStatusStore, JournalGraphSink, SpanMetadataFn,
    StoreGraphEventJournal,
};

// --- Graph: orchestration tools (ordinary harness Tool implementations) ---
pub use graph::{
    CancelledDetachedTask, DetachedTaskRegistry, DetachedTaskRegistryError, DetachedTaskSnapshot,
    DetachedTaskWaitOutcome, InMemoryTaskStore, JsonlTaskStore, OrchestrationControlOutcome,
    OrchestrationTaskFilter, OrchestrationTaskKind, OrchestrationTaskRecord,
    OrchestrationTaskResult, OrchestrationTaskSpec, OrchestrationTaskStatus, OrchestrationTool,
    OrchestrationToolKind, SteeringRegistry, TaskStore, orchestration_tool_schema,
    orchestration_tool_schemas, orchestration_tools, orchestration_tools_with_steering,
    register_orchestration_tools,
};

// --- Graph: multi-stage sub-agent delegation (plan → execute ⇄ review → finalize) ---
// The per-stage worker is injected, so the host supplies how a stage runs while
// this crate owns the routing, revision budget, checkpoint/resume classification
// and the durable human-approval interrupt. `DelegationState` is an on-disk
// format — see `graph::delegation`'s README before changing its shape.
pub use graph::delegation::{
    CURRENT_SCHEMA_VERSION as DELEGATION_SCHEMA_VERSION, DelegationConfig, DelegationOutcome,
    DelegationStage, DelegationStageOutput, DelegationState, PendingApproval, StepRecord,
    delegation_graph_topology, deny_decision, resume_delegation, run_delegation,
    run_delegation_durable, run_or_resume_delegation,
};

// --- Graph: per-thread goal (durable objective + graph-native continuation) ---
// `goal_store` is the programmatic CRUD surface (get/set/complete/account_usage);
// the tools and continuation helpers are re-exported flat for discoverability.
pub use graph::goals::store as goal_store;
pub use graph::{
    BudgetVerdict, GoalBudgetGuard, GoalProgress, GoalTool, GoalToolKind, ThreadGoal,
    ThreadGoalStatus, TurnOutcome, account_turn, accrues_usage, active_goal_context_block,
    goal_gate_node, goal_tools, note_user_turn, register_goal_tools, run_continuation_tick,
    turn_tokens,
};

// --- Graph: per-thread task board (kanban todos) ---
// `todo_store` is the programmatic CRUD surface (add/edit/claim_card/...);
// `task_run_store` is the claim/heartbeat/reclaim log a dispatcher writes
// alongside it. The tool, data model, and dispatch policy are re-exported flat
// for discoverability.
pub use graph::todos::runs::store as task_run_store;
pub use graph::todos::store as todo_store;
pub use graph::{
    ActiveRun, ActiveRunRegistry, CardPatch, PollCadence, ReclaimDetail, ReclaimResult, RunLimits,
    RunOutcome, TaskApprovalMode, TaskBoard, TaskBoardCard, TaskCardStatus, TaskPromptTools,
    TaskRun, TodoTool, TodosSnapshot, build_progress_instruction, build_task_prompt, card_urgency,
    has_card_in_progress, normalise_board, parse_status, pick_next_card, register_todo_tools,
    render_markdown, requires_plan_approval, staleness_reason, todo_tools,
};

// --- Graph: parallel map/reduce helper ---
pub use graph::parallel::{
    FailurePolicy, ItemOutcome, ParallelOptions, ParallelOutcome, map_reduce,
};

// --- Graph: shared-workspace claim arbitration ---
pub use graph::parallel::{
    ClaimConflict, ClaimPathError, DispatchMode, DispatchPlan, WorkspaceClaim,
    parse_relative_claim_paths, paths_overlap, plan_shared_workspace_dispatch,
    writes_shared_workspace,
};

// --- Graph: export / visualization ---
// Topology types are surfaced at the crate root; the `to_json`/`to_mermaid`
// free functions stay behind `graph::export::` to avoid generic-name clashes.
pub use graph::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphPolicySummary, GraphTopology, NodeInfo,
    NodePolicySummary, RouteInfo, ValidationReport, WaitingEdgeInfo,
};

// --- Graph: testkit (deterministic node doubles + run assertions) ---
// The fluent `assert_graph` builder and node-double constructors stay behind
// `graph::testkit::` (and are re-exported here) so downstream crates can test
// graphs without a live model. Names are graph-test specific to avoid clashing
// with the harness `testkit`.
pub use graph::testkit::{
    GraphAssertions, GraphEventRecorder, GraphRun, RetryCountingNode, StreamCollector,
    assert_graph, failing_node, fanout_node, interrupting_node, noop_node, run_recorded,
    scripted_route_node, scripted_update_node, subagent_fake_node, subgraph_test_node,
};
