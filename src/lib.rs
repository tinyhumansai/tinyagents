//! # TinyAgents — a recursive language-model (RLM) harness for Rust
//!
//! TinyAgents is a typed, durable runtime where **language models call models,
//! agents call agents, and graphs run graphs** — and where a model can author,
//! compile, and run the very workflow it is standing inside, all as inspectable,
//! checkpointed, policy-checked Rust.
//!
//! The "recursive" framing is the through-line of the whole crate, not a
//! footnote. It is architected around the execution model described in
//! "Recursive Language Models" (Alex L. Zhang, Tim Kraska, Omar Khattab, MIT
//! CSAIL, 2025; <https://arxiv.org/abs/2512.24601>): rather than stuffing
//! everything into one context window, a model treats long context as an
//! external *environment* it interacts with through a REPL — examining,
//! decomposing, and recursively calling sub-models over snippets. TinyAgents
//! brings that idea to Rust as a production-shaped harness (sub-model /
//! sub-agent / sub-graph calls as functions, persistent session values, depth
//! tracking, and trajectory/event logging). It is *inspired by and architected
//! around* the RLM execution model, not a reimplementation of the paper's
//! benchmarks.
//!
//! ## The five surfaces
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
//!    agents, graphs, stores, middleware, policy) that `.rag`/`.ragsh` bind by
//!    name.
//! 4. **Expressive language `.rag`** ([`language`]) — a declarative,
//!    side-effect-free blueprint format that compiles (lexer → parser →
//!    compiler) into the same graph/harness runtime; the safe boundary for
//!    agent-authored plans.
//! 5. **REPL language `.ragsh`** ([`repl`]) — imperative, capability-bound
//!    interactive orchestration; the RLM/CodeAct loop surface.
//!
//! ## The recursion story
//!
//! Both `.rag` and `.ragsh` lower into the *same* [`graph`] + [`harness`] types
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
//! [`harness::providers::MockModel`]. Three Cargo features gate optional,
//! heavier dependencies instead: `sqlite` (embedded SQLite checkpointer,
//! [`graph::checkpoint::SqliteCheckpointer`]), `repl` (embedded Rhai engine
//! powering the `.ragsh` session runtime, [`repl::session`]), and `rlm` (the
//! recursive-language-model runtime: a driver model writes code cells run in
//! a sandboxed interpreter — embedded Rhai or an external Python/JavaScript
//! process — whose only host surface is capability calls back into the
//! registry).
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
pub mod repl;
#[cfg(feature = "rlm")]
pub mod rlm;

// --- Error: the crate-wide error type and `Result` alias ---
pub use error::{Result, TinyAgentsError};

// --- Registry: named capability catalog (.rag/.ragsh binding by name) ---
pub use registry::{
    AliasBinding, CapabilityRegistry, ComponentId, ComponentKind, ComponentMetadata,
    DiagnosticSeverity, ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot,
    ModelCatalogSource, ModelPricing, RegistryDiagnostic, RegistrySnapshot,
};

// --- Language: registry → blueprint binding façade ---
// The strict, registry-backed entry points the REPL and orchestrators use to
// turn `.rag`/`.ragsh` source into validated blueprints. `compile_source` runs
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
pub use harness::observability::{LangfuseAuth, LangfuseClient, LangfuseTraceConfig};

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
    InMemoryTaskStore, JsonlTaskStore, OrchestrationControlOutcome, OrchestrationTaskFilter,
    OrchestrationTaskKind, OrchestrationTaskRecord, OrchestrationTaskResult, OrchestrationTaskSpec,
    OrchestrationTaskStatus, OrchestrationTool, OrchestrationToolKind, SteeringRegistry, TaskStore,
    orchestration_tool_schema, orchestration_tool_schemas, orchestration_tools,
    orchestration_tools_with_steering, register_orchestration_tools,
};

// --- Graph: per-thread goal (durable objective + graph-native continuation) ---
// `goal_store` is the programmatic CRUD surface (get/set/complete/account_usage);
// the tools and continuation helpers are re-exported flat for discoverability.
pub use graph::goals::store as goal_store;
pub use graph::{
    GoalProgress, GoalTool, GoalToolKind, ThreadGoal, ThreadGoalStatus, TurnOutcome,
    active_goal_context_block, goal_gate_node, goal_tools, note_user_turn, register_goal_tools,
    run_continuation_tick,
};

// --- Graph: per-thread task board (kanban todos) ---
// `todo_store` is the programmatic CRUD surface (add/edit/claim_card/...); the
// tool and data model are re-exported flat for discoverability.
pub use graph::todos::store as todo_store;
pub use graph::{
    CardPatch, TaskApprovalMode, TaskBoard, TaskBoardCard, TaskCardStatus, TodoTool, TodosSnapshot,
    normalise_board, parse_status, register_todo_tools, render_markdown, todo_tools,
};

// --- Graph: parallel map/reduce helper ---
pub use graph::parallel::{
    FailurePolicy, ItemOutcome, ParallelOptions, ParallelOutcome, map_reduce,
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

// --- REPL language `.ragsh` Rhai session runtime (feature = "repl") ---
// The imperative orchestration surface. Gated behind the `repl` feature so the
// default build does not pull in the embedded Rhai engine. `ReplSession` here is
// the scripting session from `repl::session`; the line-oriented command session
// remains available as `repl::ReplSession`.
#[cfg(feature = "repl")]
pub use repl::session::{
    LanguageCompiler, ReplCallKind, ReplCallRecord, ReplCancelFlag, ReplCapabilities, ReplPolicy,
    ReplResult, ReplSession, ReplValue, ReplVariables,
};

// --- RLM runtime (feature = "rlm") ---
// The recursive-language-model surface: a driver model writes code cells that
// run in a sandboxed interpreter (embedded Rhai or an external Python/Node
// process) whose only host surface is capability calls (`llm`, `tool`,
// `agent`) back into the registry. Config-driven end to end (`RlmConfig`).
#[cfg(feature = "rlm")]
pub use rlm::{
    CellOutcome, HostCall, InterpreterSpec, RlmCallKind, RlmCallRecord, RlmCancelFlag, RlmConfig,
    RlmHost, RlmHostApi, RlmInterpreter, RlmOutcome, RlmPolicy, RlmRunner, RlmSession, RlmStep,
    RlmStopReason, RlmTemplate, TemplateSpec,
};
