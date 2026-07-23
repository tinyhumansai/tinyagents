//! TinyAgents graph runtime.
//!
//! The graph module is TinyAgents' durable workflow runtime (LangGraph-style)
//! and one of the load-bearing surfaces of the crate's recursive language-model
//! (RLM) architecture: because a node can embed another compiled graph
//! ([`subgraph`]) or invoke a sub-agent, **graphs run graphs** and orchestration
//! recurses while every step stays typed, checkpointed, and observable. A
//! workflow authored from a `.rag` blueprint or driven from the `.ragsh` REPL
//! lowers into exactly these same types, so a model can describe, compile, and
//! re-enter the very runtime it is executing inside.
//!
//! The pieces: partial updates and reducers ([`reducer`]), commands and
//! interrupts ([`command`]), a builder/compile contract ([`builder`]), a
//! superstep executor ([`compiled`]), checkpointing ([`checkpoint`]),
//! streaming/events ([`stream`]), run-status snapshots ([`status`]), graph
//! export/visualization ([`export`]), subgraph embedding ([`subgraph`]), and
//! per-thread productivity primitives — a durable goal ([`goals`]) and a kanban
//! task board ([`todos`]) — exposed as harness tools.
//!
//! Each concern lives in its own submodule with `types.rs` (definitions),
//! `mod.rs` (implementations), and `test.rs` (unit tests).

pub mod builder;
pub mod channel;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod export;
pub mod goals;
pub mod observability;
pub mod orchestration;
pub mod parallel;
pub mod recursion;
pub mod reducer;
pub mod status;
pub mod stream;
pub mod subagent_node;
pub mod subgraph;
pub mod testkit;
pub(crate) mod thread_locks;
pub mod todos;

// --- Durable execution model ---
pub use builder::{
    END, ForkId, GraphBuilder, GraphDefaults, NodeContext, NodeFuture, NodeHandler, Route,
    RouterFn, START,
};
pub use channel::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, Delta, Ephemeral,
    LastValue, Messages, NamedBarrier, Topic, Untracked,
};
#[cfg(feature = "sqlite")]
pub use checkpoint::SqliteCheckpointer;
pub use checkpoint::{
    BarrierArrivals, Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource,
    CheckpointTuple, Checkpointer, DurabilityMode, FileCheckpointer, InMemoryCheckpointer,
    PendingActivation, PendingWrite,
};
pub use command::{Command, Interrupt, NodeResult, RouteTarget, Send};
pub use compiled::{CompiledGraph, GraphExecution, GraphInput, ResumeTarget, StateSnapshot};
pub use export::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphPolicySummary, GraphTopology, NodeInfo,
    NodePolicySummary, RouteInfo, ValidationReport, WaitingEdgeInfo, blueprint_to_json,
    blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
pub use goals::{
    GoalProgress, GoalTool, GoalToolKind, ThreadGoal, ThreadGoalStatus, TurnOutcome,
    active_goal_context_block, goal_gate_node, goal_tools, note_user_turn, register_goal_tools,
    run_continuation_tick,
};
pub use observability::{
    GraphEventJournal, GraphHealthSummary, GraphLangfuseExporter, GraphLatencyMetrics,
    GraphNodeHealth, GraphNodeLatency, GraphObservation, GraphStatusStore, GraphStepLatency,
    InMemoryGraphEventJournal, InMemoryGraphStatusStore, JournalGraphSink, SpanMetadataFn,
    StoreGraphEventJournal,
};
pub use orchestration::{
    CancelledDetachedTask, DetachedTaskRegistry, DetachedTaskRegistryError, DetachedTaskSnapshot,
    DetachedTaskWaitOutcome, InMemoryTaskStore, JsonlTaskStore, OrchestrationControlOutcome,
    OrchestrationTaskFilter, OrchestrationTaskKind, OrchestrationTaskRecord,
    OrchestrationTaskResult, OrchestrationTaskSpec, OrchestrationTaskStatus, OrchestrationTool,
    OrchestrationToolKind, SteeringRegistry, TaskStore, orchestration_tool_schema,
    orchestration_tool_schemas, orchestration_tools, orchestration_tools_with_steering,
    register_orchestration_tools,
};
pub use recursion::{
    ChildRun, ChildRunSink, RecursionFrame, RecursionPolicy, RecursionStack, RunTree,
};
pub use reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
pub use status::GraphRunStatus;
pub use stream::{CollectingSink, GraphEvent, GraphEventSink, NoopSink, StreamMode};
pub use subagent_node::{
    HarnessAgent, HarnessSubAgent, InputMapper, OutputMapper, SubAgentBudget, SubAgentInput,
    SubAgentNode, SubAgentOutput, SubAgentPolicy, subagent_node,
};
pub use subgraph::{adapter_subgraph_node, shared_subgraph_node};
pub use testkit::{
    GraphAssertions, GraphEventRecorder, GraphRun, RetryCountingNode, StreamCollector,
    assert_graph, failing_node, fanout_node, interrupting_node, noop_node, run_recorded,
    scripted_route_node, scripted_update_node, subagent_fake_node, subgraph_test_node,
};
pub use todos::{
    CardPatch, TaskApprovalMode, TaskBoard, TaskBoardCard, TaskCardStatus, TodoTool, TodosSnapshot,
    normalise_board, parse_status, register_todo_tools, render_markdown, todo_tools,
};
