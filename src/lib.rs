//! TinyAgents is a small foundation for LLM applications built around
//! composable chat models, tools, and executable state graphs.

pub mod chat;
pub mod error;
pub mod graph;
pub mod harness;
pub mod language;
pub mod model;
pub mod registry;
pub mod repl;
pub mod tool;

pub use chat::{ChatMessage, ChatRole};
pub use error::{Result, TinyAgentsError};

// --- Registry: named capability catalog (.rag/.ragsh binding by name) ---
pub use registry::{CapabilityRegistry, ComponentId, ComponentKind, ComponentMetadata};

// --- Language: registry → blueprint binding façade ---
// The strict, registry-backed entry points the REPL and orchestrators use to
// turn `.rag`/`.ragsh` source into validated blueprints. `compile_source` runs
// parse -> compile -> registry-bind in one call.
pub use language::compiler::{
    CapabilityResolver, bind_capabilities, bind_capabilities_with_registry, compile, compile_source,
};
pub use language::types::Blueprint;

pub use model::{ChatModel, ModelRequest, ModelResponse};
pub use tool::{Tool, ToolCall, ToolResult};

// --- Harness: embeddings + retrieval ---
pub use harness::embeddings::{
    EmbeddingModel, InMemoryVectorStore, MockEmbeddingModel, Retriever, ScoredDoc, VectorStore,
    cosine_similarity,
};

// --- Harness: first-class sub-agents (agent-calling-agent composition) ---
pub use harness::subagent::{SubAgent, SubAgentTool};

// --- Harness: orchestrator → sub-agent steering ---
pub use harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringOutcome, SteeringPolicy,
};

// --- Graph: legacy sequential API (milestone 1) ---
pub use graph::{Edge, GraphRun, Node, NodeOutput, StateGraph};

// --- Graph: durable execution model (LangGraph-style) ---
// Re-exported with explicit names so the durable API is discoverable at the
// crate root. The `harness::stream::StreamMode` and `graph::stream::StreamMode`
// types intentionally stay behind their module paths to avoid a name clash.
pub use graph::{
    Checkpoint, CheckpointMetadata, Checkpointer, ClosureReducer, ClosureStateReducer, Command,
    CompiledGraph, END, ForkId, GraphBuilder, GraphEvent, GraphExecution, GraphRunStatus,
    InMemoryCheckpointer, Interrupt, NodeContext, NodeResult, Reducer, START, StateReducer,
};

// --- Graph: export / visualization ---
// Topology types are surfaced at the crate root; the `to_json`/`to_mermaid`
// free functions stay behind `graph::export::` to avoid generic-name clashes.
pub use graph::{ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphTopology, NodeInfo, RouteInfo};
