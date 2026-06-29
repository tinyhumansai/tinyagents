//! RustAgents is a small foundation for LLM applications built around
//! composable chat models, tools, and executable state graphs.

pub mod chat;
pub mod error;
pub mod graph;
pub mod harness;
pub mod model;
pub mod registry;
pub mod tool;

pub use chat::{ChatMessage, ChatRole};
pub use error::{Result, RustAgentsError};
pub use model::{ChatModel, ModelRequest, ModelResponse};
pub use tool::{Tool, ToolCall, ToolResult};

// --- Graph: legacy sequential API (milestone 1) ---
pub use graph::{Edge, GraphRun, Node, NodeOutput, StateGraph};

// --- Graph: durable execution model (LangGraph-style) ---
// Re-exported with explicit names so the durable API is discoverable at the
// crate root. The `harness::stream::StreamMode` and `graph::stream::StreamMode`
// types intentionally stay behind their module paths to avoid a name clash.
pub use graph::{
    Checkpoint, CheckpointMetadata, Checkpointer, Command, CompiledGraph, END, GraphBuilder,
    GraphEvent, GraphExecution, GraphRunStatus, InMemoryCheckpointer, Interrupt, NodeContext,
    NodeResult, Reducer, START, StateReducer,
};
