//! RustAgents is a small foundation for LLM applications built around
//! composable chat models, tools, and executable state graphs.

pub mod chat;
pub mod error;
pub mod graph;
pub mod harness;
pub mod model;
pub mod tool;

pub use chat::{ChatMessage, ChatRole};
pub use error::{Result, RustAgentsError};
pub use graph::{Edge, GraphRun, Node, NodeOutput, StateGraph};
pub use model::{ChatModel, ModelRequest, ModelResponse};
pub use tool::{Tool, ToolCall, ToolResult};
