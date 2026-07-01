//! Graph-level orchestration controls.
//!
//! This module is the graph runtime's managed child-work surface. It gives
//! language-model orchestrators stable task ids and typed controls (`spawn`,
//! `await`, `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, and
//! `steer`) without exposing raw executor handles such as `tokio::JoinHandle`.
//!
//! The controls are ordinary harness tools. Use [`OrchestrationTool`] directly,
//! call [`orchestration_tools`] to build the full set, or call
//! [`register_orchestration_tools`] to insert them into a
//! [`crate::harness::tool::ToolRegistry`] alongside any other tools.

mod store;
mod tool;
mod types;

pub use store::{InMemoryTaskStore, JsonlTaskStore, TaskStore};
pub use tool::{
    OrchestrationTool, orchestration_tool_schema, orchestration_tool_schemas, orchestration_tools,
    register_orchestration_tools,
};
pub use types::*;

#[cfg(test)]
mod test;
