//! Harness runtime modules.
//!
//! The harness is intentionally split by feature. Each submodule owns one
//! substantial part of model/tool orchestration so the implementation can grow
//! without creating one large runtime file.

pub mod agent_loop;
pub mod cache;
pub mod context;
pub mod cost;
pub mod embeddings;
pub mod events;
pub mod ids;
pub mod limits;
pub mod memory;
pub mod message;
pub mod middleware;
pub mod model;
pub mod prompt;
pub mod providers;
pub mod retry;
pub mod runtime;
pub mod steering;
pub mod store;
pub mod stream;
pub mod structured;
pub mod subagent;
pub mod summarization;
pub mod testkit;
pub mod tool;
pub mod usage;

pub use cost::CostTotals;
pub use ids::*;
pub use message::{ContentBlock, Message};
pub use model::{
    CapabilitySet, Modalities, ModelProfile, ModelRequest, ModelResponse, ModelStatus, ModelStream,
    ModelStreamItem, ProviderError, ResponseFormat, StreamAccumulator, ToolChoice,
    collect_model_stream,
};
pub use tool::{
    Tool as HarnessTool, ToolCall as HarnessToolCall, ToolRegistry,
    ToolResult as HarnessToolResult, ToolSchema,
};
pub use usage::Usage;
