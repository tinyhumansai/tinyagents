//! Model adapters whose behavior depends on TinyAgents prompt dialects.

#[cfg(feature = "claude-code")]
pub mod claude_agent_sdk;
#[cfg(feature = "claude-code")]
pub mod claude_code;
