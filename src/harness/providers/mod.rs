//! Model provider integrations — the leaves of the recursion.
//!
//! Every recursive call in the runtime — an agent, a sub-agent, a graph node, a
//! `.ragsh` step — ultimately bottoms out in a concrete model invocation, and
//! that invocation goes through a provider adapter here. Adapters translate
//! between TinyAgents' provider-neutral request/response types
//! ([`ModelRequest`]/[`ModelResponse`]) and a provider's own wire API, so the
//! recursive machinery above stays provider-agnostic and no provider-specific
//! JSON leaks into core harness code.
//!
//! # Available providers
//!
//! | Provider | Status |
//! |---|---|
//! | [`MockModel`] | Implemented — deterministic, no network |
//! | [`openai`] (and OpenAI-compatible endpoints) | Implemented |
//!
//! [`MockModel`] is always compiled and needs no network, keeping the default
//! build offline and deterministic. The [`openai`] module is always compiled
//! too (it pulls no extra dependencies) and additionally serves every
//! OpenAI-compatible endpoint (Ollama, DeepSeek, Groq, xAI, OpenRouter,
//! Together, Mistral, and Anthropic's OpenAI-compat endpoint) through the same
//! Chat Completions wire format. The default build stays offline anyway: the
//! adapter only touches the network when invoked, and the live tests
//! early-return without `OPENAI_API_KEY`.
//!
//! To add a provider with a different wire protocol, gate it behind a new
//! Cargo feature and add the corresponding module declaration:
//!
//! ```text
//! pub mod openai;                          // always compiled
//! // #[cfg(feature = "anthropic")] pub mod anthropic;
//! // #[cfg(feature = "ollama")]    pub mod ollama;
//! ```

mod types;

// --- real provider integrations ---
// The OpenAI Chat Completions adapter is always compiled; it also serves every
// OpenAI-compatible endpoint. Providers with a different wire protocol would be
// added behind their own Cargo feature.
pub mod openai;
// #[cfg(feature = "anthropic")] pub mod anthropic;
// #[cfg(feature = "ollama")]    pub mod ollama;

pub use types::*;

use async_trait::async_trait;
use serde_json::Value;

use crate::Result;
use crate::error::TinyAgentsError;
use crate::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use crate::harness::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
};
use crate::harness::tool::ToolCall;
use crate::harness::usage::Usage;

mod mock;

#[cfg(test)]
mod test;
