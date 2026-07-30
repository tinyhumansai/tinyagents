//! Real OpenAI Chat Completions provider (feature `openai`).
//!
//! This is one of the concrete leaves the recursive runtime bottoms out in: a
//! single [`OpenAiModel`] backs hosted OpenAI *and* every OpenAI-compatible
//! endpoint (Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
//! Mistral) via the preset constructors below, so the sub-agent / sub-graph
//! layers above never need to know which provider answered.
//!
//! [`OpenAiModel`] implements [`ChatModel`] against the hosted OpenAI Chat
//! Completions endpoint (`POST {base_url}/chat/completions`). It translates the
//! provider-neutral [`ModelRequest`] into OpenAI's JSON wire format (see
//! [`types`]), performs the HTTP call with `reqwest`, and maps the response back
//! into a [`ModelResponse`] with a fully-populated [`AssistantMessage`],
//! [`ToolCall`]s, [`Usage`], and finish reason.
//!
//! The wire (de)serialization shapes live in [`types`]; this module owns only
//! the translation logic and the HTTP transport, keeping OpenAI-specific JSON
//! out of the rest of the harness.
//!
//! Local OpenAI-compatible runtimes (LM Studio, llama.cpp server, …) reject a
//! named `tool_choice` object and a `json_object` response format with an HTTP
//! 400. The transport degrades both to shapes they accept — `tool_choice`
//! `"required"` with the `tools` array filtered to the named tool, and a
//! permissive `json_schema` — either eagerly via
//! [`OpenAiModel::with_named_tool_choice`] / [`OpenAiModel::with_json_object_format`]
//! or automatically as a single retry when a 400 body implicates the shape. See
//! the module `README.md` "Local-server compatibility" section.
//!
//! Some OpenAI-compatible proxies go further and refuse unary calls entirely,
//! answering `stream: false` with an HTTP 400/422 such as
//! `{"detail":"Stream must be set to true"}`. [`ChatModel::invoke`] recognises
//! that family of rejections (`is_stream_required_error` in `transport`), folds
//! the SSE stream into a single [`ModelResponse`] instead, and **latches** the
//! constraint on the instance so only the first call pays the rejected round
//! trip. Declare it up front with
//! [`OpenAiModel::with_requires_streaming`] to skip even that one.
//!
//! # Example
//!
//! ```no_run
//! use tinyagents::harness::providers::openai::OpenAiModel;
//!
//! # fn main() -> tinyagents::Result<()> {
//! // Reads OPENAI_API_KEY (and optional OPENAI_MODEL / OPENAI_BASE_URL).
//! let model = OpenAiModel::from_env()?;
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod types;

pub use types::*;

use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{Map, Value, json};

use crate::error::{Result, TinyAgentsError};
use crate::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use crate::harness::model::{
    ChatModel, Modalities, ModelProfile, ModelRequest, ModelResponse, ModelStatus, ModelStream,
    ModelStreamItem, ProviderError, ResponseFormat, ToolChoice,
};
use crate::harness::tool::{ToolCall, ToolDelta};
use crate::harness::usage::Usage;

use super::ProviderSpec;

/// Default model id used when neither the request nor the builder override it.
const DEFAULT_MODEL: &str = "gpt-4.1-mini";
/// Default OpenAI API base URL.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Sane default TCP connect timeout applied to every call. Bounds connection
/// establishment without capping the (potentially long) response body, so it is
/// safe for streaming too.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;
/// Default overall timeout applied to unary calls when the request does not set
/// [`ModelRequest::timeout_ms`]. Streaming calls get no overall cap by default.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 600;

mod convert;
mod reasoning_tags;
mod relaxed_json;
mod responses;
mod sse;
mod transport;

pub use reasoning_tags::ReasoningTagExtraction;
pub use transport::{AuthStyle, OpenAiModel};

use convert::*;
use reasoning_tags::*;
use sse::*;
#[cfg(test)]
use transport::{
    Degrade, auth_headers, degrade_for_400, effective_temperature, glob_match,
    is_stream_required_error, merge_provider_options, merge_system_into_user, request_timeout,
};

#[cfg(test)]
mod test;
