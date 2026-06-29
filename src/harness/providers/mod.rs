//! Feature-gated model provider integrations — the leaves of the recursion.
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
//! | Provider | Feature flag | Status |
//! |---|---|---|
//! | [`MockModel`] | *(always available)* | Implemented — deterministic, no network |
//! | `openai` (and OpenAI-compatible endpoints) | `openai` | Implemented |
//!
//! [`MockModel`] is always compiled and needs no network, keeping the default
//! build offline and deterministic. The `openai` module is gated behind the
//! `openai` Cargo feature and additionally serves every OpenAI-compatible
//! endpoint (Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
//! Mistral) through the same Chat Completions wire format.
//!
//! To add a provider with a different wire protocol, enable a new feature in
//! `Cargo.toml` and add the corresponding gated module declaration:
//!
//! ```text
//! #[cfg(feature = "openai")]    pub mod openai;
//! // #[cfg(feature = "anthropic")] pub mod anthropic;
//! // #[cfg(feature = "ollama")]    pub mod ollama;
//! ```

mod types;

// --- real provider integrations (gated behind Cargo features) ---
#[cfg(feature = "openai")]
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

// ---------------------------------------------------------------------------
// Token-estimation helpers
// ---------------------------------------------------------------------------

/// Estimates the number of input tokens from a model request.
///
/// Uses the heuristic of 1 token ≈ 4 characters of UTF-8 text.
fn estimate_input_tokens(request: &ModelRequest) -> u64 {
    let total_chars: u64 = request.messages.iter().map(|m| m.text().len() as u64).sum();
    total_chars.div_ceil(4)
}

/// Estimates output tokens from the response text.
///
/// Uses the heuristic of 1 token ≈ 4 characters. Returns at least 1.
fn estimate_output_tokens(text: &str) -> u64 {
    let chars = text.len() as u64;
    std::cmp::max(1, chars.div_ceil(4))
}

// ---------------------------------------------------------------------------
// MockModel constructors
// ---------------------------------------------------------------------------

impl MockModel {
    /// Creates a `MockModel` that echoes the last user message back as the
    /// assistant reply.
    ///
    /// If the request contains no user message, the reply is an empty string.
    pub fn echo() -> Self {
        Self {
            behavior: MockBehavior::Echo,
            inner: std::sync::Mutex::new(MockInner::default()),
        }
    }

    /// Creates a `MockModel` that always returns the same fixed assistant text.
    pub fn constant(text: impl Into<String>) -> Self {
        Self {
            behavior: MockBehavior::Constant(text.into()),
            inner: std::sync::Mutex::new(MockInner::default()),
        }
    }

    /// Creates a `MockModel` that returns scripted responses in sequence.
    ///
    /// Responses are yielded one at a time in the order provided.  When all
    /// responses have been consumed the sequence **cycles back to the first
    /// response**, so the model never errors simply due to exhaustion.
    ///
    /// # Panics
    ///
    /// Panics at *construction time* if `responses` is empty, because an empty
    /// scripted model cannot produce any response.
    pub fn with_responses(responses: Vec<ModelResponse>) -> Self {
        assert!(
            !responses.is_empty(),
            "MockModel::with_responses: responses must not be empty"
        );
        Self {
            behavior: MockBehavior::Scripted(responses),
            inner: std::sync::Mutex::new(MockInner::default()),
        }
    }

    /// Creates a `MockModel` that always issues one tool-call request.
    ///
    /// The returned [`ModelResponse`] has:
    /// - An empty `content` block list (no text).
    /// - One [`ToolCall`] in `message.tool_calls`.
    /// - `finish_reason` set to `"tool_calls"`.
    ///
    /// `arguments` accepts anything that converts to a `serde_json::Value`
    /// (e.g. `serde_json::json!({...})`, a pre-built `Value`, or `Value::Null`).
    pub fn with_tool_call(name: impl Into<String>, arguments: impl Into<Value>) -> Self {
        Self {
            behavior: MockBehavior::ToolCall {
                name: name.into(),
                arguments: arguments.into(),
            },
            inner: std::sync::Mutex::new(MockInner::default()),
        }
    }

    /// Returns the total number of [`ChatModel::invoke`] calls made so far.
    ///
    /// `stream` calls that delegate to `invoke` also increment this counter.
    pub fn call_count(&self) -> u64 {
        self.inner
            .lock()
            .expect("MockModel inner state poisoned")
            .call_count
    }
}

// ---------------------------------------------------------------------------
// ChatModel<State> impl
// ---------------------------------------------------------------------------

/// Returns the shared permissive [`ModelProfile`] advertised by [`MockModel`].
///
/// `MockModel` can satisfy any reasonable [`CapabilitySet`][crate::harness::model::CapabilitySet]
/// and supports every structured-output strategy, so its profile enables all
/// capabilities.
fn mock_profile() -> &'static ModelProfile {
    static PROFILE: std::sync::OnceLock<ModelProfile> = std::sync::OnceLock::new();
    PROFILE.get_or_init(ModelProfile::permissive)
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for MockModel {
    /// Returns a permissive profile advertising every capability.
    fn profile(&self) -> Option<&ModelProfile> {
        Some(mock_profile())
    }

    /// Invokes the mock model and returns a deterministic response.
    ///
    /// Increments the internal call counter on every invocation.
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let call_id = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| TinyAgentsError::Model(format!("MockModel lock poisoned: {e}")))?;
            inner.call_count += 1;
            inner.call_count
        };

        let msg_id = format!("mock-msg-{call_id}");
        let input_tokens = estimate_input_tokens(&request);

        let response = match &self.behavior {
            MockBehavior::Echo => {
                let text = request
                    .messages
                    .iter()
                    .rev()
                    .find_map(|m| {
                        if let Message::User(_) = m {
                            Some(m.text())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                let output_tokens = estimate_output_tokens(&text);
                ModelResponse::assistant(text)
                    .with_usage(Usage::new(input_tokens, output_tokens))
                    .with_finish_reason("stop")
            }

            MockBehavior::Constant(text) => {
                let output_tokens = estimate_output_tokens(text);
                ModelResponse::assistant(text.clone())
                    .with_usage(Usage::new(input_tokens, output_tokens))
                    .with_finish_reason("stop")
            }

            MockBehavior::Scripted(responses) => {
                let index = {
                    let mut inner = self.inner.lock().map_err(|e| {
                        TinyAgentsError::Model(format!("MockModel lock poisoned: {e}"))
                    })?;
                    // We already incremented call_count above; derive index from
                    // call_count - 1 (0-based) cycling over the response list.
                    let idx = ((inner.call_count - 1) as usize) % responses.len();
                    inner.scripted_index = idx;
                    idx
                };
                responses[index].clone()
            }

            MockBehavior::ToolCall { name, arguments } => {
                let tool_call = ToolCall {
                    id: format!("mock-tool-{call_id}"),
                    name: name.clone(),
                    arguments: arguments.clone(),
                };
                let usage = Usage::new(input_tokens, 5);
                let message = AssistantMessage {
                    id: Some(msg_id.clone()),
                    content: Vec::new(),
                    tool_calls: vec![tool_call],
                    usage: Some(usage),
                };
                ModelResponse {
                    message,
                    usage: Some(usage),
                    finish_reason: Some("tool_calls".to_string()),
                    raw: None,
                    resolved_model: None,
                }
            }
        };

        // Stamp the message id on text-based responses for traceability.
        let mut response = response;
        if response.message.id.is_none() {
            response.message.id = Some(msg_id);
        }

        Ok(response)
    }

    /// Streams the model response as a real [`ModelStream`].
    ///
    /// Internally calls [`invoke`][MockModel::invoke], then replays the response
    /// as a [`ModelStreamItem::Started`] item, one or two
    /// [`ModelStreamItem::MessageDelta`] items, and a terminal
    /// [`ModelStreamItem::Completed`] carrying the full response. Text responses
    /// are split into two roughly equal halves (by Unicode scalar value) so
    /// streaming consumers observe multiple deltas without real network
    /// infrastructure. Tool-call (or otherwise text-less) responses emit a
    /// single empty text delta before completing.
    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let response = self.invoke(state, request).await?;
        let text = response.text();

        let mut items = vec![ModelStreamItem::Started];

        if text.is_empty() {
            items.push(ModelStreamItem::MessageDelta(MessageDelta::default()));
        } else {
            // Split by Unicode scalar values so we never bisect a multi-byte
            // char.
            let chars: Vec<char> = text.chars().collect();
            let mid = chars.len() / 2;
            let first: String = chars[..mid].iter().collect();
            let second: String = chars[mid..].iter().collect();
            items.push(ModelStreamItem::MessageDelta(MessageDelta {
                text: first,
                tool_call: None,
            }));
            items.push(ModelStreamItem::MessageDelta(MessageDelta {
                text: second,
                tool_call: None,
            }));
        }

        items.push(ModelStreamItem::Completed(response));
        Ok(Box::pin(futures::stream::iter(items)))
    }
}

// ---------------------------------------------------------------------------
// ContentBlock helper used in tests
// ---------------------------------------------------------------------------

impl MockModel {
    /// Convenience: builds a plain-text [`ModelResponse`] — useful for
    /// constructing scripted sequences in tests without importing the full
    /// harness message path.
    pub fn text_response(text: impl Into<String>) -> ModelResponse {
        let s = text.into();
        let output_tokens = estimate_output_tokens(&s);
        ModelResponse {
            message: AssistantMessage {
                id: None,
                content: vec![ContentBlock::Text(s)],
                tool_calls: Vec::new(),
                usage: Some(Usage::new(10, output_tokens)),
            },
            usage: Some(Usage::new(10, output_tokens)),
            finish_reason: Some("stop".to_string()),
            raw: None,
            resolved_model: None,
        }
    }
}

#[cfg(test)]
mod test;
