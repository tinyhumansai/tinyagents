//! Providers module types.
//!
//! All public and internal types for the `providers` module live here.
//! Implementations and trait-impls are in `mod.rs`.

use std::sync::Mutex;

use serde_json::Value;

use crate::harness::model::ModelResponse;

// ---------------------------------------------------------------------------
// Internal behavior enum
// ---------------------------------------------------------------------------

/// The scripted behavior that drives a [`MockModel`] invocation.
///
/// This is an internal type — callers interact with [`MockModel`]'s named
/// constructors instead.
pub(crate) enum MockBehavior {
    /// Echoes the text of the last [`Message::User`][crate::harness::message::Message]
    /// in the request back as the assistant reply.
    Echo,

    /// Always returns a fixed assistant text string, regardless of input.
    Constant(String),

    /// Returns responses from a pre-loaded vector in order, cycling back to
    /// the start when all responses have been consumed. See
    /// [`MockModel::with_responses`] for details.
    Scripted(Vec<ModelResponse>),

    /// Returns a single tool-call request for the named tool.  The
    /// `AssistantMessage` carries the call in its `tool_calls` field and the
    /// `finish_reason` is `"tool_calls"`.
    ToolCall {
        /// Name of the tool the model is requesting.
        name: String,
        /// JSON arguments to supply to the tool.
        arguments: Value,
    },
}

// ---------------------------------------------------------------------------
// Internal mutable state (behind a Mutex for Send + Sync)
// ---------------------------------------------------------------------------

/// Mutable runtime state for [`MockModel`], protected by a `Mutex`.
#[derive(Default)]
pub(crate) struct MockInner {
    /// Total number of [`ChatModel::invoke`][crate::harness::model::ChatModel]
    /// calls made so far (not counting `stream` calls that delegate to invoke).
    pub(crate) call_count: u64,
    /// Next index into the scripted response list (used by [`MockBehavior::Scripted`]).
    pub(crate) scripted_index: usize,
}

// ---------------------------------------------------------------------------
// MockModel
// ---------------------------------------------------------------------------

/// A deterministic, in-process chat model for tests and harness development.
///
/// `MockModel` implements [`ChatModel<State>`][crate::harness::model::ChatModel]
/// generically for *any* `State: Send + Sync`.  It never makes network calls
/// and has no external dependencies.
///
/// # Constructors
///
/// | Constructor | Behaviour |
/// |---|---|
/// | [`MockModel::echo`] | Echoes the last user message text back. |
/// | [`MockModel::constant`] | Always returns the same fixed string. |
/// | [`MockModel::with_responses`] | Returns scripted [`ModelResponse`]s in order, cycling when exhausted. |
/// | [`MockModel::with_tool_call`] | Always returns one tool-call request. |
///
/// # Streaming
///
/// The [`ChatModel::stream`][crate::harness::model::ChatModel] override
/// internally calls [`ChatModel::invoke`] and replays the response as a real
/// [`ModelStream`][crate::harness::model::ModelStream]: a
/// [`Started`][crate::harness::model::ModelStreamItem::Started] item, one or two
/// [`MessageDelta`][crate::harness::model::ModelStreamItem::MessageDelta] items
/// (text split into two equal-sized halves by Unicode scalar value), and a
/// terminal [`Completed`][crate::harness::model::ModelStreamItem::Completed]
/// item carrying the full response. This lets downstream streaming consumers be
/// exercised without any real streaming infrastructure. When the response
/// contains no text (e.g. a tool-call response), a single empty text delta is
/// emitted before completion.
///
/// # Usage estimates
///
/// Every response carries a deterministic [`Usage`][crate::harness::usage::Usage]
/// derived from character counts:
/// - `input_tokens` ≈ total characters in all request messages ÷ 4
/// - `output_tokens` ≈ total characters in the response text ÷ 4 (minimum 1)
///
/// This gives cost-accounting code realistic non-zero values to work with.
///
/// # Placement of real providers
///
/// Real network-backed providers are gated behind Cargo features and live in
/// sub-modules alongside this one:
///
/// ```text
/// // #[cfg(feature = "openai")]   pub mod openai;
/// // #[cfg(feature = "anthropic")] pub mod anthropic;
/// // #[cfg(feature = "ollama")]   pub mod ollama;
/// ```
///
/// Add the feature flag to `Cargo.toml` and implement
/// [`ChatModel`][crate::harness::model::ChatModel] in the corresponding module.
/// No changes to `mod.rs` or `harness/mod.rs` are needed beyond enabling the
/// `pub mod` declaration.
pub struct MockModel {
    pub(crate) behavior: MockBehavior,
    pub(crate) inner: Mutex<MockInner>,
}
