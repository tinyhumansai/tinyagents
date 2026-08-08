//! Types for explicit message trimming, summarization, and compression policies.
//!
//! All policy decisions — when to summarize, what to keep, and what provenance
//! to record — are expressed as data types so they can be inspected, tested,
//! and audited without coupling to any particular LLM provider.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::harness::message::Message;

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// A cheap heuristic estimate of the number of tokens in a piece of text.
///
/// The value is derived by [`estimate_tokens`] and should be treated as an
/// approximation only — it does not use a real tokenizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEstimate {
    /// The estimated token count.
    pub tokens: u64,
}

// ---------------------------------------------------------------------------
// Trim strategy
// ---------------------------------------------------------------------------

/// How to trim a message list when it grows too long.
///
/// Trimming is a best-effort, synchronous operation that does not call an LLM.
/// It simply drops messages from the slice according to the chosen rule.
/// System messages are never dropped by default unless the strategy is
/// `MaxTokens` and the budget is so tight that even system content must be
/// shed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimStrategy {
    /// Retain only the last `n` non-system messages (all system messages are
    /// kept in addition).
    KeepLast(usize),

    /// Retain the first `first` and last `last` non-system messages (all
    /// system messages are kept in addition).
    KeepFirstAndLast {
        /// Number of non-system messages to keep from the front.
        first: usize,
        /// Number of non-system messages to keep from the back.
        last: usize,
    },

    /// Drop messages from the front until the estimated token count of the
    /// remaining slice is at or below `limit`.  System messages are dropped
    /// last — only when all other messages have already been removed and the
    /// budget is still exceeded.
    MaxTokens(u64),
}

/// The role of a [`Message`], as a standalone value for role-boundary
/// predicates.
///
/// Used by [`TrimOptions::start_on`] / [`TrimOptions::end_on`], the crate's
/// port of LangChain core's `trim_messages(start_on=…, end_on=…)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// [`Message::System`].
    System,
    /// [`Message::User`].
    User,
    /// [`Message::Assistant`].
    Assistant,
    /// [`Message::Tool`].
    Tool,
}

impl MessageRole {
    /// Returns the role of `message`.
    pub fn of(message: &Message) -> Self {
        match message {
            Message::System(_) => MessageRole::System,
            Message::User(_) => MessageRole::User,
            Message::Assistant(_) => MessageRole::Assistant,
            Message::Tool(_) => MessageRole::Tool,
        }
    }
}

/// Knobs for [`trim_messages_with`][crate::harness::summarization::trim_messages_with].
///
/// [`Default`] is the safe configuration: tool-call pairing is repaired and no
/// role boundary is imposed. [`trim_messages`][crate::harness::summarization::trim_messages]
/// is exactly `trim_messages_with(.., &TrimOptions::default())`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrimOptions {
    /// Repair the strategy's cut point so it never splits an assistant
    /// tool-call turn from the tool results answering it, and never leaves an
    /// assistant tool call unanswered.
    ///
    /// **Defaults to on, and should stay on.** Turning it off restores the
    /// pre-repair behaviour, which produces transcripts that providers reject
    /// outright: OpenAI `400`s on a `role:"tool"` with no preceding
    /// `tool_calls`, and Anthropic rejects a `tool_result` with no matching
    /// `tool_use`. It exists for callers that reconstruct pairing themselves
    /// afterwards (and for tests that need to observe the unrepaired cut).
    ///
    /// `#[serde(default = …)]` so a persisted `TrimOptions` written before this
    /// field existed — or one that simply omits it — still deserialises to the
    /// safe value rather than to `false`.
    #[serde(default = "default_repair_tool_pairs")]
    pub repair_tool_pairs: bool,

    /// Drop messages from the **front** of the retained slice until it begins
    /// on one of these roles. `None` (the default) imposes no boundary.
    ///
    /// This is the mechanism LangChain core's `trim_messages` documents as the
    /// caller's responsibility for provider compatibility — e.g.
    /// `start_on: [User]` for providers that require the first non-system turn
    /// to be a user turn. It composes with, and does not replace,
    /// [`repair_tool_pairs`][Self::repair_tool_pairs].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_on: Option<Vec<MessageRole>>,

    /// Drop messages from the **back** of the retained slice until it ends on
    /// one of these roles. `None` (the default) imposes no boundary.
    ///
    /// `end_on: [Tool, Assistant]` is the usual setting for "do not end the
    /// prompt mid-tool-call".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_on: Option<Vec<MessageRole>>,
}

/// The default for [`TrimOptions::repair_tool_pairs`] (`true`).
pub(crate) fn default_repair_tool_pairs() -> bool {
    true
}

impl Default for TrimOptions {
    /// The safe configuration: pairing repaired, no role boundary.
    ///
    /// Hand-written rather than derived precisely because a derived `Default`
    /// would set `repair_tool_pairs` to `false` — silently restoring the
    /// provider-`400` behaviour this type exists to prevent.
    fn default() -> Self {
        Self {
            repair_tool_pairs: default_repair_tool_pairs(),
            start_on: None,
            end_on: None,
        }
    }
}

impl TrimOptions {
    /// Requires the retained slice to begin on one of `roles`.
    pub fn starting_on(mut self, roles: impl IntoIterator<Item = MessageRole>) -> Self {
        self.start_on = Some(roles.into_iter().collect());
        self
    }

    /// Requires the retained slice to end on one of `roles`.
    pub fn ending_on(mut self, roles: impl IntoIterator<Item = MessageRole>) -> Self {
        self.end_on = Some(roles.into_iter().collect());
        self
    }

    /// Disables tool-call pairing repair. See
    /// [`repair_tool_pairs`][Self::repair_tool_pairs] before reaching for this.
    pub fn without_pair_repair(mut self) -> Self {
        self.repair_tool_pairs = false;
        self
    }
}

/// Options for order-preserving token-budget trimming with a caller-supplied
/// message estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenTrimPolicy {
    /// Maximum estimated tokens retained after trimming.
    pub limit: u64,
    /// Never evict system messages, even when they alone exceed `limit`.
    pub preserve_system: bool,
    /// After eviction, discard leading tool results that no longer have their
    /// preceding assistant tool call. System messages may precede the first
    /// retained conversational message.
    pub drop_leading_orphan_tools: bool,
}

impl TokenTrimPolicy {
    /// Creates a strict token-budget policy. System messages may be dropped as
    /// a last resort and no structural cleanup is applied.
    pub const fn strict(limit: u64) -> Self {
        Self {
            limit,
            preserve_system: false,
            drop_leading_orphan_tools: false,
        }
    }

    /// Keeps system instructions even when they exceed the configured budget.
    pub const fn preserve_system(mut self) -> Self {
        self.preserve_system = true;
        self
    }

    /// Drops tool results left at the leading conversational boundary after
    /// their assistant tool-call message was evicted.
    pub const fn drop_leading_orphan_tools(mut self) -> Self {
        self.drop_leading_orphan_tools = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Compression provenance
// ---------------------------------------------------------------------------

/// Metadata that records *why* a set of messages was removed or replaced by a
/// summary.
///
/// Provenance is required by the summarization spec so that users can audit
/// what was compressed and under which policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionProvenance {
    /// Identifiers of the messages that were replaced.  When the underlying
    /// [`Message`] type carries no id the caller should supply synthetic
    /// positional ids such as `"msg-0"`, `"msg-1"`, …
    pub source_ids: Vec<String>,

    /// Estimated token count of the original messages before compression.
    pub original_token_estimate: u64,

    /// Estimated token count of the summary that replaced them.
    pub summary_token_estimate: u64,

    /// Human-readable reason describing the policy decision that triggered
    /// compression (e.g. `"token budget exceeded threshold 4096"`).
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Summary record
// ---------------------------------------------------------------------------

/// A single summary produced by a [`Summarizer`], together with the provenance
/// that explains which messages it replaced and why.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryRecord {
    /// The summary itself, expressed as a [`Message`].  Implementations
    /// typically produce a [`Message::System`] so the model treats the
    /// condensed history as background context.
    pub summary: Message,

    /// Provenance metadata linking this summary back to its source messages.
    pub provenance: CompressionProvenance,
}

// ---------------------------------------------------------------------------
// Summarizer trait
// ---------------------------------------------------------------------------

/// Async trait for turning a slice of messages into a [`SummaryRecord`].
///
/// Implementations range from deterministic concatenation stubs (see
/// [`ConcatSummarizer`]) to real LLM-backed compressors.  The trait is
/// object-safe so harness layers can store `Box<dyn Summarizer>`.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Condense `messages` into a single [`SummaryRecord`].
    ///
    /// Returns `Err` when summarization fails (for example, when an LLM call
    /// is rejected).  The caller is responsible for deciding how to handle the
    /// error — fall back to trimming, propagate, or surface a context error.
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord>;
}

// ---------------------------------------------------------------------------
// ConcatSummarizer
// ---------------------------------------------------------------------------

/// A deterministic, LLM-free summarizer for testing and fallback use.
///
/// It concatenates the text of all provided messages into a single system
/// message, prefixed by a header.  No external call is made; the result is
/// fully reproducible.
///
/// # Provenance
///
/// Because [`Message`] carries no stable id, `ConcatSummarizer` assigns
/// synthetic positional ids of the form `"msg-0"`, `"msg-1"`, … based on
/// the index of each message within the supplied slice.
#[derive(Clone, Debug, Default)]
pub struct ConcatSummarizer;

// ---------------------------------------------------------------------------
// Summarization policy
// ---------------------------------------------------------------------------

/// Policy describing *when* to summarize and *how much* to retain verbatim.
///
/// The policy does not perform summarization itself — it only decides whether
/// summarization is needed and splits the message list accordingly.  Pass the
/// split output to a [`Summarizer`] implementation.
///
/// # Context-window awareness
///
/// When [`context_window`][Self::context_window] is set (typically from a
/// model's [`ModelProfile::max_input_tokens`]), the policy only triggers once
/// the estimated tokens reach [`threshold_fraction`][Self::threshold_fraction]
/// of that window (default `0.9`, i.e. 90%). When `context_window` is `None`
/// the policy falls back to the raw [`trigger_tokens`][Self::trigger_tokens]
/// threshold, preserving the original behaviour.
///
/// [`ModelProfile::max_input_tokens`]: crate::harness::model::ModelProfile::max_input_tokens
///
/// # Example
///
/// ```
/// use tinyagents::harness::message::Message;
/// use tinyagents::harness::summarization::SummarizationPolicy;
///
/// let policy = SummarizationPolicy {
///     trigger_tokens: 2000,
///     keep_last: 4,
///     ..Default::default()
/// };
/// let msgs = vec![Message::user("hello"), Message::assistant("world")];
/// assert!(!policy.should_summarize(&msgs));
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummarizationPolicy {
    /// Estimated token threshold above which summarization is triggered.
    ///
    /// Used only when [`context_window`][Self::context_window] is `None`. When
    /// the total estimated tokens of all messages exceeds this value,
    /// [`should_summarize`][SummarizationPolicy::should_summarize] returns
    /// `true`.
    pub trigger_tokens: u64,

    /// Number of the most-recent non-system messages to keep verbatim after
    /// summarization.  System messages are always kept verbatim regardless of
    /// this setting.
    pub keep_last: usize,

    /// Maximum input (context) tokens of the target model, when known.
    ///
    /// When set, [`should_summarize`][SummarizationPolicy::should_summarize]
    /// triggers only once the estimated tokens reach
    /// [`threshold_fraction`][Self::threshold_fraction] of this window. When
    /// `None`, the policy falls back to the raw
    /// [`trigger_tokens`][Self::trigger_tokens] threshold.
    #[serde(default)]
    pub context_window: Option<u64>,

    /// Fraction of [`context_window`][Self::context_window] that must be
    /// reached before summarization triggers. Defaults to `0.9` (90%). Ignored
    /// when `context_window` is `None`.
    #[serde(default = "default_threshold_fraction")]
    pub threshold_fraction: f64,
}

/// The default [`SummarizationPolicy::threshold_fraction`] (90% of the context
/// window).
pub(crate) fn default_threshold_fraction() -> f64 {
    0.9
}

impl Default for SummarizationPolicy {
    fn default() -> Self {
        Self {
            trigger_tokens: 0,
            keep_last: 0,
            context_window: None,
            threshold_fraction: default_threshold_fraction(),
        }
    }
}
