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
