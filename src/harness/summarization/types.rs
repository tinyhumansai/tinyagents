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
/// # Example
///
/// ```
/// use tinyagents::harness::message::Message;
/// use tinyagents::harness::summarization::SummarizationPolicy;
///
/// let policy = SummarizationPolicy { trigger_tokens: 2000, keep_last: 4 };
/// let msgs = vec![Message::user("hello"), Message::assistant("world")];
/// assert!(!policy.should_summarize(&msgs));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizationPolicy {
    /// Estimated token threshold above which summarization is triggered.
    ///
    /// When the total estimated tokens of all messages exceeds this value,
    /// [`should_summarize`][SummarizationPolicy::should_summarize] returns
    /// `true`.
    pub trigger_tokens: u64,

    /// Number of the most-recent non-system messages to keep verbatim after
    /// summarization.  System messages are always kept verbatim regardless of
    /// this setting.
    pub keep_last: usize,
}
