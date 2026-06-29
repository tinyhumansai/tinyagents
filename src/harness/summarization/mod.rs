//! Explicit message trimming, summarization, and compression policies.
//!
//! This module provides:
//!
//! - [`estimate_tokens`] — cheap heuristic token counter (chars / 4).
//! - [`trim_messages`] — synchronous, LLM-free slice reduction via [`TrimStrategy`].
//! - [`Summarizer`] — async trait for condensing messages into a [`SummaryRecord`].
//! - [`ConcatSummarizer`] — deterministic concatenation stand-in (no LLM).
//! - [`SummarizationPolicy`] — decides when to summarize and how to split the slice.
//!
//! All policy decisions are explicit data types, never hidden behaviour. Callers
//! choose when to call, what to pass, and how to handle the result.

mod types;

pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::harness::message::Message;
use async_trait::async_trait;

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in `text` using a cheap character-count
/// heuristic: `tokens ≈ chars / 4`.
///
/// This is *not* a real tokenizer.  Real models use sub-word tokenizers whose
/// output depends on vocabulary and input encoding.  This function is suitable
/// for quick budget checks where a ±30% error margin is acceptable.
///
/// Returns at least `1` for any non-empty input to avoid zero-token
/// misclassifications.
pub fn estimate_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    // Heuristic: approximately 4 characters per token on average for English
    // prose and code. Clamp to at least 1 for non-empty strings.
    if chars == 0 { 0 } else { (chars / 4).max(1) }
}

/// Estimate the total tokens for a [`Message`] by running [`estimate_tokens`]
/// over its concatenated text.
fn message_token_estimate(msg: &Message) -> u64 {
    estimate_tokens(&msg.text())
}

/// Estimate the total tokens for a slice of messages.
fn slice_token_estimate(messages: &[Message]) -> u64 {
    messages.iter().map(message_token_estimate).sum()
}

// ---------------------------------------------------------------------------
// Trimming
// ---------------------------------------------------------------------------

/// Partition `messages` into system and non-system messages, preserving order.
///
/// Returns `(system, non_system)`.
fn partition_system(messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
    let system = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .cloned()
        .collect();
    let non_system = messages
        .iter()
        .filter(|m| !matches!(m, Message::System(_)))
        .cloned()
        .collect();
    (system, non_system)
}

/// Trim a message slice according to `strategy`, returning the retained subset.
///
/// System messages are preserved by default:
///
/// - [`TrimStrategy::KeepLast`] and [`TrimStrategy::KeepFirstAndLast`] always
///   keep all system messages and apply the rule only to non-system messages.
/// - [`TrimStrategy::MaxTokens`] drops non-system messages first (from the
///   front) and only starts dropping system messages if the budget still
///   cannot be met after all non-system messages are removed.
///
/// The returned `Vec<Message>` preserves the relative order of messages as
/// they appeared in the input.
pub fn trim_messages(messages: &[Message], strategy: &TrimStrategy) -> Vec<Message> {
    match strategy {
        TrimStrategy::KeepLast(n) => {
            let (system, non_system) = partition_system(messages);
            let keep_start = non_system.len().saturating_sub(*n);
            let mut result = system;
            result.extend_from_slice(&non_system[keep_start..]);
            result
        }

        TrimStrategy::KeepFirstAndLast { first, last } => {
            let (system, non_system) = partition_system(messages);
            let len = non_system.len();
            let first = *first;
            let last = *last;

            let mut result = system;
            if first + last >= len {
                // No overlap: keep everything.
                result.extend(non_system);
            } else {
                result.extend_from_slice(&non_system[..first]);
                result.extend_from_slice(&non_system[len - last..]);
            }
            result
        }

        TrimStrategy::MaxTokens(limit) => {
            let (system, non_system) = partition_system(messages);
            let limit = *limit;

            // Try dropping non-system messages from the front one at a time.
            let mut candidate: Vec<Message> = non_system;
            while !candidate.is_empty() {
                let total = slice_token_estimate(&system) + slice_token_estimate(&candidate);
                if total <= limit {
                    break;
                }
                candidate.remove(0);
            }

            // If we're still over budget, start dropping system messages from
            // the front as a last resort.
            let mut sys_candidate = system;
            while !sys_candidate.is_empty() {
                let total = slice_token_estimate(&sys_candidate) + slice_token_estimate(&candidate);
                if total <= limit {
                    break;
                }
                sys_candidate.remove(0);
            }

            let mut result = sys_candidate;
            result.extend(candidate);
            result
        }
    }
}

// ---------------------------------------------------------------------------
// ConcatSummarizer
// ---------------------------------------------------------------------------

#[async_trait]
impl Summarizer for ConcatSummarizer {
    /// Summarize `messages` by concatenating their text content into a single
    /// system message.
    ///
    /// Each message's text is prefixed by a role label and positional id so
    /// the summary is human-readable.  No LLM call is made.
    ///
    /// # Provenance
    ///
    /// Synthetic positional ids `"msg-0"`, `"msg-1"`, … are assigned because
    /// [`Message`] carries no stable identifier.  The `reason` field records
    /// that a `ConcatSummarizer` was used.
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        if messages.is_empty() {
            return Err(TinyAgentsError::Validation(
                "cannot summarize an empty message list".into(),
            ));
        }

        let original_token_estimate = slice_token_estimate(messages);

        let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
        parts.push("=== Conversation Summary ===".to_string());

        let source_ids: Vec<String> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let role = match msg {
                    Message::System(_) => "system",
                    Message::User(_) => "user",
                    Message::Assistant(_) => "assistant",
                    Message::Tool(_) => "tool",
                };
                let id = format!("msg-{i}");
                parts.push(format!("[{id}] {role}: {}", msg.text()));
                id
            })
            .collect();

        let summary_text = parts.join("\n");
        let summary_token_estimate = estimate_tokens(&summary_text);

        let summary = Message::system(summary_text);
        let provenance = CompressionProvenance {
            source_ids,
            original_token_estimate,
            summary_token_estimate,
            reason: "ConcatSummarizer: messages concatenated verbatim (no LLM call)".to_string(),
        };

        Ok(SummaryRecord {
            summary,
            provenance,
        })
    }
}

// ---------------------------------------------------------------------------
// SummarizationPolicy
// ---------------------------------------------------------------------------

impl SummarizationPolicy {
    /// Builds a policy from a model [`ModelProfile`], reading its
    /// [`max_input_tokens`][crate::harness::model::ModelProfile::max_input_tokens]
    /// as the context window and using `threshold` as the trigger fraction.
    ///
    /// All other fields take their [`Default`] values (`trigger_tokens = 0`,
    /// `keep_last = 0`). Chain [`with_threshold_fraction`][Self::with_threshold_fraction]
    /// or set `keep_last` afterwards to tune retention. When the profile does
    /// not advertise `max_input_tokens` the resulting `context_window` is
    /// `None`, so [`should_summarize`][Self::should_summarize] falls back to the
    /// raw `trigger_tokens` threshold.
    pub fn from_profile(profile: &crate::harness::model::ModelProfile, threshold: f64) -> Self {
        Self {
            context_window: profile.max_input_tokens,
            threshold_fraction: threshold,
            ..Self::default()
        }
    }

    /// Sets the context window (the model's maximum input tokens) and returns
    /// the updated policy. Enables context-window-aware triggering.
    pub fn with_context_window(mut self, max_input_tokens: u64) -> Self {
        self.context_window = Some(max_input_tokens);
        self
    }

    /// Sets the [`threshold_fraction`][Self::threshold_fraction] and returns the
    /// updated policy.
    pub fn with_threshold_fraction(mut self, fraction: f64) -> Self {
        self.threshold_fraction = fraction;
        self
    }

    /// Returns the effective token budget at which summarization triggers.
    ///
    /// When [`context_window`][Self::context_window] is `Some(window)`, the
    /// budget is `floor(window * threshold_fraction)`. When it is `None`, the
    /// budget is the raw [`trigger_tokens`][Self::trigger_tokens].
    pub fn trigger_budget(&self) -> u64 {
        match self.context_window {
            Some(window) => (window as f64 * self.threshold_fraction) as u64,
            None => self.trigger_tokens,
        }
    }

    /// Returns `true` when the estimated total tokens of `messages` reach the
    /// summarization threshold.
    ///
    /// - When [`context_window`][Self::context_window] is set, returns `true`
    ///   once the estimate is **at or above** `context_window *
    ///   threshold_fraction` (the window-usage gate).
    /// - When `context_window` is `None`, falls back to the original behaviour:
    ///   returns `true` when the estimate **exceeds**
    ///   [`trigger_tokens`][Self::trigger_tokens].
    pub fn should_summarize(&self, messages: &[Message]) -> bool {
        let tokens = slice_token_estimate(messages);
        match self.context_window {
            Some(_) => tokens >= self.trigger_budget(),
            None => tokens > self.trigger_tokens,
        }
    }

    /// Split `messages` into `(to_summarize, to_keep)`.
    ///
    /// `to_keep` always contains:
    /// - All system messages (verbatim, preserving order relative to each other).
    /// - The last [`keep_last`][Self::keep_last] non-system messages.
    ///
    /// `to_summarize` contains the remaining non-system messages that precede
    /// the kept window.  If there are fewer non-system messages than
    /// `keep_last`, `to_summarize` is empty and all messages are placed in
    /// `to_keep`.
    ///
    /// System messages are never placed in `to_summarize` — they must be kept
    /// verbatim to avoid losing persistent instructions.
    pub fn plan(&self, messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
        let (system, non_system) = partition_system(messages);

        if non_system.len() <= self.keep_last {
            // Nothing old enough to summarize; keep everything.
            let mut to_keep = system;
            to_keep.extend(non_system);
            return (Vec::new(), to_keep);
        }

        let split = non_system.len() - self.keep_last;
        let to_summarize = non_system[..split].to_vec();
        let to_keep_recent = non_system[split..].to_vec();

        let mut to_keep = system;
        to_keep.extend(to_keep_recent);

        (to_summarize, to_keep)
    }
}

#[cfg(test)]
mod test;
