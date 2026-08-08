//! Explicit message trimming, summarization, and compression policies.
//!
//! In the recursive architecture this is the harness's direct answer to
//! "context rot": context-window-aware gating ([`SummarizationPolicy`]) decides
//! *when* a run's transcript has grown large enough to compress, and the
//! trimming/summarization primitives decide *what* to keep verbatim versus fold
//! into a summary. This mirrors the recursive-language-model idea of treating a
//! long prompt as something to decompose rather than stuff whole into one
//! window — keeping each (sub-)agent's effective context bounded as runs nest.
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

pub mod pairing;
mod render;
mod trim;
mod types;

pub use pairing::{
    advance_past_orphan_tools, find_safe_cutoff_point, is_tool_calling_assistant,
    retract_orphan_tool_calls, tool_pairing_is_intact,
};
pub use render::render_message_for_summary;
pub use trim::{trim_messages, trim_messages_to_token_budget_with, trim_messages_with};
pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::harness::message::{Message, estimate_slice_tokens};
use async_trait::async_trait;
use trim::partition_system;

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

// ---------------------------------------------------------------------------
// ConcatSummarizer
// ---------------------------------------------------------------------------

#[async_trait]
impl Summarizer for ConcatSummarizer {
    /// Summarize `messages` by concatenating them into a single system message.
    ///
    /// Each message is rendered by [`render_message_for_summary`] and prefixed
    /// with a positional id, so the summary is human-readable. No LLM call is
    /// made.
    ///
    /// # Why not `Message::text()`
    ///
    /// [`Message::text`] returns only visible text blocks, so an assistant turn
    /// that only called tools, a JSON tool result, and model reasoning all
    /// render as an empty string. Because this is the crate's **default**
    /// summarizer, building it on `text()` meant that out of the box,
    /// compaction of a tool-driven run replaced the real history with a column
    /// of bare role labels. Rendering tool calls, tool results, and reasoning
    /// keeps the compacted transcript worth keeping — the same reason LangChain
    /// summarizes through `get_buffer_string(..., format="xml")`.
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

        let original_token_estimate = estimate_slice_tokens(messages);

        let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
        parts.push("=== Conversation Summary ===".to_string());

        let source_ids: Vec<String> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let id = format!("msg-{i}");
                parts.push(format!("[{id}] {}", render_message_for_summary(msg)));
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
        let tokens = estimate_slice_tokens(messages);
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
    ///
    /// # Tool-call pairing
    ///
    /// The split point is **not** a blind `len - keep_last` index. That index
    /// routinely lands between an assistant tool-call turn and the tool results
    /// answering it, putting the assistant message in `to_summarize` and its
    /// `tool` messages in `to_keep`; the rebuilt request then opens with a
    /// `role:"tool"` message that answers nothing, which OpenAI rejects with a
    /// `400` and Anthropic rejects as a `tool_result` with no matching
    /// `tool_use`. Since only long tool-driven runs reach a compaction
    /// threshold at all, the blind index failed on essentially every run that
    /// used it.
    ///
    /// [`find_safe_cutoff_point`] moves the split back to include the owning
    /// assistant turn (or, for a transcript with no such turn, forward past the
    /// unpairable results), so `to_keep` is always a slice a provider accepts.
    /// `keep_last` is therefore a **minimum**, not an exact count.
    pub fn plan(&self, messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
        let (system, non_system) = partition_system(messages);

        if non_system.len() <= self.keep_last {
            // Nothing old enough to summarize; keep everything.
            let mut to_keep = system;
            to_keep.extend(non_system);
            return (Vec::new(), to_keep);
        }

        let requested_split = non_system.len() - self.keep_last;
        let split = find_safe_cutoff_point(&non_system, requested_split);
        if split != requested_split {
            tracing::debug!(
                "[summarization::plan] keep_last={} moved split {requested_split} -> {split} to preserve tool-call pairing",
                self.keep_last
            );
        }
        let to_summarize = non_system[..split].to_vec();
        let to_keep_recent = non_system[split..].to_vec();

        let mut to_keep = system;
        to_keep.extend(to_keep_recent);

        (to_summarize, to_keep)
    }
}

#[cfg(test)]
mod test;
