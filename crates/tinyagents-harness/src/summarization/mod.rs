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
//! - [`ModelSummarizer`] / [`FaultTolerantCachingSummarizer`] — LLM-backed summarization
//!   that never aborts a turn on a summarizer outage.
//! - [`TaskStateSummarizer`] — typed task-state checkpoints: facts copied from tool
//!   records plus one structured state-update call (see [`task_state`]).
//!
//! All policy decisions are explicit data types, never hidden behaviour. Callers
//! choose when to call, what to pass, and how to handle the result.

pub mod cap_summary;
mod checkpoint;
pub mod compaction;
mod compaction_carry;
pub(crate) mod dispatch;
mod file_ops;
mod model_summarizer;
pub mod pairing;
pub mod payload_summarizer;
mod render;
mod resilient;
mod response_overflow;
mod split_turn;
pub mod task_state;
mod trim;
mod types;

pub use cap_summary::{CapToolResult, build_cap_digest, cap_summary_fallback};
pub use checkpoint::{CHECKPOINT_PREFIX, checkpoint_body, checkpoint_message, is_checkpoint};
pub use compaction::{
    CompactionContext, CompactionDecision, CutPoint, OverflowClassifier, OverflowInfo,
    OverflowProbe, find_cut_point, summarize_kind_with_split, summarize_with_split,
};
pub use compaction_carry::{CompactionCarry, last_user_message};
pub use file_ops::{
    DefaultFileOpExtractor, FileOpExtractor, FileOperations, MAX_LISTED_FILES,
    append_file_sections, extract_file_operations, split_file_sections,
};
pub use model_summarizer::{
    DEFAULT_SUMMARIZE_KEEP_LAST, DEFAULT_SUMMARIZE_THRESHOLD_FRACTION,
    DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS, default_threshold_fraction_for, summarization_policy,
    summarization_policy_with, summarization_policy_with_tail,
};
pub use pairing::{
    advance_past_orphan_tools, find_safe_cutoff_point, is_tool_calling_assistant,
    retract_orphan_tool_calls, tool_pairing_is_intact,
};
pub use render::{render_message_for_summary, render_message_for_summary_with};
pub use resilient::FaultTolerantCachingSummarizer;
pub use response_overflow::{ResponseOverflowDetection, detect_response_overflow};
pub use split_turn::{SPLIT_TURN_HEADING, split_turn_start, summarize_split_turn};
pub use task_state::{DEFAULT_TASK_STATE_CHUNK_TOKENS, TaskLedger, TaskState, TaskStateSummarizer};
pub use trim::{trim_messages, trim_messages_to_token_budget_with, trim_messages_with};
pub use types::*;

use crate::error::{Result, TinyAgentsError};
use crate::token_estimation::estimate_slice_tokens;
use async_trait::async_trait;
use tinyinference_llm::message::Message;
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
            usage: None,
        })
    }

    /// Carries [`SummaryRequest::previous_summary`] forward verbatim ahead of
    /// the newly concatenated messages.
    ///
    /// Compaction is incremental: once a transcript has been folded, the next
    /// compaction hands the summarizer only the messages since that fold, plus
    /// the prior summary, which must survive into the result. The trait default
    /// would forward it as a leading system message rendered through
    /// [`render_message_for_summary`] and given a positional `msg-N` id, so it
    /// would read as one more transcript entry. This override keeps it verbatim
    /// ahead of the concatenation instead, and is the only forwarding: it
    /// replaces the default rather than adding to it.
    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        let mut record = self.summarize(&request.messages).await?;
        if let Some(previous) = request.previous_summary.as_deref() {
            let text = format!("{previous}\n{}", record.summary.text());
            record.provenance.summary_token_estimate = estimate_tokens(&text);
            record.summary = Message::system(text);
        }
        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// SummarizationPolicy
// ---------------------------------------------------------------------------

impl SummarizationPolicy {
    /// Builds a policy from a model [`ModelProfile`](tinyinference_llm::model::ModelProfile), reading its
    /// [`max_input_tokens`][tinyinference_llm::model::ModelProfile::max_input_tokens]
    /// as the context window and using `threshold` as the trigger fraction.
    ///
    /// All other fields take their [`Default`] values (`trigger_tokens = 0`,
    /// `keep_last = 0`). Chain [`with_threshold_fraction`][Self::with_threshold_fraction]
    /// or set `keep_last` afterwards to tune retention. When the profile does
    /// not advertise `max_input_tokens` the resulting `context_window` is
    /// `None`, so [`should_summarize`][Self::should_summarize] falls back to the
    /// raw `trigger_tokens` threshold.
    pub fn from_profile(profile: &tinyinference_llm::model::ModelProfile, threshold: f64) -> Self {
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
        self.should_summarize_with_tools(messages, &[])
    }

    /// [`Self::should_summarize`], charging the tool declarations too.
    ///
    /// Tool schemas are re-sent with every request, so a run offering thirty
    /// verbose tools is tens of thousands of tokens into its window before the
    /// first user message. Ignoring them makes the threshold fire late by
    /// exactly that much; this is the check the compression middleware uses.
    pub fn should_summarize_with_tools(
        &self,
        messages: &[Message],
        tools: &[tinyinference_llm::tool::ToolSchema],
    ) -> bool {
        let tokens = estimate_slice_tokens(messages)
            + crate::token_estimation::count_tool_schema_tokens(
                tools,
                &crate::token_estimation::TokenCountOptions::default(),
            );
        self.exceeds_trigger(tokens)
    }

    /// Whether a request of `tokens` total input tokens (messages plus tool
    /// declarations, however they were measured) reaches the trigger.
    ///
    /// The same comparison [`Self::should_summarize_with_tools`] applies to
    /// its own estimate, exposed so a caller holding a better measurement —
    /// the provider-reported prompt size of the previous call, say — can use
    /// it instead: at or above [`Self::trigger_budget`] when a context window
    /// is set, strictly above [`Self::trigger_tokens`] otherwise.
    pub fn exceeds_trigger(&self, tokens: u64) -> bool {
        match self.context_window {
            Some(_) => tokens >= self.trigger_budget(),
            None => tokens > self.trigger_tokens,
        }
    }

    /// Pins the trigger to an absolute token count, ignoring any context
    /// window: the policy then compacts once a request exceeds `tokens`.
    ///
    /// For forcing compaction early (benchmarks, tests) and for models whose
    /// window is unknown. Clears [`Self::context_window`], which only ever
    /// fed the trigger.
    pub fn with_trigger_override(mut self, tokens: u64) -> Self {
        self.context_window = None;
        self.trigger_tokens = tokens;
        self
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
        let plan = self.plan_split(messages);
        (plan.to_summarize, plan.to_keep)
    }

    /// [`Self::plan`], also reporting where the split was taken.
    ///
    /// With [`keep_recent_tokens`][Self::keep_recent_tokens] set, the tail is
    /// sized in tokens by [`find_cut_point`] (pairing-repaired the same way)
    /// and never left empty; otherwise it is the last
    /// [`keep_last`][Self::keep_last] messages. With
    /// [`pin_turn_user_message`][Self::pin_turn_user_message] set, the turn's
    /// user message is then moved to the front of the tail (see
    /// [`CompactionPlan::pinned`]).
    pub fn plan_split(&self, messages: &[Message]) -> CompactionPlan {
        let (system, non_system) = partition_system(messages);
        let cut = match self.keep_recent_tokens {
            Some(keep_tokens) => token_tail_cut(&non_system, keep_tokens),
            None => self.count_tail_cut(&non_system),
        };
        split_at_cut(system, &non_system, cut, self.pin_turn_user_message)
    }

    /// The count-based cut: the last [`keep_last`][Self::keep_last] messages,
    /// moved back to keep tool-call pairing intact.
    fn count_tail_cut(&self, non_system: &[Message]) -> usize {
        if non_system.len() <= self.keep_last {
            // Nothing old enough to summarize; keep everything.
            return 0;
        }
        let requested_split = non_system.len() - self.keep_last;
        let split = find_safe_cutoff_point(non_system, requested_split);
        if split != requested_split {
            tracing::debug!(
                "[summarization::plan] keep_last={} moved split {requested_split} -> {split} to preserve tool-call pairing",
                self.keep_last
            );
        }
        split
    }
}

/// The token-budget cut: at least `keep_tokens` of the newest messages, and
/// never an empty tail. When the newest message alone is over the budget it is
/// still kept (with the call it answers), since a tail with no recent message
/// at all leaves the model nothing to continue from.
fn token_tail_cut(non_system: &[Message], keep_tokens: u64) -> usize {
    let Some(cut) = find_cut_point(
        non_system,
        keep_tokens,
        crate::token_estimation::estimate_message_tokens,
    ) else {
        // Everything fits in the tail: nothing to summarize.
        return 0;
    };
    let index = if cut.index >= non_system.len() {
        find_safe_cutoff_point(non_system, non_system.len() - 1)
    } else {
        cut.index
    };
    tracing::debug!(
        keep_tokens,
        cut = index,
        kept_tokens = cut.tokens_after,
        "[summarization::plan] token-budget tail"
    );
    index
}

/// Split `non_system` at `cut` (a pairing-safe index) into a
/// [`CompactionPlan`], pinning the turn's user message when `pin` is set.
///
/// The pin applies only when the kept tail `non_system[cut..]` holds no user
/// message: the most recent user message before `cut` is then left out of
/// `to_summarize` and placed at the front of the kept tail, verbatim up to
/// [`PINNED_USER_MESSAGE_MAX_TOKENS`]. A user message never sits between an
/// assistant tool-call turn and its results, so removing it from the head and
/// putting it ahead of the tail keeps both sides' tool pairing intact.
pub(crate) fn split_at_cut(
    system: Vec<Message>,
    non_system: &[Message],
    cut: usize,
    pin: bool,
) -> CompactionPlan {
    let cut = cut.min(non_system.len());
    let tail_has_user = non_system[cut..].iter().any(is_turn_user_message);
    let pinned = (pin && !tail_has_user)
        .then(|| non_system[..cut].iter().rposition(is_turn_user_message))
        .flatten();

    let to_summarize: Vec<Message> = non_system[..cut]
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != pinned)
        .map(|(_, m)| m.clone())
        .collect();
    let mut to_keep = system;
    if let Some(index) = pinned {
        tracing::debug!(
            pinned = index,
            cut,
            "[summarization::plan] pinning the turn's user message into the kept tail"
        );
        to_keep.push(cap_pinned_message(&non_system[index]));
    }
    to_keep.extend(non_system[cut..].iter().cloned());

    CompactionPlan {
        to_summarize,
        to_keep,
        cut,
        pinned,
    }
}

/// Front-drop `messages` to `budget` tokens ([`TrimStrategy::MaxTokens`]) the
/// way the compression middleware's fallback does, but keep the most recent
/// user message, size-capped to fit the residual budget, when the drop would
/// remove every user message. Its tokens are reserved
/// from `budget` first, so the result still fits.
pub(crate) fn trim_keeping_turn_user_message(messages: &[Message], budget: u64) -> Vec<Message> {
    let system_tokens = crate::token_estimation::count_tokens_approximately(
        &messages
            .iter()
            .filter(|message| matches!(message, Message::System(_)))
            .cloned()
            .collect::<Vec<_>>(),
    );
    let Some(pin) = messages
        .iter()
        .rposition(is_turn_user_message)
        .and_then(|index| {
            fit_pinned_message(&messages[index], budget.saturating_sub(system_tokens))
        })
    else {
        return enforce_approximate_budget(
            trim_messages(messages, &TrimStrategy::MaxTokens(budget)),
            budget,
            None,
        );
    };
    let reserved = crate::token_estimation::count_tokens_approximately(std::slice::from_ref(&pin));
    let mut trimmed = trim_messages(
        messages,
        &TrimStrategy::MaxTokens(budget.saturating_sub(reserved)),
    );
    if trimmed.iter().any(is_turn_user_message) {
        let retained_user = trimmed.iter().rposition(is_turn_user_message);
        return enforce_approximate_budget(trimmed, budget, retained_user);
    }
    let system_prefix = trimmed
        .iter()
        .take_while(|m| matches!(m, Message::System(_)))
        .count();
    tracing::debug!(
        budget,
        reserved,
        "[summarization::trim] re-inserting the turn's user message after a fallback front-drop"
    );
    trimmed.insert(system_prefix, pin);
    enforce_approximate_budget(trimmed, budget, Some(system_prefix))
}

/// Correct the cheaper trim estimate against the prompt-pressure estimator.
pub(crate) fn enforce_approximate_budget(
    mut messages: Vec<Message>,
    budget: u64,
    mut pinned: Option<usize>,
) -> Vec<Message> {
    use crate::token_estimation::count_tokens_approximately;

    while count_tokens_approximately(&messages) > budget {
        let oldest = messages
            .iter()
            .enumerate()
            .find(|(index, message)| {
                Some(*index) != pinned && !matches!(message, Message::System(_))
            })
            .or_else(|| {
                messages
                    .iter()
                    .enumerate()
                    .find(|(index, _)| Some(*index) != pinned)
            })
            .map(|(index, _)| index);
        let Some(index) = oldest else {
            break;
        };
        let removed = messages.remove(index);
        if let Some(pin_index) = pinned.as_mut()
            && index < *pin_index
        {
            *pin_index -= 1;
        }
        // A call and its contiguous results are one provider turn. Dropping
        // only the call would strand results behind a pinned user message.
        if is_tool_calling_assistant(&removed) {
            while matches!(messages.get(index), Some(Message::Tool(_))) {
                messages.remove(index);
                if let Some(pin_index) = pinned.as_mut()
                    && index < *pin_index
                {
                    *pin_index -= 1;
                }
            }
        }
        while let Some(index) = messages
            .iter()
            .position(|message| !matches!(message, Message::System(_)))
            && matches!(messages[index], Message::Tool(_))
        {
            messages.remove(index);
            if let Some(pin_index) = pinned.as_mut()
                && index < *pin_index
            {
                *pin_index -= 1;
            }
        }
    }
    messages
}

/// A pinned user message fitted to this fallback's residual budget.
fn fit_pinned_message(message: &Message, budget: u64) -> Option<Message> {
    let pin = cap_pinned_message(message);
    let count = |message: &Message| {
        crate::token_estimation::count_tokens_approximately(std::slice::from_ref(message))
    };
    if count(&pin) <= budget {
        return Some(pin);
    }
    let text = pin.text();
    let render = |keep: usize| {
        let cut: String = text.chars().take(keep).collect();
        Message::user(format!(
            "{cut}\n[message truncated to fit the context window]"
        ))
    };
    if count(&render(0)) > budget {
        return None;
    }
    let mut low = 0;
    let mut high = text
        .chars()
        .count()
        .min(usize::try_from(budget.saturating_mul(4)).unwrap_or(usize::MAX));
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if count(&render(mid)) <= budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Some(render(low))
}

/// Whether `message` is a user message a person (or host) wrote — not a
/// user-role compaction checkpoint, which is a summary, not an assignment.
fn is_turn_user_message(message: &Message) -> bool {
    matches!(message, Message::User(_)) && !is_checkpoint(message)
}

/// `message`, or — when it estimates above [`PINNED_USER_MESSAGE_MAX_TOKENS`]
/// — its text cut to that size with a truncation marker. A truncated message
/// keeps only its text: the cap exists to bound size, and an attachment large
/// enough to need it cannot stay either.
fn cap_pinned_message(message: &Message) -> Message {
    let tokens = crate::token_estimation::estimate_message_tokens(message);
    if tokens <= PINNED_USER_MESSAGE_MAX_TOKENS {
        return message.clone();
    }
    let keep_chars = (PINNED_USER_MESSAGE_MAX_TOKENS as usize).saturating_mul(4);
    let text = message.text();
    let kept: String = text.chars().take(keep_chars).collect();
    tracing::debug!(
        tokens,
        cap = PINNED_USER_MESSAGE_MAX_TOKENS,
        "[summarization::plan] truncating an oversized pinned user message"
    );
    Message::user(format!(
        "{kept}\n[… message truncated to fit the context window: the original was about \
         {tokens} tokens]"
    ))
}

impl SummarizationPolicy {
    /// [`Self::plan`] with a token-budgeted tail: keep the most recent
    /// `keep_recent_tokens` of non-system messages verbatim (capped at half
    /// the trigger budget, so the compacted request lands well under the
    /// trigger) and summarize the rest.
    ///
    /// The tail starts at a user or assistant message: a cut that would land
    /// on a tool result moves back to the assistant turn that called it (see
    /// [`find_safe_cutoff_point`]). The most recent message group is always
    /// kept, even when it alone exceeds the budget.
    pub fn plan_recent_tokens(
        &self,
        messages: &[Message],
        keep_recent_tokens: u64,
    ) -> (Vec<Message>, Vec<Message>) {
        let plan = self.plan_split_recent_tokens(messages, keep_recent_tokens);
        (plan.to_summarize, plan.to_keep)
    }

    /// [`Self::plan_recent_tokens`], also reporting where the split was
    /// taken, and pinning the turn's user message when
    /// [`pin_turn_user_message`][Self::pin_turn_user_message] is set (see
    /// [`CompactionPlan::pinned`]).
    pub fn plan_split_recent_tokens(
        &self,
        messages: &[Message],
        keep_recent_tokens: u64,
    ) -> CompactionPlan {
        let (system, non_system) = partition_system(messages);
        let budget = match self.trigger_budget() {
            0 => keep_recent_tokens,
            trigger => keep_recent_tokens.min(trigger / 2),
        };
        let mut kept = 0u64;
        let mut start = non_system.len();
        for (index, message) in non_system.iter().enumerate().rev() {
            let tokens = crate::token_estimation::estimate_message_tokens(message);
            if kept + tokens > budget {
                break;
            }
            kept += tokens;
            start = index;
        }
        if start == non_system.len() {
            start = non_system.len().saturating_sub(1);
        }
        let split = find_safe_cutoff_point(&non_system, start);
        tracing::debug!(
            keep_recent_tokens,
            budget,
            kept_tokens = kept,
            requested = start,
            split,
            total = non_system.len(),
            "[summarization::plan_recent_tokens] token-budgeted split"
        );
        split_at_cut(system, &non_system, split, self.pin_turn_user_message)
    }
}

#[cfg(test)]
#[path = "model_summarizer_tests.rs"]
mod model_summarizer_test;
#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
