//! The crate's shared, structurally-complete token estimator.
//!
//! # Why this module exists
//!
//! Several independent places in the harness need "roughly how many tokens is
//! this transcript?" — compaction gating, context middleware, and budget
//! preflight. Each grew its own `chars / 4` loop over
//! [`Message::text`][super::Message::text], and every one of them silently
//! under-counted the same way: a transcript's *structure* (tool calls, tool
//! result correlation ids, role labels, per-message framing) is invisible to
//! `text()`, and an assistant turn that only calls tools has **no text at
//! all**. A tool-driven run therefore estimated near zero and no gate fired.
//!
//! This module is the single correct implementation those call sites should
//! use. It is a port of LangChain's `count_tokens_approximately`
//! (`libs/core/langchain_core/messages/utils.py`), adapted to this crate's
//! [`Message`] model.
//!
//! # What is counted
//!
//! | Part | Source |
//! | ---- | ------ |
//! | content blocks (text, JSON, reasoning, provider extensions) | [`ContentBlock::estimated_char_weight`][super::ContentBlock::estimated_char_weight] |
//! | images | flat per-image weight, not the base64 length |
//! | assistant `tool_calls` | JSON rendering of the call array |
//! | tool `tool_call_id` | the id string |
//! | role label (`system` / `user` / `assistant` / `tool`) | [`TokenCountOptions::count_role`] |
//! | per-message framing overhead | [`TokenCountOptions::extra_tokens_per_message`] |
//! | tool declarations offered to the model | [`count_tool_schema_tokens`] |
//!
//! Rounding is **per message** (`ceil`), matching LangChain, so the parts of a
//! transcript sum to the whole rather than losing a fraction of a token each.
//!
//! # Two entry points, deliberately
//!
//! - [`count_tokens_approximately`] / [`count_tokens_approximately_with`] —
//!   the recommended counter. Use it for anything that compares an estimate
//!   against a real provider limit (context-window gating, budget preflight).
//! - [`estimate_message_tokens`] / [`estimate_slice_tokens`] — the crate's
//!   older bare `floor(chars / 4)` heuristic, retained because existing
//!   thresholds (and their tests) are calibrated against it. It shares the same
//!   *corrected* char weight, so it no longer under-counts tool calls; it just
//!   omits role labels and framing overhead.
//!
//! Nothing here is a real tokenizer. Treat every number as ±30%.

use super::{AssistantMessage, Message};
use crate::tool::ToolSchema;

/// Default characters per token (~4 for English prose and code).
pub const DEFAULT_CHARS_PER_TOKEN: f64 = 4.0;

/// Default per-message framing overhead in tokens (role delimiters,
/// begin/end-of-message markers). Matches LangChain's
/// `extra_tokens_per_message=3`.
pub const DEFAULT_EXTRA_TOKENS_PER_MESSAGE: f64 = 3.0;

/// Lower clamp applied to the usage-metadata calibration factor.
///
/// The factor may only *increase* the estimate: a provider reporting fewer
/// tokens than the heuristic guessed is not a licence to under-count, because
/// under-counting is the failure mode that lets a window overflow silently.
pub const USAGE_SCALE_MIN: f64 = 1.0;

/// Upper clamp applied to the usage-metadata calibration factor.
///
/// Without a ceiling a single anomalous usage report (a cached-prompt turn, a
/// provider that counts system overhead differently) would multiply every
/// subsequent estimate without bound. Matches LangChain's `min(1.25, …)`.
pub const USAGE_SCALE_MAX: f64 = 1.25;

/// Tuning knobs for [`count_tokens_approximately_with`].
///
/// [`Default`] reproduces LangChain's defaults with usage-metadata calibration
/// **off**, so the counter stays a pure function of the transcript unless a
/// caller opts in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenCountOptions {
    /// Characters that make up one token. See [`DEFAULT_CHARS_PER_TOKEN`].
    pub chars_per_token: f64,
    /// Framing tokens added per message. See
    /// [`DEFAULT_EXTRA_TOKENS_PER_MESSAGE`].
    pub extra_tokens_per_message: f64,
    /// Whether to charge for the message's role label.
    pub count_role: bool,
    /// Calibrate the character heuristic against real provider usage.
    ///
    /// When enabled, the newest assistant message carrying a non-zero
    /// [`Usage::total_tokens`][crate::usage::Usage::total_tokens] is
    /// compared against the heuristic's running estimate at that point, and the
    /// whole count is scaled by that ratio — clamped to
    /// `[USAGE_SCALE_MIN, USAGE_SCALE_MAX]` so one bad report cannot blow the
    /// estimate up or collapse it.
    ///
    /// This is the crate's advantage over LangChain's version: the usage is
    /// already on [`AssistantMessage::usage`], so no side-channel is needed.
    pub use_usage_metadata_scaling: bool,
}

impl Default for TokenCountOptions {
    fn default() -> Self {
        Self {
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            extra_tokens_per_message: DEFAULT_EXTRA_TOKENS_PER_MESSAGE,
            count_role: true,
            use_usage_metadata_scaling: false,
        }
    }
}

impl TokenCountOptions {
    /// Enables usage-metadata calibration. See
    /// [`use_usage_metadata_scaling`][Self::use_usage_metadata_scaling].
    pub fn with_usage_metadata_scaling(mut self) -> Self {
        self.use_usage_metadata_scaling = true;
        self
    }

    /// Sets the characters-per-token divisor.
    pub fn with_chars_per_token(mut self, chars_per_token: f64) -> Self {
        self.chars_per_token = chars_per_token;
        self
    }

    /// Effective divisor, guarding against a zero or negative configuration
    /// that would otherwise produce a non-finite estimate.
    fn divisor(&self) -> f64 {
        if self.chars_per_token > 0.0 {
            self.chars_per_token
        } else {
            DEFAULT_CHARS_PER_TOKEN
        }
    }
}

/// The OpenAI-style role label for a message, as counted toward its token cost.
pub fn message_role_label(message: &Message) -> &'static str {
    match message {
        Message::System(_) => "system",
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::Tool(_) => "tool",
    }
}

/// Total character weight charged for a single message: content blocks, tool
/// calls / tool-call id, and (optionally) the role label.
fn message_char_weight(message: &Message, options: &TokenCountOptions) -> usize {
    let mut chars = message.estimated_char_weight();
    if options.count_role {
        chars += message_role_label(message).len();
    }
    chars
}

/// Approximate the token count of `messages` with default options.
///
/// This is the recommended estimator for any comparison against a real provider
/// limit. See the [module docs][self] for what is counted.
pub fn count_tokens_approximately(messages: &[Message]) -> u64 {
    count_tokens_approximately_with(messages, &TokenCountOptions::default())
}

/// Approximate the token count of `messages` under explicit `options`.
///
/// Emits `[tokens]`-prefixed trace output describing the running total and any
/// usage calibration applied, so an over- or under-estimate can be traced to a
/// specific message without re-running the model.
pub fn count_tokens_approximately_with(messages: &[Message], options: &TokenCountOptions) -> u64 {
    let divisor = options.divisor();
    let mut total = 0.0_f64;

    // Newest assistant message carrying real usage, and the running heuristic
    // total at that point — the two halves of the calibration ratio.
    let mut last_reported_total: Option<u64> = None;
    let mut approx_at_last_report: Option<f64> = None;

    for (index, message) in messages.iter().enumerate() {
        let chars = message_char_weight(message, options);
        let message_tokens = (chars as f64 / divisor).ceil() + options.extra_tokens_per_message;
        total += message_tokens;

        tinyagents_tracing::trace!(
            "[tokens] message index={index} role={role} chars={chars} tokens={message_tokens} running={total}",
            role = message_role_label(message)
        );

        if options.use_usage_metadata_scaling
            && let Message::Assistant(AssistantMessage {
                usage: Some(usage), ..
            }) = message
            && usage.total_tokens > 0
        {
            last_reported_total = Some(usage.total_tokens);
            approx_at_last_report = Some(total);
        }
    }

    if options.use_usage_metadata_scaling
        && messages.len() > 1
        && let (Some(reported), Some(approx)) = (last_reported_total, approx_at_last_report)
        && approx > 0.0
    {
        let raw_factor = reported as f64 / approx;
        let factor = raw_factor.clamp(USAGE_SCALE_MIN, USAGE_SCALE_MAX);
        tinyagents_tracing::debug!(
            "[tokens] usage calibration reported={reported} approx={approx} raw_factor={raw_factor} clamped={factor}"
        );
        total *= factor;
    }

    let result = total.ceil().max(0.0) as u64;
    tinyagents_tracing::debug!(
        "[tokens] counted messages={} tokens={result}",
        messages.len()
    );
    result
}

/// Approximate the token cost of the tool declarations offered to a model.
///
/// Tool schemas are part of every request's prompt, so a run with twenty
/// verbose tool declarations starts thousands of tokens into its window before
/// the first user message. LangChain counts them the same way, dropping each
/// schema's own `title` / `description` because the tool's name and description
/// are already counted at the top level.
pub fn count_tool_schema_tokens(schemas: &[ToolSchema], options: &TokenCountOptions) -> u64 {
    if schemas.is_empty() {
        return 0;
    }
    let mut chars = 0usize;
    for schema in schemas {
        let mut parameters = schema.parameters.clone();
        if let Some(object) = parameters.as_object_mut() {
            object.remove("title");
            object.remove("description");
        }
        let rendered = serde_json::json!({
            "name": schema.name,
            "description": schema.description,
            "parameters": parameters,
        });
        chars += rendered.to_string().chars().count();
    }
    let tokens = (chars as f64 / options.divisor()).ceil().max(0.0) as u64;
    tinyagents_tracing::debug!(
        "[tokens] counted tool schemas count={} chars={chars} tokens={tokens}",
        schemas.len()
    );
    tokens
}

/// The crate's legacy `floor(chars / 4)` heuristic for a single message,
/// computed over the *corrected* character weight (content blocks **plus** tool
/// calls and tool-call ids).
///
/// Retained because the compaction thresholds in
/// [`crate::summarization`] are calibrated against it. Prefer
/// [`count_tokens_approximately`] for new code: this variant charges nothing
/// for role labels or per-message framing and so runs slightly low on
/// many-short-messages transcripts.
///
/// Returns at least `1` for any message with non-zero weight, so a short
/// message is never free.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = message.estimated_char_weight() as u64;
    if chars == 0 { 0 } else { (chars / 4).max(1) }
}

/// [`estimate_message_tokens`] summed over a slice.
pub fn estimate_slice_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}
