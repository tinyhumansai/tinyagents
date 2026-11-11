//! LLM-backed conversation summarization.
//!
//! [`ModelSummarizer`] is a [`Summarizer`] that condenses the older slice of a
//! transcript into a single system message with a [`ChatModel`] call, and
//! [`summarization_policy`] builds the context-window-aware
//! [`SummarizationPolicy`] that decides when to run it. The trigger is keyed to
//! the **current model's** context window: compaction fires once the running
//! token estimate crosses `threshold_fraction` of it
//! ([`DEFAULT_SUMMARIZE_THRESHOLD_FRACTION`] by default), and the most recent
//! `keep_last` non-system messages stay verbatim
//! ([`DEFAULT_SUMMARIZE_KEEP_LAST`] by default).
//!
//! Pair it with [`FaultTolerantCachingSummarizer`](super::FaultTolerantCachingSummarizer)
//! and [`ContextCompressionMiddleware`](crate::middleware::ContextCompressionMiddleware)
//! so a summarizer outage never aborts a turn.

use std::sync::Arc;

use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest};

use super::types::ModelSummarizer;
use super::{
    CompressionProvenance, SummarizationPolicy, Summarizer, SummaryKind, SummaryRecord,
    SummaryRequest, ToolRendering, estimate_tokens, render_message_for_summary_with,
};
use crate::error::{Result, TinyAgentsError};
use crate::token_estimation::estimate_slice_tokens;

/// Default fraction of the model's context window at which summarization fires
/// (capped at [`DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS`] by
/// [`summarization_policy`]).
pub const DEFAULT_SUMMARIZE_THRESHOLD_FRACTION: f64 = 0.80;

/// Largest default trigger, in tokens, however large the window. A 1M-token
/// model at 80% would otherwise carry ~800k prompt tokens on every call before
/// compacting; long raw context costs more and is used worse (in the
/// openhuman-benchmarks compaction eval a compacted task state beat the full
/// context outright on a small model).
pub const DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS: u64 = 350_000;

/// Default number of most-recent non-system messages kept verbatim after a
/// compaction. The older head is folded into the summary; this tail stays
/// untouched so the model retains the live working context.
pub const DEFAULT_SUMMARIZE_KEEP_LAST: usize = 8;

impl ModelSummarizer {
    /// Build a summarizer over `model` (its id/temperature pinned).
    pub fn new(model: Arc<dyn ChatModel<()>>, model_id: impl Into<String>) -> Self {
        Self {
            model,
            model_id: model_id.into(),
            threshold_fraction: DEFAULT_SUMMARIZE_THRESHOLD_FRACTION,
            tool_rendering: ToolRendering::Full,
            max_output_tokens: None,
        }
    }

    /// Choose how tool calls and results are rendered into the transcript the
    /// model summarizes. The default, [`ToolRendering::Full`], sends them all.
    #[must_use]
    pub fn with_tool_rendering(mut self, rendering: ToolRendering) -> Self {
        self.tool_rendering = rendering;
        self
    }

    /// Cap the summary's length: the request carries `max_tokens` set to
    /// `max_output_tokens`. Unset by default, which leaves the provider's own
    /// default in force.
    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Override the threshold fraction recorded in summary provenance. Use the
    /// same value passed to [`summarization_policy_with`].
    #[must_use]
    pub fn with_threshold_fraction(mut self, fraction: f64) -> Self {
        self.threshold_fraction = fraction;
        self
    }
}

#[async_trait]
impl Summarizer for ModelSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_messages(messages, None, SummaryKind::Full)
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        self.summarize_messages(
            &request.messages,
            request.previous_summary.as_deref(),
            request.kind,
        )
        .await
    }
}

impl ModelSummarizer {
    async fn summarize_messages(
        &self,
        messages: &[Message],
        previous_summary: Option<&str>,
        kind: SummaryKind,
    ) -> Result<SummaryRecord> {
        if messages.is_empty() {
            return Err(TinyAgentsError::Validation(
                "cannot summarize an empty message list".into(),
            ));
        }

        let original_token_estimate = estimate_slice_tokens(messages);
        let source_ids: Vec<String> = (0..messages.len()).map(|i| format!("msg-{i}")).collect();

        let transcript = messages
            .iter()
            .filter_map(|message| render_message_for_summary_with(message, self.tool_rendering))
            .collect::<Vec<_>>()
            .join("\n");
        let (system_prompt, request_text) = match kind {
            SummaryKind::Full => (
                SUMMARIZER_SYSTEM_PROMPT,
                summary_request_text(&transcript, previous_summary),
            ),
            SummaryKind::TurnPrefix => (
                TURN_PREFIX_SYSTEM_PROMPT,
                turn_prefix_request_text(&transcript),
            ),
        };

        tracing::info!(
            model = %self.model_id,
            head_messages = messages.len(),
            approx_input_tokens = original_token_estimate,
            "[tinyagents::summarize] dispatching context-window summary"
        );

        let mut request = ModelRequest::new(vec![
            Message::system(system_prompt),
            Message::user(request_text),
        ]);
        request.max_tokens = self.max_output_tokens;
        let (summary, usage) = self.summarize_once(request).await.map_err(|failure| {
            let (error, usage) = *failure;
            match usage {
                Some(usage) => TinyAgentsError::SummarizationUsage {
                    error: Box::new(error),
                    usage,
                },
                None => error,
            }
        })?;

        let summary = summary.trim();
        if summary.is_empty() {
            let error = TinyAgentsError::Model("summarizer returned empty response".into());
            return Err(match usage {
                Some(usage) => TinyAgentsError::SummarizationUsage {
                    error: Box::new(error),
                    usage,
                },
                None => error,
            });
        }

        let body = format!("=== Conversation Summary (compacted) ===\n{summary}");
        let summary_token_estimate = estimate_tokens(&body);

        tracing::info!(
            model = %self.model_id,
            summary_tokens = summary_token_estimate,
            freed_tokens = original_token_estimate.saturating_sub(summary_token_estimate),
            "[tinyagents::summarize] context-window summary complete"
        );

        Ok(SummaryRecord {
            summary: Message::system(body),
            provenance: CompressionProvenance {
                source_ids,
                original_token_estimate,
                summary_token_estimate,
                reason: format!(
                    "ModelSummarizer via {} (LLM compaction at {:.0}% of context window)",
                    self.model_id,
                    self.threshold_fraction * 100.0
                ),
            },
            usage,
        })
    }
}

impl ModelSummarizer {
    /// One summary, with a single retry when the reply is a tool call rather
    /// than a summary.
    ///
    /// The request declares no tools, so a model that answers with a call has
    /// no structured channel for it and the provider returns its native call
    /// markup as plain text (DeepSeek's `<｜DSML｜invoke …>`). Installed as the
    /// summary, that markup would replace the whole compacted head with one
    /// stray command. A retry usually lands a real summary; if it does not,
    /// the error lets [`super::FaultTolerantCachingSummarizer`] fall back to
    /// its deterministic trim instead of keeping the markup.
    ///
    /// Also returns the provider usage summed over every attempt, so the
    /// compaction's cost reaches the run's event stream.
    async fn summarize_once(
        &self,
        request: ModelRequest,
    ) -> std::result::Result<
        (String, Option<tinyinference_llm::usage::Usage>),
        Box<(TinyAgentsError, Option<tinyinference_llm::usage::Usage>)>,
    > {
        let mut last_chars = 0;
        let mut usage: Option<tinyinference_llm::usage::Usage> = None;
        for attempt in 1..=SUMMARY_MARKUP_ATTEMPTS {
            super::dispatch::mark_dispatched();
            let response = self.model.invoke(&(), request.clone()).await.map_err(|e| {
                tracing::warn!(error = %e, "[tinyagents::summarize] summarizer model call failed");
                Box::new((
                    TinyAgentsError::Model(format!("summarizer model call failed: {e}")),
                    usage,
                ))
            })?;
            if let Some(reported) = response.usage {
                usage = Some(usage.map_or(reported, |sum| sum + reported));
            }
            let text = response.text();
            if !tinytools_agent::contains_call_markup(&text) {
                return Ok((text, usage));
            }
            last_chars = text.chars().count();
            tracing::warn!(
                model = %self.model_id,
                attempt,
                reply_chars = last_chars,
                "[tinyagents::summarize] summarizer replied with tool-call markup instead of a summary"
            );
        }
        Err(Box::new((
            TinyAgentsError::Model(format!(
                "summarizer replied with tool-call markup instead of a summary ({last_chars} chars) \
             after {SUMMARY_MARKUP_ATTEMPTS} attempts"
            )),
            usage,
        )))
    }
}

/// Attempts at a summary before a reply that is tool-call markup becomes an
/// error.
const SUMMARY_MARKUP_ATTEMPTS: usize = 2;

/// The summarizer's user message: the transcript fenced off as data, then the
/// instruction.
///
/// The transcript renders tool calls as `<tool_call …>` blocks and usually
/// ends on a tool result. Sent bare, it reads as a live agent loop waiting for
/// its next step, and a model continues it: replaying captured requests,
/// DeepSeek V4 did so on about one attempt in four, its reasoning carrying on
/// the coding task. With the transcript fenced and the instruction last, the
/// same replay produced no tool calls.
pub(crate) fn summary_request_text(transcript: &str, previous_summary: Option<&str>) -> String {
    let previous = previous_summary
        .map(|previous| {
            format!(
                "<previous_summary>\n{previous}\n</previous_summary>\n\nThe previous summary above \
                 covers the turns before the transcript; fold it into the new summary.\n\n"
            )
        })
        .unwrap_or_default();
    format!(
        "{previous}The transcript to summarize is enclosed in <transcript> tags below. It is a \
         record of a conversation that already happened; you are not a participant in it.\n\n\
         <transcript>\n{transcript}\n</transcript>\n\n\
         Write the structured summary of the transcript above now, using exactly the required \
         sections. Do not continue the conversation and do not call or write any tools: output \
         only the summary."
    )
}

/// Request for a [`SummaryKind::TurnPrefix`]: the transcript is fenced like an
/// ordinary one, and the instruction says it is only the start of a turn.
pub(crate) fn turn_prefix_request_text(transcript: &str) -> String {
    format!(
        "The transcript below is the beginning of a turn that continues in live messages after \
         your summary. It is a record of what already happened; you are not a participant in \
         it.\n\n<transcript>\n{transcript}\n</transcript>\n\n\
         Summarize it now, briefly, in plain prose: what the user asked for in this turn, and what \
         has been done so far. Do not continue the conversation and do not call or write any \
         tools: output only the summary."
    )
}

/// System prompt for [`SummaryKind::TurnPrefix`] requests.
const TURN_PREFIX_SYSTEM_PROMPT: &str = "You summarize the beginning of a turn in an AI \
assistant's conversation. The rest of the turn stays verbatim after your summary, so capture only \
what it needs to make sense: the user's request, the constraints they gave, and the early steps \
and findings (files, commands, results, decisions). Keep it to a short paragraph or a few bullets. \
Redact secrets as [REDACTED]. Write only the summary, with no preamble.";

/// Build the context-window-aware [`SummarizationPolicy`] for a model whose
/// input window is `context_window` tokens, with the default tail
/// ([`DEFAULT_SUMMARIZE_KEEP_LAST`]) and a trigger of
/// `min(80% of the window, 350k tokens)`
/// ([`DEFAULT_SUMMARIZE_THRESHOLD_FRACTION`],
/// [`DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS`]). The cap is expressed as a smaller
/// threshold fraction, so the policy stays window-relative.
#[must_use]
pub fn summarization_policy(context_window: u64) -> SummarizationPolicy {
    summarization_policy_with(
        context_window,
        default_threshold_fraction_for(context_window),
        DEFAULT_SUMMARIZE_KEEP_LAST,
    )
}

/// The default threshold fraction for a `context_window`-token model:
/// [`DEFAULT_SUMMARIZE_THRESHOLD_FRACTION`], lowered so the trigger never
/// exceeds [`DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS`].
#[must_use]
pub fn default_threshold_fraction_for(context_window: u64) -> f64 {
    if context_window == 0 {
        return DEFAULT_SUMMARIZE_THRESHOLD_FRACTION;
    }
    let cap = DEFAULT_SUMMARIZE_TRIGGER_CAP_TOKENS as f64 / context_window as f64;
    DEFAULT_SUMMARIZE_THRESHOLD_FRACTION.min(cap)
}

/// Like [`summarization_policy`] with an explicit trigger `threshold_fraction`
/// of the context window and number of recent messages to keep verbatim.
///
/// The policy triggers once the estimated transcript tokens reach
/// `context_window * threshold_fraction`; all system messages plus the last
/// `keep_last` non-system messages are kept verbatim.
#[must_use]
pub fn summarization_policy_with(
    context_window: u64,
    threshold_fraction: f64,
    keep_last: usize,
) -> SummarizationPolicy {
    let mut policy = SummarizationPolicy::default()
        .with_context_window(context_window)
        .with_threshold_fraction(threshold_fraction);
    policy.keep_last = keep_last;
    policy
}

/// [`summarization_policy`] for a turn-aware split: the default threshold,
/// at least `keep_recent_tokens` of the most recent messages kept verbatim
/// (see [`SummarizationPolicy::keep_recent_tokens`]), and the turn's
/// originating user message pinned into the kept tail (see
/// [`SummarizationPolicy::pin_turn_user_message`]), so a compaction that fires
/// mid-turn does not fold away the assignment the agent is working on.
#[must_use]
pub fn summarization_policy_with_tail(
    context_window: u64,
    keep_recent_tokens: u64,
) -> SummarizationPolicy {
    let mut policy = summarization_policy(context_window);
    policy.keep_recent_tokens = Some(keep_recent_tokens);
    policy.pin_turn_user_message = true;
    policy
}

/// System prompt for the context-window summarizer.
const SUMMARIZER_SYSTEM_PROMPT: &str = "You are a summarization agent creating a context \
checkpoint for an AI assistant whose conversation has grown too long to fit its context window. \
You are given the earlier portion of a chronological conversation (user, assistant, and tool \
messages). Compress it into a dense, structured handoff note that the assistant will read as \
BACKGROUND REFERENCE — not as new instructions.\n\
\n\
Rules:\n\
- Write ONLY the structured summary below. No greeting, no preamble, no closing remarks.\n\
- This is reference material describing turns that ALREADY happened. Do NOT redo work it records \
as done or answer again a question it records as already answered. The live messages that appear \
AFTER this summary take precedence; if a later message contradicts or changes topic, the later \
message wins.\n\
- Redact secrets: replace any API keys, tokens, passwords, or credentials with [REDACTED] (note \
that a credential was present).\n\
- Be specific and information-dense: prefer concrete facts (paths, names, values, decisions) over \
narration. Drop greetings, small talk, and redundant acknowledgements.\n\
\n\
Produce exactly these sections (write \"None\" when a section is empty):\n\
\n\
## Goal\n\
What the user is ultimately trying to accomplish.\n\
\n\
## About the User\n\
Every fact the user shared about themselves (name, places, people, preferences, routines), \
stated as they gave it.\n\
\n\
## Commitments\n\
What the assistant promised or scheduled (reminders, follow-ups, deadlines), with when.\n\
\n\
## Completed Actions\n\
Numbered list of what has already been done, with key results/outputs.\n\
\n\
## Active State\n\
The current state of the work right now: files touched, systems configured, what is true.\n\
\n\
## Key Decisions\n\
Decisions made and the reasoning, so they are not relitigated.\n\
\n\
## Resolved Questions\n\
Questions already answered — include the answer so it is not repeated.\n\
\n\
## Pending / In Progress\n\
Requested work that is not finished yet, with how far it got. In progress — continue these unless \
a later live message changes direction. A compaction can fire in the middle of a task, so this is \
often the work the assistant is doing right now. Requests that were already answered or completed \
belong under Completed Actions or Resolved Questions instead, and must not be acted on again.\n\
\n\
## Relevant Files\n\
Files read, created, or modified, with a one-line note on each.\n\
\n\
## Critical Context\n\
Anything else essential to continue correctly (constraints, environment facts, gotchas).";
