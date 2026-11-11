//! Types for explicit message trimming, summarization, and compression policies.
//!
//! All policy decisions — when to summarize, what to keep, and what provenance
//! to record — are expressed as data types so they can be inspected, tested,
//! and audited without coupling to any particular LLM provider.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ChatModel;

/// An LLM-backed [`Summarizer`] that condenses messages with the active model.
///
/// The model id is retained for provenance, while the wrapped model carries
/// the actual provider configuration.
pub struct ModelSummarizer {
    pub(super) model: Arc<dyn ChatModel<()>>,
    /// Model id, kept for logging and provenance.
    pub(super) model_id: String,
    /// Threshold reported in summary provenance; the policy owns the trigger.
    pub(super) threshold_fraction: f64,
    /// How tool calls and results are rendered into the transcript.
    pub(super) tool_rendering: ToolRendering,
    /// Output cap sent as the request's `max_tokens`; `None` leaves it unset.
    pub(super) max_output_tokens: Option<u32>,
}

/// How a summary transcript renders tool activity.
///
/// See [`render_message_for_summary_with`](super::render_message_for_summary_with).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRendering {
    /// Every tool call with its id and arguments, and every tool result, in
    /// the tagged form of
    /// [`render_message_for_summary`](super::render_message_for_summary).
    #[default]
    Full,
    /// The conversation only: each tool call becomes one
    /// `[used <name>: <arguments>]` line with its arguments cut short, and tool
    /// results are left out. For small summarization models that should see
    /// what was said and done, not the tools' output.
    Markers,
}

impl std::fmt::Debug for ModelSummarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSummarizer")
            .field("model_id", &self.model_id)
            .field("threshold_fraction", &self.threshold_fraction)
            .field("tool_rendering", &self.tool_rendering)
            .field("max_output_tokens", &self.max_output_tokens)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// A cheap heuristic estimate of the number of tokens in a piece of text.
///
/// The value is derived by [`estimate_tokens`](super::estimate_tokens) and should be treated as an
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
    /// [`Message::Custom`].
    Custom,
}

impl MessageRole {
    /// Returns the role of `message`.
    pub fn of(message: &Message) -> Self {
        match message {
            Message::System(_) => MessageRole::System,
            Message::User(_) => MessageRole::User,
            Message::Assistant(_) => MessageRole::Assistant,
            Message::Tool(_) => MessageRole::Tool,
            Message::Custom(_) => MessageRole::Custom,
        }
    }
}

/// Knobs for [`trim_messages_with`][crate::summarization::trim_messages_with].
///
/// [`Default`] is the safe configuration: tool-call pairing is repaired and no
/// role boundary is imposed. [`trim_messages`][crate::summarization::trim_messages]
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

    /// Provider-reported usage of the summarization call(s) that produced
    /// [`Self::summary`], when the summarizer made a model call and the
    /// provider reported it. `None` for deterministic summarizers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<tinyinference_llm::usage::Usage>,
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

    /// [`Self::summarize`], but iterative: `request` also carries the
    /// previous compaction's summary text (when this is not the first
    /// compaction of a run), so an LLM-backed implementation can *refine* the
    /// running summary instead of re-deriving it from scratch every time.
    ///
    /// The default implementation prepends a previous summary as a system
    /// message before delegating to [`Self::summarize`]. Override this method
    /// to pass prior context through a specialized prompt.
    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        if let Some(previous) = request.previous_summary.as_deref() {
            let mut messages = Vec::with_capacity(request.messages.len() + 1);
            messages.push(Message::system(previous));
            messages.extend(request.messages.iter().cloned());
            self.summarize(&messages).await
        } else {
            self.summarize(&request.messages).await
        }
    }

    /// Merges two or more per-half [`SummaryRecord`]s produced by
    /// [`Self::summarize_request`] into one, for the "split turn" case where a
    /// single turn's messages exceeded the per-call summarization budget and
    /// were summarized in separate halves (see
    /// [`crate::summarization::compaction::summarize_with_split`]).
    ///
    /// The default merges deterministically by concatenating each summary's
    /// text under a numbered header, union-ing their provenance
    /// [`CompressionProvenance::source_ids`] and token estimates — no LLM call
    /// is made. An LLM-backed [`Summarizer`] may override this to ask the
    /// model to fuse the two summaries into fluent prose instead.
    ///
    /// # Panics
    ///
    /// Never panics; an empty `summaries` slice returns an empty summary with
    /// no provenance rather than panicking, since a caller invoking this with
    /// nothing to merge is a caller bug, not a data condition worth
    /// crashing over.
    async fn merge(&self, summaries: &[SummaryRecord]) -> Result<SummaryRecord> {
        let mut parts: Vec<String> = Vec::with_capacity(summaries.len() + 1);
        parts.push("=== Merged Summary ===".to_string());
        let mut source_ids = Vec::new();
        let mut original_token_estimate = 0u64;
        let mut summary_token_estimate = 0u64;
        for (i, record) in summaries.iter().enumerate() {
            parts.push(format!("[part {}] {}", i + 1, record.summary.text()));
            source_ids.extend(record.provenance.source_ids.iter().cloned());
            original_token_estimate += record.provenance.original_token_estimate;
            summary_token_estimate += record.provenance.summary_token_estimate;
        }
        let summary_text = parts.join("\n");
        Ok(SummaryRecord {
            summary: Message::system(summary_text),
            provenance: CompressionProvenance {
                source_ids,
                original_token_estimate,
                summary_token_estimate,
                reason: "merged split-turn summaries (default concatenation)".to_string(),
            },
            usage: summaries
                .iter()
                .filter_map(|record| record.usage)
                .reduce(|sum, usage| sum + usage),
        })
    }
}

/// Input to [`Summarizer::summarize_request`]: the messages to condense, plus
/// (when this compaction is not the first in a run) the previous compaction's
/// summary text so an iterative summarizer can refine rather than restart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryRequest {
    /// The messages to condense into a new summary.
    pub messages: Vec<Message>,
    /// The summary text produced by the previous [`CompactionRecord`] on this
    /// run's transcript, when one exists. `None` for the first compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_summary: Option<String>,
    /// What the messages are, so a summarizer can pick the right prompt.
    #[serde(default)]
    pub kind: SummaryKind,
}

/// What a [`SummaryRequest`]'s messages are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryKind {
    /// Older history, summarized as a standalone checkpoint.
    #[default]
    Full,
    /// The beginning of a turn whose remainder stays verbatim after the
    /// summary (a compaction cut landed inside the turn): capture what was
    /// asked and what has been done so far, briefly.
    TurnPrefix,
}

impl SummaryRequest {
    /// Builds a [`SummaryKind::TurnPrefix`] request for `messages`.
    pub fn turn_prefix(messages: Vec<Message>) -> Self {
        Self {
            kind: SummaryKind::TurnPrefix,
            ..Self::new(messages)
        }
    }

    /// Builds a request with no previous summary (the common, first-compaction
    /// case).
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            previous_summary: None,
            kind: SummaryKind::Full,
        }
    }

    /// Sets the previous summary text for iterative refinement.
    pub fn with_previous_summary(mut self, previous_summary: impl Into<String>) -> Self {
        self.previous_summary = Some(previous_summary.into());
        self
    }
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
/// [`ModelProfile::max_input_tokens`]: tinyinference_llm::model::ModelProfile::max_input_tokens
///
/// # Example
///
/// ```
/// use tinyinference_llm::message::Message;
/// use tinyagents_harness::summarization::SummarizationPolicy;
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
    /// reached before summarization triggers. Defaults to `0.9` (90%) for a
    /// bare policy; [`crate::summarization::summarization_policy`] uses
    /// `min(80% of the window, 350k tokens)` instead. Ignored
    /// when `context_window` is `None`.
    #[serde(default = "default_threshold_fraction")]
    pub threshold_fraction: f64,

    /// Keep at least this many estimated tokens of the most recent
    /// non-system messages verbatim, instead of a fixed
    /// [`keep_last`][Self::keep_last] message count.
    ///
    /// When set, [`plan`][SummarizationPolicy::plan] cuts with
    /// [`find_cut_point`][crate::summarization::find_cut_point] and
    /// `keep_last` is ignored. A count says nothing about size: eight
    /// messages of a tool loop can be a few hundred tokens or most of the
    /// window. `None` (the default) keeps the count-based split.
    #[serde(default)]
    pub keep_recent_tokens: Option<u64>,

    /// Keep the turn's originating user message verbatim across a compaction.
    ///
    /// A long tool-driven turn crosses the threshold mid-turn, and its only
    /// user message — the assignment being worked on — is the oldest message
    /// in it, so the split folds it into the summary. When this is set and
    /// the kept tail holds no user message, [`plan`][SummarizationPolicy::plan]
    /// moves the most recent user message out of the summarized head to the
    /// front of the kept tail, verbatim up to
    /// [`PINNED_USER_MESSAGE_MAX_TOKENS`] (truncated with a marker beyond).
    /// Defaults to `false`.
    #[serde(default)]
    pub pin_turn_user_message: bool,
}

/// Size cap, in estimated tokens, of a user message pinned by
/// [`SummarizationPolicy::pin_turn_user_message`]. A larger message is cut to
/// this size with a truncation marker, so a pasted document cannot pin most of
/// the window in place.
pub const PINNED_USER_MESSAGE_MAX_TOKENS: u64 = 8_192;

/// The result of [`SummarizationPolicy::plan_split`]: the split itself plus
/// where it was taken, in the coordinates of the non-system messages of the
/// slice that was planned.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPlan {
    /// The messages to fold into a summary, in order. Never contains a system
    /// message or the pinned user message.
    pub to_summarize: Vec<Message>,
    /// System messages, then the pinned user message (when one was pinned),
    /// then the recent tail kept verbatim.
    pub to_keep: Vec<Message>,
    /// Index, into the non-system messages, of the first message of the
    /// recent tail. Everything before it was summarized, except
    /// [`Self::pinned`].
    pub cut: usize,
    /// Index, into the non-system messages, of the user message moved from
    /// the summarized head to the front of the kept tail, when one was.
    /// Always `< cut`.
    pub pinned: Option<usize>,
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
            keep_recent_tokens: None,
            pin_turn_user_message: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Summary placement
// ---------------------------------------------------------------------------

/// Where a compaction summary is placed in the rebuilt transcript.
///
/// Either way the summary sits *after* the leading system prompt and *before*
/// the kept recent messages; the variants differ only in the role it carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryPlacement {
    /// A `user`-role message opening with
    /// [`crate::summarization::CHECKPOINT_PREFIX`], a reference-only marker
    /// telling the model the content is background data, not instructions.
    /// It sits after the system prompt and before the kept messages.
    ///
    /// The default: the system prompt and tool declarations stay
    /// byte-identical across a compaction, so the provider's prefix cache
    /// for them survives, and the summary cannot be mistaken for a new
    /// system instruction.
    #[default]
    User,
    /// A `system`-role message carrying the same marker (the original
    /// placement). Kept for hosts that relied on it; it adds a second system
    /// message after the prompt, which churns the cacheable prefix on every
    /// compaction.
    System,
}

// ---------------------------------------------------------------------------
// Compaction record
// ---------------------------------------------------------------------------

/// Why a compaction ran.
///
/// Mirrors pi's `before_compaction{reason: manual|threshold|overflow}` (see
/// `docs/runtime-comparison/pi.md` §4.5): the reason is carried through to the
/// durable [`CompactionRecord`] and to
/// [`crate::events::AgentEvent::Compacted`] so a host or auditor can tell a
/// proactive threshold-triggered compaction apart from a reactive
/// overflow-recovery one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    /// Triggered explicitly by a caller (a `/compact`-style host command).
    Manual,
    /// Triggered by [`SummarizationPolicy::should_summarize`] crossing its
    /// configured token threshold.
    Threshold,
    /// Triggered reactively by
    /// [`crate::summarization::compaction::OverflowClassifier`] classifying a
    /// model provider error as a context-window overflow, as part of the
    /// overflow → compact → retry recovery path.
    Overflow,
}

impl CompactionReason {
    /// A stable, lowercase label for this reason (matches the `serde` wire
    /// form), for logging and event payloads that want a plain string.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompactionReason::Manual => "manual",
            CompactionReason::Threshold => "threshold",
            CompactionReason::Overflow => "overflow",
        }
    }
}

/// A durable record of one compaction operation, returned by the compaction
/// step (see `crate::summarization::compaction`) and, when a
/// [`CompactionSink`] is attached to the run, handed to it for persistence.
///
/// Mirrors pi's `CompactionEntry{summary, firstKeptEntryId, tokensBefore,
/// usage, details}` (`docs/runtime-comparison/pi.md` §4.5). A host that needs
/// durable records can persist this value through [`CompactionSink`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionRecord {
    /// The replacement summary text installed as the new leading context.
    pub summary: String,
    /// Role used when restoring this checkpoint. Older records used system.
    #[serde(default = "legacy_compaction_placement")]
    pub placement: SummaryPlacement,
    /// Index, into the non-system message slice compaction operated over, of
    /// the first message that survives verbatim (everything before it was
    /// folded into [`Self::summary`]). Matches [`crate::summarization::CutPoint::index`] when the
    /// record was produced from a [`crate::summarization::CutPoint`].
    ///
    /// One exception: when the policy pinned a user message
    /// ([`SummarizationPolicy::pin_turn_user_message`]), that message lies
    /// before this index yet was kept verbatim, right after the summary. Its
    /// index (same coordinates) is in [`Self::details`] as
    /// `pinned_user_index`; a sink rebuilding the compacted transcript must
    /// restore it there.
    pub first_kept_index: usize,
    /// Estimated total tokens of the transcript immediately before
    /// compaction.
    pub tokens_before: u64,
    /// Estimated total tokens of the transcript immediately after
    /// compaction (summary + kept messages).
    pub tokens_after: u64,
    /// Usage/cost of the summarization call(s) that produced
    /// [`Self::summary`], when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<tinyinference_llm::usage::Usage>,
    /// Additional host- or policy-defined provenance: which cut-point rule
    /// fired, split-turn bookkeeping, the hook (if any) that authored or
    /// replaced the summary, and so on. Defaults to JSON `null`.
    #[serde(default)]
    pub details: serde_json::Value,
    /// Why this compaction ran.
    pub reason: CompactionReason,
}

fn legacy_compaction_placement() -> SummaryPlacement {
    SummaryPlacement::System
}

/// A durable sink a host attaches to a [`crate::context::RunContext`] to
/// persist [`CompactionRecord`]s from compactions with a trusted live boundary
/// through a host-provided store,
/// instead of only living as long as the in-process
/// [`crate::middleware::ContextCompressionMiddleware::records`] buffer.
///
/// The harness does not prescribe the persistence backend, so this trait is
/// what [`crate::context::RunContext::compaction_sink`] holds.
pub trait CompactionSink: Send + Sync {
    /// Persists `record`. Implementations should be idempotent-safe to call
    /// once per record received (the compaction step calls this at most once
    /// for each successful compaction) and should not block the run indefinitely — a
    /// slow or failing sink should return promptly with an error rather than
    /// stall the agent loop.
    fn persist(&self, record: &CompactionRecord) -> Result<()>;
}
