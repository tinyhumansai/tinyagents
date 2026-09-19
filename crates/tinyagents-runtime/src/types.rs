use std::{ops::Range, path::PathBuf, sync::Arc};

use tinyagents_harness::{
    CancellationToken,
    context::{RunConfig, RunContext},
};
use tinyagents_session::transcript::{TranscriptLocator, TranscriptMessage, TranscriptMeta};
use tinyinference_llm::message::Message;

use crate::{PrefixSnapshot, ToolSnapshot};

/// Selects the durable transcript a turn should load before execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResumeMode {
    /// Keep this session's current in-memory history.
    #[default]
    Never,
    /// Load the most recent transcript for the configured agent/stem key.
    LatestForAgent,
    /// Load the most recent root transcript matching `TurnOptions::thread_id`.
    Thread,
}

/// Explicit runtime controls for one session turn.
pub struct TurnOptions<C = ()> {
    /// Opaque correlation identifier persisted with transcript rows.
    pub request_id: Option<String>,
    /// Optional conversation thread identifier used for resume and metadata.
    pub thread_id: Option<String>,
    /// Whether the driver should use its streaming invocation path.
    pub stream: bool,
    /// Transcript resume behavior requested for this turn.
    pub resume: ResumeMode,
    /// Cooperative cancellation shared with the caller.
    pub cancellation: CancellationToken,
    /// Explicit live execution context consumed by the driver.
    pub run_context: RunContext<C>,
}

/// The codec-visible, durable subset of one turn's explicit options.
///
/// `RunContext` itself is live and consumed by the driver. A clone of its host
/// context is captured before that handoff so transcript reconciliation can
/// stamp host-owned data after the driver returns without relying on task-local
/// state or a lossy default context.
#[derive(Clone, Debug)]
pub struct TranscriptTurnOptions<C = ()> {
    /// Opaque correlation identifier for the current turn.
    pub request_id: Option<String>,
    /// Conversation thread selected for this turn.
    pub thread_id: Option<String>,
    /// Whether this turn used the streaming driver path.
    pub stream: bool,
    /// Resume mode selected before execution.
    pub resume: ResumeMode,
    /// Host-owned context cloned from `TurnOptions::run_context.data`.
    pub context: C,
}

/// A transcript destination selected lazily by a host for a session.
///
/// Constructing a target performs no I/O. The runtime opens it only when a
/// requested resume or the first append needs a bound history handle.
#[derive(Clone)]
pub struct TranscriptTarget {
    pub locator: Arc<dyn TranscriptLocator>,
    /// The durable stem used for every append and write.
    pub stem: String,
    /// Optional agent key used only by `ResumeMode::LatestForAgent` lookup.
    /// When absent, the write stem is also the resume lookup key.
    pub resume_agent: Option<String>,
    pub meta: TranscriptMeta,
}

impl TranscriptTarget {
    pub fn new(
        locator: Arc<dyn TranscriptLocator>,
        stem: impl Into<String>,
        meta: TranscriptMeta,
    ) -> Self {
        Self {
            locator,
            stem: stem.into(),
            resume_agent: None,
            meta,
        }
    }

    /// Uses a distinct agent key when looking up the latest transcript.
    pub fn with_resume_agent(mut self, resume_agent: impl Into<String>) -> Self {
        self.resume_agent = Some(resume_agent.into());
        self
    }

    pub(crate) fn same_binding(&self, other: &Self) -> bool {
        self.stem == other.stem
            && self.resume_agent == other.resume_agent
            && Arc::ptr_eq(&self.locator, &other.locator)
    }
}

/// Values prepared by `SessionHooks::before_resume` before transcript loading.
#[derive(Clone, Default)]
pub struct ResumePreparation {
    /// A lazy transcript destination. It can be selected or replaced before
    /// the first history handle is bound, but cannot be redirected afterwards.
    pub transcript: Option<TranscriptTarget>,
}

/// Values prepared by `SessionHooks::before_turn` for exactly one driver call.
#[derive(Clone, Default)]
pub struct TurnPreparation {
    /// A replacement prefix allowed before the first committed turn. It is
    /// reconciled against any decoded resumed history without duplication.
    pub prefix: Option<PrefixSnapshot>,
    /// The immutable tool declarations for this driver request. `None` uses
    /// the builder's compatibility default and is never retained from a prior
    /// preparation.
    pub tools: Option<ToolSnapshot>,
}

impl TurnPreparation {
    pub fn with_tools(tools: ToolSnapshot) -> Self {
        Self {
            tools: Some(tools),
            ..Self::default()
        }
    }
}

/// Read-only session state supplied to `before_turn`.
#[derive(Clone, Copy)]
pub struct SessionStateView<'a> {
    pub history: &'a [Message],
    pub raw_history: &'a [TranscriptMessage],
    pub prefix: &'a PrefixSnapshot,
    pub transcript_target: Option<&'a TranscriptTarget>,
    pub committed_turns: usize,
    /// `true` only when this call loaded and decoded a durable transcript
    /// before `before_turn` ran.
    pub resumed: bool,
}

/// The shape of a successful logical transcript transition.
///
/// An append extends the prior logical rows. Any rewrite, including a context
/// compaction with a longer replacement, is reported as `Replace` rather than
/// pretending that a suffix range was appended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptDelta {
    Append {
        previous_len: usize,
        appended: Range<usize>,
    },
    Replace {
        previous_len: usize,
        next_len: usize,
    },
}

/// Durable transcript information supplied after a successful append.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptCommitReceipt {
    pub path: PathBuf,
    pub delta: TranscriptDelta,
}

/// Exactly-once post-durability observation data.
#[derive(Clone, Debug)]
pub struct CommitReceipt<C = ()> {
    pub outcome: SessionTurnOutcome,
    pub options: TranscriptTurnOptions<C>,
    /// `None` when the host selected no durable transcript target.
    pub transcript: Option<TranscriptCommitReceipt>,
}

impl<C: Clone> TurnOptions<C> {
    pub(crate) fn transcript_options(&self) -> TranscriptTurnOptions<C> {
        TranscriptTurnOptions {
            request_id: self.request_id.clone(),
            thread_id: self.thread_id.clone(),
            stream: self.stream,
            resume: self.resume,
            context: self.run_context.data.clone(),
        }
    }
}

impl Default for TurnOptions<()> {
    fn default() -> Self {
        let cancellation = CancellationToken::new();
        Self {
            request_id: None,
            thread_id: None,
            stream: false,
            resume: ResumeMode::Never,
            run_context: RunContext::new(RunConfig::new("session"), ())
                .with_cancellation(cancellation.clone()),
            cancellation,
        }
    }
}

/// The input a host asks a session to execute.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnRequest {
    /// The next user/application message. Hooks may replace it before the
    /// runtime performs trailing-input deduplication.
    pub input: Message,
}

impl SessionTurnRequest {
    /// Creates a request with one next input message.
    pub fn new(input: Message) -> Self {
        Self { input }
    }
}

/// A committed turn result.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnOutcome {
    /// The full logical history after this turn.
    pub history: Vec<Message>,
    /// The driver's final visible output, when it produced one.
    pub output: Option<String>,
    /// `true` when the driver intentionally ended at an interruptible point.
    pub interrupted: bool,
}

/// The result of loading a transcript into a session.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionResume {
    /// Whether a transcript was found and decoded.
    pub loaded: bool,
    /// The loaded model history, or the existing history when none was found.
    pub history: Vec<Message>,
}

/// The one terminal observation emitted for each call to [`crate::Session::turn`].
#[derive(Clone, Debug, PartialEq)]
pub enum SessionTerminal {
    /// The turn committed. The outcome supplies durable finalization data.
    Completed(SessionTurnOutcome),
    /// The turn was cooperatively cancelled.
    Cancelled,
    /// The turn ended with an error after any recoverable partial persistence.
    Failed(String),
}
