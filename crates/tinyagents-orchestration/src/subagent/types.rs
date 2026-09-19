use std::collections::BTreeMap;

use tinyagents_harness::{CancellationToken, context::RunContext};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::{message::Message, usage::UsageTotals};

/// A host-provided request to run a task through a subagent.
///
/// `run_context` is owned live execution data, deliberately not a serializable
/// host DTO. It may already be a lineage-preserving child created from a
/// borrowed parent. Durable identity is carried separately in `task_key` so a
/// continuation can use a fresh execution context without changing its
/// original persistence key.
pub struct SubagentRequest<C = (), H = ()> {
    task_key: SubagentTaskKey,
    run_context: RunContext<C>,
    host_request: H,
    input: String,
    resume: Option<SubagentResume>,
}

/// One-way planner view of a validated request.
///
/// This is intentionally produced only by [`SubagentRequest::into_parts`]; it
/// cannot be turned back into a lifecycle request without a validated
/// constructor.
pub struct SubagentRequestParts<C = (), H = ()> {
    /// Durable identity selected by the validated constructor.
    pub task_key: SubagentTaskKey,
    /// Owned execution context for this invocation.
    pub run_context: RunContext<C>,
    /// Opaque host options for this invocation.
    pub host_request: H,
    /// Host-visible task input.
    pub input: String,
    /// Loaded or caller-supplied resume state.
    pub resume: Option<SubagentResume>,
}

/// Durable, host-neutral identity for one subagent lifecycle.
///
/// A task id is only unique inside the recursive run that created it. The
/// parent run and root run therefore scope durable persistence, in-memory
/// coalescing, and terminal cache entries. The optional thread adds the host's
/// durable conversation partition when it is available.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubagentTaskKey {
    /// Top-level recursive run that owns this task tree.
    pub root_run_id: String,
    /// Immediate run that requested this subagent lifecycle.
    pub parent_run_id: String,
    /// Host conversation partition, if either request or parent supplies one.
    pub thread_id: Option<String>,
    /// Host-local task id within the scoped recursive run.
    pub task_id: String,
}

impl SubagentTaskKey {
    /// Creates an initial durable identity from a context at task creation.
    /// Continuations must retain this returned key rather than derive another
    /// one from their fresh execution context.
    pub fn from_context<C>(
        run_context: &RunContext<C>,
        task_id: impl Into<String>,
        thread_id: Option<String>,
    ) -> Self {
        let task_id = task_id.into();
        SubagentTaskKey {
            root_run_id: run_context.lineage().root_run_id.as_str().to_owned(),
            parent_run_id: run_context.run_id().as_str().to_owned(),
            thread_id: thread_id
                .or_else(|| run_context.thread_id().map(|id| id.as_str().to_owned())),
            task_id,
        }
    }
}

impl<C, H> SubagentRequest<C, H> {
    /// Creates a fresh lifecycle from an actual parent and its owned child.
    ///
    /// Both `RunConfig` run ids must be durable unique host ids; this API can
    /// reject an equal parent/child id but cannot prove global uniqueness.
    pub fn fresh_from_parent<P>(
        parent: &RunContext<P>,
        owned_child: RunContext<C>,
        task_id: impl Into<String>,
        host_request: H,
        input: impl Into<String>,
        resume: Option<SubagentResume>,
    ) -> Result<Self, SubagentError> {
        let task_key = SubagentTaskKey::from_context(parent, task_id, None);
        Self::validate_key(&task_key)?;
        if owned_child.lineage().root_run_id != parent.lineage().root_run_id
            || owned_child.lineage().parent_run_id.as_ref() != Some(parent.run_id())
            || owned_child.run_id() == parent.run_id()
        {
            return Err(SubagentError::InvalidRequest(
                "owned execution context is not a distinct direct child of the supplied parent"
                    .into(),
            ));
        }
        Self::validate_thread(&task_key, &owned_child)?;
        Ok(Self {
            task_key,
            run_context: owned_child,
            host_request,
            input: input.into(),
            resume,
        })
    }

    /// Continues an authorized durable lifecycle with a fresh owned context.
    ///
    /// Hosts must authorize and recover `original_key` from their own durable
    /// record; do not derive a replacement key from the fresh context.
    pub fn continue_with_key(
        original_key: SubagentTaskKey,
        fresh_owned_context: RunContext<C>,
        host_request: H,
        input: impl Into<String>,
        resume: Option<SubagentResume>,
    ) -> Result<Self, SubagentError> {
        Self::validate_key(&original_key)?;
        Self::validate_thread(&original_key, &fresh_owned_context)?;
        Ok(Self {
            task_key: original_key,
            run_context: fresh_owned_context,
            host_request,
            input: input.into(),
            resume,
        })
    }

    /// Returns the sole durable identity for this lifecycle.
    pub fn task_key(&self) -> &SubagentTaskKey {
        &self.task_key
    }

    /// Returns the durable task id, derived from [`Self::task_key`].
    pub fn task_id(&self) -> &str {
        &self.task_key.task_id
    }

    /// Borrows the owned execution context before the request is consumed.
    pub fn run_context(&self) -> &RunContext<C> {
        &self.run_context
    }

    pub(crate) fn resume(&self) -> Option<&SubagentResume> {
        self.resume.as_ref()
    }

    pub(crate) fn set_resume(&mut self, resume: Option<SubagentResume>) {
        self.resume = resume;
    }

    /// Consumes this validated request for planner use.
    pub fn into_parts(self) -> SubagentRequestParts<C, H> {
        SubagentRequestParts {
            task_key: self.task_key,
            run_context: self.run_context,
            host_request: self.host_request,
            input: self.input,
            resume: self.resume,
        }
    }

    fn validate_key(key: &SubagentTaskKey) -> Result<(), SubagentError> {
        if key.root_run_id.is_empty() || key.parent_run_id.is_empty() || key.task_id.is_empty() {
            return Err(SubagentError::InvalidRequest(
                "durable task keys require non-empty root, parent, and task ids".into(),
            ));
        }
        Ok(())
    }

    fn validate_thread(
        key: &SubagentTaskKey,
        context: &RunContext<C>,
    ) -> Result<(), SubagentError> {
        let context_thread = context.thread_id().map(|thread| thread.as_str());
        if context_thread.is_some() && context_thread != key.thread_id.as_deref() {
            return Err(SubagentError::InvalidRequest(
                "execution context thread disagrees with durable task key".into(),
            ));
        }
        Ok(())
    }
}

/// A neutral checkpoint offered to a planner for resumption.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SubagentResume {
    /// Lossless model history available to the host planner.
    pub history: Vec<Message>,
    /// Opaque checkpoint token; its interpretation remains host-owned.
    pub checkpoint: Option<String>,
    /// Small neutral metadata. This deliberately excludes paths and credentials.
    pub metadata: BTreeMap<String, String>,
}

/// Fully resolved, immutable execution input produced by a host planner.
///
/// The planner, not this orchestration layer, resolves agent identity, prompt
/// messages, limits, workspace policy and the tool declaration allowlist.
pub struct PreparedSubagent<C = ()> {
    /// Host-stable task identity.
    pub task_id: String,
    /// Host-resolved agent identity.
    pub agent_key: String,
    /// Complete model input, including any resume history and prompt prefix.
    pub input: Vec<Message>,
    /// Frozen model-visible tool declarations for this one execution.
    pub tools: ToolSnapshot,
    /// Explicit child run context, including lineage and host context data.
    pub run_context: RunContext<C>,
}

/// The one execution handed to a [`crate::subagent::SubagentExecutor`].
pub struct SubagentExecution<C = ()> {
    /// The planner's complete, host-resolved execution description.
    pub prepared: PreparedSubagent<C>,
    /// Cooperative cancellation shared with the lifecycle owner.
    pub cancellation: CancellationToken,
}

/// A neutral reference to a host-owned artifact.
///
/// The reference intentionally contains no filesystem path or URL. Hosts own
/// artifact authorization and resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactReference {
    /// Stable host artifact identifier.
    pub id: String,
    /// Optional neutral media-type hint.
    pub media_type: Option<String>,
    /// Opaque, non-location metadata for a host to interpret.
    pub metadata: BTreeMap<String, String>,
}

/// A neutral suspension point that can later be supplied to a planner.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SubagentPause {
    /// Why execution needs input or an external host action.
    pub reason: String,
    /// Resume state captured at the suspension point.
    pub resume: SubagentResume,
}

/// A neutral non-successful but terminal completion.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubagentIncomplete {
    /// A host-safe explanation for incomplete work.
    pub reason: String,
}

/// The visible status of one subagent run.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SubagentStatus {
    /// The subagent completed normally.
    Completed,
    /// The subagent stopped at a resumable input boundary.
    AwaitingInput(SubagentPause),
    /// The subagent terminated without a complete result.
    Incomplete(SubagentIncomplete),
    /// Cooperative cancellation won the lifecycle race.
    Cancelled,
}

/// The durable lifecycle action that made a run result observable.
///
/// Hosts use this to attach side effects such as event-bus notifications and
/// progress records to the same successful commit boundary as the outcome.
/// In particular, a terminal result loaded from durable storage must be
/// returned to a caller without publishing a second terminal notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentPersistenceDisposition {
    /// This invocation committed a resumable pause record.
    PauseCommitted,
    /// This invocation atomically advanced an existing durable pause after a
    /// continuation.  It owns the new pause's host effects just as the first
    /// pause writer does; observers that lost the compare-and-swap receive
    /// [`Self::PauseExisting`] instead.
    PauseReplaced,
    /// This invocation atomically inserted the terminal record.
    TerminalInserted,
    /// A pause record was already made visible by another invocation.
    PauseExisting,
    /// A terminal record was already made visible by another invocation.
    TerminalExisting,
    /// This caller stopped observing an in-flight lifecycle before it reached
    /// a durable boundary. It must not publish terminal or pause effects.
    ObserverCancelled,
}

impl SubagentPersistenceDisposition {
    /// Whether this invocation owns host terminal/pause side effects.
    pub const fn should_emit_host_effects(self) -> bool {
        matches!(
            self,
            Self::PauseCommitted | Self::PauseReplaced | Self::TerminalInserted
        )
    }
}

/// Result of a neutral subagent lifecycle.
///
/// The outcome is lossless host-neutral execution data. The disposition says
/// whether this caller won durable persistence, preventing a second driver or
/// a coalesced caller from repeating host-visible lifecycle effects.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentRunResult {
    /// The authoritative execution outcome.
    pub outcome: SubagentOutcome,
    /// The successful persistence boundary for this invocation.
    pub disposition: SubagentPersistenceDisposition,
}

impl SubagentRunResult {
    /// Creates a result at a known persistence disposition.
    pub const fn new(
        outcome: SubagentOutcome,
        disposition: SubagentPersistenceDisposition,
    ) -> Self {
        Self {
            outcome,
            disposition,
        }
    }

    /// Whether the caller must publish host effects for this result.
    pub const fn should_emit_host_effects(&self) -> bool {
        self.disposition.should_emit_host_effects()
    }
}

/// Complete neutral result of one subagent execution.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SubagentOutcome {
    /// Host-stable task identity.
    pub task_id: String,
    /// Final visible text, retained even when cancellation arrives after work.
    pub output: String,
    /// Complete model history without lossy transcript conversion.
    pub history: Vec<Message>,
    /// Terminal or pause state.
    pub status: SubagentStatus,
    /// Model usage reported by the nested execution exactly once.
    pub usage: UsageTotals,
    /// Host-owned artifacts represented by neutral references.
    pub artifacts: Vec<ArtifactReference>,
}

impl SubagentOutcome {
    /// Creates the truthful empty result used when cancellation prevents a
    /// planner or executor from starting.
    pub fn cancelled(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            output: String::new(),
            history: Vec::new(),
            status: SubagentStatus::Cancelled,
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        }
    }

    pub(crate) fn cancelled_preserving(mut self) -> Self {
        self.status = SubagentStatus::Cancelled;
        self
    }
}

/// The durable input to [`crate::subagent::SubagentPersistence::save_pause`].
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedSubagentPause {
    /// Durable scoped lifecycle identity.
    pub key: SubagentTaskKey,
    /// The complete paused outcome. Keeping the visible output, history,
    /// usage, and artifacts with the suspension means a duplicate driver can
    /// return the durable winner rather than its own uncommitted result.
    pub outcome: SubagentOutcome,
    /// The exact resumable state this invocation consumed, when it is a
    /// continuation. Persistence implementations compare this under their
    /// scoped-key transaction before replacing a pause. `None` means this
    /// attempt creates the first pause for a fresh lifecycle.
    pub replaces: Option<SubagentResume>,
}

/// Result of atomically persisting a pause transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentPausePersistenceDisposition {
    /// A fresh lifecycle installed its first pause.
    Inserted,
    /// A continuation consumed and replaced the exact prior pause.
    Replaced,
    /// Another lifecycle already owns the current pause.
    Existing,
    /// A terminal outcome already closed the lifecycle.
    TerminalExisting,
}

/// Result of atomically persisting a terminal transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubagentTerminalPersistenceDisposition {
    /// This invocation atomically closed an unpaused lifecycle or consumed the
    /// exact pause it had loaded.
    Inserted,
    /// Another invocation already committed a terminal outcome.
    Existing,
    /// Another invocation owns a newer or unconsumed pause. The caller must
    /// return that pause outcome rather than manufacture a terminal result.
    PauseExisting,
}

/// Typed lifecycle failures. Adapters classify their errors at the seam that
/// owns them; the driver never flattens them into an untyped host error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubagentError {
    /// A supplied context or durable key violates neutral lifecycle invariants.
    InvalidRequest(String),
    /// The planner rejected or could not resolve a request.
    Planning(String),
    /// The executor could not finish a prepared run.
    Execution(String),
    /// Resume loading or outcome persistence failed.
    Persistence(String),
    /// A required planner, executor, or persistence seam was not provided.
    MissingCapability(&'static str),
    /// A host seam returned an outcome for a task other than the one reserved
    /// by this lifecycle. The driver rejects it before any persistence or
    /// terminal cache write can corrupt another task's record.
    TaskIdMismatch {
        /// Task id the caller reserved.
        expected: String,
        /// Task id returned by the planner or executor.
        actual: String,
    },
    /// Cooperative cancellation interrupted the lifecycle.
    Cancelled,
}

impl std::fmt::Display for SubagentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid subagent request: {message}"),
            Self::Planning(message) => write!(f, "subagent planning failed: {message}"),
            Self::Execution(message) => write!(f, "subagent execution failed: {message}"),
            Self::Persistence(message) => write!(f, "subagent persistence failed: {message}"),
            Self::MissingCapability(capability) => {
                write!(f, "subagent host capability is unavailable: {capability}")
            }
            Self::TaskIdMismatch { expected, actual } => write!(
                f,
                "subagent host seam returned task id {actual:?}, expected {expected:?}"
            ),
            Self::Cancelled => write!(f, "subagent execution was cancelled"),
        }
    }
}

impl std::error::Error for SubagentError {}
