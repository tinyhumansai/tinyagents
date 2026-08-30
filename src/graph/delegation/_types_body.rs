/// Which stage a delegation node is asking the injected worker to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationStage {
    /// Produce a plan for the task.
    Plan,
    /// Execute the current plan (re-run on revision).
    Execute,
    /// Review the latest execution; may approve or request a revision.
    Review,
}

/// What an injected stage worker returns.
#[derive(Debug, Clone)]
pub struct DelegationStageOutput {
    /// The stage's textual output (plan text, execution result, or review note).
    pub text: String,
    /// Only meaningful for [`DelegationStage::Review`]: `true` approves the
    /// execution and ends the loop; `false` requests another revision.
    pub approved: bool,
    /// The exact prompt handed to this stage's worker, when it surfaces one.
    /// Persisted into [`StepRecord::prompt`] for per-step provenance (read only
    /// for the execute stage; ignored elsewhere). `None` when the worker does not
    /// surface a prompt — e.g. the deterministic test mock.
    pub prompt: Option<String>,
}

impl DelegationStageOutput {
    /// A plain non-review stage output (the `approved` flag is unused and no
    /// prompt is surfaced).
    pub fn done(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            approved: true,
            prompt: None,
        }
    }
}

/// Current on-disk schema version for a checkpointed [`DelegationState`]. Bumped
/// only on a breaking state-shape change — introduced with the
/// `executions: Vec<String>` → `Vec<StepRecord>` migration (issue #3884).
/// Pre-versioned records deserialize to `0` via `#[serde(default)]`, so a resume
/// can tell a stale checkpoint from a current one.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// One completed execute-stage pass, recorded durably so a resumed run knows
/// exactly how far it got and can render/finalize per step rather than from a
/// flat text log. Replaces the former `executions: Vec<String>` (issue #3884).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    /// 0-based execute pass: `0` is the first execution, `n` the n-th revision.
    pub index: usize,
    /// The exact prompt handed to the execute sub-agent for this pass — per-step
    /// provenance and the seam a later plan-edit slice (#3881) diffs against.
    /// Empty when the worker did not surface one.
    #[serde(default)]
    pub prompt: String,
    /// The sub-agent's result text — the value the former `Vec<String>` entry held.
    pub result: String,
}

/// Typed working state threaded through (and checkpointed across) the delegation
/// graph. Serde-serializable so a [`Checkpointer`] can persist and restore it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DelegationState {
    /// The plan produced by the `plan` stage.
    pub plan: Option<String>,
    /// One record per execution pass (the first plus each revision), typed so a
    /// resumed run can render/finalize per step (widened from `Vec<String>`,
    /// issue #3884).
    pub executions: Vec<StepRecord>,
    /// One entry per review pass.
    pub reviews: Vec<String>,
    /// Number of revisions the reviewer requested (loops back to `execute`).
    pub revisions: usize,
    /// Set once the reviewer approves or the revision cap is hit.
    pub approved: bool,
    /// The final synthesized output (set by `finalize`).
    pub final_output: Option<String>,
    /// Set when the run short-circuited because its token was cancelled.
    pub cancelled: bool,
    /// The durable human-approval decision, once a resume delivers one:
    /// `Some(true)` = approved, `Some(false)` = denied, `None` = not gated /
    /// still awaiting. Only meaningful when `require_review_approval` is set.
    #[serde(default)]
    pub human_approved: Option<bool>,
    /// Set when the durable human-approval gate denied the delegated result
    /// (deny semantics: block the action, finalize as denied).
    #[serde(default)]
    pub denied: bool,
    /// On-disk schema version, stamped [`CURRENT_SCHEMA_VERSION`] on a fresh run
    /// and defaulting to `0` for pre-versioned checkpoints.
    /// [`run_or_resume_delegation`] expires any checkpoint whose version is below
    /// `CURRENT_SCHEMA_VERSION` (and any that fails to deserialize) instead of
    /// resuming or returning it — so a shape change that stays structurally
    /// decodable is still not misread.
    #[serde(default)]
    pub schema_version: u32,
}

impl DelegationState {
    /// A fresh run's initial state, stamped with the current schema version so
    /// its checkpoints are self-identifying.
    fn new_run() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            ..Self::default()
        }
    }

    /// The latest execution result text, if any — the projection the review
    /// prompt and the finalize summary read.
    pub fn last_result(&self) -> Option<&str> {
        self.executions.last().map(|r| r.result.as_str())
    }

    /// The execution result texts in order — the flat projection used for the
    /// durable approval-interrupt payload, kept `Vec<&str>` so that wire shape
    /// is unchanged from the pre-#3884 `Vec<String>`.
    pub fn executions_texts(&self) -> Vec<&str> {
        self.executions.iter().map(|r| r.result.as_str()).collect()
    }
}

/// Reducer updates emitted by the delegation nodes.
pub(crate) enum DelegationUpdate {
    Plan(String),
    Execution {
        prompt: String,
        result: String,
    },
    Review {
        note: String,
        approved: bool,
    },
    /// A durable human-approval decision delivered by a resume command.
    HumanDecision {
        approved: bool,
    },
    Final(String),
    Cancelled,
}

/// Configuration for a delegation run.
pub struct DelegationConfig {
    /// Upper bound on reviewer-requested revisions before forcing `finalize`.
    pub max_revisions: usize,
    /// Optional durable checkpointer (e.g. a `FileCheckpointer`). When set with a
    /// `thread_id`, the run persists its state at every super-step boundary.
    pub checkpointer: Option<Arc<dyn Checkpointer<DelegationState>>>,
    /// Thread id for checkpoint keying; required for the checkpointer to persist.
    pub thread_id: Option<String>,
    /// Cooperative cancellation; checked at each node boundary.
    pub cancel: CancellationToken,
    /// When set, an approved review does not finalize directly: the run reaches
    /// a durable **human-approval** interrupt (`NodeResult::Interrupt`) that is
    /// persisted via the checkpointer (Sync durability) and survives a process
    /// restart. The pause is only released by [`resume_delegation`] carrying the
    /// approver's decision. Requires `checkpointer` + `thread_id` (interrupts
    /// require durability).
    ///
    /// This is the **durable** approval boundary — distinct from the interactive
    /// chat-turn approval gate (the 10-min TTL steering pause surfaced via
    /// `ApprovalRequestCard`), which parks a live chat turn in memory and is left
    /// exactly as-is. Durable graphs pause by checkpoint; chat turns pause by
    /// steering. See the `approval` node below.
    pub require_review_approval: bool,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            max_revisions: 2,
            checkpointer: None,
            thread_id: None,
            cancel: CancellationToken::new(),
            require_review_approval: false,
        }
    }
}

/// A durable human-approval pause the delegation graph is parked on.
///
/// Produced when a run reaches the `approval` interrupt (see
/// [`DelegationConfig::require_review_approval`]). The pause is already
/// persisted as a checkpoint keyed by `thread_id`; the approver's decision is
/// delivered later via [`resume_delegation`], which survives a process restart.
#[derive(Debug, Clone)]
pub struct PendingApproval {
    /// Stable id of the emitted interrupt (matches a resume value to this pause).
    pub interrupt_id: String,
    /// The node that emitted the interrupt (always `"approval"` here).
    pub node: String,
    /// Approval-request payload presented to the approver (review notes, etc.).
    pub payload: Value,
    /// Thread id the paused graph is checkpointed under; the resume key.
    pub thread_id: String,
}

/// Outcome of a durable delegation run or resume.
#[derive(Debug, Clone)]
pub struct DelegationOutcome {
    /// The latest committed [`DelegationState`] at the run/resume boundary.
    pub state: DelegationState,
    /// `Some` when the run is parked on a durable human-approval interrupt;
    /// `None` when the run reached a terminal (finalized) boundary.
    pub pending: Option<PendingApproval>,
}
