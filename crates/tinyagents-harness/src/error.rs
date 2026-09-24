//! Crate-wide error type and `Result` alias.
//!
//! Every fallible surface of the recursive runtime — graph execution, the
//! harness agent loop, sub-agent recursion, and
//! registry binding — funnels through [`TinyAgentsError`] so failures from a
//! deeply nested child run roll up to the caller through one uniform type.
//! Downstream code should prefer the [`Result`] alias exported here.

use thiserror::Error;

/// Convenience alias for `std::result::Result<T, TinyAgentsError>` used
/// throughout the crate's public API.
pub type Result<T> = std::result::Result<T, TinyAgentsError>;

/// The single error type returned by every fallible TinyAgents operation.
///
/// Variants are grouped by the surface that raises them: graph construction and
/// execution, model/tool invocation, run limits and policy, graph durability,
/// and graph execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TinyAgentsError {
    /// A graph was compiled or run without a configured `START` edge, so there
    /// is no entry node to begin execution from.
    #[error("graph start node is not configured")]
    MissingStart,

    /// An edge, route, or run referenced a node name that is not present in the
    /// graph. The payload is the missing node name.
    #[error("node `{0}` does not exist")]
    MissingNode(String),

    /// An edge declares a destination node that does not exist. The payload is
    /// the missing target name.
    #[error("edge points to missing node `{0}`")]
    MissingEdgeTarget(String),

    /// A conditional router returned a `route` label that is not wired to any
    /// destination from `node`.
    #[error("conditional route `{route}` from node `{node}` does not exist")]
    MissingRoute { node: String, route: String },

    /// Graph execution performed more super-steps than the configured recursion
    /// limit allows (typically an unintended cycle). The payload is the limit
    /// that was hit. Contrast with [`TinyAgentsError::SubAgentDepth`], which
    /// counts nested run-tree levels rather than super-steps.
    #[error("graph exceeded the recursion limit of {0} steps")]
    RecursionLimit(usize),

    /// A sub-agent invocation would exceed the configured maximum recursion
    /// depth. The payload is the `max_depth` cap that was reached.
    ///
    /// This is distinct from [`TinyAgentsError::RecursionLimit`], which counts
    /// graph *super-steps*; `SubAgentDepth` counts nested run-tree *levels*
    /// (parent → child → grandchild …) so the two limits can be reasoned about
    /// and surfaced independently.
    #[error("sub-agent recursion exceeded the maximum depth of {0}")]
    SubAgentDepth(usize),

    /// A single graph node was activated more times within one run than the
    /// `RecursionPolicy`'s (in `tinyagents-graph`) `max_visits_per_node` allows (an
    /// unbounded node-loop). This is node-loop recursion, tracked separately
    /// from [`TinyAgentsError::RecursionLimit`] (total super-steps) and
    /// [`TinyAgentsError::SubAgentDepth`] (run-tree depth).
    #[error("node `{node}` exceeded its visit limit of {limit}")]
    NodeVisitLimit {
        /// The node that was over-visited.
        node: String,
        /// The configured per-node visit cap.
        limit: usize,
    },

    /// A model provider call failed (transport error, non-2xx status, or a
    /// malformed response). The payload is a human-readable, provider-normalized
    /// description.
    ///
    /// Prefer [`TinyAgentsError::Provider`] when the structured failure detail
    /// (HTTP status, provider error code, retryability) is available — this
    /// variant remains for transport-level and parsing failures that have no
    /// such structure to preserve.
    #[error("model error: {0}")]
    Model(String),

    /// A model provider call failed with the full structured detail preserved
    /// — HTTP status, provider error code, and whether retrying the same
    /// request may succeed — instead of flattened into a display string.
    ///
    /// Real provider adapters (for example the OpenAI unary and streaming
    /// paths) raise this instead of [`TinyAgentsError::Model`] whenever they
    /// have a [`tinyinference_llm::model::ProviderError`] in hand, so
    /// [`crate::retry::is_retryable`] can classify retryability from
    /// [`tinyinference_llm::model::ProviderError::retryable`] (a 429 is
    /// retryable; a 401 is not) rather than retrying every provider failure
    /// indiscriminately. Boxed so this one variant's larger payload does not
    /// inflate every `Result<T, TinyAgentsError>` in the crate
    /// (`clippy::result_large_err`).
    #[error("model error: {0}")]
    Provider(Box<tinyinference_llm::model::ProviderError>),

    /// The request did not fit in the model's context window.
    ///
    /// Distinguished from the generic [`TinyAgentsError::Provider`] because the
    /// remedy is specific and mechanical — compact or drop transcript history
    /// and retry — where a generic provider failure has none. A caller that can
    /// summarise its own transcript (see
    /// [`crate::summarization`]) can match on this variant instead of
    /// string-matching a provider message that differs per vendor and changes
    /// without notice. Port of LangChain's `ContextOverflowError`.
    ///
    /// # Detection is best-effort, and asymmetric
    ///
    /// Hosted providers raise an explicit 400 for this, which
    /// [`tinyinference_llm::providers::openai::CONTEXT_OVERFLOW_CODE`] classifies.
    /// **Local servers usually truncate the front of the prompt silently
    /// instead**, so the absence of this error is not evidence that the prompt
    /// fitted — pair it with a probed real context window
    /// ([`tinyinference_llm::providers::openai::LocalProbe`]) rather than relying
    /// on it alone.
    #[error("context overflow: {message}")]
    ContextOverflow {
        /// Provider family identifier, for example `openai` or `ollama`.
        provider: String,
        /// Provider model id, when known.
        model: Option<String>,
        /// The provider's own message, preserved verbatim.
        message: String,
    },

    /// A tool invocation returned an error. The payload describes the failure.
    #[error("tool error: {0}")]
    Tool(String),

    /// A tool or an output validator ([`crate::structured::OutputValidator`])
    /// reported a *recoverable* failure the model should be asked to fix and
    /// retry, rather than one that ends the run.
    ///
    /// Mirrors Pydantic AI's `ModelRetry`. A tool returning this from
    /// [`tinytools::Tool::execute`] is folded into a recoverable
    /// [`tinytools::ToolResult::retry`] result instead of aborting the run —
    /// see `agent_loop/tools.rs`'s `map_tool_dispatch_error`. An
    /// [`crate::structured::OutputValidator`] returning it on the agent
    /// loop's final turn drives the output-validation retry loop (A3): the
    /// message is pushed back to the model as a repair prompt and the turn
    /// continues, bounded by
    /// [`crate::runtime::RunPolicy::output_retry`]'s `max_attempts`.
    /// Contrast with [`Self::ToolFailed`], which is permanent.
    #[error("retryable failure: {0}")]
    ModelRetry(String),

    /// A tool reported a **permanent** failure that must not be retried —
    /// the counterpart to [`Self::ModelRetry`]. Folded into a
    /// [`tinytools::ToolResult::failed`] result (still recoverable at the
    /// transcript level — the model sees the message — but
    /// [`crate::middleware::library::RetryMiddleware`] and any other retry policy treat it
    /// as non-retryable rather than re-attempting the call).
    #[error("permanent tool failure: {0}")]
    ToolFailed(String),

    /// A tool (from [`tinytools::Tool::execute`]) or a `before_tool`
    /// middleware asked for **human approval** before this call runs (A2).
    ///
    /// The agent loop does not treat this as a failure: it finishes the rest
    /// of the batch, lists the call under
    /// [`crate::tool::DeferredToolRequests::approvals`] with `metadata`
    /// attached, and exits with `AgentRun::deferred` set (or resolves it
    /// inline through a registered
    /// [`crate::tool::DeferredToolHandler`]). Mirrors Pydantic AI's
    /// `ApprovalRequired`. Never retried by [`crate::retry::is_retryable`].
    #[error("tool call requires approval")]
    ApprovalRequired {
        /// Host-only context for the approver (never shown to the model).
        metadata: serde_json::Value,
    },

    /// A tool asked the **host** to execute this call out of band (A2):
    /// the loop lists it under [`crate::tool::DeferredToolRequests::calls`]
    /// and expects a [`crate::tool::DeferredCallResult`] on resume. Raised
    /// automatically for a tool registered through
    /// [`crate::tool::ToolRegistry::register_external`], and equally by
    /// [`crate::tool::toolset::ExternalToolSet`]'s
    /// [`crate::tool::toolset::ToolSet::call`] (gap B3, mirroring Pydantic
    /// AI's `defer_loading`/deferred-tools model) — both call sites raise
    /// this same variant so a host sees one deferred-call signal regardless
    /// of which registration path advertised the tool. `metadata` is
    /// host-only context describing how to execute the call; the call's own
    /// name and arguments are already carried on the
    /// [`crate::tool::DeferredToolRequests`] entry, so most callers leave it
    /// `Value::Null`. Mirrors Pydantic AI's `CallDeferred`. Never retried by
    /// [`crate::retry::is_retryable`].
    #[error("tool call deferred to the host")]
    CallDeferred {
        /// Host-only context describing how to execute the call.
        metadata: serde_json::Value,
    },

    /// A run referenced a tool name that is not present in the
    /// [`crate::tool::ToolRegistry`]. The payload is the tool name.
    #[error("tool `{0}` is not registered")]
    ToolNotFound(String),

    /// A run referenced a model name that is not registered. The payload is the
    /// model name.
    #[error("model `{0}` is not registered")]
    ModelNotFound(String),

    /// Input failed validation before a call was made (for example a missing
    /// API key or an empty required field). The payload describes the problem.
    #[error("validation error: {0}")]
    Validation(String),

    /// Parsing or validating a model's structured (JSON-schema) output failed.
    #[error("structured output error: {0}")]
    StructuredOutput(String),

    // --- run/limit/policy errors ---
    /// A configured run limit (model calls, tool calls, wall clock) was exceeded.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    /// The provider returned an empty completion — no text, no tool calls, and
    /// no structured output — while
    /// [`crate::runtime::RunPolicy`]'s `error_on_empty_response` guard
    /// was enabled. Raised in the agent loop's finalization branch instead of
    /// terminating the run with a blank final answer, so the caller can
    /// re-prompt or surface a real failure rather than silently succeeding with
    /// empty content.
    #[error("model returned an empty response")]
    EmptyResponse,

    /// The run exceeded its wall-clock deadline.
    ///
    /// Terminal: the run itself is out of time, so retrying or falling back
    /// to another model would just spin until the next deadline check fails
    /// identically. See [`TinyAgentsError::CallTimeout`] for the per-call
    /// counterpart, which *is* retryable.
    #[error("run timed out: {0}")]
    Timeout(String),

    /// A single call (currently: a model call bounded by
    /// [`crate::limits::RunLimits::max_model_call_ms`]) ran past its own
    /// ceiling while the run still has wall-clock budget left.
    ///
    /// Unlike [`TinyAgentsError::Timeout`], this does not mean the run is out
    /// of time — it means *this one call* wedged. [`crate::retry::is_retryable`]
    /// treats it as transient, and the model-resolution retry/fallback loop
    /// (`invoke_model_resolving`) does not treat it as a reason to skip the
    /// fallback chain the way it does a run-deadline `Timeout`.
    #[error("call timed out: {0}")]
    CallTimeout(String),

    /// The run was cancelled before completion.
    #[error("run cancelled")]
    Cancelled,

    /// A middleware hook reported a failure.
    #[error("middleware error: {0}")]
    Middleware(String),

    /// A steering command was rejected because the run's
    /// [`crate::steering::SteeringPolicy`] does not allow it, or it
    /// could not be applied. The payload is a human-readable description naming
    /// the offending command kind.
    #[error("steering error: {0}")]
    Steering(String),

    /// A memory backend operation failed.
    #[error("memory error: {0}")]
    Memory(String),

    /// An embedding model, vector store, or retriever operation failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    // --- graph durability errors ---
    /// Generic graph runtime error.
    #[error("graph error: {0}")]
    Graph(String),

    /// Execution was interrupted (human-in-the-loop / external approval).
    #[error("graph interrupted at node `{node}`: {message}")]
    Interrupted { node: String, message: String },

    /// Two or more concurrent branches in a single superstep wrote the same
    /// non-aggregate channel (for example a `LastValue` channel, in
    /// `tinyagents-graph`), so the merge cannot pick a single deterministic winner. Use an
    /// aggregate channel (one whose `Channel::allows_concurrent` is `true`, such as
    /// `Topic` or `BinaryAggregate`) when fan-out branches must
    /// write the same key. The payload describes the offending channel.
    #[error("invalid concurrent update: {0}")]
    InvalidConcurrentUpdate(String),

    /// A checkpoint could not be written, read, or located.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// Resume was requested but checkpointing was not configured or no
    /// checkpoint was found.
    #[error("cannot resume: {0}")]
    Resume(String),

    /// A capability (model, tool, route fn) referenced by source is not
    /// registered or is not allowlisted.
    #[error("capability error: {0}")]
    Capability(String),

    /// A named capability with the same `ComponentKind` and
    /// name is already registered in a
    /// `CapabilityRegistry` (in `tinyagents-registry`). The payload names the offending
    /// kind and name. Use an explicit `replace_*` method to overwrite an
    /// existing registration instead.
    #[error("duplicate component: {0}")]
    DuplicateComponent(String),

    /// A `serde_json` (de)serialization failure, automatically converted from
    /// [`serde_json::Error`] via `?` wherever JSON is read or written
    /// (checkpoints, model wire formats, and structured output).
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A durable-storage operation failed — opening, migrating, reading, or
    /// writing a backing database for the session store and run ledger
    /// (`tinyagents-session`).
    ///
    /// Distinct from [`TinyAgentsError::Checkpoint`], which covers graph
    /// checkpoint durability: a session-store failure means run *history* could
    /// not be recorded or queried, while a checkpoint failure means a run
    /// cannot be resumed. The payload carries the operation context and the
    /// underlying driver message.
    #[error("storage error: {0}")]
    Storage(String),
}

impl From<tinyinference_llm::Error> for TinyAgentsError {
    fn from(error: tinyinference_llm::Error) -> Self {
        match error {
            tinyinference_llm::Error::Model(message) => Self::Model(message),
            tinyinference_llm::Error::Provider(error) => Self::from_provider_error(*error),
            tinyinference_llm::Error::Validation(message) => Self::Validation(message),
            tinyinference_llm::Error::Serialization(error) => Self::Serialization(error),
            tinyinference_llm::Error::Catalog(message) => Self::Model(message),
            tinyinference_llm::Error::Unsupported(message) => Self::Validation(message),
        }
    }
}

impl From<tinyinference_embeddings::Error> for TinyAgentsError {
    fn from(error: tinyinference_embeddings::Error) -> Self {
        match error {
            tinyinference_embeddings::Error::Validation(message) => Self::Validation(message),
            tinyinference_embeddings::Error::Serialization(error) => Self::Serialization(error),
            tinyinference_embeddings::Error::Embedding(message) => Self::Embedding(message),
            tinyinference_embeddings::Error::Cancelled => Self::Cancelled,
        }
    }
}

#[cfg(feature = "media")]
impl From<tinyinference_image::Error> for TinyAgentsError {
    fn from(error: tinyinference_image::Error) -> Self {
        match error {
            tinyinference_image::Error::Serialization(error) => Self::Serialization(error),
            error @ (tinyinference_image::Error::Validation(_)
            | tinyinference_image::Error::Unsupported { .. }) => {
                Self::Validation(error.to_string())
            }
            other => Self::Model(other.to_string()),
        }
    }
}

#[cfg(feature = "media")]
impl From<tinyinference_video::Error> for TinyAgentsError {
    fn from(error: tinyinference_video::Error) -> Self {
        match error {
            tinyinference_video::Error::Media(error) => error.into(),
            error @ tinyinference_video::Error::Timeout { .. } => Self::Timeout(error.to_string()),
            other => Self::Model(other.to_string()),
        }
    }
}

impl TinyAgentsError {
    /// Builds the right error for a structured provider failure, promoting a
    /// recognised context overflow to [`TinyAgentsError::ContextOverflow`].
    ///
    /// Provider adapters classify the overflow and stamp
    /// [`CONTEXT_OVERFLOW_CODE`][code] on
    /// [`ProviderError::code`][pc]; this is where that code becomes a type. Use
    /// it in place of `TinyAgentsError::Provider(Box::new(error))` at every
    /// site that has a `ProviderError` in hand — the generic variant is still
    /// correct for everything else, and is what this returns when the code is
    /// absent or unrecognised.
    ///
    /// [code]: tinyinference_llm::providers::openai::CONTEXT_OVERFLOW_CODE
    /// [pc]: tinyinference_llm::model::ProviderError::code
    pub fn from_provider_error(error: tinyinference_llm::model::ProviderError) -> Self {
        if error.code.as_deref()
            == Some(tinyinference_llm::providers::openai::CONTEXT_OVERFLOW_CODE)
        {
            tracing::debug!(
                "[error] promoting provider `{}` context-overflow code to a typed error",
                error.provider
            );
            return Self::ContextOverflow {
                provider: error.provider,
                model: error.model,
                message: error.message,
            };
        }
        Self::Provider(Box::new(error))
    }

    /// Whether this error means the request did not fit the model's context
    /// window.
    ///
    /// Recognises **both** the typed [`TinyAgentsError::ContextOverflow`] and a
    /// [`TinyAgentsError::Provider`] still carrying the classification code, so
    /// a caller's compact-and-retry logic behaves identically no matter which
    /// construction site produced the error. Call sites are migrating to
    /// [`Self::from_provider_error`]; until every one has, the two shapes must
    /// classify the same or the same failure would be handled two ways.
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::ContextOverflow { .. } => true,
            Self::Provider(error) => {
                error.code.as_deref()
                    == Some(tinyinference_llm::providers::openai::CONTEXT_OVERFLOW_CODE)
            }
            _ => false,
        }
    }
}

/// Converts a raw `rusqlite` failure into [`TinyAgentsError::Storage`] so the
/// session store and run ledger can use `?` on driver calls directly. Call
/// sites that have useful context to add should still map explicitly rather
/// than relying on this bare conversion.
#[cfg(feature = "sqlite")]
impl From<rusqlite::Error> for TinyAgentsError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Storage(err.to_string())
    }
}
