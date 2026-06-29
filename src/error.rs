//! Crate-wide error type and `Result` alias.
//!
//! Every fallible surface of the recursive runtime — graph execution, the
//! harness agent loop, sub-agent recursion, `.rag`/`.ragsh` compilation, and
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
/// and `.rag`/`.ragsh` language processing.
#[derive(Debug, Error)]
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

    /// A model provider call failed (transport error, non-2xx status, or a
    /// malformed response). The payload is a human-readable, provider-normalized
    /// description.
    #[error("model error: {0}")]
    Model(String),

    /// A tool invocation returned an error. The payload describes the failure.
    #[error("tool error: {0}")]
    Tool(String),

    /// A run referenced a tool name that is not present in the
    /// [`crate::harness::tool::ToolRegistry`]. The payload is the tool name.
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

    /// The run exceeded its wall-clock deadline.
    #[error("run timed out: {0}")]
    Timeout(String),

    /// The run was cancelled before completion.
    #[error("run cancelled")]
    Cancelled,

    /// A middleware hook reported a failure.
    #[error("middleware error: {0}")]
    Middleware(String),

    /// A steering command was rejected because the run's
    /// [`crate::harness::steering::SteeringPolicy`] does not allow it, or it
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

    /// A checkpoint could not be written, read, or located.
    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    /// Resume was requested but checkpointing was not configured or no
    /// checkpoint was found.
    #[error("cannot resume: {0}")]
    Resume(String),

    // --- language / blueprint errors ---
    /// A `.rag`/`.ragsh` source could not be tokenised or parsed.
    #[error("parse error at line {line}, column {column}: {message}")]
    Parse {
        message: String,
        line: usize,
        column: usize,
    },

    /// Lowering a parsed blueprint into graph/harness structures failed.
    #[error("compile error: {0}")]
    Compile(String),

    /// A capability (model, tool, route fn) referenced by source is not
    /// registered or is not allowlisted.
    #[error("capability error: {0}")]
    Capability(String),

    /// A named capability with the same [`crate::registry::ComponentKind`] and
    /// name is already registered in a
    /// [`crate::registry::CapabilityRegistry`]. The payload names the offending
    /// kind and name. Use an explicit `replace_*` method to overwrite an
    /// existing registration instead.
    #[error("duplicate component: {0}")]
    DuplicateComponent(String),

    /// A `serde_json` (de)serialization failure, automatically converted from
    /// [`serde_json::Error`] via `?` wherever JSON is read or written
    /// (checkpoints, model wire formats, structured output, blueprints).
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
