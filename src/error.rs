use thiserror::Error;

pub type Result<T> = std::result::Result<T, RustAgentsError>;

#[derive(Debug, Error)]
pub enum RustAgentsError {
    #[error("graph start node is not configured")]
    MissingStart,

    #[error("node `{0}` does not exist")]
    MissingNode(String),

    #[error("edge points to missing node `{0}`")]
    MissingEdgeTarget(String),

    #[error("conditional route `{route}` from node `{node}` does not exist")]
    MissingRoute { node: String, route: String },

    #[error("graph exceeded the recursion limit of {0} steps")]
    RecursionLimit(usize),

    #[error("model error: {0}")]
    Model(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("tool `{0}` is not registered")]
    ToolNotFound(String),

    #[error("model `{0}` is not registered")]
    ModelNotFound(String),

    #[error("validation error: {0}")]
    Validation(String),

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

    /// A memory backend operation failed.
    #[error("memory error: {0}")]
    Memory(String),

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

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
