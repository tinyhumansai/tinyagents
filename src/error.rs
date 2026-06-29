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

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
