/// Errors emitted by the host-neutral session runtime.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeError {
    /// A caller cancelled the turn at a runtime await boundary.
    #[error("session turn cancelled")]
    Cancelled,
    /// A driver could not invoke the model/tool loop.
    #[error("driver failed: {0}")]
    Driver(String),
    /// Transcript decoding or encoding failed.
    #[error("transcript codec failed: {0}")]
    Codec(String),
    /// Durable transcript persistence failed.
    #[error("transcript persistence failed: {0}")]
    Persistence(String),
    /// A host lifecycle hook failed.
    #[error("session hook failed: {0}")]
    Hook(String),
    /// The host supplied two distinct tool declarations with one name.
    #[error("tool snapshot has conflicting declarations for `{0}`")]
    ToolNameCollision(String),
    /// A harness cannot bind its registered tools to this turn's frozen snapshot.
    #[error("harness tools do not match the session tool snapshot")]
    ToolSnapshotMismatch,
    /// The builder did not receive a required dependency.
    #[error("session builder requires {0}")]
    MissingDependency(&'static str),
    /// A host attempted a state transition which would invalidate a committed
    /// session invariant.
    #[error("invalid session state: {0}")]
    InvalidSessionState(String),
}
