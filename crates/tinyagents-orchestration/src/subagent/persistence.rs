use async_trait::async_trait;

use super::{
    PersistedSubagentPause, SubagentError, SubagentOutcome, SubagentPausePersistenceDisposition,
    SubagentResume, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

/// Host boundary for durable pause, resume, and terminal lifecycle state.
///
/// Implementations must make `record_terminal` idempotent by
/// [`SubagentTaskKey`] across process boundaries. The key is scoped by root
/// run, immediate parent run, and (when supplied) thread, so a bare task id is
/// never a global lifecycle identity. The driver additionally suppresses
/// duplicate records from repeated calls made through the same driver instance. A persistence
/// future's successful return is its commit boundary: implementations must not
/// make a write visible and then await again before returning `Ok(true)`. The
/// driver races that boundary with cancellation and, when cancellation wins,
/// records one truthful `Cancelled` terminal outcome instead.
#[async_trait]
pub trait SubagentPersistence: Send + Sync {
    /// Returns a terminal outcome committed by another driver/process, if any.
    /// The lifecycle consults this before planning so a durable terminal never
    /// reopens merely because this process has an empty in-memory cache.
    async fn load_terminal(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError>;
    /// Loads the most recent resumable state, if a caller did not supply one.
    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError>;

    /// Loads the complete durable pause outcome for observers that lost a
    /// pause compare-and-swap. This is deliberately richer than [`Self::load`]
    /// so a caller never returns its own discarded output, usage, or artifacts.
    async fn load_pause(
        &self,
        key: &SubagentTaskKey,
    ) -> Result<Option<SubagentOutcome>, SubagentError>;

    /// Saves one resumable pause. The driver never also records a terminal for
    /// that same committed outcome. A paused outcome is deliberately not cached
    /// by the driver; a later call reloads this state and resumes execution.
    /// Atomically creates or advances a pause record. A continuation must
    /// replace only the exact durable pause it loaded; competing continuations
    /// receive `Existing` and return that winner without emitting effects.
    async fn save_pause(
        &self,
        pause: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition, SubagentError>;

    /// Atomically records a terminal outcome. A continuation can consume only
    /// the exact pause it loaded; a fresh lifecycle can close only an unpaused
    /// key. This prevents an independent pause and terminal execution from
    /// both claiming host effects.
    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        replaces: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition, SubagentError>;
}
