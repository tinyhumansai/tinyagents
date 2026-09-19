use async_trait::async_trait;

use crate::{
    CommitReceipt, ResumePreparation, RuntimeError, SessionStateView, SessionTerminal,
    SessionTurnOutcome, SessionTurnRequest, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

/// Host observation/preparation around a session turn.
///
/// Hooks do not grant tools, select models, compose product prompts, or own
/// transcript state. A session serializes these mutable calls, so a host can
/// keep preparation state without task-local runtime state.
#[async_trait]
pub trait SessionHooks<C: Clone + Send + Sync + 'static = ()>: Send + Sync {
    /// Runs after the turn's initial cancellation check and before transcript
    /// target binding or resume. It may mutate the request and live options.
    /// Its target remains lazy until resume or the first append needs it.
    async fn before_resume(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions<C>,
        _: SessionStateView<'_>,
    ) -> Result<ResumePreparation, RuntimeError> {
        Ok(ResumePreparation::default())
    }
    /// Runs before the driver sees the request and consumes the explicit
    /// options, after any requested transcript has been loaded. Returned
    /// values apply only to this driver invocation.
    async fn before_turn(
        &self,
        request: &mut SessionTurnRequest,
        options: &mut TurnOptions<C>,
        state: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError>;
    /// Runs after the driver has produced a candidate and before it commits.
    /// Returning an error or observing cancellation therefore leaves no
    /// durable session mutation behind.
    async fn before_commit(
        &self,
        outcome: &SessionTurnOutcome,
        options: &TranscriptTurnOptions<C>,
    ) -> Result<(), RuntimeError>;
    /// Runs exactly once after a successful durable transcript commit.
    /// Errors and cancellation observed here are deliberately observational:
    /// the result has already become durable and remains successful.
    async fn after_commit(&self, _: CommitReceipt<C>) -> Result<(), RuntimeError> {
        Ok(())
    }
    /// Runs exactly once for every terminal turn result.
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError>;
}

/// A no-op hook set for hosts that need no lifecycle observation.
#[derive(Default)]
pub struct NoopSessionHooks;

#[async_trait]
impl<C: Clone + Send + Sync + 'static> SessionHooks<C> for NoopSessionHooks {
    async fn before_resume(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions<C>,
        _: SessionStateView<'_>,
    ) -> Result<ResumePreparation, RuntimeError> {
        Ok(ResumePreparation::default())
    }
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions<C>,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions<C>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn after_commit(&self, _: CommitReceipt<C>) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}
