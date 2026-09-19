//! Stateful, host-neutral sessions for TinyAgents.
//!
//! A [`Session<C>`] owns conversation history and its durable transcript view.
//! Hosts supply the model/tool execution through [`SessionDriver`] and own all
//! policy, prompt composition, model selection, and authorization.  This
//! crate deliberately has no host configuration or product dependencies.

mod builder;
mod driver;
mod error;
mod hooks;
mod prefix;
mod session;
mod tools;
mod types;

pub use builder::SessionBuilder;
pub use driver::{DriverFailure, DriverOutcome, DriverRequest, HarnessDriver, SessionDriver};
pub use error::RuntimeError;
pub use hooks::{NoopSessionHooks, SessionHooks};
pub use prefix::PrefixSnapshot;
pub use session::Session;
pub use tinyagents_session::transcript::TranscriptPartial;
pub use tools::ToolSnapshot;
pub use types::{
    CommitReceipt, ResumeMode, ResumePreparation, SessionResume, SessionStateView, SessionTerminal,
    SessionTurnOutcome, SessionTurnRequest, TranscriptCommitReceipt, TranscriptDelta,
    TranscriptTarget, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

/// Converts between a host's lossless durable transcript dialect and the
/// inference messages driven by a [`SessionDriver<C>`].
///
/// It is object-safe so a host can keep its conversion and metadata ownership
/// outside this reusable runtime.  The runtime never narrows transcript data
/// itself; a codec must explicitly decide how every field is represented.
pub trait TranscriptCodec<C: Clone + Send + Sync + 'static = ()>: Send + Sync {
    /// Decodes a durable transcript for model execution.
    fn decode_history(
        &self,
        transcript: &tinyagents_session::transcript::SessionTranscript,
    ) -> Result<Vec<tinyinference_llm::message::Message>, RuntimeError>;

    /// Reconciles a model-history transition with the prior lossless durable
    /// rows. `prior` is authoritative for fields absent from inference
    /// messages (provider metadata, raw arguments, reasoning, ids, and host
    /// extensions): unchanged model positions must retain their corresponding
    /// raw row, including when `next` is a compaction replacement. The returned
    /// rows are the complete next logical durable set; the history layer writes
    /// its delta atomically. `options` is captured after `before_turn` mutates
    /// the explicit turn options and before the driver consumes `RunContext`.
    fn reconcile(
        &self,
        prior: &[tinyagents_session::transcript::TranscriptMessage],
        previous: &[tinyinference_llm::message::Message],
        next: &[tinyinference_llm::message::Message],
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Vec<tinyagents_session::transcript::TranscriptMessage>, RuntimeError>;

    /// Returns host-derived provider usage for this transition, if available.
    ///
    /// The runtime calls this after the driver has completed (and therefore
    /// after a host's explicit context sidecars may have been updated), but
    /// before opening or appending a transcript.  The returned value travels
    /// through the same atomic [`tinyagents_session::transcript::TranscriptTurn`]
    /// append as the reconciled rows, where the history implementation attaches
    /// it to the turn's final assistant row.  Generic codecs need no usage
    /// policy, so the default remains `None`.
    fn turn_usage(
        &self,
        _: &TranscriptTurnOptions<C>,
    ) -> Result<Option<tinyagents_session::transcript::TurnUsage>, RuntimeError> {
        Ok(None)
    }
}

#[cfg(test)]
mod test;
