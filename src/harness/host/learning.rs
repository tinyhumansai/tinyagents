//! Completed work handed to the embedder for durable knowledge extraction.
//!
//! See [`LearningSink`] for the trait contract and [`NoopLearningSink`] for the
//! inert default.

use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::harness::ids::{RunId, ThreadId};

/// Receives completed work so the embedder can derive durable knowledge from
/// it.
///
/// Fire-and-forget by contract: the runtime does not wait on the result and
/// treats an `Err` as a logged failure, never a run failure. Implementations
/// must return promptly and defer real work to a background task.
#[async_trait]
pub trait LearningSink<State: Send + Sync>: Send + Sync {
    /// Called once after a turn produces its final output.
    async fn on_turn_completed(&self, state: &State, record: &TurnRecord) -> Result<()>;

    /// Called after a turn's messages are durably written, naming the artifact
    /// so an ingester can read it without holding the messages in memory.
    ///
    /// The default does nothing.
    async fn on_transcript_committed(
        &self,
        state: &State,
        commit: &TranscriptCommit<'_>,
    ) -> Result<()> {
        let _ = (state, commit);
        Ok(())
    }
}

/// One completed turn, summarized for downstream extraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRecord {
    /// The run the turn belongs to.
    pub run_id: RunId,
    /// Thread the run belongs to, when it belongs to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<ThreadId>,
    /// Host-defined identity of the agent that took the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Host-defined label for how the run was entered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<String>,
    /// The user input that opened the turn.
    pub input: String,
    /// The assistant's final output for the turn.
    pub output: String,
    /// Tool calls made during the turn, in call order.
    #[serde(default)]
    pub tool_calls: Vec<ToolOutcomeRecord>,
    /// Number of model calls the turn consumed.
    pub model_calls: u32,
    /// Wall-clock duration of the turn.
    pub elapsed_ms: u64,
}

/// One tool call within a [`TurnRecord`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcomeRecord {
    /// Name of the tool that ran.
    pub name: String,
    /// Arguments the model supplied.
    pub arguments: Value,
    /// Whether the call succeeded.
    pub succeeded: bool,
    /// Bounded, non-sensitive description of the result. Never raw output.
    pub summary: String,
    /// Wall-clock duration of the call.
    pub elapsed_ms: u64,
}

/// A durably written transcript, named rather than inlined.
#[derive(Clone, Debug)]
pub struct TranscriptCommit<'a> {
    /// The run whose messages were written.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Where the transcript lives on disk.
    pub path: &'a Path,
    /// How many messages this commit appended.
    pub appended_messages: u64,
}

/// A [`LearningSink`] that learns nothing.
///
/// Registering it is indistinguishable from registering nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopLearningSink;

impl NoopLearningSink {
    /// Creates the sink.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> LearningSink<State> for NoopLearningSink {
    async fn on_turn_completed(&self, _state: &State, _record: &TurnRecord) -> Result<()> {
        Ok(())
    }
}
