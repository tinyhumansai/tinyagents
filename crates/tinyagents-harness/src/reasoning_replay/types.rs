//! Types for choosing which turns' reasoning a request replays.

use serde::{Deserialize, Serialize};

/// Which turns' reasoning blocks the requests of a run carry.
///
/// Applied when a request is built; the stored transcript always keeps every
/// reasoning block. A turn is the span after a user message, so the current
/// turn is everything after the last one. Defaults to [`Self::All`], which
/// sends the history untouched, so a request is byte-identical unless the
/// host opts in. A session overrides it per run with
/// [`crate::context::RunConfig::replay_reasoning`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningReplay {
    /// Every assistant message keeps its reasoning blocks (today's behaviour).
    #[default]
    All,
    /// Only the current turn's reasoning is sent. Reasoning from completed
    /// user turns is omitted from the request.
    CurrentTurn,
}
