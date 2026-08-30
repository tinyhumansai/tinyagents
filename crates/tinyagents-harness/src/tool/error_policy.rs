//! Per-tool error handling policy.
//!
//! # The gap this closes
//!
//! [`Tool::call`][super::Tool::call] returns `Result<ToolResult>`, and an `Err`
//! propagates out of the agent loop and kills the run. The only way for a tool
//! to signal a *recoverable* failure — one the model should see and retry
//! differently — is to return `Ok(ToolResult::error(...))`. That convention is
//! documented, but it is a documentation-level contract only: nothing stops a
//! tool from returning `Err` for a routine "file not found", and when one does,
//! a whole agent run dies over something the model could have handled in one
//! more turn.
//!
//! LangChain solves this with `handle_tool_error: bool | str | Callable` on the
//! tool itself, resolved at execution time: a handled error flips the result's
//! status and comes back as a `ToolMessage(status="error")` rather than
//! propagating. [`ToolErrorPolicy`] is the same idea with the variants named.
//!
//! # The one rule that must not be broken
//!
//! **Cancellation and interruption always bubble.** See
//! [`ToolErrorPolicy::apply`].

use crate::error::{Result, TinyAgentsError};

use super::types::{ToolCall, ToolResult};

/// What the harness should do when a tool's [`call`][super::Tool::call] returns
/// `Err`.
///
/// Declared per tool via [`Tool::error_policy`][super::Tool::error_policy].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolErrorPolicy {
    /// Propagate the error and fail the run. The default, and the right choice
    /// for a tool whose failure means the run's premise is broken.
    #[default]
    Fail,

    /// Convert the error into an error [`ToolResult`] carrying the error's own
    /// message, so the model sees what went wrong and can adapt.
    ///
    /// Equivalent to LangChain's `handle_tool_error=True`.
    ReturnToError,

    /// Convert the error into an error [`ToolResult`] carrying this fixed
    /// message instead of the error's own text.
    ///
    /// Use it when the underlying error is not safe or not useful to show a
    /// model — a database error containing a connection string, a stack trace,
    /// an internal identifier. Equivalent to LangChain's
    /// `handle_tool_error="<message>"`.
    Message(String),
}

impl ToolErrorPolicy {
    /// Applies the policy to a tool invocation's outcome.
    ///
    /// # Control-flow errors always bubble
    ///
    /// [`TinyAgentsError::Cancelled`] and [`TinyAgentsError::Interrupted`] are
    /// re-raised **regardless of policy**. They are not tool failures: they are
    /// the run being told to stop, or a human-in-the-loop pause. Converting one
    /// into a tool result would report "the tool failed" to the model, let the
    /// loop continue, and silently defeat a cancel or an interrupt — the exact
    /// reason LangGraph re-raises `GraphBubbleUp` unconditionally in
    /// `ToolNode`, and why its tool-retry middleware does the same. Swallowing
    /// an interrupt is a correctness bug, not a robustness feature.
    ///
    /// Everything else is routed by the policy. `Ok` results pass straight
    /// through untouched — a tool that already returned
    /// `Ok(ToolResult::error(..))` has made its own choice and the policy does
    /// not second-guess it.
    pub fn apply(&self, call: &ToolCall, outcome: Result<ToolResult>) -> Result<ToolResult> {
        let error = match outcome {
            Ok(result) => return Ok(result),
            Err(error) => error,
        };

        if is_control_flow_error(&error) {
            tinyagents_tracing::debug!(
                "[tool::error_policy] bubbling control-flow error from `{}`: {error}",
                call.name
            );
            return Err(error);
        }

        match self {
            ToolErrorPolicy::Fail => {
                tinyagents_tracing::debug!(
                    "[tool::error_policy] failing run on `{}` error: {error}",
                    call.name
                );
                Err(error)
            }
            ToolErrorPolicy::ReturnToError => {
                tinyagents_tracing::debug!(
                    "[tool::error_policy] converting `{}` error to a tool result: {error}",
                    call.name
                );
                Ok(ToolResult::error(
                    call.id.clone(),
                    call.name.clone(),
                    error.to_string(),
                ))
            }
            ToolErrorPolicy::Message(message) => {
                tinyagents_tracing::debug!(
                    "[tool::error_policy] masking `{}` error behind a fixed message: {error}",
                    call.name
                );
                Ok(ToolResult::error(
                    call.id.clone(),
                    call.name.clone(),
                    message.clone(),
                ))
            }
        }
    }
}

/// Whether `error` is a control-flow signal that must never be converted into a
/// tool result.
///
/// Exposed so any execution site — including middleware that wraps tool calls
/// with its own retry or fallback logic — can apply the same rule, rather than
/// each re-deriving which variants are safe to swallow.
pub fn is_control_flow_error(error: &TinyAgentsError) -> bool {
    matches!(
        error,
        TinyAgentsError::Cancelled | TinyAgentsError::Interrupted { .. }
    )
}
