use std::sync::Arc;

use async_trait::async_trait;
use tinyagents_harness::{context::RunContext, runtime::AgentHarness};
use tinyinference_llm::message::Message;

use crate::{RuntimeError, ToolSnapshot, TranscriptPartial};

/// Immutable values passed from a [`crate::Session`] to one driver invocation.
pub struct DriverRequest<C = ()> {
    /// Full model history, including the stable prefix and deduplicated input.
    pub history: Vec<Message>,
    /// Frozen model-visible tool declarations selected by the host.
    pub tools: ToolSnapshot,
    /// Explicit live run context for this invocation.
    pub run_context: RunContext<C>,
    /// Requests the driver's streaming execution path.
    pub stream: bool,
}

/// A driver's complete or partial model result.
#[derive(Clone, Debug, PartialEq)]
pub struct DriverOutcome {
    /// Complete history accumulated by the harness/driver.
    pub history: Vec<Message>,
    /// Final visible output, when available.
    pub output: Option<String>,
    /// Display-only partial text when execution was interrupted.
    pub partial: Option<TranscriptPartial>,
    /// Whether the driver ended at an interruptible point.
    pub interrupted: bool,
}

/// A driver error which may retain model history and display-only partial text.
#[derive(Clone, Debug)]
pub struct DriverFailure {
    /// The reason execution failed.
    pub error: RuntimeError,
    /// Work produced before failure, safe for partial transcript persistence.
    pub partial: Option<DriverOutcome>,
}

/// Object-safe model/tool-loop invocation boundary.
///
/// The generic `AgentHarness<State, Ctx>` cannot be stored directly in a
/// reusable `Session` without leaking a host's state type. Implement this seam
/// directly, or use [`HarnessDriver`] for `AgentHarness<State, C>`.
#[async_trait]
pub trait SessionDriver<C: Send + Sync + 'static = ()>: Send + Sync {
    /// Executes one turn from an immutable runtime snapshot.
    async fn execute(&self, request: DriverRequest<C>) -> Result<DriverOutcome, DriverFailure>;
}

/// Adapter for the ordinary, explicit-model TinyAgents harness entry points.
///
/// Tool selection remains outside this adapter: callers configure the harness
/// from the same already-authorized [`ToolSnapshot`] that they expose here.
pub struct HarnessDriver<State: Send + Sync + 'static, C: Send + Sync + 'static = ()> {
    harness: Arc<AgentHarness<State, C>>,
    state: Arc<State>,
}

impl<State: Send + Sync + 'static, C: Send + Sync + 'static> HarnessDriver<State, C> {
    /// Binds reusable harness mechanics and immutable host state to a driver.
    pub fn new(harness: Arc<AgentHarness<State, C>>, state: Arc<State>) -> Self {
        Self { harness, state }
    }
}

#[async_trait]
impl<State: Send + Sync + 'static, C: Send + Sync + 'static> SessionDriver<C>
    for HarnessDriver<State, C>
{
    async fn execute(&self, request: DriverRequest<C>) -> Result<DriverOutcome, DriverFailure> {
        // The harness has no per-invocation registry override.  Executing when
        // its provider-visible declarations differ from the frozen request
        // would advertise or dispatch capabilities the session did not grant.
        if !same_tools(
            &self.harness.tools().declared_specs(),
            request.tools.specs(),
        ) {
            return Err(DriverFailure {
                error: RuntimeError::ToolSnapshotMismatch,
                partial: None,
            });
        }
        let partial = if request.stream {
            self.harness
                .invoke_streaming_in_context_collecting_partial(
                    self.state.as_ref(),
                    request.run_context,
                    request.history,
                )
                .await
        } else {
            self.harness
                .invoke_in_context_collecting_partial(
                    self.state.as_ref(),
                    request.run_context,
                    request.history,
                )
                .await
        };
        let output =
            partial.run.messages.iter().rev().find_map(|message| {
                matches!(message, Message::Assistant(_)).then(|| message.text())
            });
        let outcome = DriverOutcome {
            history: partial.run.messages,
            output: output.clone(),
            // The pinned harness retains partial model history but exposes no
            // separate streamed text delta, reasoning, or iteration here. The
            // accumulated final assistant text is the only display-safe value.
            partial: partial
                .error
                .as_ref()
                .and(output)
                .map(TranscriptPartial::new),
            interrupted: partial.run.paused.is_some(),
        };
        match partial.error {
            Some(error) => Err(DriverFailure {
                error: RuntimeError::Driver(error.to_string()),
                partial: Some(outcome),
            }),
            None => Ok(outcome),
        }
    }
}

fn same_tools(left: &[tinytools::ToolSpec], right: &[tinytools::ToolSpec]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.name == right.name
                && left.description == right.description
                && left.parameters == right.parameters
        })
}
