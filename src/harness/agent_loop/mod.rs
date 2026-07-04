//! Default model-tool-model agent loop.
//!
//! This loop is the innermost turn of the recursive-language-model (RLM)
//! runtime: it is where one model call is driven to completion, and because a
//! whole harness can be exposed as a tool
//! ([`crate::harness::subagent::SubAgentTool`]), the very tools this loop
//! executes may themselves be other agents — so "a model calling a model" is
//! just this loop nested inside one of its own tool calls. Each invocation runs
//! inside a [`RunContext`] that tracks recursion depth, fans usage/cost up to a
//! parent run, and observes cooperative cancellation and steering at safe
//! checkpoints.
//!
//! This module implements the harness's standard execution loop as inherent
//! methods on [`crate::harness::runtime::AgentHarness`]: build a model request,
//! invoke the model (with retry and fallback), execute any requested tools,
//! append the tool results, and repeat until the model produces a final
//! assistant message with no tool calls or a configured limit is reached.
//!
//! # Lifecycle
//!
//! 1. Build a [`RunContext`] from the [`RunConfig`] and emit
//!    [`AgentEvent::RunStarted`].
//! 2. Run `before_agent` middleware.
//! 3. Repeatedly:
//!    - enforce the model-call cap and wall-clock deadline (fail-closed),
//!    - build the [`ModelRequest`] from the working messages, registered tool
//!      schemas, and the policy's default response format,
//!    - run `before_model` middleware, emit [`AgentEvent::ModelStarted`],
//!    - resolve and invoke the model with retry + fallback,
//!    - run `after_model` middleware, emit [`AgentEvent::ModelCompleted`], fold
//!      usage into the [`AgentRun`], append the assistant message,
//!    - if the assistant requested tools, execute them (enforcing the tool-call
//!      cap, running `before_tool`/`after_tool`, emitting tool events) and
//!      append the tool results, then continue. Multi-call turns run
//!      concurrently when no tool-wrap middleware is registered — see the
//!      `tools` submodule for the dispatch rules, the semantics preserved in
//!      each mode, and why tool-wrap middleware forces serial execution,
//!    - otherwise extract structured output when configured and break.
//! 4. Run `after_agent` middleware and emit [`AgentEvent::RunCompleted`].
//!
//! On any error the loop emits [`AgentEvent::RunFailed`], fans the error out
//! through `on_error` middleware, and returns the error.
//!
//! # Limits
//!
//! Model and tool caps are enforced by the run context's own
//! [`crate::harness::limits::LimitTracker`], which is synced with
//! [`RunPolicy::limits`][crate::harness::runtime::RunPolicy] once at the start
//! of each run (see [`crate::harness::limits::LimitTracker::sync_call_limits`])
//! so the harness policy and the per-run [`RunConfig`] agree on a single
//! enforced cap instead of silently disagreeing. Each call is checked
//! *before* it is made, returning [`TinyAgentsError::LimitExceeded`] whose
//! message always names the limit that actually tripped. The wall-clock
//! deadline (from the run config) is checked each iteration and surfaces as
//! [`TinyAgentsError::Timeout`].
//!
//! # Backoff
//!
//! Retry backoff durations are *computed* via
//! [`crate::harness::retry::RetryPolicy::backoff_for_attempt`]. Whether the loop
//! actually sleeps for that duration is opt-in: it is off by default (keeping
//! tests fast and deterministic) and enabled per policy via
//! [`crate::harness::retry::RetryPolicy::with_backoff_sleep`], so a real
//! provider integration retries after a genuine, growing delay while unit tests
//! stay sleep-free.

mod types;

pub use types::*;

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{Result, TinyAgentsError};
use crate::harness::cache::{ResponseCache, cache_key};
use crate::harness::context::{MiddlewareControl, RunConfig, RunContext};
use crate::harness::events::{AgentEvent, HarnessRunStatus, LimitKind};
use crate::harness::ids::{CallId, ComponentId, HarnessPhase};
use crate::harness::message::{Message, MessageDelta};
use crate::harness::middleware::{
    AgentRun, BoxModelFuture, BoxToolFuture, ModelBaseCall, ToolBaseCall,
};
use crate::harness::model::{
    ChatModel, ModelDelta, ModelRequest, ModelResolutionSource, ModelResponse, ModelStreamItem,
    ResolvedModel, ResolvedModelBinding, ResponseFormat, StreamAccumulator, ToolChoice,
};
use crate::harness::runtime::{AgentHarness, UnknownToolPolicy};
use crate::harness::structured::{StructuredExtractor, StructuredStrategy};
use crate::harness::tool::{Tool, ToolCall, ToolSchema};
use futures::StreamExt;
use serde_json::Value;

mod entry;
mod model_call;
mod run_loop;
mod stream;
mod tools;

pub use stream::AgentStreamItem;

#[cfg(test)]
mod test;
