//! Built-in middleware library.
//!
//! This module collects the ready-to-use middleware that ship with the harness.
//! They are split across two extension surfaces from
//! [`crate::harness::middleware`]:
//!
//! - **Resilience (wrap)** — [`RetryMiddleware`], [`TimeoutMiddleware`],
//!   [`ModelFallbackMiddleware`], and [`RateLimitMiddleware`] implement the
//!   around-call [`ModelMiddleware`] trait and surround the real model call.
//! - **Policy / guard / observation (lifecycle)** —
//!   [`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`], and
//!   [`TracingMiddleware`] implement the lifecycle [`Middleware`] trait.
//!
//! Type definitions live in [`types`]; this file holds the constructors and
//! trait impls. Tests live in `test.rs`.
//!
//! # Testability
//!
//! None of these middleware sleep on the wall clock in a way that tests cannot
//! control: [`RetryMiddleware`] sleeps on backoff only when its policy opts in
//! via [`RetryPolicy::with_backoff_sleep`] (off by default),
//! [`TimeoutMiddleware`] is exercised under `tokio::time` paused-time tests, and
//! [`RateLimitMiddleware`] takes an injectable clock and a configurable poll
//! interval so its wait loop can be driven deterministically.

mod types;

pub use types::*;

use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::AgentEvent;
use crate::harness::ids::CallId;
use crate::harness::message::{ContentBlock, Message};
use crate::harness::middleware::{
    Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware,
};
use crate::harness::model::{ModelDelta, ModelRequest, ModelResponse, ResponseFormat};
use crate::harness::retry::{RateLimiter, RetryPolicy, is_retryable};
use crate::harness::structured::{StructuredExtractor, StructuredStrategy};
use crate::harness::tool::{ToolCall, ToolDelta, ToolResult, ToolSchema};

mod budget;
mod observe;
mod resilience;
mod tool_policy;

#[cfg(test)]
mod test;
