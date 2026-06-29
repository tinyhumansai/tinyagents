//! Harness runtime facade.
//!
//! Owns the high-level [`AgentHarness`] builder and the [`RunPolicy`] bundle
//! that wires registries, middleware, and run policy into a single ergonomic
//! entry point. The agent loop driven by this facade lives in the sibling
//! [`crate::harness::agent_loop`] module.
//!
//! # Layout
//!
//! - [`types`] holds the public type definitions ([`RunPolicy`] and
//!   [`AgentHarness`]).
//! - This file holds the builder, registration, and accessor methods.
//! - `test.rs` holds focused tests for construction and registration.

mod types;

pub use types::*;

use std::sync::Arc;

use crate::harness::middleware::{Middleware, MiddlewareStack};
use crate::harness::model::{ChatModel, ModelRegistry};
use crate::harness::tool::{Tool, ToolRegistry};

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Creates an empty harness with default policy and no models, tools, or
    /// middleware registered.
    pub fn new() -> Self {
        Self {
            models: ModelRegistry::new(),
            tools: ToolRegistry::new(),
            middleware: MiddlewareStack::new(),
            policy: RunPolicy::default(),
        }
    }

    /// Registers a model under `name`. The first registered model becomes the
    /// registry default unless one is already set. Returns `&mut Self` for
    /// chaining.
    pub fn register_model(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        self.models.register(name, model);
        self
    }

    /// Sets the default model name used when a request specifies no override.
    pub fn set_default_model(&mut self, name: impl Into<String>) -> &mut Self {
        self.models.set_default(name);
        self
    }

    /// Registers a tool, keyed by its [`Tool::name`]. Returns `&mut Self` for
    /// chaining.
    pub fn register_tool(&mut self, tool: Arc<dyn Tool<State>>) -> &mut Self {
        self.tools.register(tool);
        self
    }

    /// Appends a middleware to the stack. Registration order is the onion
    /// order: the first pushed middleware is the outermost layer.
    pub fn push_middleware(&mut self, middleware: Arc<dyn Middleware<State, Ctx>>) -> &mut Self {
        self.middleware.push(middleware);
        self
    }

    /// Replaces the run policy. Returns `&mut Self` for chaining.
    pub fn with_policy(&mut self, policy: RunPolicy) -> &mut Self {
        self.policy = policy;
        self
    }

    /// Returns a reference to the model registry.
    pub fn models(&self) -> &ModelRegistry<State> {
        &self.models
    }

    /// Returns a reference to the tool registry.
    pub fn tools(&self) -> &ToolRegistry<State> {
        &self.tools
    }

    /// Returns a reference to the middleware stack.
    pub fn middleware(&self) -> &MiddlewareStack<State, Ctx> {
        &self.middleware
    }

    /// Returns a reference to the active run policy.
    pub fn policy(&self) -> &RunPolicy {
        &self.policy
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Default for AgentHarness<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
