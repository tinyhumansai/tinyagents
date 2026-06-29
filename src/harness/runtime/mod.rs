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

use crate::harness::cache::ResponseCache;
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
            response_cache: None,
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

    /// Attaches a [`ResponseCache`] shared across every run this harness drives.
    ///
    /// Once attached, the agent loop computes a stable
    /// [`cache_key`][crate::harness::cache::cache_key] for each model request
    /// and consults the cache before calling the provider. On a hit the
    /// provider is **not** invoked and the cached
    /// [`crate::harness::model::ModelResponse`] is reused; on a miss the
    /// provider is called and the successful response is stored back. Whether
    /// caching is active for a given call is governed by the effective
    /// [`CachePolicy`][crate::harness::cache::CachePolicy] (the per-request
    /// [`crate::harness::model::ModelRequest::cache_policy`] overriding
    /// [`RunPolicy::cache`]).
    ///
    /// Because the cache lives on the harness rather than a single run, two
    /// identical requests issued across separate runs share a key, so the
    /// second run can be served entirely from cache. Returns `&mut Self` for
    /// chaining.
    pub fn with_response_cache(&mut self, cache: Arc<dyn ResponseCache>) -> &mut Self {
        self.response_cache = Some(cache);
        self
    }

    /// Returns a reference to the attached response cache, if any.
    pub fn response_cache(&self) -> Option<&Arc<dyn ResponseCache>> {
        self.response_cache.as_ref()
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
