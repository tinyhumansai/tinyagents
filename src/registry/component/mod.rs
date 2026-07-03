//! Component identity, kind, and metadata for the named capability registry.
//!
//! These are the vocabulary of the recursive catalog: a [`ComponentKind`]
//! ([`Model`](ComponentKind::Model), [`Tool`](ComponentKind::Tool),
//! [`Graph`](ComponentKind::Graph), [`Agent`](ComponentKind::Agent), …) plus a
//! [`ComponentId`] name is exactly what a `.rag`/`.ragsh` reference carries, and
//! [`ComponentMetadata`] is the durable, serializable description that lets a
//! capability be discovered, listed, and bound by name long after the process
//! that registered it has exited.
//!
//! See [`types`] for the definitions. This module adds constructors, accessors,
//! and string conversions used by the [`crate::registry::CapabilityRegistry`].

mod types;

use std::fmt;

pub use types::*;

impl ComponentId {
    /// Creates a component id from any string-like value.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the underlying name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ComponentId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ComponentId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl ComponentKind {
    /// All component kinds, in a stable order, for discovery iteration.
    pub const ALL: [ComponentKind; 12] = [
        ComponentKind::Model,
        ComponentKind::Tool,
        ComponentKind::Graph,
        ComponentKind::Router,
        ComponentKind::Reducer,
        ComponentKind::Store,
        ComponentKind::Agent,
        ComponentKind::Script,
        ComponentKind::Middleware,
        ComponentKind::Checkpointer,
        ComponentKind::TaskStore,
        ComponentKind::Listener,
    ];

    /// Returns the lowercase string name of this kind, matching its serialized
    /// form. Used in error messages and discovery output.
    pub fn as_str(&self) -> &'static str {
        match self {
            ComponentKind::Model => "model",
            ComponentKind::Tool => "tool",
            ComponentKind::Graph => "graph",
            ComponentKind::Router => "router",
            ComponentKind::Reducer => "reducer",
            ComponentKind::Store => "store",
            ComponentKind::Agent => "agent",
            ComponentKind::Script => "script",
            ComponentKind::Middleware => "middleware",
            ComponentKind::Checkpointer => "checkpointer",
            ComponentKind::TaskStore => "task_store",
            ComponentKind::Listener => "listener",
        }
    }
}

impl fmt::Display for ComponentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ComponentMetadata {
    /// Creates minimal metadata for a registered component: just its id and
    /// kind, with no description, tags, or aliases.
    pub fn new(id: impl Into<ComponentId>, kind: ComponentKind) -> Self {
        Self {
            id: id.into(),
            kind,
            description: None,
            tags: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Sets the description. Returns `self` for chaining.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a discovery tag. Returns `self` for chaining.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Returns the registered name (the id string).
    pub fn name(&self) -> &str {
        self.id.as_str()
    }
}

#[cfg(test)]
mod test;
