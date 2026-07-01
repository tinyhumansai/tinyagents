//! Identity and discovery types for registered named capabilities — the
//! by-name handles a recursive workflow uses to address sub-capabilities.
//!
//! These are the durable, provider-neutral descriptors used by the
//! [`crate::registry::CapabilityRegistry`]. They intentionally do *not* carry
//! executable handles: a [`ComponentMetadata`] can be serialized, diffed, and
//! rendered in a UI long after the process that registered the live model or
//! tool has exited.

use serde::{Deserialize, Serialize};

/// A stable, durable identifier for a registered capability.
///
/// This is a newtype over [`String`] rather than a raw Rust type path so a
/// capability can survive module moves, renames, and serialization. The string
/// is the registered name (for example `"gpt-4o"` or `"lookup_user"`); it is
/// scoped to a [`ComponentKind`] by the registry, so the same name may be used
/// for, say, a model and a tool without collision.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ComponentId(pub String);

/// The kind of capability a registered component provides.
///
/// The kind partitions the registry namespace: lookups, duplicate detection,
/// aliasing, and discovery are all scoped by kind, so `(Model, "x")` and
/// `(Tool, "x")` are independent entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    /// A chat model handle.
    Model,
    /// A callable tool.
    Tool,
    /// A compiled graph blueprint.
    Graph,
    /// A conditional-routing function descriptor (name-only for now).
    Router,
    /// A state-channel reducer descriptor (name-only for now).
    Reducer,
    /// A memory/vector/checkpoint store descriptor (name-only for now).
    Store,
    /// An executable agent configuration descriptor (name-only for now).
    Agent,
    /// A middleware descriptor (name-only for now).
    Middleware,
    /// A graph/harness checkpointer descriptor (name-only for now).
    Checkpointer,
    /// An orchestration task-store descriptor (name-only for now).
    TaskStore,
    /// An event listener descriptor (name-only for now).
    Listener,
}

/// Discovery and UI metadata for one registered capability.
///
/// Every entry in a [`crate::registry::CapabilityRegistry`] has metadata, even
/// when it was registered through a minimal name-only API; in that case the
/// description and tags are empty and [`ComponentMetadata::aliases`] is filled
/// in as aliases are declared.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// The durable identifier (the registered name).
    pub id: ComponentId,
    /// The kind this component was registered under.
    pub kind: ComponentKind,
    /// Optional human/UI-facing description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free-form discovery tags (for example `fast`, `cheap`, `local`).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Alternate names that resolve to this component.
    #[serde(default)]
    pub aliases: Vec<String>,
}
