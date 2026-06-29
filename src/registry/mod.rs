//! Registry coordination and discovery primitives.
//!
//! The registry owns named runtime components and local metadata catalogs. The
//! first concrete piece is the model catalog, which keeps a checked-in snapshot
//! of provider model prices, context windows, and capabilities for deterministic
//! local lookup.

pub mod capability;
pub mod catalog;
pub mod component;

pub use capability::CapabilityRegistry;
pub use catalog::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
    ModelPricing,
};
pub use component::{ComponentId, ComponentKind, ComponentMetadata};
