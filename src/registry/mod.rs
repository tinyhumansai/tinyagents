//! Registry coordination and discovery primitives.
//!
//! The registry owns named runtime components and local metadata catalogs. The
//! first concrete piece is the model catalog, which keeps a checked-in snapshot
//! of provider model prices, context windows, and capabilities for deterministic
//! local lookup.

pub mod catalog;

pub use catalog::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
    ModelPricing,
};
