//! Subgraph node adapters.
//!
//! A [`crate::graph::CompiledGraph`] can be embedded as a node in a parent
//! graph in two modes:
//!
//! - **shared-state**: parent and child share the same `State`/`Update`
//!   channel. The child runs over the parent's state and its final state is
//!   returned as the parent update.
//! - **adapter**: parent and child use different state shapes. A `to_child`
//!   mapping projects parent state into the child input, and a `from_child`
//!   mapping folds the child's final state back into a parent update.
//!
//! Both adapters append the embedding node id to the child's checkpoint
//! namespace so parent and child checkpoint ids never collide.

// This module is documentation-only for types; the adapter constructors live in
// `mod.rs` because they return closures rather than named types.
