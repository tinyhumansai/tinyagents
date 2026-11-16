//! Crate-owned session configuration.
//!
//! A relocated agent runtime cannot read a host's config schema — the whole
//! point of the move is that the runtime is generic over its host. This module
//! is the alternative: the crate declares the configuration *it* needs, and
//! each host maps its own schema into these structs once, at session-build
//! time.
//!
//! # Why structs and not a `ConfigProvider` trait
//!
//! A trait with ~40 getters was considered and rejected. It turns every config
//! read into a virtual call, it has no natural boundary so it becomes a dumping
//! ground, and it makes "what does the runtime actually depend on?" unanswerable
//! without reading every implementation. A struct answers that question by
//! existing. The mapping layer is the cost, and it is a cost worth paying: it is
//! the one place a host's schema meets the runtime's expectations, so it is also
//! the one place to test that meeting.
//!
//! # Layering
//!
//! [`SessionConfig`] is the root and owns the two path roots plus model
//! selection. [`TurnConfig`], [`ToolConfig`], and [`MemoryLimits`] hang off it
//! and are separable so a caller can override one turn's limits without
//! rebuilding the session.
//!
//! The types are inert — `serde` + `schemars` + `std` only, see `types`; `required_output`
//! adds pure validate/repair functions over [`RequiredOutput`]. Capabilities
//! (memory, security, budget, progress) are trait objects supplied separately;
//! nothing in this module is a behaviour seam.

mod context;
mod required_output;
mod types;

pub use context::*;
pub use required_output::{
    find_required_block, output_satisfies_contract, repair_instruction, synthesize_block,
};
pub use types::*;

#[cfg(test)]
#[path = "required_output_tests.rs"]
mod required_output_test;
#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
