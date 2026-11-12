//! Durable push-completion for detached children.
//!
//! A detached child runs on its own and finishes whenever it finishes. The
//! parent still has to hear about it, once, even if the parent is idle,
//! mid-turn, or the process restarts in between. This module owns that
//! hand-off:
//!
//! - [`CompletionRecord`]: one finished child, keyed by task id, with a
//!   delivery [`CompletionState`] and an attempt counter.
//! - [`CompletionStore`]: [`InMemoryCompletionStore`] and the durable
//!   [`DriverCompletionStore`] over `tinystoragedrivers` ports.
//! - [`CompletionRouter`]: dedupe, tombstones, batch claims with attempt
//!   counting, give-up, and routing by [`NotifyMode`] onto the parent's queue
//!   lanes.
//! - [`CompletionFormatter`]: the host's wording. The harness ships only a
//!   neutral default.
//!
//! What stays with the host: when a parent is idle enough to receive a
//! delivery turn, the turn itself, and what to do with a record that gave up.
//! Live pushes are at-least-once: the host acknowledges with
//! `mark_delivered`. Nothing here relaunches a child.

mod admission;
mod delivery;
mod driver;
mod durable_lease;
mod format;
mod lifecycle;
mod registry;
mod router;
mod store;
mod types;

pub use admission::CompletionAdmission;
pub use delivery::{
    CompletionDelivery, CompletionDeliveryConfig, DeliveryFence, DeliveryOutcome, DeliveryTarget,
    ParentBusyGate, ParentBusyLease, TenantThreadKey, ThreadDeliveryLeases,
};
pub use driver::DriverCompletionStore;
pub use durable_lease::{DurableThreadDeliveryLease, DurableThreadDeliveryLeases};
pub use format::{CompletionFormatter, NeutralCompletionFormatter};
#[doc(hidden)]
pub use lifecycle::ParentLifecycle;
pub use registry::{
    CompletionRouterPool, CompletionRouterPoolError, CompletionRouterReplaceOutcome,
    CompletionStoreFallback, open_driver_completion_store, open_driver_completion_store_or_memory,
    open_driver_completion_store_or_memory_with_coordination_domain,
    open_driver_completion_store_with_coordination_domain,
};
pub use router::{
    CompletionRouter, CompletionStoreReplaceOutcome, DEFAULT_MAX_ATTEMPTS, RecordOutcome,
    TombstoneOutcome,
};
pub use store::{
    ClaimResolution, CompletionClaimMode, CompletionStore, InMemoryCompletionStore,
    ParentGateClaim, PendingClaimRequest, RenewClaimRequest, ResolveClaimRequest, TaskClaimRequest,
};
pub use types::{
    CompletionArtifact, CompletionClaim, CompletionReceipt, CompletionRecord, CompletionResult,
    CompletionState, CompletionStatus, NotifyMode,
};

#[cfg(test)]
#[path = "durable_lease_tests.rs"]
mod durable_lease_tests;

#[cfg(test)]
#[path = "router_tests.rs"]
mod router_tests;

#[cfg(test)]
#[path = "delivery_tests.rs"]
mod delivery_tests;

#[cfg(test)]
#[path = "registry_tests.rs"]
mod registry_tests;
