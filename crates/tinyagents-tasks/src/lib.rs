//! Graph-independent detached-task machinery for TinyAgents.
//!
//! This crate is the managed child-work surface: task model, `TaskStore`
//! implementations, the detached task registry and restart reconciliation.
//! It does not depend on the graph engine, so
//! `tinyagents-orchestration` builds on it.
//!
//! Hosts can track child work by stable task id without exposing raw executor
//! handles such as `tokio::JoinHandle`.

mod completions;
mod driver_store;
mod reconcile;
mod runtime;
mod status;
mod steering_registry;
mod storage;
mod store;
mod store_registry;
mod types;

pub use completions::{
    ClaimResolution, CompletionAdmission, CompletionArtifact, CompletionClaim, CompletionClaimMode,
    CompletionFormatter, CompletionReceipt, CompletionRecord, CompletionResult, CompletionRouter,
    CompletionState, CompletionStatus, CompletionStore, CompletionStoreReplaceOutcome,
    DEFAULT_MAX_ATTEMPTS, DriverCompletionStore, InMemoryCompletionStore,
    NeutralCompletionFormatter, NotifyMode, ParentGateClaim, ParentLifecycle, PendingClaimRequest,
    RecordOutcome, RenewClaimRequest, ResolveClaimRequest, TaskClaimRequest, TombstoneOutcome,
};
pub use completions::{
    CompletionDelivery, CompletionDeliveryConfig, CompletionRouterPool, CompletionRouterPoolError,
    CompletionRouterReplaceOutcome, CompletionStoreFallback, DeliveryFence, DeliveryOutcome,
    DeliveryTarget, DurableThreadDeliveryLease, DurableThreadDeliveryLeases, ParentBusyGate,
    ParentBusyLease, TenantThreadKey, ThreadDeliveryLeases, open_driver_completion_store,
    open_driver_completion_store_or_memory,
    open_driver_completion_store_or_memory_with_coordination_domain,
    open_driver_completion_store_with_coordination_domain,
};
pub use driver_store::DriverTaskStore;
pub use reconcile::{
    ReconcileOutcome, ReconcileReport, ReconciledTask, reconcile_orphaned_tasks, task_status_label,
};
pub use runtime::{DetachedAbortEvent, DetachedTaskRegistry};
pub use steering_registry::SteeringRegistry;
pub use storage::{open_storage, storage_schema};
pub use store::{InMemoryTaskStore, TaskStore};
pub use store_registry::{
    TaskStoreRegistry, TaskStoreRegistryError, open_driver_task_store,
    open_driver_task_store_or_memory,
};
pub use types::*;

#[cfg(test)]
#[path = "conditional_sqlite.rs"]
mod conditional_sqlite;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod test;
