//! The v2 storage declarations of the run ledger (owner `ledger`) and the
//! session history (owner `history`), for the host's `Registry::compose`.
//!
//! One [`Schema`] per owner, as the registry requires; a host composes both
//! (and the session crate's `storage_schema`) and runs `Registry::migrate`
//! per agent scope.
//!
//! | v1 | v2 |
//! |---|---|
//! | `run_ledger_{workflow_runs,telemetry,agent_runs,teams,members,tool_effects,event_outbox,event_heads,tasks}` | `ledger.{workflow_runs,telemetry,agent_runs,teams,members,tool_effects,event_outbox,event_heads,tasks}`, id `Key::of` of the record's ids, snake_case body with `*_ms` times, no `{schemaVersion, record, *Sql}` envelope |
//! | stream `run_events:<sha>` | stream `ledger.events/<run>` (the run id is read from the entries) |
//! | `session_history_{sessions,messages,tool_calls}` | `history.{sessions,messages,tool_calls}` (µs sort keys dropped, times in ms) |
//! | `session_history_fts` | `history.search`, rebuilt from the migrated records |
//!
//! Outside a document step (a step writes only its own owner's collections):
//!
//! - the embedded `WorkflowRun` lease becomes a `tsd.leases` record, the dense
//!   history ids `tsd.counters` and the agent-import receipt a `tsd.migrations`
//!   record; [`super::import_legacy_ledger`] moves the first two (call it
//!   **before** `Registry::migrate`, which deletes the v1 workflow rows).
//! - `run_ledger_event_heads` is dropped: the v1 id was a digest of the run id,
//!   and the next sequence of a migrated run is recomputed from its events.

use super::storage::{
    EVENT_STREAM, HISTORY_OWNER, INPUTS, LEDGER_OWNER, SCHEMA_VERSION, history_specs, ledger_specs,
};
use tinyagents_storage_kit::{Owner, Schema, ScopeClass};

/// What the run ledger stores (owner `ledger`, scope class Agent), with the
/// steps that bring the v1 names and shapes forward.
#[must_use]
pub fn ledger_storage_schema() -> Schema {
    let owner = Owner::new(LEDGER_OWNER).expect("`ledger` is a valid owner");
    let mut schema = Schema::new(owner);
    for spec in ledger_specs() {
        // Inputs are new, with no legacy v1 envelope to migrate. Their initial
        // schema is version one; declaring v2 would demand a nonexistent step.
        let version = if spec.name == INPUTS {
            1
        } else {
            SCHEMA_VERSION
        };
        schema = schema.collection_with(version, spec, ScopeClass::Agent);
    }
    schema = schema.stream(EVENT_STREAM, ScopeClass::Agent);
    for step in super::steps::steps() {
        schema = schema.step(step);
    }
    schema
}

/// What the session history stores (owner `history`, scope class Agent), with
/// the steps that bring the v1 names and shapes forward.
#[must_use]
pub fn history_storage_schema() -> Schema {
    let owner = Owner::new(HISTORY_OWNER).expect("`history` is a valid owner");
    let mut schema = Schema::new(owner);
    for spec in history_specs() {
        schema = schema.collection_with(SCHEMA_VERSION, spec, ScopeClass::Agent);
    }
    for step in super::steps_history::steps() {
        schema = schema.step(step);
    }
    schema
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
