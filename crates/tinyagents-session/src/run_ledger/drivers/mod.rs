//! Run-ledger implementation over an explicitly scoped storage handle.
//!
//! Operations whose native atomic contracts are not implemented yet report a
//! typed unsupported error. They never approximate a transaction with weaker
//! individual writes or switch to a local SQLite database.

mod agents;
mod dto;
mod dto_history;
mod effects;
mod events;
mod history;
mod history_retention;
mod imports;
mod inbox;
mod legacy_state;
mod retention;
mod schema;
mod steps;
mod steps_history;
mod storage;
mod team;
pub use legacy_state::{LegacyLedgerImport, import_legacy_ledger};
pub use schema::{history_storage_schema, ledger_storage_schema};
use std::sync::Arc;
use storage::{
    LEDGER_OWNER, MEMBERS, TASKS, TEAMS, TELEMETRY, WORKFLOWS, eq_nonblank, now, trunc,
    workflow_lease_key,
};
use tinyagents_storage_kit::ScopedStorage;
use tinyagents_storage_kit::primitives::Leases;
use tinyagents_storage_kit::{Filter, Query, Scope, Sort, StorageError};
use tokio::sync::OnceCell;

use super::LedgerResult as Result;
use chrono::{DateTime, Utc};

use super::command_center::{AgentWorkRow, CommandCenterView, ControlVerb};
use super::port::RunLedgerStore;
use super::tool_effects::{ToolEffectRow, ToolEffectSettle, ToolEffectStart};
use super::types::RunTransition;
use super::types::*;
use crate::retention::RetentionReport;

/// Ledger storage bound to the supplied tenant/agent scope.
///
/// The host supplies an admitted handle. Records and model arguments never
/// select a namespace. This adapter does not open a workspace or local file.
#[derive(Debug, Clone)]
pub struct DriverRunLedger {
    scoped: ScopedStorage,
    declared: Arc<OnceCell<()>>,
    validate_collections: bool,
}

impl DriverRunLedger {
    /// Bind the already admitted scoped ports without opening any local database.
    pub fn new(scoped: ScopedStorage) -> Self {
        Self {
            scoped,
            declared: Arc::new(OnceCell::new()),
            validate_collections: false,
        }
    }

    /// Validate existing declarations rather than lazily calling ensure_collection.
    ///
    /// The caller must supply native ExistingSchema ports to prohibit other
    /// backend-internal schema initialization. This constructor controls only
    /// this adapter's document declaration policy, not backend authority.
    pub fn new_existing_schema(scoped: ScopedStorage) -> Self {
        Self {
            validate_collections: true,
            ..Self::new(scoped)
        }
    }

    /// Exact document declarations shared by bootstrap preparation and lazy admission.
    pub fn collection_specs() -> Vec<tinystoragedrivers_types::CollectionSpec> {
        storage::specs()
    }

    /// Prepare ledger declarations through an explicitly trusted document handle.
    ///
    /// # Errors
    /// The first native declaration refusal, with no fallback or rollback claim
    /// for earlier successful declarations. This does not prepare native internal schema.
    pub async fn prepare_schema(
        documents: &dyn tinystoragedrivers_api::DocumentStore,
    ) -> std::result::Result<(), StorageError> {
        // The shared `tsd.*` collections (leases, counters, migrations) are
        // declared with them: the ledger reads and writes those too.
        for spec in Self::collection_specs()
            .into_iter()
            .chain(storage::primitive_specs())
        {
            documents.ensure_collection(&spec).await?;
        }
        Ok(())
    }

    /// The immutable namespace this handle was supplied for.
    pub fn scope(&self) -> &Scope {
        self.scoped.scope()
    }
}

impl DriverRunLedger {
    /// Driver leases (`tsd.leases`) timed at `now`, so expiry is tested without sleeps.
    fn leases_at(&self, now: DateTime<Utc>) -> Leases {
        let ms = u64::try_from(now.timestamp_millis()).unwrap_or(0);
        Leases::new(self.scoped.documents()).with_clock(Arc::new(move || ms))
    }

    /// `row` with the live driver lease of its run, if one is held.
    async fn with_lease(&self, mut row: WorkflowRun, now: DateTime<Utc>) -> Result<WorkflowRun> {
        let lease = self
            .leases_at(now)
            .current(LEDGER_OWNER, &workflow_lease_key(&row.id)?)
            .await?;
        if let Some(lease) = lease {
            row.lease_owner = Some(lease.holder);
            row.lease_expires_at = Some(dto::from_ms(lease.expires_ms)?);
        }
        Ok(row)
    }

    /// End the live lease of `run_id`, whoever holds it. Nothing happens when none is held.
    async fn release_workflow_lease(&self, run_id: &str, now: DateTime<Utc>) -> Result<()> {
        let leases = self.leases_at(now);
        let lease_key = workflow_lease_key(run_id)?;
        if let Some(lease) = leases.current(LEDGER_OWNER, &lease_key).await? {
            leases
                .release(LEDGER_OWNER, &lease_key, &lease.holder, lease.epoch)
                .await?;
        }
        Ok(())
    }

    /// The timestamp is explicit internally so the expiry boundary is tested without sleeps.
    ///
    /// The lease is a `tsd.leases` record; the run record only gets its revision
    /// bumped, in a second write, so a claim is no longer one atomic document
    /// change. A crash between the two leaves the lease held and the revision
    /// unchanged, which the next claim after expiry resolves.
    async fn claim_workflow_at(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
        now: DateTime<Utc>,
    ) -> Result<WorkflowLeaseClaim> {
        let Some(found) = self.read::<WorkflowRun>(WORKFLOWS, id).await? else {
            return Ok(WorkflowLeaseClaim::Missing);
        };
        let lease_key = workflow_lease_key(id)?;
        let ttl_ms = lease_for.num_milliseconds();
        let leases = self.leases_at(now);
        let Some(mut lease) = leases
            .acquire(LEDGER_OWNER, &lease_key, owner, ttl_ms.max(1))
            .await?
        else {
            return Ok(WorkflowLeaseClaim::Busy(
                self.with_lease(found.doc, now).await?,
            ));
        };
        if ttl_ms <= 0 {
            // A lease that is already over (the classic claim accepted any
            // duration): take the holder's tenure, then end it at once.
            leases
                .release(LEDGER_OWNER, &lease_key, owner, lease.epoch)
                .await?;
            lease.expires_ms = now.timestamp_millis() + ttl_ms;
        }
        let bumped = self
            .change(WORKFLOWS, id, |previous: Option<WorkflowRun>| {
                let Some(mut row) = previous else {
                    return Ok((None, None));
                };
                row.updated_at = now;
                row.revision = next_revision(row.revision)?;
                Ok((Some(row.clone()), Some(row)))
            })
            .await?;
        let Some(mut row) = bumped else {
            return Ok(WorkflowLeaseClaim::Missing);
        };
        row.lease_owner = Some(lease.holder);
        row.lease_expires_at = Some(dto::from_ms(lease.expires_ms)?);
        Ok(WorkflowLeaseClaim::Acquired(row))
    }

    /// Renewal changes the native version, but preserves the workflow lifecycle revision.
    async fn renew_workflow_at(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let leases = self.leases_at(now);
        let lease_key = workflow_lease_key(id)?;
        let Some(current) = leases.current(LEDGER_OWNER, &lease_key).await? else {
            return Ok(false);
        };
        if current.holder != owner {
            return Ok(false);
        }
        let ttl_ms = lease_for.num_milliseconds().max(1);
        let renewed = leases
            .renew(LEDGER_OWNER, &lease_key, owner, current.epoch, ttl_ms)
            .await?;
        if renewed.is_none() {
            return Ok(false);
        }
        self.change(WORKFLOWS, id, |previous: Option<WorkflowRun>| {
            Ok(match previous {
                Some(mut row) => {
                    row.updated_at = now;
                    (Some(row), true)
                }
                None => (None, false),
            })
        })
        .await
        .map_err(Into::into)
    }
}

/// Workflow revisions retain the classic signed SQLite range rather than overflowing on corrupt input.
fn next_revision(revision: u64) -> std::result::Result<u64, StorageError> {
    revision
        .checked_add(1)
        .filter(|revision| *revision <= i64::MAX as u64)
        .ok_or_else(|| {
            StorageError::serialization("workflow revision exceeds classic signed range")
        })
}

/// Merge lifecycle fields without rewriting the immutable started-at timestamp.
fn apply_workflow(
    row: &mut WorkflowRun,
    upsert: &WorkflowRunUpsert,
    now: DateTime<Utc>,
) -> std::result::Result<(), StorageError> {
    row.definition_id = upsert.definition_id.clone();
    row.parent_thread_id = upsert
        .parent_thread_id
        .clone()
        .or(row.parent_thread_id.take());
    row.input = upsert.input.clone();
    row.phase_states = upsert.phase_states.clone();
    row.child_run_ids = upsert.child_run_ids.clone();
    row.status = upsert.status;
    row.summary = upsert.summary.clone().or(row.summary.take());
    row.updated_at = now;
    row.completed_at = upsert.completed_at.map(trunc).or(row.completed_at);
    row.revision = next_revision(row.revision)?;
    Ok(())
}

#[async_trait::async_trait]
impl RunLedgerStore for DriverRunLedger {
    async fn import_agent_family(
        &self,
        import: super::import::AgentFamilyImport,
    ) -> Result<super::import::AgentFamilyImportReport> {
        self.import_agent_family_rows(import).await
    }

    async fn import_legacy_rows(
        &self,
        import: super::import::LegacyLedgerRows,
    ) -> Result<super::import::LegacyLedgerRowsReport> {
        self.import_legacy_rows_inner(import).await
    }

    /// Implement [`RunLedgerStore::upsert_agent_run`] without a weaker fallback.
    async fn upsert_agent_run(&self, upsert: AgentRunUpsert) -> Result<AgentRun> {
        self.upsert_agent(upsert).await
    }

    /// Implement [`RunLedgerStore::upsert_workflow_run`] without a weaker fallback.
    async fn upsert_workflow_run(&self, upsert: WorkflowRunUpsert) -> Result<WorkflowRun> {
        let now = now();
        let row = self
            .change(WORKFLOWS, &upsert.id, |previous: Option<WorkflowRun>| {
                let first = previous.is_none();
                let mut row = previous.unwrap_or(WorkflowRun {
                    id: upsert.id.clone(),
                    definition_id: upsert.definition_id.clone(),
                    parent_thread_id: None,
                    input: upsert.input.clone(),
                    phase_states: upsert.phase_states.clone(),
                    child_run_ids: upsert.child_run_ids.clone(),
                    status: upsert.status,
                    summary: None,
                    started_at: upsert.started_at.map(trunc).unwrap_or(now),
                    updated_at: now,
                    completed_at: None,
                    revision: 0,
                    lease_owner: None,
                    lease_expires_at: None,
                });
                row.definition_id = upsert.definition_id.clone();
                row.parent_thread_id = upsert.parent_thread_id.clone().or(row.parent_thread_id);
                row.input = upsert.input.clone();
                row.phase_states = upsert.phase_states.clone();
                row.child_run_ids = upsert.child_run_ids.clone();
                row.status = upsert.status;
                row.summary = upsert.summary.clone().or(row.summary);
                row.updated_at = now;
                row.completed_at = upsert.completed_at.map(trunc).or(row.completed_at);
                if !first {
                    row.revision = next_revision(row.revision)?;
                }
                Ok((Some(row.clone()), row))
            })
            .await?;
        if row.status.is_terminal() {
            self.release_workflow_lease(&row.id, now).await?;
            return Ok(row);
        }
        self.with_lease(row, now).await
    }

    async fn try_claim_workflow_run(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<WorkflowLeaseClaim> {
        self.claim_workflow_at(id, owner, lease_for, now()).await
    }

    /// The lease is checked first and renewed (or ended, for a terminal status)
    /// after the revision-fenced record write; the two are separate documents,
    /// so a lease that lapses in between does not undo the write.
    async fn compare_and_swap_workflow_run(
        &self,
        upsert: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<Option<WorkflowRun>> {
        let now = now();
        let leases = self.leases_at(now);
        let lease_key = workflow_lease_key(&upsert.id)?;
        let Some(held) = leases
            .current(LEDGER_OWNER, &lease_key)
            .await?
            .filter(|lease| lease.holder == owner)
        else {
            return Ok(None);
        };
        let written = self
            .change(WORKFLOWS, &upsert.id, |previous: Option<WorkflowRun>| {
                let Some(mut row) = previous else {
                    return Ok((None, None));
                };
                if row.revision != expected_revision {
                    return Ok((None, None));
                }
                apply_workflow(&mut row, &upsert, now)?;
                Ok((Some(row.clone()), Some(row)))
            })
            .await?;
        let Some(mut row) = written else {
            return Ok(None);
        };
        if row.status.is_terminal() {
            leases
                .release(LEDGER_OWNER, &lease_key, owner, held.epoch)
                .await?;
        } else {
            let ttl_ms = lease_for.num_milliseconds().max(1);
            if let Some(lease) = leases
                .renew(LEDGER_OWNER, &lease_key, owner, held.epoch, ttl_ms)
                .await?
            {
                row.lease_owner = Some(lease.holder);
                row.lease_expires_at = Some(dto::from_ms(lease.expires_ms)?);
            }
        }
        Ok(Some(row))
    }

    async fn renew_workflow_run_lease(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<bool> {
        self.renew_workflow_at(id, owner, lease_for, now()).await
    }

    async fn compare_and_swap_workflow_run_lifecycle(
        &self,
        upsert: WorkflowRunUpsert,
        expected_revision: u64,
    ) -> Result<Option<WorkflowRun>> {
        let now = now();
        let written = self
            .change(WORKFLOWS, &upsert.id, |previous: Option<WorkflowRun>| {
                let Some(mut row) = previous else {
                    return Ok((None, None));
                };
                if row.revision != expected_revision {
                    return Ok((None, None));
                }
                apply_workflow(&mut row, &upsert, now)?;
                Ok((Some(row.clone()), Some(row)))
            })
            .await?;
        if let Some(row) = &written {
            self.release_workflow_lease(&row.id, now).await?;
        }
        Ok(written)
    }

    /// Implement [`RunLedgerStore::append_run_event`] without a weaker fallback.
    async fn append_run_event(&self, event: RunEventAppend) -> Result<RunEvent> {
        self.commit_event(event).await
    }

    /// Implement [`RunLedgerStore::upsert_run_telemetry`] without a weaker fallback.
    async fn upsert_run_telemetry(&self, upsert: RunTelemetryUpsert) -> Result<RunTelemetry> {
        let now = now();
        self.merge(
            TELEMETRY,
            &upsert.run_id,
            |previous: Option<RunTelemetry>| {
                let mut row = previous.unwrap_or(RunTelemetry {
                    run_id: upsert.run_id.clone(),
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    cost_usd: 0.0,
                    elapsed_ms: None,
                    tool_count: 0,
                    model: None,
                    provider: None,
                    error: None,
                    updated_at: Some(now),
                });
                row.input_tokens = upsert.input_tokens.unwrap_or(row.input_tokens);
                row.output_tokens = upsert.output_tokens.unwrap_or(row.output_tokens);
                row.cached_input_tokens = upsert
                    .cached_input_tokens
                    .unwrap_or(row.cached_input_tokens);
                row.cost_usd = upsert.cost_usd.unwrap_or(row.cost_usd);
                row.elapsed_ms = upsert.elapsed_ms.or(row.elapsed_ms);
                row.tool_count = upsert.tool_count.unwrap_or(row.tool_count);
                row.model = upsert.model.clone().or(row.model);
                row.provider = upsert.provider.clone().or(row.provider);
                row.error = upsert.error.clone().or(row.error);
                row.updated_at = Some(now);
                row
            },
        )
        .await
        .map_err(Into::into)
    }

    /// Implement [`RunLedgerStore::get_agent_run`] without a weaker fallback.
    async fn get_agent_run(&self, id: &str) -> Result<Option<AgentRun>> {
        self.agent(id).await
    }

    /// Implement [`RunLedgerStore::transition_agent_run_status`] without a weaker fallback.
    async fn transition_agent_run_status(
        &self,
        id: &str,
        status: AgentRunStatus,
        error: Option<&str>,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<Option<AgentRun>> {
        self.transition_agent(id, None, status, error, completed_at, None)
            .await
            .map(|result| match result {
                RunTransition::Applied(row) | RunTransition::StatusChanged(row) => Some(row),
                RunTransition::NotFound => None,
            })
    }

    /// Implement [`RunLedgerStore::transition_agent_run_status_from`] without a weaker fallback.
    async fn transition_agent_run_status_from(
        &self,
        id: &str,
        expected: AgentRunStatus,
        status: AgentRunStatus,
        error: Option<&str>,
        completed_at: Option<DateTime<Utc>>,
        event: Option<RunEventAppend>,
    ) -> Result<RunTransition> {
        self.transition_agent(id, Some(expected), status, error, completed_at, event)
            .await
    }

    /// Implement [`RunLedgerStore::interrupt_orphaned_agent_runs`] without a weaker fallback.
    async fn interrupt_orphaned_agent_runs(&self) -> Result<usize> {
        self.interrupt_agents().await
    }

    /// Implement [`RunLedgerStore::list_agent_runs`] without a weaker fallback.
    async fn list_agent_runs(&self, request: &AgentRunListRequest) -> Result<AgentRunListResponse> {
        self.agents(request).await
    }

    /// Implement [`RunLedgerStore::list_recent_run_events`] without a weaker fallback.
    async fn list_recent_run_events(
        &self,
        request: &RunEventListRequest,
    ) -> Result<RunEventListResponse> {
        self.events(request).await
    }

    /// Implement [`RunLedgerStore::get_workflow_run`] without a weaker fallback.
    async fn get_workflow_run(&self, id: &str) -> Result<Option<WorkflowRun>> {
        let Some(found) = self.read::<WorkflowRun>(WORKFLOWS, id).await? else {
            return Ok(None);
        };
        Ok(Some(self.with_lease(found.doc, now()).await?))
    }

    /// Implement [`RunLedgerStore::list_workflow_runs`] without a weaker fallback.
    async fn list_workflow_runs(
        &self,
        request: &WorkflowRunListRequest,
    ) -> Result<WorkflowRunListResponse> {
        let filter = eq_nonblank(
            Filter::default(),
            "definition_id",
            request.definition_id.as_deref(),
        );
        let filter = eq_nonblank(filter, "status", request.status.as_deref());
        let filter = eq_nonblank(
            filter,
            "parent_thread_id",
            request.parent_thread_id.as_deref(),
        );
        let (runs, count) = self
            .page(
                WORKFLOWS,
                filter,
                request.limit.unwrap_or(50).min(500),
                request.offset.unwrap_or(0),
            )
            .await?;
        let at = now();
        let mut leased = Vec::with_capacity(runs.len());
        for run in runs {
            leased.push(self.with_lease(run, at).await?);
        }
        Ok(WorkflowRunListResponse {
            runs: leased,
            count,
        })
    }

    async fn upsert_agent_team(&self, upsert: AgentTeamUpsert) -> Result<AgentTeam> {
        let now = now();
        Ok(self
            .merge(TEAMS, &upsert.id, |previous: Option<AgentTeam>| AgentTeam {
                id: upsert.id.clone(),
                parent_thread_id: upsert.parent_thread_id.clone().or_else(|| {
                    previous
                        .as_ref()
                        .and_then(|row| row.parent_thread_id.clone())
                }),
                lead_agent_id: upsert.lead_agent_id.clone(),
                status: upsert.status,
                summary: upsert
                    .summary
                    .clone()
                    .or_else(|| previous.as_ref().and_then(|row| row.summary.clone())),
                created_at: previous
                    .as_ref()
                    .map(|row| row.created_at)
                    .unwrap_or(upsert.created_at.map(trunc).unwrap_or(now)),
                updated_at: now,
                closed_at: upsert
                    .closed_at
                    .or_else(|| previous.as_ref().and_then(|row| row.closed_at)),
            })
            .await?)
    }

    async fn get_agent_team(&self, id: &str) -> Result<Option<AgentTeam>> {
        Ok(self.read(TEAMS, id).await?.map(|row| row.doc))
    }

    async fn list_agent_teams(
        &self,
        request: &AgentTeamListRequest,
    ) -> Result<AgentTeamListResponse> {
        let filter = eq_nonblank(
            Filter::default(),
            "parent_thread_id",
            request.parent_thread_id.as_deref(),
        );
        let filter = eq_nonblank(filter, "status", request.status.as_deref());
        let (teams, count) = self
            .page(
                TEAMS,
                filter,
                request.limit.unwrap_or(50).min(500),
                request.offset.unwrap_or(0),
            )
            .await?;
        Ok(AgentTeamListResponse { teams, count })
    }

    async fn upsert_agent_team_member(
        &self,
        upsert: AgentTeamMemberUpsert,
    ) -> Result<AgentTeamMember> {
        let now = now();
        Ok(self
            .merge(MEMBERS, &upsert.id, |previous: Option<AgentTeamMember>| {
                AgentTeamMember {
                    id: upsert.id.clone(),
                    team_id: previous
                        .as_ref()
                        .map(|row| row.team_id.clone())
                        .unwrap_or_else(|| upsert.team_id.clone()),
                    name: upsert.name.clone(),
                    agent_id: upsert
                        .agent_id
                        .clone()
                        .or_else(|| previous.as_ref().and_then(|row| row.agent_id.clone())),
                    member_status: upsert.member_status,
                    current_task_id: upsert.current_task_id.clone().or_else(|| {
                        previous
                            .as_ref()
                            .and_then(|row| row.current_task_id.clone())
                    }),
                    worker_thread_id: upsert.worker_thread_id.clone().or_else(|| {
                        previous
                            .as_ref()
                            .and_then(|row| row.worker_thread_id.clone())
                    }),
                    run_id: upsert
                        .run_id
                        .clone()
                        .or_else(|| previous.as_ref().and_then(|row| row.run_id.clone())),
                    created_at: previous
                        .as_ref()
                        .map(|row| row.created_at)
                        .unwrap_or(upsert.created_at.map(trunc).unwrap_or(now)),
                    updated_at: now,
                }
            })
            .await?)
    }

    async fn get_agent_team_member(&self, id: &str) -> Result<Option<AgentTeamMember>> {
        Ok(self.read(MEMBERS, id).await?.map(|row| row.doc))
    }

    async fn list_agent_team_members(&self, team_id: &str) -> Result<Vec<AgentTeamMember>> {
        Ok(self
            .records(
                MEMBERS,
                Query::filter(Filter::eq("team_id", team_id)).sort(Sort::asc("created_ms")),
            )
            .await?)
    }

    async fn upsert_agent_team_task(&self, upsert: AgentTeamTaskUpsert) -> Result<AgentTeamTask> {
        let now = now();
        Ok(self
            .merge(TASKS, &upsert.id, |previous: Option<AgentTeamTask>| {
                let live = previous
                    .as_ref()
                    .filter(|_| upsert.status == AgentTeamTaskStatus::InProgress);
                AgentTeamTask {
                    id: upsert.id.clone(),
                    team_id: previous
                        .as_ref()
                        .map(|row| row.team_id.clone())
                        .unwrap_or_else(|| upsert.team_id.clone()),
                    title: upsert.title.clone(),
                    objective: upsert
                        .objective
                        .clone()
                        .or_else(|| previous.as_ref().and_then(|row| row.objective.clone())),
                    status: upsert.status,
                    owner_member_id: upsert.owner_member_id.clone().or_else(|| {
                        previous
                            .as_ref()
                            .and_then(|row| row.owner_member_id.clone())
                    }),
                    claimed_by_member_id: live.and_then(|row| row.claimed_by_member_id.clone()),
                    claim_token: live.and_then(|row| row.claim_token.clone()),
                    depends_on: upsert.depends_on.clone(),
                    gate_status: upsert
                        .gate_status
                        .clone()
                        .unwrap_or_else(|| "pending".into()),
                    gate_reason: upsert
                        .gate_reason
                        .clone()
                        .or_else(|| previous.as_ref().and_then(|row| row.gate_reason.clone())),
                    evidence: upsert.evidence.clone(),
                    source_run_id: upsert
                        .source_run_id
                        .clone()
                        .or_else(|| previous.as_ref().and_then(|row| row.source_run_id.clone())),
                    order_index: upsert.order_index,
                    created_at: previous
                        .as_ref()
                        .map(|row| row.created_at)
                        .unwrap_or(upsert.created_at.map(trunc).unwrap_or(now)),
                    updated_at: now,
                }
            })
            .await?)
    }

    async fn get_agent_team_task(&self, id: &str) -> Result<Option<AgentTeamTask>> {
        Ok(self.read(TASKS, id).await?.map(|row| row.doc))
    }

    async fn list_agent_team_tasks(&self, team_id: &str) -> Result<Vec<AgentTeamTask>> {
        Ok(self
            .records(
                TASKS,
                Query::filter(Filter::eq("team_id", team_id))
                    .sort(Sort::asc("order_index"))
                    .sort(Sort::asc("created_ms")),
            )
            .await?)
    }

    /// Implement [`RunLedgerStore::claim_agent_team_task`] without a weaker fallback.
    async fn claim_agent_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome> {
        self.claim_team_task(team_id, task_id, member_id, claim_token)
            .await
    }

    /// Implement [`RunLedgerStore::complete_agent_team_task`] without a weaker fallback.
    async fn complete_agent_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        self.complete_team_task(team_id, task_id, member_id, evidence, require_evidence)
            .await
    }

    async fn complete_agent_team_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        self.complete_team_task_if_claimed(
            team_id,
            task_id,
            member_id,
            Some(claim_token),
            evidence,
            require_evidence,
        )
        .await
    }

    /// Implement [`RunLedgerStore::shutdown_agent_team_member`] without a weaker fallback.
    async fn shutdown_agent_team_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
        self.shutdown_team_member(team_id, member_id).await
    }

    async fn mark_agent_team_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>> {
        let now = now();
        self.change(MEMBERS, member_id, |previous: Option<AgentTeamMember>| {
            let Some(mut row) = previous.filter(|row| row.team_id == team_id) else {
                return Ok((None, None));
            };
            row.member_status = AgentTeamMemberStatus::Active;
            row.current_task_id = Some(task_id.into());
            row.worker_thread_id = Some(worker_thread_id.into());
            row.run_id = Some(run_id.into());
            row.updated_at = now;
            Ok((Some(row.clone()), Some(row)))
        })
        .await
        .map_err(Into::into)
    }

    async fn try_start_agent_team_member(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<AgentTeamMemberRunAdmission> {
        let now = Utc::now();
        self.change(
            MEMBERS,
            member_id,
            move |previous: Option<AgentTeamMember>| {
                let Some(mut row) = previous.filter(|row| row.team_id == team_id) else {
                    return Ok((None, AgentTeamMemberRunAdmission::Missing));
                };
                match row.member_status {
                    AgentTeamMemberStatus::Active => {
                        Ok((None, AgentTeamMemberRunAdmission::AlreadyActive(row)))
                    }
                    AgentTeamMemberStatus::Stopped => {
                        Ok((None, AgentTeamMemberRunAdmission::NotAdmissible(row)))
                    }
                    AgentTeamMemberStatus::Pending | AgentTeamMemberStatus::Idle => {
                        row.member_status = AgentTeamMemberStatus::Active;
                        row.current_task_id = Some(task_id.to_owned());
                        row.worker_thread_id = Some(worker_thread_id.to_owned());
                        row.run_id = Some(run_id.to_owned());
                        row.updated_at = now;
                        Ok((Some(row.clone()), AgentTeamMemberRunAdmission::Started(row)))
                    }
                }
            },
        )
        .await
        .map_err(Into::into)
    }

    async fn mark_agent_team_member_idle(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<AgentTeamMember>> {
        let now = now();
        self.change(MEMBERS, member_id, |previous: Option<AgentTeamMember>| {
            let Some(mut row) = previous.filter(|row| row.team_id == team_id) else {
                return Ok((None, None));
            };
            row.member_status = AgentTeamMemberStatus::Idle;
            row.current_task_id = None;
            row.updated_at = now;
            Ok((Some(row.clone()), Some(row)))
        })
        .await
        .map_err(Into::into)
    }

    async fn mark_agent_team_member_idle_if_current(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>> {
        let now = Utc::now();
        self.change(MEMBERS, member_id, |previous: Option<AgentTeamMember>| {
            let Some(mut row) = previous.filter(|row| {
                row.team_id == team_id
                    && row.member_status == AgentTeamMemberStatus::Active
                    && row.current_task_id.as_deref() == Some(task_id)
                    && row.worker_thread_id.as_deref() == Some(worker_thread_id)
                    && row.run_id.as_deref() == Some(run_id)
            }) else {
                return Ok((None, None));
            };
            row.member_status = AgentTeamMemberStatus::Idle;
            row.current_task_id = None;
            row.updated_at = now;
            Ok((Some(row.clone()), Some(row)))
        })
        .await
        .map_err(Into::into)
    }

    async fn release_agent_team_task(&self, team_id: &str, task_id: &str) -> Result<bool> {
        let now = now();
        self.change(TASKS, task_id, |previous: Option<AgentTeamTask>| {
            let Some(mut row) = previous.filter(|row| {
                row.team_id == team_id && row.status == AgentTeamTaskStatus::InProgress
            }) else {
                return Ok((None, false));
            };
            row.status = AgentTeamTaskStatus::Todo;
            row.claimed_by_member_id = None;
            row.claim_token = None;
            row.gate_status = "pending".into();
            row.gate_reason = None;
            row.updated_at = now;
            Ok((Some(row), true))
        })
        .await
        .map_err(Into::into)
    }

    async fn release_agent_team_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<bool> {
        let now = Utc::now();
        self.change(TASKS, task_id, |previous: Option<AgentTeamTask>| {
            let Some(mut row) = previous.filter(|row| {
                row.team_id == team_id
                    && row.status == AgentTeamTaskStatus::InProgress
                    && row.claimed_by_member_id.as_deref() == Some(member_id)
                    && row.claim_token.as_deref() == Some(claim_token)
            }) else {
                return Ok((None, false));
            };
            row.status = AgentTeamTaskStatus::Todo;
            row.claimed_by_member_id = None;
            row.claim_token = None;
            row.gate_status = "pending".into();
            row.gate_reason = None;
            row.updated_at = now;
            Ok((Some(row), true))
        })
        .await
        .map_err(Into::into)
    }

    async fn record_tool_started(&self, start: ToolEffectStart) -> Result<ToolEffectRow> {
        self.effect_start(start).await
    }

    async fn settle_tool_effect(&self, settle: ToolEffectSettle) -> Result<Option<ToolEffectRow>> {
        self.effect_settle(settle).await
    }

    async fn list_unresolved_tool_effects(&self, run_id: &str) -> Result<Vec<ToolEffectRow>> {
        self.effect_unresolved(run_id).await
    }

    async fn mark_tool_effect_interrupted(
        &self,
        run_id: &str,
        call_id: &str,
    ) -> Result<Option<ToolEffectRow>> {
        self.effect_settle(ToolEffectSettle {
            run_id: run_id.into(),
            call_id: call_id.into(),
            status: super::tool_effects::ToolEffectStatus::Interrupted,
            effect_summary: None,
        })
        .await
    }

    /// Implement [`RunLedgerStore::prune_sessions_before`] without a weaker fallback.
    async fn prune_sessions_before(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.history_prune_sessions(older_than).await
    }

    /// Implement [`RunLedgerStore::trim_session_messages`] without a weaker fallback.
    async fn trim_session_messages(&self, session_id: &str, keep_last: usize) -> Result<usize> {
        self.history_trim_messages(session_id, keep_last).await
    }

    /// Implement [`RunLedgerStore::prune_tool_calls_before`] without a weaker fallback.
    async fn prune_tool_calls_before(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.history_prune_tool_calls(older_than).await
    }

    /// Implement [`RunLedgerStore::prune_run_events_before`] without a weaker fallback.
    async fn prune_run_events_before(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.history_prune_events(older_than).await
    }

    async fn prune_run_telemetry_before(&self, older_than: DateTime<Utc>) -> Result<usize> {
        self.prune_telemetry(older_than).await.map_err(Into::into)
    }

    /// Implement [`RunLedgerStore::apply_retention`] without a weaker fallback.
    async fn apply_retention(&self, older_than: DateTime<Utc>) -> Result<RetentionReport> {
        self.history_apply_retention(older_than).await
    }

    /// Implement [`RunLedgerStore::reindex_fts`] without a weaker fallback.
    async fn reindex_fts(&self) -> Result<usize> {
        self.history_reindex().await
    }

    /// Implement [`RunLedgerStore::list_agent_work`] without a weaker fallback.
    async fn list_agent_work(
        &self,
        limit: Option<u32>,
        display_name: &(dyn for<'a> Fn(&'a str) -> Option<String> + Sync),
    ) -> Result<CommandCenterView> {
        let response = self
            .list_agent_runs(&AgentRunListRequest {
                limit: Some(limit.unwrap_or(200).min(500)),
                ..Default::default()
            })
            .await?;
        Ok(super::command_center::build_view(
            response.runs,
            display_name,
        ))
    }

    /// Implement [`RunLedgerStore::apply_control`] without a weaker fallback.
    async fn apply_control(
        &self,
        run_id: &str,
        verb: ControlVerb,
        message: Option<&str>,
        reason: Option<&str>,
        display_name: &(dyn for<'a> Fn(&'a str) -> Option<String> + Sync),
    ) -> Result<AgentWorkRow> {
        super::command_center::native::apply_control(
            self,
            run_id,
            verb,
            message,
            reason,
            display_name,
        )
        .await
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "agent_tests.rs"]
mod agent_tests;

#[cfg(test)]
#[cfg(test)]
#[path = "retention_tests.rs"]
mod retention_tests;

#[cfg(test)]
#[path = "events_bson_tests.rs"]
mod event_bson_tests;

#[cfg(test)]
#[path = "events_query_tests.rs"]
mod event_query_tests;

#[cfg(test)]
#[path = "team_tests.rs"]
mod team_tests;

#[cfg(test)]
#[path = "history_tests.rs"]
mod history_tests;

#[cfg(test)]
#[path = "imports_tests.rs"]
mod imports_tests;
