//! Atomic document outbox and explicit replay into the native event stream.
//!
//! Status and event receipts commit together. Stream projection is separate:
//! no post-commit projection failure is represented as a rolled-back status CAS.
use super::dto::{StreamEvent, stored_as_is};
use super::storage::{AGENTS, CAS_ATTEMPTS, EVENT_HEADS, EVENTS, event_stream, key, put_op};
use super::*;
use serde::{Deserialize, Serialize};
use tinyagents_storage_kit::{ErrorKind, Precondition, Versioned, WriteOp};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EventHead {
    /// Null marks exhaustion without an unsigned BSON-overflow sentinel.
    pub(super) next: Option<u64>,
}

stored_as_is!(EventHead);

/// The head after `highest` is the last sequence a run used: the next ordinal,
/// or `None` once `i64::MAX` has been handed out.
pub(super) fn head_after(highest: u64) -> Option<u64> {
    (highest < i64::MAX as u64).then_some(highest + 1)
}

pub(super) struct EventCommit {
    pub(super) event: RunEvent,
    pub(super) writes: Vec<WriteOp>,
}

impl DriverRunLedger {
    /// Prepare an event and head update for the caller's atomic document batch.
    pub(super) async fn event_commit(&self, input: &RunEventAppend) -> Result<EventCommit> {
        let head = self.read::<EventHead>(EVENT_HEADS, &input.run_id).await?;
        let sequence = match &head {
            // A run migrated from v1 has events but no head (the v1 head's id
            // was a digest of the run id); its next sequence follows its last event.
            None => match self.highest_sequence(&input.run_id).await? {
                None => 1,
                Some(highest) => head_after(highest)
                    .ok_or_else(|| StorageError::serialization("event sequence exhausted"))?,
            },
            Some(head) => head
                .doc
                .next
                .ok_or_else(|| StorageError::serialization("event sequence exhausted"))?,
        };
        if sequence == 0 || sequence > i64::MAX as u64 {
            return Err(StorageError::serialization("event sequence exceeds classic range").into());
        }
        // Keep every persisted integer within BSON’s signed range, while still
        // allowing the final legal event. Existing numeric heads deserialize
        // as Some; the portable terminal representation is JSON null.
        let next = (sequence < i64::MAX as u64).then_some(sequence + 1);
        let event = RunEvent {
            run_id: input.run_id.clone(),
            sequence,
            event_type: input.event_type.clone(),
            payload: input.payload.clone(),
            timestamp: now(),
        };
        Ok(EventCommit {
            writes: vec![
                put_op(
                    EVENT_HEADS,
                    key(&[&input.run_id])?,
                    &EventHead { next },
                    condition(&head),
                )?,
                put_op(
                    EVENTS,
                    key(&[&input.run_id, &sequence.to_string()])?,
                    &event,
                    Precondition::Absent,
                )?,
            ],
            event,
        })
    }

    /// The highest sequence a run has committed, read from its receipts.
    async fn highest_sequence(&self, run_id: &str) -> Result<Option<u64>> {
        self.declare().await?;
        let page = self
            .scoped
            .documents()
            .query(
                EVENTS,
                &Query::filter(Filter::eq("run_id", run_id.to_owned()))
                    .sort(Sort::desc("sequence"))
                    .limit(1),
            )
            .await?;
        page.items
            .into_iter()
            .next()
            .map(|row| Ok(super::storage::decode::<RunEvent>(row.doc)?.sequence))
            .transpose()
    }

    pub(super) async fn commit_event(&self, input: RunEventAppend) -> Result<RunEvent> {
        for _ in 0..CAS_ATTEMPTS {
            let commit = self.event_commit(&input).await?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![], commit.writes)
                .await
            {
                Ok(_) => return Ok(commit.event),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("event allocation contention budget exhausted").into())
    }

    pub(super) async fn transition_agent(
        &self,
        id: &str,
        expected: Option<AgentRunStatus>,
        status: AgentRunStatus,
        error: Option<&str>,
        completed_at: Option<DateTime<Utc>>,
        event: Option<RunEventAppend>,
    ) -> Result<RunTransition> {
        for _ in 0..CAS_ATTEMPTS {
            let Some(previous) = self.read::<AgentRun>(AGENTS, id).await? else {
                return Ok(RunTransition::NotFound);
            };
            let (telemetry, check) = self.telemetry_dependency(id).await?;
            if expected.is_some_and(|expected| previous.doc.status != expected) {
                let checks = vec![
                    check,
                    tinystoragedrivers_types::ReadCheck {
                        collection: AGENTS.into(),
                        id: key(&[id])?,
                        expected: tinystoragedrivers_types::ExpectedDocument::Version(
                            previous.version,
                        ),
                    },
                ];
                match self
                    .scoped
                    .documents()
                    .atomic_batch_checked(checks, vec![])
                    .await
                {
                    Ok(_) => {
                        let mut row = previous.doc;
                        row.telemetry = telemetry;
                        return Ok(RunTransition::StatusChanged(row));
                    }
                    Err(error) if error.kind() == ErrorKind::Conflict => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            let condition = Precondition::Version(previous.version);
            let mut row = previous.doc;
            row.status = status;
            row.error = error.map(str::to_owned);
            row.completed_at = completed_at.map(trunc);
            row.updated_at = now();
            row.telemetry = None;
            let mut writes = vec![put_op(AGENTS, key(&[id])?, &row, condition)?];
            if let Some(input) = &event {
                writes.extend(self.event_commit(input).await?.writes);
            }
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![check], writes)
                .await
            {
                Ok(_) => {
                    row.telemetry = telemetry;
                    return Ok(RunTransition::Applied(row));
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent transition contention budget exhausted").into())
    }

    pub(super) async fn events(
        &self,
        request: &RunEventListRequest,
    ) -> Result<RunEventListResponse> {
        let limit = request.limit.unwrap_or(100).min(1000) as usize;
        let after = request.after_sequence.unwrap_or(0);
        let filter =
            Filter::eq("run_id", request.run_id.clone()).and(Filter::gt("sequence", after));
        self.declare().await?;
        let mut events = Vec::with_capacity(limit);
        let mut query = Query::filter(filter)
            .sort(Sort::asc("sequence"))
            .limit(limit.max(1));
        loop {
            // Even an empty result request obtains native read admission.
            let page = self.scoped.documents().query(EVENTS, &query).await?;
            for row in page.items {
                if events.len() == limit {
                    break;
                }
                events.push(super::storage::decode(row.doc)?);
            }
            match page.next {
                Some(cursor) if events.len() < limit => {
                    query.cursor = Some(cursor);
                    query.limit = Some(limit - events.len());
                }
                _ => break,
            }
        }
        Ok(RunEventListResponse {
            count: events.len(),
            events,
        })
    }

    async fn validate_projection(&self, run_id: &str, stream: &str, next: u64) -> Result<u64> {
        let mut offset = 0;
        let mut after = 0;
        while offset < next {
            let limit = usize::try_from((next - offset).min(256)).expect("bounded projection page");
            let entries = self
                .scoped
                .streams()
                .read_window(stream, offset, limit)
                .await?;
            if entries.len() != limit {
                return Err(StorageError::serialization("event projection was truncated").into());
            }
            let receipts = self
                .events(&RunEventListRequest {
                    run_id: run_id.into(),
                    after_sequence: Some(after),
                    limit: Some(limit as u32),
                })
                .await?
                .events;
            if receipts.len() != entries.len() {
                return Err(StorageError::serialization(
                    "event projection exceeds committed receipts",
                )
                .into());
            }
            for (entry, expected) in entries.into_iter().zip(receipts) {
                if entry.offset != offset {
                    return Err(
                        StorageError::serialization("event projection offset has a gap").into(),
                    );
                }
                let event = serde_json::from_value::<StreamEvent>(entry.value)
                    .map_err(StorageError::from)?
                    .into_event(run_id)?;
                if event != expected {
                    return Err(StorageError::serialization(
                        "event projection differs from ordered committed receipt",
                    )
                    .into());
                }
                offset += 1;
                after = event.sequence;
            }
        }
        Ok(after)
    }

    /// Replay committed event receipts into their native stream, without duplicate appends.
    ///
    /// Document receipts remain authoritative for atomic status/event visibility.
    /// Hosts may retry projection after a crash or native refusal. This operation
    /// never changes status, allocates a sequence, or selects a different backend.
    ///
    /// # Errors
    /// Retains native stream refusal and conditional-append capability errors;
    /// a conflicting or truncated projection is refused rather than overwritten.
    pub async fn reconcile_run_events(&self, run_id: &str) -> Result<u64> {
        self.declare().await?;
        let stream = event_stream(run_id)?;
        let mut next = self.scoped.streams().len(&stream).await?;
        let mut after = self.validate_projection(run_id, &stream, next).await?;
        let mut conflicts = 0;
        loop {
            let records = self
                .events(&RunEventListRequest {
                    run_id: run_id.into(),
                    after_sequence: Some(after),
                    limit: Some(1000),
                })
                .await?
                .events;
            if records.is_empty() {
                return Ok(next);
            }
            let last_sequence = records.last().expect("nonempty projection batch").sequence;
            let count = records.len() as u64;
            let values = records
                .into_iter()
                .map(|event| serde_json::to_value(StreamEvent::of(&event)))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(StorageError::from)?;
            match self
                .scoped
                .streams()
                .append_batch_if_offset(&stream, next, values)
                .await
            {
                Ok(_) => {
                    // Our conditional append extends the prefix already validated.
                    // Progress is not contention and does not spend a retry.
                    next += count;
                    after = last_sequence;
                    conflicts = 0;
                }
                Err(error) if error.kind() == ErrorKind::Conflict => {
                    conflicts += 1;
                    if conflicts == CAS_ATTEMPTS {
                        return Err(StorageError::conflict(
                            "event projection contention budget exhausted",
                        )
                        .into());
                    }
                    next = self.scoped.streams().len(&stream).await?;
                    after = self.validate_projection(run_id, &stream, next).await?;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

fn condition<T>(previous: &Option<Versioned<T>>) -> Precondition {
    previous.as_ref().map_or(Precondition::Absent, |previous| {
        Precondition::Version(previous.version)
    })
}
