//! Atomic durable input admission with persisted parent-lineage fencing.

use super::dto::{from_ms, stored_as_is};
use super::storage::{AGENTS, CAS_ATTEMPTS, INPUTS, key, now, put_op};
use super::*;
use crate::run_ledger::inbox::{AddressedInput, QueuedInput, RunInboxStore};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tinyagents_storage_kit::{ErrorKind, ExpectedDocument, Precondition, ReadCheck};

#[derive(Clone, Serialize, Deserialize)]
struct StoredInput {
    #[serde(flatten)]
    input: AddressedInput,
    sequence: u64,
    created_ms: i64,
}
stored_as_is!(StoredInput);

impl StoredInput {
    /// Convert the persisted admission into its stable public receipt.
    fn receipt(self) -> Result<QueuedInput> {
        Ok(QueuedInput {
            input: self.input,
            sequence: self.sequence,
            created_at: from_ms(self.created_ms)?,
        })
    }
}

/// Reject invalid identities and empty or oversized input before storage access.
fn validate(input: &AddressedInput) -> Result<()> {
    for id in [
        &input.submission_id,
        &input.root_thread_id,
        &input.recipient_run_id,
    ] {
        if id.trim().is_empty() || id.len() > 512 || id.contains('\0') {
            return Err(StorageError::serialization("invalid addressed input identity").into());
        }
    }
    if input.text.trim().is_empty() || input.text.len() > 1024 * 1024 {
        return Err(StorageError::serialization(
            "addressed input must contain 1..=1048576 bytes of nonblank text",
        )
        .into());
    }
    Ok(())
}

impl DriverRunLedger {
    /// Walk exact persisted run links and fence every observed lineage version.
    /// Display names and the live worker registry cannot establish ownership.
    async fn input_lineage(&self, root: &str, recipient: &str) -> Result<Vec<ReadCheck>> {
        let mut current = recipient.to_owned();
        let mut visited = HashSet::new();
        let mut checks = Vec::new();
        for _ in 0..128 {
            if !visited.insert(current.clone()) {
                return Err(StorageError::serialization("cyclic input recipient lineage").into());
            }
            let row = self
                .read::<AgentRun>(AGENTS, &current)
                .await?
                .ok_or_else(|| StorageError::serialization("input recipient lineage is missing"))?;
            checks.push(ReadCheck {
                collection: AGENTS.into(),
                id: key(&[&current])?,
                expected: ExpectedDocument::Version(row.version),
            });
            match row.doc.parent_run_id {
                Some(parent) => current = parent,
                None if row.doc.parent_thread_id.as_deref() == Some(root) => return Ok(checks),
                None => {
                    return Err(StorageError::serialization(
                        "input recipient belongs to another root",
                    )
                    .into());
                }
            }
        }
        Err(StorageError::serialization("input recipient lineage exceeds 128 runs").into())
    }
}

#[async_trait::async_trait]
impl RunInboxStore for DriverRunLedger {
    /// Atomically admit an input and event, or return its original dedup receipt.
    async fn enqueue_input(&self, input: AddressedInput) -> Result<QueuedInput> {
        validate(&input)?;
        self.declare().await?;
        let id = key(&[&input.root_thread_id, &input.submission_id])?;
        for _ in 0..CAS_ATTEMPTS {
            let mut checks = self
                .input_lineage(&input.root_thread_id, &input.recipient_run_id)
                .await?;
            let previous = self.scoped.documents().get(INPUTS, &id).await?;
            if let Some(previous) = previous {
                let stored: StoredInput = super::storage::decode(previous.doc)?;
                if stored.input != input {
                    return Err(StorageError::conflict(
                        "submission id already addresses different input",
                    )
                    .into());
                }
                checks.push(ReadCheck {
                    collection: INPUTS.into(),
                    id: id.clone(),
                    expected: ExpectedDocument::Version(previous.version),
                });
                match self
                    .scoped
                    .documents()
                    .atomic_batch_checked(checks, vec![])
                    .await
                {
                    Ok(_) => return stored.receipt(),
                    Err(error) if error.kind() == ErrorKind::Conflict => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            let commit = self.event_commit(&RunEventAppend {
                run_id: input.recipient_run_id.clone(),
                event_type: "input_queued".into(),
                payload: serde_json::json!({"submission_id": input.submission_id, "mode": input.mode}),
            }).await?;
            let stored = StoredInput {
                input: input.clone(),
                sequence: commit.event.sequence,
                created_ms: now().timestamp_millis(),
            };
            let mut writes = commit.writes;
            writes.push(put_op(INPUTS, id.clone(), &stored, Precondition::Absent)?);
            match self
                .scoped
                .documents()
                .atomic_batch_checked(checks, writes)
                .await
            {
                Ok(_) => return stored.receipt(),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("input admission contention budget exhausted").into())
    }

    /// Read an exclusive sequence page while rechecking persisted root ownership.
    async fn pending_inputs(
        &self,
        root: &str,
        recipient: &str,
        after_sequence: Option<u64>,
        limit: u32,
    ) -> Result<Vec<QueuedInput>> {
        if root.trim().is_empty() || recipient.trim().is_empty() {
            return Err(
                StorageError::serialization("input root and recipient are required").into(),
            );
        }
        for _ in 0..CAS_ATTEMPTS {
            let checks = self.input_lineage(root, recipient).await?;
            let mut filter =
                Filter::eq("root_thread_id", root).and(Filter::eq("recipient_run_id", recipient));
            if let Some(cursor) = after_sequence {
                filter = filter.and(Filter::gt("sequence", cursor));
            }
            let page = self
                .scoped
                .documents()
                .query(
                    INPUTS,
                    &Query::filter(filter)
                        .sort(Sort::asc("sequence"))
                        .limit(limit.clamp(1, 500) as usize),
                )
                .await?;
            let inputs = page
                .items
                .into_iter()
                .map(|row| super::storage::decode::<StoredInput>(row.doc)?.receipt())
                .collect::<Result<Vec<_>>>()?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(checks, vec![])
                .await
            {
                Ok(_) => return Ok(inputs),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("input read contention budget exhausted").into())
    }
}

#[cfg(test)]
#[path = "inbox_tests.rs"]
mod tests;
