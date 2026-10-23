//! Writer-only reconciliation; a stream row is invisible until metadata publishes it.
use super::pending::{APPENDS, AppendIntent};
use super::*;
use tinyagents_storage_kit::{Capability, ExpectedDocument, ReadCheck, WriteOp, WriteResult};

impl DriverConversationStore {
    /// Thread metadata plus the companion durable append-intent declaration
    /// (`conversations.appends`).
    ///
    /// Readers and metadata-only handles retain [`Self::collection_specs`].
    /// Compatible native required-feature supersets are accepted.
    pub fn append_collection_specs() -> Vec<CollectionSpec> {
        let mut specs = Self::collection_specs();
        specs.push(append_spec());
        specs
    }

    /// Prepare append declarations under explicitly trusted bootstrap authority.
    ///
    /// Actor append validates declarations anew and never declares them. This
    /// prepares no native internal schemas and establishes no scope authority.
    ///
    /// # Errors
    /// The original native declaration failure, with its kind and source.
    pub async fn prepare_append_schema(
        documents: &dyn DocumentStore,
    ) -> tinystoragedrivers_types::Result<()> {
        for spec in Self::append_collection_specs() {
            documents.ensure_collection(&spec).await?;
        }
        Ok(())
    }

    async fn admit_append(&self) -> ConversationResult<()> {
        for spec in Self::append_collection_specs() {
            self.docs.validate_collection_spec(&spec).await?;
        }
        if !self
            .docs
            .capabilities()
            .contains(Capability::ValidatedBatch)
        {
            return Err(StorageError::unsupported(
                Capability::ValidatedBatch,
                "conversation append requires checked documents",
            )
            .into());
        }
        if !self
            .streams
            .capabilities()
            .contains(Capability::ConditionalStreamAppend)
        {
            return Err(StorageError::unsupported(
                Capability::ConditionalStreamAppend,
                "conversation append requires conditional streams",
            )
            .into());
        }
        Ok(())
    }

    async fn pending(
        &self,
        thread_id: &str,
        incarnation: &str,
    ) -> ConversationResult<Option<Versioned<AppendIntent>>> {
        let key = AppendIntent::key(thread_id, incarnation)?;
        self.docs
            .get(APPENDS, &key)
            .await?
            .map(|row| AppendIntent::decode(row, thread_id, incarnation))
            .transpose()
    }

    fn check(collection: &str, id: &str, version: tinystoragedrivers_types::Version) -> ReadCheck {
        ReadCheck {
            collection: collection.into(),
            id: id.into(),
            expected: ExpectedDocument::Version(version),
        }
    }

    fn uncertain(error: StorageError) -> ConversationError {
        ConversationError::Uncertain {
            operation: "append_message",
            source: Box::new(error),
        }
    }

    fn acknowledgment_may_be_lost(error: &StorageError) -> bool {
        matches!(error.kind(), ErrorKind::Backend | ErrorKind::Unavailable)
    }

    async fn duplicate(
        &self,
        thread_id: &str,
        message: &ThreadMessage,
    ) -> ConversationResult<Option<ThreadMessage>> {
        if !is_deterministic_message_id(&message.id)
            && message
                .extra_metadata
                .get("scope")
                .and_then(serde_json::Value::as_str)
                != Some("channel")
        {
            return Ok(None);
        }
        Ok(self
            .messages(thread_id)
            .await?
            .into_iter()
            .find(|row| row.id == message.id))
    }

    async fn exact_row(&self, intent: &AppendIntent, stream: &str) -> ConversationResult<bool> {
        let offset = intent.offset();
        let rows = self.streams.read_window(stream, offset, 1).await?;
        let Some(row) = rows.first() else {
            return Ok(false);
        };
        if rows.len() != 1 || row.offset != offset || row.value != *intent.record_value() {
            return Err(StorageError::serialization(
                "conversation stream row disagrees with durable append intent",
            )
            .into());
        }
        Ok(true)
    }

    // Called only after this exact token was observed durably or reserved with
    // an acknowledged successful batch. Row equality alone cannot prove an
    // unobserved ordinary-ID reservation belonged to this attempt.
    async fn committed(&self, intent: &AppendIntent) -> ConversationResult<bool> {
        let offset = intent.offset();
        let Some(row) = self.load(&intent.thread_id).await? else {
            return Ok(false);
        };
        if row.doc.deleted
            || row.doc.incarnation != intent.incarnation
            || row.doc.visible_end <= offset
        {
            return Ok(false);
        }
        if !self.exact_row(intent, &row.doc.stream()?).await? {
            return Err(
                StorageError::serialization("missing committed conversation append").into(),
            );
        }
        Ok(self.load(&intent.thread_id).await?.is_some_and(|next| {
            !next.doc.deleted
                && next.doc.incarnation == intent.incarnation
                && next.doc.visible_end > offset
        }))
    }

    async fn finish_append(&self, intent: &AppendIntent) -> ConversationResult<ThreadMessage> {
        let offset = intent.offset();
        for _ in 0..RETRIES {
            self.admit_append().await?;
            if self.committed(intent).await? {
                return intent.message();
            }
            let current = self.load(&intent.thread_id).await?;
            let thread = Self::live(current.as_ref().map(|row| &row.doc))?;
            if thread.incarnation != intent.incarnation {
                return Err(
                    StorageError::conflict("conversation append incarnation retired").into(),
                );
            }
            let stream = thread.stream()?;
            if thread.visible_end != offset {
                return Err(StorageError::serialization(
                    "conversation append prefix is inconsistent",
                )
                .into());
            }
            let Some(pending) = self.pending(&intent.thread_id, &intent.incarnation).await? else {
                continue;
            };
            if pending.doc != *intent {
                continue;
            }
            match self
                .streams
                .append_batch_if_offset(&stream, offset, vec![intent.record_value().clone()])
                .await
            {
                Ok(actual) if actual == offset => {}
                Ok(_) => {
                    return Err(StorageError::serialization(
                        "conversation append returned wrong offset",
                    )
                    .into());
                }
                Err(error)
                    if error.kind() == ErrorKind::Conflict
                        || Self::acknowledgment_may_be_lost(&error) =>
                {
                    match self.exact_row(intent, &stream).await {
                        Ok(true) => {}
                        Ok(false) => return Err(error.into()),
                        Err(ConversationError::Driver(read))
                            if Self::acknowledgment_may_be_lost(&read) =>
                        {
                            return Err(Self::uncertain(error));
                        }
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => return Err(error.into()),
            }
            if !self.exact_row(intent, &stream).await? {
                return Err(StorageError::serialization(
                    "conversation append row missing after acknowledgment",
                )
                .into());
            }
            let Some(mut current) = self.load(&intent.thread_id).await? else {
                return Err(StorageError::not_found("conversation thread not found").into());
            };
            if current.doc.deleted || current.doc.incarnation != intent.incarnation {
                return Err(
                    StorageError::conflict("conversation append incarnation retired").into(),
                );
            }
            if current.doc.visible_end != offset {
                continue;
            }
            let Some(pending) = self.pending(&intent.thread_id, &intent.incarnation).await? else {
                continue;
            };
            if pending.doc != *intent {
                continue;
            }
            current.doc.visible_end = offset
                .checked_add(1)
                .ok_or_else(|| StorageError::invalid_input("conversation prefix overflow"))?;
            current.doc.thread.last_message_at = intent.message()?.created_at;
            let body = serde_json::to_value(current.doc.to_dto()?).map_err(|error| {
                StorageError::serialization("encode committed conversation metadata")
                    .with_source(error)
            })?;
            let checks = vec![
                Self::check(COLLECTION, &current.id, current.version),
                Self::check(APPENDS, &pending.id, pending.version),
            ];
            let writes = vec![
                WriteOp::Put {
                    collection: COLLECTION.into(),
                    id: current.id.clone(),
                    doc: body,
                    precondition: current.unchanged(),
                },
                WriteOp::Delete {
                    collection: APPENDS.into(),
                    id: pending.id.clone(),
                    precondition: pending.unchanged(),
                },
            ];
            match self.docs.atomic_batch_checked(checks, writes).await {
                Ok(_) => return intent.message(),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) if Self::acknowledgment_may_be_lost(&error) => {
                    return match self.committed(intent).await {
                        Ok(true) => intent.message(),
                        Ok(false) => Err(error.into()),
                        Err(ConversationError::Driver(read))
                            if Self::acknowledgment_may_be_lost(&read) =>
                        {
                            Err(Self::uncertain(error))
                        }
                        Err(other) => Err(other),
                    };
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(
            StorageError::conflict("conversation append reconciliation retry bound exhausted")
                .into(),
        )
    }

    pub(super) async fn append_native(
        &self,
        thread_id: &str,
        message: ThreadMessage,
    ) -> ConversationResult<ThreadMessage> {
        self.admit_append().await?;
        for _ in 0..RETRIES {
            let current = self.load(thread_id).await?;
            let thread = Self::live(current.as_ref().map(|row| &row.doc))?;
            if let Some(pending) = self.pending(thread_id, &thread.incarnation).await? {
                match self.finish_append(&pending.doc).await {
                    Ok(_) => {}
                    // This caller has not reserved its own message. Only the
                    // helped writer has an uncertain accepted attempt.
                    Err(ConversationError::Uncertain { source, .. }) => {
                        return Err(ConversationError::Driver(source));
                    }
                    Err(error) => return Err(error),
                }
                continue;
            }
            if let Some(existing) = self.duplicate(thread_id, &message).await? {
                return Ok(existing);
            }
            let stream = thread.stream()?;
            match self
                .streams
                .append_batch_if_offset(&stream, thread.visible_end, Vec::new())
                .await
            {
                Ok(actual) if actual == thread.visible_end => {}
                Ok(_) => {
                    return Err(StorageError::serialization(
                        "conversation probe returned wrong offset",
                    )
                    .into());
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
            let Some(current) = current else {
                return Err(StorageError::not_found("conversation thread not found").into());
            };
            let intent = AppendIntent::new(&thread, &message)?;
            let pending_key = AppendIntent::key(thread_id, &thread.incarnation)?;
            let checks = vec![
                Self::check(COLLECTION, &current.id, current.version),
                ReadCheck {
                    collection: APPENDS.into(),
                    id: pending_key.clone(),
                    expected: ExpectedDocument::Absent,
                },
            ];
            let writes = vec![WriteOp::Put {
                collection: APPENDS.into(),
                id: pending_key,
                doc: intent.value()?,
                precondition: Precondition::Absent,
            }];
            match self.docs.atomic_batch_checked(checks, writes).await {
                Ok(results) => {
                    if !matches!(results.as_slice(), [WriteResult::Put { .. }]) {
                        return Err(StorageError::serialization(
                            "conversation reservation returned invalid result",
                        )
                        .into());
                    }
                    return self.finish_append(&intent).await;
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) if Self::acknowledgment_may_be_lost(&error) => {
                    if let Ok(Some(pending)) = self.pending(thread_id, &thread.incarnation).await
                        && pending.doc == intent
                    {
                        return self.finish_append(&intent).await;
                    }
                    if let Ok(Some(existing)) = self.duplicate(thread_id, &message).await {
                        return Ok(existing);
                    }
                    return Err(Self::uncertain(error));
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("conversation append reservation retry bound exhausted").into())
    }
}
