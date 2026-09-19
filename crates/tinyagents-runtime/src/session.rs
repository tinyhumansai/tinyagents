use std::{future::Future, sync::Arc};

use tinyagents_harness::CancellationToken;
use tinyagents_session::transcript::{
    TranscriptHistory, TranscriptMessage, TranscriptPartial, TranscriptTurn, TurnUsage,
};
use tinyinference_llm::message::Message;

use crate::{
    CommitReceipt, DriverRequest, PrefixSnapshot, ResumeMode, ResumePreparation, RuntimeError,
    SessionDriver, SessionHooks, SessionResume, SessionStateView, SessionTerminal,
    SessionTurnOutcome, SessionTurnRequest, ToolSnapshot, TranscriptCodec, TranscriptCommitReceipt,
    TranscriptDelta, TranscriptTarget, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

/// Host-neutral mutable state for one conversation session.
pub struct Session<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    prefix: PrefixSnapshot,
    default_tools: ToolSnapshot,
    history: Vec<Message>,
    persisted: Vec<TranscriptMessage>,
    target: Option<TranscriptTarget>,
    transcript: Option<Arc<dyn TranscriptHistory>>,
    committed_turns: usize,
}

impl<C: Clone + Send + Sync + 'static> Session<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        driver: Arc<dyn SessionDriver<C>>,
        codec: Option<Arc<dyn TranscriptCodec<C>>>,
        hooks: Arc<dyn SessionHooks<C>>,
        prefix: PrefixSnapshot,
        default_tools: ToolSnapshot,
        target: Option<TranscriptTarget>,
    ) -> Self {
        Self {
            driver,
            codec,
            hooks,
            history: prefix.messages().to_vec(),
            prefix,
            default_tools,
            persisted: Vec::new(),
            target,
            transcript: None,
            committed_turns: 0,
        }
    }

    /// Returns the currently committed model history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns the stable prefix currently applied to this session.
    pub fn prefix_snapshot(&self) -> &PrefixSnapshot {
        &self.prefix
    }

    /// Returns the builder compatibility default used only when a preparation
    /// supplies no per-turn tool snapshot.
    pub fn tool_snapshot(&self) -> &ToolSnapshot {
        &self.default_tools
    }

    /// Seeds an uncommitted session from an explicit, lossless host snapshot.
    ///
    /// This replaces neither the host's raw rows nor their metadata. It is the
    /// supported alternative to a host keeping a shadow history beside the
    /// runtime. Seeding after any durable transition is rejected.
    pub fn seed_history(
        &mut self,
        history: Vec<Message>,
        raw: Vec<TranscriptMessage>,
    ) -> Result<(), RuntimeError> {
        if self.committed_turns != 0 {
            return Err(RuntimeError::InvalidSessionState(
                "cannot seed history after a committed turn".into(),
            ));
        }
        self.history = self.with_prefix(history);
        self.persisted = raw;
        Ok(())
    }

    /// Loads the selected durable transcript, retaining its lossless raw rows
    /// as the base for the next append-only delta.
    pub async fn resume(
        &mut self,
        options: &TurnOptions<C>,
    ) -> Result<SessionResume, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let Some(target) = self.target.as_ref() else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let read = match options.resume {
            ResumeMode::Never => None,
            ResumeMode::LatestForAgent => target
                .locator
                .latest_for_agent(target.resume_agent.as_deref().unwrap_or(&target.stem)),
            ResumeMode::Thread => options
                .thread_id
                .as_deref()
                .and_then(|thread| target.locator.root_for_thread(thread)),
        };
        let Some(read) = read else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let Some(transcript) = read
            .read_session()
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?
        else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let codec = self
            .codec
            .as_ref()
            .ok_or(RuntimeError::MissingDependency("TranscriptCodec"))?;
        let history = self.with_prefix(codec.decode_history(&transcript)?);
        self.history = history.clone();
        self.persisted = transcript.messages;
        // The discovered metadata, not the builder seed, is authoritative for
        // the subsequent append. This keeps resume-only host fields intact.
        if let Some(target) = self.target.as_mut() {
            target.meta = transcript.meta;
        }
        // A successful explicit resume also fixes the target's history handle
        // for later appends. Builder construction itself remains I/O-free.
        if self.transcript.is_none() {
            let target = self.target.as_ref().expect("target checked above");
            self.transcript = Some(
                target
                    .locator
                    .open_stem(&target.stem, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            );
        }
        Ok(SessionResume {
            loaded: true,
            history,
        })
    }

    /// Executes and commits one state transition.
    pub async fn turn(
        &mut self,
        mut request: SessionTurnRequest,
        mut options: TurnOptions<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        let mut terminal_guard = TerminalGuard::new(self.hooks.clone());
        let result = self
            .turn_inner(&mut request, &mut options, &mut terminal_guard)
            .await;
        if !terminal_guard.is_committed() {
            let terminal = match &result {
                Ok(outcome) => SessionTerminal::Completed(outcome.clone()),
                Err(RuntimeError::Cancelled) => SessionTerminal::Cancelled,
                Err(error) => SessionTerminal::Failed(error.to_string()),
            };
            terminal_guard.set(terminal);
        }
        // Terminal observation cannot revoke a durable successful commit.
        let _ = terminal_guard.finish().await;
        result
    }

    async fn turn_inner(
        &mut self,
        request: &mut SessionTurnRequest,
        options: &mut TurnOptions<C>,
        terminal_guard: &mut TerminalGuard<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let cancellation = options.cancellation.clone();
        let resume_preparation = cancelable(
            &cancellation,
            self.hooks
                .before_resume(request, options, self.state_view(false)),
        )
        .await?;
        self.apply_resume_preparation(resume_preparation)?;
        let resumed = if options.resume == ResumeMode::Never {
            false
        } else {
            self.resume(options).await?.loaded
        };
        // `resume` is synchronous after its read, so this explicit boundary
        // makes cancellation between loading and before-turn preparation
        // observable without handing work to the driver.
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let preparation = cancelable(
            &cancellation,
            self.hooks
                .before_turn(request, options, self.state_view(resumed)),
        )
        .await?;
        let (tools, prepared_prefix) = self.apply_preparation(preparation)?;
        if let Some(prefix) = prepared_prefix {
            self.apply_prefix(prefix)?;
        }

        let mut input = self.history.clone();
        if input.last() != Some(&request.input) {
            input.push(request.input.clone());
        }
        let codec_options = options.transcript_options();
        let request_id = options.request_id.clone();
        let thread_id = options.thread_id.clone();
        let stream = options.stream;
        let cancellation = options.cancellation.clone();
        // `RunContext` is consumed exactly once. The host context captured in
        // `codec_options` is the one after preparation and before handoff.
        let run_context = std::mem::replace(
            &mut options.run_context,
            tinyagents_harness::context::RunContext::new(
                tinyagents_harness::context::RunConfig::new("consumed-session-context"),
                codec_options.context.clone(),
            ),
        )
        .with_cancellation(cancellation.clone());
        let driver_result = tokio::select! {
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
            result = self.driver.execute(DriverRequest { history: input, tools, run_context, stream }) => result,
        };
        let outcome = match driver_result {
            Ok(outcome) => outcome,
            Err(failure) => {
                if let Some(partial) = failure.partial {
                    let partial_history = self.with_prefix(partial.history);
                    let raw = self.encode(&self.history, &partial_history, &codec_options)?;
                    let turn_usage = self.turn_usage(&codec_options)?;
                    let receipt = self.persist(
                        &raw,
                        request_id.as_deref(),
                        thread_id.as_deref(),
                        partial.partial.as_ref(),
                        turn_usage.as_ref(),
                    )?;
                    self.history = partial_history;
                    self.persisted = raw;
                    if receipt.is_some() {
                        self.committed_turns += 1;
                    }
                }
                return Err(failure.error);
            }
        };
        let candidate = self.with_prefix(outcome.history);
        let committed = SessionTurnOutcome {
            history: candidate.clone(),
            output: outcome.output,
            interrupted: outcome.interrupted,
        };
        cancelable(
            &cancellation,
            self.hooks.before_commit(&committed, &codec_options),
        )
        .await?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let raw = self.encode(&self.history, &candidate, &codec_options)?;
        let turn_usage = self.turn_usage(&codec_options)?;
        let transcript = self.persist(
            &raw,
            request_id.as_deref(),
            thread_id.as_deref(),
            None,
            turn_usage.as_ref(),
        )?;
        self.history = committed.history.clone();
        self.persisted = raw;
        self.committed_turns += 1;
        // The receipt is constructed only after append and state replacement.
        // Its hook and the completed terminal are owned by one task: errors or
        // cancellation cannot relabel the successful durable transition, and
        // dropping the caller future cannot drop finalization mid-flight.
        let receipt = CommitReceipt {
            outcome: committed.clone(),
            options: codec_options,
            transcript,
        };
        let finalization = terminal_guard.finalize_commit(receipt);
        // This await deliberately does not observe cancellation. If this turn
        // future is dropped, dropping `JoinHandle` detaches rather than aborts
        // the owned finalization task.
        let _ = finalization.await;
        Ok(committed)
    }

    fn apply_preparation(
        &mut self,
        preparation: TurnPreparation,
    ) -> Result<(ToolSnapshot, Option<PrefixSnapshot>), RuntimeError> {
        // A returned snapshot never updates `default_tools`: it applies only
        // to the `DriverRequest` being built by this call.
        Ok((
            preparation
                .tools
                .unwrap_or_else(|| self.default_tools.clone()),
            preparation.prefix,
        ))
    }

    fn apply_resume_preparation(
        &mut self,
        preparation: ResumePreparation,
    ) -> Result<(), RuntimeError> {
        if let Some(target) = preparation.transcript {
            if self.transcript.is_some() || self.committed_turns != 0 {
                if !self
                    .target
                    .as_ref()
                    .is_some_and(|bound| bound.same_binding(&target))
                {
                    return Err(RuntimeError::InvalidSessionState(
                        "cannot change a transcript target after it is bound or committed".into(),
                    ));
                }
            } else {
                self.target = Some(target);
            }
        }
        if self.target.is_some() && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        Ok(())
    }

    fn apply_prefix(&mut self, prefix: PrefixSnapshot) -> Result<(), RuntimeError> {
        if prefix == self.prefix {
            return Ok(());
        }
        if self.committed_turns != 0 {
            return Err(RuntimeError::InvalidSessionState(
                "cannot change a session prefix after a committed turn".into(),
            ));
        }
        let history = std::mem::take(&mut self.history);
        let history = history
            .strip_prefix(self.prefix.messages())
            .unwrap_or(&history)
            .to_vec();
        self.prefix = prefix;
        self.history = self.with_prefix(history);
        Ok(())
    }

    fn state_view(&self, resumed: bool) -> SessionStateView<'_> {
        SessionStateView {
            history: &self.history,
            raw_history: &self.persisted,
            prefix: &self.prefix,
            transcript_target: self.target.as_ref(),
            committed_turns: self.committed_turns,
            resumed,
        }
    }

    fn encode(
        &self,
        previous: &[Message],
        next: &[Message],
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.reconcile(&self.persisted, previous, next, options),
            None => Ok(Vec::new()),
        }
    }

    fn turn_usage(
        &self,
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Option<TurnUsage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.turn_usage(options),
            None => Ok(None),
        }
    }

    fn persist(
        &mut self,
        raw: &[TranscriptMessage],
        request_id: Option<&str>,
        thread_id: Option<&str>,
        partial: Option<&TranscriptPartial>,
        turn_usage: Option<&TurnUsage>,
    ) -> Result<Option<TranscriptCommitReceipt>, RuntimeError> {
        let Some(target) = self.target.as_mut() else {
            return Ok(None);
        };
        if self.transcript.is_none() {
            self.transcript = Some(
                target
                    .locator
                    .open_stem(&target.stem, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            );
        }
        let transcript = self.transcript.as_ref().expect("bound above");
        let mut meta = target.meta.clone();
        meta.turn_count += 1;
        meta.updated = chrono::Utc::now().to_rfc3339();
        meta.thread_id = thread_id.map(str::to_owned).or(meta.thread_id);
        transcript
            .append_turn_with_partial(
                TranscriptTurn {
                    prev: &self.persisted,
                    next: raw,
                    meta: &meta,
                    turn_usage,
                    request_id,
                },
                partial,
            )
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
        target.meta = meta;
        let previous_len = self.persisted.len();
        let next_len = raw.len();
        let common_len = previous_len.min(next_len);
        let delta = if next_len >= previous_len && raw[..common_len] == self.persisted[..common_len]
        {
            TranscriptDelta::Append {
                previous_len,
                appended: previous_len..next_len,
            }
        } else {
            TranscriptDelta::Replace {
                previous_len,
                next_len,
            }
        };
        Ok(Some(TranscriptCommitReceipt {
            path: transcript.path().to_path_buf(),
            delta,
        }))
    }

    fn with_prefix(&self, history: Vec<Message>) -> Vec<Message> {
        let prefix = self.prefix.messages();
        let overlap = (0..=prefix.len().min(history.len()))
            .rev()
            .find(|&len| prefix[prefix.len() - len..] == history[..len])
            .unwrap_or_default();
        let mut reconciled = prefix[..prefix.len() - overlap].to_vec();
        reconciled.extend(history);
        reconciled
    }
}

/// Ensures a terminal hook is scheduled once even if a caller drops a turn
/// future while it is awaiting preparation, driving, persistence, or hooks.
struct TerminalGuard<C: Clone + Send + Sync + 'static> {
    hooks: Arc<dyn SessionHooks<C>>,
    terminal: Option<SessionTerminal>,
    committed: bool,
}

impl<C: Clone + Send + Sync + 'static> TerminalGuard<C> {
    fn new(hooks: Arc<dyn SessionHooks<C>>) -> Self {
        Self {
            hooks,
            terminal: Some(SessionTerminal::Failed("session turn dropped".into())),
            committed: false,
        }
    }

    fn set(&mut self, terminal: SessionTerminal) {
        self.terminal = Some(terminal);
    }

    fn finalize_commit(&mut self, receipt: CommitReceipt<C>) -> tokio::task::JoinHandle<()> {
        let terminal = SessionTerminal::Completed(receipt.outcome.clone());
        // Removing the guard's terminal transfers exactly-once ownership to
        // the finalizer. `finish` and `Drop` then become no-ops for this turn.
        self.terminal = None;
        self.committed = true;
        let hooks = self.hooks.clone();
        tokio::spawn(async move {
            let _ = hooks.after_commit(receipt).await;
            let _ = hooks.on_terminal(terminal).await;
        })
    }

    fn is_committed(&self) -> bool {
        self.committed
    }

    async fn finish(mut self) -> Result<(), RuntimeError> {
        let Some(terminal) = self.terminal.take() else {
            return Ok(());
        };
        self.hooks.on_terminal(terminal).await
    }
}

impl<C: Clone + Send + Sync + 'static> Drop for TerminalGuard<C> {
    fn drop(&mut self) {
        let Some(terminal) = self.terminal.take() else {
            return;
        };
        let hooks = self.hooks.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = hooks.on_terminal(terminal).await;
            });
        }
    }
}

async fn cancelable<T>(
    cancellation: &CancellationToken,
    future: impl Future<Output = Result<T, RuntimeError>>,
) -> Result<T, RuntimeError> {
    if cancellation.is_cancelled() {
        return Err(RuntimeError::Cancelled);
    }
    tokio::select! {
        _ = cancellation.cancelled() => Err(RuntimeError::Cancelled),
        result = future => result,
    }
}
