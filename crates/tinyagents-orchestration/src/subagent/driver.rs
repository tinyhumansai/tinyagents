use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use tinyagents_tasks::CompletionRouter;
use tokio::sync::{Mutex as AsyncMutex, Notify};

use super::completion::{CompletionOrigin, deliver};
use super::policy::{AttemptSource, apply_outcome_policies, may_retry};
use super::{IncompleteKind, SpawnAdmission, SpawnRejection, SubagentIncomplete, restrict_tools};
use super::{
    PersistedSubagentPause, SubagentError, SubagentExecution, SubagentExecutor, SubagentOutcome,
    SubagentOutcomeKind, SubagentPausePersistenceDisposition, SubagentPersistence,
    SubagentPersistenceDisposition, SubagentPlanner, SubagentRequest, SubagentRunResult,
    SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};
use tinyagents_harness::error::TinyAgentsError;

const LOG_PREFIX: &str = "[subagent-driver]";
use tinyagents_harness::{CancellationToken, context::RunConfig, ids::RunId};

/// Optional host seams accepted by [`SubagentDriver::new`].
///
/// Hosts that cannot provide every seam must receive a typed construction error
/// rather than accidentally executing a partial lifecycle.
pub struct SubagentCapabilities<C: Send + 'static = (), H: Send + 'static = ()> {
    /// Host planner, which resolves policy and explicit execution inputs.
    pub planner: Option<Arc<dyn SubagentPlanner<C, H>>>,
    /// Host executor, which drives the prepared run.
    pub executor: Option<Arc<dyn SubagentExecutor<C>>>,
    /// Host persistence for resume and one lifecycle record.
    pub persistence: Option<Arc<dyn SubagentPersistence>>,
}

/// Generic lifecycle driver for one host's subagent runs.
///
/// Concurrent calls for one scoped task key coalesce, while task ids from
/// distinct parents, roots, or threads remain independent. Hosts still need
/// idempotent persistence for multiple processes/drivers.
pub struct SubagentDriver<C: Send + 'static = (), H: Send + 'static = ()> {
    planner: Arc<dyn SubagentPlanner<C, H>>,
    executor: Arc<dyn SubagentExecutor<C>>,
    persistence: Arc<dyn SubagentPersistence>,
    terminal_outcomes: AsyncMutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    in_flight: Arc<Mutex<HashMap<SubagentTaskKey, Arc<InFlight>>>>,
    admission: SpawnAdmission,
    completions: Option<Arc<CompletionRouter>>,
}

/// Result shared by callers that arrived while the same task was executing.
///
/// The map only protects reservation and removal. Planner, executor, and
/// persistence futures never run while it is locked.
struct InFlight {
    result: Mutex<Option<Result<SubagentRunResult, SubagentError>>>,
    notify: Notify,
}

impl InFlight {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            notify: Notify::new(),
        }
    }

    async fn wait(
        &self,
        cancellation: &CancellationToken,
        task_id: &str,
    ) -> Result<SubagentRunResult, SubagentError> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self
                .result
                .lock()
                .expect("in-flight result poisoned")
                .clone()
            {
                return result;
            }
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Ok(SubagentRunResult::new(SubagentOutcome::cancelled(task_id), SubagentPersistenceDisposition::ObserverCancelled)),
                _ = &mut notified => {}
            }
        }
    }

    fn complete(&self, result: Result<SubagentRunResult, SubagentError>) {
        *self.result.lock().expect("in-flight result poisoned") = Some(result);
        self.notify.notify_waiters();
    }
}

/// Owns the leader's in-flight reservation until its result is published.
///
/// Dropping a caller future must also release its reservation: otherwise a
/// later caller becomes a follower of work that will never publish a result.
struct LeaderReservation {
    in_flight: Arc<Mutex<HashMap<SubagentTaskKey, Arc<InFlight>>>>,
    task_key: SubagentTaskKey,
    entry: Arc<InFlight>,
    finished: bool,
}

impl LeaderReservation {
    fn finish(mut self, result: Result<SubagentRunResult, SubagentError>) {
        self.entry.complete(result);
        let mut in_flight = self.in_flight.lock().expect("in-flight map poisoned");
        if in_flight
            .get(&self.task_key)
            .is_some_and(|current| Arc::ptr_eq(current, &self.entry))
        {
            in_flight.remove(&self.task_key);
        }
        self.finished = true;
    }
}

impl Drop for LeaderReservation {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let in_flight = self.in_flight.clone();
        let task_key = self.task_key.clone();
        let entry = self.entry.clone();
        // Publish a typed result before removal so existing followers cannot
        // wait indefinitely, even if the future is dropped outside a runtime.
        entry.complete(Err(SubagentError::Cancelled));
        let mut in_flight = in_flight.lock().expect("in-flight map poisoned");
        if in_flight
            .get(&task_key)
            .is_some_and(|current| Arc::ptr_eq(current, &entry))
        {
            in_flight.remove(&task_key);
        }
    }
}

impl<C: Send + 'static, H: Send + 'static> SubagentDriver<C, H> {
    /// Validates host capability availability before exposing a runnable driver.
    pub fn new(capabilities: SubagentCapabilities<C, H>) -> Result<Self, SubagentError> {
        Ok(Self {
            planner: capabilities
                .planner
                .ok_or(SubagentError::MissingCapability("planner"))?,
            executor: capabilities
                .executor
                .ok_or(SubagentError::MissingCapability("executor"))?,
            persistence: capabilities
                .persistence
                .ok_or(SubagentError::MissingCapability("persistence"))?,
            terminal_outcomes: AsyncMutex::new(HashMap::new()),
            in_flight: Arc::new(Mutex::new(HashMap::new())),
            admission: SpawnAdmission::default(),
            completions: None,
        })
    }

    /// Records every child this driver finishes with `router`, so its parent is
    /// told according to the child's
    /// [`NotifyMode`](tinyagents_tasks::NotifyMode).
    ///
    /// Only the invocation that wins the durable terminal write records, so a
    /// coalesced follower or a replayed terminal never produces a second
    /// completion. Only a spawn that set a notify mode is recorded. A
    /// cancellation is not recorded (the parent asked for it), neither is a
    /// pause (the same task completes later), nor an executor error (nothing
    /// terminal was persisted, so the task may be re-run; the caller of `run`
    /// has the error). A failure to record
    /// is logged and never fails the run. Without this call the driver behaves
    /// exactly as before.
    pub fn with_completion_router(mut self, router: Arc<CompletionRouter>) -> Self {
        self.completions = Some(router);
        self
    }

    /// Enforces spawn limits on every lifecycle this driver launches.
    ///
    /// A slot is reserved after cancellation and resume loading but before the
    /// planner runs, and released when the lifecycle returns; a lifecycle that
    /// fails before the executor launches also refunds its total-budget unit.
    /// Coalesced followers and cached terminal results never reserve a slot.
    /// Without this call spawning is unlimited.
    pub fn with_spawn_admission(mut self, admission: SpawnAdmission) -> Self {
        self.admission = admission;
        self
    }

    /// The admission ledger this driver reserves from.
    pub fn spawn_admission(&self) -> &SpawnAdmission {
        &self.admission
    }

    /// Runs `load -> prepare -> execute -> one persistence action`.
    ///
    /// A caller-supplied resume bypasses loading. Cancellation is checked before
    /// preparation, after every awaited lifecycle stage, and after execution;
    /// a cancellation observed after an executor result preserves its output,
    /// history, usage, and artifact references while changing only the status.
    pub async fn run(
        &self,
        request: SubagentRequest<C, H>,
        cancellation: CancellationToken,
    ) -> Result<SubagentRunResult, SubagentError> {
        let task_key = request.task_key().clone();
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }
        if let Some(outcome) = self.persistence.load_terminal(&task_key).await? {
            self.cache_terminal(task_key, &outcome).await;
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }

        let (entry, is_leader) = {
            let mut in_flight = self.in_flight.lock().expect("in-flight map poisoned");
            match in_flight.get(&task_key) {
                Some(entry) => (entry.clone(), false),
                None => {
                    let entry = Arc::new(InFlight::new());
                    in_flight.insert(task_key.clone(), entry.clone());
                    (entry, true)
                }
            }
        };
        if !is_leader {
            return entry
                .wait(&cancellation, &task_key.task_id)
                .await
                .map(|result| {
                    SubagentRunResult::new(
                        result.outcome,
                        match result.disposition {
                            SubagentPersistenceDisposition::PauseCommitted
                            | SubagentPersistenceDisposition::PauseReplaced
                            | SubagentPersistenceDisposition::PauseExisting => {
                                SubagentPersistenceDisposition::PauseExisting
                            }
                            SubagentPersistenceDisposition::TerminalInserted
                            | SubagentPersistenceDisposition::TerminalExisting => {
                                SubagentPersistenceDisposition::TerminalExisting
                            }
                            SubagentPersistenceDisposition::ObserverCancelled => {
                                SubagentPersistenceDisposition::ObserverCancelled
                            }
                        },
                    )
                });
        }
        let reservation = LeaderReservation {
            in_flight: self.in_flight.clone(),
            task_key: task_key.clone(),
            entry: entry.clone(),
            finished: false,
        };
        // A preceding caller may have committed a terminal result between the
        // initial cache check and this reservation. Do not reopen that task
        // after its in-flight entry has been removed.
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            reservation.finish(Ok(SubagentRunResult::new(
                outcome.clone(),
                SubagentPersistenceDisposition::TerminalExisting,
            )));
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }

        let result = self
            .run_reserved(request, task_key.clone(), cancellation)
            .await;
        reservation.finish(result.clone());
        result
    }

    async fn run_reserved(
        &self,
        mut request: SubagentRequest<C, H>,
        task_key: SubagentTaskKey,
        cancellation: CancellationToken,
    ) -> Result<SubagentRunResult, SubagentError> {
        let task_id = request.task_id().to_owned();

        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(task_key, SubagentOutcome::cancelled(task_id), None)
                .await;
        }

        if request.resume().is_none() {
            request.set_resume(self.persistence.load(&task_key).await?);
        }
        // Keep the exact pause seen at the durable read boundary. The
        // persistence seam uses it as a scoped compare-and-swap expectation
        // if execution pauses again, so a resumed pause can advance while a
        // duplicate continuation returns the newer durable winner.
        let expected_pause = request.resume().cloned();
        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(
                    task_key,
                    SubagentOutcome::cancelled(task_id),
                    expected_pause,
                )
                .await;
        }

        // Reserve before any planner/executor work. A resumed lifecycle is an
        // existing child, so it takes a live slot but no new total budget.
        // The scope resolves from the parent as the key describes it: the
        // durable thread when there is one, else the parent run id. Fresh spawns
        // and continuations resolve identically.
        //
        // The key carries the parent's run id, thread and the tree's root run
        // id, so a custom scope rule that groups by root
        // (`cfg.lineage.root_run_id`) sees the real root, not the immediate
        // parent. Tags and metadata are not part of the durable key and so are
        // not available to a resolver here.
        let mut parent_config = RunConfig::new(task_key.parent_run_id.as_str());
        parent_config.lineage.root_run_id = RunId::new(task_key.root_run_id.as_str());
        if let Some(thread) = &task_key.thread_id {
            parent_config = parent_config.with_thread(thread.as_str());
        }
        let target = match request.target() {
            Some(target) => target,
            // Fail closed: an allowlist cannot vet a target it was never told.
            None if self.admission.policy().allowed_targets.is_some() => {
                return Err(SubagentError::SpawnRejected(
                    SpawnRejection::TargetNotAllowed {
                        target: String::new(),
                    },
                ));
            }
            None => "",
        };
        let reservation = if request.resume().is_some() {
            self.admission
                .try_reserve_continuation(&parent_config, target)
        } else {
            self.admission.try_reserve(&parent_config, target)
        };
        let mut reservation = reservation.map_err(SubagentError::SpawnRejected)?;

        let mut prepared = self.planner.prepare(request).await?;
        if prepared.task_id != task_id {
            return Err(SubagentError::TaskIdMismatch {
                expected: task_id,
                actual: prepared.task_id,
            });
        }
        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(
                    task_key,
                    SubagentOutcome::cancelled(prepared.task_id),
                    expected_pause,
                )
                .await;
        }
        prepared.tools = restrict_tools(
            &prepared.tools,
            prepared.role,
            prepared.tool_ceiling.as_ref(),
            &prepared.delegation_tools,
        );
        prepared
            .policy
            .budget
            .apply_call_caps(&mut prepared.run_context.config);
        // The child runs under its own token, linked to the lifecycle's: a
        // lifecycle cancellation reaches it, but a policy timeout cancels only
        // the child and never the caller's token.
        let child_token = cancellation.child_token();
        prepared.run_context = prepared.run_context.with_cancellation(child_token.clone());
        // The child is about to launch: from here its total-budget unit stays
        // spent, and the live slot is released when this function returns.
        reservation.commit();

        let policy = prepared.policy.clone();
        let result_policy = prepared.result_policy.clone();
        let completion_origin = self
            .completions
            .as_ref()
            .and_then(|_| CompletionOrigin::new(&task_key, &prepared));
        let mut attempts = AttemptSource::from_prepared(&prepared);
        let mut attempt = 0usize;
        let mut execution = SubagentExecution {
            prepared,
            cancellation: child_token,
        };
        let mut timed_out = false;
        let mut omitted_chars = 0usize;
        let mut overflow_artifact = None;
        let mut token;
        let executed = loop {
            token = execution.cancellation.clone();
            let run = self.executor.execute(execution);
            let result = match policy.timeout {
                Some(limit) => match tokio::time::timeout(limit, run).await {
                    Ok(result) => result,
                    Err(_) => {
                        token.cancel();
                        timed_out = true;
                        tracing::debug!(
                            "{LOG_PREFIX} timeout task_id={task_id} after_ms={}",
                            limit.as_millis()
                        );
                        break Ok(SubagentOutcome::incomplete(
                            task_id.clone(),
                            SubagentIncomplete::new(format!("subagent timed out after {limit:?}"))
                                .with_kind(IncompleteKind::Timeout),
                        ));
                    }
                },
                None => run.await,
            };
            let Err(SubagentError::Transient { message, tools_ran }) = &result else {
                break result;
            };
            let error = TinyAgentsError::Tool(message.clone());
            if cancellation.is_cancelled()
                || token.is_cancelled()
                || !may_retry(&policy, attempt, &error, *tools_ran)
                || !attempts.can_retry()
            {
                break result;
            }
            attempt += 1;
            tracing::debug!("{LOG_PREFIX} retry task_id={task_id} attempt={attempt}");
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => break Err(SubagentError::Cancelled),
                _ = policy.retry.sleep_backoff(attempt) => {}
            }
            let next_token = cancellation.child_token();
            match attempts.next(attempt as u32, next_token.clone()) {
                Ok(next) => {
                    execution = SubagentExecution {
                        prepared: next,
                        cancellation: next_token,
                    };
                }
                Err(error) => break Err(error),
            }
        };

        let outcome = match executed {
            Ok(outcome) => {
                if outcome.task_id != task_id {
                    return Err(SubagentError::TaskIdMismatch {
                        expected: task_id,
                        actual: outcome.task_id,
                    });
                }
                // The execution token is the lifecycle's own or a child of it;
                // an executor that cancels it has cancelled the run, but the
                // driver's own timeout cancel is not a cancellation.
                if cancellation.is_cancelled() || (!timed_out && token.is_cancelled()) {
                    outcome.cancelled_preserving()
                } else {
                    let (outcome, omitted, overflow) =
                        apply_outcome_policies(outcome, &policy, &result_policy).await;
                    omitted_chars = omitted;
                    overflow_artifact = overflow;
                    outcome
                }
            }
            Err(SubagentError::Cancelled) => SubagentOutcome::cancelled(task_id),
            Err(_) if cancellation.is_cancelled() => SubagentOutcome::cancelled(task_id),
            Err(error) => return Err(error),
        };
        let result = self
            .persist(task_key, outcome, expected_pause, &cancellation)
            .await?;
        if let (Some(router), Some(origin)) = (&self.completions, &completion_origin)
            && result.should_emit_host_effects()
            && let Some(record) = origin.record_for_outcome(
                &result.outcome,
                omitted_chars,
                overflow_artifact.as_ref(),
            )
        {
            deliver(router, record).await;
        }
        Ok(result)
    }

    async fn persist_cancelled(
        &self,
        task_key: SubagentTaskKey,
        mut outcome: SubagentOutcome,
        expected_pause: Option<super::SubagentResume>,
    ) -> Result<SubagentRunResult, SubagentError> {
        outcome.status = SubagentOutcomeKind::Cancelled;
        // Once cancellation has won, this is the one terminal action. Do not
        // race it with the already-latched token or a caller could observe an
        // indeterminate terminal write.
        match self
            .persistence
            .record_terminal(&task_key, &outcome, expected_pause.as_ref())
            .await?
        {
            SubagentTerminalPersistenceDisposition::Inserted => {
                self.cache_terminal(task_key, &outcome).await;
                Ok(SubagentRunResult::new(
                    outcome,
                    SubagentPersistenceDisposition::TerminalInserted,
                ))
            }
            SubagentTerminalPersistenceDisposition::Existing => {
                self.load_terminal_winner(task_key).await
            }
            SubagentTerminalPersistenceDisposition::PauseExisting => {
                self.load_pause_winner(task_key).await
            }
        }
    }

    async fn persist(
        &self,
        task_key: SubagentTaskKey,
        outcome: SubagentOutcome,
        expected_pause: Option<super::SubagentResume>,
        cancellation: &CancellationToken,
    ) -> Result<SubagentRunResult, SubagentError> {
        if matches!(&outcome.status, SubagentOutcomeKind::Cancelled) {
            return self
                .persist_cancelled(task_key, outcome, expected_pause)
                .await;
        }
        let disposition = match &outcome.status {
            SubagentOutcomeKind::AwaitingInput(_) => {
                match self
                    .commit_or_cancel(
                        self.persistence.save_pause(PersistedSubagentPause {
                            key: task_key.clone(),
                            outcome: outcome.clone(),
                            replaces: expected_pause.clone(),
                        }),
                        cancellation,
                    )
                    .await?
                {
                    Some(SubagentPausePersistenceDisposition::Inserted) => {
                        SubagentPersistenceDisposition::PauseCommitted
                    }
                    Some(SubagentPausePersistenceDisposition::Replaced) => {
                        SubagentPersistenceDisposition::PauseReplaced
                    }
                    Some(SubagentPausePersistenceDisposition::Existing) => {
                        return self.load_pause_winner(task_key).await;
                    }
                    Some(SubagentPausePersistenceDisposition::TerminalExisting) => {
                        return self.load_terminal_winner(task_key).await;
                    }
                    None => {
                        return self
                            .persist_cancelled(task_key, outcome, expected_pause)
                            .await;
                    }
                }
            }
            SubagentOutcomeKind::Completed | SubagentOutcomeKind::Incomplete(_) => {
                match self
                    .commit_or_cancel(
                        async {
                            self.persistence
                                .record_terminal(&task_key, &outcome, expected_pause.as_ref())
                                .await
                        },
                        cancellation,
                    )
                    .await?
                {
                    Some(SubagentTerminalPersistenceDisposition::Inserted) => {
                        SubagentPersistenceDisposition::TerminalInserted
                    }
                    Some(SubagentTerminalPersistenceDisposition::Existing) => {
                        SubagentPersistenceDisposition::TerminalExisting
                    }
                    Some(SubagentTerminalPersistenceDisposition::PauseExisting) => {
                        return self.load_pause_winner(task_key).await;
                    }
                    None => {
                        return self
                            .persist_cancelled(task_key, outcome, expected_pause)
                            .await;
                    }
                }
            }
            SubagentOutcomeKind::Cancelled => unreachable!("handled before persistence race"),
        };
        if matches!(
            &outcome.status,
            SubagentOutcomeKind::Completed | SubagentOutcomeKind::Incomplete(_)
        ) {
            let terminal = if matches!(
                disposition,
                SubagentPersistenceDisposition::TerminalExisting
            ) {
                self.persistence
                    .load_terminal(&task_key)
                    .await?
                    .ok_or_else(|| {
                        SubagentError::Persistence(
                            "terminal insert lost without durable outcome".into(),
                        )
                    })?
            } else {
                outcome.clone()
            };
            self.cache_terminal(task_key, &terminal).await;
            return Ok(SubagentRunResult::new(terminal, disposition));
        }
        Ok(SubagentRunResult::new(outcome, disposition))
    }

    async fn load_pause_winner(
        &self,
        task_key: SubagentTaskKey,
    ) -> Result<SubagentRunResult, SubagentError> {
        // A terminal can win after the pause CAS reports contention. It is
        // authoritative over every prior pause and must never be reopened.
        if let Some(terminal) = self.persistence.load_terminal(&task_key).await? {
            self.cache_terminal(task_key, &terminal).await;
            return Ok(SubagentRunResult::new(
                terminal,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }
        let Some(paused) = self.persistence.load_pause(&task_key).await? else {
            // A different driver can consume the pause into a terminal between
            // the first terminal read and this pause read. Recheck the
            // authoritative terminal before classifying that interleaving as a
            // persistence failure.
            if let Some(terminal) = self.persistence.load_terminal(&task_key).await? {
                self.cache_terminal(task_key, &terminal).await;
                return Ok(SubagentRunResult::new(
                    terminal,
                    SubagentPersistenceDisposition::TerminalExisting,
                ));
            }
            return Err(SubagentError::Persistence(
                "pause compare-and-swap lost without a durable pause or terminal outcome".into(),
            ));
        };
        if !matches!(paused.status, SubagentOutcomeKind::AwaitingInput(_)) {
            return Err(SubagentError::Persistence(
                "durable pause record did not contain an awaiting-input outcome".into(),
            ));
        }
        Ok(SubagentRunResult::new(
            paused,
            SubagentPersistenceDisposition::PauseExisting,
        ))
    }

    async fn load_terminal_winner(
        &self,
        task_key: SubagentTaskKey,
    ) -> Result<SubagentRunResult, SubagentError> {
        let terminal = self
            .persistence
            .load_terminal(&task_key)
            .await?
            .ok_or_else(|| {
                SubagentError::Persistence(
                    "terminal compare-and-swap lost without a durable terminal outcome".into(),
                )
            })?;
        self.cache_terminal(task_key, &terminal).await;
        Ok(SubagentRunResult::new(
            terminal,
            SubagentPersistenceDisposition::TerminalExisting,
        ))
    }

    async fn commit_or_cancel<F, T>(
        &self,
        operation: F,
        cancellation: &CancellationToken,
    ) -> Result<Option<T>, SubagentError>
    where
        F: Future<Output = Result<T, SubagentError>>,
    {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Ok(None),
            result = operation => {
                Ok(Some(result?))
            }
        }
    }

    async fn cache_terminal(&self, key: SubagentTaskKey, outcome: &SubagentOutcome) {
        self.terminal_outcomes
            .lock()
            .await
            .insert(key, outcome.clone());
    }
}
