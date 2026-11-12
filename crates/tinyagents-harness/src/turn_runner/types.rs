use std::{
    collections::HashMap,
    future::Future,
    hash::Hash,
    sync::{Arc, Weak},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio::time::Instant;

use crate::progress::AgentProgress;
use crate::run_queue::{QueueLane, QueueMode, QueueStatus, QueuedMessage, RunQueue};

/// Identity that isolates session locks and active turns between agents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopeKey {
    pub agent_id: String,
    pub profile_id: Option<String>,
}

/// A request after the host has supplied its channel-independent metadata.
#[derive(Debug, Clone)]
pub struct TurnRequest<P = serde_json::Value> {
    pub scope: ScopeKey,
    pub thread_id: String,
    pub request_id: String,
    pub text: String,
    /// Lossless host-defined input metadata such as attachments, access flags,
    /// locale, or channel attribution.
    pub payload: P,
    pub host_authored: bool,
    pub disposition: QueueDisposition,
    pub deadline: Option<Duration>,
    pub idle_timeout: Option<Duration>,
}

/// A request item stored by a runner and shared with its host's running agent
/// loop. Implementations must preserve the complete request in both directions.
pub trait TurnQueueItem<P>: Clone + Send + Sync + 'static {
    /// Wraps an accepted request without changing its identity or payload.
    fn from_turn_request(request: TurnRequest<P>) -> Self;

    /// Borrows the request represented by this item.
    fn as_turn_request(&self) -> &TurnRequest<P>;

    /// Consumes this item into the request it represents.
    fn into_turn_request(self) -> TurnRequest<P>;
}

impl<P> TurnQueueItem<P> for TurnRequest<P>
where
    P: Clone + Send + Sync + 'static,
{
    fn from_turn_request(request: TurnRequest<P>) -> Self {
        request
    }

    fn as_turn_request(&self) -> &TurnRequest<P> {
        self
    }

    fn into_turn_request(self) -> TurnRequest<P> {
        self
    }
}

impl<P> QueuedMessage for TurnRequest<P>
where
    P: RequeuePayload + Send + 'static,
{
    fn id(&self) -> &str {
        &self.request_id
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn requeued(id: String, text: String, thread_label: &str, queued_at_ms: u64) -> Self {
        Self {
            scope: ScopeKey {
                agent_id: String::new(),
                profile_id: None,
            },
            thread_id: thread_label.to_owned(),
            request_id: id,
            text,
            payload: P::requeued(thread_label, queued_at_ms),
            host_authored: false,
            disposition: QueueDisposition::Steer,
            deadline: None,
            idle_timeout: None,
        }
    }

    fn requeued_in_lane(
        id: String,
        text: String,
        thread_label: &str,
        lane: QueueLane,
        queued_at_ms: u64,
    ) -> Self {
        let mut request = Self::requeued(id, text, thread_label, queued_at_ms);
        request.disposition = match lane {
            QueueLane::Steer => QueueDisposition::Steer,
            QueueLane::Followup => QueueDisposition::Followup,
            QueueLane::Collect => QueueDisposition::Collect,
        };
        request
    }
}

// Untyped JSON payloads have no host schema, so only known turn-scoped fields
// are inherited; request-local values such as attachments stay with their
// original request.
impl RequeuePayload for serde_json::Value {
    fn requeued(thread_id: &str, queued_at_ms: u64) -> Self {
        serde_json::json!({
            "thread_id": thread_id,
            "queued_at_ms": queued_at_ms,
        })
    }

    fn adopt(&mut self, running: &Self) {
        if let (Self::Object(requeued), Self::Object(active)) = (self, running) {
            for key in [
                "access",
                "locale",
                "attribution",
                "channel_attribution",
                "client_id",
                "model_override",
                "temperature",
            ] {
                if let Some(value) = active.get(key) {
                    requeued
                        .entry(key.to_owned())
                        .or_insert_with(|| value.clone());
                }
            }
        }
    }

    fn restore_thread_id(&mut self, thread_id: &str) {
        if let Self::Object(payload) = self
            && payload.contains_key("thread_id")
        {
            payload.insert(
                "thread_id".into(),
                serde_json::Value::String(thread_id.into()),
            );
        }
    }
}

/// Host-owned identity and fingerprint for the checked-out durable session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnSession {
    pub session_id: String,
    pub fingerprint: Option<String>,
}

/// Pipeline stages are called in this order for every submitted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressStage {
    StageAttachments,
    Guardrail,
    ApprovalInterception,
    BeforeSubmit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngressDecision {
    Continue,
    /// The policy consumed the request, for example by routing an approval reply.
    Consumed,
    Reject(String),
}

#[async_trait]
pub trait IngressPolicy<P = serde_json::Value>: Send + Sync {
    async fn apply(
        &self,
        stage: IngressStage,
        request: &mut TurnRequest<P>,
    ) -> Result<IngressDecision, String>;
}

/// The five acceptance behaviors exposed by channel turn APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueDisposition {
    Interrupt,
    Steer,
    Followup,
    Collect,
    Parallel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalEvent {
    Completed {
        request_id: String,
        reply_id: String,
    },
    Failed {
        request_id: String,
        message: String,
        class: String,
    },
    Cancelled {
        request_id: String,
    },
}

/// Activity updates sent by the host so the idle guard distinguishes silence
/// from a model or tool call that is still running.
#[derive(Debug, Clone)]
pub enum TurnSignal {
    Progress(Box<AgentProgress>),
    WorkStarted,
    WorkFinished,
}

/// Host-defined metadata for a classified turn failure.
///
/// The runner owns the stable class and user-facing message. Hosts can attach
/// structured details for their own presentation layer without teaching the
/// harness about host-specific localization or wire formats.
#[derive(Debug, Clone, PartialEq)]
pub struct FailureClassification {
    /// Stable machine-readable failure class.
    pub class: String,
    /// User-facing summary for the failed turn.
    pub message: String,
    /// Optional host-defined structured presentation details.
    pub details: Option<serde_json::Value>,
}

/// Creates the host-owned payload for a queued turn and adopts safe metadata
/// from the active turn when the host needs to synthesize a follow-up request.
///
/// Payload implementations decide which fields may be inherited. Request text,
/// ids, and queue timestamps remain owned by the newly queued request.
pub trait RequeuePayload: Sized {
    /// Create an empty payload for a request queued on `thread_id`.
    fn requeued(thread_id: &str, queued_at_ms: u64) -> Self;

    /// Copy any safe, turn-scoped host metadata from the running payload.
    fn adopt(&mut self, running: &Self);

    /// Replace a thread identity embedded in the payload with the queue owner's
    /// routing identity. Payloads without their own thread field may leave this
    /// as the default no-op.
    fn restore_thread_id(&mut self, _thread_id: &str) {}
}

#[async_trait]
pub trait TurnHost<P = serde_json::Value, Q = TurnRequest<P>>: Send + Sync {
    /// Check out or create the session for this request and verify its fingerprint.
    async fn checkout(&self, request: &TurnRequest<P>) -> Result<TurnSession, String>;
    /// Check out an independent transcript/session snapshot for Parallel mode.
    async fn checkout_fork(&self, request: &TurnRequest<P>) -> Result<TurnSession, String>;
    /// Confirm the checked-out session still matches the request's expected
    /// transcript/session identity before any model work begins.
    async fn validate_fingerprint(
        &self,
        _request: &TurnRequest<P>,
        _session: &TurnSession,
    ) -> Result<(), String> {
        Ok(())
    }
    /// Build the host's origin value (channel, actor and request attribution).
    fn run_origin(&self, request: &TurnRequest<P>) -> serde_json::Value;
    /// Run the actual harness/session turn. Progress is forwarded to the sink.
    async fn run(
        &self,
        request: &TurnRequest<P>,
        session: &TurnSession,
        origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<Q>>,
        progress: mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String>;
    /// Release the session after the run, including on cancellation or failure.
    async fn checkin(&self, request: &TurnRequest<P>);
    /// Cancel child runs belonging to this parent request.
    async fn cancel_children(&self, request: &TurnRequest<P>);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnResult {
    pub reply: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

#[async_trait]
pub trait ReplyStore<P = serde_json::Value>: Send + Sync {
    async fn persist(
        &self,
        request: &TurnRequest<P>,
        reply_id: &str,
        result: &TurnResult,
    ) -> Result<(), String>;
}

#[async_trait]
pub trait EventSink<P = serde_json::Value>: Send + Sync
where
    P: Send + Sync,
{
    async fn progress(&self, request: &TurnRequest<P>, event: AgentProgress);
    async fn terminal(&self, request: &TurnRequest<P>, event: TerminalEvent);
    /// Deliver a terminal failure with its structured host classification.
    ///
    /// The default keeps existing event sinks source-compatible and forwards
    /// the wire-neutral terminal event. Hosts that render richer failure copy
    /// can override this method.
    async fn classified_failure(
        &self,
        request: &TurnRequest<P>,
        event: TerminalEvent,
        _classification: FailureClassification,
    ) {
        self.terminal(request, event).await;
    }
    async fn suggestions(&self, _request: &TurnRequest<P>, _suggestions: Vec<String>) {}
    async fn collected(&self, _request: &TurnRequest<P>, _items: Vec<TurnRequest<P>>) {}
}

pub trait FailureClassifier: Send + Sync {
    fn classify(&self, error: &str) -> (String, String);

    /// Classify an error with optional host-defined presentation details.
    ///
    /// The default preserves classifiers that only return a class and message.
    fn classify_failure(&self, error: &str) -> FailureClassification {
        let (class, message) = self.classify(error);
        FailureClassification {
            class,
            message,
            details: None,
        }
    }
}

#[async_trait]
pub trait Summarizer<P = serde_json::Value>: Send + Sync {
    /// Optional next-message suggestions for a successfully completed turn.
    async fn suggestions(&self, _request: &TurnRequest<P>, _result: &TurnResult) -> Vec<String> {
        Vec::new()
    }
    /// Ground a bounded close in the work completed so far after a guard fires.
    async fn grounded_close(&self, request: &TurnRequest<P>, reason: &str) -> Option<TurnResult>;
}

/// Runner guard defaults. The idle timeout catches silence; the deadline is a
/// separate total wall-clock ceiling and can be overridden per request.
#[derive(Debug, Clone, Copy)]
pub struct TurnRunnerConfig {
    pub idle_timeout: Duration,
    pub deadline: Option<Duration>,
    pub close_timeout: Duration,
    /// How long a cancelled host run keeps being polled, with its own
    /// cancellation token already fired, before it is dropped. A host uses the
    /// window to hand back and persist the rounds it completed. Zero drops the
    /// run immediately.
    pub cancel_grace: Duration,
}

impl Default for TurnRunnerConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(900),
            deadline: None,
            close_timeout: Duration::from_secs(60),
            cancel_grace: Duration::ZERO,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TurnRunnerError {
    #[error("turn was consumed by ingress policy")]
    Consumed,
    #[error("turn was accepted into the host queue")]
    Queued,
    #[error("turn rejected: {0}")]
    Rejected(String),
    #[error("turn failed: {0}")]
    Failed(String),
    #[error("turn cancelled")]
    Cancelled,
}

#[derive(Clone)]
pub(crate) struct ActiveTurn<P, Q = TurnRequest<P>> {
    pub(crate) request_id: String,
    pub(crate) cancel: crate::CancellationToken,
    pub(crate) cancellable: bool,
    pub(crate) generation: u64,
    pub(crate) queue: Arc<RunQueue<Q>>,
    pub(crate) payload: P,
}

#[derive(Clone)]
struct ActiveParallelTurn {
    cancel: crate::CancellationToken,
    cancellable: bool,
}

/// A live turn owned by a runner, suitable for host-side cancellation and
/// observability without exposing cancellation tokens or queue internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTurn {
    /// Agent/profile scope that owns this turn.
    pub scope: ScopeKey,
    /// Conversation thread supplied by the host.
    pub thread_id: String,
    /// Host request identity used to scope cancellation and terminal events.
    pub request_id: String,
    /// Whether this turn runs on an independent parallel fork.
    pub parallel: bool,
}

type ThreadLockKey = (ScopeKey, String);
type ThreadLocks = Mutex<HashMap<ThreadLockKey, Weak<Mutex<()>>>>;
type ParallelQueueKey = (ScopeKey, String, String);
type ParallelQueueMap<Q> = Mutex<HashMap<ParallelQueueKey, Arc<RunQueue<Q>>>>;

fn queue_lane(disposition: QueueDisposition) -> Option<QueueLane> {
    match disposition {
        QueueDisposition::Steer => Some(QueueLane::Steer),
        QueueDisposition::Followup => Some(QueueLane::Followup),
        QueueDisposition::Collect => Some(QueueLane::Collect),
        QueueDisposition::Interrupt | QueueDisposition::Parallel => None,
    }
}

fn restore_queue_context<P: RequeuePayload>(
    requests: &mut [TurnRequest<P>],
    scope: &ScopeKey,
    thread_id: &str,
    running_payload: &P,
) {
    for request in requests {
        request.scope.clone_from(scope);
        request.thread_id = thread_id.to_owned();
        request.payload.adopt(running_payload);
        request.payload.restore_thread_id(thread_id);
    }
}

/// Reusable turn lifecycle coordinator, parameterized entirely by host ports.
pub struct TurnRunner<H, I, R, E, F, S, P = serde_json::Value, Q = TurnRequest<P>> {
    host: Arc<H>,
    ingress: Arc<I>,
    replies: Arc<R>,
    events: Arc<E>,
    failures: Arc<F>,
    summarizer: Arc<S>,
    config: TurnRunnerConfig,
    locks: ThreadLocks,
    parallel_queues: ParallelQueueMap<Q>,
    pub(crate) active: Mutex<HashMap<(ScopeKey, String), ActiveTurn<P, Q>>>,
    parallel_active: Mutex<HashMap<(ScopeKey, String, String), ActiveParallelTurn>>,
    next_generation: std::sync::atomic::AtomicU64,
    payload: std::marker::PhantomData<fn() -> P>,
}

impl<H, I, R, E, F, S, P, Q> TurnRunner<H, I, R, E, F, S, P, Q>
where
    H: TurnHost<P, Q> + 'static,
    I: IngressPolicy<P> + 'static,
    R: ReplyStore<P> + 'static,
    E: EventSink<P> + 'static,
    F: FailureClassifier + 'static,
    S: Summarizer<P> + 'static,
    P: RequeuePayload + Clone + Send + Sync + 'static,
    Q: TurnQueueItem<P>,
{
    pub fn new(
        host: Arc<H>,
        ingress: Arc<I>,
        replies: Arc<R>,
        events: Arc<E>,
        failures: Arc<F>,
        summarizer: Arc<S>,
        config: TurnRunnerConfig,
    ) -> Self {
        Self {
            host,
            ingress,
            replies,
            events,
            failures,
            summarizer,
            config,
            locks: Mutex::new(HashMap::new()),
            parallel_queues: Mutex::new(HashMap::new()),
            active: Mutex::new(HashMap::new()),
            parallel_active: Mutex::new(HashMap::new()),
            next_generation: std::sync::atomic::AtomicU64::new(1),
            payload: std::marker::PhantomData,
        }
    }

    /// Cancel only the active generation named by `request_id`; a stale cancel
    /// cannot target a newer turn that reused the same thread.
    pub async fn cancel(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: Option<&str>,
    ) -> bool {
        !self
            .cancel_requests(scope, thread_id, request_id)
            .await
            .is_empty()
    }

    /// Cancel matching active turns and return their request IDs, with the
    /// primary turn first followed by parallel turns in request-ID order.
    /// A stale request ID is a no-op and cannot cancel a newer generation.
    pub async fn cancel_requests(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: Option<&str>,
    ) -> Vec<String> {
        let key = (scope.clone(), thread_id.to_owned());
        let mut active = self.active.lock().await;
        let mut cancelled = Vec::new();
        if let Some(turn) = active
            .get_mut(&key)
            .filter(|turn| turn.cancellable && request_id.is_none_or(|id| turn.request_id == id))
        {
            turn.cancellable = false;
            turn.queue.clear().await;
            turn.cancel.cancel();
            cancelled.push(turn.request_id.clone());
        }
        drop(active);
        let mut parallel = self.parallel_active.lock().await;
        let mut keys: Vec<_> = parallel
            .iter()
            .filter(|((owner, thread, req), turn)| {
                turn.cancellable
                    && owner == scope
                    && thread == thread_id
                    && request_id.is_none_or(|id| req == id)
            })
            .map(|(key, _)| key.clone())
            .collect();
        keys.sort_by(|left, right| left.2.cmp(&right.2));
        for key in &keys {
            if let Some(turn) = parallel.get_mut(key) {
                turn.cancellable = false;
                turn.cancel.cancel();
            }
            cancelled.push(key.2.clone());
        }
        cancelled
    }

    /// Whether the thread owns a primary user-authored turn cancellation slot.
    pub async fn is_in_flight(&self, scope: &ScopeKey, thread_id: &str) -> bool {
        self.active
            .lock()
            .await
            .contains_key(&(scope.clone(), thread_id.to_owned()))
    }

    pub async fn in_flight_request_id(&self, scope: &ScopeKey, thread_id: &str) -> Option<String> {
        self.active
            .lock()
            .await
            .get(&(scope.clone(), thread_id.to_owned()))
            .map(|turn| turn.request_id.clone())
    }

    /// Snapshot every active primary and parallel turn in deterministic order.
    /// The result contains identities only; cancellation remains owned by the
    /// runner and must go through [`Self::cancel_requests`].
    pub async fn live_turns(&self) -> Vec<LiveTurn> {
        let mut live = Vec::new();
        for ((scope, thread_id), turn) in self.active.lock().await.iter() {
            live.push(LiveTurn {
                scope: scope.clone(),
                thread_id: thread_id.clone(),
                request_id: turn.request_id.clone(),
                parallel: false,
            });
        }
        for (scope, thread_id, request_id) in self.parallel_active.lock().await.keys() {
            live.push(LiveTurn {
                scope: scope.clone(),
                thread_id: thread_id.clone(),
                request_id: request_id.clone(),
                parallel: true,
            });
        }
        live.sort_by(|left, right| {
            (
                left.scope.agent_id.as_str(),
                left.scope.profile_id.as_deref(),
                left.thread_id.as_str(),
                left.parallel,
                left.request_id.as_str(),
            )
                .cmp(&(
                    right.scope.agent_id.as_str(),
                    right.scope.profile_id.as_deref(),
                    right.thread_id.as_str(),
                    right.parallel,
                    right.request_id.as_str(),
                ))
        });
        live
    }

    pub async fn run(&self, request: TurnRequest<P>) -> Result<TurnResult, TurnRunnerError> {
        let mut state = None;
        let mut next = None;
        let result = self
            .run_one(request.clone(), None, &mut state, &mut next)
            .await;
        if result.is_ok() && request.disposition != QueueDisposition::Parallel {
            while let Some((followup, reserved)) = next.take() {
                let mut current = Some(reserved.clone());
                let mut following = None;
                if self
                    .run_one(
                        followup.clone(),
                        Some(reserved),
                        &mut current,
                        &mut following,
                    )
                    .await
                    .is_err()
                {
                    if let Some(state) = current.as_ref() {
                        self.clear_generation_queue(
                            &followup.scope,
                            &followup.thread_id,
                            state.generation,
                        )
                        .await;
                        self.clear_tracking(&followup, state.generation).await;
                    }
                    break;
                }
                next = following;
            }
        } else {
            if let Some(state) = state.as_ref() {
                if result.is_err() {
                    self.clear_generation_queue(
                        &request.scope,
                        &request.thread_id,
                        state.generation,
                    )
                    .await;
                }
                self.clear_tracking(&request, state.generation).await;
            } else if request.disposition == QueueDisposition::Parallel {
                self.clear_parallel_tracking(&request).await;
            }
        }
        result
    }

    async fn run_one(
        &self,
        mut request: TurnRequest<P>,
        reservation: Option<ActiveTurn<P, Q>>,
        state_out: &mut Option<ActiveTurn<P, Q>>,
        next_out: &mut Option<(TurnRequest<P>, ActiveTurn<P, Q>)>,
    ) -> Result<TurnResult, TurnRunnerError> {
        let deadline = TurnDeadline::new(request.deadline.or(self.config.deadline));
        let cancellation = reservation
            .as_ref()
            .map(|state| state.cancel.clone())
            .unwrap_or_default();
        if reservation.is_none() {
            for stage in [
                IngressStage::StageAttachments,
                IngressStage::Guardrail,
                IngressStage::ApprovalInterception,
                IngressStage::BeforeSubmit,
            ] {
                match self
                    .before_deadline(
                        deadline.at,
                        &cancellation,
                        self.ingress.apply(stage, &mut request),
                    )
                    .await
                {
                    Ok(Ok(IngressDecision::Continue)) => {}
                    Ok(Ok(IngressDecision::Consumed)) => return Err(TurnRunnerError::Consumed),
                    Ok(Ok(IngressDecision::Reject(reason))) => {
                        return Err(TurnRunnerError::Rejected(reason));
                    }
                    Ok(Err(error)) => return self.fail_generation(&request, error, None).await,
                    Err(PreRunStop::Cancelled) => {
                        return self.cancel_generation(&request, None).await;
                    }
                    Err(PreRunStop::Deadline) => {
                        return self
                            .fail_generation(&request, "turn wall-clock deadline".into(), None)
                            .await;
                    }
                }
            }
        }

        let key = (request.scope.clone(), request.thread_id.clone());
        let mut primary_queue = reservation.as_ref().map(|state| state.queue.clone());
        let mut active_generation = reservation.as_ref().map(|state| state.generation);
        if !request.host_authored {
            if request.disposition == QueueDisposition::Parallel {
                self.parallel_active.lock().await.insert(
                    (
                        request.scope.clone(),
                        request.thread_id.clone(),
                        request.request_id.clone(),
                    ),
                    ActiveParallelTurn {
                        cancel: cancellation.clone(),
                        cancellable: true,
                    },
                );
            } else if let Some(reserved) = reservation.as_ref() {
                let mut active = self.active.lock().await;
                if active
                    .get(&key)
                    .is_none_or(|current| current.generation != reserved.generation)
                {
                    drop(active);
                    return self
                        .cancel_generation(&request, Some(reserved.generation))
                        .await;
                }
                let mut state = reserved.clone();
                state.request_id = request.request_id.clone();
                state.cancel = cancellation.clone();
                state.payload.clone_from(&request.payload);
                active.insert(key.clone(), state.clone());
                active_generation = Some(state.generation);
                *state_out = Some(state);
            } else {
                let mut active = self.active.lock().await;
                if let Some(lane) = queue_lane(request.disposition)
                    && let Some(current) = active.get(&key)
                {
                    current
                        .queue
                        .push(lane, Q::from_turn_request(request.clone()))
                        .await;
                    return Err(TurnRunnerError::Queued);
                }
                let queue = Arc::new(RunQueue::new());
                let state = ActiveTurn {
                    request_id: request.request_id.clone(),
                    cancel: cancellation.clone(),
                    cancellable: true,
                    generation: self
                        .next_generation
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    queue: queue.clone(),
                    payload: request.payload.clone(),
                };
                let previous = active.insert(key.clone(), state.clone());
                active_generation = Some(state.generation);
                *state_out = Some(state);
                primary_queue = Some(queue);
                if let Some(previous) = previous {
                    if request.disposition == QueueDisposition::Interrupt {
                        previous.queue.clear().await;
                    }
                    previous.cancel.cancel();
                }
            }
        }
        let run_queue = if request.disposition == QueueDisposition::Parallel {
            self.parallel_queue_for((
                request.scope.clone(),
                request.thread_id.clone(),
                request.request_id.clone(),
            ))
            .await
        } else if let Some(queue) = primary_queue {
            queue
        } else {
            Arc::new(RunQueue::new())
        };
        let lock = {
            let mut locks = self.locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(key.clone(), Arc::downgrade(&lock));
                lock
            }
        };
        let _session_guard = if request.disposition == QueueDisposition::Parallel {
            None
        } else {
            match self
                .before_deadline(deadline.at, &cancellation, lock.lock_owned())
                .await
            {
                Ok(guard) => Some(guard),
                Err(PreRunStop::Cancelled) => {
                    return self.cancel_generation(&request, active_generation).await;
                }
                Err(PreRunStop::Deadline) => {
                    return self
                        .fail_generation(
                            &request,
                            "turn wall-clock deadline".into(),
                            active_generation,
                        )
                        .await;
                }
            }
        };
        let checkout = if request.disposition == QueueDisposition::Parallel {
            self.before_deadline(
                deadline.at,
                &cancellation,
                self.host.checkout_fork(&request),
            )
            .await
        } else {
            self.before_deadline(deadline.at, &cancellation, self.host.checkout(&request))
                .await
        };
        let session = match checkout {
            Ok(Ok(session)) => session,
            Ok(Err(error)) => {
                return self
                    .fail_generation(&request, error, active_generation)
                    .await;
            }
            Err(PreRunStop::Cancelled) => {
                return self.cancel_generation(&request, active_generation).await;
            }
            Err(PreRunStop::Deadline) => {
                return self
                    .fail_generation(
                        &request,
                        "turn wall-clock deadline".into(),
                        active_generation,
                    )
                    .await;
            }
        };
        match self
            .before_deadline(
                deadline.at,
                &cancellation,
                self.host.validate_fingerprint(&request, &session),
            )
            .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.host.checkin(&request).await;
                return self
                    .fail_generation(&request, error, active_generation)
                    .await;
            }
            Err(PreRunStop::Cancelled) => {
                self.host.checkin(&request).await;
                return self.cancel_generation(&request, active_generation).await;
            }
            Err(PreRunStop::Deadline) => {
                self.host.checkin(&request).await;
                return self
                    .fail_generation(
                        &request,
                        "turn wall-clock deadline".into(),
                        active_generation,
                    )
                    .await;
            }
        }
        let origin = self.host.run_origin(&request);

        let result = self
            .run_guarded(
                &request,
                &session,
                &origin,
                cancellation.clone(),
                Arc::clone(&run_queue),
                deadline,
            )
            .await;
        if result.is_ok()
            && !self
                .close_cancellation_window(&request, active_generation, &cancellation)
                .await
        {
            self.host.checkin(&request).await;
            drop(_session_guard);
            return self.cancel_generation(&request, active_generation).await;
        }
        self.host.checkin(&request).await;
        drop(_session_guard);

        match result {
            Ok(result) => {
                let reply_id =
                    deterministic_reply_id(&request.scope, &request.thread_id, &request.request_id);
                if let Err(error) = self.replies.persist(&request, &reply_id, &result).await {
                    return self
                        .fail_generation(&request, error, active_generation)
                        .await;
                }
                let suggestions = self.summarizer.suggestions(&request, &result).await;
                self.events.suggestions(&request, suggestions).await;
                if !request.host_authored && request.disposition != QueueDisposition::Parallel {
                    let (collected, next) = match active_generation {
                        Some(generation) => {
                            self.finalize_generation(&request.scope, &request.thread_id, generation)
                                .await
                        }
                        None => (Vec::new(), None),
                    };
                    *next_out = next.clone();
                    *state_out = next.map(|(_, state)| state);
                    self.events.collected(&request, collected).await;
                }
                self.events
                    .terminal(
                        &request,
                        TerminalEvent::Completed {
                            request_id: request.request_id.clone(),
                            reply_id,
                        },
                    )
                    .await;
                Ok(result)
            }
            Err(_error) if cancellation.is_cancelled() => {
                self.cancel_generation(&request, active_generation).await
            }
            Err(error) => {
                self.fail_generation(&request, error, active_generation)
                    .await
            }
        }
    }

    async fn parallel_queue_for(&self, key: ParallelQueueKey) -> Arc<RunQueue<Q>> {
        let mut queues = self.parallel_queues.lock().await;
        Arc::clone(
            queues
                .entry(key)
                .or_insert_with(|| Arc::new(RunQueue::new())),
        )
    }

    async fn before_deadline<T>(
        &self,
        deadline: Option<Instant>,
        cancellation: &crate::CancellationToken,
        future: impl Future<Output = T>,
    ) -> Result<T, PreRunStop> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(PreRunStop::Cancelled),
            _ = async {
                if let Some(deadline) = deadline {
                    tokio::time::sleep_until(deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            }, if deadline.is_some() => Err(PreRunStop::Deadline),
            result = future => Ok(result),
        }
    }

    pub(crate) async fn retire_generation(
        &self,
        request: &TurnRequest<P>,
        generation: Option<u64>,
    ) -> Vec<TurnRequest<P>> {
        let Some(generation) = generation else {
            return Vec::new();
        };
        let key = (request.scope.clone(), request.thread_id.clone());
        let mut active = self.active.lock().await;
        let Some(turn) = active
            .get(&key)
            .filter(|turn| turn.generation == generation)
            .cloned()
        else {
            return Vec::new();
        };
        active.remove(&key);
        let mut queued = Vec::new();
        for lane in [QueueLane::Steer, QueueLane::Followup, QueueLane::Collect] {
            queued.extend(
                turn.queue
                    .take(lane, QueueMode::All)
                    .await
                    .into_iter()
                    .map(TurnQueueItem::into_turn_request),
            );
        }
        restore_queue_context(
            &mut queued,
            &request.scope,
            &request.thread_id,
            &turn.payload,
        );
        queued
    }

    async fn cancel_generation(
        &self,
        request: &TurnRequest<P>,
        generation: Option<u64>,
    ) -> Result<TurnResult, TurnRunnerError> {
        let queued = self.retire_generation(request, generation).await;
        if request.disposition == QueueDisposition::Parallel {
            self.clear_parallel_tracking(request).await;
        }
        for queued_request in queued {
            self.events
                .terminal(
                    &queued_request,
                    TerminalEvent::Cancelled {
                        request_id: queued_request.request_id.clone(),
                    },
                )
                .await;
        }
        self.events
            .terminal(
                request,
                TerminalEvent::Cancelled {
                    request_id: request.request_id.clone(),
                },
            )
            .await;
        Err(TurnRunnerError::Cancelled)
    }

    async fn fail_generation<T>(
        &self,
        request: &TurnRequest<P>,
        error: String,
        generation: Option<u64>,
    ) -> Result<T, TurnRunnerError> {
        let classification = self.failures.classify_failure(&error);
        let class = classification.class.clone();
        let message = classification.message.clone();
        let queued = self.retire_generation(request, generation).await;
        if request.disposition == QueueDisposition::Parallel {
            self.clear_parallel_tracking(request).await;
        }
        for queued_request in queued {
            let queued_message = format!("queued turn was not run: {message}");
            let mut queued_classification = classification.clone();
            queued_classification.message.clone_from(&queued_message);
            self.events
                .classified_failure(
                    &queued_request,
                    TerminalEvent::Failed {
                        request_id: queued_request.request_id.clone(),
                        message: queued_message,
                        class: class.clone(),
                    },
                    queued_classification,
                )
                .await;
        }
        self.events
            .classified_failure(
                request,
                TerminalEvent::Failed {
                    request_id: request.request_id.clone(),
                    message: message.clone(),
                    class,
                },
                classification,
            )
            .await;
        Err(TurnRunnerError::Failed(message))
    }

    pub(crate) async fn finalize_generation(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        owner_generation: u64,
    ) -> (
        Vec<TurnRequest<P>>,
        Option<(TurnRequest<P>, ActiveTurn<P, Q>)>,
    ) {
        let key = (scope.clone(), thread_id.to_owned());
        let mut active = self.active.lock().await;
        if active
            .get(&key)
            .is_none_or(|turn| turn.generation != owner_generation)
        {
            return (Vec::new(), None);
        }
        let Some(current) = active.get(&key).cloned() else {
            return (Vec::new(), None);
        };
        let mut collected: Vec<TurnRequest<P>> = current
            .queue
            .take(QueueLane::Collect, QueueMode::All)
            .await
            .into_iter()
            .map(TurnQueueItem::into_turn_request)
            .collect();
        restore_queue_context(&mut collected, scope, thread_id, &current.payload);
        let next = current
            .queue
            .take(QueueLane::Steer, QueueMode::OneAtATime)
            .await
            .into_iter()
            .next()
            .map(TurnQueueItem::into_turn_request);
        let next = if next.is_none() {
            current
                .queue
                .take(QueueLane::Followup, QueueMode::OneAtATime)
                .await
                .into_iter()
                .next()
                .map(TurnQueueItem::into_turn_request)
        } else {
            next
        };
        if let Some(mut request) = next {
            restore_queue_context(
                std::slice::from_mut(&mut request),
                scope,
                thread_id,
                &current.payload,
            );
            let reserved = ActiveTurn {
                request_id: request.request_id.clone(),
                cancel: crate::CancellationToken::new(),
                cancellable: true,
                generation: self
                    .next_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                queue: current.queue,
                payload: request.payload.clone(),
            };
            active.insert(key, reserved.clone());
            (collected, Some((request, reserved)))
        } else {
            active.remove(&key);
            (collected, None)
        }
    }

    /// Take queued work from one lane. `OneAtATime` is useful for steering;
    /// `All` preserves FIFO while applying every accepted item at a boundary.
    pub async fn take_queued(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        lane: QueueLane,
        mode: QueueMode,
    ) -> Vec<Q> {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => {
                let mut requests: Vec<TurnRequest<P>> = turn
                    .queue
                    .take(lane, mode)
                    .await
                    .into_iter()
                    .map(TurnQueueItem::into_turn_request)
                    .collect();
                restore_queue_context(&mut requests, scope, thread_id, &turn.payload);
                requests.into_iter().map(Q::from_turn_request).collect()
            }
            None => Vec::new(),
        }
    }

    pub async fn queue_status(&self, scope: &ScopeKey, thread_id: &str) -> QueueStatus {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.status().await,
            None => QueueStatus {
                steers: 0,
                followups: 0,
                collects: 0,
                total: 0,
            },
        }
    }

    /// Non-destructively snapshot the active primary turn's host queue items in
    /// lane order, preserving FIFO order within each lane.
    pub async fn queue_snapshot(&self, scope: &ScopeKey, thread_id: &str) -> Vec<(QueueLane, Q)> {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.snapshot().await,
            None => Vec::new(),
        }
    }

    pub async fn clear_queue(&self, scope: &ScopeKey, thread_id: &str) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.clear().await,
            None => 0,
        }
    }

    async fn clear_generation_queue(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        generation: u64,
    ) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active
            .get(&key)
            .filter(|turn| turn.generation == generation)
        {
            Some(turn) => turn.queue.clear().await,
            None => 0,
        }
    }

    pub async fn remove_queued(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: &str,
    ) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => {
                turn.queue
                    .remove_where(|turn| turn.as_turn_request().request_id == request_id)
                    .await
            }
            None => 0,
        }
    }

    async fn run_guarded(
        &self,
        request: &TurnRequest<P>,
        session: &TurnSession,
        origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<Q>>,
        deadline: TurnDeadline,
    ) -> Result<TurnResult, String> {
        let (progress_tx, mut progress_rx) = mpsc::channel(32);
        let task_cancellation = crate::CancellationToken::new();
        let run = self.host.run(
            request,
            session,
            origin,
            task_cancellation.clone(),
            queue,
            progress_tx,
        );
        let idle = request.idle_timeout.unwrap_or(self.config.idle_timeout);
        let mut last_progress = deadline.started;
        let mut work_in_flight = 0usize;
        let mut progress_open = true;
        tokio::pin!(run);
        // The progress event being handed to the sink. It is a branch of the
        // loop below, never awaited inline, so the turn keeps being polled
        // while it is delivered: a delivery can need the turn to move (directly,
        // or through a lock permit already handed to a waiter inside it), and
        // awaiting it inline would freeze both, every deadline inside the turn
        // included. No further progress is received until it lands, which keeps
        // the bounded channel's backpressure on the host.
        let mut delivering: Option<futures::future::BoxFuture<'_, ()>> = None;
        loop {
            if deadline
                .at
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                task_cancellation.cancel();
                self.host.cancel_children(request).await;
                return self
                    .close_or_timeout(request, "wall-clock deadline", &cancellation)
                    .await;
            }
            let idle_left = idle.saturating_sub(last_progress.elapsed());
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    if !self.config.cancel_grace.is_zero() {
                        // Let the host unwind with its token fired; its result
                        // is discarded either way.
                        let unwound = tokio::time::timeout(self.config.cancel_grace, &mut run).await;
                        tracing::debug!(
                            thread_id = %request.thread_id,
                            request_id = %request.request_id,
                            unwound = unwound.is_ok(),
                            "turn runner: cancelled run left its grace window"
                        );
                    }
                    return Err("turn cancelled".into());
                }
                _ = async { if let Some(deadline) = deadline.at { tokio::time::sleep_until(deadline).await } else { std::future::pending().await } }, if deadline.at.is_some() => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    return self.close_or_timeout(request, "wall-clock deadline", &cancellation).await;
                }
                result = &mut run => {
                    // A host can emit progress and finish in the same poll. Close
                    // the channel before draining so retained child senders
                    // cannot extend completion indefinitely, then finish the
                    // event in flight and deliver every event already accepted
                    // by the bounded queue, in order.
                    progress_rx.close();
                    if let Some(delivery) = delivering.take()
                        && let Some(stopped) = self
                            .deliver_within_turn(request, delivery, &cancellation, &task_cancellation, &deadline)
                            .await
                    {
                        return stopped;
                    }
                    while let Ok(signal) = progress_rx.try_recv() {
                        if let TurnSignal::Progress(event) = signal
                            && let Some(stopped) = self
                                .deliver_within_turn(
                                    request,
                                    self.events.progress(request, *event),
                                    &cancellation,
                                    &task_cancellation,
                                    &deadline,
                                )
                                .await
                        {
                            return stopped;
                        }
                    }
                    return result;
                }
                () = async { if let Some(delivery) = delivering.as_mut() { delivery.await } }, if delivering.is_some() => {
                    delivering = None;
                }
                signal = progress_rx.recv(), if progress_open && delivering.is_none() => match signal {
                    Some(signal) => match signal {
                        TurnSignal::Progress(event) => {
                            last_progress = Instant::now();
                            delivering = Some(self.events.progress(request, *event));
                        }
                        TurnSignal::WorkStarted => work_in_flight = work_in_flight.saturating_add(1),
                        TurnSignal::WorkFinished => {
                            work_in_flight = work_in_flight.saturating_sub(1);
                            if work_in_flight == 0 {
                                last_progress = Instant::now();
                            }
                        }
                    },
                    None => progress_open = false,
                },
                _ = tokio::time::sleep(idle_left), if work_in_flight == 0 && delivering.is_none() => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    return self.close_or_timeout(request, "idle timeout", &cancellation).await;
                }
            }
        }
    }

    /// Delivers one progress event after the turn finished, still honouring
    /// cancellation and the wall-clock deadline. `Some` is the turn's outcome
    /// when either stops it first.
    async fn deliver_within_turn(
        &self,
        request: &TurnRequest<P>,
        delivery: futures::future::BoxFuture<'_, ()>,
        cancellation: &crate::CancellationToken,
        task_cancellation: &crate::CancellationToken,
        deadline: &TurnDeadline,
    ) -> Option<Result<TurnResult, String>> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                task_cancellation.cancel();
                self.host.cancel_children(request).await;
                Some(Err("turn cancelled".into()))
            }
            _ = async { if let Some(deadline) = deadline.at { tokio::time::sleep_until(deadline).await } else { std::future::pending().await } }, if deadline.at.is_some() => {
                task_cancellation.cancel();
                self.host.cancel_children(request).await;
                Some(self.close_or_timeout(request, "wall-clock deadline", cancellation).await)
            }
            () = delivery => None,
        }
    }

    async fn close_or_timeout(
        &self,
        request: &TurnRequest<P>,
        reason: &str,
        cancellation: &crate::CancellationToken,
    ) -> Result<TurnResult, String> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err("turn cancelled".into()),
            result = tokio::time::timeout(
                self.config.close_timeout,
                self.summarizer.grounded_close(request, reason),
            ) => match result {
                Ok(Some(result)) if !cancellation.is_cancelled() => Ok(result),
                _ if cancellation.is_cancelled() => Err("turn cancelled".into()),
                _ => Err(format!("turn {reason}")),
            }
        }
    }

    async fn clear_tracking(&self, request: &TurnRequest<P>, generation: u64) {
        let key = (request.scope.clone(), request.thread_id.clone());
        let mut active = self.active.lock().await;
        if active
            .get(&key)
            .is_some_and(|turn| turn.generation == generation)
        {
            active.remove(&key);
        }
    }

    async fn close_cancellation_window(
        &self,
        request: &TurnRequest<P>,
        generation: Option<u64>,
        cancellation: &crate::CancellationToken,
    ) -> bool {
        if cancellation.is_cancelled() {
            return false;
        }
        if request.host_authored {
            return true;
        }
        if request.disposition == QueueDisposition::Parallel {
            let key = (
                request.scope.clone(),
                request.thread_id.clone(),
                request.request_id.clone(),
            );
            let mut active = self.parallel_active.lock().await;
            let Some(turn) = active.get_mut(&key) else {
                return !cancellation.is_cancelled();
            };
            if turn.cancel.is_cancelled() {
                return false;
            }
            turn.cancellable = false;
            return true;
        }
        let Some(generation) = generation else {
            return true;
        };
        let key = (request.scope.clone(), request.thread_id.clone());
        let mut active = self.active.lock().await;
        let Some(turn) = active
            .get_mut(&key)
            .filter(|turn| turn.generation == generation)
        else {
            return !cancellation.is_cancelled();
        };
        if turn.cancel.is_cancelled() {
            return false;
        }
        turn.cancellable = false;
        true
    }

    async fn clear_parallel_tracking(&self, request: &TurnRequest<P>) {
        let key = (
            request.scope.clone(),
            request.thread_id.clone(),
            request.request_id.clone(),
        );
        self.parallel_active.lock().await.remove(&key);
        self.parallel_queues.lock().await.remove(&key);
    }
}

#[derive(Debug, Clone, Copy)]
enum PreRunStop {
    Cancelled,
    Deadline,
}

#[derive(Debug, Clone, Copy)]
struct TurnDeadline {
    started: Instant,
    at: Option<Instant>,
}

impl TurnDeadline {
    fn new(duration: Option<Duration>) -> Self {
        let started = Instant::now();
        Self {
            started,
            at: duration.map(|duration| started + duration),
        }
    }
}

/// Stable reply identity for idempotent host persistence.
pub fn deterministic_reply_id(scope: &ScopeKey, thread_id: &str, request_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(
        serde_json::to_vec(&(&scope.agent_id, &scope.profile_id, thread_id, request_id))
            .expect("serializing a tuple of strings cannot fail"),
    );
    let encoded = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("reply-{encoded}")
}
