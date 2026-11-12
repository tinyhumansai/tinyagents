use super::*;
use crate::run_queue::{QueueLane, QueueMode, QueuedMessage, RunQueue};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone)]
struct WrappedQueueRequest(TurnRequest);

impl TurnQueueItem<serde_json::Value> for WrappedQueueRequest {
    fn from_turn_request(request: TurnRequest) -> Self {
        Self(request)
    }

    fn as_turn_request(&self) -> &TurnRequest {
        &self.0
    }

    fn into_turn_request(self) -> TurnRequest {
        self.0
    }
}

#[derive(Default)]
struct Host {
    started: Option<Arc<tokio::sync::Notify>>,
    release: Option<Arc<tokio::sync::Notify>>,
    fingerprint_error: bool,
    hold_checkin: bool,
    checkin_started: Option<Arc<tokio::sync::Notify>>,
    fail_run: bool,
    fingerprint_started: Option<Arc<tokio::sync::Notify>>,
    fingerprint_release: Option<Arc<tokio::sync::Notify>>,
    checkins: std::sync::atomic::AtomicUsize,
    runs: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl TurnHost for Host {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: Some("stable".into()),
        })
    }

    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: format!("fork:{}", request.thread_id),
            fingerprint: Some("fork".into()),
        })
    }

    async fn validate_fingerprint(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
    ) -> Result<(), String> {
        if let (Some(started), Some(release)) =
            (&self.fingerprint_started, &self.fingerprint_release)
        {
            started.notify_one();
            release.notified().await;
        }
        if self.fingerprint_error {
            Err("fingerprint mismatch".into())
        } else {
            Ok(())
        }
    }

    fn run_origin(&self, request: &TurnRequest) -> serde_json::Value {
        serde_json::json!({"request": request.request_id})
    }

    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let (Some(started), Some(release)) = (&self.started, &self.release) {
            started.notify_one();
            tokio::select! {
                _ = release.notified() => {},
                _ = cancellation.cancelled() => return Err("cancelled".into()),
            }
        }
        if self.fail_run {
            return Err("host run failed".into());
        }
        Ok(TurnResult {
            reply: "done".into(),
            data: serde_json::Value::Null,
        })
    }

    async fn checkin(&self, request: &TurnRequest) {
        if self.hold_checkin
            && request.request_id == "first"
            && let Some(started) = &self.checkin_started
        {
            started.notify_one();
        }
        if self.hold_checkin
            && request.request_id == "first"
            && let Some(release) = &self.release
        {
            release.notified().await;
        }
        self.checkins
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

struct WrappedHost {
    inner: Host,
    wrapped_queue: Mutex<Option<Arc<RunQueue<WrappedQueueRequest>>>>,
}

#[async_trait::async_trait]
impl TurnHost<serde_json::Value, WrappedQueueRequest> for WrappedHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        <Host as TurnHost>::checkout(&self.inner, request).await
    }

    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        <Host as TurnHost>::checkout_fork(&self.inner, request).await
    }

    async fn validate_fingerprint(
        &self,
        request: &TurnRequest,
        session: &TurnSession,
    ) -> Result<(), String> {
        <Host as TurnHost>::validate_fingerprint(&self.inner, request, session).await
    }

    fn run_origin(&self, request: &TurnRequest) -> serde_json::Value {
        <Host as TurnHost>::run_origin(&self.inner, request)
    }

    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<WrappedQueueRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        *self.wrapped_queue.lock().unwrap() = Some(queue);
        self.inner
            .runs
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let (Some(started), Some(release)) = (&self.inner.started, &self.inner.release) {
            started.notify_one();
            tokio::select! {
                _ = release.notified() => {},
                _ = cancellation.cancelled() => return Err("cancelled".into()),
            }
        }
        Ok(TurnResult {
            reply: "done".into(),
            data: serde_json::Value::Null,
        })
    }

    async fn checkin(&self, request: &TurnRequest) {
        <Host as TurnHost>::checkin(&self.inner, request).await
    }

    async fn cancel_children(&self, request: &TurnRequest) {
        <Host as TurnHost>::cancel_children(&self.inner, request).await
    }
}

#[tokio::test]
async fn completed_execution_is_not_reported_as_cancelled_during_checkin() {
    let checkin_started = Arc::new(tokio::sync::Notify::new());
    let release_checkin = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        hold_checkin: true,
        checkin_started: Some(checkin_started.clone()),
        release: Some(release_checkin.clone()),
        ..Host::default()
    });
    let replies = Arc::new(Replies::default());
    let events = Arc::new(Events::default());
    let runner = Arc::new(runner(
        host,
        Arc::new(Ingress::default()),
        replies.clone(),
        events.clone(),
    ));
    let mut request = request(QueueDisposition::Interrupt, false);
    request.request_id = "first".into();
    let scope = request.scope.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(request).await })
    };

    checkin_started.notified().await;
    assert!(
        runner
            .cancel_requests(&scope, "thread-a", Some("first"))
            .await
            .is_empty()
    );
    release_checkin.notify_one();

    assert!(run.await.unwrap().is_ok());
    assert_eq!(replies.0.lock().unwrap().len(), 1);
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [TerminalEvent::Completed { request_id, .. }] if request_id == "first"
    ));
}

#[tokio::test]
async fn cancelled_parallel_turn_remains_visible_until_cleanup_finishes() {
    let started = Arc::new(tokio::sync::Notify::new());
    let checkin_started = Arc::new(tokio::sync::Notify::new());
    let release_checkin = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release_checkin.clone()),
        hold_checkin: true,
        checkin_started: Some(checkin_started.clone()),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host,
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let mut parallel_request = request(QueueDisposition::Parallel, false);
    parallel_request.request_id = "first".into();
    let scope = parallel_request.scope.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(parallel_request).await })
    };

    started.notified().await;
    assert_eq!(
        runner
            .cancel_requests(&scope, "thread-a", Some("first"))
            .await,
        ["first"]
    );
    checkin_started.notified().await;
    assert_eq!(
        runner
            .live_turns()
            .await
            .iter()
            .map(|turn| (turn.request_id.as_str(), turn.parallel))
            .collect::<Vec<_>>(),
        [("first", true)]
    );
    assert!(
        runner
            .cancel_requests(&scope, "thread-a", Some("first"))
            .await
            .is_empty()
    );

    release_checkin.notify_one();
    assert!(matches!(
        run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(runner.live_turns().await.is_empty());
}

#[tokio::test]
async fn primary_cancel_result_is_reported_once_until_cleanup_finishes() {
    let started = Arc::new(tokio::sync::Notify::new());
    let checkin_started = Arc::new(tokio::sync::Notify::new());
    let release_checkin = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release_checkin.clone()),
        hold_checkin: true,
        checkin_started: Some(checkin_started.clone()),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host,
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let mut primary_request = request(QueueDisposition::Interrupt, false);
    primary_request.request_id = "first".into();
    let scope = primary_request.scope.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };

    started.notified().await;
    assert_eq!(
        runner
            .cancel_requests(&scope, "thread-a", Some("first"))
            .await,
        ["first"]
    );
    checkin_started.notified().await;
    assert!(
        runner
            .cancel_requests(&scope, "thread-a", Some("first"))
            .await
            .is_empty()
    );
    assert_eq!(
        runner
            .live_turns()
            .await
            .iter()
            .map(|turn| turn.request_id.as_str())
            .collect::<Vec<_>>(),
        ["first"]
    );

    release_checkin.notify_one();
    assert!(matches!(
        run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(runner.live_turns().await.is_empty());
}

#[tokio::test]
async fn custom_queue_item_shares_runner_queue_and_preserves_complete_request() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(WrappedHost {
        inner: Host {
            started: Some(started.clone()),
            release: Some(release),
            ..Host::default()
        },
        wrapped_queue: Mutex::default(),
    });
    let runner = Arc::new(TurnRunner::<
        _,
        _,
        _,
        _,
        _,
        _,
        serde_json::Value,
        WrappedQueueRequest,
    >::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));

    let mut primary = request(QueueDisposition::Interrupt, false);
    primary.request_id = "primary".into();
    primary.scope.profile_id = Some("profile-x".into());
    let scope = primary.scope.clone();
    let primary_run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary).await })
    };
    started.notified().await;

    let mut runner_queued = request(QueueDisposition::Followup, false);
    runner_queued.scope = scope.clone();
    runner_queued.thread_id = "thread-a".into();
    runner_queued.request_id = "runner-queued-id".into();
    runner_queued.text = "runner text".into();
    runner_queued.payload = serde_json::json!({"client":"runner","attachment":[1,2]});
    runner_queued.deadline = Some(Duration::from_secs(31));
    runner_queued.idle_timeout = Some(Duration::from_secs(7));
    assert!(matches!(
        runner.run(runner_queued.clone()).await,
        Err(TurnRunnerError::Queued)
    ));

    let shared_queue = host.wrapped_queue.lock().unwrap().clone().unwrap();
    let runner_item = shared_queue
        .snapshot()
        .await
        .into_iter()
        .find(|(_, item)| item.as_turn_request().request_id == "runner-queued-id")
        .expect("runner-enqueued item is in the host's queue");
    assert_eq!(runner_item.0, QueueLane::Followup);
    let stored_runner_request = runner_item.1.as_turn_request();
    assert_eq!(stored_runner_request.scope, runner_queued.scope);
    assert_eq!(stored_runner_request.thread_id, runner_queued.thread_id);
    assert_eq!(stored_runner_request.request_id, runner_queued.request_id);
    assert_eq!(stored_runner_request.text, runner_queued.text);
    assert_eq!(stored_runner_request.payload, runner_queued.payload);
    assert_eq!(
        stored_runner_request.host_authored,
        runner_queued.host_authored
    );
    assert_eq!(stored_runner_request.disposition, runner_queued.disposition);
    assert_eq!(stored_runner_request.deadline, runner_queued.deadline);
    assert_eq!(
        stored_runner_request.idle_timeout,
        runner_queued.idle_timeout
    );
    let round_tripped = runner_item.1.clone().into_turn_request();
    assert_eq!(round_tripped.scope, runner_queued.scope);
    assert_eq!(round_tripped.thread_id, runner_queued.thread_id);
    assert_eq!(round_tripped.request_id, runner_queued.request_id);
    assert_eq!(round_tripped.text, runner_queued.text);
    assert_eq!(round_tripped.payload, runner_queued.payload);
    assert_eq!(round_tripped.host_authored, runner_queued.host_authored);
    assert_eq!(round_tripped.disposition, runner_queued.disposition);
    assert_eq!(round_tripped.deadline, runner_queued.deadline);
    assert_eq!(round_tripped.idle_timeout, runner_queued.idle_timeout);

    let mut host_queued = request(QueueDisposition::Steer, false);
    host_queued.scope = scope.clone();
    host_queued.request_id = "host-queued-id".into();
    host_queued.text = "host text".into();
    host_queued.payload = serde_json::json!({"client":"host","locale":"en"});
    shared_queue
        .push(
            QueueLane::Steer,
            WrappedQueueRequest::from_turn_request(host_queued.clone()),
        )
        .await;
    let runner_snapshot = runner.queue_snapshot(&scope, "thread-a").await;
    let host_item = runner_snapshot
        .iter()
        .find(|(_, item)| item.as_turn_request().request_id == "host-queued-id")
        .expect("host-enqueued item is visible in runner's snapshot");
    assert_eq!(host_item.0, QueueLane::Steer);
    assert_eq!(host_item.1.as_turn_request().scope, host_queued.scope);
    assert_eq!(
        host_item.1.as_turn_request().request_id,
        host_queued.request_id
    );
    assert_eq!(host_item.1.as_turn_request().text, host_queued.text);
    assert_eq!(host_item.1.as_turn_request().payload, host_queued.payload);

    assert_eq!(
        runner
            .cancel_requests(&scope, "thread-a", Some("primary"))
            .await,
        ["primary"]
    );
    assert!(matches!(
        primary_run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
}

#[derive(Default)]
struct Ingress(Mutex<Vec<IngressStage>>);

#[async_trait::async_trait]
impl IngressPolicy for Ingress {
    async fn apply(
        &self,
        stage: IngressStage,
        _request: &mut TurnRequest,
    ) -> Result<IngressDecision, String> {
        self.0.lock().unwrap().push(stage);
        Ok(IngressDecision::Continue)
    }
}

#[derive(Default)]
struct Replies(Mutex<Vec<String>>);

#[async_trait::async_trait]
impl ReplyStore for Replies {
    async fn persist(
        &self,
        _request: &TurnRequest,
        reply_id: &str,
        _result: &TurnResult,
    ) -> Result<(), String> {
        self.0.lock().unwrap().push(reply_id.to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct Events {
    terminal: Mutex<Vec<TerminalEvent>>,
    collected: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl EventSink for Events {
    async fn progress(&self, _request: &TurnRequest, _event: crate::progress::AgentProgress) {}
    async fn terminal(&self, _request: &TurnRequest, event: TerminalEvent) {
        self.terminal.lock().unwrap().push(event);
    }
    async fn collected(&self, _request: &TurnRequest, items: Vec<TurnRequest>) {
        self.collected
            .lock()
            .unwrap()
            .push(items.into_iter().map(|item| item.request_id).collect());
    }
}

struct Failures;
impl FailureClassifier for Failures {
    fn classify(&self, error: &str) -> (String, String) {
        ("internal".into(), error.into())
    }
}

struct Summary;
#[async_trait::async_trait]
impl Summarizer for Summary {
    async fn grounded_close(&self, _request: &TurnRequest, _reason: &str) -> Option<TurnResult> {
        None
    }
}

fn runner(
    host: Arc<Host>,
    ingress: Arc<Ingress>,
    replies: Arc<Replies>,
    events: Arc<Events>,
) -> TurnRunner<Host, Ingress, Replies, Events, Failures, Summary> {
    TurnRunner::new(
        host,
        ingress,
        replies,
        events,
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    )
}

fn request(disposition: QueueDisposition, host_authored: bool) -> TurnRequest {
    TurnRequest {
        scope: ScopeKey {
            agent_id: "agent-a".into(),
            profile_id: None,
        },
        thread_id: "thread-a".into(),
        request_id: "request-a".into(),
        text: "hello".into(),
        payload: serde_json::json!({"attachment": "file-token"}),
        host_authored,
        disposition,
        deadline: None,
        idle_timeout: Some(std::time::Duration::from_secs(5)),
    }
}

#[tokio::test]
async fn taking_rebuilt_queue_requests_restores_the_active_scope() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    let scope = ScopeKey {
        agent_id: "agent-a".into(),
        profile_id: Some("profile-a".into()),
    };
    let queue = Arc::new(RunQueue::new());
    let mut recovered = request(QueueDisposition::Steer, false);
    recovered.scope = ScopeKey {
        agent_id: String::new(),
        profile_id: None,
    };
    queue.push_now(QueueLane::Steer, recovered);
    runner.active.lock().await.insert(
        (scope.clone(), "thread-a".into()),
        ActiveTurn {
            request_id: "active".into(),
            cancel: crate::CancellationToken::new(),
            cancellable: true,
            generation: 7,
            queue,
            payload: serde_json::Value::Null,
        },
    );

    let recovered = runner
        .take_queued(&scope, "thread-a", QueueLane::Steer, QueueMode::OneAtATime)
        .await;

    assert_eq!(recovered[0].scope, scope);
}

#[tokio::test]
async fn taking_rebuilt_queue_requests_restores_the_owner_thread_id() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    let scope = ScopeKey {
        agent_id: "agent-a".into(),
        profile_id: Some("profile-a".into()),
    };
    let queue = Arc::new(RunQueue::new());
    let mut recovered = <TurnRequest as QueuedMessage>::requeued(
        "recovered".into(),
        "follow up".into(),
        "subagent-task-label",
        42,
    );
    recovered.disposition = QueueDisposition::Steer;
    queue.push_now(QueueLane::Steer, recovered);
    runner.active.lock().await.insert(
        (scope.clone(), "thread-a".into()),
        ActiveTurn {
            request_id: "active".into(),
            cancel: crate::CancellationToken::new(),
            cancellable: true,
            generation: 7,
            queue,
            payload: serde_json::json!({
                "access": "full",
                "locale": "fr",
                "attachments": ["prior-turn-file"],
            }),
        },
    );

    let recovered = runner
        .take_queued(&scope, "thread-a", QueueLane::Steer, QueueMode::OneAtATime)
        .await;

    assert_eq!(recovered[0].thread_id, "thread-a");
    assert_eq!(recovered[0].payload["access"], "full");
    assert_eq!(recovered[0].payload["locale"], "fr");
    assert_eq!(recovered[0].payload["thread_id"], "thread-a");
    assert_eq!(recovered[0].payload["queued_at_ms"], 42);
    assert!(recovered[0].payload["attachments"].is_null());
}

#[tokio::test]
async fn finalizing_a_rebuilt_queue_request_restores_the_active_scope() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    let scope = ScopeKey {
        agent_id: "agent-a".into(),
        profile_id: Some("profile-a".into()),
    };
    let queue = Arc::new(RunQueue::new());
    let mut recovered = <TurnRequest as QueuedMessage>::requeued(
        "recovered".into(),
        "follow up".into(),
        "subagent-task-label",
        42,
    );
    recovered.disposition = QueueDisposition::Steer;
    queue.push_now(QueueLane::Steer, recovered);
    runner.active.lock().await.insert(
        (scope.clone(), "thread-a".into()),
        ActiveTurn {
            request_id: "active".into(),
            cancel: crate::CancellationToken::new(),
            cancellable: true,
            generation: 7,
            queue,
            payload: serde_json::json!({"access": "full"}),
        },
    );

    let (_, next) = runner.finalize_generation(&scope, "thread-a", 7).await;

    let (next, _) = next.expect("queued steer is reserved");
    assert_eq!(next.scope, scope);
    assert_eq!(next.thread_id, "thread-a");
    assert_eq!(next.payload["access"], "full");
    assert_eq!(next.payload["queued_at_ms"], 42);
}

#[tokio::test]
async fn finalizing_rebuilt_collect_requests_restores_the_active_scope() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    let scope = ScopeKey {
        agent_id: "agent-a".into(),
        profile_id: Some("profile-a".into()),
    };
    let queue = Arc::new(RunQueue::new());
    let mut recovered = request(QueueDisposition::Collect, false);
    recovered.scope = ScopeKey {
        agent_id: String::new(),
        profile_id: None,
    };
    queue.push_now(QueueLane::Collect, recovered);
    runner.active.lock().await.insert(
        (scope.clone(), "thread-a".into()),
        ActiveTurn {
            request_id: "active".into(),
            cancel: crate::CancellationToken::new(),
            cancellable: true,
            generation: 7,
            queue,
            payload: serde_json::Value::Null,
        },
    );

    let (collected, _) = runner.finalize_generation(&scope, "thread-a", 7).await;

    assert_eq!(collected[0].scope, scope);
}

#[tokio::test]
async fn retiring_rebuilt_queue_requests_restores_the_active_scope() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    let scope = ScopeKey {
        agent_id: "agent-a".into(),
        profile_id: Some("profile-a".into()),
    };
    let queue = Arc::new(RunQueue::new());
    for lane in [QueueLane::Steer, QueueLane::Followup, QueueLane::Collect] {
        let mut recovered = request(QueueDisposition::Steer, false);
        recovered.request_id = format!("{lane:?}");
        recovered.scope = ScopeKey {
            agent_id: String::new(),
            profile_id: None,
        };
        queue.push_now(lane, recovered);
    }
    runner.active.lock().await.insert(
        (scope.clone(), "thread-a".into()),
        ActiveTurn {
            request_id: "active".into(),
            cancel: crate::CancellationToken::new(),
            cancellable: true,
            generation: 7,
            queue,
            payload: serde_json::Value::Null,
        },
    );

    let mut owner = request(QueueDisposition::Interrupt, false);
    owner.scope = scope.clone();
    let retired = runner.retire_generation(&owner, Some(7)).await;

    assert_eq!(retired.len(), 3);
    assert!(
        retired
            .iter()
            .all(|request| { request.scope == scope && request.thread_id == "thread-a" })
    );
}

#[tokio::test]
async fn ingress_stages_run_in_order_before_queue_acceptance() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        ..Host::default()
    });
    let ingress = Arc::new(Ingress::default());
    let runner = Arc::new(runner(
        host.clone(),
        ingress.clone(),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));

    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    started.notified().await;
    for (mode, id) in [
        (QueueDisposition::Steer, "steered"),
        (QueueDisposition::Followup, "followup"),
        (QueueDisposition::Collect, "collect"),
    ] {
        let mut incoming = request(mode, false);
        incoming.request_id = id.into();
        assert!(matches!(
            runner.run(incoming).await,
            Err(TurnRunnerError::Queued)
        ));
    }
    let scope = request(QueueDisposition::Interrupt, false).scope;
    let status = runner.queue_status(&scope, "thread-a").await;
    assert_eq!(
        (status.steers, status.followups, status.collects),
        (1, 1, 1)
    );
    let steers = runner
        .take_queued(&scope, "thread-a", QueueLane::Steer, QueueMode::OneAtATime)
        .await;
    assert_eq!(steers[0].request_id, "steered");
    assert_eq!(
        steers[0].payload,
        serde_json::json!({"attachment": "file-token"})
    );
    assert_eq!(
        runner
            .take_queued(&scope, "thread-a", QueueLane::Followup, QueueMode::All)
            .await[0]
            .request_id,
        "followup"
    );
    assert_eq!(
        runner
            .take_queued(&scope, "thread-a", QueueLane::Collect, QueueMode::All)
            .await[0]
            .request_id,
        "collect"
    );
    assert_eq!(runner.clear_queue(&scope, "thread-a").await, 0);
    release.notify_one();
    primary.await.unwrap().unwrap();
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    let ingress = ingress.0.lock().unwrap();
    assert_eq!(ingress.len(), 4 * 4);
    assert_eq!(
        ingress[12..],
        [
            IngressStage::StageAttachments,
            IngressStage::Guardrail,
            IngressStage::ApprovalInterception,
            IngressStage::BeforeSubmit,
        ]
    );
}

#[tokio::test]
async fn accepted_followup_is_dispatched_as_a_fresh_turn_after_primary_finishes() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        ..Host::default()
    });
    let events = Arc::new(Events::default());
    let ingress = Arc::new(Ingress::default());
    let runner = Arc::new(runner(
        host.clone(),
        ingress.clone(),
        Arc::new(Replies::default()),
        events.clone(),
    ));

    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    started.notified().await;
    let mut followup = request(QueueDisposition::Followup, false);
    followup.request_id = "followup-fresh".into();
    assert!(matches!(
        runner.run(followup).await,
        Err(TurnRunnerError::Queued)
    ));

    release.notify_one();
    started.notified().await;
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    release.notify_one();
    primary.await.unwrap().unwrap();
    assert_eq!(ingress.0.lock().unwrap().len(), 8);

    assert_eq!(events.terminal.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn natural_finish_dispatches_steer_before_followup() {
    let host = Arc::new(OrderedHost::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    host.started.notified().await;
    for (request_id, disposition) in [
        ("deferred", QueueDisposition::Followup),
        ("urgent", QueueDisposition::Steer),
    ] {
        let mut queued = request(disposition, false);
        queued.request_id = request_id.into();
        assert!(matches!(
            runner.run(queued).await,
            Err(TurnRunnerError::Queued)
        ));
    }
    host.release.notify_one();
    primary.await.unwrap().unwrap();

    assert_eq!(
        host.started_ids.lock().unwrap().as_slice(),
        ["request-a", "urgent", "deferred"]
    );
}

#[derive(Default)]
struct OrderedHost {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    started_ids: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl TurnHost for OrderedHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.request_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        _cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        self.started_ids
            .lock()
            .unwrap()
            .push(request.request_id.clone());
        if request.request_id == "request-a" {
            self.started.notify_one();
            self.release.notified().await;
        }
        Ok(TurnResult {
            reply: "ok".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test]
async fn failed_generation_is_retired_before_a_slow_terminal_callback() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        fail_run: true,
        ..Host::default()
    });
    let events = Arc::new(BlockingFailureEvents::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let first = {
        let runner = runner.clone();
        tokio::spawn(async move {
            let mut first = request(QueueDisposition::Interrupt, false);
            first.request_id = "failing-owner".into();
            runner.run(first).await
        })
    };
    started.notified().await;
    let mut queued = request(QueueDisposition::Followup, false);
    queued.request_id = "accepted-before-failure".into();
    assert!(matches!(
        runner.run(queued).await,
        Err(TurnRunnerError::Queued)
    ));
    release.notify_one();
    events.terminal_started.notified().await;

    let mut late = request(QueueDisposition::Steer, false);
    late.request_id = "late-steer".into();
    let late_result = runner.run(late).await;

    events.release_terminal.notify_one();
    assert!(matches!(
        first.await.unwrap(),
        Err(TurnRunnerError::Failed(_))
    ));
    assert!(matches!(late_result, Err(TurnRunnerError::Failed(_))));
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(
        events
            .terminal
            .lock()
            .unwrap()
            .iter()
            .any(|(id, event)| id == "accepted-before-failure"
                && matches!(event, TerminalEvent::Failed { .. }))
    );
    let classified = events.classified.lock().unwrap();
    assert_eq!(
        classified
            .iter()
            .find(|(id, _)| id == "accepted-before-failure")
            .map(|(_, classification)| classification.message.as_str()),
        Some("queued turn was not run: host run failed")
    );
}

#[tokio::test]
async fn cancelled_generation_is_retired_before_a_slow_terminal_callback() {
    let host = Arc::new(CancelOwnerHost::default());
    let events = Arc::new(BlockingCancelEvents::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move {
            let mut owner = request(QueueDisposition::Interrupt, false);
            owner.request_id = "cancel-owner".into();
            runner.run(owner).await
        })
    };
    host.owner_started.notified().await;
    let scope = request(QueueDisposition::Interrupt, false).scope;
    assert!(
        runner
            .cancel(&scope, "thread-a", Some("cancel-owner"))
            .await
    );
    events.terminal_started.notified().await;

    let mut late = request(QueueDisposition::Steer, false);
    late.request_id = "after-cancel".into();
    assert!(runner.run(late).await.is_ok());
    events.release_terminal.notify_one();
    assert!(matches!(
        run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
}

#[derive(Default)]
struct CancelOwnerHost {
    owner_started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl TurnHost for CancelOwnerHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        if request.request_id == "cancel-owner" {
            self.owner_started.notify_one();
            cancellation.cancelled().await;
            return Err("cancelled".into());
        }
        Ok(TurnResult {
            reply: "ok".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[derive(Default)]
struct BlockingCancelEvents {
    terminal_started: tokio::sync::Notify,
    release_terminal: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl EventSink for BlockingCancelEvents {
    async fn progress(&self, _request: &TurnRequest, _event: crate::progress::AgentProgress) {}
    async fn terminal(&self, request: &TurnRequest, event: TerminalEvent) {
        if request.request_id == "cancel-owner" && matches!(event, TerminalEvent::Cancelled { .. })
        {
            self.terminal_started.notify_one();
            self.release_terminal.notified().await;
        }
    }
}

#[derive(Default)]
struct BlockingFailureEvents {
    terminal_started: tokio::sync::Notify,
    release_terminal: tokio::sync::Notify,
    terminal: Mutex<Vec<(String, TerminalEvent)>>,
    classified: Mutex<Vec<(String, FailureClassification)>>,
}

#[async_trait::async_trait]
impl EventSink for BlockingFailureEvents {
    async fn progress(&self, _request: &TurnRequest, _event: crate::progress::AgentProgress) {}
    async fn terminal(&self, request: &TurnRequest, event: TerminalEvent) {
        self.terminal
            .lock()
            .unwrap()
            .push((request.request_id.clone(), event));
        if request.request_id == "failing-owner" {
            self.terminal_started.notify_one();
            self.release_terminal.notified().await;
        }
    }
    async fn classified_failure(
        &self,
        request: &TurnRequest,
        event: TerminalEvent,
        classification: FailureClassification,
    ) {
        self.classified
            .lock()
            .unwrap()
            .push((request.request_id.clone(), classification));
        self.terminal(request, event).await;
    }
}

#[tokio::test]
async fn host_authored_turn_persists_reply_without_an_inflight_slot() {
    let events = Arc::new(Events::default());
    let replies = Arc::new(Replies::default());
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        replies.clone(),
        events.clone(),
    );
    let request = request(QueueDisposition::Interrupt, true);
    let scope = request.scope.clone();
    let thread = request.thread_id.clone();

    let result = runner.run(request.clone()).await.unwrap();

    assert_eq!(result.reply, "done");
    assert!(!runner.is_in_flight(&scope, &thread).await);
    assert_eq!(replies.0.lock().unwrap().len(), 1);
    assert_eq!(events.terminal.lock().unwrap().len(), 1);
    assert_eq!(
        replies.0.lock().unwrap()[0],
        deterministic_reply_id(&scope, &thread, &request.request_id)
    );
}

#[tokio::test]
async fn host_authored_turn_never_enters_inflight_while_running() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let runner = Arc::new(runner(
        Arc::new(Host {
            started: Some(started.clone()),
            release: Some(release.clone()),
            ..Host::default()
        }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let request = request(QueueDisposition::Interrupt, true);
    let scope = request.scope.clone();
    let thread = request.thread_id.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(request).await })
    };

    started.notified().await;
    assert!(!runner.is_in_flight(&scope, &thread).await);
    release.notify_one();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn stale_or_cross_agent_cancel_cannot_cancel_the_current_turn() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let runner = Arc::new(runner(
        Arc::new(Host {
            started: Some(started.clone()),
            release: Some(release.clone()),
            ..Host::default()
        }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let mut active_request = request(QueueDisposition::Interrupt, false);
    active_request.request_id = "current-request".into();
    let scope = active_request.scope.clone();
    let thread = active_request.thread_id.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(active_request).await })
    };

    started.notified().await;
    assert!(!runner.cancel(&scope, &thread, Some("stale-request")).await);
    let another_agent = ScopeKey {
        agent_id: "agent-b".into(),
        profile_id: None,
    };
    assert!(
        !runner
            .cancel(&another_agent, &thread, Some("current-request"))
            .await
    );
    for mode in [
        QueueDisposition::Steer,
        QueueDisposition::Followup,
        QueueDisposition::Collect,
    ] {
        assert!(matches!(
            runner.run(request(mode, false)).await,
            Err(TurnRunnerError::Queued)
        ));
    }
    assert_eq!(runner.queue_status(&scope, &thread).await.total, 3);
    assert!(
        runner
            .cancel(&scope, &thread, Some("current-request"))
            .await
    );
    assert_eq!(runner.queue_status(&scope, &thread).await.total, 0);
    assert!(matches!(
        run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
}

#[tokio::test]
async fn unscoped_cancel_stops_primary_and_parallel_turns_together() {
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(Arc::new(tokio::sync::Notify::new())),
        release: Some(release.clone()),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let primary_request = request(QueueDisposition::Interrupt, false);
    let parallel_request = request(QueueDisposition::Parallel, false);
    let scope = primary_request.scope.clone();
    let thread = primary_request.thread_id.clone();
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };
    let parallel = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(parallel_request).await })
    };
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while host.runs.load(std::sync::atomic::Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(runner.cancel(&scope, &thread, None).await);
    assert!(matches!(
        primary.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(matches!(
        parallel.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
}

#[tokio::test]
async fn cancelling_a_followup_discards_later_followups_still_in_the_queue() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let primary_request = request(QueueDisposition::Interrupt, false);
    let scope = primary_request.scope.clone();
    let thread = primary_request.thread_id.clone();
    let outer = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };
    started.notified().await;
    for id in ["first-followup", "second-followup"] {
        let mut followup = request(QueueDisposition::Followup, false);
        followup.request_id = id.into();
        assert!(matches!(
            runner.run(followup).await,
            Err(TurnRunnerError::Queued)
        ));
    }
    release.notify_one();
    started.notified().await;
    assert!(runner.cancel(&scope, &thread, Some("first-followup")).await);
    assert!(outer.await.unwrap().is_ok());
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(runner.queue_status(&scope, &thread).await.total, 0);
}

#[tokio::test]
async fn parallel_turn_gets_a_private_queue_and_cannot_steal_primary_work() {
    let host = Arc::new(QueueOwnershipHost::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut primary_request = request(QueueDisposition::Interrupt, false);
    primary_request.request_id = "primary-owner".into();
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };
    host.primary_started.notified().await;

    let mut collect = request(QueueDisposition::Collect, false);
    collect.request_id = "primary-collect".into();
    assert!(matches!(
        runner.run(collect).await,
        Err(TurnRunnerError::Queued)
    ));
    assert_eq!(
        runner
            .queue_status(
                &request(QueueDisposition::Interrupt, false).scope,
                "thread-a"
            )
            .await
            .collects,
        1
    );

    let mut parallel_request = request(QueueDisposition::Parallel, false);
    parallel_request.request_id = "parallel-private".into();
    runner.run(parallel_request).await.unwrap();

    let (primary_queue, parallel_queue) = {
        let queues = host.queues.lock().unwrap();
        (
            queues.get("primary-owner").unwrap().clone(),
            queues.get("parallel-private").unwrap().clone(),
        )
    };
    assert_eq!(primary_queue.status().await.collects, 1);
    assert_eq!(parallel_queue.status().await.total, 0);

    host.release_primary.notify_one();
    primary.await.unwrap().unwrap();
}

#[tokio::test]
async fn host_authored_turn_gets_private_queue_and_preserves_primary_collects() {
    let host = Arc::new(QueueOwnershipHost::default());
    let events = Arc::new(Events::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut primary_request = request(QueueDisposition::Interrupt, false);
    primary_request.request_id = "primary-owner".into();
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };
    host.primary_started.notified().await;

    let mut collect = request(QueueDisposition::Collect, false);
    collect.request_id = "primary-collect".into();
    assert!(matches!(
        runner.run(collect).await,
        Err(TurnRunnerError::Queued)
    ));
    assert_eq!(
        runner
            .queue_status(
                &request(QueueDisposition::Interrupt, false).scope,
                "thread-a"
            )
            .await
            .collects,
        1
    );
    let mut background = request(QueueDisposition::Interrupt, true);
    background.request_id = "host-background".into();
    let background_task = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(background).await })
    };
    host.release_primary.notify_one();
    primary.await.unwrap().unwrap();
    background_task.await.unwrap().unwrap();
    assert_eq!(
        events.collected.lock().unwrap().as_slice(),
        [vec!["primary-collect".to_owned()]]
    );
}

#[tokio::test]
async fn stale_completion_cannot_consume_a_replacement_generations_queue() {
    let host = Arc::new(ReplacementHost::default());
    let replies = Arc::new(BlockingOldReplyStore::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        replies.clone(),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut old = request(QueueDisposition::Interrupt, false);
    old.request_id = "old-generation".into();
    let old_task = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(old).await })
    };
    replies.persist_started.notified().await;

    let mut replacement = request(QueueDisposition::Interrupt, false);
    replacement.request_id = "replacement-generation".into();
    let new_task = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(replacement).await })
    };
    host.replacement_started.notified().await;
    for (id, disposition) in [
        ("replacement-collect", QueueDisposition::Collect),
        ("replacement-followup", QueueDisposition::Followup),
        ("replacement-steer", QueueDisposition::Steer),
    ] {
        let mut queued = request(disposition, false);
        queued.request_id = id.into();
        assert!(matches!(
            runner.run(queued).await,
            Err(TurnRunnerError::Queued)
        ));
    }

    replies.release_old.notify_one();
    old_task.await.unwrap().unwrap();
    let status = runner
        .queue_status(
            &request(QueueDisposition::Interrupt, false).scope,
            "thread-a",
        )
        .await;
    assert_eq!((status.collects, status.followups), (1, 1));

    host.release_replacement.notify_one();
    new_task.await.unwrap().unwrap();
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 4);
}

#[tokio::test]
async fn collect_arriving_during_terminal_delivery_runs_as_a_fresh_turn() {
    let host = Arc::new(Host::default());
    let events = Arc::new(BlockingTerminalEvents::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut owner = request(QueueDisposition::Interrupt, false);
    owner.request_id = "callback-owner".into();
    let owner_task = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(owner).await })
    };
    events.terminal_started.notified().await;

    let mut late_collect = request(QueueDisposition::Collect, false);
    late_collect.request_id = "late-collect".into();
    runner.run(late_collect).await.unwrap();
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 2);

    events.release_terminal.notify_one();
    owner_task.await.unwrap().unwrap();
}

#[derive(Default)]
struct BlockingTerminalEvents {
    terminal_started: tokio::sync::Notify,
    release_terminal: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl EventSink for BlockingTerminalEvents {
    async fn progress(&self, _request: &TurnRequest, _event: crate::progress::AgentProgress) {}
    async fn terminal(&self, request: &TurnRequest, _event: TerminalEvent) {
        if request.request_id == "callback-owner" {
            self.terminal_started.notify_one();
            self.release_terminal.notified().await;
        }
    }
}

#[derive(Default)]
struct BlockingOldReplyStore {
    persist_started: tokio::sync::Notify,
    release_old: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl ReplyStore for BlockingOldReplyStore {
    async fn persist(
        &self,
        request: &TurnRequest,
        _reply_id: &str,
        _result: &TurnResult,
    ) -> Result<(), String> {
        if request.request_id == "old-generation" {
            self.persist_started.notify_one();
            self.release_old.notified().await;
        }
        Ok(())
    }
}

#[derive(Default)]
struct ReplacementHost {
    replacement_started: tokio::sync::Notify,
    release_replacement: tokio::sync::Notify,
    runs: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl TurnHost for ReplacementHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.request_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if request.request_id == "replacement-generation" {
            self.replacement_started.notify_one();
            tokio::select! {
                _ = self.release_replacement.notified() => {},
                _ = cancellation.cancelled() => return Err("cancelled".into()),
            }
        }
        Ok(TurnResult {
            reply: "ok".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test]
async fn simultaneous_idle_steer_and_followup_admission_has_one_owner() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        ..Host::default()
    });
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(SubmitBarrierIngress(tokio::sync::Barrier::new(2))),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut steer = request(QueueDisposition::Steer, false);
    steer.request_id = "simultaneous-steer".into();
    let mut followup = request(QueueDisposition::Followup, false);
    followup.request_id = "simultaneous-followup".into();
    let a = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(steer).await })
    };
    let b = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(followup).await })
    };
    started.notified().await;
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        runner
            .queue_status(
                &request(QueueDisposition::Interrupt, false).scope,
                "thread-a"
            )
            .await
            .total,
        1
    );
    let scope = request(QueueDisposition::Interrupt, false).scope;
    let active_id = runner
        .in_flight_request_id(&scope, "thread-a")
        .await
        .unwrap();
    assert!(runner.cancel(&scope, "thread-a", Some(&active_id)).await);
    let results = [a.await.unwrap(), b.await.unwrap()];
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(TurnRunnerError::Cancelled)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(TurnRunnerError::Queued)))
            .count(),
        1
    );
}

struct SubmitBarrierIngress(tokio::sync::Barrier);

#[async_trait::async_trait]
impl IngressPolicy for SubmitBarrierIngress {
    async fn apply(
        &self,
        stage: IngressStage,
        _request: &mut TurnRequest,
    ) -> Result<IngressDecision, String> {
        if stage == IngressStage::BeforeSubmit {
            self.0.wait().await;
        }
        Ok(IngressDecision::Continue)
    }
}

#[derive(Default)]
struct QueueOwnershipHost {
    primary_started: tokio::sync::Notify,
    release_primary: tokio::sync::Notify,
    queues: Mutex<std::collections::HashMap<String, Arc<RunQueue<TurnRequest>>>>,
}

#[async_trait::async_trait]
impl TurnHost for QueueOwnershipHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.request_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        _cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        self.queues
            .lock()
            .unwrap()
            .insert(request.request_id.clone(), queue);
        if request.request_id == "primary-owner" {
            self.primary_started.notify_one();
            self.release_primary.notified().await;
        }
        Ok(TurnResult {
            reply: "ok".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test]
async fn closed_progress_stream_does_not_starve_the_completed_run() {
    let runner = runner(
        Arc::new(Host::default()),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        runner.run(request(QueueDisposition::Interrupt, false)),
    )
    .await
    .expect("closed progress stream must not starve the run result")
    .unwrap();
}

#[derive(Default)]
struct OrderedEvents(Mutex<Vec<&'static str>>);

#[async_trait::async_trait]
impl EventSink for OrderedEvents {
    async fn progress(&self, _request: &TurnRequest, event: crate::progress::AgentProgress) {
        let label = match event {
            crate::progress::AgentProgress::TurnStarted => "progress",
            _ => "other-progress",
        };
        self.0.lock().unwrap().push(label);
    }

    async fn terminal(&self, _request: &TurnRequest, _event: TerminalEvent) {
        self.0.lock().unwrap().push("terminal");
    }
}

struct FastProgressHost;

#[async_trait::async_trait]
impl TurnHost for FastProgressHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        _cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        progress
            .send(TurnSignal::Progress(Box::new(
                crate::progress::AgentProgress::TurnStarted,
            )))
            .await
            .unwrap();
        Ok(TurnResult {
            reply: "fast".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test]
async fn accepted_progress_is_delivered_before_terminal_when_run_finishes_immediately() {
    let events = Arc::new(OrderedEvents::default());
    let runner = TurnRunner::new(
        Arc::new(FastProgressHost),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    );

    runner
        .run(request(QueueDisposition::Interrupt, false))
        .await
        .unwrap();

    assert_eq!(*events.0.lock().unwrap(), ["progress", "terminal"]);
}

#[tokio::test]
async fn a_superseded_turn_waiting_for_the_session_lock_never_starts() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release.clone()),
        hold_checkin: true,
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let mut first = request(QueueDisposition::Interrupt, false);
    first.request_id = "first".into();
    let first_run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(first).await })
    };
    started.notified().await;

    let mut waiting = request(QueueDisposition::Interrupt, false);
    waiting.request_id = "waiting".into();
    let waiting_run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(waiting).await })
    };
    let scope = request(QueueDisposition::Interrupt, false).scope;
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while runner
            .in_flight_request_id(&scope, "thread-a")
            .await
            .as_deref()
            != Some("waiting")
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("waiting request must own the cancellation generation");

    let mut replacement = request(QueueDisposition::Interrupt, false);
    replacement.request_id = "replacement".into();
    let replacement_run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(replacement).await })
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while runner
            .in_flight_request_id(&scope, "thread-a")
            .await
            .as_deref()
            != Some("replacement")
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("replacement must supersede the waiting request");

    release.notify_one();
    started.notified().await;
    release.notify_one();
    assert!(matches!(
        first_run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(matches!(
        waiting_run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    replacement_run.await.unwrap().unwrap();
    assert_eq!(host.runs.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn checked_out_session_is_checked_in_when_fingerprint_validation_fails() {
    let host = Arc::new(Host {
        fingerprint_error: true,
        ..Host::default()
    });
    let runner = runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    );

    assert!(matches!(
        runner
            .run(request(QueueDisposition::Interrupt, false))
            .await,
        Err(TurnRunnerError::Failed(_))
    ));
    assert_eq!(host.checkins.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn wall_deadline_covers_fingerprint_validation_and_checks_session_back_in() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        fingerprint_started: Some(started.clone()),
        fingerprint_release: Some(release),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let mut turn = request(QueueDisposition::Interrupt, false);
    turn.deadline = Some(std::time::Duration::from_secs(5));
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(turn).await })
    };
    started.notified().await;
    tokio::time::advance(std::time::Duration::from_secs(6)).await;

    assert!(
        matches!(run.await.unwrap(), Err(TurnRunnerError::Failed(message)) if message.contains("deadline"))
    );
    assert_eq!(host.checkins.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn two_agents_can_run_the_same_thread_id_concurrently() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let host = Arc::new(ConcurrentHost {
        barrier: barrier.clone(),
        sessions: Mutex::default(),
    });
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut first = request(QueueDisposition::Interrupt, false);
    first.thread_id = "shared-name".into();
    let mut second = first.clone();
    second.scope.agent_id = "agent-b".into();
    second.request_id = "request-b".into();
    let first = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(first).await })
    };
    let second = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(second).await })
    };

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    })
    .await
    .expect("agent-scoped sessions must not share a lock");
    assert_eq!(host.sessions.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn separate_threads_for_one_agent_do_not_share_a_session_lock() {
    let host = Arc::new(ConcurrentHost {
        barrier: Arc::new(tokio::sync::Barrier::new(2)),
        sessions: Mutex::default(),
    });
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let first = request(QueueDisposition::Interrupt, false);
    let mut second = first.clone();
    second.thread_id = "thread-b".into();
    second.request_id = "request-b".into();
    let first = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(first).await })
    };
    let second = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(second).await })
    };

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    })
    .await
    .expect("distinct threads must own independent session locks");
}

struct ConcurrentHost {
    barrier: Arc<tokio::sync::Barrier>,
    sessions: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl TurnHost for ConcurrentHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.request_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: format!("fork:{}", request.request_id),
            fingerprint: None,
        })
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        request: &TurnRequest,
        session: &TurnSession,
        _origin: &serde_json::Value,
        _cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        let expected_session = if request.disposition == QueueDisposition::Parallel {
            format!("fork:{}", request.request_id)
        } else {
            request.request_id.clone()
        };
        assert_eq!(session.session_id, expected_session);
        self.sessions
            .lock()
            .unwrap()
            .push(session.session_id.clone());
        self.barrier.wait().await;
        Ok(TurnResult {
            reply: "ok".into(),
            data: serde_json::Value::Null,
        })
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test]
async fn parallel_turns_use_independent_fork_sessions_and_run_concurrently() {
    let host = Arc::new(ConcurrentHost {
        barrier: Arc::new(tokio::sync::Barrier::new(2)),
        sessions: Mutex::default(),
    });
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    ));
    let mut first = request(QueueDisposition::Parallel, false);
    first.request_id = "parallel-a".into();
    let mut second = first.clone();
    second.request_id = "parallel-b".into();
    let first = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(first).await })
    };
    let second = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(second).await })
    };

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    })
    .await
    .expect("parallel branches must not serialize");
    let sessions = host.sessions.lock().unwrap();
    assert_eq!(sessions.as_slice(), ["fork:parallel-a", "fork:parallel-b"]);
}

struct QuietWorkHost {
    started: Arc<tokio::sync::Notify>,
    finish: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl TurnHost for QuietWorkHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        _cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        progress.send(TurnSignal::WorkStarted).await.unwrap();
        self.started.notify_one();
        self.finish.notified().await;
        progress.send(TurnSignal::WorkFinished).await.unwrap();
        std::future::pending().await
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test(start_paused = true)]
async fn idle_guard_restarts_when_quiet_work_finishes() {
    let started = Arc::new(tokio::sync::Notify::new());
    let finish = Arc::new(tokio::sync::Notify::new());
    let runner = Arc::new(TurnRunner::new(
        Arc::new(QuietWorkHost {
            started: started.clone(),
            finish: finish.clone(),
        }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig {
            idle_timeout: std::time::Duration::from_secs(5),
            deadline: None,
            close_timeout: std::time::Duration::from_secs(1),
            ..TurnRunnerConfig::default()
        },
    ));
    let mut turn = request(QueueDisposition::Interrupt, false);
    turn.idle_timeout = Some(std::time::Duration::from_secs(5));
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(turn).await })
    };
    started.notified().await;
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(10)).await;
    finish.notify_one();
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(6)).await;

    assert!(
        matches!(run.await.unwrap(), Err(TurnRunnerError::Failed(message)) if message.contains("idle timeout"))
    );
}

#[tokio::test(start_paused = true)]
async fn continuous_progress_cannot_starve_the_wall_deadline() {
    let started = Arc::new(tokio::sync::Notify::new());
    let runner = Arc::new(TurnRunner::new(
        Arc::new(FloodProgressHost {
            started: started.clone(),
        }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig {
            idle_timeout: std::time::Duration::from_secs(60),
            deadline: Some(std::time::Duration::from_secs(5)),
            close_timeout: std::time::Duration::from_secs(1),
            ..TurnRunnerConfig::default()
        },
    ));
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    started.notified().await;
    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(6)).await;

    assert!(
        matches!(run.await.unwrap(), Err(TurnRunnerError::Failed(message)) if message.contains("wall-clock deadline"))
    );
}

#[tokio::test(start_paused = true)]
async fn wall_deadline_is_polled_while_progress_delivery_is_pending() {
    let started = Arc::new(tokio::sync::Notify::new());
    let events = Arc::new(BlockingProgressEvents::default());
    let runner = Arc::new(TurnRunner::new(
        Arc::new(FloodProgressHost {
            started: started.clone(),
        }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig {
            idle_timeout: std::time::Duration::from_secs(60),
            deadline: Some(std::time::Duration::from_secs(5)),
            close_timeout: std::time::Duration::from_secs(1),
            ..TurnRunnerConfig::default()
        },
    ));
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    started.notified().await;
    events.started.notified().await;
    tokio::time::advance(std::time::Duration::from_secs(6)).await;

    assert!(
        matches!(run.await.unwrap(), Err(TurnRunnerError::Failed(message)) if message.contains("wall-clock deadline"))
    );
}

#[derive(Default)]
struct BlockingProgressEvents {
    started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl EventSink for BlockingProgressEvents {
    async fn progress(&self, _request: &TurnRequest, _event: crate::progress::AgentProgress) {
        self.started.notify_one();
        std::future::pending().await
    }
    async fn terminal(&self, _request: &TurnRequest, _event: TerminalEvent) {}
}

struct FloodProgressHost {
    started: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl TurnHost for FloodProgressHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        let sender = progress.clone();
        tokio::spawn(async move {
            while sender
                .send(TurnSignal::Progress(Box::new(
                    crate::progress::AgentProgress::TurnStarted,
                )))
                .await
                .is_ok()
            {}
        });
        self.started.notify_one();
        cancellation.cancelled().await;
        Err("cancelled".into())
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_grounded_close_prevents_persisting_completion() {
    let summary = Arc::new(BlockingSummary::default());
    let replies = Arc::new(Replies::default());
    let runner = Arc::new(TurnRunner::new(
        Arc::new(PendingHost),
        Arc::new(Ingress::default()),
        replies.clone(),
        Arc::new(Events::default()),
        Arc::new(Failures),
        summary.clone(),
        TurnRunnerConfig {
            idle_timeout: std::time::Duration::from_secs(5),
            deadline: None,
            close_timeout: std::time::Duration::from_secs(60),
            ..TurnRunnerConfig::default()
        },
    ));
    let mut turn = request(QueueDisposition::Interrupt, false);
    turn.idle_timeout = Some(std::time::Duration::from_secs(5));
    let scope = turn.scope.clone();
    let run = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(turn).await })
    };
    summary.started.notified().await;
    let close_started_at = tokio::time::Instant::now();
    assert!(runner.cancel(&scope, "thread-a", Some("request-a")).await);
    assert!(matches!(
        run.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(close_started_at.elapsed() < std::time::Duration::from_secs(1));
    assert!(replies.0.lock().unwrap().is_empty());
}

struct PendingHost;

#[async_trait::async_trait]
impl TurnHost for PendingHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        cancellation.cancelled().await;
        Err("cancelled".into())
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

#[derive(Default)]
struct BlockingSummary {
    started: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl Summarizer for BlockingSummary {
    async fn grounded_close(&self, _request: &TurnRequest, _reason: &str) -> Option<TurnResult> {
        self.started.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn live_turns_and_cancel_requests_report_scoped_primary_before_parallel() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));

    let mut primary_request = request(QueueDisposition::Interrupt, false);
    primary_request.request_id = "primary".into();
    let scope = primary_request.scope.clone();
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(primary_request).await })
    };
    started.notified().await;

    let mut parallel_request = request(QueueDisposition::Parallel, false);
    parallel_request.request_id = "parallel".into();
    let parallel = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(parallel_request).await })
    };

    let mut other_agent_request = request(QueueDisposition::Interrupt, false);
    other_agent_request.request_id = "other-agent".into();
    other_agent_request.scope.agent_id = "agent-b".into();
    let other_agent_scope = other_agent_request.scope.clone();
    let other_agent = {
        let runner = runner.clone();
        tokio::spawn(async move { runner.run(other_agent_request).await })
    };
    while host.runs.load(std::sync::atomic::Ordering::SeqCst) < 3 {
        tokio::task::yield_now().await;
    }

    let live = runner.live_turns().await;
    assert_eq!(
        live.iter()
            .map(|turn| (
                turn.scope.agent_id.as_str(),
                turn.request_id.as_str(),
                turn.parallel,
            ))
            .collect::<Vec<_>>(),
        [
            ("agent-a", "primary", false),
            ("agent-a", "parallel", true),
            ("agent-b", "other-agent", false),
        ]
    );

    assert_eq!(
        runner.cancel_requests(&scope, "thread-a", None).await,
        ["primary".to_owned(), "parallel".to_owned(),]
    );
    assert!(matches!(
        primary.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(matches!(
        parallel.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert_eq!(
        runner
            .cancel_requests(&other_agent_scope, "thread-a", None)
            .await,
        ["other-agent".to_owned()]
    );
    assert!(matches!(
        other_agent.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    assert!(runner.live_turns().await.is_empty());
}

#[tokio::test]
async fn queue_snapshot_preserves_lane_fifo_and_does_not_drain_work() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let host = Arc::new(Host {
        started: Some(started.clone()),
        release: Some(release),
        ..Host::default()
    });
    let runner = Arc::new(runner(
        host,
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
    ));
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    started.notified().await;

    for (disposition, id) in [
        (QueueDisposition::Steer, "steer-1"),
        (QueueDisposition::Steer, "steer-2"),
        (QueueDisposition::Followup, "followup-1"),
        (QueueDisposition::Collect, "collect-1"),
    ] {
        let mut queued = request(disposition, false);
        queued.request_id = id.into();
        assert!(matches!(
            runner.run(queued).await,
            Err(TurnRunnerError::Queued)
        ));
    }
    let scope = request(QueueDisposition::Interrupt, false).scope;

    let snapshot = runner.queue_snapshot(&scope, "thread-a").await;
    assert_eq!(
        snapshot
            .iter()
            .map(|(lane, item)| (*lane, item.request_id.as_str()))
            .collect::<Vec<_>>(),
        [
            (QueueLane::Steer, "steer-1"),
            (QueueLane::Steer, "steer-2"),
            (QueueLane::Followup, "followup-1"),
            (QueueLane::Collect, "collect-1"),
        ]
    );
    assert_eq!(runner.queue_status(&scope, "thread-a").await.total, 4);
    assert_eq!(
        runner
            .take_queued(&scope, "thread-a", QueueLane::Steer, QueueMode::OneAtATime)
            .await[0]
            .request_id,
        "steer-1"
    );

    runner.cancel_requests(&scope, "thread-a", None).await;
    assert!(matches!(
        primary.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
}

#[derive(Default)]
struct UnwindingHost {
    started: tokio::sync::Notify,
    persisted: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl TurnHost for UnwindingHost {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: None,
        })
    }
    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }
    fn run_origin(&self, _request: &TurnRequest) -> serde_json::Value {
        serde_json::Value::Null
    }
    async fn run(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
        _origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        _queue: Arc<RunQueue<TurnRequest>>,
        _progress: tokio::sync::mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String> {
        self.started.notify_one();
        cancellation.cancelled().await;
        // Hand back and persist the rounds completed so far.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        self.persisted
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Err("stopped".into())
    }
    async fn checkin(&self, _request: &TurnRequest) {}
    async fn cancel_children(&self, _request: &TurnRequest) {}
}

async fn cancel_unwinding_turn(grace: std::time::Duration) -> bool {
    let host = Arc::new(UnwindingHost::default());
    let runner = Arc::new(TurnRunner::new(
        host.clone(),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        Arc::new(Events::default()),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig {
            cancel_grace: grace,
            ..TurnRunnerConfig::default()
        },
    ));
    let primary = {
        let runner = runner.clone();
        tokio::spawn(async move {
            runner
                .run(request(QueueDisposition::Interrupt, false))
                .await
        })
    };
    host.started.notified().await;
    let req = request(QueueDisposition::Interrupt, false);
    assert!(runner.cancel(&req.scope, &req.thread_id, None).await);
    assert!(matches!(
        primary.await.unwrap(),
        Err(TurnRunnerError::Cancelled)
    ));
    host.persisted.load(std::sync::atomic::Ordering::SeqCst)
}

#[tokio::test]
async fn cancelled_run_unwinds_within_the_cancel_grace() {
    assert!(cancel_unwinding_turn(std::time::Duration::from_secs(2)).await);
}

#[tokio::test]
async fn cancelled_run_is_dropped_at_once_without_a_cancel_grace() {
    assert!(!cancel_unwinding_turn(std::time::Duration::ZERO).await);
}

#[path = "progress_delivery_tests.rs"]
mod progress_delivery;
