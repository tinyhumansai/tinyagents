//! The runner keeps polling the turn while it hands a progress event to the
//! sink.
//!
//! A sink's delivery can depend on the turn moving: directly, or through a
//! resource the turn's future already holds or is queued for (a lock whose
//! permit was handed to a waiter inside the turn). If the runner awaited the
//! delivery without polling the turn, the two would wait on each other for
//! good, with every deadline inside the turn frozen with it.

use super::*;
use crate::progress::AgentProgress;

/// Sends one progress event, yields, then signals that it was polled again.
struct SignalsAfterProgress {
    polled_again: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl TurnHost for SignalsAfterProgress {
    async fn checkout(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        Ok(TurnSession {
            session_id: request.thread_id.clone(),
            fingerprint: Some("stable".into()),
        })
    }

    async fn checkout_fork(&self, request: &TurnRequest) -> Result<TurnSession, String> {
        self.checkout(request).await
    }

    async fn validate_fingerprint(
        &self,
        _request: &TurnRequest,
        _session: &TurnSession,
    ) -> Result<(), String> {
        Ok(())
    }

    fn run_origin(&self, request: &TurnRequest) -> serde_json::Value {
        serde_json::json!({"request": request.request_id})
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
                AgentProgress::TurnStarted,
            )))
            .await
            .map_err(|_| "progress channel closed".to_owned())?;
        // Hand control back so the runner sees the event before the turn
        // takes its next step.
        tokio::task::yield_now().await;
        self.polled_again.notify_one();
        Ok(TurnResult {
            reply: "done".into(),
            data: serde_json::Value::Null,
        })
    }

    async fn checkin(&self, _request: &TurnRequest) {}

    async fn cancel_children(&self, _request: &TurnRequest) {}
}

/// Delivers a progress event only once the turn has taken its next step.
struct WaitsForTheTurn {
    polled_again: Arc<tokio::sync::Notify>,
    terminal: Mutex<Vec<TerminalEvent>>,
}

#[async_trait::async_trait]
impl EventSink for WaitsForTheTurn {
    async fn progress(&self, _request: &TurnRequest, _event: AgentProgress) {
        self.polled_again.notified().await;
    }

    async fn terminal(&self, _request: &TurnRequest, event: TerminalEvent) {
        self.terminal.lock().unwrap().push(event);
    }

    async fn collected(&self, _request: &TurnRequest, _items: Vec<TurnRequest>) {}
}

#[tokio::test]
async fn a_progress_delivery_that_waits_on_the_turn_does_not_freeze_it() {
    let polled_again = Arc::new(tokio::sync::Notify::new());
    let events = Arc::new(WaitsForTheTurn {
        polled_again: polled_again.clone(),
        terminal: Mutex::default(),
    });
    let runner = TurnRunner::new(
        Arc::new(SignalsAfterProgress { polled_again }),
        Arc::new(Ingress::default()),
        Arc::new(Replies::default()),
        events.clone(),
        Arc::new(Failures),
        Arc::new(Summary),
        TurnRunnerConfig::default(),
    );

    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        runner.run(request(QueueDisposition::Interrupt, false)),
    )
    .await
    .expect("the turn must keep running while its progress is delivered");

    assert!(outcome.is_ok(), "{outcome:?}");
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [TerminalEvent::Completed { .. }]
    ));
}
