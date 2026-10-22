use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tinyagents_harness::CancellationToken;
use tinyagents_harness::cost::{CallCost, CostSource};
use tinyinference_llm::usage::{ChargedAmount, Usage};
use tokio::sync::{Notify, oneshot};

#[derive(Default)]
struct FakeHost {
    events: Vec<&'static str>,
    run: Option<Result<ChildTurnRun<String>, ChildTurnFailure<String, &'static str>>>,
    summary: Option<ChildTurnCapSummary>,
    persisted: Option<ChildTurnResult<String>>,
    failure_history: Option<Vec<String>>,
    fail_persist: bool,
    wait_on_run: bool,
    run_started: Option<oneshot::Sender<()>>,
    wait_on_summary: bool,
    summary_started: Option<oneshot::Sender<()>>,
    wait_on_persist: bool,
    persist_started: Option<oneshot::Sender<()>>,
}

#[async_trait::async_trait]
impl ChildTurnHost for FakeHost {
    type Message = String;
    type Error = &'static str;

    async fn run_turn(
        &mut self,
        cancellation: CancellationToken,
    ) -> Result<ChildTurnRun<String>, ChildTurnFailure<String, &'static str>> {
        if let Some(started) = self.run_started.take() {
            let _ = started.send(());
        }
        if self.wait_on_run {
            cancellation.cancelled().await;
        }
        self.events.push("run");
        self.run.take().unwrap()
    }

    async fn summarize_cap(
        &mut self,
        _: &ChildTurnRun<String>,
    ) -> Result<ChildTurnCapSummary, &'static str> {
        self.events.push("summary");
        if let Some(started) = self.summary_started.take() {
            let _ = started.send(());
        }
        if self.wait_on_summary {
            std::future::pending().await
        }
        Ok(self.summary.take().unwrap())
    }

    fn append_final_message(
        &mut self,
        history: &mut Vec<String>,
        text: &str,
    ) -> Result<(), &'static str> {
        self.events.push("append-final");
        history.push(format!("native:{text}"));
        Ok(())
    }

    async fn persist_success(
        &mut self,
        result: &ChildTurnResult<String>,
    ) -> Result<(), &'static str> {
        self.events.push("persist-success");
        if let Some(started) = self.persist_started.take() {
            let _ = started.send(());
        }
        if self.wait_on_persist {
            std::future::pending().await
        }
        if self.fail_persist {
            return Err("persist failed");
        }
        self.persisted = Some(result.clone());
        Ok(())
    }

    async fn persist_failure(
        &mut self,
        history: &[String],
        _: &ChildTurnAccounting,
    ) -> Result<(), &'static str> {
        self.events.push("persist-failure");
        self.failure_history = Some(history.to_vec());
        Ok(())
    }

    async fn after_commit(
        &mut self,
        _: &ChildTurnResult<String>,
        _: CancellationToken,
    ) -> Result<(), &'static str> {
        self.events.push("after-commit");
        Ok(())
    }
}

fn active_control() -> ChildTurnControl {
    ChildTurnControl::new(CancellationToken::new(), Arc::new(|| true))
}

struct RevocableFence {
    admitted: AtomicBool,
    lost: Notify,
}

impl RevocableFence {
    fn new() -> Self {
        Self {
            admitted: AtomicBool::new(true),
            lost: Notify::new(),
        }
    }

    fn revoke(&self) {
        self.admitted.store(false, Ordering::SeqCst);
        self.lost.notify_waiters();
    }
}

impl tinyagents_tasks::DeliveryFence for RevocableFence {
    fn admitted(&self) -> bool {
        self.admitted.load(Ordering::SeqCst)
    }

    fn lost(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move { self.lost.notified().await })
    }
}

fn totals(
    input: u64,
    output: u64,
    cache_read: u64,
    cache_create: u64,
    reasoning: u64,
) -> UsageTotals {
    let mut totals = UsageTotals::default();
    totals.record(Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        cache_read_tokens: cache_read,
        cache_creation_tokens: cache_create,
        reasoning_tokens: reasoning,
        charged_amount: Some(ChargedAmount::usd_micros(250)),
        context_window_tokens: Some(32_000),
    });
    totals
}

#[tokio::test]
async fn cap_summary_folds_complete_usage_and_runs_after_transcript_commit() {
    let accounting = ChildTurnAccounting {
        usage: totals(10, 4, 3, 2, 1),
        cost: {
            let mut cost = CostTally::default();
            cost.add(CallCost::Charged(1.25));
            cost
        },
        model_calls: 5,
    };
    let mut host = FakeHost {
        run: Some(Ok(ChildTurnRun {
            accepted_history: vec!["accepted-native-assistant-tool-call".into()],
            output: "partial".into(),
            model_calls: 5,
            accounting,
            hit_cap: true,
            early_exit: false,
        })),
        summary: Some(ChildTurnCapSummary {
            text: "resume from checkpoint".into(),
            usage: totals(7, 5, 2, 1, 3),
            cost: CallCost::Estimated(2.25),
        }),
        ..FakeHost::default()
    };

    let result = drive_child_turn(&mut host, active_control()).await.unwrap();

    assert_eq!(
        host.events,
        [
            "run",
            "summary",
            "append-final",
            "persist-success",
            "after-commit"
        ]
    );
    assert_eq!(
        result.accepted_history,
        [
            "accepted-native-assistant-tool-call",
            "native:resume from checkpoint"
        ]
    );
    assert_eq!(result.output, "resume from checkpoint");
    assert_eq!(result.accounting.model_calls, 5);
    assert_eq!(result.accounting.usage.calls, 2);
    assert_eq!(result.accounting.usage.usage.input_tokens, 17);
    assert_eq!(result.accounting.usage.usage.output_tokens, 9);
    assert_eq!(result.accounting.usage.usage.total_tokens, 26);
    assert_eq!(result.accounting.usage.usage.cache_read_tokens, 5);
    assert_eq!(result.accounting.usage.usage.cache_creation_tokens, 3);
    assert_eq!(result.accounting.usage.usage.reasoning_tokens, 4);
    assert_eq!(
        result.accounting.usage.usage.charged_amount,
        Some(ChargedAmount::usd_micros(500))
    );
    assert_eq!(result.accounting.cost.known_usd, 3.5);
    assert_eq!(result.accounting.cost.source, CostSource::Estimated);
    assert_eq!(host.persisted, Some(result));
}

#[tokio::test]
async fn early_exit_appends_typed_question_without_cap_summary() {
    let mut host = FakeHost {
        run: Some(Ok(ChildTurnRun {
            accepted_history: vec!["accepted-native-tool-result".into()],
            output: "Which account should I use?".into(),
            model_calls: 3,
            accounting: ChildTurnAccounting::default(),
            hit_cap: false,
            early_exit: true,
        })),
        ..FakeHost::default()
    };

    let result = drive_child_turn(&mut host, active_control()).await.unwrap();

    assert_eq!(
        host.events,
        ["run", "append-final", "persist-success", "after-commit"]
    );
    assert_eq!(
        result.accepted_history,
        [
            "accepted-native-tool-result",
            "native:Which account should I use?"
        ]
    );
    assert!(result.early_exit);
    assert!(!result.hit_cap);
    assert_eq!(result.accounting.model_calls, 3);
    assert_eq!(host.persisted, Some(result));
}

#[tokio::test]
async fn failed_turn_persists_only_accepted_history_before_returning_error() {
    let mut host = FakeHost {
        run: Some(Err(ChildTurnFailure {
            error: "provider failed",
            accepted_history: vec!["accepted-user".into(), "accepted-assistant".into()],
            accounting: ChildTurnAccounting {
                model_calls: 1,
                usage: totals(3, 2, 0, 0, 0),
                ..ChildTurnAccounting::default()
            },
        })),
        ..FakeHost::default()
    };

    assert_eq!(
        drive_child_turn(&mut host, active_control()).await,
        Err(ChildTurnError::Host("provider failed"))
    );
    assert_eq!(host.events, ["run", "persist-failure"]);
    assert_eq!(
        host.failure_history,
        Some(vec!["accepted-user".into(), "accepted-assistant".into()])
    );
    assert!(host.persisted.is_none());
}

#[tokio::test]
async fn failed_transcript_commit_never_runs_deferred_terminal_effects() {
    let mut host = FakeHost {
        run: Some(Ok(ChildTurnRun {
            accepted_history: vec!["accepted-xml-suffix".into()],
            output: "done".into(),
            model_calls: 2,
            accounting: ChildTurnAccounting::default(),
            hit_cap: false,
            early_exit: false,
        })),
        fail_persist: true,
        ..FakeHost::default()
    };

    assert_eq!(
        drive_child_turn(&mut host, active_control()).await,
        Err(ChildTurnError::Host("persist failed"))
    );
    assert_eq!(host.events, ["run", "persist-success"]);
    assert!(host.persisted.is_none());
}

#[tokio::test]
async fn losing_turn_fence_cancels_host_work_and_skips_persistence() {
    let fence = Arc::new(RevocableFence::new());
    let cancellation = CancellationToken::new();
    let control = ChildTurnControl::new(cancellation.clone(), fence.clone());
    let (started, started_rx) = oneshot::channel();
    let mut host = FakeHost {
        wait_on_run: true,
        run_started: Some(started),
        ..FakeHost::default()
    };

    let (outcome, ()) = tokio::join!(drive_child_turn(&mut host, control), async {
        started_rx.await.unwrap();
        fence.revoke();
    });

    assert_eq!(outcome, Err(ChildTurnError::LeaseLost));
    assert!(cancellation.is_cancelled());
    assert!(host.persisted.is_none());
    assert!(!host.events.contains(&"persist-success"));
    assert!(!host.events.contains(&"after-commit"));
}

#[tokio::test]
async fn losing_turn_fence_during_cap_summary_skips_transcript_commit() {
    let fence = Arc::new(RevocableFence::new());
    let cancellation = CancellationToken::new();
    let control = ChildTurnControl::new(cancellation.clone(), fence.clone());
    let (started, started_rx) = oneshot::channel();
    let mut host = FakeHost {
        run: Some(Ok(ChildTurnRun {
            accepted_history: vec!["accepted".into()],
            output: "partial".into(),
            model_calls: 4,
            accounting: ChildTurnAccounting::default(),
            hit_cap: true,
            early_exit: false,
        })),
        wait_on_summary: true,
        summary_started: Some(started),
        ..FakeHost::default()
    };

    let (outcome, ()) = tokio::join!(drive_child_turn(&mut host, control), async {
        started_rx.await.unwrap();
        fence.revoke();
    });

    assert_eq!(outcome, Err(ChildTurnError::LeaseLost));
    assert!(cancellation.is_cancelled());
    assert!(host.persisted.is_none());
    assert!(!host.events.contains(&"persist-success"));
    assert!(!host.events.contains(&"after-commit"));
}

#[tokio::test]
async fn losing_turn_fence_during_transcript_commit_skips_deferred_terminal_work() {
    let fence = Arc::new(RevocableFence::new());
    let cancellation = CancellationToken::new();
    let control = ChildTurnControl::new(cancellation.clone(), fence.clone());
    let (started, started_rx) = oneshot::channel();
    let mut host = FakeHost {
        run: Some(Ok(ChildTurnRun {
            accepted_history: vec!["accepted".into()],
            output: "done".into(),
            model_calls: 1,
            accounting: ChildTurnAccounting::default(),
            hit_cap: false,
            early_exit: false,
        })),
        wait_on_persist: true,
        persist_started: Some(started),
        ..FakeHost::default()
    };

    let (outcome, ()) = tokio::join!(drive_child_turn(&mut host, control), async {
        started_rx.await.unwrap();
        fence.revoke();
    });

    assert_eq!(outcome, Err(ChildTurnError::LeaseLost));
    assert!(cancellation.is_cancelled());
    assert!(host.persisted.is_none());
    assert!(!host.events.contains(&"after-commit"));
}
