//! Tests for the streaming model idle timeout and its per-model
//! consecutive-timeout breaker.
//!
//! Every test runs on tokio's paused clock: a silent stream parks on a timer
//! the runtime auto-advances, so there are no real sleeps and the elapsed
//! virtual time is exact.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::Instant;

use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::limits::RunLimits;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{ScriptedModel, SlowModel};
use tinyinference_llm::message::{Message, MessageDelta};
use tinyinference_llm::model::{
    BlockDelta, ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
};
use tinyinference_llm::usage::Usage;

/// One step of a scripted stream.
#[derive(Clone)]
enum Step {
    /// Yield this item.
    Item(Box<ModelStreamItem>),
    /// Wait this long (on the tokio clock) before the next step.
    Sleep(Duration),
    /// Never produce anything again.
    Hang,
}

fn delta(text: &str) -> Step {
    Step::Item(Box::new(ModelStreamItem::MessageDelta(MessageDelta {
        text: text.to_string(),
        reasoning: String::new(),
        tool_call: None,
    })))
}

fn reasoning(text: &str) -> Step {
    Step::Item(Box::new(ModelStreamItem::MessageDelta(MessageDelta {
        text: String::new(),
        reasoning: text.to_string(),
        tool_call: None,
    })))
}

fn usage() -> Step {
    Step::Item(Box::new(ModelStreamItem::UsageDelta(Usage::default())))
}

fn started() -> Step {
    Step::Item(Box::new(ModelStreamItem::Started))
}

fn completed(text: &str) -> Step {
    Step::Item(Box::new(ModelStreamItem::Completed(
        ModelResponse::assistant(text),
    )))
}

/// A streaming model that plays one script per call; the last script repeats.
struct ScriptedStreams {
    scripts: Mutex<VecDeque<Vec<Step>>>,
    calls: Mutex<usize>,
    /// When set, `stream(...)` itself never resolves (a provider that hangs
    /// while the stream is being opened).
    hang_on_open: bool,
}

impl ScriptedStreams {
    fn new(scripts: Vec<Vec<Step>>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            calls: Mutex::new(0),
            hang_on_open: false,
        })
    }

    /// A model whose `stream(...)` call never returns.
    fn hanging_open() -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(VecDeque::new()),
            calls: Mutex::new(0),
            hang_on_open: true,
        })
    }

    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ScriptedStreams {
    async fn invoke(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        unreachable!("these tests only drive the streaming path")
    }

    async fn stream(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        *self.calls.lock().unwrap() += 1;
        if self.hang_on_open {
            futures::future::pending::<()>().await;
        }
        let script = {
            let mut scripts = self.scripts.lock().unwrap();
            if scripts.len() > 1 {
                scripts.pop_front().unwrap()
            } else {
                scripts.front().cloned().unwrap_or_default()
            }
        };
        let stream = futures::stream::unfold(VecDeque::from(script), |mut steps| async move {
            loop {
                match steps.pop_front()? {
                    Step::Item(item) => return Some((*item, steps)),
                    Step::Sleep(delay) => tokio::time::sleep(delay).await,
                    Step::Hang => futures::future::pending::<()>().await,
                }
            }
        });
        Ok(ModelStream::new(Box::pin(stream)))
    }
}

fn harness_with(
    primary: Arc<dyn ChatModel<()>>,
    limits: RunLimits,
    attempts: usize,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary);
    harness.with_policy(RunPolicy {
        limits,
        retry: RetryPolicy::default()
            .with_max_attempts(attempts)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness
}

/// A harness whose primary model is `primary` and whose fallback chain is
/// `[primary, others...]`.
fn harness_with_chain(
    primary: Arc<dyn ChatModel<()>>,
    others: Vec<(&str, Arc<dyn ChatModel<()>>)>,
    limits: RunLimits,
    attempts: usize,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary);
    let mut models = vec!["primary".to_string()];
    for (name, model) in others {
        harness.register_model(name, model);
        models.push(name.to_string());
    }
    harness.with_policy(RunPolicy {
        limits,
        retry: RetryPolicy::default()
            .with_max_attempts(attempts)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy { models }),
        ..RunPolicy::default()
    });
    harness
}

/// The silent-stream limits most tests use: a 1s idle window, no breaker.
fn idle_1s() -> RunLimits {
    RunLimits::default()
        .with_stream_idle_timeout_ms(Some(1_000))
        .with_max_consecutive_stream_idle_timeouts(None)
}

const SECOND: Duration = Duration::from_secs(1);

/// Runs a streaming turn, failing the test (instead of hanging) when the run
/// does not finish within a virtual hour.
async fn run(
    harness: &AgentHarness<()>,
    config: RunConfig,
) -> crate::error::Result<crate::middleware::AgentRun> {
    tokio::time::timeout(
        Duration::from_secs(3600),
        harness.invoke_streaming(&(), (), config, vec![Message::user("hi")]),
    )
    .await
    .expect("the run hung: no timeout ever fired on a silent stream")
}

// ── idle timeout (after the first output event) ──────────────────────────────

#[tokio::test(start_paused = true)]
async fn stream_that_goes_silent_after_output_fails_with_a_retryable_call_timeout() {
    let model = ScriptedStreams::new(vec![vec![started(), delta("par"), Step::Hang]]);
    let harness = harness_with(model.clone(), idle_1s(), 3);

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("idle-run"))
        .await
        .expect_err("a stream that goes silent must fail");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("between output events"), "{message}");
            assert!(message.contains("1000 ms"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    assert!(crate::retry::is_retryable(&err));
    // Three attempts (the policy's cap), each dying after exactly one idle
    // window of silence.
    assert_eq!(model.calls(), 3);
    assert_eq!(began.elapsed(), 3 * SECOND);
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_is_rearmed_by_every_output_event() {
    // 30 events 800ms apart: 24s in total, far past the 1s idle window, but no
    // single gap is. A total-duration timer would kill this call.
    let mut script = vec![started()];
    for _ in 0..30 {
        script.push(Step::Sleep(Duration::from_millis(800)));
        script.push(delta("x"));
    }
    script.push(completed(&"x".repeat(30)));
    let model = ScriptedStreams::new(vec![script]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        1,
    );

    let began = Instant::now();
    let run = run(&harness, RunConfig::new("steady-run"))
        .await
        .expect("a steadily streaming call must not trip the idle timeout");

    assert_eq!(run.text().as_deref(), Some("x".repeat(30).as_str()));
    assert_eq!(model.calls(), 1);
    assert_eq!(began.elapsed(), Duration::from_millis(30 * 800));
}

#[tokio::test(start_paused = true)]
async fn usage_updates_do_not_extend_the_idle_deadline() {
    // After the first token the provider sends only usage updates, 900ms
    // apart. The idle deadline (1s after the token) must not move.
    let mut script = vec![started(), delta("tok")];
    for _ in 0..5 {
        script.push(Step::Sleep(Duration::from_millis(900)));
        script.push(usage());
    }
    script.push(Step::Hang);
    let model = ScriptedStreams::new(vec![script]);
    let harness = harness_with(model.clone(), idle_1s(), 1);

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("usage-trickle"))
        .await
        .expect_err("usage updates must not keep a stalled stream alive");

    assert!(matches!(err, TinyAgentsError::CallTimeout(_)), "{err:?}");
    assert_eq!(began.elapsed(), model.calls() as u32 * SECOND);
}

#[tokio::test(start_paused = true)]
async fn disabled_idle_timeout_waits_for_a_slow_stream() {
    for disabled in [None, Some(0)] {
        let model = ScriptedStreams::new(vec![vec![
            started(),
            delta("early"),
            Step::Sleep(Duration::from_secs(900)),
            delta("late"),
            completed("earlylate"),
        ]]);
        let harness = harness_with(
            model.clone(),
            RunLimits::default().with_stream_idle_timeout_ms(disabled),
            1,
        );

        let run = run(&harness, RunConfig::new("disabled-run"))
            .await
            .expect("with the idle timeout disabled a slow stream must complete");
        assert_eq!(run.text().as_deref(), Some("earlylate"), "{disabled:?}");
    }
}

// ── first output event ───────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn default_limits_allow_the_first_output_after_ten_minutes() {
    // Hidden-reasoning models and local CPU prefill can be silent for many
    // minutes before the first token; the default limits must not cut them off.
    let model = ScriptedStreams::new(vec![vec![
        started(),
        Step::Sleep(Duration::from_secs(600)),
        delta("answer"),
        completed("answer"),
    ]]);
    let harness = harness_with(model.clone(), RunLimits::default(), 1);

    let began = Instant::now();
    let run = run(&harness, RunConfig::new("slow-first-token"))
        .await
        .expect("the default idle timeout must not apply before the first output");

    assert_eq!(run.text().as_deref(), Some("answer"));
    assert_eq!(model.calls(), 1);
    assert_eq!(began.elapsed(), Duration::from_secs(600));
}

#[tokio::test(start_paused = true)]
async fn first_event_window_is_opt_in_and_bounds_the_first_output() {
    let model = ScriptedStreams::new(vec![vec![Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        idle_1s().with_stream_first_event_timeout_ms(Some(5_000)),
        2,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("no-first-output"))
        .await
        .expect_err("a stream that never produces output must fail");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("first output event"), "{message}");
            assert!(message.contains("5000 ms"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    // The long first-output window applied, not the 1s idle one.
    assert_eq!(model.calls(), 2);
    assert_eq!(began.elapsed(), 10 * SECOND);
}

#[tokio::test(start_paused = true)]
async fn first_event_window_gives_slow_starts_room_and_hands_over_to_the_idle_window() {
    // Output at 3s is past the 1s idle window but inside the 5s first-output
    // window; afterwards the idle window governs and the stream keeps up.
    let model = ScriptedStreams::new(vec![vec![
        started(),
        Step::Sleep(Duration::from_secs(3)),
        delta("a"),
        Step::Sleep(Duration::from_millis(900)),
        delta("b"),
        completed("ab"),
    ]]);
    let harness = harness_with(
        model.clone(),
        idle_1s().with_stream_first_event_timeout_ms(Some(5_000)),
        1,
    );

    let run = run(&harness, RunConfig::new("slow-start-run"))
        .await
        .expect("a first output inside the first-event window must succeed");
    assert_eq!(run.text().as_deref(), Some("ab"));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn reasoning_only_delta_counts_as_the_first_output() {
    // A reasoning-only delta is output: it ends the 1s first-event phase, so
    // the 10s gap that follows is governed by the 60s idle window.
    let model = ScriptedStreams::new(vec![vec![
        started(),
        reasoning("thinking"),
        Step::Sleep(Duration::from_secs(10)),
        delta("done"),
        completed("done"),
    ]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(60_000))
            .with_stream_first_event_timeout_ms(Some(1_000)),
        1,
    );

    let run = run(&harness, RunConfig::new("reasoning-first"))
        .await
        .expect("reasoning output must end the first-event phase");
    assert_eq!(run.text().as_deref(), Some("done"));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn start_marker_and_usage_updates_do_not_end_the_first_event_phase() {
    // A start marker and usage updates arrive at 900ms/1800ms, but the 2s
    // first-output deadline set when the stream opened does not move.
    let model = ScriptedStreams::new(vec![vec![
        started(),
        Step::Sleep(Duration::from_millis(900)),
        usage(),
        Step::Sleep(Duration::from_millis(900)),
        usage(),
        Step::Hang,
    ]]);
    let harness = harness_with(
        model.clone(),
        idle_1s().with_stream_first_event_timeout_ms(Some(2_000)),
        1,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("usage-first-phase"))
        .await
        .expect_err("non-output events must not satisfy the first-event window");

    assert!(matches!(err, TinyAgentsError::CallTimeout(_)), "{err:?}");
    assert_eq!(began.elapsed(), model.calls() as u32 * 2 * SECOND);
}

#[tokio::test(start_paused = true)]
async fn zero_windows_mean_no_bound() {
    let model = ScriptedStreams::new(vec![vec![
        started(),
        Step::Sleep(Duration::from_secs(900)),
        delta("late"),
        completed("late"),
    ]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(0))
            .with_stream_first_event_timeout_ms(Some(0)),
        1,
    );

    let run = run(&harness, RunConfig::new("zero-windows"))
        .await
        .expect("a zero window must not fail the stream instantly");
    assert_eq!(run.text().as_deref(), Some("late"));
}

// ── retry and fallback ───────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn idle_timeout_is_retried_and_the_retry_can_succeed() {
    let model = ScriptedStreams::new(vec![
        vec![started(), delta("par"), Step::Hang],
        vec![started(), delta("recovered"), completed("recovered")],
    ]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        3,
    );

    let run = run(&harness, RunConfig::new("retry-run"))
        .await
        .expect("the retry after an idle timeout must succeed");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert_eq!(model.calls(), 2);
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_falls_back_to_the_next_model_with_the_breaker_enabled() {
    let primary = ScriptedStreams::new(vec![vec![started(), delta("par"), Step::Hang]]);
    let fallback = Arc::new(ScriptedModel::replies(vec!["fallback answer"]));
    let harness = harness_with_chain(
        primary.clone(),
        vec![("fallback", fallback.clone())],
        RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        1,
    );

    let run = run(&harness, RunConfig::new("fallback-run"))
        .await
        .expect("an idle timeout must fall through to the fallback chain");

    assert_eq!(run.text().as_deref(), Some("fallback answer"));
    assert_eq!(fallback.requests().len(), 1);
}

// ── breaker ──────────────────────────────────────────────────────────────────
//
// The breaker counts idle timeouts with no output event in between, so a
// stalled stream is scripted to open and then stay silent *before* any output,
// with the opt-in first-event window as the bound that fires.

#[tokio::test(start_paused = true)]
async fn breaker_stops_retrying_a_stalled_model_but_lets_the_fallback_answer() {
    let primary = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let fallback = Arc::new(ScriptedModel::replies(vec!["fallback answer"]));
    let harness = harness_with_chain(
        primary.clone(),
        vec![("fallback", fallback.clone())],
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(3))
            .with_max_retries_per_call(10),
        10,
    );

    let began = Instant::now();
    let run = run(&harness, RunConfig::new("breaker-fallback"))
        .await
        .expect("a tripped breaker must not pre-empt the fallback chain");

    assert_eq!(run.text().as_deref(), Some("fallback answer"));
    assert_eq!(
        primary.calls(),
        3,
        "the stalled model stops at the threshold"
    );
    assert_eq!(fallback.requests().len(), 1);
    assert_eq!(began.elapsed(), 3 * SECOND);
}

#[tokio::test(start_paused = true)]
async fn breaker_fails_the_run_only_when_the_chain_is_exhausted() {
    // Both models stall. The count is per model: each gets its own three
    // strikes before the chain runs out.
    let primary = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let second = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let harness = harness_with_chain(
        primary.clone(),
        vec![("second", second.clone())],
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(3))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("breaker-exhausted"))
        .await
        .expect_err("every model stalled, so the run must fail");

    match &err {
        TinyAgentsError::StreamIdleTimeout(message) => {
            assert!(message.contains("3 consecutive"), "{message}");
        }
        other => panic!("expected StreamIdleTimeout, got {other:?}"),
    }
    assert!(!crate::retry::is_retryable(&err));
    assert_eq!(primary.calls(), 3);
    assert_eq!(second.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn breaker_without_a_fallback_fails_the_run_at_the_threshold() {
    let model = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(3))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("breaker-solo"))
        .await
        .expect_err("the breaker must fail the run");

    assert!(
        matches!(err, TinyAgentsError::StreamIdleTimeout(_)),
        "{err:?}"
    );
    assert_eq!(model.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn reasoning_output_resets_the_breaker() {
    // Threshold 2. Attempt 1 is silent (count 1); attempt 2 delivers reasoning
    // (count back to 0) before going silent (count 1); attempt 3 is silent
    // (count 2, tripped). Without the reset it would trip on attempt 2.
    let model = ScriptedStreams::new(vec![
        vec![Step::Hang],
        vec![started(), reasoning("hm"), Step::Hang],
        vec![Step::Hang],
    ]);
    let harness = harness_with(
        model.clone(),
        idle_1s()
            .with_max_consecutive_stream_idle_timeouts(Some(2))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("reset-run"))
        .await
        .expect_err("the breaker must eventually trip");

    assert!(
        matches!(err, TinyAgentsError::StreamIdleTimeout(_)),
        "{err:?}"
    );
    assert_eq!(model.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn usage_updates_do_not_reset_the_breaker() {
    // Each attempt sends a start marker and a usage update, then goes silent.
    // Neither is output, so the count keeps climbing and trips at 2.
    let model = ScriptedStreams::new(vec![vec![started(), usage(), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        idle_1s()
            .with_max_consecutive_stream_idle_timeouts(Some(2))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("usage-no-reset"))
        .await
        .expect_err("the breaker must trip");

    assert!(
        matches!(err, TinyAgentsError::StreamIdleTimeout(_)),
        "{err:?}"
    );
    assert_eq!(model.calls(), 2);
}

#[tokio::test(start_paused = true)]
async fn breaker_stops_retry_middleware_retries() {
    // With a `RetryMiddleware` registered the base call skips its own retry
    // loop and the middleware retries the whole call. The breaker's
    // `StreamIdleTimeout` must stop middleware retries at the threshold, not at the
    // middleware's much larger attempt cap.
    let model = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let mut harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(3)),
        10,
    );
    harness.push_model_middleware(Arc::new(crate::middleware::library::RetryMiddleware::new(
        RetryPolicy::default()
            .with_max_attempts(10)
            .with_backoff_sleep(false),
    )));

    let err = run(&harness, RunConfig::new("breaker-middleware"))
        .await
        .expect_err("the breaker must fail the run");

    assert!(
        matches!(err, TinyAgentsError::StreamIdleTimeout(_)),
        "{err:?}"
    );
    assert_eq!(model.calls(), 3);
}

// ── interaction with the other bounds ────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn per_call_ceiling_is_a_plain_call_timeout_that_does_not_feed_the_breaker() {
    // The 500ms ceiling is tighter than the 60s idle window, so it fires
    // first. A breaker with threshold 1 would trip on a single idle timeout;
    // the run must instead surface the ceiling's own retryable error.
    let model = ScriptedStreams::new(vec![vec![started(), delta("a"), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_max_model_call_ms(Some(500))
            .with_stream_idle_timeout_ms(Some(60_000))
            .with_max_consecutive_stream_idle_timeouts(Some(1)),
        2,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("ceiling-run"))
        .await
        .expect_err("the per-call ceiling must fire");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("per-model-call ceiling"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    assert_eq!(model.calls(), 2);
    assert_eq!(began.elapsed(), Duration::from_millis(1_000));
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_the_idle_wait_is_cancelled_not_a_timeout() {
    let model = ScriptedStreams::new(vec![vec![started(), delta("a"), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(1)),
        5,
    );
    let token = crate::cancel::CancellationToken::new();
    let ctx = RunContext::new(RunConfig::new("cancel-run"), ()).with_cancellation(token.clone());

    let began = Instant::now();
    let (result, ()) = tokio::join!(
        harness.invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")]),
        async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            token.cancel();
        }
    );

    assert!(
        matches!(result, Err(TinyAgentsError::Cancelled)),
        "{result:?}"
    );
    assert_eq!(model.calls(), 1);
    assert_eq!(began.elapsed(), Duration::from_millis(300));
}

/// Burns **real** time (not virtual) before every model call, standing in for
/// scheduling delay on a loaded CI box.
struct RealStall(Duration);

#[async_trait]
impl crate::middleware::Middleware<(), ()> for RealStall {
    fn name(&self) -> &str {
        "real_stall"
    }
    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> crate::error::Result<()> {
        std::thread::sleep(self.0);
        Ok(())
    }
}

/// The run deadline is measured by the limit tracker on the **real** clock
/// (`std::time::Instant`, restarted when the run begins), while this test's
/// stream runs on tokio's paused virtual clock. Real time spent between the run
/// starting and the call being issued (scheduling, a loaded CI box) is
/// therefore subtracted from the budget the call gets, so the virtual time to
/// the timeout is `500ms - real_elapsed`, not exactly `500ms`. The test used to
/// assert exact equality and failed whenever more than ~1ms of real time passed
/// (it passed locally only because the loop is that fast). `stall` makes that
/// real delay explicit and deterministic.
async fn run_against_a_500ms_deadline(stall: Duration) -> (usize, Duration, TinyAgentsError) {
    let model = ScriptedStreams::new(vec![vec![started(), delta("a"), Step::Hang]]);
    let mut harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(60_000)),
        3,
    );
    harness.push_middleware(Arc::new(RealStall(stall)));
    let ctx = RunContext::new(RunConfig::new("deadline-run").with_timeout_ms(500), ());

    let began = Instant::now();
    let err = tokio::time::timeout(
        Duration::from_secs(3600),
        harness.invoke_streaming_in_context(&(), ctx, vec![Message::user("hi")]),
    )
    .await
    .expect("the run hung: the run deadline never fired")
    .expect_err("the run deadline must fire first");
    (model.calls(), began.elapsed(), err)
}

#[tokio::test(start_paused = true)]
async fn run_deadline_still_wins_as_a_terminal_timeout() {
    let (calls, elapsed, err) = run_against_a_500ms_deadline(Duration::ZERO).await;

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert_eq!(calls, 1, "a run-deadline timeout is not retried");
    // Never later than the deadline (plus the timer wheel's 1ms rounding), and
    // far sooner than the 60s idle window it must beat.
    assert!(elapsed <= Duration::from_millis(501), "elapsed {elapsed:?}");
    assert!(elapsed >= Duration::from_millis(400), "elapsed {elapsed:?}");
}

#[tokio::test(start_paused = true)]
async fn run_deadline_budget_shrinks_by_real_time_already_spent() {
    // 100ms of real time spent before the call leaves ~400ms of the 500ms
    // budget; the timeout must fire after that remainder, not a fresh 500ms.
    let (calls, elapsed, err) = run_against_a_500ms_deadline(Duration::from_millis(100)).await;

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert_eq!(calls, 1);
    assert!(elapsed <= Duration::from_millis(401), "elapsed {elapsed:?}");
    assert!(elapsed >= Duration::from_millis(100), "elapsed {elapsed:?}");
}

#[tokio::test(start_paused = true)]
async fn non_streaming_calls_ignore_the_idle_timeout() {
    // A buffered call has no inter-event gaps to measure: a 300s call under a
    // 1s idle timeout must still complete.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_secs(300), "done")),
    );
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(1_000)),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke(
            &(),
            (),
            RunConfig::new("unary-run"),
            vec![Message::user("hi")],
        )
        .await
        .expect("the idle timeout must not apply to non-streaming calls");
    assert_eq!(run.text().as_deref(), Some("done"));
}

// ── progress accounting ──────────────────────────────────────────────────────

/// Chunks of a tool call a model writes as text (DeepSeek DSML). The streaming
/// scrubber consumes every one of them, so none reaches a consumer, but each is
/// real model output that proves the provider is alive.
const DSML_CHUNKS: [&str; 6] = [
    "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name=\"shell\">\n",
    "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">cd /app && ",
    "grep -rn jsonpath pkg/ ",
    "&& cat pkg/jsonpath/mod.rs ",
    "</｜｜DSML｜｜ parameter>\n",
    "</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>",
];

/// `chunks` streamed 800ms apart, then the terminal response.
fn slow_chunks(prefix: Vec<Step>, chunks: &[&str]) -> Vec<Step> {
    let mut script = prefix;
    for chunk in chunks {
        script.push(Step::Sleep(Duration::from_millis(800)));
        script.push(delta(chunk));
    }
    script.push(completed(&chunks.concat()));
    script
}

#[tokio::test(start_paused = true)]
async fn scrubbed_tool_call_text_keeps_the_idle_deadline_alive() {
    // Visible text arms the 1s idle window; the 4.8s of tool-call writing that
    // follows is entirely consumed by the scrubber, yet no gap exceeds 800ms.
    let model = ScriptedStreams::new(vec![slow_chunks(
        vec![started(), delta("Let me look. ")],
        &DSML_CHUNKS,
    )]);
    let harness = harness_with(model.clone(), idle_1s().with_max_model_calls(1), 1);

    let run = run(&harness, RunConfig::new("scrubbed-idle"))
        .await
        .expect("a model still writing a tool call as text is not idle");

    assert_eq!(model.calls(), 1);
    assert!(!run.text().unwrap_or_default().contains("DSML"));
}

#[tokio::test(start_paused = true)]
async fn scrubbed_tool_call_text_satisfies_the_first_event_window() {
    // The very first output is a tool call written as text: no chunk survives
    // scrubbing, but the 1s first-event window must still be satisfied.
    let model = ScriptedStreams::new(vec![slow_chunks(vec![started()], &DSML_CHUNKS)]);
    let harness = harness_with(
        model.clone(),
        idle_1s()
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_model_calls(1),
        1,
    );

    run(&harness, RunConfig::new("scrubbed-first"))
        .await
        .expect("scrubbed tool-call text is output for the first-event window");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn scrubbed_tool_call_text_resets_the_breaker() {
    // Threshold 2. Attempt 1 is silent (count 1). Attempt 2 writes a tool call
    // as text (fully scrubbed) before going silent: that output resets the
    // count to 0, so attempt 3's silence only makes it 1 and attempt 4's
    // makes it 2. Without the reset it would trip on attempt 2.
    let model = ScriptedStreams::new(vec![
        vec![Step::Hang],
        vec![started(), delta(DSML_CHUNKS[0]), Step::Hang],
        vec![Step::Hang],
    ]);
    let harness = harness_with(
        model.clone(),
        idle_1s()
            .with_max_consecutive_stream_idle_timeouts(Some(2))
            .with_stream_first_event_timeout_ms(Some(1_000))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("scrubbed-reset"))
        .await
        .expect_err("the breaker must eventually trip");

    assert!(
        matches!(err, TinyAgentsError::StreamIdleTimeout(_)),
        "{err:?}"
    );
    assert_eq!(model.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn empty_deltas_do_not_extend_the_idle_deadline() {
    // After the first token the provider sends only empty deltas, 900ms apart.
    let mut script = vec![started(), delta("tok")];
    for _ in 0..5 {
        script.push(Step::Sleep(Duration::from_millis(900)));
        script.push(delta(""));
    }
    script.push(Step::Hang);
    let model = ScriptedStreams::new(vec![script]);
    let harness = harness_with(model.clone(), idle_1s(), 1);

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("empty-trickle"))
        .await
        .expect_err("empty deltas must not keep a stalled stream alive");

    assert!(matches!(err, TinyAgentsError::CallTimeout(_)), "{err:?}");
    assert_eq!(began.elapsed(), SECOND);
}

#[tokio::test(start_paused = true)]
async fn block_deltas_rearm_the_idle_deadline_but_empty_ones_do_not() {
    let block = |text: &str| {
        Step::Item(Box::new(ModelStreamItem::BlockDelta {
            index: 0,
            delta: BlockDelta::Text(text.to_string()),
        }))
    };
    // Non-empty block deltas 800ms apart keep a 1s window alive...
    let mut alive = vec![started()];
    for _ in 0..5 {
        alive.push(Step::Sleep(Duration::from_millis(800)));
        alive.push(block("x"));
    }
    alive.push(completed("xxxxx"));
    let model = ScriptedStreams::new(vec![alive]);
    let harness = harness_with(
        model.clone(),
        idle_1s().with_stream_first_event_timeout_ms(Some(1_000)),
        1,
    );
    run(&harness, RunConfig::new("block-alive"))
        .await
        .expect("non-empty block deltas are progress");

    // ...while empty ones do not.
    let mut stalled = vec![started(), block("x")];
    for _ in 0..5 {
        stalled.push(Step::Sleep(Duration::from_millis(800)));
        stalled.push(block(""));
    }
    stalled.push(Step::Hang);
    let model = ScriptedStreams::new(vec![stalled]);
    let harness = harness_with(model.clone(), idle_1s(), 1);
    let err = run(&harness, RunConfig::new("block-empty"))
        .await
        .expect_err("empty block deltas are not progress");
    assert!(matches!(err, TinyAgentsError::CallTimeout(_)), "{err:?}");
}

#[tokio::test(start_paused = true)]
async fn first_event_window_bounds_a_stream_that_hangs_while_opening() {
    let model = ScriptedStreams::hanging_open();
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_first_event_timeout_ms(Some(3_000)),
        1,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("hang-on-open"))
        .await
        .expect_err("a stream that never opens must hit the first-event window");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("first output event"), "{message}");
            assert!(message.contains("3000 ms"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    assert_eq!(model.calls(), 1);
    assert_eq!(began.elapsed(), 3 * SECOND);
}

/// Blanks every delta's text and reasoning (a redaction policy), and gives an
/// empty delta some text of its own (a middleware that injects content).
struct BlankOrInject;

#[async_trait]
impl crate::middleware::Middleware<(), ()> for BlankOrInject {
    fn name(&self) -> &str {
        "blank-or-inject"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> crate::error::Result<()> {
        if delta.content.is_empty() && delta.reasoning.is_empty() {
            delta.content = "injected".to_string();
        } else {
            delta.content.clear();
            delta.reasoning.clear();
        }
        Ok(())
    }
}

#[tokio::test(start_paused = true)]
async fn redacted_provider_output_still_counts_as_progress() {
    // The provider keeps sending real text 800ms apart; the middleware blanks
    // all of it, so no consumer sees anything, but the provider is not idle.
    let model = ScriptedStreams::new(vec![slow_chunks(vec![started()], &["a", "b", "c", "d"])]);
    let mut harness = harness_with(
        model.clone(),
        idle_1s().with_stream_first_event_timeout_ms(Some(1_000)),
        1,
    );
    harness.push_middleware(Arc::new(BlankOrInject));

    run(&harness, RunConfig::new("redacted-progress"))
        .await
        .expect("a provider streaming redacted text is not idle");
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn text_injected_by_middleware_into_an_empty_delta_is_not_progress() {
    // Empty provider deltas arrive 900ms apart; the middleware turns each into
    // visible text, which is the middleware's output, not the provider's.
    let mut script = vec![started(), delta("tok")];
    for _ in 0..5 {
        script.push(Step::Sleep(Duration::from_millis(900)));
        script.push(delta(""));
    }
    script.push(Step::Hang);
    let model = ScriptedStreams::new(vec![script]);
    let mut harness = harness_with(model.clone(), idle_1s(), 1);
    harness.push_middleware(Arc::new(BlankOrInject));

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("injected-progress"))
        .await
        .expect_err("injected text must not keep a stalled stream alive");
    assert!(matches!(err, TinyAgentsError::CallTimeout(_)), "{err:?}");
    assert_eq!(began.elapsed(), SECOND);
}
