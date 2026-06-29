//! Type definitions for the harness testkit.
//!
//! Every public struct and enum in `crate::harness::testkit` is declared here.
//! Implementations, constructors, and trait impls live in `mod.rs`; focused
//! tests live in `test.rs`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::harness::events::{AgentEvent, EventSink, RecordingListener};
use crate::harness::model::{ModelRequest, ModelResponse, ModelStreamItem};
use crate::harness::tool::ToolCall;

// ---------------------------------------------------------------------------
// StreamingMock
// ---------------------------------------------------------------------------

/// A [`crate::harness::model::ChatModel`] that yields a scripted sequence of
/// [`ModelStreamItem`]s, exercising the real streaming pipeline deterministically.
///
/// Each call to [`crate::harness::model::ChatModel::stream`] replays the same
/// scripted items, and [`crate::harness::model::ChatModel::invoke`] returns the
/// merged response those items fold into (via
/// [`crate::harness::model::StreamAccumulator`]), so the mock behaves
/// consistently on both the streaming and unary paths.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::StreamingMock;
/// // Streams "Hello, world" as three message deltas plus a merged completion.
/// let model = StreamingMock::from_text_chunks(["Hello", ", ", "world"]);
/// ```
#[allow(dead_code)]
pub struct StreamingMock {
    /// The scripted items replayed on every `stream` call.
    pub(crate) items: Vec<ModelStreamItem>,
    /// Number of `stream`/`invoke` calls made so far.
    pub(crate) calls: Mutex<u64>,
}

// ---------------------------------------------------------------------------
// SlowModel
// ---------------------------------------------------------------------------

/// A [`crate::harness::model::ChatModel`] that sleeps for a fixed delay before
/// replying, used to deterministically trigger the agent loop's per-model-call
/// wall-clock timeout.
///
/// Both [`crate::harness::model::ChatModel::invoke`] and
/// [`crate::harness::model::ChatModel::stream`] first
/// `tokio::time::sleep(delay).await` and then return a fixed assistant reply.
/// (The `stream` path inherits the delay because the default trait
/// implementation delegates to `invoke`.) Configure the run with a wall-clock
/// timeout much smaller than `delay` (for example a 20&nbsp;ms timeout against a
/// 200&nbsp;ms delay) so the call is reliably interrupted with
/// [`crate::error::TinyAgentsError::Timeout`].
///
/// # Example
///
/// ```rust
/// # use std::time::Duration;
/// # use tinyagents::harness::testkit::SlowModel;
/// // Sleeps 200ms before echoing a fixed reply.
/// let model = SlowModel::new(Duration::from_millis(200), "slow reply");
/// ```
pub struct SlowModel {
    /// How long [`crate::harness::model::ChatModel::invoke`] sleeps before
    /// replying.
    pub(crate) delay: Duration,
    /// The fixed assistant text returned after the delay.
    pub(crate) reply: String,
    /// Number of `invoke` calls made so far.
    pub(crate) calls: Mutex<u64>,
}

// ---------------------------------------------------------------------------
// ScriptedModel
// ---------------------------------------------------------------------------

/// A [`crate::harness::model::ChatModel`] that returns pre-loaded responses in
/// order, making model behavior fully deterministic in tests.
///
/// Responses are consumed from the front of the queue one per `invoke` call.
/// When the queue is exhausted `invoke` returns
/// [`crate::error::TinyAgentsError::Model`] rather than panicking so tests get
/// a clear error instead of a thread panic.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::ScriptedModel;
/// # use tinyagents::harness::model::ModelResponse;
/// let model = ScriptedModel::replies(vec!["Hello", "World"]);
/// // Use in tests as a ChatModel<()>.
/// ```
pub struct ScriptedModel {
    /// Queued responses returned in FIFO order.
    pub(crate) queue: Mutex<VecDeque<ModelResponse>>,
    /// Every `ModelRequest` received by `invoke`, in call order.
    pub(crate) received: Mutex<Vec<ModelRequest>>,
}

// ---------------------------------------------------------------------------
// FakeTool
// ---------------------------------------------------------------------------

/// The runtime behavior chosen when a [`FakeTool`] is invoked.
pub(crate) enum FakeToolBehavior {
    /// Return a fixed text content string.
    Return(String),
    /// Return `Err(TinyAgentsError::Tool(...))` with the provided message.
    Fail(String),
}

/// A configurable [`crate::harness::tool::Tool`] for testing.
///
/// Created with one of three factory methods:
///
/// - [`FakeTool::new`] — returns an empty string result.
/// - [`FakeTool::returning`] — returns a fixed text content.
/// - [`FakeTool::failing`] — returns a [`crate::error::TinyAgentsError::Tool`] error.
///
/// Every received [`ToolCall`] is recorded and available via
/// [`FakeTool::calls`].
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::FakeTool;
/// let tool = FakeTool::returning("search", "42");
/// // Use as Tool<()> in tests.
/// ```
pub struct FakeTool {
    /// Canonical name returned by [`crate::harness::tool::Tool::name`].
    pub(crate) tool_name: String,
    /// Human-readable description returned by
    /// [`crate::harness::tool::Tool::description`].
    pub(crate) tool_description: String,
    /// What to do when invoked.
    pub(crate) behavior: FakeToolBehavior,
    /// Recorded calls for post-invocation assertions.
    pub(crate) received: Mutex<Vec<ToolCall>>,
}

// ---------------------------------------------------------------------------
// DeterministicClock
// ---------------------------------------------------------------------------

/// A controllable, monotonic clock for deterministic test scenarios.
///
/// Unlike `SystemTime`, `DeterministicClock` never advances on its own. Tests
/// call [`DeterministicClock::advance`] to move time forward in a controlled
/// way, making time-sensitive assertions reproducible.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::DeterministicClock;
/// let clock = DeterministicClock::new(1_000);
/// assert_eq!(clock.now_millis(), 1_000);
/// clock.advance(500);
/// assert_eq!(clock.now_millis(), 1_500);
/// ```
pub struct DeterministicClock {
    /// Current time in milliseconds since an arbitrary epoch.
    pub(crate) millis: Mutex<u64>,
}

// ---------------------------------------------------------------------------
// DeterministicIds
// ---------------------------------------------------------------------------

/// A monotonically incrementing identifier generator for stable test output.
///
/// Produces ids in the form `"{prefix}-0"`, `"{prefix}-1"`, ... so tests can
/// assert exact id values without relying on UUIDs or wall-clock timestamps.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::DeterministicIds;
/// let ids = DeterministicIds::new("call");
/// assert_eq!(ids.next(), "call-0");
/// assert_eq!(ids.next(), "call-1");
/// ```
pub struct DeterministicIds {
    /// Stable prefix prepended to every generated id.
    pub(crate) prefix: String,
    /// Counter incremented on each [`DeterministicIds::next`] call.
    pub(crate) counter: Mutex<u64>,
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

/// Captures [`AgentEvent`]s emitted through an [`EventSink`] for later
/// inspection.
///
/// The recorder owns an internal [`RecordingListener`] subscribed to a shared
/// [`EventSink`]. Callers obtain the sink via [`EventRecorder::sink`] and pass
/// it to the component under test. After the run, [`EventRecorder::events`]
/// and [`EventRecorder::kinds`] provide access to what was emitted.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::EventRecorder;
/// # use tinyagents::harness::events::AgentEvent;
/// # use tinyagents::harness::ids::RunId;
/// let recorder = EventRecorder::new();
/// let sink = recorder.sink();
/// sink.emit(AgentEvent::RunStarted { run_id: RunId::new("r1"), thread_id: None });
/// assert_eq!(recorder.kinds(), vec!["run.started"]);
/// ```
pub struct EventRecorder {
    /// Internal listener that buffers every record.
    pub(crate) listener: Arc<RecordingListener>,
    /// The shared sink the listener is subscribed to.
    pub(crate) sink: EventSink,
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

/// An ordered view of [`AgentEvent`]s with structural assertion helpers.
///
/// `Trajectory` lets tests make deterministic claims about *what happened*
/// during a run (which tools were called, how many model calls occurred, whether
/// the run completed) without depending on exact LLM prose or wall-clock timing.
///
/// ## `assert_*` variants vs predicate methods
///
/// Each feature is exposed both as a predicate (`tool_was_called`) and as an
/// asserting helper (`assert_tool_called`) that panics with a descriptive
/// message on failure, matching Rust's `assert!` / `assert_eq!` ergonomics.
///
/// ## Assertion methods returning `Result`
///
/// [`Trajectory::assert_order`] returns
/// [`crate::error::Result`]`<()>` rather than panicking so callers can
/// propagate the error or inspect the message programmatically.
///
/// # Example
///
/// ```rust
/// # use tinyagents::harness::testkit::Trajectory;
/// # use tinyagents::harness::events::AgentEvent;
/// # use tinyagents::harness::ids::{RunId, CallId};
/// let events = vec![
///     AgentEvent::RunStarted { run_id: RunId::new("r1"), thread_id: None },
///     AgentEvent::ModelStarted { call_id: CallId::new("c1"), model: "gpt".into() },
///     AgentEvent::ModelCompleted { call_id: CallId::new("c1"), usage: None },
///     AgentEvent::RunCompleted { run_id: RunId::new("r1") },
/// ];
/// let traj = Trajectory::from_events(events);
/// assert_eq!(traj.model_call_count(), 1);
/// traj.assert_completed();
/// ```
pub struct Trajectory {
    /// The ordered sequence of events that make up this trajectory.
    pub(crate) events: Vec<AgentEvent>,
}
