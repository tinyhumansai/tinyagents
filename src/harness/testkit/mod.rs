//! Test-support toolkit for the harness.
//!
//! In the recursive architecture this is how nested, model-driven behaviour is
//! made *deterministically testable*: scripted/streaming model doubles, fake
//! tools, controllable clocks/ids, and a [`Trajectory`] over recorded
//! [`AgentEvent`]s let tests assert exactly what an agent — and the sub-agents
//! and sub-graphs it spawns — did, all without a live provider. The same
//! [`EventRecorder`] that observes a top-level run also captures child-run
//! events fanned onto a shared sink, so recursion is observable in tests.
//!
//! Provides deterministic doubles and trajectory assertions that make it
//! possible to test model-and-tool workflows without live providers.
//!
//! # Contents
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`ScriptedModel`] | Pre-loaded `ChatModel` returning queued responses |
//! | [`SlowModel`] | `ChatModel` that sleeps before replying (timeout testing) |
//! | [`FakeTool`] | Configurable `Tool` recording invocations |
//! | [`DeterministicClock`] | Controllable millisecond clock |
//! | [`DeterministicIds`] | Monotonic `"{prefix}-N"` id generator |
//! | [`EventRecorder`] | Captures `AgentEvent`s from an `EventSink` |
//! | [`Trajectory`] | Structural assertions over a sequence of events |

mod types;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::error::{Result, TinyAgentsError};
use crate::harness::events::{AgentEvent, EventSink, RecordingListener};
use crate::harness::message::MessageDelta;
use crate::harness::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem, StreamAccumulator,
};
use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

pub use types::*;

// ---------------------------------------------------------------------------
// StreamingMock
// ---------------------------------------------------------------------------

impl StreamingMock {
    /// Creates a streaming mock that replays the given scripted items verbatim.
    ///
    /// The items should follow the streaming contract: a leading
    /// [`ModelStreamItem::Started`], any number of delta items, and a terminal
    /// [`ModelStreamItem::Completed`] or [`ModelStreamItem::Failed`].
    pub fn new(items: Vec<ModelStreamItem>) -> Self {
        Self {
            items,
            calls: Mutex::new(0),
        }
    }

    /// Builds a streaming mock from text chunks.
    ///
    /// Produces a [`ModelStreamItem::Started`], one
    /// [`ModelStreamItem::MessageDelta`] per chunk, and a terminal
    /// [`ModelStreamItem::Completed`] carrying the concatenated text as the
    /// merged assistant response.
    pub fn from_text_chunks<S: AsRef<str>>(chunks: impl IntoIterator<Item = S>) -> Self {
        let mut items = vec![ModelStreamItem::Started];
        let mut full = String::new();
        for chunk in chunks {
            let text = chunk.as_ref().to_string();
            full.push_str(&text);
            items.push(ModelStreamItem::MessageDelta(MessageDelta {
                text,
                tool_call: None,
            }));
        }
        items.push(ModelStreamItem::Completed(ModelResponse::assistant(full)));
        Self::new(items)
    }

    /// Returns the number of `stream`/`invoke` calls made so far.
    pub fn call_count(&self) -> u64 {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned")
    }

    /// Folds the scripted items into the response they merge to.
    fn merged_response(&self) -> Result<ModelResponse> {
        let mut accumulator = StreamAccumulator::new();
        for item in &self.items {
            accumulator.push(item);
        }
        accumulator.finish()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for StreamingMock {
    /// Returns the merged response the scripted stream folds into.
    async fn invoke(&self, _state: &State, _request: ModelRequest) -> Result<ModelResponse> {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned") += 1;
        self.merged_response()
    }

    /// Replays the scripted items as a real [`ModelStream`].
    async fn stream(&self, _state: &State, _request: ModelRequest) -> Result<ModelStream> {
        *self
            .calls
            .lock()
            .expect("StreamingMock calls lock poisoned") += 1;
        let items = self.items.clone();
        Ok(Box::pin(futures::stream::iter(items)))
    }
}

// ---------------------------------------------------------------------------
// SlowModel
// ---------------------------------------------------------------------------

impl SlowModel {
    /// Creates a slow model that sleeps `delay` before returning `reply`.
    pub fn new(delay: Duration, reply: impl Into<String>) -> Self {
        Self {
            delay,
            reply: reply.into(),
            calls: Mutex::new(0),
        }
    }

    /// Returns the number of `invoke` calls made so far.
    pub fn call_count(&self) -> u64 {
        *self.calls.lock().expect("SlowModel calls lock poisoned")
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for SlowModel {
    /// Sleeps for the configured delay, then returns the fixed reply.
    async fn invoke(&self, _state: &State, _request: ModelRequest) -> Result<ModelResponse> {
        {
            // Bump the counter in its own scope so the guard is dropped before
            // the `.await` (a `MutexGuard` is not `Send`).
            *self.calls.lock().expect("SlowModel calls lock poisoned") += 1;
        }
        tokio::time::sleep(self.delay).await;
        Ok(ModelResponse::assistant(self.reply.clone()))
    }
}

// ---------------------------------------------------------------------------
// ScriptedModel
// ---------------------------------------------------------------------------

impl ScriptedModel {
    /// Creates a scripted model that will return `responses` in order.
    ///
    /// The first element is returned on the first `invoke`, the second on the
    /// second call, and so on. When the queue is drained, subsequent calls
    /// return [`TinyAgentsError::Model`].
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            queue: Mutex::new(VecDeque::from(responses)),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a scripted model from a list of plain text replies.
    ///
    /// Each string in `texts` becomes a [`ModelResponse::assistant`] wrapping
    /// that text. This is the most concise constructor for text-only tests.
    pub fn replies<S: AsRef<str>>(texts: Vec<S>) -> Self {
        let responses = texts
            .into_iter()
            .map(|t| ModelResponse::assistant(t.as_ref()))
            .collect();
        Self::new(responses)
    }

    /// Returns a snapshot of every [`ModelRequest`] received by `invoke`, in
    /// call order.
    ///
    /// Use this to assert on the exact messages, tools, or parameters passed to
    /// the model by the component under test.
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.received
            .lock()
            .expect("ScriptedModel received lock poisoned")
            .clone()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ScriptedModel {
    /// Pops the next response from the queue and records the received request.
    ///
    /// Returns [`TinyAgentsError::Model`] when the queue is exhausted so the
    /// test gets a clear message rather than a thread panic.
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        self.received
            .lock()
            .expect("ScriptedModel received lock poisoned")
            .push(request);

        self.queue
            .lock()
            .expect("ScriptedModel queue lock poisoned")
            .pop_front()
            .ok_or_else(|| {
                TinyAgentsError::Model(
                    "ScriptedModel: response queue is exhausted; no more scripted responses"
                        .to_string(),
                )
            })
    }
}

// ---------------------------------------------------------------------------
// FakeTool
// ---------------------------------------------------------------------------

impl FakeTool {
    /// Creates a `FakeTool` with the given name that returns an empty string
    /// result on every invocation.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Return(String::new()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a `FakeTool` that returns `content` as plain text on every
    /// successful invocation.
    pub fn returning(name: impl Into<String>, content: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Return(content.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Creates a `FakeTool` that always returns
    /// `Err(`[`TinyAgentsError::Tool`]`(message))`.
    pub fn failing(name: impl Into<String>, message: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            tool_description: format!("Fake tool: {name}"),
            tool_name: name,
            behavior: FakeToolBehavior::Fail(message.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// Returns a snapshot of every [`ToolCall`] received by this tool, in
    /// invocation order.
    pub fn calls(&self) -> Vec<ToolCall> {
        self.received
            .lock()
            .expect("FakeTool received lock poisoned")
            .clone()
    }
}

#[async_trait]
impl<State: Send + Sync> Tool<State> for FakeTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    /// Returns a minimal schema advertising no required parameters.
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.tool_name.clone(),
            self.tool_description.clone(),
            json!({ "type": "object", "properties": {}, "required": [] }),
        )
    }

    /// Records the call and then either returns a fixed result or an error,
    /// depending on how the tool was constructed.
    async fn call(&self, _state: &State, call: ToolCall) -> Result<ToolResult> {
        self.received
            .lock()
            .expect("FakeTool received lock poisoned")
            .push(call.clone());

        match &self.behavior {
            FakeToolBehavior::Return(content) => {
                Ok(ToolResult::text(call.id, call.name, content.clone()))
            }
            FakeToolBehavior::Fail(message) => Err(TinyAgentsError::Tool(message.clone())),
        }
    }
}

// ---------------------------------------------------------------------------
// DeterministicClock
// ---------------------------------------------------------------------------

impl DeterministicClock {
    /// Creates a new clock starting at `start_millis` milliseconds.
    pub fn new(start_millis: u64) -> Self {
        Self {
            millis: Mutex::new(start_millis),
        }
    }

    /// Returns the current clock time in milliseconds.
    pub fn now_millis(&self) -> u64 {
        *self
            .millis
            .lock()
            .expect("DeterministicClock lock poisoned")
    }

    /// Advances the clock forward by `ms` milliseconds.
    ///
    /// The clock never advances on its own; this is the only way to move it
    /// forward, keeping test timing fully deterministic.
    pub fn advance(&self, ms: u64) {
        *self
            .millis
            .lock()
            .expect("DeterministicClock lock poisoned") += ms;
    }
}

impl Default for DeterministicClock {
    /// Creates a clock starting at epoch zero (0 ms).
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// DeterministicIds
// ---------------------------------------------------------------------------

impl DeterministicIds {
    /// Creates a new generator with the given `prefix`.
    ///
    /// The first call to [`DeterministicIds::next`] returns `"{prefix}-0"`, the
    /// second returns `"{prefix}-1"`, and so on.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: Mutex::new(0),
        }
    }

    /// Returns the next id in the sequence and increments the internal counter.
    pub fn next(&self) -> String {
        let mut counter = self.counter.lock().expect("DeterministicIds lock poisoned");
        let id = format!("{}-{}", self.prefix, *counter);
        *counter += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// EventRecorder
// ---------------------------------------------------------------------------

impl EventRecorder {
    /// Creates a new recorder with an empty buffer.
    ///
    /// The internal [`RecordingListener`] is subscribed to the internal
    /// [`EventSink`] immediately; callers only need to obtain the sink via
    /// [`EventRecorder::sink`] and pass it to the component under test.
    pub fn new() -> Self {
        let listener = Arc::new(RecordingListener::new());
        let sink = EventSink::new();
        sink.subscribe(listener.clone());
        Self { listener, sink }
    }

    /// Returns a clone of the internal [`EventSink`] that the recorder is
    /// listening to.
    ///
    /// Pass this sink to the component under test so its emitted events are
    /// captured.
    pub fn sink(&self) -> EventSink {
        self.sink.clone()
    }

    /// Returns a snapshot of the raw [`AgentEvent`] payloads captured so far,
    /// in arrival order.
    pub fn events(&self) -> Vec<AgentEvent> {
        self.listener
            .events()
            .into_iter()
            .map(|r| r.event)
            .collect()
    }

    /// Returns the `kind()` string for each captured event, in arrival order.
    ///
    /// Useful for quick assertions like:
    ///
    /// ```rust
    /// # use tinyagents::harness::testkit::EventRecorder;
    /// # use tinyagents::harness::events::AgentEvent;
    /// # use tinyagents::harness::ids::RunId;
    /// let recorder = EventRecorder::new();
    /// recorder.sink().emit(AgentEvent::RunStarted {
    ///     run_id: RunId::new("r1"),
    ///     thread_id: None,
    /// });
    /// assert_eq!(recorder.kinds(), vec!["run.started"]);
    /// ```
    pub fn kinds(&self) -> Vec<String> {
        self.listener
            .events()
            .into_iter()
            .map(|r| r.event.kind().to_string())
            .collect()
    }
}

impl Default for EventRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

impl Trajectory {
    /// Constructs a `Trajectory` from an owned sequence of [`AgentEvent`]s.
    pub fn from_events(events: Vec<AgentEvent>) -> Self {
        Self { events }
    }

    // ── Tool assertions ──────────────────────────────────────────────────────

    /// Returns `true` when at least one [`AgentEvent::ToolStarted`] with the
    /// given `name` is present in the trajectory.
    pub fn tool_was_called(&self, name: &str) -> bool {
        self.tool_call_count(name) > 0
    }

    /// Panics with a descriptive message when the named tool was not called.
    ///
    /// Use in tests for ergonomic assertions:
    ///
    /// ```rust
    /// # use tinyagents::harness::testkit::Trajectory;
    /// # use tinyagents::harness::events::AgentEvent;
    /// # use tinyagents::harness::ids::CallId;
    /// let events = vec![
    ///     AgentEvent::ToolStarted {
    ///         call_id: CallId::new("c1"),
    ///         tool_name: "search".into(),
    ///     },
    /// ];
    /// Trajectory::from_events(events).assert_tool_called("search");
    /// ```
    pub fn assert_tool_called(&self, name: &str) {
        assert!(
            self.tool_was_called(name),
            "Trajectory: expected tool '{name}' to have been called, but it was not found in the \
             event sequence"
        );
    }

    /// Returns the number of times [`AgentEvent::ToolStarted`] with the given
    /// `name` appears in the trajectory.
    pub fn tool_call_count(&self, name: &str) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolStarted { tool_name, .. } if tool_name == name))
            .count()
    }

    // ── Model assertions ─────────────────────────────────────────────────────

    /// Returns the number of [`AgentEvent::ModelStarted`] events in the
    /// trajectory.
    pub fn model_call_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ModelStarted { .. }))
            .count()
    }

    /// Panics when the number of model calls does not equal `n`.
    pub fn assert_model_called_times(&self, n: usize) {
        let actual = self.model_call_count();
        assert_eq!(
            actual, n,
            "Trajectory: expected {n} model call(s) but found {actual}"
        );
    }

    // ── Ordering assertions ──────────────────────────────────────────────────

    /// Asserts that `labels` appear as a subsequence of the trajectory events.
    ///
    /// Each label is matched against the events in order. A label matches the
    /// first unmatched event for which *either*:
    ///
    /// - the event's [`AgentEvent::kind()`] equals the label (e.g.
    ///   `"tool.started"`, `"model.completed"`), **or**
    /// - the event is a `ToolStarted` or `ToolCompleted` whose `tool_name`
    ///   equals the label.
    ///
    /// The check is a *subsequence* match: there may be other events between
    /// the matched ones.
    ///
    /// Returns [`TinyAgentsError::Validation`] with a descriptive message on
    /// failure.
    pub fn assert_order(&self, labels: &[&str]) -> Result<()> {
        let mut event_iter = self.events.iter();
        for &label in labels {
            let found = event_iter.any(|e| Self::event_matches_label(e, label));
            if !found {
                return Err(TinyAgentsError::Validation(format!(
                    "Trajectory: expected label '{label}' in order but it was not found after the \
                     previous matched label"
                )));
            }
        }
        Ok(())
    }

    /// Returns `true` when the trajectory contains a [`AgentEvent::RunCompleted`]
    /// event.
    pub fn completed(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunCompleted { .. }))
    }

    /// Panics when the trajectory does not contain a `RunCompleted` event.
    pub fn assert_completed(&self) {
        assert!(
            self.completed(),
            "Trajectory: expected RunCompleted event but none was found"
        );
    }

    /// Returns `true` when the trajectory contains at least one
    /// [`AgentEvent::RunFailed`] event.
    pub fn failed(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunFailed { .. }))
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Returns `true` when `event` should be counted as a match for `label`.
    ///
    /// Matches on the event kind string (e.g. `"tool.started"`) **or** on the
    /// `tool_name` field of `ToolStarted`/`ToolCompleted` events.
    fn event_matches_label(event: &AgentEvent, label: &str) -> bool {
        if event.kind() == label {
            return true;
        }
        match event {
            AgentEvent::ToolStarted { tool_name, .. }
            | AgentEvent::ToolCompleted { tool_name, .. } => tool_name == label,
            AgentEvent::RouteSelected { route } => route == label,
            _ => false,
        }
    }
}

#[cfg(test)]
mod test;
