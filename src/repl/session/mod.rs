//! Rhai-backed `.ragsh` session runtime (design milestone R2).
//!
//! [`ReplSession`] is the imperative counterpart to the declarative `.rag`
//! language: an orchestrator (a human, or a model acting as one) drives a
//! session one *cell* at a time. Each cell is a small Rhai script evaluated
//! against a **persistent namespace** — top-level `let` bindings survive into
//! the next cell, exactly like the persistent locals of a Recursive Language
//! Model REPL — while model, tool, and graph capabilities are exposed as
//! host-registered functions rather than script-native side effects.
//!
//! This module implements the runtime core of milestone R2:
//!
//! - an [`rhai::Engine`] configured with
//!   [`set_max_operations`](rhai::Engine::set_max_operations) so a runaway
//!   script *fails closed* instead of hanging the host;
//! - a persistent [`rhai::Scope`] shared across cells;
//! - captured `print` output, returned values, and changed variables;
//! - `emit(...)` and `answer(...)` built-ins recorded as typed data;
//! - byte limits on both script input and captured output;
//! - restoration of reserved core names after every cell so a script can add
//!   locals but cannot permanently replace `context`, `answer`, `model_query`,
//!   `graph_run`, or any other reserved capability.
//!
//! Capability functions (`model_query`, `tool_call`, `graph_run`, the
//! `graph_*` blueprint surface, …) are wired to the real registries by later
//! slices; this slice establishes the session, policy, and result types they
//! plug into. Generated graph topology is never installed directly — it must
//! pass through the `.rag` compiler, the capability resolver, and the policy
//! review gate.
//!
//! The whole module is gated behind the `repl` cargo feature so the default
//! build does not pull in the Rhai engine.

mod builtins;
mod types;

#[cfg(test)]
mod test;

pub use types::*;

use builtins::{CallCounters, HostContext};

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rhai::{Dynamic, Engine, EvalAltResult, Scope};

use crate::error::{Result, TinyAgentsError};
use crate::harness::context::RunContext;
use crate::harness::events::EventSink;
use crate::harness::ids::{SessionId, new_session_id};

/// Shared host-side buffers the registered Rhai built-ins write into.
///
/// Cloned (cheaply, via `Arc`) into the engine's `on_print`, `emit`,
/// `answer`, and capability closures at construction, and read back by
/// [`ReplSession::eval_cell`] after each cell.
///
/// `host_error` lets a fallible capability function (which can only surface a
/// stringly-typed [`rhai::EvalAltResult`] across the engine boundary) stash the
/// precise [`TinyAgentsError`] it failed with; [`ReplSession::eval_cell`]
/// prefers that error over the generic Rhai runtime error so callers see the
/// real diagnostic (`ModelNotFound`, `LimitExceeded`, …). `vars_snapshot` holds
/// the persistent namespace as of the start of the current cell so the
/// `show_vars()` built-in can print it.
#[derive(Clone, Default)]
pub(super) struct CellBuffers {
    stdout: Arc<Mutex<String>>,
    calls: Arc<Mutex<Vec<ReplCallRecord>>>,
    answer: Arc<Mutex<Option<String>>>,
    host_error: Arc<Mutex<Option<TinyAgentsError>>>,
    vars_snapshot: Arc<Mutex<BTreeMap<String, String>>>,
}

/// The persistent variable namespace of a session.
///
/// Wraps the Rhai [`Scope`] that survives across cells together with the
/// baseline values of the reserved names, which are restored after each cell.
pub struct ReplVariables {
    scope: Scope<'static>,
    reserved_baseline: BTreeMap<String, Dynamic>,
}

impl ReplVariables {
    /// Seeds a fresh namespace with the reserved built-in variables set to unit.
    fn seeded() -> Self {
        let mut scope = Scope::new();
        let mut reserved_baseline = BTreeMap::new();
        for name in reserved_names() {
            let value = Dynamic::UNIT;
            scope.push(name.to_string(), value.clone());
            reserved_baseline.insert(name.to_string(), value);
        }
        Self {
            scope,
            reserved_baseline,
        }
    }

    /// Sets a persistent (non-reserved) variable from a [`ReplValue`].
    ///
    /// Reserved names are rejected so callers cannot smuggle a capability
    /// replacement through the variable surface; use [`ReplSession::set_context`]
    /// and friends for the reserved data slots.
    pub fn set(&mut self, name: impl Into<String>, value: ReplValue) -> Result<()> {
        let name = name.into();
        if reserved_names().any(|r| name == r) {
            return Err(TinyAgentsError::Capability(format!(
                "`{name}` is a reserved REPL name and cannot be set as a variable"
            )));
        }
        self.scope.set_value(name, repl_value_to_dynamic(&value));
        Ok(())
    }

    /// Returns the current value of a variable, if present.
    pub fn get(&self, name: &str) -> Option<ReplValue> {
        self.scope
            .get_value::<Dynamic>(name)
            .map(|d| dynamic_to_repl_value(&d))
    }

    /// Overwrites a reserved data slot's baseline and current value.
    fn set_reserved(&mut self, name: &str, value: Dynamic) {
        self.scope.set_value(name.to_string(), value.clone());
        self.reserved_baseline.insert(name.to_string(), value);
    }

    /// Snapshots the current `name -> debug-string` view of the scope, used to
    /// detect which variables a cell changed.
    fn snapshot(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for (name, _is_const, value) in self.scope.iter() {
            map.insert(name.to_string(), format!("{value:?}"));
        }
        map
    }

    /// Restores every reserved name to its session baseline, discarding any
    /// script-level reassignment or shadowing from the cell just evaluated.
    fn restore_reserved(&mut self) {
        for (name, value) in &self.reserved_baseline {
            self.scope.set_value(name.clone(), value.clone());
        }
    }
}

impl Default for ReplVariables {
    fn default() -> Self {
        Self::seeded()
    }
}

/// An interactive Rhai-backed `.ragsh` session.
///
/// See the [module docs](self) for the runtime model. Construct a default,
/// stateless session with [`ReplSession::new`]; supply registries, a custom
/// policy, or a run context with [`ReplSession::builder`]-style `with_*`
/// methods.
///
/// # Not to be confused with `repl::ReplSession`
///
/// This crate has two distinct types named `ReplSession`:
///
/// - **This type** (`repl::session::ReplSession`, feature `repl` only) — the
///   Rhai-backed scripting session described above; also reachable as
///   `crate::ReplSession` (the crate-root re-export) when the `repl` feature
///   is enabled.
/// - [`crate::repl::ReplSession`] (always available, no feature required) —
///   the line-oriented command skeleton (verbs like `set`/`get`/`run`/`call`
///   parsed from a single line). It is *not* re-exported at the crate root
///   under this feature, so `crate::ReplSession` only ever means this
///   scripting session once `repl` is enabled.
///
/// The two are unrelated types serving different layers of the `.ragsh`
/// design; always check which module path (`crate::repl::session::ReplSession`
/// vs. `crate::repl::ReplSession`) you imported from.
pub struct ReplSession<State = (), Ctx = ()>
where
    State: Send + Sync,
{
    /// Unique id for this session.
    pub session_id: SessionId,
    /// The harness run context this session executes within.
    pub run_context: RunContext<Ctx>,
    /// The persistent variable namespace.
    pub variables: ReplVariables,
    /// The named capabilities this session may bind against.
    pub capabilities: ReplCapabilities<State>,
    /// The resource limits bounding this session.
    pub policy: ReplPolicy,
    /// The event sink REPL events are emitted on.
    pub events: EventSink,
    /// The application state capability calls (`model_query`, `tool_call`,
    /// `agent_query`, …) are invoked against. For a stateless session this is
    /// `Arc::new(())`. Distinct from the reserved Rhai `state` *variable*, which
    /// is a script-visible data slot.
    state: Arc<State>,
    /// Session-cumulative capability-call counters, enforced against the
    /// `max_*_calls` policy limits. Shared with the engine's capability
    /// closures and persisted across cells.
    counters: Arc<Mutex<CallCounters>>,
    /// Graph blueprints drafted by `graph_define` in this session, keyed by
    /// graph name. Persisted across cells so a graph defined in one cell can be
    /// validated, compiled, diffed, or registered in another. The actual
    /// topology is never installed here — it stays a draft until it passes the
    /// `.rag` compiler, the capability resolver, and the policy review gate.
    drafts: Arc<Mutex<BTreeMap<String, GraphBlueprintHandle>>>,
    /// The configured Rhai engine. Private: its registered functions are the
    /// capability boundary and must not be mutated by callers.
    engine: Engine,
    /// Shared buffers the engine's built-ins write into.
    buffers: CellBuffers,
}

impl<State: Send + Sync + Default + 'static> ReplSession<State, ()> {
    /// Creates a default, stateless session with empty capabilities and the
    /// default [`ReplPolicy`].
    pub fn new() -> Self {
        Self::from_parts(
            ReplCapabilities::default(),
            ReplPolicy::default(),
            RunContext::new(
                crate::harness::context::RunConfig::new(format!(
                    "repl-run-{}",
                    crate::harness::ids::next_seq()
                )),
                (),
            ),
        )
    }
}

impl<State: Send + Sync + Default + 'static> Default for ReplSession<State, ()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State: Send + Sync + Default + 'static, Ctx> ReplSession<State, Ctx> {
    /// Assembles a session from its capabilities, policy, and run context, with
    /// a default application state.
    ///
    /// The session id is generated from the crate's monotonic id source (no
    /// wall-clock time or randomness), and the session's [`EventSink`] is shared
    /// with the run context so REPL events compose with harness events. Supply a
    /// non-default application state with [`with_state`](Self::with_state).
    pub fn from_parts(
        capabilities: ReplCapabilities<State>,
        policy: ReplPolicy,
        run_context: RunContext<Ctx>,
    ) -> Self {
        let buffers = CellBuffers::default();
        let events = run_context.events.clone();
        let mut session = Self {
            session_id: new_session_id(),
            run_context,
            variables: ReplVariables::seeded(),
            capabilities,
            policy,
            events,
            state: Arc::new(State::default()),
            counters: Arc::new(Mutex::new(CallCounters::default())),
            drafts: Arc::new(Mutex::new(BTreeMap::new())),
            engine: Engine::new(),
            buffers,
        };
        session.rebuild_engine();
        session
    }
}

impl<State: Send + Sync + 'static, Ctx> ReplSession<State, Ctx> {
    /// (Re)builds the sandboxed Rhai engine from the session's current policy,
    /// capabilities, and application state, registering every host-backed
    /// built-in function against the live registries. Called after any change to
    /// policy, capabilities, or state.
    fn rebuild_engine(&mut self) {
        let ctx = Arc::new(HostContext {
            registry: self.capabilities.registry.clone(),
            state: self.state.clone(),
            policy: self.policy.clone(),
            language: self.capabilities.language.clone(),
            session_label: self.session_id.as_str().to_string(),
            run_depth: self.run_context.config.depth,
            events: self.events.clone(),
            buffers: self.buffers.clone(),
            counters: self.counters.clone(),
            drafts: self.drafts.clone(),
        });
        self.engine = builtins::build_engine(ctx);
    }

    /// Replaces the session policy and rebuilds the engine to honor the new
    /// operation and call limits.
    pub fn with_policy(mut self, policy: ReplPolicy) -> Self {
        self.policy = policy;
        self.rebuild_engine();
        self
    }

    /// Replaces the session capabilities and rebuilds the engine so the
    /// capability functions resolve against the new registries.
    pub fn with_capabilities(mut self, capabilities: ReplCapabilities<State>) -> Self {
        self.capabilities = capabilities;
        self.rebuild_engine();
        self
    }

    /// Replaces the application state capability calls are invoked against and
    /// rebuilds the engine.
    pub fn with_state(mut self, state: Arc<State>) -> Self {
        self.state = state;
        self.rebuild_engine();
        self
    }

    /// Returns a shared handle to the application state capability calls are
    /// invoked against.
    ///
    /// The CodeAct driver ([`crate::repl::codeact`]) needs this to invoke the
    /// session's driver model through the same state the in-cell capability
    /// functions use, without exposing the private field.
    pub fn app_state(&self) -> Arc<State> {
        self.state.clone()
    }

    /// Sets the reserved `context` variable.
    pub fn set_context(&mut self, value: ReplValue) {
        self.variables
            .set_reserved("context", repl_value_to_dynamic(&value));
    }

    /// Sets the reserved `state` variable.
    pub fn set_state_var(&mut self, value: ReplValue) {
        self.variables
            .set_reserved("state", repl_value_to_dynamic(&value));
    }

    /// Evaluates a single `.ragsh` cell against the persistent namespace.
    ///
    /// Captures stdout, the cell's return value, the persistent variables it
    /// changed, recorded `emit`/`answer` calls, and elapsed time. Reserved core
    /// names are restored afterward so the next cell starts from a clean
    /// capability baseline.
    ///
    /// # Errors
    ///
    /// * [`TinyAgentsError::LimitExceeded`] — the script exceeds
    ///   [`ReplPolicy::max_script_bytes`], the output exceeds
    ///   [`ReplPolicy::max_output_bytes`], or the script exceeds the engine
    ///   operation limit (fail-closed runaway protection).
    /// * [`TinyAgentsError::Validation`] — the script failed to compile or
    ///   raised a runtime error.
    pub fn eval_cell(&mut self, script: &str) -> Result<ReplResult> {
        let start = Instant::now();

        if script.len() > self.policy.max_script_bytes {
            return Err(TinyAgentsError::LimitExceeded(format!(
                "ragsh cell is {} bytes, exceeding the max_script_bytes limit of {}",
                script.len(),
                self.policy.max_script_bytes
            )));
        }

        // Reset per-cell shared buffers.
        self.buffers.reset();

        let before = self.variables.snapshot();
        // Expose the pre-cell namespace to `show_vars()`.
        *self
            .buffers
            .vars_snapshot
            .lock()
            .expect("vars_snapshot poisoned") = before.clone();

        // Disjoint field borrows: the engine is read-only while the scope is
        // mutated in place, so top-level `let` bindings persist into the scope.
        let eval = self
            .engine
            .eval_with_scope::<Dynamic>(&mut self.variables.scope, script);

        // Always restore reserved names, even on error, so a failed cell cannot
        // leave a half-overwritten capability baseline behind.
        self.variables.restore_reserved();

        let value_dynamic = match eval {
            Ok(value) => value,
            Err(err) => {
                // A fallible capability function stashes its precise crate error
                // here; prefer it over the generic Rhai runtime wrapper so the
                // caller sees the real diagnostic.
                if let Some(host_err) = self.buffers.take_host_error() {
                    return Err(host_err);
                }
                return Err(map_rhai_error(*err));
            }
        };

        let value = if value_dynamic.is_unit() {
            None
        } else {
            Some(dynamic_to_repl_value(&value_dynamic))
        };

        let stdout = self.buffers.stdout();
        let calls = self.buffers.take_calls();
        let final_answer = self.buffers.answer();

        // Enforce the output byte limit fail-closed.
        let value_bytes = value.as_ref().map(ReplValue::byte_len).unwrap_or(0);
        if stdout.len() + value_bytes > self.policy.max_output_bytes {
            return Err(TinyAgentsError::LimitExceeded(format!(
                "ragsh cell produced {} bytes of output, exceeding the max_output_bytes limit of {}",
                stdout.len() + value_bytes,
                self.policy.max_output_bytes
            )));
        }

        let after = self.variables.snapshot();
        let variables_changed = diff_changed(&before, &after);

        Ok(ReplResult {
            stdout,
            value,
            variables_changed,
            calls,
            final_answer,
            elapsed: start.elapsed(),
        })
    }
}

impl CellBuffers {
    fn reset(&self) {
        self.stdout.lock().expect("stdout poisoned").clear();
        self.calls.lock().expect("calls poisoned").clear();
        *self.answer.lock().expect("answer poisoned") = None;
        *self.host_error.lock().expect("host_error poisoned") = None;
    }

    fn stdout(&self) -> String {
        self.stdout.lock().expect("stdout poisoned").clone()
    }

    fn take_calls(&self) -> Vec<ReplCallRecord> {
        std::mem::take(&mut *self.calls.lock().expect("calls poisoned"))
    }

    fn answer(&self) -> Option<String> {
        self.answer.lock().expect("answer poisoned").clone()
    }

    fn take_host_error(&self) -> Option<TinyAgentsError> {
        self.host_error.lock().expect("host_error poisoned").take()
    }

    // ── Accessors used by the capability built-ins (in `builtins.rs`). ──

    /// Pushes a recorded capability call/event.
    pub(super) fn push_call(&self, record: ReplCallRecord) {
        self.calls.lock().expect("calls poisoned").push(record);
    }

    /// Appends a line to the captured stdout buffer.
    pub(super) fn push_stdout_line(&self, line: &str) {
        let mut out = self.stdout.lock().expect("stdout poisoned");
        out.push_str(line);
        out.push('\n');
    }

    /// Sets the session's final answer.
    pub(super) fn set_answer(&self, content: String) {
        *self.answer.lock().expect("answer poisoned") = Some(content);
    }

    /// Stashes the precise crate error a fallible capability function failed
    /// with, so `eval_cell` can surface it verbatim.
    pub(super) fn set_host_error(&self, err: TinyAgentsError) {
        *self.host_error.lock().expect("host_error poisoned") = Some(err);
    }

    /// Returns the pre-cell namespace snapshot for `show_vars()`.
    pub(super) fn vars_snapshot(&self) -> BTreeMap<String, String> {
        self.vars_snapshot
            .lock()
            .expect("vars_snapshot poisoned")
            .clone()
    }
}

/// Maps a Rhai evaluation error to a crate error, distinguishing the
/// fail-closed operation-limit case from other compile/runtime failures.
fn map_rhai_error(err: EvalAltResult) -> TinyAgentsError {
    match err {
        EvalAltResult::ErrorTooManyOperations(pos) => TinyAgentsError::LimitExceeded(format!(
            "ragsh cell exceeded the operation limit (max_operations) at {pos}"
        )),
        other => TinyAgentsError::Validation(format!("ragsh evaluation error: {other}")),
    }
}

/// Returns the names whose values were added or changed between two snapshots,
/// excluding reserved names (which are restored after each cell).
fn diff_changed(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut changed: Vec<String> = after
        .iter()
        .filter(|(name, value)| {
            !reserved_names().any(|r| r == name.as_str())
                && before.get(*name).map(|b| b != *value).unwrap_or(true)
        })
        .map(|(name, _)| name.clone())
        .collect();
    changed.sort();
    changed.dedup();
    changed
}

/// Converts a [`ReplValue`] into a Rhai [`Dynamic`].
pub(super) fn repl_value_to_dynamic(value: &ReplValue) -> Dynamic {
    match value {
        ReplValue::Unit => Dynamic::UNIT,
        ReplValue::Bool(b) => Dynamic::from_bool(*b),
        ReplValue::Int(i) => Dynamic::from_int(*i),
        ReplValue::Float(f) => Dynamic::from_float(*f),
        ReplValue::String(s) => Dynamic::from(s.clone()),
        ReplValue::Array(items) => {
            let arr: rhai::Array = items.iter().map(repl_value_to_dynamic).collect();
            Dynamic::from_array(arr)
        }
        ReplValue::Map(map) => {
            let mut rmap = rhai::Map::new();
            for (k, v) in map {
                rmap.insert(k.as_str().into(), repl_value_to_dynamic(v));
            }
            Dynamic::from_map(rmap)
        }
    }
}

/// Converts a Rhai [`Dynamic`] into a typed [`ReplValue`].
///
/// Unsupported or opaque host values are stringified rather than leaking a Rhai
/// type across the capability boundary.
pub(super) fn dynamic_to_repl_value(value: &Dynamic) -> ReplValue {
    if value.is_unit() {
        return ReplValue::Unit;
    }
    if value.is_bool() {
        return ReplValue::Bool(value.as_bool().unwrap_or(false));
    }
    if value.is_int() {
        return ReplValue::Int(value.as_int().unwrap_or(0));
    }
    if value.is_float() {
        return ReplValue::Float(value.as_float().unwrap_or(0.0));
    }
    if value.is_string() {
        return ReplValue::String(value.clone().into_string().unwrap_or_default());
    }
    if value.is_array() {
        let arr = value.clone().into_array().unwrap_or_default();
        return ReplValue::Array(arr.iter().map(dynamic_to_repl_value).collect());
    }
    if value.is_map()
        && let Some(map) = value.read_lock::<rhai::Map>()
    {
        let mut out = BTreeMap::new();
        for (k, v) in map.iter() {
            out.insert(k.to_string(), dynamic_to_repl_value(v));
        }
        return ReplValue::Map(out);
    }
    ReplValue::String(value.to_string())
}

/// Converts a [`serde_json::Value`] (as returned by a tool or model call) into
/// a typed [`ReplValue`] so capability results cross back into the script as
/// native Rhai values.
pub(super) fn json_to_repl_value(value: &serde_json::Value) -> ReplValue {
    match value {
        serde_json::Value::Null => ReplValue::Unit,
        serde_json::Value::Bool(b) => ReplValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ReplValue::Int(i)
            } else {
                ReplValue::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => ReplValue::String(s.clone()),
        serde_json::Value::Array(items) => {
            ReplValue::Array(items.iter().map(json_to_repl_value).collect())
        }
        serde_json::Value::Object(map) => ReplValue::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_repl_value(v)))
                .collect(),
        ),
    }
}
