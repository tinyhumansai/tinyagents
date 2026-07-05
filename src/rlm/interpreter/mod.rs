//! Interpreter backends for the RLM runtime.
//!
//! [`RlmInterpreter`] is the pluggable execution API: the session hands a
//! backend one code cell at a time plus the shared [`RlmHostApi`] handle, and
//! the backend returns a raw [`CellEval`]. State (variables, imports,
//! definitions) persists across cells within one backend instance, so the
//! driving model can build up a workspace incrementally like a notebook.
//!
//! Two backends ship built in:
//!
//! - [`rhai_cell::RhaiInterpreter`] — the embedded Rhai engine. Hermetic: no
//!   filesystem, network, or process access; the registered capability
//!   functions are its entire host surface.
//! - [`external::ExternalInterpreter`] — a child process (Python, Node, or
//!   any command speaking the wire protocol). The binary is configuration,
//!   so embedders choose the exact interpreter (virtualenv Python, Deno, a
//!   containerized runner, …).
//!
//! Construct a backend from configuration with [`build_interpreter`].

pub mod external;
pub mod rhai_cell;

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;

use super::host::RlmHostApi;
use super::types::{InterpreterSpec, RlmCancelFlag};
use crate::error::{Result, TinyAgentsError};

/// The raw output of evaluating one cell, before the session merges in the
/// host-side call records and final answer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellEval {
    /// Captured print/console output.
    pub stdout: String,
    /// The cell's final expression value, if any.
    pub value: Option<Value>,
    /// A recoverable script error (exception / runtime error).
    pub error: Option<String>,
}

/// A pluggable code-cell execution backend.
#[async_trait]
pub trait RlmInterpreter: Send {
    /// The language cells are written in (`"rhai"`, `"python"`, …), used for
    /// prompt rendering and code-fence extraction.
    fn language(&self) -> &str;

    /// A prompt fragment teaching the driver model how to reach the host
    /// capabilities *in this language* (function signatures, examples).
    fn usage_guide(&self) -> String;

    /// Sets (or replaces) a global variable visible to subsequent cells.
    ///
    /// Used to inject the task context (`context`) without string-splicing
    /// user data into script source.
    async fn set_variable(&mut self, name: &str, value: Value) -> Result<()>;

    /// Evaluates one code cell against the host.
    ///
    /// Returns `Ok(CellEval)` for both success and *recoverable* script
    /// errors (carried in [`CellEval::error`]); returns `Err` only for fatal
    /// conditions (policy limits, timeout, cancellation, a dead backend).
    async fn eval_cell(&mut self, code: &str, host: Arc<dyn RlmHostApi>) -> Result<CellEval>;

    /// Releases backend resources (kills a child process). Idempotent.
    async fn shutdown(&mut self) -> Result<()>;
}

/// Builds the interpreter backend described by an [`InterpreterSpec`].
///
/// `max_operations` bounds the embedded Rhai engine; external backends are
/// bounded by the cell deadline instead (a wedged child is killed).
pub fn build_interpreter(
    spec: &InterpreterSpec,
    max_operations: u64,
) -> Result<Box<dyn RlmInterpreter>> {
    match spec {
        InterpreterSpec::Rhai => Ok(Box::new(rhai_cell::RhaiInterpreter::new(max_operations))),
        InterpreterSpec::Python { binary, args } => Ok(Box::new(
            external::ExternalInterpreter::python(binary.as_deref(), args.clone()),
        )),
        InterpreterSpec::Javascript { binary, args } => Ok(Box::new(
            external::ExternalInterpreter::javascript(binary.as_deref(), args.clone()),
        )),
        InterpreterSpec::Command { binary, args } => Ok(Box::new(
            external::ExternalInterpreter::command(binary.clone(), args.clone()),
        )),
    }
}

/// How often the watcher thread wakes to observe cancellation while a
/// capability call is in flight inside the blocking bridge.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Why the watcher tripped a bounded bridge call.
enum BridgeStop {
    Deadline,
    Cancelled,
}

/// Drives an async host-capability future to completion **synchronously**,
/// bounded by the cell deadline and the session cancel flag.
///
/// This is the same fail-closed blocking bridge the `.ragsh` REPL uses (see
/// `repl::session::builtins`): the embedded Rhai engine is synchronous, so a
/// capability closure must block its thread — but never unboundedly. A
/// detached watcher thread races the future; if the deadline elapses or the
/// flag trips first, the future is dropped (cancelling the underlying
/// request) and a `Timeout` / `Cancelled` error is returned.
pub(super) fn bridge_block_on<T, F>(
    deadline: Option<Instant>,
    cancel: &RlmCancelFlag,
    future: F,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    if cancel.is_cancelled() {
        return Err(TinyAgentsError::Cancelled);
    }
    if let Some(deadline) = deadline
        && Instant::now() >= deadline
    {
        return Err(TinyAgentsError::Timeout(
            "rlm cell deadline elapsed before a host capability call could start".to_string(),
        ));
    }

    let (tx, rx) = futures::channel::oneshot::channel::<BridgeStop>();
    let watcher_cancel = cancel.clone();
    std::thread::spawn(move || {
        loop {
            if tx.is_canceled() {
                return;
            }
            if watcher_cancel.is_cancelled() {
                let _ = tx.send(BridgeStop::Cancelled);
                return;
            }
            match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        let _ = tx.send(BridgeStop::Deadline);
                        return;
                    }
                    std::thread::sleep((deadline - now).min(CANCEL_POLL_INTERVAL));
                }
                None => std::thread::sleep(CANCEL_POLL_INTERVAL),
            }
        }
    });

    match futures::executor::block_on(futures::future::select(Box::pin(future), rx)) {
        futures::future::Either::Left((output, _watcher)) => output,
        futures::future::Either::Right((stop, _fut)) => match stop {
            Ok(BridgeStop::Cancelled) => Err(TinyAgentsError::Cancelled),
            Ok(BridgeStop::Deadline) | Err(_) => Err(TinyAgentsError::Timeout(
                "rlm cell deadline elapsed during a host capability call".to_string(),
            )),
        },
    }
}
