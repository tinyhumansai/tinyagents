//! An RLM session: one interpreter instance bound to one capability host.
//!
//! The session is the programmatic surface — "the interpreter exposed as an
//! API". Embedders that want direct cell execution (their own loop, a
//! notebook UI, a test) drive [`RlmSession::eval`] themselves; the
//! model-driven loop in [`super::runner`] is built on exactly this surface.

use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use super::host::{RlmHost, RlmHostApi};
use super::interpreter::{RlmInterpreter, build_interpreter};
use super::types::{CellOutcome, InterpreterSpec};
use crate::error::{Result, TinyAgentsError};

/// Marker appended when captured output exceeds
/// [`RlmPolicy::max_output_bytes`] and is truncated.
const TRUNCATION_MARKER: &str = "\n… [output truncated by rlm policy]";

/// One sandboxed script workspace: a persistent interpreter plus the
/// capability host its cells call back into.
pub struct RlmSession<State: Send + Sync + 'static> {
    interpreter: Box<dyn RlmInterpreter>,
    host: Arc<RlmHost<State>>,
    cells_run: usize,
}

impl<State: Send + Sync + 'static> RlmSession<State> {
    /// Builds a session from an interpreter spec and a configured host.
    pub fn new(spec: &InterpreterSpec, host: Arc<RlmHost<State>>) -> Result<Self> {
        let interpreter = build_interpreter(spec, host.policy().max_operations)?;
        Ok(Self {
            interpreter,
            host,
            cells_run: 0,
        })
    }

    /// Builds a session over an already-constructed interpreter backend
    /// (for custom [`RlmInterpreter`] implementations).
    pub fn from_interpreter(
        interpreter: Box<dyn RlmInterpreter>,
        host: Arc<RlmHost<State>>,
    ) -> Self {
        Self {
            interpreter,
            host,
            cells_run: 0,
        }
    }

    /// The capability host this session's cells call back into.
    pub fn host(&self) -> &Arc<RlmHost<State>> {
        &self.host
    }

    /// The language cells are written in.
    pub fn language(&self) -> String {
        self.interpreter.language().to_string()
    }

    /// The interpreter-specific capability usage guide (prompt fragment).
    pub fn usage_guide(&self) -> String {
        self.interpreter.usage_guide()
    }

    /// Number of cells evaluated so far.
    pub fn cells_run(&self) -> usize {
        self.cells_run
    }

    /// Injects a global variable visible to subsequent cells — the safe way
    /// to hand a task context to scripts without splicing it into source.
    pub async fn set_variable(&mut self, name: &str, value: Value) -> Result<()> {
        self.interpreter.set_variable(name, value).await
    }

    /// Evaluates one code cell, enforcing the session policy fail-closed.
    pub async fn eval(&mut self, code: &str) -> Result<CellOutcome> {
        let policy = self.host.policy().clone();
        if self.host.cancel_flag().is_cancelled() {
            return Err(TinyAgentsError::Cancelled);
        }
        if self.cells_run >= policy.max_cells {
            return Err(TinyAgentsError::LimitExceeded(format!(
                "cell limit ({}) exceeded",
                policy.max_cells
            )));
        }
        if code.len() > policy.max_script_bytes {
            return Err(TinyAgentsError::LimitExceeded(format!(
                "cell source is {} bytes, over the {}-byte limit",
                code.len(),
                policy.max_script_bytes
            )));
        }
        self.cells_run += 1;

        self.host.begin_cell();
        let start = Instant::now();
        let eval = self
            .interpreter
            .eval_cell(code, self.host.clone() as Arc<dyn RlmHostApi>)
            .await;
        let (calls, final_answer) = self.host.end_cell();
        let mut eval = eval?;

        // Bound what flows back into the driver conversation. Truncation is
        // explicit (marked) so the model knows it saw a prefix.
        if eval.stdout.len() > policy.max_output_bytes {
            eval.stdout.truncate(policy.max_output_bytes);
            eval.stdout.push_str(TRUNCATION_MARKER);
        }
        if let Some(value) = &eval.value {
            let rendered = value.to_string();
            if rendered.len() > policy.max_output_bytes {
                let mut clipped = rendered;
                clipped.truncate(policy.max_output_bytes);
                clipped.push_str(TRUNCATION_MARKER);
                eval.value = Some(Value::String(clipped));
            }
        }

        Ok(CellOutcome {
            stdout: eval.stdout,
            value: eval.value,
            error: eval.error,
            calls,
            final_answer,
            elapsed: start.elapsed(),
        })
    }

    /// Releases interpreter resources (kills an external child process).
    pub async fn shutdown(&mut self) -> Result<()> {
        self.interpreter.shutdown().await
    }
}
