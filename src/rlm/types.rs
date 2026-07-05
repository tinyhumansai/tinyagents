//! Public data types for the recursive-language-model (RLM) runtime.
//!
//! Everything here is `serde`-serializable on purpose: an RLM run is meant to
//! be **config-driven**, so an external harness (a CLI, a service, another
//! agent runtime) can describe an entire run — interpreter choice, resource
//! policy, prompt template — as a JSON document and hand it to
//! [`RlmConfig::from_json`].
//!
//! Logic lives in the sibling modules: the host capability boundary in
//! [`super::host`], the interpreter backends in [`super::interpreter`], the
//! session in [`super::session`], and the model-driven loop in
//! [`super::runner`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

// ── Interpreter selection ───────────────────────────────────────────────────

/// Which interpreter executes the code cells of an RLM session.
///
/// The embedded [`Rhai`](InterpreterSpec::Rhai) engine is the default and the
/// only *hermetically* sandboxed choice: it has no filesystem, network, or
/// process access — the capability functions registered by the host are its
/// entire world. External interpreters run as a child **process** provided by
/// the embedding application (the binary and args are configuration, exactly
/// so a harness can point at a virtualenv Python, a Deno binary, a container
/// entrypoint, …); they speak the line-delimited JSON wire protocol described
/// in [`super::interpreter::external`]. The host still enforces every
/// [`RlmPolicy`] limit fail-closed (killing the child on violation), but the
/// child process itself has whatever OS access the embedder's environment
/// grants it — isolate it externally (container, seccomp, jail) when running
/// untrusted models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InterpreterSpec {
    /// The embedded Rhai engine (feature-default, hermetic sandbox).
    #[default]
    Rhai,
    /// An external CPython-compatible interpreter.
    Python {
        /// The interpreter binary (defaults to `python3` on `PATH`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binary: Option<String>,
        /// Extra arguments placed before the bootstrap `-c` program.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
    },
    /// An external Node.js-compatible JavaScript interpreter.
    Javascript {
        /// The interpreter binary (defaults to `node` on `PATH`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binary: Option<String>,
        /// Extra arguments placed before the bootstrap `-e` program.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
    },
    /// An arbitrary command that already speaks the RLM wire protocol on its
    /// stdin/stdout (for embedders that ship their own runner, e.g. a
    /// container image or a jailed interpreter).
    Command {
        /// The command binary.
        binary: String,
        /// Arguments passed verbatim.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
    },
}

impl InterpreterSpec {
    /// The language name scripts are written in, used in prompts and code
    /// fence extraction (```rhai / ```python / ```javascript).
    pub fn language(&self) -> &'static str {
        match self {
            InterpreterSpec::Rhai => "rhai",
            InterpreterSpec::Python { .. } => "python",
            InterpreterSpec::Javascript { .. } => "javascript",
            InterpreterSpec::Command { .. } => "python",
        }
    }
}

// ── Policy ──────────────────────────────────────────────────────────────────

/// Resource limits bounding an RLM session and its model-driven loop.
///
/// Every limit is enforced **fail closed**: exceeding a bound aborts the cell
/// (and, for an external interpreter, kills the child process) instead of
/// silently truncating or running unbounded work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlmPolicy {
    /// Maximum code cells one [`super::RlmRunner::run`] loop may execute.
    pub max_cells: usize,
    /// Maximum source size, in bytes, of a single cell.
    pub max_script_bytes: usize,
    /// Maximum captured stdout + value size, in bytes, per cell.
    pub max_output_bytes: usize,
    /// Maximum sub-LLM (`llm`) calls per session.
    pub max_llm_calls: usize,
    /// Maximum `tool` calls per session.
    pub max_tool_calls: usize,
    /// Maximum sub-agent (`agent`) calls per session.
    pub max_agent_calls: usize,
    /// Maximum recursion depth for sub-agent calls, enforced through the
    /// shared harness guard
    /// ([`RunConfig::checked_child_depth`](crate::harness::context::RunConfig::checked_child_depth)).
    pub max_depth: usize,
    /// Wall-clock timeout per cell (script + in-flight capability calls).
    #[serde(with = "humantime_millis")]
    pub cell_timeout: Option<Duration>,
    /// Maximum Rhai operations per cell (embedded interpreter only; `0`
    /// means unlimited).
    pub max_operations: u64,
}

impl Default for RlmPolicy {
    fn default() -> Self {
        Self {
            max_cells: 16,
            max_script_bytes: 64 * 1024,
            max_output_bytes: 256 * 1024,
            max_llm_calls: 64,
            max_tool_calls: 128,
            max_agent_calls: 32,
            max_depth: 8,
            cell_timeout: Some(Duration::from_secs(120)),
            max_operations: 5_000_000,
        }
    }
}

/// Serializes the optional cell timeout as integer milliseconds so an RLM
/// config is a plain JSON document (`"cell_timeout": 120000`).
mod humantime_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(duration) => serializer.serialize_some(&(duration.as_millis() as u64)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(deserializer)?.map(Duration::from_millis))
    }
}

// ── Cancellation ────────────────────────────────────────────────────────────

/// A shared, sticky cancellation flag for an RLM session (the same contract as
/// the REPL's flag: once cancelled, the session refuses further work until a
/// fresh flag is installed).
#[derive(Clone, Debug, Default)]
pub struct RlmCancelFlag(Arc<AtomicBool>);

impl RlmCancelFlag {
    /// Creates a fresh, un-cancelled flag.
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Requests cancellation; idempotent, observed by every clone.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

// ── The host-call boundary ──────────────────────────────────────────────────

/// One capability call a script makes back into the host.
///
/// This is the **entire** host surface a sandboxed script sees, across every
/// interpreter backend: the embedded Rhai closures build these values
/// directly, and the external wire protocol carries them as the `call` field
/// of a `{"op":"call"}` frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "capability", rename_all = "snake_case")]
pub enum HostCall {
    /// A sub-LLM query (`llm(...)` in scripts).
    Llm {
        /// Registry model name; `None` selects the session's default model.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// The user prompt.
        prompt: String,
        /// Optional system prompt.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        system: Option<String>,
    },
    /// A tool invocation (`tool(name, args)` in scripts).
    Tool {
        /// Registry tool name.
        tool: String,
        /// JSON arguments matching the tool's schema.
        #[serde(default)]
        arguments: Value,
    },
    /// A sub-agent delegation (`agent(name, input)` in scripts).
    Agent {
        /// Registry agent name.
        agent: String,
        /// The prompt the child run is seeded with.
        input: String,
        /// Optional structured side-channel payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    /// The script's final answer (`final_answer(text)`); ends the run loop.
    FinalAnswer {
        /// The answer text handed back to the caller.
        answer: String,
    },
}

impl HostCall {
    /// The capability name used in call records and telemetry.
    pub fn name(&self) -> String {
        match self {
            HostCall::Llm { model, .. } => model.clone().unwrap_or_else(|| "default".to_string()),
            HostCall::Tool { tool, .. } => tool.clone(),
            HostCall::Agent { agent, .. } => agent.clone(),
            HostCall::FinalAnswer { .. } => "final_answer".to_string(),
        }
    }

    /// The record kind for this call.
    pub fn kind(&self) -> RlmCallKind {
        match self {
            HostCall::Llm { .. } => RlmCallKind::Llm,
            HostCall::Tool { .. } => RlmCallKind::Tool,
            HostCall::Agent { .. } => RlmCallKind::Agent,
            HostCall::FinalAnswer { .. } => RlmCallKind::FinalAnswer,
        }
    }
}

/// The kind of capability an [`RlmCallRecord`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlmCallKind {
    /// A sub-LLM query.
    Llm,
    /// A tool invocation.
    Tool,
    /// A sub-agent delegation.
    Agent,
    /// The final answer.
    FinalAnswer,
}

/// A record of one capability call a cell performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmCallRecord {
    /// Which capability kind was invoked.
    pub kind: RlmCallKind,
    /// The capability name (model, tool, or agent registry name).
    pub name: String,
    /// Structured detail about the call (argument summary, sizes).
    pub detail: Value,
    /// Wall-clock time the call took.
    pub elapsed: Duration,
}

// ── Cell + run outcomes ─────────────────────────────────────────────────────

/// The structured result of evaluating one code cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellOutcome {
    /// Captured print/console output, bounded by
    /// [`RlmPolicy::max_output_bytes`].
    pub stdout: String,
    /// The cell's final expression value, if it produced one.
    pub value: Option<Value>,
    /// A script-level error (exception / runtime error), when the cell
    /// failed *recoverably* — the driving model sees this and may adapt.
    pub error: Option<String>,
    /// Capability calls recorded during the cell, in order.
    pub calls: Vec<RlmCallRecord>,
    /// The final answer, if the cell called `final_answer(...)`.
    pub final_answer: Option<String>,
    /// Wall-clock time the cell took to evaluate.
    pub elapsed: Duration,
}

/// Why an [`RlmOutcome`] run loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlmStopReason {
    /// A cell called `final_answer(...)`.
    Answered,
    /// The driver model replied with prose and no code cell; the prose is
    /// taken as the answer.
    ModelAnswered,
    /// The [`RlmPolicy::max_cells`] budget was exhausted without an answer.
    CellBudgetExhausted,
}

/// One executed step of the model-driven loop: the code the driver model
/// wrote and what evaluating it produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmStep {
    /// The code cell the driver model emitted.
    pub code: String,
    /// The evaluation outcome fed back to the model.
    pub outcome: CellOutcome,
}

/// The result of one complete [`super::RlmRunner::run`] loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmOutcome {
    /// The final answer, when the run produced one.
    pub answer: Option<String>,
    /// Why the loop stopped.
    pub stop_reason: RlmStopReason,
    /// Every executed step, in order (the full trajectory).
    pub steps: Vec<RlmStep>,
    /// Driver-model calls made by the loop itself (excludes sub-LLM calls
    /// made *by scripts*, which are counted in [`RlmOutcome::sub_llm_calls`]).
    pub driver_calls: usize,
    /// Sub-LLM calls scripts made through the `llm` capability.
    pub sub_llm_calls: usize,
    /// Tool calls scripts made through the `tool` capability.
    pub tool_calls: usize,
    /// Sub-agent calls scripts made through the `agent` capability.
    pub agent_calls: usize,
}

// ── Templates ───────────────────────────────────────────────────────────────

/// A named prompt scaffold for the driver model.
///
/// The `system_prompt` may reference these placeholders, substituted at run
/// time by [`super::templates::render_system_prompt`]:
///
/// - `{{language}}` — the interpreter language (`rhai`, `python`, …)
/// - `{{usage}}` — the interpreter-specific capability usage guide
/// - `{{capabilities}}` — the live model/tool/agent registry listing
/// - `{{limits}}` — a human-readable summary of the [`RlmPolicy`]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlmTemplate {
    /// The template name (used by [`TemplateSpec::Named`]).
    pub name: String,
    /// The system prompt scaffold with `{{placeholder}}` slots.
    pub system_prompt: String,
}

/// How a config selects its prompt template: one of the built-in named
/// templates, or a fully inline scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateSpec {
    /// A built-in template by name (`"general"`, `"context-explorer"`,
    /// `"orchestrator"`).
    Named(String),
    /// An inline template document.
    Inline(RlmTemplate),
}

impl Default for TemplateSpec {
    fn default() -> Self {
        TemplateSpec::Named("general".to_string())
    }
}

// ── Config ──────────────────────────────────────────────────────────────────

/// A complete, serializable description of an RLM run — the document an
/// external harness hands to [`super::RlmRunner::from_config`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RlmConfig {
    /// Which interpreter executes code cells.
    pub interpreter: InterpreterSpec,
    /// The registry name of the driver model that writes cells; `None`
    /// selects the registry default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_model: Option<String>,
    /// The registry name of the default sub-LLM scripts reach with
    /// `llm(...)` when they don't name a model; `None` falls back to the
    /// driver model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_model: Option<String>,
    /// Resource limits for the session.
    pub policy: RlmPolicy,
    /// The driver prompt template.
    pub template: TemplateSpec,
}

impl RlmConfig {
    /// Parses a config from a JSON document.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(TinyAgentsError::Serialization)
    }

    /// Serializes this config to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(TinyAgentsError::Serialization)
    }
}
