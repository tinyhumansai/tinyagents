//! Public data types for the Rhai-backed `.ragsh` session runtime.
//!
//! These are the typed values that cross the host/script boundary: the
//! [`ReplPolicy`] limits that bound a session, the [`ReplCapabilities`] that
//! wire a session to the named registries, and the [`ReplResult`] /
//! [`ReplValue`] / [`ReplCallRecord`] values a single evaluated cell produces.
//!
//! Logic (engine construction, cell evaluation, reserved-name restoration)
//! lives in [`super`]; tests live in `test.rs`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::language::types::{Blueprint, Origin};
use crate::registry::CapabilityRegistry;

// ── Reserved names ──────────────────────────────────────────────────────────

/// Reserved built-in *variable* names seeded into every session scope.
///
/// These are restored to their session baseline after each cell so a script
/// can read or temporarily shadow them but cannot permanently replace the
/// session's context, state, or final-answer slots.
pub const RESERVED_VARIABLES: &[&str] =
    &["context", "state", "messages", "history", "run", "answer"];

/// Reserved built-in *capability function* names.
///
/// Rhai resolves a call expression against the registered-function namespace,
/// which is independent of the variable namespace, so these names cannot be
/// replaced by a script-level `let`. They are listed here so the runtime can
/// also scrub any same-named variable a script introduces, matching the design
/// document's "scripts may add locals but not permanently replace
/// capabilities" rule.
pub const RESERVED_FUNCTIONS: &[&str] = &[
    "model_query",
    "model_query_batched",
    "agent_query",
    "agent_query_batched",
    "graph_run",
    "graph_run_batched",
    "graph_define",
    "graph_validate",
    "graph_compile",
    "graph_diff",
    "graph_register",
    "tool_call",
    "tool_call_batched",
    "emit",
    "show_vars",
    "answer",
];

/// Returns every reserved name (variables and capability functions) the
/// runtime must protect across cells.
pub fn reserved_names() -> impl Iterator<Item = &'static str> {
    RESERVED_VARIABLES
        .iter()
        .copied()
        .chain(RESERVED_FUNCTIONS.iter().copied())
}

// ── Policy ──────────────────────────────────────────────────────────────────

/// Resource limits that bound a [`super::ReplSession`].
///
/// Every limit is enforced "fail closed": when a script would exceed a bound,
/// cell evaluation returns an error rather than truncating silently or running
/// unbounded work. The defaults are conservative and tuned for an in-process,
/// model-driven orchestration loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplPolicy {
    /// Maximum Rhai operations per cell, wired to
    /// [`rhai::Engine::set_max_operations`]. `0` means unlimited.
    pub max_operations: u64,
    /// Maximum CodeAct loop iterations a model-driven session may run.
    pub max_iterations: usize,
    /// Maximum source size, in bytes, of a single cell.
    pub max_script_bytes: usize,
    /// Maximum captured stdout/value size, in bytes, per cell.
    pub max_output_bytes: usize,
    /// Maximum `model_query` calls per session.
    pub max_model_calls: usize,
    /// Maximum `tool_call` calls per session.
    pub max_tool_calls: usize,
    /// Maximum `graph_run` calls per session.
    pub max_graph_calls: usize,
    /// Maximum `graph_define` blueprints per session.
    pub max_graph_definitions: usize,
    /// Maximum recursion depth for sub-model/sub-agent/sub-graph calls.
    pub max_depth: usize,
    /// Optional wall-clock timeout per cell.
    pub timeout: Option<Duration>,
    /// Maximum concurrency for batched capability calls.
    pub max_concurrency: usize,
    /// When `true`, model-generated graphs require a review token before they
    /// can be registered.
    pub generated_graphs_require_review: bool,
}

impl Default for ReplPolicy {
    fn default() -> Self {
        Self {
            max_operations: 1_000_000,
            max_iterations: 16,
            max_script_bytes: 64 * 1024,
            max_output_bytes: 256 * 1024,
            max_model_calls: 64,
            max_tool_calls: 128,
            max_graph_calls: 32,
            max_graph_definitions: 8,
            max_depth: 8,
            timeout: Some(Duration::from_secs(30)),
            max_concurrency: 4,
            generated_graphs_require_review: true,
        }
    }
}

// ── Values ──────────────────────────────────────────────────────────────────

/// A typed, serializable projection of a Rhai value returned from a cell.
///
/// The Rhai engine is dynamically typed; `ReplValue` is the explicit conversion
/// at the capability boundary the design document requires. Unsupported or
/// opaque Rhai values are stringified rather than leaking host types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum ReplValue {
    /// Rhai unit `()` — used when a cell produces no meaningful value.
    Unit,
    /// A boolean.
    Bool(bool),
    /// A 64-bit signed integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// A string.
    String(String),
    /// An ordered array of values.
    Array(Vec<ReplValue>),
    /// A string-keyed map of values.
    Map(BTreeMap<String, ReplValue>),
}

impl ReplValue {
    /// Converts this value into a [`serde_json::Value`] for event/store writes.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            ReplValue::Unit => serde_json::Value::Null,
            ReplValue::Bool(b) => serde_json::Value::Bool(*b),
            ReplValue::Int(i) => serde_json::Value::from(*i),
            ReplValue::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            ReplValue::String(s) => serde_json::Value::String(s.clone()),
            ReplValue::Array(items) => {
                serde_json::Value::Array(items.iter().map(ReplValue::to_json).collect())
            }
            ReplValue::Map(map) => serde_json::Value::Object(
                map.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
            ),
        }
    }

    /// Returns the approximate serialized size of this value, in bytes, used
    /// to enforce [`ReplPolicy::max_output_bytes`].
    pub fn byte_len(&self) -> usize {
        serde_json::to_string(&self.to_json())
            .map(|s| s.len())
            .unwrap_or(0)
    }
}

// ── Call records ────────────────────────────────────────────────────────────

/// The kind of capability a [`ReplCallRecord`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplCallKind {
    /// A `model_query` call.
    Model,
    /// A `tool_call` call.
    Tool,
    /// A `graph_run` call.
    Graph,
    /// An `agent_query` call.
    Agent,
    /// A custom `emit` event.
    Emit,
}

/// A record of one capability call (or emitted event) a cell performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplCallRecord {
    /// Unique id for this call within the session.
    pub call_id: crate::harness::ids::CallId,
    /// Which capability kind was invoked.
    pub kind: ReplCallKind,
    /// The capability or event name.
    pub name: String,
    /// Structured detail about the call (arguments or payload).
    pub detail: serde_json::Value,
    /// Wall-clock time the call took.
    pub elapsed: Duration,
}

// ── Cell result ─────────────────────────────────────────────────────────────

/// The structured result of evaluating one `.ragsh` cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplResult {
    /// Captured `print`/`debug` output, truncated-free up to the policy bound.
    pub stdout: String,
    /// The cell's final expression value, if it produced one.
    pub value: Option<ReplValue>,
    /// Names of persistent variables created or changed by this cell.
    pub variables_changed: Vec<String>,
    /// Capability calls and emitted events recorded during the cell.
    pub calls: Vec<ReplCallRecord>,
    /// The final answer, if the cell called `answer(...)`.
    pub final_answer: Option<String>,
    /// Wall-clock time the cell took to evaluate.
    pub elapsed: Duration,
}

// ── Graph blueprint handle ──────────────────────────────────────────────────

/// An opaque, script-carryable handle to a `.rag` graph blueprint drafted
/// inside a session.
///
/// `graph_define` lowers `.rag` source through the Cluster H compiler and
/// returns one of these as a Rhai value; `graph_validate`, `graph_compile`,
/// `graph_diff`, and `graph_register` accept it back. The handle carries the
/// compiled [`Blueprint`] together with the original source and its
/// [`Origin`] provenance (always [`Origin::Generated`] for REPL-authored
/// graphs) so a review tool can trace topology back to the producing session.
///
/// Generated topology is **never** installed directly: a handle is only marked
/// `compiled` after passing the capability resolver, and registration through
/// `graph_register` still honors [`ReplPolicy::generated_graphs_require_review`].
#[derive(Debug, Clone)]
pub struct GraphBlueprintHandle {
    /// The graph name (its `graph_id`).
    pub name: String,
    /// The original `.rag` source the blueprint was drafted from.
    pub source: String,
    /// The compiled blueprint.
    pub blueprint: Blueprint,
    /// Source provenance — generated, labelled with the session id.
    pub origin: Origin,
    /// `true` once the handle has passed `graph_compile` (resolver-bound).
    pub compiled: bool,
    /// Whether registering this generated graph requires a review token, copied
    /// from the session policy at compile time.
    pub requires_review: bool,
}

// ── Language compiler handle ────────────────────────────────────────────────

/// A thin handle marking that a session may draft and compile `.rag` graph
/// blueprints through the expressive-language compiler.
///
/// Generated graph topology is never installed directly: the actual
/// `graph_define`/`graph_compile`/`graph_register` wiring routes through the
/// `.rag` compiler, the capability resolver, and the policy review gate. This
/// handle records the provenance label applied to generated blueprints and is
/// fleshed out by the graph-capability slice; here it establishes the typed
/// slot the design document's `ReplCapabilities::language` field describes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageCompiler {
    /// Provenance label stamped on blueprints generated in this session.
    pub provenance_label: String,
}

impl Default for LanguageCompiler {
    fn default() -> Self {
        Self {
            provenance_label: "ragsh-generated".to_string(),
        }
    }
}

// ── Capabilities ────────────────────────────────────────────────────────────

/// The named capabilities a session may bind against.
///
/// The design document sketches separate `ModelRegistry`, `ToolRegistry`,
/// `GraphRegistry`, and `AgentRegistry` fields. In this crate those four kinds
/// are already unified under the single name-addressable
/// [`CapabilityRegistry`], so `ReplCapabilities` wraps that registry (shared via
/// `Arc` so a session can be cheaply cloned into a graph node) plus an optional
/// [`LanguageCompiler`]. The per-kind accessors ([`models`](Self::models),
/// [`tools`](Self::tools), [`graphs`](Self::graphs), [`agents`](Self::agents))
/// preserve the documented surface.
///
/// A prior revision also carried a [`crate::harness::store::StoreRegistry`]
/// field, but no built-in
/// (`model_query`, `tool_call`, …) ever read or wrote through it — it was dead
/// weight advertising a capability the engine did not actually expose. It was
/// removed rather than left half-wired; long-term store access can be added
/// back as real `store_get`/`store_set` built-ins (see [`super::builtins`])
/// once that surface is designed.
pub struct ReplCapabilities<State = ()>
where
    State: Send + Sync,
{
    /// The unified capability catalog (models, tools, graphs, agents).
    pub registry: Arc<CapabilityRegistry<State>>,
    /// Optional expressive-language compiler handle for graph drafting.
    pub language: Option<LanguageCompiler>,
}

impl<State: Send + Sync> ReplCapabilities<State> {
    /// Builds capabilities over an existing capability registry.
    pub fn new(registry: Arc<CapabilityRegistry<State>>) -> Self {
        Self {
            registry,
            language: None,
        }
    }

    /// Enables the expressive-language compiler handle for this session.
    pub fn with_language(mut self, language: LanguageCompiler) -> Self {
        self.language = Some(language);
        self
    }

    /// Returns the registered model names.
    pub fn models(&self) -> Vec<String> {
        self.registry.names(crate::registry::ComponentKind::Model)
    }

    /// Returns the registered tool names.
    pub fn tools(&self) -> Vec<String> {
        self.registry.names(crate::registry::ComponentKind::Tool)
    }

    /// Returns the registered graph-blueprint names.
    pub fn graphs(&self) -> Vec<String> {
        self.registry.names(crate::registry::ComponentKind::Graph)
    }

    /// Returns the registered agent names.
    pub fn agents(&self) -> Vec<String> {
        self.registry.names(crate::registry::ComponentKind::Agent)
    }
}

impl<State: Send + Sync> Default for ReplCapabilities<State> {
    fn default() -> Self {
        Self::new(Arc::new(CapabilityRegistry::new()))
    }
}
