//! Inert configuration value types for the harness.
//!
//! **Dependency rule:** this module is `serde` + `std` only. No engine types,
//! no tokio, no host types. A host must be able to depend on these structs to
//! build a mapper without pulling in the rest of the crate's machinery — the
//! same carve-out that keeps a gated domain's type module always-compiled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Defaults ──────────────────────────────────────────────────────────────────
//
// These mirror the values a host is expected to supply. They exist so a
// partially-specified config still runs, NOT as an authority: a host with its
// own schema always maps its values in explicitly. Changing one silently
// changes behaviour for hosts that rely on `..Default::default()`, so treat
// them as a compatibility surface.

fn default_max_tool_iterations() -> usize {
    10
}

fn default_max_history_messages() -> usize {
    50
}

fn default_max_parallel_tools() -> usize {
    4
}

fn default_tool_dispatcher() -> ToolDispatcher {
    ToolDispatcher::Auto
}

fn default_max_memory_context_chars() -> usize {
    2000
}

fn default_tool_result_budget_bytes() -> usize {
    16 * 1024
}

fn default_agent_timeout_secs() -> u64 {
    120
}

fn default_agents_md_enabled() -> bool {
    true
}

// ── ToolDispatcher ────────────────────────────────────────────────────────────

/// How the runtime decides to dispatch tool calls for a turn.
///
/// Modelled as an enum rather than the host's free-form string so an
/// unrecognised mode is a mapping error at the boundary instead of a silent
/// fallthrough deep in the turn loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolDispatcher {
    /// Provider-native structured tool calls when the provider supports them,
    /// otherwise [`Xml`](Self::Xml).
    #[default]
    Auto,
    /// Force provider-native structured tool calls.
    Native,
    /// Force JSON-in-tag: `<tool_call>{…}</tool_call>`.
    Xml,
    /// Force compact positional P-Format: `tool[a|b]`. The most token-efficient
    /// encoding, but it mis-parses on some models, so it is opt-in only and
    /// never selected by [`Auto`](Self::Auto).
    Pformat,
}

// ── RequiredOutput ────────────────────────────────────────────────────────────

/// A contract asserting the model's reply carries a particular JSON block.
///
/// Satisfied when the reply contains a JSON object carrying `block_key` — plus
/// every entry in `required_keys` — with a non-null value. A blank `block_key`
/// makes the contract **inert** (enforcement skipped), so it is a safe default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RequiredOutput {
    /// The JSON key identifying the required block, e.g. `"thoughts"`.
    pub block_key: String,
    /// Sibling keys that must also be present and non-null in the same block.
    /// `block_key` is always required and need not be repeated here.
    #[serde(default)]
    pub required_keys: Vec<String>,
}

impl RequiredOutput {
    /// A contract on `block_key` with no extra required siblings.
    pub fn new(block_key: impl Into<String>) -> Self {
        Self {
            block_key: block_key.into(),
            required_keys: Vec::new(),
        }
    }

    /// Whether this contract actually enforces anything. A blank `block_key`
    /// is inert — callers should skip enforcement rather than fail every turn.
    pub fn is_active(&self) -> bool {
        !self.block_key.trim().is_empty()
    }
}

// ── MemoryLimits ──────────────────────────────────────────────────────────────

/// Character budgets for memory injected into a turn's context.
///
/// Characters, not tokens, deliberately: the runtime applies these before a
/// tokenizer is necessarily available, and the host's own schema is expressed
/// the same way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Ceiling on recalled memory text injected into one turn.
    #[serde(default = "default_max_memory_context_chars")]
    pub max_memory_context_chars: usize,
    /// Ceiling on summary text drawn from any single namespace.
    #[serde(default)]
    pub per_namespace_max_chars: usize,
    /// Ceiling on summary-tree text across all namespaces combined.
    #[serde(default)]
    pub total_tree_max_chars: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_memory_context_chars: default_max_memory_context_chars(),
            per_namespace_max_chars: 0,
            total_tree_max_chars: 0,
        }
    }
}

// ── TurnConfig ────────────────────────────────────────────────────────────────

/// Per-turn knobs: how long a turn may run and how much it may carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnConfig {
    /// Maximum tool-call/model round trips in a single turn before the runtime
    /// stops and returns what it has.
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: usize,
    /// How many prior messages to carry into the model request.
    #[serde(default = "default_max_history_messages")]
    pub max_history_messages: usize,
    /// Whether to compact context before a call when it grows large.
    #[serde(default)]
    pub compact_context: bool,
    /// Whether independent tool calls in one step may run concurrently.
    #[serde(default)]
    pub parallel_tools: bool,
    /// Concurrency cap when `parallel_tools` is set. Ignored otherwise.
    #[serde(default = "default_max_parallel_tools")]
    pub max_parallel_tools: usize,
    /// Byte budget for a single tool result before it is truncated or offloaded.
    #[serde(default = "default_tool_result_budget_bytes")]
    pub tool_result_budget_bytes: usize,
    /// Wall-clock ceiling for one turn.
    #[serde(default = "default_agent_timeout_secs")]
    pub timeout_secs: u64,
    /// Optional structured-output contract the reply must satisfy.
    #[serde(default)]
    pub required_output: Option<RequiredOutput>,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            max_tool_iterations: default_max_tool_iterations(),
            max_history_messages: default_max_history_messages(),
            compact_context: false,
            parallel_tools: false,
            max_parallel_tools: default_max_parallel_tools(),
            tool_result_budget_bytes: default_tool_result_budget_bytes(),
            timeout_secs: default_agent_timeout_secs(),
            required_output: None,
        }
    }
}

// ── ToolConfig ────────────────────────────────────────────────────────────────

/// How tools are dispatched and which are reachable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool-call dispatch strategy.
    #[serde(default = "default_tool_dispatcher")]
    pub dispatcher: ToolDispatcher,
    /// Per-channel permission grants, keyed by host-defined channel id. Opaque
    /// to the runtime — passed to the host's `SecurityGate`, never interpreted
    /// here.
    #[serde(default)]
    pub channel_permissions: HashMap<String, String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            dispatcher: default_tool_dispatcher(),
            channel_permissions: HashMap::new(),
        }
    }
}

// ── SessionConfig ─────────────────────────────────────────────────────────────

/// Everything a session needs that is *configuration* rather than *capability*.
///
/// Capabilities (memory, security, budget, …) arrive as trait objects; this
/// struct carries only inert values. A host maps its own schema into this once
/// at session-build time — see the mapper on the host side rather than making
/// the runtime read a host config type.
// No `Eq`: `temperature` is an `f64`, which is only `PartialEq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Root for the runtime's own internal state. Agent tools must **not**
    /// write here; the host enforces that, but the runtime should not treat
    /// this as a working directory either.
    pub workspace_dir: PathBuf,
    /// The agent's read/write root — where relative paths from acting tools
    /// resolve.
    pub action_dir: PathBuf,
    /// Default model id for this session.
    pub model: String,
    /// Overrides the model for the top-level (lead) agent when set.
    #[serde(default)]
    pub lead_model: Option<String>,
    /// Overrides the model for spawned subagents when set.
    #[serde(default)]
    pub subagent_model: Option<String>,
    /// Sampling temperature. `None` leaves the provider default in place.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Maximum subagent nesting depth. Zero disables delegation.
    #[serde(default)]
    pub max_depth: u32,
    /// Whether to load a repository's `AGENTS.md` into context.
    #[serde(default = "default_agents_md_enabled")]
    pub agents_md_enabled: bool,
    /// Per-turn limits.
    #[serde(default)]
    pub turn: TurnConfig,
    /// Tool dispatch and reachability.
    #[serde(default)]
    pub tools: ToolConfig,
    /// Budgets for memory injected into context.
    #[serde(default)]
    pub memory: MemoryLimits,
}

impl SessionConfig {
    /// A config rooted at `workspace_dir` / `action_dir` using `model`, with
    /// every other knob at its default.
    ///
    /// The three required values have no sensible default — a wrong guess for
    /// a path root would write agent output somewhere the user did not ask
    /// for — so they are constructor arguments rather than `Default` fields.
    /// This is also why `SessionConfig` deliberately does **not** implement
    /// `Default`.
    pub fn new(
        workspace_dir: impl Into<PathBuf>,
        action_dir: impl Into<PathBuf>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
            action_dir: action_dir.into(),
            model: model.into(),
            lead_model: None,
            subagent_model: None,
            temperature: None,
            max_depth: 0,
            agents_md_enabled: default_agents_md_enabled(),
            turn: TurnConfig::default(),
            tools: ToolConfig::default(),
            memory: MemoryLimits::default(),
        }
    }

    /// The model a lead (top-level) agent should use: `lead_model` when set,
    /// otherwise [`model`](Self::model).
    pub fn effective_lead_model(&self) -> &str {
        self.lead_model.as_deref().unwrap_or(&self.model)
    }

    /// The model a spawned subagent should use: `subagent_model` when set,
    /// otherwise [`model`](Self::model).
    ///
    /// Note this does **not** fall back to `lead_model` — a host that overrides
    /// the lead model usually wants subagents on the cheaper default, so
    /// inheriting the lead override would silently multiply cost.
    pub fn effective_subagent_model(&self) -> &str {
        self.subagent_model.as_deref().unwrap_or(&self.model)
    }

    /// Whether this session may spawn subagents at `depth`.
    pub fn may_delegate_at(&self, depth: u32) -> bool {
        depth < self.max_depth
    }
}
