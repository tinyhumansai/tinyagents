//! Tool layer types used by the harness.
//!
//! These types define the call boundary every harness capability shares —
//! including sub-agents exposed as tools (see
//! [`crate::harness::subagent::SubAgentTool`]), which is how the recursive
//! architecture turns "agents calling agents" into ordinary tool calls.
//!
//! Here a [`ToolCall`] carries a required `id` so results can be correlated
//! back to the originating call, matching provider tool-call semantics.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use crate::harness::context::RunContext;
use crate::harness::events::EventSink;
use crate::harness::ids::{RunId, ThreadId};

/// The model-visible syntax a tool declaration prefers.
///
/// Tool execution remains provider-neutral: after parsing, the harness invokes
/// tools with [`ToolCall::arguments`] as JSON so local schema validation,
/// middleware, tracing, and replay use one stable representation. This format
/// tells prompt renderers and provider adapters how a tool should be exposed to
/// a model when the provider does not force a native tool-calling shape.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolFormat {
    /// Native JSON/function-call style. This is the default and maps directly to
    /// providers such as OpenAI Chat Completions.
    #[default]
    Json,
    /// XML tag style, for example
    /// `<search><query>rust</query></search>`.
    Xml,
    /// Parametric p-type style: a compact ordered-parameter call syntax such as
    /// `search("rust", 5)`.
    PType {
        /// Ordered parameter names used by compact renderers. The names should
        /// correspond to fields in [`ToolSchema::parameters`].
        parameters: Vec<String>,
    },
}

/// A model-visible declaration of a tool: its name, description,
/// JSON-schema-compatible parameter shape, and preferred tool-call format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Canonical tool name (ASCII `snake_case` by convention).
    pub name: String,
    /// Human/model readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the model-visible input arguments.
    pub parameters: Value,
    /// Preferred model-visible tool-call format.
    #[serde(default, skip_serializing_if = "ToolFormat::is_json")]
    pub format: ToolFormat,
}

/// A request from the model to invoke a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id, required for result correlation.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments supplied by the model, as raw JSON.
    #[serde(default)]
    pub arguments: Value,
}

/// The outcome of executing a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Id of the [`ToolCall`] this result answers.
    pub call_id: String,
    /// Name of the tool that produced the result.
    pub name: String,
    /// Model-facing textual content.
    pub content: String,
    /// Optional structured value for application code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Error message when the tool failed; `None` on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock execution time in milliseconds.
    #[serde(default)]
    pub elapsed_ms: u64,
}

/// Live run context visible to a tool invoked by an agent loop.
///
/// The legacy [`Tool::call`] entry point remains available for direct calls and
/// tests. The agent loop uses [`Tool::call_with_context`] so recursive tools
/// such as sub-agents can inherit caller lineage while still isolating child
/// threads.
#[derive(Clone)]
pub struct ToolExecutionContext {
    /// Run that invoked the tool.
    pub run_id: RunId,
    /// Caller thread id, when the parent run is threaded.
    pub thread_id: Option<ThreadId>,
    /// Caller recursion depth.
    pub depth: usize,
    /// Maximum output tokens requested for each model turn in the caller's run.
    pub max_turn_output_tokens: Option<u32>,
    /// Shared event sink for nested run observability.
    pub events: EventSink,
    /// The isolated workspace/sandbox the tool may operate in, when the run was
    /// configured with a
    /// [`WorkspaceIsolation`][crate::harness::workspace::WorkspaceIsolation]
    /// provider. A tool discovers its allowed root here instead of an
    /// application global; `None` means no workspace policy is in effect.
    pub workspace: Option<crate::harness::workspace::WorkspaceDescriptor>,
}

impl ToolExecutionContext {
    /// Captures the non-generic tool-visible parts of a live [`RunContext`].
    pub fn from_run_context<Ctx>(ctx: &RunContext<Ctx>) -> Self {
        Self {
            run_id: ctx.config.run_id.clone(),
            thread_id: ctx.config.thread_id.clone(),
            depth: ctx.config.depth,
            max_turn_output_tokens: ctx.config.max_turn_output_tokens,
            events: ctx.events.clone(),
            workspace: None,
        }
    }

    /// Attaches an isolated workspace descriptor the tool may operate in.
    pub fn with_workspace(
        mut self,
        workspace: crate::harness::workspace::WorkspaceDescriptor,
    ) -> Self {
        self.workspace = Some(workspace);
        self
    }
}

/// How strictly a tool must be sandboxed when it executes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxMode {
    /// Inherit whatever the run's execution environment provides (the default).
    #[default]
    Inherit,
    /// The tool is safe to run without any sandbox.
    Disabled,
    /// The tool must run inside an isolated execution environment; policy
    /// enforcement fails closed if no sandbox is available.
    Required,
}

/// How a tool is allowed to reach the caller's workspace / filesystem root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceAccess {
    /// The tool needs no filesystem/workspace access (the safe default).
    #[default]
    None,
    /// The tool may only touch explicitly declared
    /// [`ToolAccess::trusted_roots`].
    Scoped,
    /// The tool may touch any path the process can reach.
    Any,
}

/// Declared side effects a tool may cause.
///
/// Used by policy enforcement (see
/// [`ToolPolicyMiddleware`][crate::harness::middleware::ToolPolicyMiddleware]) to
/// decide whether a tool may be exposed to the model or executed under a given
/// run's constraints. Every flag defaults to `false`, matching a pure,
/// side-effect-free tool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSideEffects {
    /// The tool reads state but never mutates anything observable.
    pub read_only: bool,
    /// The tool creates, modifies, or deletes files.
    pub writes_files: bool,
    /// The tool performs network I/O.
    pub network: bool,
    /// The tool installs packages or otherwise mutates the toolchain.
    pub installs_dependencies: bool,
    /// The tool can perform irreversible / destructive actions.
    pub destructive: bool,
    /// The tool calls an external third-party service.
    pub external_service: bool,
    /// The tool can move money or incur a charge.
    pub payment: bool,
}

/// Runtime requirements a tool declares for safe execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRuntime {
    /// Suggested per-call wall-clock timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Maximum automatic retries permitted for this tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Whether repeating the call with identical arguments is safe.
    pub idempotent: bool,
    /// Whether the tool honors cooperative cancellation.
    pub cancelable: bool,
    /// How strictly the tool must be sandboxed.
    pub sandbox: SandboxMode,
    /// Maximum result payload the harness should accept, in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_result_bytes: Option<usize>,
    /// Whether the tool can emit [`ToolDelta`] streaming fragments.
    pub streaming: bool,
}

/// Access requirements a tool declares before it can be exposed or run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAccess {
    /// Workspace/filesystem reach the tool needs.
    pub workspace: WorkspaceAccess,
    /// Filesystem roots the tool is allowed to touch (for
    /// [`WorkspaceAccess::Scoped`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trusted_roots: Vec<String>,
    /// Named credentials the tool requires to be present.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<String>,
    /// Whether an explicit human approval is required before each call.
    pub approval_required: bool,
    /// Whether the tool is safe to run in a background/non-interactive run.
    pub background_safe: bool,
}

/// SDK-owned safety and runtime metadata attached to a [`Tool`].
///
/// A tool advertises its policy through [`Tool::policy`]. The default is
/// **unclassified** (`classified == false`): policy enforcement can be
/// configured to fail closed on unclassified tools so that adding a new tool
/// without declaring its safety profile does not silently widen the attack
/// surface. The plain [`ToolSchema`] remains the model-visible projection;
/// `ToolPolicy` is the audit/enforcement projection and is fully serializable
/// for registry introspection.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    /// Whether the tool author has explicitly classified this policy. A default
    /// (`false`) marks the tool as *unclassified*, which strict enforcement
    /// treats as untrusted.
    pub classified: bool,
    /// Declared side effects.
    pub side_effects: ToolSideEffects,
    /// Declared runtime requirements.
    pub runtime: ToolRuntime,
    /// Declared access requirements.
    pub access: ToolAccess,
}

/// An incremental progress update emitted while a tool runs (streaming).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDelta {
    /// Id of the [`ToolCall`] this delta belongs to.
    pub call_id: String,
    /// Incremental content fragment.
    pub content: String,
}

/// A tool the harness can invoke during an agent loop.
///
/// Generic over the application `State` so tools can read shared context
/// without exposing it to model-visible schemas.
#[async_trait]
pub trait Tool<State: Send + Sync>: Send + Sync {
    /// Canonical tool name.
    fn name(&self) -> &str;

    /// Human/model readable description.
    fn description(&self) -> &str;

    /// Returns the model-visible schema for this tool.
    fn schema(&self) -> ToolSchema;

    /// Returns the SDK-owned safety/runtime policy for this tool.
    ///
    /// The default is an *unclassified* [`ToolPolicy`]; tools that touch the
    /// filesystem, network, money, or otherwise carry risk should override this
    /// so policy enforcement can make fail-closed decisions. Enforcement lives in
    /// [`ToolPolicyMiddleware`][crate::harness::middleware::ToolPolicyMiddleware].
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::default()
    }

    /// Executes the tool against application state and a validated call.
    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult>;

    /// Executes the tool with access to caller run context.
    ///
    /// Implementors that do not need caller lineage can rely on the default,
    /// which delegates to [`Self::call`].
    async fn call_with_context(
        &self,
        state: &State,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> Result<ToolResult> {
        let _ = context;
        self.call(state, call).await
    }
}

/// A name-keyed registry of tools available to the harness.
pub struct ToolRegistry<State> {
    pub(crate) tools: HashMap<String, Arc<dyn Tool<State>>>,
}
