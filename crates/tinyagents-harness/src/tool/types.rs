//! Tool layer types used by the harness.
//!
//! These types define the call boundary every harness capability shares —
//! including sub-agents exposed as tools (see
//! [`crate::subagent::SubAgentTool`]), which is how the recursive
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
use crate::cancel::CancellationToken;
use crate::context::RunContext;
use crate::events::EventSink;
use crate::ids::{RunId, ThreadId};
use crate::tool::{ToolErrorPolicy, context_detail_from_args, humanize_tool_name};

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
    ///
    /// When [`Self::invalid`] is set, this preserves the raw (unparseable)
    /// arguments string the model emitted (as a JSON string value) instead of a
    /// parsed object.
    #[serde(default)]
    pub arguments: Value,
    /// Set when the model emitted arguments that could not be parsed as JSON.
    ///
    /// Small local models (Ollama, LM Studio, llama.cpp, vLLM) occasionally emit
    /// malformed argument JSON. Rather than fail the whole model call, the
    /// provider surfaces the call with this error message and the raw arguments
    /// preserved in [`Self::arguments`]; the agent loop feeds the message back to
    /// the model as an error tool result so it can retry (mirroring LangChain's
    /// `invalid_tool_calls` and the AI SDK's invalid dynamic tool parts). `None`
    /// on a well-formed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid: Option<String>,
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

/// `raw` key under which [`ToolResult::mark_trusted_verbatim`] records the
/// request that this result reach the model byte-for-byte.
///
/// Kept in `raw` rather than as a dedicated field so the opt-in costs nothing
/// for the results that never use it, and so a host can round-trip the flag
/// through its own serialisation without a schema change.
pub const TRUSTED_VERBATIM_KEY: &str = "trusted_verbatim";

impl ToolResult {
    /// Asks hosts to deliver this result's content to the model unchanged.
    ///
    /// A host is free to summarise, truncate, batch, or re-frame tool output —
    /// and usually should. This marks the results where that is not safe: an
    /// input schema whose argument names the model must copy exactly, a
    /// signature, a diff. Reshaping those yields content that still reads well
    /// and is wrong.
    ///
    /// Advisory: the crate carries the flag to [`super::super::message::ToolMessage`]
    /// via [`tinyinference::message::Message::tool_from_result`]; honouring it
    /// is the host's job.
    /// # Returns
    ///
    /// `true` when the request was recorded. `false` when [`Self::raw`] already
    /// holds a **non-object** value (a string, array, or number): the flag lives
    /// under a key in `raw`, and a scalar has nowhere to put a key. The crate
    /// will not clobber a value the tool deliberately set, so the request is
    /// refused rather than silently lost — make room by moving that value under
    /// a key of its own, then call this again.
    ///
    /// `#[must_use]` precisely because the failure is otherwise invisible:
    /// [`Self::is_trusted_verbatim`] would keep answering `false` with no
    /// diagnostic, and the caller would believe the opt-in took effect.
    #[must_use = "returns false when `raw` holds a non-object value and the request was refused"]
    pub fn mark_trusted_verbatim(&mut self) -> bool {
        let raw = self
            .raw
            .get_or_insert_with(|| Value::Object(Default::default()));
        match raw.as_object_mut() {
            Some(map) => {
                map.insert(TRUSTED_VERBATIM_KEY.to_string(), Value::Bool(true));
                true
            }
            None => false,
        }
    }

    /// Whether [`Self::mark_trusted_verbatim`] was set on this result.
    ///
    /// `false` when `raw` is absent, is not an object, or holds anything other
    /// than `true` — an opt-in that cannot be tripped by unrelated metadata.
    pub fn is_trusted_verbatim(&self) -> bool {
        self.raw
            .as_ref()
            .and_then(|raw| raw.get(TRUSTED_VERBATIM_KEY))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
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
    /// The caller run's cancellation token. A recursive tool such as a
    /// sub-agent installs this on its child run so one `cancel()` unwinds the
    /// whole nested-run tree, as
    /// [`crate::cancel`] documents.
    pub cancellation: CancellationToken,
    /// Whether the caller run is being driven through the streaming loop path.
    /// A sub-agent tool uses this to run its child in the matching mode so the
    /// child's deltas propagate onto the shared [`EventSink`].
    pub streaming: bool,
    /// The isolated workspace/sandbox the tool may operate in, when the run was
    /// configured with a
    /// [`WorkspaceIsolation`][crate::workspace::WorkspaceIsolation]
    /// provider. A tool discovers its allowed root here instead of an
    /// application global; `None` means no workspace policy is in effect.
    pub workspace: Option<crate::workspace::WorkspaceDescriptor>,
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
            cancellation: ctx.cancellation.clone(),
            streaming: ctx.streaming,
            workspace: ctx.workspace.clone(),
        }
    }

    /// Attaches an isolated workspace descriptor the tool may operate in.
    pub fn with_workspace(mut self, workspace: crate::workspace::WorkspaceDescriptor) -> Self {
        self.workspace = Some(workspace);
        self
    }
}

/// Lets a tool read this context without depending on the harness.
///
/// A tool is written against `tinytools`, which cannot name this type: this
/// crate depends on `tinytools`, so an edge back would be a cycle. The
/// vocabulary therefore declares a narrow trait and this crate implements it,
/// which is what lets a host hand a live [`ToolExecutionContext`] to a tool
/// that has never heard of the harness.
///
/// Only the facts a tool actually reads are exposed. The run id, event sink,
/// cancellation token and streaming flag stay harness-internal — a tool that
/// wanted them would be reaching into the run rather than doing its job.
///
/// `workspace` needs no conversion: [`WorkspaceDescriptor`] is `tinytools`'
/// type, re-exported by this crate, so the field is already the right one.
impl tinytools::ToolRunContext for ToolExecutionContext {
    fn workspace(&self) -> Option<&tinytools::WorkspaceDescriptor> {
        self.workspace.as_ref()
    }

    fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_ref().map(ThreadId::as_str)
    }

    fn max_turn_output_tokens(&self) -> Option<u32> {
        self.max_turn_output_tokens
    }
}

// `SandboxMode` rides on `WorkspaceDescriptor`, which is `tinytools`' type, so
// the mode has to be the same type on both sides of that field.
pub use tinytools::SandboxMode;

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
/// [`ToolPolicyMiddleware`][crate::middleware::ToolPolicyMiddleware]) to
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

/// How the harness should bound a single tool invocation in wall-clock time.
///
/// Most tools should inherit the run's global tool timeout. Long-running
/// scripting or build tools can opt out with [`ToolTimeout::Unbounded`] when the
/// caller did not supply a deadline, and can return [`ToolTimeout::Millis`] for
/// an explicit per-call budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "timeout_ms")]
pub enum ToolTimeout {
    /// Use the run/global timeout policy.
    #[default]
    Inherit,
    /// Run without a harness-imposed wall-clock deadline.
    Unbounded,
    /// Enforce this exact deadline in milliseconds.
    Millis(u64),
}

/// Human-facing presentation metadata for a tool invocation.
///
/// This metadata is never sent to the model as part of [`ToolSchema`]. It is
/// intended for timelines, audit logs, and application UIs that need compact
/// labels such as `Read(src/lib.rs)` instead of raw machine names.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDisplay {
    /// Short verb phrase or title-cased label shown for the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional static detail. Dynamic details usually come from call args via
    /// [`Tool::display_detail`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Runtime requirements a tool declares for safe execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRuntime {
    /// Suggested per-call wall-clock timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Invocation timeout behavior when a simple numeric timeout is not enough.
    #[serde(default, skip_serializing_if = "ToolTimeout::is_inherit")]
    pub timeout: ToolTimeout,
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
    /// Human-facing presentation metadata.
    #[serde(default, skip_serializing_if = "ToolDisplay::is_empty")]
    pub display: ToolDisplay,
}

/// An incremental progress update emitted while a tool runs (streaming).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDelta {
    /// Id of the [`ToolCall`] this delta belongs to.
    pub call_id: String,
    /// Incremental content fragment.
    pub content: String,
    /// The tool's name, when known. Providers that surface it on the first
    /// (call-opening) delta populate this so consumers can label a tool call as
    /// soon as it begins — before its arguments have streamed. Subsequent
    /// argument fragments leave it `None`; [`StreamAccumulator`] remembers the
    /// first non-empty name per `call_id` and stamps it onto the reconstructed
    /// [`ToolCall`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
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
    /// [`ToolPolicyMiddleware`][crate::middleware::ToolPolicyMiddleware].
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::default()
    }

    /// Returns the human-facing label for this specific call.
    ///
    /// The default prefers [`ToolPolicy::display`] and otherwise derives a
    /// compact title-cased label from [`Self::name`]. Applications can use this
    /// for timelines and audit logs without exposing presentation text to model
    /// tool declarations.
    fn display_label(&self, _call: &ToolCall) -> Option<String> {
        self.policy()
            .display
            .label
            .or_else(|| Some(humanize_tool_name(self.name())))
    }

    /// Returns the human-facing detail for this specific call.
    ///
    /// The default prefers a static [`ToolPolicy::display`] detail and
    /// otherwise extracts the most relevant common argument from the call.
    fn display_detail(&self, call: &ToolCall) -> Option<String> {
        self.policy()
            .display
            .detail
            .or_else(|| context_detail_from_args(&call.arguments))
    }

    /// Returns the invocation timeout behavior for this specific call.
    ///
    /// The default reads [`ToolPolicy::runtime`]. Static `timeout_ms` values are
    /// promoted to [`ToolTimeout::Millis`] for callers that consume the richer
    /// timeout vocabulary; tools with argument-dependent deadlines can override
    /// this method.
    fn timeout_policy(&self, _call: &ToolCall) -> ToolTimeout {
        let runtime = self.policy().runtime;
        match (runtime.timeout, runtime.timeout_ms) {
            (ToolTimeout::Inherit, Some(timeout_ms)) => ToolTimeout::Millis(timeout_ms),
            (timeout, _) => timeout,
        }
    }

    /// Names of arguments this tool receives from the **host**, never from the
    /// model.
    ///
    /// These keys are projected out of the model-facing declaration by
    /// [`ToolRegistry::schemas`] — removed from both `properties` and
    /// `required` — so the model neither sees them nor is asked to supply them.
    /// The host fills them in at execution time.
    ///
    /// This is the declarative replacement for reaching around the argument
    /// schema into [`ToolExecutionContext`], which is what a tool needing the
    /// caller's thread id or recursion depth has to do today. Port of
    /// LangChain's `InjectedToolArg` / `InjectedToolCallId`.
    ///
    /// # Execution-time contract
    ///
    /// A model-supplied value for an injected key must be **stripped before
    /// schema validation**, then replaced by the host's real value — see the
    /// ordering rule in [`crate::tool::injected`]. Getting that order
    /// wrong lets a model forge a hidden argument.
    ///
    /// The default is an empty list: a tool opts in.
    fn injected_arguments(&self) -> &[&str] {
        &[]
    }

    /// What the harness should do when this tool returns `Err`.
    ///
    /// The default is [`ToolErrorPolicy::Fail`], preserving today's behaviour
    /// exactly: an `Err` propagates and ends the run. A tool whose failures are
    /// routine and recoverable — a lookup that misses, a network call that
    /// times out — should return [`ToolErrorPolicy::ReturnToError`] so the
    /// model sees the failure and can adapt, or
    /// [`ToolErrorPolicy::Message`] to show a fixed message instead of an error
    /// text that may not be safe to expose.
    ///
    /// [`TinyAgentsError::Cancelled`][crate::error::TinyAgentsError::Cancelled]
    /// and
    /// [`TinyAgentsError::Interrupted`][crate::error::TinyAgentsError::Interrupted]
    /// are re-raised regardless of this policy; see
    /// [`ToolErrorPolicy::apply`].
    fn error_policy(&self) -> ToolErrorPolicy {
        ToolErrorPolicy::default()
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
