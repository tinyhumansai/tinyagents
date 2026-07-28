//! System-prompt and per-turn context assembly supplied by the embedder.
//!
//! See [`ContextComposer`] for the trait contract and
//! [`PassthroughContextComposer`] for the inert default.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::harness::ids::{RunId, ThreadId};
use crate::harness::tool::ToolSchema;
use crate::harness::workspace::WorkspaceDescriptor;

/// Builds the run's system prompt and the ordered, per-turn context fragments
/// that precede user input.
///
/// This is the typed turn-preparation seam: enrichment becomes an ordered set
/// of registered fragments with placement and priority instead of a bespoke
/// method body. The crate owns ordering, placement, and assembly; the embedder
/// owns every byte of text.
#[async_trait]
pub trait ContextComposer<State: Send + Sync>: Send + Sync {
    /// Renders the run's system prompt. Called once per run so the prompt
    /// prefix stays byte-stable across turns for provider prompt caching.
    ///
    /// Synchronous by contract. Every input is already in hand at the call
    /// site, and hosts have cold-boot resume paths that build a prompt without
    /// being allowed to fan out to a memory store; an `async` signature would
    /// make that fan-out structurally unavoidable. Per-turn work that genuinely
    /// needs I/O belongs in [`prepare_turn`][Self::prepare_turn].
    fn compose_system_prompt(
        &self,
        state: &State,
        request: &SystemPromptRequest<'_>,
    ) -> Result<String>;

    /// Produces this turn's context fragments.
    ///
    /// The default returns nothing, so a host that only needs a system prompt
    /// implements one method.
    async fn prepare_turn(
        &self,
        state: &State,
        request: &TurnPreparationRequest<'_>,
    ) -> Result<TurnPreparation> {
        let _ = (state, request);
        Ok(TurnPreparation::default())
    }
}

/// Everything a composer may consult when rendering the run's system prompt.
#[derive(Clone, Debug)]
pub struct SystemPromptRequest<'a> {
    /// The run being started.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Host-defined identity of the agent being instantiated.
    pub agent_id: &'a str,
    /// Model the run will call.
    pub model_id: &'a str,
    /// Full schemas for the tools registered on the run.
    pub tools: &'a [ToolSchema],
    /// Names the policy layer decided to advertise; see
    /// [`SecurityGate::filter_tools`][super::SecurityGate::filter_tools].
    pub visible_tool_names: &'a [String],
    /// Dispatcher-rendered instructions for invoking tools, when the model is
    /// driven prompt-guided rather than with native tool calls.
    pub tool_call_instructions: &'a str,
    /// The environment the run may touch, when one was prepared. Carries the
    /// primary root plus any additional trusted roots; the crate models no
    /// other named root.
    pub workspace: Option<&'a WorkspaceDescriptor>,
}

/// Everything a composer may consult when preparing one turn.
#[derive(Clone, Debug)]
pub struct TurnPreparationRequest<'a> {
    /// The run this turn belongs to.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Host-defined identity of the agent taking the turn.
    pub agent_id: &'a str,
    /// The user input this turn will act on.
    pub input: &'a str,
    /// Zero-based position of this turn within the run.
    pub turn_index: u32,
    /// `true` when no prior turn exists in this run's history.
    pub first_turn: bool,
    /// `true` when history was seeded from durable storage rather than built
    /// in-process; a first turn of a resumed thread is both.
    pub resumed: bool,
}

/// The fragments one turn contributes, plus opaque host passthrough.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnPreparation {
    /// Fragments to splice into the request, in any order — the runtime sorts
    /// them by placement and priority.
    #[serde(default)]
    pub blocks: Vec<ContextBlock>,
    /// Opaque host payload returned to the caller untouched.
    ///
    /// Deliberately untyped: presentation contracts (source attribution chips,
    /// cost footers, timeline rows) are the embedder's, and the crate must not
    /// grow a rendering opinion by modelling them.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub extras: Value,
}

impl TurnPreparation {
    /// Creates a preparation carrying `blocks` and no extras.
    pub fn new(blocks: Vec<ContextBlock>) -> Self {
        Self {
            blocks,
            extras: Value::Null,
        }
    }

    /// Attaches an opaque host payload.
    pub fn with_extras(mut self, extras: Value) -> Self {
        self.extras = extras;
        self
    }
}

/// One rendered fragment of turn context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextBlock {
    /// Stable identity for the fragment, used for de-duplication and logging.
    pub id: String,
    /// Host-authored text, spliced in verbatim.
    pub body: String,
    /// Where the fragment is spliced into the request.
    #[serde(default)]
    pub placement: ContextPlacement,
    /// Higher sorts earlier within a placement; ties keep insertion order.
    #[serde(default)]
    pub priority: i32,
}

impl ContextBlock {
    /// Creates a [`ContextPlacement::TurnPrefix`] block at priority `0`.
    pub fn new(id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            body: body.into(),
            placement: ContextPlacement::default(),
            priority: 0,
        }
    }

    /// Sets where the fragment is spliced in.
    pub fn with_placement(mut self, placement: ContextPlacement) -> Self {
        self.placement = placement;
        self
    }

    /// Sets the sort priority within the fragment's placement.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Where a fragment is spliced into the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextPlacement {
    /// Prepended to the turn's user message. Per-turn content belongs here:
    /// putting it in the system prompt invalidates the cached prefix, and
    /// history trimming commonly hoists system messages to the front, which
    /// reorders content that was meant to ride one specific turn.
    #[default]
    TurnPrefix,
    /// Prepended to the system prompt. Run-stable content only.
    SystemPrefix,
}

/// A [`ContextComposer`] that contributes nothing.
///
/// `compose_system_prompt` returns an empty string and `prepare_turn` takes the
/// trait default, so composing it is observationally identical to composing no
/// composer at all.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassthroughContextComposer;

impl PassthroughContextComposer {
    /// Creates the composer.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> ContextComposer<State> for PassthroughContextComposer {
    fn compose_system_prompt(
        &self,
        _state: &State,
        _request: &SystemPromptRequest<'_>,
    ) -> Result<String> {
        Ok(String::new())
    }
}
