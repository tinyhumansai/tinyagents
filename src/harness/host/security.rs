//! Host policy consulted before the runtime widens what an agent can reach.
//!
//! See [`SecurityGate`] for the trait contract and
//! [`RootContainedSecurityGate`] for the fail-closed default.

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};
use crate::harness::ids::{CallId, RunId, ThreadId};

/// Host policy consulted before the runtime widens what an agent can see or
/// touch.
///
/// The crate defines no policy of its own here: it defines the questions and
/// fails closed on the answers. Complements
/// [`WorkspaceIsolation`][crate::harness::workspace::WorkspaceIsolation],
/// which prepares an environment; this decides what is permitted inside one.
///
/// Only [`authorize_call`][Self::authorize_call] is `async`: it is the one
/// question a realistic host answers with I/O (a policy store lookup, or an
/// operator approval round-trip). Tool-set narrowing, path resolution, input
/// screening, and redaction are in-process inspections, and forcing them
/// `async` would push `async` up through synchronous session-construction and
/// artifact-persistence paths for no benefit.
#[async_trait]
pub trait SecurityGate<State: Send + Sync>: Send + Sync {
    /// Narrows the tool set advertised to a model.
    ///
    /// This is *advertisement*, not enforcement: it decides what the model is
    /// told exists. Enforcement of an individual call — which depends on the
    /// arguments, not just the name — is
    /// [`authorize_call`][Self::authorize_call], and a gate that narrows here
    /// without also deciding there admits anything the model names from
    /// memory.
    ///
    /// The default exposes everything.
    fn filter_tools(
        &self,
        state: &State,
        request: &ToolExposureRequest<'_>,
    ) -> Result<ToolExposure> {
        let _ = state;
        Ok(ToolExposure {
            visible: request.available.to_vec(),
            withheld: Vec::new(),
            boundary_note: None,
        })
    }

    /// Decides whether one concrete tool call may proceed.
    ///
    /// Three-valued on purpose: a policy that can only allow or deny cannot
    /// express "ask a human first", which then silently degrades into one of
    /// the other two. The decision sees the arguments, because the same tool
    /// name is routinely both safe and unsafe depending on them.
    ///
    /// The default allows every call.
    async fn authorize_call(
        &self,
        state: &State,
        request: &ToolCallRequest<'_>,
    ) -> Result<CallVerdict> {
        let _ = (state, request);
        Ok(CallVerdict::Allow)
    }

    /// Resolves a caller-supplied path to an absolute path the run may use, or
    /// errors. Implementations must reject traversal out of `root` and must not
    /// depend on the target existing.
    fn resolve_path(&self, state: &State, request: &PathRequest<'_>) -> Result<PathBuf>;

    /// Screens untrusted inbound text before it becomes model input.
    ///
    /// The default admits everything.
    fn screen_input(
        &self,
        state: &State,
        request: &InputScreenRequest<'_>,
    ) -> Result<InputVerdict> {
        let _ = (state, request);
        Ok(InputVerdict::Admit)
    }

    /// Rewrites text the runtime is about to persist, render, or feed back to a
    /// model — masking secrets, or fencing untrusted content so a model treats
    /// it as data rather than instruction.
    ///
    /// Separate from [`screen_input`][Self::screen_input] because the answer is
    /// modified text rather than a verdict, and because it runs in both
    /// directions: inbound text on its way into a prompt, and outbound tool
    /// output on its way into storage or a preview.
    ///
    /// The default returns the text unchanged.
    fn redact(&self, state: &State, request: &RedactionRequest<'_>) -> Result<Redaction> {
        let _ = state;
        Ok(Redaction::unchanged(request.text))
    }
}

/// The tool set a run could advertise, and how it was entered.
#[derive(Clone, Debug)]
pub struct ToolExposureRequest<'a> {
    /// The run whose tool set is being narrowed.
    pub run_id: &'a RunId,
    /// Host-defined identity of the agent taking the run.
    pub agent_id: &'a str,
    /// Host-defined label for how this run was entered.
    pub entrypoint: &'a str,
    /// Every tool name registered on the run, before narrowing.
    pub available: &'a [String],
}

/// The narrowed tool set, plus optional host-authored explanation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolExposure {
    /// Names the model is told about.
    pub visible: Vec<String>,
    /// Names deliberately withheld, for logging and boundary rendering.
    #[serde(default)]
    pub withheld: Vec<String>,
    /// Optional host-authored text describing the restriction, for the caller
    /// to render into the prompt. The crate never generates this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_note: Option<String>,
}

/// One concrete tool call awaiting authorization.
#[derive(Clone, Debug)]
pub struct ToolCallRequest<'a> {
    /// The run issuing the call.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Provider-assigned identity of this call.
    pub call_id: &'a CallId,
    /// Name of the tool the model asked for.
    pub tool_name: &'a str,
    /// Arguments the model supplied. Policy decisions routinely depend on
    /// these, not only on `tool_name`.
    pub arguments: &'a Value,
    /// Host-defined label for how this run was entered.
    pub entrypoint: &'a str,
}

/// The decision on one tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "verdict")]
pub enum CallVerdict {
    /// Execute the call.
    Allow,
    /// Refuse the call outright. `code` is a stable host-defined discriminant;
    /// `message` is host-authored text the runtime may hand back to the model
    /// in place of a result.
    Deny {
        /// Stable host-defined discriminant.
        code: String,
        /// Host-authored explanation.
        message: String,
    },
    /// Hold the call pending an out-of-band decision. The runtime does not
    /// define how approval is obtained; it only distinguishes "not now" from
    /// "never" so a held call is not reported to the model as a refusal.
    RequireApproval {
        /// Stable host-defined discriminant.
        code: String,
        /// Host-authored explanation.
        message: String,
    },
}

/// Whether a path is being resolved for reading or for writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathIntent {
    /// The caller intends to read the resolved path.
    Read,
    /// The caller intends to create or modify the resolved path.
    Write,
}

/// A caller-supplied path awaiting containment checks.
#[derive(Clone, Debug)]
pub struct PathRequest<'a> {
    /// The run requesting the path.
    pub run_id: &'a RunId,
    /// The path as supplied, relative or absolute.
    pub path: &'a Path,
    /// The root the resolved path must stay inside.
    pub root: &'a Path,
    /// What the caller intends to do with the result.
    pub intent: PathIntent,
}

/// Untrusted inbound text awaiting screening.
#[derive(Clone, Debug)]
pub struct InputScreenRequest<'a> {
    /// The run the text would enter.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Host-defined label for where the text came from.
    pub source: &'a str,
    /// The text itself.
    pub text: &'a str,
}

/// The decision on one piece of inbound text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "verdict")]
pub enum InputVerdict {
    /// Let the text through unchanged.
    Admit,
    /// Refuse the input. `code` is a stable host-defined discriminant;
    /// `message` is host-authored user-facing text.
    Refuse {
        /// Stable host-defined discriminant.
        code: String,
        /// Host-authored user-facing text.
        message: String,
    },
}

/// Which way text is flowing through [`SecurityGate::redact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionDirection {
    /// Text on its way into a model request.
    Inbound,
    /// Text on its way out of the runtime — persisted, previewed, or logged.
    Outbound,
}

/// Text awaiting redaction, with the direction it is travelling.
#[derive(Clone, Debug)]
pub struct RedactionRequest<'a> {
    /// The run the text belongs to.
    pub run_id: &'a RunId,
    /// Thread the run belongs to, when it belongs to one.
    pub thread_id: Option<&'a ThreadId>,
    /// Which way the text is flowing.
    pub direction: RedactionDirection,
    /// Host-defined label for what produced the text.
    pub source: &'a str,
    /// The text itself.
    pub text: &'a str,
}

/// The result of a redaction pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Redaction {
    /// The text to use from here on.
    pub value: String,
    /// `true` when `value` differs from the input, so callers can stamp a
    /// "modified" marker without diffing.
    pub changed: bool,
    /// Optional host-authored note describing what was changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Redaction {
    /// Returns `text` untouched.
    pub fn unchanged(text: &str) -> Self {
        Self {
            value: text.to_string(),
            changed: false,
            note: None,
        }
    }

    /// Returns `value` as a modification of the original text.
    pub fn changed(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            changed: true,
            note: None,
        }
    }

    /// Attaches a host-authored note describing the change.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// A [`SecurityGate`] whose only real decision is filesystem containment.
///
/// [`resolve_path`][SecurityGate::resolve_path] normalises the request
/// lexically — dropping `.`, popping on `..`, rejecting embedded roots — and
/// errors with [`TinyAgentsError::Validation`] if the result would escape
/// `root`. The check never touches the filesystem, so a path that does not
/// exist yet resolves exactly like one that does; it is also therefore not a
/// defence against symlinks, which a host that cares about them must layer on.
///
/// Every other method takes the permissive trait default: this gate fails
/// closed on the one question it actually answers and declines to invent policy
/// for the rest.
#[derive(Clone, Copy, Debug, Default)]
pub struct RootContainedSecurityGate;

impl RootContainedSecurityGate {
    /// Creates the gate.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State: Send + Sync> SecurityGate<State> for RootContainedSecurityGate {
    fn resolve_path(&self, _state: &State, request: &PathRequest<'_>) -> Result<PathBuf> {
        let escape = || {
            TinyAgentsError::Validation(format!(
                "path {} escapes root {}",
                request.path.display(),
                request.root.display()
            ))
        };

        // An absolute request is only meaningful if it is already inside the
        // root; strip the root off and treat the remainder as relative so the
        // component walk below is the single containment check.
        let relative = if request.path.is_absolute() {
            request
                .path
                .strip_prefix(request.root)
                .map_err(|_| escape())?
        } else {
            request.path
        };

        let mut resolved = request.root.to_path_buf();
        for component in relative.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(part) => resolved.push(part),
                Component::ParentDir => {
                    if !resolved.pop() || !resolved.starts_with(request.root) {
                        return Err(escape());
                    }
                }
                Component::RootDir | Component::Prefix(_) => return Err(escape()),
            }
        }

        if !resolved.starts_with(request.root) {
            return Err(escape());
        }
        Ok(resolved)
    }
}
