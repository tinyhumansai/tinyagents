//! Host capability: choosing *which* model answers a turn.
//!
//! [`tinyinference::model::ChatModel`] already covers *how* a model is called.
//! What the runtime cannot supply is the decision of **which** model a given
//! turn should use. That decision is product policy — route the lead agent to a
//! frontier model and its subagents to a cheap one, send background/"thinking"
//! work to a lower tier, pin a particular role to a local model — and it is
//! exactly the kind of rule that differs per host and changes without notice.
//! Encoding any of it here would bake one product's economics into a
//! redistributed crate.
//!
//! So the seam is a question, not an answer: the runtime describes the turn it
//! is about to run ([`ModelResolveRequest`]) and the host hands back a model to
//! call. The runtime never inspects the reasoning, never learns which tier it
//! got, and never caches the decision — a host is free to answer differently
//! for two identical requests (A/B routing, failover after an outage, a quota
//! that just tripped).
//!
//! # When a host supplies this
//!
//! Always. Unlike the optional sinks, `ModelResolver` is a **required**
//! capability: a turn with no model cannot run at all, so there is no
//! meaningful "absent" state to distinguish from failure. A host with exactly
//! one model wires [`FixedModelResolver`] and is done; a host with routing
//! rules implements the trait over its own policy.
//!
//! # Naming hazard — read before "fixing" this
//!
//! The crate already has [`tinyinference::model::ModelRequest`], and it is a
//! **different thing**: that type is the provider call payload (messages,
//! tools, sampling parameters) handed to a model that has already been chosen.
//! The type here, [`ModelResolveRequest`], is the *routing question* asked
//! before any payload exists. The near-miss in names is unfortunate but the
//! two are not interchangeable, and renaming this one to `ModelRequest` would
//! collide outright. Keep the `Resolve` in the name.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use tinyinference::model::ChatModel;

// ── ModelResolveRequest ───────────────────────────────────────────────────────

/// The routing question: everything the host needs to pick a model for one turn.
///
/// Deliberately tiny and inert (serde + std only). It carries *identity and
/// position*, never policy: there is no "tier", no "cheap/expensive" flag, no
/// budget hint. Those are conclusions the host draws from these facts, and
/// putting them here would mean the runtime had already made the decision it is
/// supposed to be delegating.
///
/// [`model_pin`](Self::model_pin) is consistent with that rule rather than an
/// exception to it. A pin is a *declared fact* — this agent's definition names
/// this model — not a conclusion the runtime reached. Passing it along still
/// leaves the host to decide whether to honour it; see the field docs.
///
/// Fields are public so a host can pattern-match without accessor ceremony; the
/// builder methods exist for call sites that construct one inline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelResolveRequest {
    /// Identifier of the agent about to take the turn.
    ///
    /// A plain `String` because **this crate has no `AgentId` newtype** — the
    /// RFC's signature writes `&AgentId`, but introducing one here purely for
    /// this trait would create a second identity type competing with the host's
    /// own. Ids are opaque to the runtime: never parsed, never compared against
    /// a known list, only passed through.
    pub agent_id: String,

    /// Host-defined role of the agent, when the host models roles at all.
    ///
    /// `None` means "no role supplied", which is distinct from a role whose
    /// name happens to be empty. The runtime does not interpret this string —
    /// role vocabularies are product taxonomy and differ per host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Whether this agent is the lead of its team rather than a delegate.
    ///
    /// Kept as a separate flag instead of a magic role name because leadership
    /// is *structural* — the runtime genuinely knows it from the call graph,
    /// whereas a role is something only the host can assert. It is the one
    /// routing input the runtime can supply honestly, and it is the split most
    /// hosts key on first (lead model vs subagent model).
    #[serde(default)]
    pub is_team_lead: bool,

    /// Exact model id the agent's definition pinned, if any.
    ///
    /// Distinct from [`role`](Self::role) and the distinction is the whole
    /// point: a role is a **host taxonomy** ("chat", "background", "thinking"),
    /// whereas this is a **concrete model id** (`claude-3-5-sonnet`, a BYOK id,
    /// a local model name). Before this field existed, `role` was the only
    /// string on the request, so a wiring author needing a pin honoured would
    /// naturally put the model id there — and a host resolver reasonably
    /// treating `role` as a role vocabulary would fail to recognise it and fall
    /// back to a default. The pin would be dropped silently, which is precisely
    /// the ambiguity this field removes.
    ///
    /// **Advisory, not binding.** The host decides whether it can honour the
    /// pin — the model may be unconfigured, the credentials absent, the
    /// provider down, or the id simply unknown — and is free to resolve
    /// something else or return an error. Keeping that judgement host-side is
    /// deliberate: the runtime has no view of credentials or provider health,
    /// so a runtime that honoured pins itself would route to models the host
    /// cannot actually call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_pin: Option<String>,
}

impl ModelResolveRequest {
    /// A request for `agent_id` with no role, no model pin, and not a team lead.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            role: None,
            is_team_lead: false,
            model_pin: None,
        }
    }

    /// Sets the host-defined role.
    ///
    /// Pass a *role*, never a model id — see
    /// [`with_model_pin`](Self::with_model_pin) for the latter.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Sets the exact model id the agent's definition pinned.
    pub fn with_model_pin(mut self, model: impl Into<String>) -> Self {
        self.model_pin = Some(model.into());
        self
    }

    /// Marks this agent as the lead of its team.
    pub fn as_team_lead(mut self) -> Self {
        self.is_team_lead = true;
        self
    }

    /// The role as a borrowed string, or `None` when unset.
    ///
    /// A blank or whitespace-only role is reported as `None`: a host mapping a
    /// missing field to `Some("")` almost certainly means "no role", and
    /// letting that reach a routing `match` would silently select a
    /// role-specific branch keyed on the empty string.
    pub fn role(&self) -> Option<&str> {
        self.role
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty())
    }

    /// The pinned model id as a borrowed string, or `None` when unset.
    ///
    /// Blank-trims for the same reason [`role`](Self::role) does, and the
    /// consequence here is worse: a definition with `model = ""` reaching a
    /// resolver as `Some("")` would be looked up as a model literally named
    /// empty-string, missing, and reported as an unroutable pin — turning a
    /// cosmetic config blank into a failed turn instead of the intended
    /// "no pin, route normally".
    pub fn model_pin(&self) -> Option<&str> {
        self.model_pin
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
    }
}

// ── ModelResolver ─────────────────────────────────────────────────────────────

/// Resolves the model that should answer a turn.
///
/// Generic over the application `State` for the same reason
/// [`ChatModel`] is: the returned model is invoked with the host's state, so
/// the resolver has to produce a model of the *same* state type. This is why
/// the trait cannot be state-erased into a single object shared across
/// differently-typed harnesses.
///
/// Returning `Arc` (not `Box`) is deliberate: a host almost always hands back
/// one of a small fixed set of long-lived clients, and a per-turn clone of a
/// provider client would throw away its connection pool.
///
/// # Errors
///
/// Return an error only when no model can be produced — an unroutable agent,
/// an unconfigured provider, a tripped quota. There is no `Ok(None)` here on
/// purpose: unlike the optional capabilities, "no model" is never a normal
/// state the runtime can proceed past, so it must not be expressible as a
/// success.
#[async_trait]
pub trait ModelResolver<State: Send + Sync>: Send + Sync {
    /// Chooses the model for the turn described by `req`.
    ///
    /// Called once per model call, not once per session — a host may return
    /// different models across the turns of a single conversation (escalating
    /// after a failure, downgrading once a budget tightens). Implementations
    /// should therefore be cheap and must not assume the answer is memoized by
    /// the caller.
    async fn resolve(&self, req: &ModelResolveRequest) -> Result<Arc<dyn ChatModel<State>>>;
}

// ── FixedModelResolver ────────────────────────────────────────────────────────

/// The default resolver: always returns the same model, whatever is asked.
///
/// This is the honest single-model case, not a stub — a host with one
/// configured model has genuinely finished routing. It ignores every field of
/// [`ModelResolveRequest`], which is the correct behaviour rather than a
/// limitation to work around: a host that wants role-based or lead/subagent
/// routing implements [`ModelResolver`] itself instead of teaching this type
/// about product tiers.
pub struct FixedModelResolver<State: Send + Sync> {
    model: Arc<dyn ChatModel<State>>,
}

impl<State: Send + Sync> FixedModelResolver<State> {
    /// Wraps `model` as the answer to every resolution request.
    pub fn new(model: Arc<dyn ChatModel<State>>) -> Self {
        Self { model }
    }

    /// The wrapped model.
    pub fn model(&self) -> &Arc<dyn ChatModel<State>> {
        &self.model
    }
}

// Hand-written: `#[derive(Clone)]` would demand `State: Clone`, which is wrong —
// only the `Arc` is cloned and `State` never appears by value.
impl<State: Send + Sync> Clone for FixedModelResolver<State> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
        }
    }
}

// Hand-written for the same reason, and because `dyn ChatModel` is not `Debug`.
impl<State: Send + Sync> std::fmt::Debug for FixedModelResolver<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedModelResolver").finish_non_exhaustive()
    }
}

#[async_trait]
impl<State: Send + Sync> ModelResolver<State> for FixedModelResolver<State> {
    async fn resolve(&self, _req: &ModelResolveRequest) -> Result<Arc<dyn ChatModel<State>>> {
        Ok(Arc::clone(&self.model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tinyinference::model::{ModelRequest, ModelResponse};

    /// Minimal model double: replies with a fixed string so a resolved model can
    /// be identified by the text it produces as well as by pointer identity.
    struct EchoModel(&'static str);

    #[async_trait]
    impl<State: Send + Sync> ChatModel<State> for EchoModel {
        async fn invoke(
            &self,
            _state: &State,
            _request: ModelRequest,
        ) -> tinyinference::Result<ModelResponse> {
            Ok(ModelResponse::assistant(self.0))
        }
    }

    fn fixed(reply: &'static str) -> FixedModelResolver<()> {
        FixedModelResolver::new(Arc::new(EchoModel(reply)))
    }

    // ── ModelResolveRequest invariants ───────────────────────────────────────

    #[test]
    fn new_request_has_no_role_and_is_not_lead() {
        let req = ModelResolveRequest::new("planner");
        assert_eq!(req.agent_id, "planner");
        assert_eq!(req.role(), None);
        assert!(!req.is_team_lead);
        assert_eq!(req.model_pin(), None);
    }

    #[test]
    fn builders_set_role_and_lead() {
        let req = ModelResolveRequest::new("planner")
            .with_role("researcher")
            .as_team_lead();
        assert_eq!(req.role(), Some("researcher"));
        assert!(req.is_team_lead);
    }

    #[test]
    fn model_pin_is_a_separate_channel_from_role() {
        // The bug #89 exists to prevent: with no pin field, a wiring author
        // puts the model id in `role`, the host reads it as an unknown role and
        // silently falls back to a default. The two must never collapse into
        // one string.
        let req = ModelResolveRequest::new("planner")
            .with_role("researcher")
            .with_model_pin("claude-3-5-sonnet");
        assert_eq!(req.role(), Some("researcher"));
        assert_eq!(req.model_pin(), Some("claude-3-5-sonnet"));
    }

    #[test]
    fn a_pin_can_be_set_without_a_role() {
        // The common case for a user-authored agent: it pins a model and
        // declares no role at all.
        let req = ModelResolveRequest::new("planner").with_model_pin("local-llama");
        assert_eq!(req.role(), None);
        assert_eq!(req.model_pin(), Some("local-llama"));
    }

    #[test]
    fn blank_model_pin_reads_as_absent() {
        // `model = ""` in a definition means "no pin", not a model named "".
        // Reaching a resolver as Some("") would fail the lookup and turn a
        // cosmetic config blank into an unroutable turn.
        for blank in ["", "   ", "\t\n"] {
            let req = ModelResolveRequest::new("planner").with_model_pin(blank);
            assert_eq!(
                req.model_pin(),
                None,
                "blank pin {blank:?} must read as absent"
            );
        }
    }

    #[test]
    fn model_pin_accessor_trims_surrounding_whitespace() {
        let req = ModelResolveRequest::new("planner").with_model_pin("  gpt-4o  ");
        assert_eq!(req.model_pin(), Some("gpt-4o"));
    }

    #[test]
    fn blank_role_reads_as_absent() {
        // A host mapping a missing field to `Some("")` must not select a
        // role-specific routing branch keyed on the empty string.
        assert_eq!(ModelResolveRequest::new("a").with_role("").role(), None);
        assert_eq!(
            ModelResolveRequest::new("a").with_role("  \t ").role(),
            None
        );
    }

    #[test]
    fn role_accessor_trims_surrounding_whitespace() {
        let req = ModelResolveRequest::new("a").with_role("  lead  ");
        assert_eq!(req.role(), Some("lead"));
        // The stored value is untouched — trimming is a read-side courtesy, not
        // normalization the host's own round-trip has to anticipate.
        assert_eq!(req.role.as_deref(), Some("  lead  "));
    }

    #[test]
    fn default_request_is_empty_and_not_lead() {
        let req = ModelResolveRequest::default();
        assert!(req.agent_id.is_empty());
        assert_eq!(req.role(), None);
        assert!(!req.is_team_lead);
        assert_eq!(req.model_pin(), None);
    }

    #[test]
    fn request_round_trips_through_serde() {
        let req = ModelResolveRequest::new("planner")
            .with_role("researcher")
            .with_model_pin("claude-3-5-sonnet")
            .as_team_lead();
        let json = serde_json::to_string(&req).expect("serialize");
        let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, req);
    }

    #[test]
    fn absent_role_is_omitted_and_restored() {
        let req = ModelResolveRequest::new("planner");
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(
            !json.contains("role"),
            "unset role must not serialize: {json}"
        );
        let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, req);
    }

    #[test]
    fn absent_model_pin_is_omitted_and_restored() {
        let req = ModelResolveRequest::new("planner");
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(
            !json.contains("model_pin"),
            "unset model pin must not serialize: {json}"
        );
        let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, req);
        assert_eq!(back.model_pin(), None);
    }

    #[test]
    fn request_deserializes_from_agent_id_alone() {
        // Hosts hand-writing this in config should not have to spell out
        // defaults for the two optional facts.
        let back: ModelResolveRequest =
            serde_json::from_str(r#"{"agent_id":"planner"}"#).expect("deserialize");
        assert_eq!(back, ModelResolveRequest::new("planner"));
    }

    // ── FixedModelResolver behaviour ─────────────────────────────────────────

    #[tokio::test]
    async fn fixed_resolver_returns_the_wrapped_model() {
        let resolver = fixed("hello");
        let model = resolver
            .resolve(&ModelResolveRequest::new("planner"))
            .await
            .expect("resolve");
        let response = model
            .invoke(&(), ModelRequest::default())
            .await
            .expect("invoke");
        assert_eq!(response.text(), "hello");
    }

    #[tokio::test]
    async fn fixed_resolver_ignores_every_routing_field() {
        let resolver = fixed("hello");
        let lead = resolver
            .resolve(
                &ModelResolveRequest::new("lead")
                    .with_role("planner")
                    .as_team_lead(),
            )
            .await
            .expect("resolve lead");
        let worker = resolver
            .resolve(&ModelResolveRequest::new("worker").with_role("scribe"))
            .await
            .expect("resolve worker");
        // A pin is advisory: a host with exactly one model is entitled to
        // ignore it, and must not be expected to look the id up.
        let pinned = resolver
            .resolve(&ModelResolveRequest::new("pinned").with_model_pin("gpt-4o"))
            .await
            .expect("resolve pinned");
        assert!(
            Arc::ptr_eq(&lead, &worker) && Arc::ptr_eq(&lead, &pinned),
            "the fixed resolver must not route on agent id, role, lead status, or model pin"
        );
    }

    #[tokio::test]
    async fn resolved_model_shares_the_wrapped_allocation() {
        // Resolution must hand back a handle to the same long-lived client, not
        // a clone — a per-turn copy would discard its connection pool.
        let resolver = fixed("hello");
        let resolved = resolver
            .resolve(&ModelResolveRequest::new("planner"))
            .await
            .expect("resolve");
        assert!(Arc::ptr_eq(resolver.model(), &resolved));
    }

    #[tokio::test]
    async fn clone_resolves_to_the_same_model() {
        let resolver = fixed("hello");
        let clone = resolver.clone();
        let a = resolver
            .resolve(&ModelResolveRequest::new("a"))
            .await
            .expect("resolve a");
        let b = clone
            .resolve(&ModelResolveRequest::new("b"))
            .await
            .expect("resolve b");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn usable_behind_a_trait_object() {
        // The runtime holds `Arc<dyn ModelResolver<State>>`, so object safety is
        // part of the contract, not an implementation detail.
        let resolver: Arc<dyn ModelResolver<()>> = Arc::new(fixed("dyn"));
        let model = resolver
            .resolve(&ModelResolveRequest::new("planner"))
            .await
            .expect("resolve");
        let response = model
            .invoke(&(), ModelRequest::default())
            .await
            .expect("invoke");
        assert_eq!(response.text(), "dyn");
    }
}
