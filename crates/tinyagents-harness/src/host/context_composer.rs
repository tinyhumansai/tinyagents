//! Host capability: turn-context composition (RFC §3.2, **required**).
//!
//! The runtime knows the *mechanical* parts of a turn — which agent is running,
//! which thread it belongs to, and what the user just said. It deliberately does
//! not know the parts that make an assistant feel like a particular product:
//! identity/soul text, descriptions of whichever integrations the user has
//! actually connected, and whatever learned context the host has accumulated
//! about this person. Those are host material, they change per deployment, and
//! several of them are business rules that must never ship inside a
//! redistributed crate.
//!
//! [`ContextComposer`] is the seam. Before each turn the runtime hands the host
//! a [`TurnContextRequest`] and receives back the system prompt (and optionally
//! a few messages to prepend). The runtime treats both as **opaque** — it does
//! not parse, re-order, truncate, or otherwise interpret what comes back.
//!
//! Unlike the optional capabilities in this module family, a host must always
//! supply one: a turn with no system prompt at all is not a meaningful default,
//! so "absent" is not a state the runtime can act on. Hosts that genuinely have
//! nothing to inject use [`StaticContextComposer`] with a fixed prompt rather
//! than passing `None`.
//!
//! # Empty is not failure
//!
//! [`ContextComposer::preamble`] returning an empty `Vec` is the *normal* case,
//! not an error and not a signal that something is missing. Only hosts with
//! pinned context or explicit goals to inject return anything, and the runtime
//! must proceed unchanged when the vector is empty.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::ThreadId;
use crate::message::Message;

// ── TurnContextRequest ────────────────────────────────────────────────────────

/// Everything the host needs in order to compose the context for one turn.
///
/// **Inert by construction:** `serde` + `std` only. A host must be able to build
/// a composer against this struct without linking the rest of the runtime, which
/// is the same carve-out that lets a gated domain keep its type module
/// always-compiled.
///
/// The agent is identified by a plain `String` rather than a newtype because
/// this crate has no `AgentId` — the RFC's other signatures name one, but
/// introducing it here would mint a crate-wide identifier from a single module
/// and force every host call site to convert. A bare id keeps the seam narrow;
/// if an `AgentId` newtype ever lands crate-wide, this field is the one place
/// that changes.
///
/// It carries no security context, no config, and no tool state on purpose:
/// anything the host must *enforce* belongs behind a gate the runtime calls, not
/// smuggled into a struct the runtime also treats as advisory.
// No `Default`: `ThreadId` has none, and a turn context with a blank agent id
// or thread would compose the wrong identity rather than fail — see `new`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnContextRequest {
    /// Host-defined id of the agent running this turn. Opaque to the runtime —
    /// it is passed through so the host can vary identity text per agent.
    pub agent_id: String,
    /// Thread this turn belongs to. The host uses it to scope learned context
    /// and any thread-level prior state it wants reflected in the prompt.
    pub thread_id: ThreadId,
    /// The user's turn text, verbatim and untransformed.
    ///
    /// Deliberately **not** pre-screened here: screening untrusted text is the
    /// security gate's job, and a composer that silently received sanitized text
    /// would make it impossible to tell whether screening ran at all.
    pub user_text: String,
}

impl TurnContextRequest {
    /// A request for `agent_id` on `thread_id` carrying `user_text`.
    ///
    /// All three values are required. There is no partial constructor because a
    /// composer keyed on a missing agent id would silently compose the wrong
    /// identity — a failure that produces plausible output and so is not caught
    /// by a smoke test.
    pub fn new(
        agent_id: impl Into<String>,
        thread_id: impl Into<ThreadId>,
        user_text: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            thread_id: thread_id.into(),
            user_text: user_text.into(),
        }
    }

    /// Whether the turn carries any user text at all once surrounding
    /// whitespace is ignored.
    ///
    /// Useful to a host that varies its prompt for a turn triggered by something
    /// other than typed input (a resumed run, a scheduled kick). It reports; it
    /// does not decide — an empty turn is still a valid turn and the runtime
    /// does not reject one on this basis.
    pub fn has_user_text(&self) -> bool {
        !self.user_text.trim().is_empty()
    }
}

// ── ContextComposer ───────────────────────────────────────────────────────────

/// Assembles the system prompt and any preamble messages for a turn.
///
/// Implementations run on the critical path of every turn, before the first
/// model call, so they should be cheap or internally cached. They are also
/// consulted for *each* turn rather than once per session: a host whose learned
/// context or connected-integration set changes mid-session needs the later
/// turns to see it, and caching that decision inside the runtime would freeze
/// stale identity text into a long conversation.
#[async_trait]
pub trait ContextComposer: Send + Sync {
    /// Builds the system prompt for a turn.
    ///
    /// This is where the host injects identity/soul text, descriptions of the
    /// integrations the user has connected, and learned-context blocks. The
    /// returned string is used **verbatim** — the runtime does not append its
    /// own instructions to it, so a host that wants the turn's mechanical
    /// framing present must include it here.
    ///
    /// Returning `Ok("")` is legal and means "no system prompt for this turn".
    /// An `Err` means the host could not compose one and the turn should fail —
    /// reserve it for genuine faults, never for "nothing to add", because a
    /// caller cannot tell the two apart from an error alone.
    async fn compose_system_prompt(&self, req: &TurnContextRequest) -> Result<String>;

    /// Extra messages to prepend after the system prompt — goals, pinned
    /// context, a few-shot exchange the host maintains.
    ///
    /// **Empty is normal.** Most turns return `Ok(vec![])`; the runtime must
    /// treat that as a complete, successful answer and not as a degraded one.
    ///
    /// Order is preserved and the messages land between the system prompt and
    /// the conversation history, so a host that returns roles out of order (a
    /// second `system` message, say) is making a provider-visible choice the
    /// runtime will not correct for it.
    async fn preamble(&self, req: &TurnContextRequest) -> Result<Vec<Message>>;
}

// ── StaticContextComposer ─────────────────────────────────────────────────────

/// The default [`ContextComposer`]: one fixed system prompt, no preamble.
///
/// This is the Phase-1 stand-in that lets the runtime land the seam before any
/// host implements it, and it stays useful afterwards for tests and for hosts
/// with genuinely static framing.
///
/// It ignores the [`TurnContextRequest`] entirely — by design, not by omission.
/// The point of a static composer is that its output is independent of the turn,
/// which makes it a reproducible baseline: a test that varies the request and
/// sees the prompt change knows a real composer is wired in.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StaticContextComposer {
    /// The prompt returned for every turn.
    system_prompt: String,
}

impl StaticContextComposer {
    /// A composer that returns `system_prompt` for every turn.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
        }
    }

    /// A composer that returns an empty system prompt and no preamble.
    ///
    /// The neutral choice when a host has not been wired up yet. It contributes
    /// nothing rather than guessing at framing — a placeholder prompt would be
    /// worse than none, because the model would follow it.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The fixed prompt this composer was constructed with.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}

#[async_trait]
impl ContextComposer for StaticContextComposer {
    async fn compose_system_prompt(&self, _req: &TurnContextRequest) -> Result<String> {
        Ok(self.system_prompt.clone())
    }

    async fn preamble(&self, _req: &TurnContextRequest) -> Result<Vec<Message>> {
        Ok(Vec::new())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TurnContextRequest {
        TurnContextRequest::new("planner", "thread-1", "what changed today?")
    }

    #[test]
    fn new_populates_every_field() {
        let req = request();
        assert_eq!(req.agent_id, "planner");
        assert_eq!(req.thread_id.as_str(), "thread-1");
        assert_eq!(req.user_text, "what changed today?");
    }

    #[test]
    fn has_user_text_ignores_surrounding_whitespace() {
        assert!(request().has_user_text());
        assert!(!TurnContextRequest::new("planner", "t", "   \n\t ").has_user_text());
        assert!(!TurnContextRequest::new("planner", "t", "").has_user_text());
        // A single visible character still counts — the check is "blank", not
        // "meaningful".
        assert!(TurnContextRequest::new("planner", "t", "?").has_user_text());
    }

    #[test]
    fn request_round_trips_through_serde() {
        let req = request();
        let json = serde_json::to_string(&req).expect("serialize");
        let back: TurnContextRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(req, back);
    }

    #[test]
    fn a_blank_user_text_reports_no_user_text() {
        // `TurnContextRequest` has no `Default` on purpose (`ThreadId` has
        // none, and a blank agent id would compose the wrong identity), so
        // construct the empty-text case explicitly.
        let req = TurnContextRequest::new("agent", ThreadId::from("thread-1"), "   ");
        assert!(!req.has_user_text());
    }

    #[tokio::test]
    async fn static_composer_returns_its_fixed_prompt() {
        let composer = StaticContextComposer::new("you are a careful assistant");
        assert_eq!(composer.system_prompt(), "you are a careful assistant");
        assert_eq!(
            composer
                .compose_system_prompt(&request())
                .await
                .expect("prompt"),
            "you are a careful assistant"
        );
    }

    #[tokio::test]
    async fn static_composer_prompt_is_independent_of_the_request() {
        let composer = StaticContextComposer::new("fixed");
        let a = TurnContextRequest::new("planner", "thread-1", "first");
        let b = TurnContextRequest::new("researcher", "thread-2", "second");
        assert_eq!(
            composer.compose_system_prompt(&a).await.expect("prompt a"),
            composer.compose_system_prompt(&b).await.expect("prompt b"),
        );
    }

    #[tokio::test]
    async fn static_composer_preamble_is_empty_and_not_an_error() {
        let composer = StaticContextComposer::new("fixed");
        let preamble = composer.preamble(&request()).await.expect("preamble");
        assert!(preamble.is_empty());
    }

    #[tokio::test]
    async fn empty_composer_contributes_nothing() {
        let composer = StaticContextComposer::empty();
        assert_eq!(
            composer
                .compose_system_prompt(&request())
                .await
                .expect("prompt"),
            ""
        );
        assert!(
            composer
                .preamble(&request())
                .await
                .expect("preamble")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn usable_as_a_trait_object() {
        // Pins object safety: the harness stores this capability as
        // `Arc<dyn ContextComposer>`, so a non-dyn-safe signature would only
        // fail at the wiring site, not here.
        let composer: std::sync::Arc<dyn ContextComposer> =
            std::sync::Arc::new(StaticContextComposer::new("dyn"));
        assert_eq!(
            composer
                .compose_system_prompt(&request())
                .await
                .expect("prompt"),
            "dyn"
        );
    }
}
