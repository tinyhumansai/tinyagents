//! Unit tests for [`ModelRouter`](super::ModelRouter) / [`WorkloadRoute`].

use super::{ModelRouter, WorkloadRoute};
use crate::harness::model::CapabilitySet;

/// Mirrors OpenHuman's workload tiers so the tests exercise the exact projection
/// the host router needs: fast/chat siblings, heavy reasoning siblings, and a
/// capability-gated vision primary-only tier.
fn oh_like_router() -> ModelRouter {
    ModelRouter::new()
        .with_route(WorkloadRoute::new("chat-v1", "chat-v1").with_fallbacks(["burst-v1"]))
        .with_route(WorkloadRoute::new("burst-v1", "burst-v1").with_fallbacks(["chat-v1"]))
        .with_route(
            WorkloadRoute::new("reasoning-v1", "reasoning-v1").with_fallbacks(["agentic-v1"]),
        )
        .with_route(WorkloadRoute::new("agentic-v1", "agentic-v1").with_fallbacks(["reasoning-v1"]))
        .with_route(
            WorkloadRoute::new("vision-v1", "vision-v1").requiring(CapabilitySet {
                image_in: true,
                ..CapabilitySet::default()
            }),
        )
        .with_default("chat-v1")
}

#[test]
fn resolves_alias_to_target_model() {
    let r = oh_like_router();
    assert_eq!(r.target_model("reasoning-v1"), Some("reasoning-v1"));
    assert_eq!(r.target_model("nope"), None);
    assert_eq!(r.default_alias(), Some("chat-v1"));
}

#[test]
fn fallback_policy_heads_with_primary_then_alternates() {
    let r = oh_like_router();
    let policy = r.fallback_policy("chat-v1").expect("chat has a sibling");
    // The chain leads with the primary so `next_after(primary)` yields the alternate.
    assert_eq!(
        policy.models,
        vec!["chat-v1".to_string(), "burst-v1".to_string()]
    );
    assert_eq!(policy.next_after("chat-v1"), Some("burst-v1"));
    assert_eq!(policy.next_after("burst-v1"), None);
}

#[test]
fn fallback_policy_is_none_without_alternates() {
    let r = oh_like_router();
    // vision is primary-only (a text fallback can't satisfy image_in).
    assert!(r.fallback_policy("vision-v1").is_none());
    // Unknown alias installs no policy.
    assert!(r.fallback_policy("ghost").is_none());
}

#[test]
fn capability_gate_only_for_gated_routes() {
    let r = oh_like_router();
    let gate = r
        .required_capabilities("vision-v1")
        .expect("vision requires image_in");
    assert!(gate.image_in);
    // A plain text tier imposes no gate, so the common turn is unaffected.
    assert!(r.required_capabilities("chat-v1").is_none());
    assert!(r.required_capabilities("unknown").is_none());
}

#[test]
fn aliases_preserve_registration_order() {
    let r = oh_like_router();
    let aliases: Vec<&str> = r.aliases().collect();
    assert_eq!(
        aliases,
        vec![
            "chat-v1",
            "burst-v1",
            "reasoning-v1",
            "agentic-v1",
            "vision-v1"
        ]
    );
}

#[test]
fn register_rejects_duplicate_alias() {
    let mut r = ModelRouter::new();
    r.register(WorkloadRoute::new("chat-v1", "chat-v1"))
        .unwrap();
    let err = r
        .register(WorkloadRoute::new("chat-v1", "other"))
        .unwrap_err();
    assert!(err.to_string().contains("chat-v1"), "err: {err}");
    // The original mapping is intact.
    assert_eq!(r.target_model("chat-v1"), Some("chat-v1"));
}

#[test]
fn with_route_last_write_wins_and_keeps_position() {
    let r = ModelRouter::new()
        .with_route(WorkloadRoute::new("a", "a-model"))
        .with_route(WorkloadRoute::new("b", "b-model"))
        // Overwrite `a` in place — position preserved, target updated.
        .with_route(WorkloadRoute::new("a", "a-model-v2"));
    assert_eq!(r.target_model("a"), Some("a-model-v2"));
    assert_eq!(r.aliases().collect::<Vec<_>>(), vec!["a", "b"]);
    assert_eq!(r.routes().len(), 2);
}

#[test]
fn empty_router_answers_nothing() {
    let r = ModelRouter::new();
    assert!(r.is_empty());
    assert!(r.default_alias().is_none());
    assert!(r.target_model("x").is_none());
    assert!(r.fallback_policy("x").is_none());
    assert!(r.required_capabilities("x").is_none());
}

#[test]
fn workload_route_serde_skips_defaults() {
    // A bare route round-trips compactly (no `requires`/`fallbacks` noise).
    let route = WorkloadRoute::new("chat-v1", "chat-v1");
    let json = serde_json::to_string(&route).unwrap();
    assert_eq!(json, r#"{"alias":"chat-v1","model":"chat-v1"}"#);
    let back: WorkloadRoute = serde_json::from_str(&json).unwrap();
    assert_eq!(back, route);
}
