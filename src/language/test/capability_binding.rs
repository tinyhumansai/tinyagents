//! Capability-binding tests (the legacy `bind_capabilities` allowlist
//! gate).
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Capability binding
// ---------------------------------------------------------------------------

#[test]
fn bind_capabilities_accepts_allowed_references() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::from_lists(
        ["default".to_string()],
        ["lookup_user".to_string(), "create_ticket".to_string()],
    );
    bind_capabilities(&bp, &resolver).unwrap();
}

#[test]
fn bind_capabilities_rejects_unknown_model() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new()
        .allow_tool("lookup_user")
        .allow_tool("create_ticket");
    let err = bind_capabilities(&bp, &resolver).unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Capability(_)));
    assert!(err.to_string().contains("unknown model"), "{err}");
}

#[test]
fn bind_capabilities_rejects_unknown_tool() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let resolver = CapabilityResolver::new().allow_model("default");
    let err = bind_capabilities(&bp, &resolver).unwrap_err();
    assert!(err.to_string().contains("unknown tool"), "{err}");
}
