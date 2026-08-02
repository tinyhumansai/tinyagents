use super::*;
use std::path::PathBuf;

fn sample() -> SessionConfig {
    SessionConfig::new("/ws", "/act", "sonnet")
}

#[test]
fn new_sets_the_three_required_values_and_defaults_the_rest() {
    let c = sample();
    assert_eq!(c.workspace_dir, PathBuf::from("/ws"));
    assert_eq!(c.action_dir, PathBuf::from("/act"));
    assert_eq!(c.model, "sonnet");
    assert_eq!(c.turn, TurnConfig::default());
    assert_eq!(c.tools, ToolConfig::default());
    assert_eq!(c.memory, MemoryLimits::default());
}

#[test]
fn defaults_match_the_documented_values() {
    // These are a compatibility surface for hosts using `..Default::default()`.
    // A change here silently changes their behaviour, so pin them.
    let t = TurnConfig::default();
    assert_eq!(t.max_tool_iterations, 10);
    assert_eq!(t.max_history_messages, 50);
    assert_eq!(t.max_parallel_tools, 4);
    assert_eq!(t.tool_result_budget_bytes, 16 * 1024);
    assert_eq!(t.timeout_secs, 120);
    assert!(!t.compact_context);
    assert!(!t.parallel_tools);
    assert!(t.required_output.is_none());

    assert_eq!(MemoryLimits::default().max_memory_context_chars, 2000);
    assert_eq!(ToolConfig::default().dispatcher, ToolDispatcher::Auto);
    assert!(sample().agents_md_enabled);
}

#[test]
fn lead_model_falls_back_to_model_then_overrides_it() {
    let mut c = sample();
    assert_eq!(c.effective_lead_model(), "sonnet");
    c.lead_model = Some("opus".into());
    assert_eq!(c.effective_lead_model(), "opus");
}

#[test]
fn subagent_model_does_not_inherit_the_lead_override() {
    // A host overriding the lead model usually wants subagents left on the
    // cheaper default; inheriting would silently multiply cost.
    let mut c = sample();
    c.lead_model = Some("opus".into());
    assert_eq!(c.effective_subagent_model(), "sonnet");

    c.subagent_model = Some("haiku".into());
    assert_eq!(c.effective_subagent_model(), "haiku");
    assert_eq!(c.effective_lead_model(), "opus");
}

#[test]
fn max_depth_zero_disables_delegation() {
    let mut c = sample();
    assert!(!c.may_delegate_at(0));

    c.max_depth = 2;
    assert!(c.may_delegate_at(0));
    assert!(c.may_delegate_at(1));
    assert!(!c.may_delegate_at(2), "depth 2 is the ceiling, not allowed");
}

#[test]
fn a_blank_required_output_block_key_is_inert() {
    assert!(!RequiredOutput::default().is_active());
    assert!(!RequiredOutput::new("   ").is_active());
    assert!(RequiredOutput::new("thoughts").is_active());
}

#[test]
fn all_keys_leads_with_the_block_key_and_dedupes_siblings() {
    let r = RequiredOutput {
        block_key: "  thoughts ".into(),
        required_keys: vec![
            "next_action".into(),
            "  ".into(),        // blank entries are dropped
            "next_action".into(), // duplicates are dropped
            "thoughts".into(),  // repeating the block key is a no-op
            " reason ".into(),  // trimmed
        ],
    };
    assert_eq!(r.all_keys(), vec!["thoughts", "next_action", "reason"]);
}

#[test]
fn a_blank_block_key_makes_the_contract_inert_even_with_siblings() {
    // The block key is the contract's defining key. Siblings alone must not
    // resurrect it, or enforcement would demand a block it can never name.
    let r = RequiredOutput {
        block_key: "  ".into(),
        required_keys: vec!["next_action".into()],
    };
    assert!(r.all_keys().is_empty());
    assert!(!r.is_active());
}

#[test]
fn required_output_new_leaves_sibling_keys_empty() {
    let r = RequiredOutput::new("thoughts");
    assert_eq!(r.block_key, "thoughts");
    assert!(r.required_keys.is_empty());
}

#[test]
fn tool_dispatcher_round_trips_as_snake_case() {
    for (variant, wire) in [
        (ToolDispatcher::Auto, "\"auto\""),
        (ToolDispatcher::Native, "\"native\""),
        (ToolDispatcher::Xml, "\"xml\""),
        (ToolDispatcher::Pformat, "\"pformat\""),
    ] {
        let json = serde_json::to_string(&variant).expect("serializes");
        assert_eq!(json, wire);
        let back: ToolDispatcher = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, variant);
    }
}

#[test]
fn an_unknown_dispatcher_is_a_deserialization_error_not_a_silent_default() {
    // The enum exists precisely so a typo'd host value fails at the boundary
    // rather than falling through to `auto` deep in the turn loop.
    assert!(serde_json::from_str::<ToolDispatcher>("\"nativ\"").is_err());
}

#[test]
fn session_config_round_trips_through_json() {
    let mut c = sample();
    c.temperature = Some(0.3);
    c.max_depth = 3;
    c.turn.required_output = Some(RequiredOutput::new("thoughts"));
    c.tools
        .channel_permissions
        .insert("telegram".into(), "read".into());

    let json = serde_json::to_string(&c).expect("serializes");
    let back: SessionConfig = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(back, c);
}

#[test]
fn omitted_sections_deserialize_to_their_defaults() {
    // A host that writes only the required fields still gets a usable config —
    // this is what keeps the mapper from having to name every knob.
    let json = r#"{
        "workspace_dir": "/ws",
        "action_dir": "/act",
        "model": "sonnet"
    }"#;
    let c: SessionConfig = serde_json::from_str(json).expect("deserializes");
    assert_eq!(c, sample());
}
