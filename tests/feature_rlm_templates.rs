//! Feature tests for the **RLM prompt templates**: the built-in template
//! catalogue, fail-closed resolution of unknown names, inline template
//! pass-through, and the placeholder renderer's substitution of language,
//! usage guide, live capabilities, and policy limits.
//!
//! The renderer is pure and offline, so these build a `CapabilityListing`
//! directly and assert on the rendered prompt text.

#![cfg(feature = "rlm")]

use tinyagents::rlm::templates;
use tinyagents::rlm::{CapabilityListing, RlmPolicy, RlmTemplate, TemplateSpec};

#[test]
fn resolves_each_built_in_template_by_name() {
    for name in ["general", "context-explorer", "orchestrator"] {
        let template =
            templates::resolve(&TemplateSpec::Named(name.to_string())).expect("resolve built-in");
        assert_eq!(template.name, name);
        assert!(template.system_prompt.contains("{{language}}"));
        assert!(template.system_prompt.contains("final_answer("));
    }
}

#[test]
fn the_orchestrator_template_describes_delegation() {
    let template = templates::orchestrator();
    assert!(template.system_prompt.contains("agent("));
    assert!(template.system_prompt.to_lowercase().contains("delegate"));
}

#[test]
fn the_context_explorer_template_describes_the_context_variable() {
    let template = templates::context_explorer();
    assert!(template.system_prompt.contains("`context`"));
}

#[test]
fn an_unknown_named_template_fails_closed_with_the_catalogue() {
    let err = templates::resolve(&TemplateSpec::Named("does-not-exist".to_string()))
        .expect_err("unknown template must fail");
    match err {
        tinyagents::TinyAgentsError::Validation(message) => {
            assert!(message.contains("does-not-exist"));
            assert!(message.contains("general"));
            assert!(message.contains("orchestrator"));
        }
        other => panic!("expected Validation, got {other:?}"),
    }
}

#[test]
fn an_inline_template_is_returned_verbatim() {
    let inline = RlmTemplate {
        name: "bespoke".to_string(),
        system_prompt: "custom {{language}} scaffold".to_string(),
    };
    let resolved =
        templates::resolve(&TemplateSpec::Inline(inline.clone())).expect("resolve inline");
    assert_eq!(resolved, inline);
}

#[test]
fn rendering_substitutes_every_placeholder() {
    let listing = CapabilityListing {
        models: vec!["gpt".to_string()],
        tools: vec![("search".to_string(), "Searches the web.".to_string())],
        agents: vec!["planner".to_string()],
    };
    let prompt = templates::render_system_prompt(
        &templates::general(),
        "python",
        "USAGE-GUIDE-MARKER",
        &listing,
        &RlmPolicy::default(),
    );
    assert!(
        !prompt.contains("{{"),
        "no placeholder must remain: {prompt}"
    );
    assert!(prompt.contains("```python"));
    assert!(prompt.contains("USAGE-GUIDE-MARKER"));
    assert!(prompt.contains("search: Searches the web."));
    assert!(prompt.contains("planner"));
    assert!(prompt.contains("max cells: 16"));
}

#[test]
fn rendering_marks_an_empty_registry_as_none_registered() {
    let prompt = templates::render_system_prompt(
        &templates::general(),
        "rhai",
        "guide",
        &CapabilityListing::default(),
        &RlmPolicy::default(),
    );
    assert!(prompt.contains("models: (none registered)"));
    assert!(prompt.contains("tools: (none registered)"));
    assert!(prompt.contains("agents: (none registered)"));
}

#[test]
fn rendering_reflects_a_customised_policy_and_absent_timeout() {
    let policy = RlmPolicy {
        max_cells: 3,
        max_llm_calls: 5,
        max_tool_calls: 7,
        max_agent_calls: 9,
        cell_timeout: None,
        ..RlmPolicy::default()
    };
    let prompt = templates::render_system_prompt(
        &templates::general(),
        "rhai",
        "guide",
        &CapabilityListing::default(),
        &policy,
    );
    assert!(prompt.contains("max cells: 3"));
    assert!(prompt.contains("max sub-LLM calls: 5"));
    assert!(prompt.contains("max tool calls: 7"));
    assert!(prompt.contains("max agent calls: 9"));
    assert!(prompt.contains("per-cell timeout: none"));
}
