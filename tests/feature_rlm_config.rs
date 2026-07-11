//! Feature tests for the RLM **config surface**: the serde documents an
//! external harness hands to the runtime (`RlmConfig`, `InterpreterSpec`,
//! `RlmPolicy`, `TemplateSpec`) and the host-call wire enum (`HostCall`).
//!
//! These focus on the config-as-data contract — round trips, documented
//! defaults, millisecond timeout encoding, untagged template selection, and
//! the `HostCall` name/kind accessors — which the module's own unit tests only
//! touch in part. Everything here is offline and deterministic.

#![cfg(feature = "rlm")]

use std::time::Duration;

use serde_json::json;
use tinyagents::rlm::{
    HostCall, InterpreterSpec, RlmCallKind, RlmConfig, RlmPolicy, RlmTemplate, TemplateSpec,
};

#[test]
fn interpreter_spec_language_name_per_variant() {
    assert_eq!(InterpreterSpec::Rhai.language(), "rhai");
    assert_eq!(
        InterpreterSpec::Python {
            binary: None,
            args: vec![]
        }
        .language(),
        "python"
    );
    assert_eq!(
        InterpreterSpec::Javascript {
            binary: None,
            args: vec![]
        }
        .language(),
        "javascript"
    );
    // An embedder-provided command speaks the Python-flavoured wire protocol.
    assert_eq!(
        InterpreterSpec::Command {
            binary: "runner".to_string(),
            args: vec![]
        }
        .language(),
        "python"
    );
}

#[test]
fn interpreter_spec_default_is_the_hermetic_rhai_engine() {
    assert_eq!(InterpreterSpec::default(), InterpreterSpec::Rhai);
}

#[test]
fn config_defaults_are_the_documented_shape() {
    let config = RlmConfig::default();
    assert_eq!(config.interpreter, InterpreterSpec::Rhai);
    assert_eq!(config.driver_model, None);
    assert_eq!(config.sub_model, None);
    assert_eq!(config.template, TemplateSpec::Named("general".to_string()));
    assert_eq!(config.policy, RlmPolicy::default());
}

#[test]
fn policy_defaults_match_the_documented_bounds() {
    let policy = RlmPolicy::default();
    assert_eq!(policy.max_cells, 16);
    assert_eq!(policy.max_script_bytes, 64 * 1024);
    assert_eq!(policy.max_output_bytes, 256 * 1024);
    assert_eq!(policy.max_llm_calls, 64);
    assert_eq!(policy.max_tool_calls, 128);
    assert_eq!(policy.max_agent_calls, 32);
    assert_eq!(policy.max_depth, 8);
    assert_eq!(policy.cell_timeout, Some(Duration::from_secs(120)));
    assert_eq!(policy.max_operations, 5_000_000);
}

#[test]
fn cell_timeout_is_encoded_as_integer_milliseconds() {
    let config = RlmConfig::from_json(
        r#"{ "interpreter": {"kind": "rhai"}, "policy": { "cell_timeout": 4500 } }"#,
    )
    .expect("parse");
    assert_eq!(
        config.policy.cell_timeout,
        Some(Duration::from_millis(4500))
    );

    let json = config.to_json().expect("serialize");
    assert!(
        json.contains("\"cell_timeout\": 4500"),
        "timeout should serialize as plain millis, got: {json}"
    );
}

#[test]
fn a_null_cell_timeout_round_trips_as_no_deadline() {
    let policy = RlmPolicy {
        cell_timeout: None,
        ..RlmPolicy::default()
    };
    let config = RlmConfig {
        policy,
        ..RlmConfig::default()
    };
    let back = RlmConfig::from_json(&config.to_json().expect("serialize")).expect("parse");
    assert_eq!(back.policy.cell_timeout, None);
}

#[test]
fn absent_optional_model_names_are_omitted_from_json() {
    let json = RlmConfig::default().to_json().expect("serialize");
    assert!(
        !json.contains("driver_model"),
        "None driver_model must be skipped, got: {json}"
    );
    assert!(
        !json.contains("sub_model"),
        "None sub_model must be skipped, got: {json}"
    );
}

#[test]
fn interpreter_spec_python_round_trips_binary_and_args() {
    let config = RlmConfig {
        interpreter: InterpreterSpec::Python {
            binary: Some("/opt/venv/bin/python".to_string()),
            args: vec!["-B".to_string()],
        },
        ..RlmConfig::default()
    };
    let back = RlmConfig::from_json(&config.to_json().expect("serialize")).expect("parse");
    assert_eq!(config, back);
}

#[test]
fn template_spec_is_untagged_named_or_inline() {
    let named: TemplateSpec =
        serde_json::from_value(json!("orchestrator")).expect("parse named template");
    assert_eq!(named, TemplateSpec::Named("orchestrator".to_string()));

    let inline: TemplateSpec = serde_json::from_value(json!({
        "name": "custom",
        "system_prompt": "do the thing"
    }))
    .expect("parse inline template");
    assert_eq!(
        inline,
        TemplateSpec::Inline(RlmTemplate {
            name: "custom".to_string(),
            system_prompt: "do the thing".to_string(),
        })
    );
}

#[test]
fn host_call_reports_a_name_and_kind_per_variant() {
    let llm_default = HostCall::Llm {
        model: None,
        prompt: "hi".to_string(),
        system: None,
    };
    assert_eq!(llm_default.name(), "default");
    assert_eq!(llm_default.kind(), RlmCallKind::Llm);

    let llm_named = HostCall::Llm {
        model: Some("gpt".to_string()),
        prompt: "hi".to_string(),
        system: None,
    };
    assert_eq!(llm_named.name(), "gpt");

    let tool = HostCall::Tool {
        tool: "search".to_string(),
        arguments: json!({}),
    };
    assert_eq!(tool.name(), "search");
    assert_eq!(tool.kind(), RlmCallKind::Tool);

    let agent = HostCall::Agent {
        agent: "researcher".to_string(),
        input: "go".to_string(),
        data: None,
    };
    assert_eq!(agent.name(), "researcher");
    assert_eq!(agent.kind(), RlmCallKind::Agent);

    let answer = HostCall::FinalAnswer {
        answer: "done".to_string(),
    };
    assert_eq!(answer.name(), "final_answer");
    assert_eq!(answer.kind(), RlmCallKind::FinalAnswer);
}

#[test]
fn host_call_variants_round_trip_through_their_tagged_wire_shape() {
    for call in [
        HostCall::Llm {
            model: Some("m".to_string()),
            prompt: "p".to_string(),
            system: Some("s".to_string()),
        },
        HostCall::Tool {
            tool: "t".to_string(),
            arguments: json!({ "q": 1 }),
        },
        HostCall::Agent {
            agent: "a".to_string(),
            input: "i".to_string(),
            data: Some(json!({ "k": "v" })),
        },
        HostCall::FinalAnswer {
            answer: "the end".to_string(),
        },
    ] {
        let json = serde_json::to_string(&call).expect("serialize call");
        let back: HostCall = serde_json::from_str(&json).expect("parse call");
        assert_eq!(call, back);
    }
}
