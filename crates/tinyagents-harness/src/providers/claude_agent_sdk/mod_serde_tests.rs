//! Serde and JSON-schema parity tests for [`ClaudeAgentSdkConfig`].
//!
//! OpenHuman reads this config from the `[claude_agent_sdk]` section of its
//! TOML config through its own mirror type. The mirror is reproduced verbatim
//! below so these tests pin that the upstream derives read the same files,
//! write the same bytes and describe the same JSON schema — the guarantee a
//! host needs to swap its mirror for this type.

use super::*;
use serde_json::json;

/// OpenHuman's `config::schema::claude_agent_sdk::ClaudeAgentSdkConfig`,
/// verbatim (derives, attributes, field docs and default functions), as the
/// parity oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct OpenHumanClaudeAgentSdkConfig {
    /// Whether the Claude Agent SDK provider is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// Path to the `claude` CLI binary. Defaults to `"claude"` (PATH lookup).
    #[serde(default = "default_claude_binary")]
    pub binary: String,

    /// Default model passed via `--model` when the provider string is just
    /// `"claude_agent_sdk"` with no model suffix.
    #[serde(default = "default_claude_model")]
    pub default_model: String,

    /// Maximum spend in USD before aborting a request (`--max-turns` controls
    /// loop depth; `--budget` controls cost). `None` means no cap.
    #[serde(default)]
    pub max_budget_usd: Option<f64>,
}

impl Default for OpenHumanClaudeAgentSdkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: default_claude_binary(),
            default_model: default_claude_model(),
            max_budget_usd: None,
        }
    }
}

/// How OpenHuman embeds the section: `#[serde(default)] claude_agent_sdk`.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct HostConfig<T: Default> {
    #[serde(default)]
    claude_agent_sdk: T,
}

/// TOML inputs a host config may carry for the section.
const TOML_CASES: &[&str] = &[
    "[claude_agent_sdk]\nenabled = true\nbinary = \"/opt/bin/claude\"\n\
     default_model = \"claude-opus-4-1\"\nmax_budget_usd = 2.5\n",
    "[claude_agent_sdk]\nenabled = true\n",
    "[claude_agent_sdk]\nmax_budget_usd = 0.25\n",
    "[claude_agent_sdk]\n",
    "",
];

fn fields(c: &ClaudeAgentSdkConfig) -> (bool, &str, &str, Option<f64>) {
    (c.enabled, &c.binary, &c.default_model, c.max_budget_usd)
}

fn mirror_fields(c: &OpenHumanClaudeAgentSdkConfig) -> (bool, &str, &str, Option<f64>) {
    (c.enabled, &c.binary, &c.default_model, c.max_budget_usd)
}

#[test]
fn reads_a_full_toml_section() {
    let host: HostConfig<ClaudeAgentSdkConfig> = toml::from_str(TOML_CASES[0]).unwrap();
    assert_eq!(
        fields(&host.claude_agent_sdk),
        (true, "/opt/bin/claude", "claude-opus-4-1", Some(2.5))
    );
}

#[test]
fn omitted_fields_take_the_default_values() {
    let only_enabled: ClaudeAgentSdkConfig =
        serde_json::from_value(json!({"enabled": true})).unwrap();
    assert_eq!(
        fields(&only_enabled),
        (true, "claude", "claude-sonnet-4-6", None)
    );

    let empty: ClaudeAgentSdkConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(fields(&empty), fields(&ClaudeAgentSdkConfig::default()));

    let absent: HostConfig<ClaudeAgentSdkConfig> = toml::from_str("").unwrap();
    assert_eq!(
        fields(&absent.claude_agent_sdk),
        fields(&ClaudeAgentSdkConfig::default())
    );
}

#[test]
fn serializes_every_field_and_none_as_null_or_omitted() {
    assert_eq!(
        serde_json::to_value(ClaudeAgentSdkConfig::default()).unwrap(),
        json!({
            "enabled": false,
            "binary": "claude",
            "default_model": "claude-sonnet-4-6",
            "max_budget_usd": null
        })
    );
    // TOML has no null: a `None` budget is left out, and reads back as `None`.
    assert_eq!(
        toml::to_string(&ClaudeAgentSdkConfig::default()).unwrap(),
        "enabled = false\nbinary = \"claude\"\ndefault_model = \"claude-sonnet-4-6\"\n"
    );
}

#[test]
fn toml_and_json_match_openhumans_mirror_byte_for_byte() {
    for input in TOML_CASES {
        let ours: HostConfig<ClaudeAgentSdkConfig> = toml::from_str(input).unwrap();
        let theirs: HostConfig<OpenHumanClaudeAgentSdkConfig> = toml::from_str(input).unwrap();
        assert_eq!(
            fields(&ours.claude_agent_sdk),
            mirror_fields(&theirs.claude_agent_sdk),
            "parsed values differ for {input:?}"
        );
        let ours_toml = toml::to_string(&ours).unwrap();
        assert_eq!(
            ours_toml,
            toml::to_string(&theirs).unwrap(),
            "TOML output differs for {input:?}"
        );
        assert_eq!(
            serde_json::to_string(&ours).unwrap(),
            serde_json::to_string(&theirs).unwrap(),
            "JSON output differs for {input:?}"
        );
        // The re-serialized TOML reads back to the same values.
        let again: HostConfig<ClaudeAgentSdkConfig> = toml::from_str(&ours_toml).unwrap();
        assert_eq!(
            fields(&again.claude_agent_sdk),
            fields(&ours.claude_agent_sdk)
        );
    }
}

#[test]
fn mistyped_fields_are_rejected_like_the_mirror() {
    let bad = json!({"enabled": "yes"});
    assert!(serde_json::from_value::<ClaudeAgentSdkConfig>(bad.clone()).is_err());
    assert!(serde_json::from_value::<OpenHumanClaudeAgentSdkConfig>(bad).is_err());
}

/// The root schema of `T` without the type-level `title`/`description`, which
/// name the Rust type rather than describe the wire shape.
fn wire_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap();
    let object = schema.as_object_mut().unwrap();
    object.remove("title");
    object.remove("description");
    schema
}

#[test]
fn json_schema_matches_openhumans_mirror() {
    let ours = wire_schema::<ClaudeAgentSdkConfig>();
    assert_eq!(ours, wire_schema::<OpenHumanClaudeAgentSdkConfig>());
    assert!(ours.get("required").is_none(), "no field is required");
    let props = &ours["properties"];
    assert_eq!(props["enabled"]["default"], json!(false));
    assert_eq!(props["binary"]["default"], json!("claude"));
    assert_eq!(
        props["default_model"]["default"],
        json!("claude-sonnet-4-6")
    );
    assert_eq!(props["max_budget_usd"]["default"], json!(null));
    assert_eq!(
        schemars::schema_for!(ClaudeAgentSdkConfig).as_value()["title"],
        json!("ClaudeAgentSdkConfig")
    );
}
