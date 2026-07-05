//! Tests for the harness tool layer.
//!
//! Cover [`ToolSchema`]/[`ToolCall`]/[`ToolResult`] construction (including the
//! error path that preserves the call id) and [`ToolRegistry`] registration,
//! lookup, name listing, and schema collection.

use super::*;
use async_trait::async_trait;
use serde_json::{Value, json};

struct EchoTool;

#[async_trait]
impl Tool<()> for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "echoes its input"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo",
            "echoes its input",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        )
    }

    async fn call(&self, _state: &(), call: ToolCall) -> crate::Result<ToolResult> {
        let text = call
            .arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Ok(ToolResult::text(call.id, "echo", text))
    }
}

struct PolicyTool {
    policy: ToolPolicy,
}

#[async_trait]
impl Tool<()> for PolicyTool {
    fn name(&self) -> &str {
        "mcp_file-read"
    }

    fn description(&self) -> &str {
        "reads files"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name(),
            self.description(),
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )
    }

    fn policy(&self) -> ToolPolicy {
        self.policy.clone()
    }

    async fn call(&self, _state: &(), call: ToolCall) -> crate::Result<ToolResult> {
        Ok(ToolResult::text(call.id, self.name(), "ok"))
    }
}

#[test]
fn registry_register_get_names_schemas() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    assert!(registry.get("echo").is_some());
    assert!(registry.get("missing").is_none());
    assert_eq!(registry.names(), vec!["echo".to_string()]);
    assert_eq!(registry.schemas().len(), 1);
    assert_eq!(registry.schemas()[0].name, "echo");
}

#[test]
fn default_policy_is_unclassified() {
    let policy = EchoTool.policy();
    assert!(!policy.classified);
    assert!(!policy.has_side_effects());
}

#[test]
fn read_only_policy_is_classified_and_pure() {
    let policy = ToolPolicy::read_only();
    assert!(policy.classified);
    assert!(policy.side_effects.read_only);
    assert!(!policy.has_side_effects());
    assert!(policy.access.background_safe);
    assert!(policy.runtime.idempotent);
}

#[test]
fn policy_builders_mark_classified_and_serialize() {
    let policy = ToolPolicy::classified()
        .with_side_effects(ToolSideEffects {
            network: true,
            payment: true,
            ..ToolSideEffects::default()
        })
        .requiring_approval();
    assert!(policy.classified);
    assert!(policy.has_side_effects());
    assert!(policy.access.approval_required);
    // Round-trips for audit/registry introspection.
    let json = serde_json::to_value(&policy).unwrap();
    let back: ToolPolicy = serde_json::from_value(json).unwrap();
    assert_eq!(policy, back);
}

#[test]
fn registry_exposes_policy_snapshot() {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool));
    let policies = registry.policies();
    assert!(policies.contains_key("echo"));
    assert!(!policies["echo"].classified);
}

#[test]
fn default_display_label_humanizes_tool_name() {
    let tool = PolicyTool {
        policy: ToolPolicy::read_only(),
    };
    let call = ToolCall::new("c-1", tool.name(), json!({}));

    assert_eq!(tool.display_label(&call).as_deref(), Some("File Read"));
    assert_eq!(
        humanize_tool_name("gmail_read_message"),
        "Gmail Read Message"
    );
    assert_eq!(
        humanize_tool_name("composio_gmail_send_email"),
        "Gmail Send Email"
    );
    assert_eq!(humanize_tool_name("read-diff"), "Read Diff");
    assert_eq!(humanize_tool_name("___"), "___");
}

#[test]
fn default_display_detail_extracts_common_context_args() {
    let tool = EchoTool;

    let missing = ToolCall::new("c-1", "echo", Value::Null);
    assert!(tool.display_detail(&missing).is_none());

    let call = ToolCall::new(
        "c-2",
        "echo",
        json!({"name": "ignored", "to": "steven@example.com"}),
    );
    assert_eq!(
        tool.display_detail(&call).as_deref(),
        Some("steven@example.com")
    );

    let spaced = ToolCall::new("c-3", "echo", json!({"command": "  ls   -la  "}));
    assert_eq!(tool.display_detail(&spaced).as_deref(), Some("ls -la"));

    let long = "x".repeat(200);
    let long_call = ToolCall::new("c-4", "echo", json!({"query": long}));
    let detail = tool.display_detail(&long_call).unwrap();
    assert!(detail.chars().count() <= 80);
    assert!(detail.ends_with("..."));
}

#[test]
fn policy_display_metadata_overrides_derived_defaults() {
    let tool = PolicyTool {
        policy: ToolPolicy::read_only()
            .with_display(ToolDisplay::label("Reading file").with_detail("README.md")),
    };
    let call = ToolCall::new("c-1", tool.name(), json!({"path": "src/lib.rs"}));

    assert_eq!(tool.display_label(&call).as_deref(), Some("Reading file"));
    assert_eq!(tool.display_detail(&call).as_deref(), Some("README.md"));

    let serialized = serde_json::to_value(tool.policy()).unwrap();
    assert_eq!(serialized["display"]["label"], "Reading file");
}

#[test]
fn tool_policy_deserializes_without_display_metadata() {
    let mut json = serde_json::to_value(ToolPolicy::read_only()).unwrap();
    json.as_object_mut().unwrap().remove("display");

    let policy: ToolPolicy = serde_json::from_value(json).unwrap();
    assert!(policy.display.is_empty());
}

#[test]
fn timeout_policy_uses_richer_timeout_semantics() {
    let call = ToolCall::new("c-1", "mcp_file-read", json!({}));

    let inherited = PolicyTool {
        policy: ToolPolicy::read_only(),
    };
    assert_eq!(inherited.timeout_policy(&call), ToolTimeout::Inherit);

    let legacy_ms = PolicyTool {
        policy: ToolPolicy::read_only().with_runtime(ToolRuntime {
            timeout_ms: Some(12_000),
            ..ToolRuntime::default()
        }),
    };
    assert_eq!(legacy_ms.timeout_policy(&call), ToolTimeout::Millis(12_000));

    let unbounded = PolicyTool {
        policy: ToolPolicy::read_only().with_runtime(ToolRuntime {
            timeout: ToolTimeout::Unbounded,
            timeout_ms: Some(12_000),
            ..ToolRuntime::default()
        }),
    };
    assert_eq!(unbounded.timeout_policy(&call), ToolTimeout::Unbounded);
}

#[tokio::test]
async fn tool_call_round_trips() {
    let tool = EchoTool;
    let call = ToolCall::new("c-1", "echo", json!({"text": "hi"}));
    let result = tool.call(&(), call).await.unwrap();
    assert_eq!(result.call_id, "c-1");
    assert_eq!(result.content, "hi");
    assert!(!result.is_error());
}

#[test]
fn error_result_preserves_call_id() {
    let result = ToolResult::error("c-9", "echo", "boom");
    assert!(result.is_error());
    assert_eq!(result.call_id, "c-9");
    assert_eq!(result.error.as_deref(), Some("boom"));
}

#[test]
fn tool_schema_defaults_to_json_format() {
    let schema = ToolSchema::new("lookup", "looks up a user", json!({"type": "object"}));

    assert_eq!(schema.format, ToolFormat::Json);
    let value = serde_json::to_value(&schema).unwrap();
    assert!(value.get("format").is_none());
}

#[test]
fn tool_schema_can_declare_xml_and_ptype_formats() {
    let xml = ToolSchema::new("lookup", "looks up a user", json!({"type": "object"}))
        .with_format(ToolFormat::Xml);
    let ptype = ToolSchema::new(
        "search",
        "searches docs",
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": { "type": "string" },
                "limit": { "type": "integer" }
            }
        }),
    )
    .with_format(ToolFormat::PType {
        parameters: vec!["query".to_string(), "limit".to_string()],
    });

    assert_eq!(xml.format, ToolFormat::Xml);
    assert_eq!(
        ptype.format,
        ToolFormat::PType {
            parameters: vec!["query".to_string(), "limit".to_string()],
        }
    );
    let round_tripped: ToolSchema =
        serde_json::from_value(serde_json::to_value(&ptype).unwrap()).unwrap();
    assert_eq!(round_tripped, ptype);
}

#[test]
fn schema_validation_accepts_matching_arguments() {
    let schema = ToolSchema::new(
        "lookup",
        "looks up a user",
        json!({
            "type": "object",
            "required": ["user"],
            "additionalProperties": false,
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["id"],
                    "additionalProperties": false,
                    "properties": {
                        "id": { "type": "string" },
                        "roles": { "type": "array", "items": { "type": "string" } }
                    }
                },
                "include_disabled": { "type": "boolean" }
            }
        }),
    );
    let call = ToolCall::new(
        "c-1",
        "lookup",
        json!({
            "user": { "id": "u-1", "roles": ["admin", "editor"] },
            "include_disabled": false
        }),
    );

    schema.validate_call(&call).expect("valid call");
}

#[test]
fn schema_validation_rejects_missing_required_fields() {
    let schema = ToolSchema::new(
        "lookup",
        "looks up a user",
        json!({
            "type": "object",
            "required": ["user_id"],
            "properties": { "user_id": { "type": "string" } }
        }),
    );
    let call = ToolCall::new("c-1", "lookup", json!({}));

    let err = schema.validate_call(&call).expect_err("missing field");
    assert!(err.to_string().contains("user_id"));
}

#[test]
fn schema_validation_rejects_missing_required_without_properties() {
    // A schema may declare `required` without listing `properties`. The
    // required check must still fail closed rather than silently accept a call
    // that omits the required field.
    let schema = ToolSchema::new(
        "lookup",
        "looks up a user",
        json!({
            "type": "object",
            "required": ["user_id"]
        }),
    );

    let missing = ToolCall::new("c-1", "lookup", json!({}));
    let err = schema.validate_call(&missing).expect_err("missing field");
    assert!(err.to_string().contains("user_id"));

    // A non-object argument for a required-bearing schema also fails closed.
    let not_object = ToolCall::new("c-2", "lookup", json!("nope"));
    schema
        .validate_call(&not_object)
        .expect_err("non-object arguments");

    let present = ToolCall::new("c-3", "lookup", json!({ "user_id": "u-1" }));
    schema.validate_call(&present).expect("valid call");
}

#[test]
fn schema_validation_rejects_wrong_types_and_extra_fields() {
    let schema = ToolSchema::new(
        "lookup",
        "looks up a user",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "user_id": { "type": "string" },
                "limit": { "type": "integer" }
            }
        }),
    );

    let wrong_type = ToolCall::new("c-1", "lookup", json!({ "user_id": 42 }));
    let err = schema.validate_call(&wrong_type).expect_err("wrong type");
    assert!(err.to_string().contains("user_id"));

    let extra = ToolCall::new("c-2", "lookup", json!({ "user_id": "u-1", "extra": true }));
    let err = schema.validate_call(&extra).expect_err("extra field");
    assert!(err.to_string().contains("extra"));
}
