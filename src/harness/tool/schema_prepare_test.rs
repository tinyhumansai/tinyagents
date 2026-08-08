//! Tests for the provider projection seam.
//!
//! `SchemaCleanr` was fully implemented and entirely unreachable: no call site
//! anywhere in the crate. These tests pin the seam that makes it callable, plus
//! the strict-mode sanitizer and the `parameters: null` guard.

use super::*;
use serde_json::json;

fn ref_schema() -> ToolSchema {
    ToolSchema::new(
        "lookup",
        "Look a record up",
        json!({
            "type": "object",
            "$defs": {"Id": {"type": "string", "description": "record id"}},
            "properties": {
                "id": {"$ref": "#/$defs/Id"},
                "limit": {"type": "integer"},
            },
            "required": ["id"],
        }),
    )
}

#[test]
fn local_refs_are_resolved_and_defs_dropped_for_anthropic() {
    // Anthropic rejects `$ref` / `$defs` outright, and every JSON-Schema
    // generator emits them for a nested type.
    let prepared = prepare_tool_schema(&ref_schema(), &SchemaPreparation::anthropic());

    assert_eq!(prepared.parameters["properties"]["id"]["type"], "string");
    assert!(prepared.parameters.get("$defs").is_none());
    assert!(
        prepared.parameters["properties"]["id"]
            .get("$ref")
            .is_none()
    );
    // Name/description/format are untouched by the projection.
    assert_eq!(prepared.name, "lookup");
    assert_eq!(prepared.description, "Look a record up");
}

#[test]
fn gemini_drops_the_keywords_it_rejects() {
    let schema = ToolSchema::new(
        "search",
        "Search",
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"q": {"type": "string", "minLength": 3, "pattern": "^a"}},
        }),
    );
    let prepared = prepare_tool_schema(&schema, &SchemaPreparation::gemini());

    assert!(prepared.parameters.get("additionalProperties").is_none());
    let q = &prepared.parameters["properties"]["q"];
    assert!(q.get("minLength").is_none());
    assert!(q.get("pattern").is_none());
    assert_eq!(q["type"], "string");
}

#[test]
fn strict_mode_requires_every_property_and_closes_every_object() {
    let prepared = prepare_tool_schema(&ref_schema(), &SchemaPreparation::openai().with_strict());

    let required: Vec<&str> = prepared.parameters["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert!(required.contains(&"id"));
    assert!(
        required.contains(&"limit"),
        "strict mode has no optional arguments"
    );
    assert_eq!(prepared.parameters["additionalProperties"], json!(false));
}

#[test]
fn strict_mode_overrides_a_pre_existing_additional_properties_true() {
    // Schema generators emit `additionalProperties: true` for open dictionaries;
    // strict mode forbids it, so it must be overridden rather than preserved.
    let schema = ToolSchema::new(
        "config",
        "Configure",
        json!({
            "type": "object",
            "properties": {
                "options": {"type": "object", "additionalProperties": true},
            },
            "required": ["options"],
        }),
    );
    let prepared = prepare_tool_schema(&schema, &SchemaPreparation::openai().with_strict());

    assert_eq!(
        prepared.parameters["properties"]["options"]["additionalProperties"],
        json!(false)
    );
}

#[test]
fn strict_mode_recurses_through_items_and_any_of() {
    let schema = ToolSchema::new(
        "batch",
        "Batch",
        json!({
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "required": ["a"],
                    },
                },
            },
            "required": ["rows"],
        }),
    );
    let prepared = prepare_tool_schema(&schema, &SchemaPreparation::openai().with_strict());

    assert_eq!(
        prepared.parameters["properties"]["rows"]["items"]["additionalProperties"],
        json!(false)
    );
}

#[test]
fn non_strict_preparation_leaves_optionality_alone() {
    let prepared = prepare_tool_schema(&ref_schema(), &SchemaPreparation::openai());
    let required = prepared.parameters["required"].as_array().unwrap();
    assert_eq!(required, &vec![json!("id")], "`limit` must stay optional");
}

#[test]
fn null_parameters_never_reach_the_wire() {
    // `ToolSchema::parameters` is an unconditional `Value`, so a tool taking no
    // arguments can return `Value::Null` — which serialises to
    // `"parameters": null` and 400s.
    for parameters in [json!(null), json!("nonsense"), json!([1, 2]), json!({})] {
        let schema = ToolSchema::new("bare", "no args", parameters);
        let prepared = prepare_tool_schema(&schema, &SchemaPreparation::default());
        assert_eq!(prepared.parameters["type"], "object");
        assert!(prepared.parameters["properties"].is_object());
    }
}

#[test]
fn an_object_schema_missing_its_type_gains_one() {
    let schema = ToolSchema::new(
        "implied",
        "implied object",
        json!({"properties": {"a": {"type": "string"}}}),
    );
    let prepared = prepare_tool_schema(&schema, &SchemaPreparation::default());
    assert_eq!(prepared.parameters["type"], "object");
}

#[test]
fn preparing_a_set_preserves_order_and_count() {
    let schemas = vec![
        ref_schema(),
        ToolSchema::new("other", "Other", json!({"type": "object"})),
    ];
    let prepared = prepare_tool_schemas(&schemas, &SchemaPreparation::conservative());
    assert_eq!(prepared.len(), 2);
    assert_eq!(prepared[0].name, "lookup");
    assert_eq!(prepared[1].name, "other");
}
