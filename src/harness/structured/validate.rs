//! Local validation of an extracted structured value against its declared
//! JSON Schema.
//!
//! # Why this exists
//!
//! [`StructuredExtractor`][super::StructuredExtractor] stored its schema and
//! never read it: provider-schema mode ran a bare
//! [`serde_json::from_str`] and tool-call mode cloned the call's arguments
//! straight through. So `{"wrong_key": 1}` against a `score` schema *succeeded*,
//! and `run.structured` came back holding something no caller had asked for —
//! a failure that surfaces later, somewhere else, as a missing field.
//!
//! Validating here also gives the repair loop something to say: an error naming
//! the exact failing instance path is a message that can be handed back to the
//! model, where "deserialisation failed" is not.
//!
//! # The supported subset
//!
//! The same subset the tool-call boundary enforces: `type` (including union
//! types), object `properties`, `required`, `additionalProperties: false`,
//! array `items`, and `enum`. Unknown keywords are ignored, so a richer schema
//! can still be sent to a provider while the local boundary fails closed on
//! exactly the structural constraints it understands. An empty or null schema
//! imposes no constraints.
//!
//! It is intentionally **not** a general JSON Schema implementation: no
//! `$ref`, no `allOf`/`anyOf`/`oneOf`, no numeric or string facets. Those
//! belong in a dedicated validator crate if the need ever arises; guessing at
//! them here would produce confident wrong answers.

use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

/// Validates `value` against `schema`, reporting the failing instance path.
///
/// `root` names the value in error messages — the caller passes something like
/// `schema 'review'` so the message reads `schema 'review'.items[2].id must be
/// integer, got string`.
pub fn validate_value(schema: &Value, value: &Value, root: &str) -> Result<()> {
    validate_at(schema, value, root)
}

fn validate_at(schema: &Value, value: &Value, path: &str) -> Result<()> {
    if schema.is_null() || schema.as_object().is_some_and(|map| map.is_empty()) {
        return Ok(());
    }

    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.iter().any(|candidate| candidate == value)
    {
        return Err(invalid(format!(
            "{path} must be one of the declared enum values"
        )));
    }

    if let Some(type_spec) = schema.get("type") {
        validate_type(type_spec, value, path)?;
    }

    // `required` is enforced independently of `properties`: a schema may name
    // required fields without describing them, and nesting the check under
    // `properties` would let such a schema fail open.
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        if let Some(object) = value.as_object() {
            for field in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(field) {
                    return Err(invalid(format!("{path}.{field} is required")));
                }
            }
        } else if schema.get("type").is_none() {
            return Err(invalid(format!(
                "{path} must be an object with the declared fields, got {}",
                kind_of(value)
            )));
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        if let Some(object) = value.as_object() {
            if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(invalid(format!("{path}.{field} is not allowed")));
                    }
                }
            }
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_at(field_schema, field_value, &format!("{path}.{field}"))?;
                }
            }
        } else if schema.get("type").is_none() {
            return Err(invalid(format!(
                "{path} must be an object with the declared fields, got {}",
                kind_of(value)
            )));
        }
    }

    if let Some(items_schema) = schema.get("items")
        && let Some(items) = value.as_array()
    {
        for (index, item) in items.iter().enumerate() {
            validate_at(items_schema, item, &format!("{path}[{index}]"))?;
        }
    }

    Ok(())
}

fn validate_type(type_spec: &Value, value: &Value, path: &str) -> Result<()> {
    if let Some(kind) = type_spec.as_str() {
        if matches_type(value, kind) {
            return Ok(());
        }
        return Err(invalid(format!(
            "{path} must be {kind}, got {}",
            kind_of(value)
        )));
    }

    if let Some(kinds) = type_spec.as_array() {
        let allowed: Vec<&str> = kinds.iter().filter_map(Value::as_str).collect();
        if allowed.iter().any(|kind| matches_type(value, kind)) {
            return Ok(());
        }
        return Err(invalid(format!(
            "{path} must be one of {}, got {}",
            allowed.join(", "),
            kind_of(value)
        )));
    }

    Ok(())
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        // An unknown type keyword must not fail closed: providers accept richer
        // vocabularies than this subset understands.
        _ => true,
    }
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn invalid(message: String) -> TinyAgentsError {
    TinyAgentsError::StructuredOutput(message)
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;

    fn score_schema() -> Value {
        json!({
            "type": "object",
            "properties": { "score": { "type": "integer" } },
            "required": ["score"],
            "additionalProperties": false
        })
    }

    #[test]
    fn accepts_a_conforming_value() {
        validate_value(&score_schema(), &json!({ "score": 4 }), "schema 'score'").unwrap();
    }

    #[test]
    fn rejects_a_missing_required_field_by_path() {
        let err = validate_value(&score_schema(), &json!({ "wrong_key": 1 }), "schema 'score'")
            .expect_err("a missing required field is not valid");
        assert!(err.to_string().contains("schema 'score'.score is required"), "{err}");
    }

    #[test]
    fn rejects_a_wrong_type_by_path() {
        let err = validate_value(
            &score_schema(),
            &json!({ "score": "four" }),
            "schema 'score'",
        )
        .expect_err("a string is not an integer");
        assert!(
            err.to_string().contains("schema 'score'.score must be integer, got string"),
            "{err}"
        );
    }

    #[test]
    fn reports_a_nested_array_index() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": { "type": "array", "items": { "type": "object", "properties": { "id": { "type": "integer" } } } }
            }
        });
        let err = validate_value(
            &schema,
            &json!({ "items": [{ "id": 1 }, { "id": "two" }] }),
            "schema 'batch'",
        )
        .expect_err("the second item is invalid");
        assert!(err.to_string().contains("items[1].id"), "{err}");
    }

    #[test]
    fn rejects_an_undeclared_field_when_additional_properties_is_false() {
        let err = validate_value(
            &score_schema(),
            &json!({ "score": 4, "extra": true }),
            "schema 'score'",
        )
        .expect_err("`extra` is not declared");
        assert!(err.to_string().contains("extra is not allowed"), "{err}");
    }

    #[test]
    fn an_empty_schema_constrains_nothing() {
        validate_value(&json!({}), &json!("anything at all"), "schema 'free'").unwrap();
        validate_value(&Value::Null, &json!(7), "schema 'free'").unwrap();
    }

    #[test]
    fn accepts_a_union_type() {
        let schema = json!({ "type": ["string", "null"] });
        validate_value(&schema, &json!(null), "schema 'maybe'").unwrap();
        validate_value(&schema, &json!("x"), "schema 'maybe'").unwrap();
        assert!(validate_value(&schema, &json!(3), "schema 'maybe'").is_err());
    }

    #[test]
    fn ignores_unknown_type_keywords() {
        // A provider may accept a richer vocabulary than this subset knows.
        validate_value(&json!({ "type": "date-time" }), &json!("2026-01-01"), "s").unwrap();
    }

    #[test]
    fn enforces_an_enum() {
        let schema = json!({ "enum": ["a", "b"] });
        validate_value(&schema, &json!("a"), "schema 'choice'").unwrap();
        assert!(validate_value(&schema, &json!("c"), "schema 'choice'").is_err());
    }
}
