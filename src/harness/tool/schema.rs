//! JSON Schema cleaning and validation for LLM tool-calling compatibility.
//!
//! Different providers accept different JSON Schema subsets for model-visible
//! tool declarations. This module normalizes schemas while preserving semantic
//! intent: it resolves local refs, removes provider-rejected keywords, flattens
//! simple literal unions, strips nullable variants, converts `const` to `enum`,
//! and breaks circular local refs safely.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::{Result, TinyAgentsError};

/// Keywords that Gemini rejects for tool schemas.
pub const GEMINI_UNSUPPORTED_KEYWORDS: &[&str] = &[
    "$ref",
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "additionalProperties",
    "patternProperties",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "examples",
];

const SCHEMA_META_KEYS: &[&str] = &["description", "title", "default"];

/// Schema cleaning strategies for different LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleaningStrategy {
    /// Gemini / Google AI / Vertex AI: most restrictive.
    Gemini,
    /// Anthropic Claude: moderately permissive, but local refs must be resolved.
    Anthropic,
    /// OpenAI: most permissive.
    OpenAI,
    /// Conservative common subset.
    Conservative,
}

impl CleaningStrategy {
    /// Returns the schema keywords rejected by this strategy.
    pub fn unsupported_keywords(self) -> &'static [&'static str] {
        match self {
            Self::Gemini => GEMINI_UNSUPPORTED_KEYWORDS,
            Self::Anthropic => &["$ref", "$defs", "definitions"],
            Self::OpenAI => &[],
            Self::Conservative => &["$ref", "$defs", "definitions", "additionalProperties"],
        }
    }
}

/// JSON Schema cleaner optimized for model tool-calling declarations.
pub struct SchemaCleanr;

impl SchemaCleanr {
    /// Cleans a schema for Gemini compatibility.
    pub fn clean_for_gemini(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::Gemini)
    }

    /// Cleans a schema for Anthropic compatibility.
    pub fn clean_for_anthropic(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::Anthropic)
    }

    /// Cleans a schema for OpenAI compatibility.
    pub fn clean_for_openai(schema: Value) -> Value {
        Self::clean(schema, CleaningStrategy::OpenAI)
    }

    /// Cleans a schema with the specified strategy.
    pub fn clean(schema: Value, strategy: CleaningStrategy) -> Value {
        let defs = if let Some(obj) = schema.as_object() {
            Self::extract_defs(obj)
        } else {
            HashMap::new()
        };

        Self::clean_with_defs(schema, &defs, strategy, &mut HashSet::new())
    }

    /// Validates that a schema is suitable for model tool calling.
    pub fn validate(schema: &Value) -> Result<()> {
        let obj = schema
            .as_object()
            .ok_or_else(|| TinyAgentsError::Validation("schema must be an object".to_string()))?;

        if !obj.contains_key("type") {
            return Err(TinyAgentsError::Validation(
                "schema missing required `type` field".to_string(),
            ));
        }

        if let Some(Value::String(schema_type)) = obj.get("type") {
            if schema_type == "object" && !obj.contains_key("properties") {
                // Valid but often rejected by providers; callers can decide how
                // strict they want to be after validation succeeds.
            }
        }

        Ok(())
    }

    fn extract_defs(obj: &Map<String, Value>) -> HashMap<String, Value> {
        let mut defs = HashMap::new();

        if let Some(Value::Object(defs_obj)) = obj.get("$defs") {
            for (key, value) in defs_obj {
                defs.insert(key.clone(), value.clone());
            }
        }

        if let Some(Value::Object(defs_obj)) = obj.get("definitions") {
            for (key, value) in defs_obj {
                defs.insert(key.clone(), value.clone());
            }
        }

        defs
    }

    fn clean_with_defs(
        schema: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        match schema {
            Value::Object(obj) => Self::clean_object(obj, defs, strategy, ref_stack),
            Value::Array(arr) => Value::Array(
                arr.into_iter()
                    .map(|value| Self::clean_with_defs(value, defs, strategy, ref_stack))
                    .collect(),
            ),
            other => other,
        }
    }

    fn clean_object(
        obj: Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if let Some(Value::String(ref_value)) = obj.get("$ref") {
            return Self::resolve_ref(ref_value, &obj, defs, strategy, ref_stack);
        }

        if (obj.contains_key("anyOf") || obj.contains_key("oneOf"))
            && let Some(simplified) = Self::try_simplify_union(&obj, defs, strategy, ref_stack)
        {
            return simplified;
        }

        let mut cleaned = Map::new();
        let unsupported: HashSet<&str> = strategy.unsupported_keywords().iter().copied().collect();
        let has_union = obj.contains_key("anyOf") || obj.contains_key("oneOf");

        for (key, value) in obj {
            if unsupported.contains(key.as_str()) {
                continue;
            }

            match key.as_str() {
                "const" => {
                    cleaned.insert("enum".to_string(), json!([value]));
                }
                "type" if has_union => {}
                "type" if matches!(value, Value::Array(_)) => {
                    cleaned.insert(key, Self::clean_type_array(value));
                }
                "properties" => {
                    cleaned.insert(
                        key,
                        Self::clean_properties(value, defs, strategy, ref_stack),
                    );
                }
                "items" => {
                    cleaned.insert(key, Self::clean_with_defs(value, defs, strategy, ref_stack));
                }
                "anyOf" | "oneOf" | "allOf" => {
                    cleaned.insert(key, Self::clean_union(value, defs, strategy, ref_stack));
                }
                _ => {
                    let cleaned_value = match value {
                        Value::Object(_) | Value::Array(_) => {
                            Self::clean_with_defs(value, defs, strategy, ref_stack)
                        }
                        other => other,
                    };
                    cleaned.insert(key, cleaned_value);
                }
            }
        }

        Value::Object(cleaned)
    }

    fn resolve_ref(
        ref_value: &str,
        obj: &Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if ref_stack.contains(ref_value) {
            return Self::preserve_meta(obj, Value::Object(Map::new()));
        }

        if let Some(def_name) = Self::parse_local_ref(ref_value) {
            if let Some(definition) = defs.get(def_name.as_str()) {
                ref_stack.insert(ref_value.to_string());
                let cleaned = Self::clean_with_defs(definition.clone(), defs, strategy, ref_stack);
                ref_stack.remove(ref_value);
                return Self::preserve_meta(obj, cleaned);
            }
        }

        Self::preserve_meta(obj, Value::Object(Map::new()))
    }

    fn parse_local_ref(ref_value: &str) -> Option<String> {
        ref_value
            .strip_prefix("#/$defs/")
            .or_else(|| ref_value.strip_prefix("#/definitions/"))
            .map(Self::decode_json_pointer)
    }

    fn decode_json_pointer(segment: &str) -> String {
        if !segment.contains('~') {
            return segment.to_string();
        }

        let mut decoded = String::with_capacity(segment.len());
        let mut chars = segment.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '~' {
                match chars.peek().copied() {
                    Some('0') => {
                        chars.next();
                        decoded.push('~');
                    }
                    Some('1') => {
                        chars.next();
                        decoded.push('/');
                    }
                    _ => decoded.push('~'),
                }
            } else {
                decoded.push(ch);
            }
        }

        decoded
    }

    fn try_simplify_union(
        obj: &Map<String, Value>,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Option<Value> {
        let union_key = if obj.contains_key("anyOf") {
            "anyOf"
        } else if obj.contains_key("oneOf") {
            "oneOf"
        } else {
            return None;
        };

        let variants = obj.get(union_key)?.as_array()?;
        let cleaned_variants: Vec<Value> = variants
            .iter()
            .map(|value| Self::clean_with_defs(value.clone(), defs, strategy, ref_stack))
            .collect();
        let non_null: Vec<Value> = cleaned_variants
            .into_iter()
            .filter(|value| !Self::is_null_schema(value))
            .collect();

        if non_null.len() == 1 {
            return Some(Self::preserve_meta(obj, non_null[0].clone()));
        }

        if let Some(enum_value) = Self::try_flatten_literal_union(&non_null) {
            return Some(Self::preserve_meta(obj, enum_value));
        }

        None
    }

    fn is_null_schema(value: &Value) -> bool {
        if let Some(obj) = value.as_object() {
            if let Some(Value::Null) = obj.get("const") {
                return true;
            }
            if let Some(Value::Array(arr)) = obj.get("enum") {
                if arr.len() == 1 && matches!(arr[0], Value::Null) {
                    return true;
                }
            }
            if let Some(Value::String(schema_type)) = obj.get("type") {
                if schema_type == "null" {
                    return true;
                }
            }
        }
        false
    }

    fn try_flatten_literal_union(variants: &[Value]) -> Option<Value> {
        if variants.is_empty() {
            return None;
        }

        let mut values = Vec::new();
        let mut common_type: Option<String> = None;

        for variant in variants {
            let obj = variant.as_object()?;
            let literal = if let Some(const_value) = obj.get("const") {
                const_value.clone()
            } else if let Some(Value::Array(arr)) = obj.get("enum") {
                if arr.len() == 1 {
                    arr[0].clone()
                } else {
                    return None;
                }
            } else {
                return None;
            };

            let variant_type = obj.get("type")?.as_str()?;
            match &common_type {
                None => common_type = Some(variant_type.to_string()),
                Some(existing) if existing != variant_type => return None,
                _ => {}
            }

            values.push(literal);
        }

        common_type.map(|schema_type| {
            json!({
                "type": schema_type,
                "enum": values
            })
        })
    }

    fn clean_type_array(value: Value) -> Value {
        if let Value::Array(types) = value {
            let non_null: Vec<Value> = types
                .into_iter()
                .filter(|value| value.as_str() != Some("null"))
                .collect();

            match non_null.len() {
                0 => Value::String("null".to_string()),
                1 => non_null
                    .into_iter()
                    .next()
                    .unwrap_or(Value::String("null".to_string())),
                _ => Value::Array(non_null),
            }
        } else {
            value
        }
    }

    fn clean_properties(
        value: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if let Value::Object(props) = value {
            let cleaned: Map<String, Value> = props
                .into_iter()
                .map(|(key, value)| (key, Self::clean_with_defs(value, defs, strategy, ref_stack)))
                .collect();
            Value::Object(cleaned)
        } else {
            value
        }
    }

    fn clean_union(
        value: Value,
        defs: &HashMap<String, Value>,
        strategy: CleaningStrategy,
        ref_stack: &mut HashSet<String>,
    ) -> Value {
        if let Value::Array(variants) = value {
            let cleaned: Vec<Value> = variants
                .into_iter()
                .map(|value| Self::clean_with_defs(value, defs, strategy, ref_stack))
                .collect();
            Value::Array(cleaned)
        } else {
            value
        }
    }

    fn preserve_meta(source: &Map<String, Value>, mut target: Value) -> Value {
        if let Value::Object(target_obj) = &mut target {
            for &key in SCHEMA_META_KEYS {
                if let Some(value) = source.get(key) {
                    target_obj.insert(key.to_string(), value.clone());
                }
            }
        }
        target
    }
}
