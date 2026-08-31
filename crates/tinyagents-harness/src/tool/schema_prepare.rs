//! The provider projection seam for tool declarations.
//!
//! # Why this module exists
//!
//! [`SchemaCleanr`][super::SchemaCleanr] has always known how to resolve
//! `$ref`/`$defs`, strip per-provider unsupported keywords, flatten literal
//! unions, and break `$ref` cycles — and nothing in the crate ever called it.
//! It was dead code for a structural reason, not an oversight: the tool layer
//! had no *place* where a provider-specific projection of a
//! [`ToolSchema`] happens. Every adapter took `Tool::schema()` and put it on
//! the wire unchanged, so a schema written with `$defs` (which is what every
//! JSON-Schema generator emits for a nested type) reached Gemini and Anthropic
//! in a shape they reject.
//!
//! LangChain has exactly one such seam and applies it unconditionally:
//! `convert_to_openai_function` dereferences refs and pops `definitions`/`$defs`
//! for **every** conversion. This module is that seam.
//!
//! # Using it
//!
//! A provider adapter converts its declarations through
//! [`prepare_tool_schemas`] instead of reading `Tool::schema()` directly:
//!
//! ```
//! use tinyagents_harness::tool::{CleaningStrategy, SchemaPreparation, prepare_tool_schemas};
//! use tinyinference::tool::ToolSchema;
//! use serde_json::json;
//!
//! let declared = vec![ToolSchema::new(
//!     "lookup",
//!     "Look a record up",
//!     json!({
//!         "type": "object",
//!         "$defs": {"Id": {"type": "string"}},
//!         "properties": {"id": {"$ref": "#/$defs/Id"}},
//!     }),
//! )];
//!
//! let wire = prepare_tool_schemas(&declared, &SchemaPreparation::anthropic());
//! // The `$ref` is resolved and `$defs` is gone — Anthropic accepts this.
//! assert_eq!(wire[0].parameters["properties"]["id"]["type"], "string");
//! assert!(wire[0].parameters.get("$defs").is_none());
//! ```
//!
//! # Strict mode
//!
//! OpenAI's structured tool calling (`strict: true`) additionally demands that
//! every declared property be listed in `required` and that
//! `additionalProperties` be `false` at every object level.
//! [`SchemaPreparation::strict`] applies both, mirroring
//! `convert_to_openai_function(..., strict=True)`.

use serde_json::{Map, Value, json};

use super::schema::{CleaningStrategy, SchemaCleanr};
use super::types::ToolSchema;

/// How a [`ToolSchema`] should be projected for a specific provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaPreparation {
    /// Which keyword subset the target provider accepts.
    pub strategy: CleaningStrategy,
    /// Apply the OpenAI strict-mode sanitizer: force `required` to list every
    /// declared property, and set `additionalProperties: false` at every object
    /// level (overriding a pre-existing `true`).
    ///
    /// Off by default. Turning it on **changes the contract the model is given**
    /// — a previously optional argument becomes mandatory — so it belongs to the
    /// adapter that actually sends `strict: true`, not to the tool author.
    pub strict: bool,
}

impl SchemaPreparation {
    /// Projection for Gemini / Google AI / Vertex AI (the most restrictive
    /// keyword set).
    pub const fn gemini() -> Self {
        Self {
            strategy: CleaningStrategy::Gemini,
            strict: false,
        }
    }

    /// Projection for Anthropic (local refs must be resolved).
    pub const fn anthropic() -> Self {
        Self {
            strategy: CleaningStrategy::Anthropic,
            strict: false,
        }
    }

    /// Projection for OpenAI (most permissive; refs are still resolved).
    pub const fn openai() -> Self {
        Self {
            strategy: CleaningStrategy::OpenAI,
            strict: false,
        }
    }

    /// The conservative common subset, for an unknown OpenAI-compatible route.
    pub const fn conservative() -> Self {
        Self {
            strategy: CleaningStrategy::Conservative,
            strict: false,
        }
    }

    /// Enables the strict-mode sanitizer. See [`Self::strict`].
    pub const fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }
}

impl Default for SchemaPreparation {
    fn default() -> Self {
        Self::conservative()
    }
}

/// The parameter schema substituted for a missing or non-object declaration.
///
/// [`ToolSchema::parameters`] is an unconditional [`Value`], so a tool is free
/// to return `Value::Null`. Serialised as-is that becomes `"parameters": null`,
/// which every provider rejects with a `400` — a tool that simply takes no
/// arguments should not be able to break a whole request that way.
fn empty_object_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

/// Normalises a parameter schema into something a provider will accept as an
/// object schema.
///
/// `null`, a bare scalar, and an array are all replaced by
/// `{"type":"object","properties":{}}`. An object schema missing `type` gains
/// `type: "object"` when it declares `properties` or `required`, since that is
/// unambiguously what it meant.
pub fn normalize_parameters(parameters: &Value) -> Value {
    let Some(object) = parameters.as_object() else {
        tinyagents_tracing::debug!(
            "[tool::schema] non-object tool parameters ({}) replaced with an empty object schema",
            parameters_kind(parameters)
        );
        return empty_object_schema();
    };

    if object.is_empty() {
        return empty_object_schema();
    }

    let mut normalized = object.clone();
    if !normalized.contains_key("type")
        && (normalized.contains_key("properties") || normalized.contains_key("required"))
    {
        normalized.insert("type".to_string(), Value::String("object".to_string()));
    }
    Value::Object(normalized)
}

fn parameters_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Forces every declared property to be `required`.
///
/// OpenAI strict mode has no notion of an optional argument: a property absent
/// from `required` is rejected outright. Port of the `required` half of
/// `convert_to_openai_function(..., strict=True)`. A schema with no properties
/// is left alone.
pub fn require_all_properties(mut schema: Value) -> Value {
    let Some(object) = schema.as_object_mut() else {
        return schema;
    };
    let Some(Value::Object(properties)) = object.get("properties") else {
        return schema;
    };
    if properties.is_empty() {
        return schema;
    }
    let required: Vec<Value> = properties
        .keys()
        .map(|key| Value::String(key.clone()))
        .collect();
    object.insert("required".to_string(), Value::Array(required));
    schema
}

/// Sets `additionalProperties: false` at every object level that needs it,
/// **overriding** a pre-existing `true`.
///
/// Port of LangChain's `_recursive_set_additional_properties_false`, including
/// its three trigger conditions: the level declares `required`, it declares an
/// empty `properties`, or it already carries an `additionalProperties` key
/// (schema generators emit `additionalProperties: true` for open dictionaries,
/// and strict mode forbids that). Recurses through `anyOf`, `properties`, and
/// `items`.
pub fn set_additional_properties_false(mut schema: Value) -> Value {
    let Some(object) = schema.as_object_mut() else {
        return schema;
    };

    let empty_properties =
        matches!(object.get("properties"), Some(Value::Object(p)) if p.is_empty());
    if object.contains_key("required")
        || empty_properties
        || object.contains_key("additionalProperties")
    {
        object.insert("additionalProperties".to_string(), Value::Bool(false));
    }

    if let Some(Value::Array(variants)) = object.remove("anyOf") {
        let cleaned = variants
            .into_iter()
            .map(set_additional_properties_false)
            .collect();
        object.insert("anyOf".to_string(), Value::Array(cleaned));
    }

    if let Some(Value::Object(properties)) = object.remove("properties") {
        let cleaned: Map<String, Value> = properties
            .into_iter()
            .map(|(key, value)| (key, set_additional_properties_false(value)))
            .collect();
        object.insert("properties".to_string(), Value::Object(cleaned));
    }

    if let Some(items) = object.remove("items") {
        object.insert("items".to_string(), set_additional_properties_false(items));
    }

    schema
}

/// Projects one parameter schema for a provider: normalise, clean, then (when
/// requested) apply the strict-mode sanitizer.
///
/// The order matters. Cleaning runs before the strict sanitizer so that
/// `required` is computed from the *resolved* property set — a schema whose
/// properties arrive through a `$ref` would otherwise be marked as having none.
pub fn prepare_parameters(parameters: &Value, preparation: &SchemaPreparation) -> Value {
    let normalized = normalize_parameters(parameters);
    let cleaned = SchemaCleanr::clean(normalized, preparation.strategy);
    if !preparation.strict {
        return cleaned;
    }
    let required = require_all_properties(cleaned);
    set_additional_properties_false(required)
}

/// Projects one [`ToolSchema`] for a provider, leaving name, description, and
/// format untouched.
pub fn prepare_tool_schema(schema: &ToolSchema, preparation: &SchemaPreparation) -> ToolSchema {
    let prepared = ToolSchema {
        name: schema.name.clone(),
        description: schema.description.clone(),
        parameters: prepare_parameters(&schema.parameters, preparation),
        format: schema.format.clone(),
    };
    tinyagents_tracing::trace!(
        "[tool::schema] prepared `{}` for {:?} (strict={})",
        schema.name,
        preparation.strategy,
        preparation.strict
    );
    prepared
}

/// Projects a whole declaration set. This is the call a provider adapter makes.
pub fn prepare_tool_schemas(
    schemas: &[ToolSchema],
    preparation: &SchemaPreparation,
) -> Vec<ToolSchema> {
    tinyagents_tracing::debug!(
        "[tool::schema] preparing {} tool declaration(s) for {:?} (strict={})",
        schemas.len(),
        preparation.strategy,
        preparation.strict
    );
    schemas
        .iter()
        .map(|schema| prepare_tool_schema(schema, preparation))
        .collect()
}
