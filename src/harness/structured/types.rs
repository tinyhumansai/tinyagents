//! Public types for the structured-output module.
//!
//! All user-visible structs and enums live here so [`super`] can provide clean
//! implementations without mixing type definitions and method bodies.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// How the harness extracts a structured value from a model response.
///
/// * [`ProviderSchema`] – the provider was asked to produce JSON conforming to
///   a schema via a native response-format API; the structured value is parsed
///   from the raw response text.
/// * [`ToolCall`] – an artificial tool was exposed to the model; the structured
///   value is read from the matching tool-call's `arguments` field.
///
/// [`ProviderSchema`]: StructuredStrategy::ProviderSchema
/// [`ToolCall`]: StructuredStrategy::ToolCall
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredStrategy {
    /// Parse the JSON from the model's text response (provider-native mode).
    ProviderSchema,
    /// Read the arguments of a matching tool call.
    ToolCall,
}

// ---------------------------------------------------------------------------
// StructuredOutput
// ---------------------------------------------------------------------------

/// A validated structured value extracted from a [`ModelResponse`].
///
/// Carries the parsed JSON [`Value`] and, when available, the raw assistant
/// text that was parsed (useful for debugging or provider-native mode).
///
/// [`ModelResponse`]: crate::harness::model::ModelResponse
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutput {
    /// The extracted JSON value.
    pub value: Value,
    /// The raw assistant text that was parsed, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
}

// ---------------------------------------------------------------------------
// StructuredExtractor
// ---------------------------------------------------------------------------

/// Extracts a [`StructuredOutput`] from a [`ModelResponse`] using the
/// configured [`StructuredStrategy`].
///
/// # Example
///
/// ```rust
/// use rustagents::harness::structured::{StructuredExtractor, StructuredStrategy};
/// use rustagents::harness::model::ModelResponse;
/// use serde_json::json;
///
/// let extractor = StructuredExtractor::new(
///     StructuredStrategy::ProviderSchema,
///     "answer",
///     json!({ "type": "object", "properties": { "value": { "type": "string" } } }),
/// );
/// let response = ModelResponse::assistant(r#"{"value":"hello"}"#);
/// let output = extractor.extract(&response).unwrap();
/// assert_eq!(output.value["value"], "hello");
/// ```
#[derive(Clone, Debug)]
pub struct StructuredExtractor {
    /// How to locate the structured value in the response.
    pub(crate) strategy: StructuredStrategy,
    /// Name used to match the artificial tool call (for [`StructuredStrategy::ToolCall`])
    /// or to label errors.
    pub(crate) schema_name: String,
    /// The JSON Schema document (kept for potential future local validation).
    pub(crate) schema: Value,
}
