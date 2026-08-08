//! Public types for the structured-output module.
//!
//! All user-visible structs and enums live here so [`super`] can provide clean
//! implementations without mixing type definitions and method bodies.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::model::ModelResponse;

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
// StructuredOutcome
// ---------------------------------------------------------------------------

/// The result of attempting structured extraction, **including the failures**.
///
/// [`StructuredExtractor::extract`][ex] returns `Result<StructuredOutput>`, so
/// a parse or validation failure is fatal to whatever is driving it — on the
/// agent loop's final turn that discards a whole run, every tool call and token
/// already spent, over one malformed brace. This type is the non-fatal
/// alternative: the value when there is one, the raw response either way, and
/// the error as *data* rather than control flow.
///
/// Modelled on LangChain's `include_raw=True`, which wraps the parser in a
/// fallback yielding `{"raw", "parsed": None, "parsing_error"}` instead of
/// raising.
///
/// [ex]: StructuredExtractor::extract
#[derive(Clone, Debug)]
pub struct StructuredOutcome {
    /// The extracted and validated value, when extraction succeeded.
    pub value: Option<Value>,
    /// The model response extraction was attempted on, always preserved — it
    /// is the only evidence of what the model actually said, and the input a
    /// repair or re-ask turn needs.
    pub raw: ModelResponse,
    /// Why extraction failed, when it did. Written to be handed back to a model
    /// verbatim: it names the schema and, for a validation failure, the exact
    /// failing instance path.
    pub error: Option<String>,
}

impl StructuredOutcome {
    /// Whether a value was extracted.
    pub fn is_success(&self) -> bool {
        self.value.is_some()
    }

    /// The extracted value, or the recorded error as a
    /// [`TinyAgentsError::StructuredOutput`][err].
    ///
    /// Use this at a boundary that genuinely cannot proceed without a value;
    /// prefer matching on [`Self::value`] where a repair or re-ask is possible.
    ///
    /// [err]: crate::error::TinyAgentsError::StructuredOutput
    pub fn into_result(self) -> crate::error::Result<Value> {
        match self.value {
            Some(value) => Ok(value),
            None => Err(crate::error::TinyAgentsError::StructuredOutput(
                self.error
                    .unwrap_or_else(|| "structured extraction failed".to_string()),
            )),
        }
    }
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
/// use tinyagents::harness::structured::{StructuredExtractor, StructuredStrategy};
/// use tinyagents::harness::model::ModelResponse;
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
    /// The JSON Schema document. **Enforced**: every extracted value is checked
    /// against it by [`super::validate`] before it is returned, so a
    /// well-formed value of the wrong shape is a reported error rather than
    /// silent garbage in `run.structured`.
    pub(crate) schema: Value,
}
