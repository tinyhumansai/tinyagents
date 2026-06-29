//! Structured output.
//!
//! Owns response formats, JSON schema validation, provider-native structured
//! output, tool-call fallback structured output, parsed typed responses, and
//! validation errors.
//!
//! # Overview
//!
//! Two strategies are supported:
//!
//! | [`StructuredStrategy`]  | How it works                                                  |
//! |-------------------------|---------------------------------------------------------------|
//! | `ProviderSchema`        | Provider returns JSON text; [`StructuredExtractor`] parses it |
//! | `ToolCall`              | An artificial tool call carries the arguments as JSON         |
//!
//! Use [`response_format_for_strategy`] to obtain the correct
//! [`ResponseFormat`] to include in a [`ModelRequest`], then call
//! [`StructuredExtractor::extract`] on the completed [`ModelResponse`].
//!
//! # Example
//!
//! ```rust
//! use tinyagents::harness::structured::{
//!     StructuredExtractor, StructuredStrategy, response_format_for_strategy,
//! };
//! use tinyagents::harness::model::ModelResponse;
//! use serde_json::json;
//!
//! let schema = json!({ "type": "object", "properties": { "score": { "type": "number" } } });
//! let _fmt = response_format_for_strategy(StructuredStrategy::ProviderSchema, "score_result", schema.clone());
//!
//! let extractor = StructuredExtractor::new(StructuredStrategy::ProviderSchema, "score_result", schema);
//! let response = ModelResponse::assistant(r#"{"score":42}"#);
//! let output = extractor.extract(&response).unwrap();
//! assert_eq!(output.value["score"], 42);
//! ```

mod types;

pub use types::*;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};
use crate::harness::model::{ModelProfile, ModelResponse, ResponseFormat};

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

impl StructuredStrategy {
    /// Chooses a strategy for [`ResponseFormat::Auto`] based on a model profile.
    ///
    /// Returns [`StructuredStrategy::ProviderSchema`] when the model advertises
    /// native structured output *and* JSON Schema support, or when no profile is
    /// available (the conservative default). Otherwise returns
    /// [`StructuredStrategy::ToolCall`], which works on any tool-calling model.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tinyagents::harness::structured::StructuredStrategy;
    /// use tinyagents::harness::model::ModelProfile;
    ///
    /// // No profile -> provider-native schema mode.
    /// assert_eq!(
    ///     StructuredStrategy::for_profile(None),
    ///     StructuredStrategy::ProviderSchema
    /// );
    ///
    /// // A tool-calling model without native structured output -> tool call.
    /// let mut profile = ModelProfile { tool_calling: true, ..ModelProfile::default() };
    /// assert_eq!(
    ///     StructuredStrategy::for_profile(Some(&profile)),
    ///     StructuredStrategy::ToolCall
    /// );
    ///
    /// // A model with native structured output -> provider schema.
    /// profile.native_structured_output = true;
    /// profile.json_schema = true;
    /// assert_eq!(
    ///     StructuredStrategy::for_profile(Some(&profile)),
    ///     StructuredStrategy::ProviderSchema
    /// );
    /// ```
    pub fn for_profile(profile: Option<&ModelProfile>) -> StructuredStrategy {
        match profile {
            Some(p) if !(p.native_structured_output && p.json_schema) => {
                StructuredStrategy::ToolCall
            }
            _ => StructuredStrategy::ProviderSchema,
        }
    }
}

// ---------------------------------------------------------------------------
// StructuredOutput
// ---------------------------------------------------------------------------

impl StructuredOutput {
    /// Returns a reference to the inner JSON [`Value`].
    pub fn as_value(&self) -> &Value {
        &self.value
    }

    /// Deserialises the inner JSON value into `T`.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::StructuredOutput`] when the value cannot be
    /// deserialised into `T`.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.value.clone())
            .map_err(|e| TinyAgentsError::StructuredOutput(format!("deserialisation failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// StructuredExtractor
// ---------------------------------------------------------------------------

impl StructuredExtractor {
    /// Creates a new extractor.
    ///
    /// * `strategy` – whether to use provider-schema or tool-call extraction.
    /// * `schema_name` – the schema's logical name; used as the tool name when
    ///   matching tool calls in [`StructuredStrategy::ToolCall`] mode.
    /// * `schema` – the JSON Schema document (retained for future local
    ///   validation, not yet applied).
    pub fn new(
        strategy: StructuredStrategy,
        schema_name: impl Into<String>,
        schema: Value,
    ) -> Self {
        Self {
            strategy,
            schema_name: schema_name.into(),
            schema,
        }
    }

    /// Returns the JSON Schema document this extractor was configured with.
    ///
    /// Retained for local validation and for echoing the schema back into a
    /// [`ResponseFormat`] when re-requesting structured output.
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Extracts a [`StructuredOutput`] from `response` using the configured
    /// strategy.
    ///
    /// # Strategies
    ///
    /// * **[`StructuredStrategy::ProviderSchema`]** – calls
    ///   [`ModelResponse::text`] and parses the result as JSON.  Returns
    ///   [`TinyAgentsError::StructuredOutput`] when the text is not valid JSON.
    ///
    /// * **[`StructuredStrategy::ToolCall`]** – scans the response's tool
    ///   calls for the first one whose `name` matches
    ///   [`StructuredExtractor::schema_name`] and returns its `arguments` as
    ///   the structured value.  Returns [`TinyAgentsError::Validation`] when no
    ///   matching call is found.
    ///
    /// # Errors
    ///
    /// See strategy descriptions above.
    pub fn extract(&self, response: &ModelResponse) -> Result<StructuredOutput> {
        match self.strategy {
            StructuredStrategy::ProviderSchema => self.extract_provider_schema(response),
            StructuredStrategy::ToolCall => self.extract_tool_call(response),
        }
    }

    // -- private helpers --

    fn extract_provider_schema(&self, response: &ModelResponse) -> Result<StructuredOutput> {
        let raw = response.text();
        let value: Value = serde_json::from_str(&raw).map_err(|e| {
            TinyAgentsError::StructuredOutput(format!(
                "schema '{}': response text is not valid JSON: {e}",
                self.schema_name
            ))
        })?;
        Ok(StructuredOutput {
            value,
            raw_text: Some(raw),
        })
    }

    fn extract_tool_call(&self, response: &ModelResponse) -> Result<StructuredOutput> {
        let call = response
            .tool_calls()
            .iter()
            .find(|tc| tc.name == self.schema_name)
            .ok_or_else(|| {
                TinyAgentsError::Validation(format!(
                    "schema '{}': no tool call with that name found in response",
                    self.schema_name
                ))
            })?;
        Ok(StructuredOutput {
            value: call.arguments.clone(),
            raw_text: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Returns the [`ResponseFormat`] appropriate for the given `strategy`.
///
/// | strategy        | result                          |
/// |-----------------|---------------------------------|
/// | `ProviderSchema`| `ResponseFormat::JsonSchema`    |
/// | `ToolCall`      | `ResponseFormat::Text`          |
///
/// For `ProviderSchema` the caller should also call
/// [`StructuredExtractor::extract`] after the model responds.
///
/// For `ToolCall` the caller is responsible for registering an artificial
/// tool with the given `name` and `schema` in the [`ModelRequest`]; the
/// response format is plain text because the structure arrives via tool
/// arguments.
pub fn response_format_for_strategy(
    strategy: StructuredStrategy,
    name: impl Into<String>,
    schema: Value,
) -> ResponseFormat {
    match strategy {
        StructuredStrategy::ProviderSchema => ResponseFormat::json_schema(name, schema),
        StructuredStrategy::ToolCall => ResponseFormat::Text,
    }
}

#[cfg(test)]
mod test;
