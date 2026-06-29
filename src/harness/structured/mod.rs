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
//! use rustagents::harness::structured::{
//!     StructuredExtractor, StructuredStrategy, response_format_for_strategy,
//! };
//! use rustagents::harness::model::ModelResponse;
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

use crate::error::{Result, RustAgentsError};
use crate::harness::model::{ModelResponse, ResponseFormat};

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
    /// Returns [`RustAgentsError::StructuredOutput`] when the value cannot be
    /// deserialised into `T`.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.value.clone()).map_err(|e| {
            RustAgentsError::StructuredOutput(format!("deserialisation failed: {e}"))
        })
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

    /// Extracts a [`StructuredOutput`] from `response` using the configured
    /// strategy.
    ///
    /// # Strategies
    ///
    /// * **[`StructuredStrategy::ProviderSchema`]** – calls
    ///   [`ModelResponse::text`] and parses the result as JSON.  Returns
    ///   [`RustAgentsError::StructuredOutput`] when the text is not valid JSON.
    ///
    /// * **[`StructuredStrategy::ToolCall`]** – scans the response's tool
    ///   calls for the first one whose `name` matches
    ///   [`StructuredExtractor::schema_name`] and returns its `arguments` as
    ///   the structured value.  Returns [`RustAgentsError::Validation`] when no
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
            RustAgentsError::StructuredOutput(format!(
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
                RustAgentsError::Validation(format!(
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
