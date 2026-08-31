//! Structured output.
//!
//! In the recursive architecture this is how a parent gets *typed values* back
//! out of a model call rather than free-form prose — the boundary that lets a
//! model's output become a program input. It underpins agents that return
//! machine-checkable results to their caller and, at the deepest level, a model
//! emitting a schema-conformant `.rag` blueprint or REPL plan that the same
//! runtime then compiles and runs.
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
//! # Repair, validation, and non-fatal extraction
//!
//! Extraction is not a bare `serde_json::from_str` any more. Three things
//! happen around it, each in its own submodule:
//!
//! * [`repair`] climbs a conservative ladder — code fence, prose slice,
//!   relaxed JSON, truncation close — so a fenced, chatty, or cut-off answer is
//!   recovered instead of ending a run. It never invents structure: a rung is
//!   accepted only when the repaired text parses strictly.
//! * [`validate`] checks the parsed value against the declared schema, so
//!   `{"wrong_key": 1}` against a `score` schema is a reported error naming the
//!   failing instance path — not a silent success.
//! * [`StructuredExtractor::extract_outcome`] returns a [`StructuredOutcome`]
//!   instead of `Result`, recording a failure as data so a caller can repair,
//!   re-ask, or return the raw response rather than losing the run.
//!
//! # Example
//!
//! ```rust
//! use tinyagents_harness::structured::{
//!     StructuredExtractor, StructuredStrategy, response_format_for_strategy,
//! };
//! use tinyinference::model::ModelResponse;
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

mod repair;
mod types;
mod validate;

pub use repair::JsonRepair;
pub use types::*;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};
use tinyinference::model::{ModelProfile, ModelResponse, ResponseFormat};

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

impl StructuredStrategy {
    /// Chooses a strategy for [`ResponseFormat::Auto`] based on a model profile.
    ///
    /// Returns [`StructuredStrategy::ProviderSchema`] when the model advertises
    /// native structured output *and* JSON Schema support, or when no profile is
    /// available (the conservative default). Otherwise returns
    /// [`StructuredStrategy::ToolCall`] — but **only for a model that can
    /// actually call tools**.
    ///
    /// # Why the `tool_calling` check matters
    ///
    /// This used to select `ToolCall` for *any* profile lacking native
    /// structured output, including profiles that declare `tool_calling:
    /// false`. That strategy declares an artificial tool and forces
    /// [`ToolChoice::Tool`][tc]; on a model with no tool support the harness
    /// runs prompt-guided instead, so the wire `tools` array is empty and the
    /// forced choice is dropped — leaving a request that asks for nothing in
    /// particular and an extractor waiting for a tool call that can never
    /// arrive. Provider-schema mode at least asks for JSON and, with the repair
    /// ladder in [`super::structured::repair`], parses what a JSON-mode model
    /// actually returns.
    ///
    /// [tc]: tinyinference::model::ToolChoice::Tool
    ///
    /// # Example
    ///
    /// ```rust
    /// use tinyagents_harness::structured::StructuredStrategy;
    /// use tinyinference::model::ModelProfile;
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
    ///
    /// // A model that can do neither -> JSON in the text, not a tool call.
    /// let plain = ModelProfile::default();
    /// assert_eq!(
    ///     StructuredStrategy::for_profile(Some(&plain)),
    ///     StructuredStrategy::ProviderSchema
    /// );
    /// ```
    pub fn for_profile(profile: Option<&ModelProfile>) -> StructuredStrategy {
        match profile {
            Some(p) if p.native_structured_output && p.json_schema => {
                StructuredStrategy::ProviderSchema
            }
            Some(p) if p.tool_calling => StructuredStrategy::ToolCall,
            // No native schema support and no tool calling: ask for JSON in the
            // text and lean on the repair ladder. A dedicated `JsonMode` arm
            // (plain JSON object + schema in the prompt, LangChain's third
            // `method`) is the refinement — see the module docs.
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
    /// * `schema` – the JSON Schema document. Enforced: every extracted value
    ///   is validated against it (see [`validate`]).
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
    /// Used for local validation and for echoing the schema back into a
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
    /// # Validation
    ///
    /// Both strategies validate the extracted value against this extractor's
    /// schema before returning it (see [`validate`]). A value that parses but
    /// does not conform is an error naming the failing instance path — not a
    /// success carrying the wrong shape.
    ///
    /// # Errors
    ///
    /// See strategy descriptions above.
    pub fn extract(&self, response: &ModelResponse) -> Result<StructuredOutput> {
        let output = match self.strategy {
            StructuredStrategy::ProviderSchema => self.extract_provider_schema(response)?,
            StructuredStrategy::ToolCall => self.extract_tool_call(response)?,
        };
        validate::validate_value(&self.schema, &output.value, &self.instance_root())?;
        Ok(output)
    }

    /// Extracts without failing: records the error instead of raising it.
    ///
    /// The difference is who decides what a failed extraction costs. `extract`
    /// decides for the caller — it returns `Err`, and on the agent loop's final
    /// turn that discards the entire run. This returns a
    /// [`StructuredOutcome`] instead, so a caller can log the failure and
    /// return the raw response, hand [`StructuredOutcome::error`] back to the
    /// model as a repair prompt (LangChain's `OutputFixingParser`), or re-ask
    /// with the original prompt (`RetryOutputParser`) — none of which are
    /// possible once the run has already been thrown away.
    ///
    /// Mirrors LangChain's `include_raw=True`.
    pub fn extract_outcome(&self, response: &ModelResponse) -> StructuredOutcome {
        match self.extract(response) {
            Ok(output) => StructuredOutcome {
                value: Some(output.value),
                raw: response.clone(),
                error: None,
            },
            Err(error) => {
                let error = error.to_string();
                tinyagents_tracing::debug!(
                    "[structured] extraction failed for schema '{}': {error}",
                    self.schema_name
                );
                StructuredOutcome {
                    value: None,
                    raw: response.clone(),
                    error: Some(error),
                }
            }
        }
    }

    // -- private helpers --

    /// The label validation errors are rooted at, for example `schema 'review'`.
    fn instance_root(&self) -> String {
        format!("schema '{}'", self.schema_name)
    }

    fn extract_provider_schema(&self, response: &ModelResponse) -> Result<StructuredOutput> {
        let raw = response.text();

        // A reasoning model can spend its entire output budget thinking and
        // return no content at all. The provider says so plainly through
        // `finish_reason: "length"`, but a naive parse of the empty string
        // reports "expected value at line 1 column 1" — which sends the reader
        // hunting for a malformed response that was never sent, and hides the
        // one-line fix of raising `max_tokens` or capping reasoning.
        if raw.trim().is_empty() {
            let truncated = response
                .finish_reason
                .as_deref()
                .is_some_and(|reason| reason == "length");
            return Err(TinyAgentsError::StructuredOutput(if truncated {
                format!(
                    "schema '{}': the model returned no content because it hit its output limit \
                     (finish_reason = \"length\"). Reasoning models can consume the whole budget \
                     before emitting an answer: raise `max_tokens`, or cap/disable reasoning.",
                    self.schema_name
                )
            } else {
                format!(
                    "schema '{}': the model returned no content (finish_reason = {:?})",
                    self.schema_name,
                    response.finish_reason.as_deref().unwrap_or("unknown")
                )
            }));
        }

        // Climb the repair ladder rather than a bare `from_str`. The crate
        // already repairs the *other* JSON a model emits (tool-call arguments);
        // there is no reason a fenced, chatty, or truncated structured answer
        // should end a run when the same repairs recover it.
        let Some((value, repair)) = repair::parse_lenient(&raw) else {
            return Err(TinyAgentsError::StructuredOutput(format!(
                "schema '{}': response text is not valid JSON and no conservative repair \
                 recovered it (finish_reason = {:?})",
                self.schema_name,
                response.finish_reason.as_deref().unwrap_or("unknown")
            )));
        };
        if repair.is_repaired() {
            tinyagents_tracing::debug!(
                "[structured] schema '{}': recovered the value with repair `{}`",
                self.schema_name,
                repair.as_str()
            );
        }
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

        // A provider that could not parse the call's arguments preserves them
        // as a raw string (`ToolCall::invalid`). Running the same repair ladder
        // here means a small local model's malformed arguments are recovered
        // rather than handed on as a JSON string masquerading as the value.
        if let Some(raw) = call.arguments.as_str()
            && let Some((value, repair)) = repair::parse_lenient(raw)
        {
            if repair.is_repaired() {
                tinyagents_tracing::debug!(
                    "[structured] schema '{}': recovered tool-call arguments with repair `{}`",
                    self.schema_name,
                    repair.as_str()
                );
            }
            return Ok(StructuredOutput {
                value,
                raw_text: Some(raw.to_string()),
            });
        }

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
