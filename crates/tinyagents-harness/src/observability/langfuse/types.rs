//! Type definitions for the Langfuse ingestion exporter.
//!
//! Split out of `langfuse/mod.rs`; see that module's doc comment for the
//! exporter overview.

use serde::Serialize;
use serde_json::Value;

/// Authentication mode for [`LangfuseClient`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LangfuseAuth {
    /// Send `Authorization: Basic base64(public_key:secret_key)`.
    Basic {
        /// Langfuse project public key.
        public_key: String,
        /// Langfuse project secret key.
        secret_key: String,
    },
    /// Send `Authorization: Bearer <token>`.
    ///
    /// Use this when targeting the TinyHumans backend proxy at
    /// `/telemetry/langfuse/ingestion`; the backend injects Langfuse Basic Auth.
    Bearer {
        /// Backend access token.
        token: String,
    },
}

/// Configuration for a Langfuse trace export.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LangfuseTraceConfig {
    /// Stable Langfuse trace id. Defaults to the first observation's root run id.
    pub trace_id: Option<String>,
    /// Human-readable trace name.
    pub name: Option<String>,
    /// End-user id to filter by in Langfuse.
    pub user_id: Option<String>,
    /// Session/thread id to group related traces.
    pub session_id: Option<String>,
    /// Langfuse environment name.
    pub environment: Option<String>,
    /// Release identifier.
    pub release: Option<String>,
    /// Version identifier.
    pub version: Option<String>,
    /// Tags attached to the trace.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Extra trace metadata.
    #[serde(default)]
    pub metadata: Value,
}

/// The value of a [`LangfuseScore`], typed the way Langfuse classifies scores.
///
/// Langfuse stores three score data types; the variant selected here maps to
/// the `dataType` and the shape of the `value` field in the `score-create`
/// ingestion event (a number for numeric/boolean, a string for categorical).
#[derive(Clone, Debug, PartialEq)]
pub enum LangfuseScoreValue {
    /// A continuous numeric score (`dataType: "NUMERIC"`).
    Numeric(f64),
    /// A discrete label (`dataType: "CATEGORICAL"`), e.g. `"correct"`.
    Categorical(String),
    /// A pass/fail score (`dataType: "BOOLEAN"`), serialized as `1`/`0`.
    Boolean(bool),
}

impl LangfuseScoreValue {
    /// Returns the Langfuse `dataType` string for this value.
    pub fn data_type(&self) -> &'static str {
        match self {
            LangfuseScoreValue::Numeric(_) => "NUMERIC",
            LangfuseScoreValue::Categorical(_) => "CATEGORICAL",
            LangfuseScoreValue::Boolean(_) => "BOOLEAN",
        }
    }

    /// Returns the JSON `value` payload: a number for numeric/boolean scores, a
    /// string for categorical ones.
    pub fn to_value(&self) -> Value {
        match self {
            LangfuseScoreValue::Numeric(n) => Value::from(*n),
            LangfuseScoreValue::Categorical(s) => Value::from(s.clone()),
            LangfuseScoreValue::Boolean(b) => Value::from(if *b { 1 } else { 0 }),
        }
    }
}

/// An evaluation score attached to a Langfuse trace or a single observation.
///
/// Mirrors Langfuse's `createScore` / `score-create` ingestion event: a named,
/// typed value scoped to a `trace_id` and optionally narrowed to one
/// `observation_id` (a specific generation or span), with an optional free-text
/// comment. This is how post-hoc evaluations — human ratings, automated
/// LLM-as-judge checks, regression metrics — are correlated back to the run
/// that produced them.
#[derive(Clone, Debug, PartialEq)]
pub struct LangfuseScore {
    /// The trace the score is attached to.
    pub trace_id: String,
    /// A specific observation (generation/span) to scope the score to, when the
    /// score grades one step rather than the whole trace.
    pub observation_id: Option<String>,
    /// The score name (its metric key), e.g. `"helpfulness"`.
    pub name: String,
    /// The typed score value.
    pub value: LangfuseScoreValue,
    /// Optional free-text rationale stored alongside the score.
    pub comment: Option<String>,
    /// Stable score id for idempotent re-ingestion. Defaults (in
    /// [`LangfuseClient::build_score_batch`]) to a deterministic id derived from
    /// the trace, observation, and name, so re-scoring the same target updates
    /// the existing score instead of creating a duplicate.
    pub id: Option<String>,
}

impl LangfuseScore {
    /// Builds a trace-level numeric score.
    pub fn numeric(trace_id: impl Into<String>, name: impl Into<String>, value: f64) -> Self {
        Self {
            trace_id: trace_id.into(),
            observation_id: None,
            name: name.into(),
            value: LangfuseScoreValue::Numeric(value),
            comment: None,
            id: None,
        }
    }

    /// Builds a trace-level categorical score.
    pub fn categorical(
        trace_id: impl Into<String>,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            trace_id: trace_id.into(),
            observation_id: None,
            name: name.into(),
            value: LangfuseScoreValue::Categorical(value.into()),
            comment: None,
            id: None,
        }
    }

    /// Builds a trace-level boolean score.
    pub fn boolean(trace_id: impl Into<String>, name: impl Into<String>, value: bool) -> Self {
        Self {
            trace_id: trace_id.into(),
            observation_id: None,
            name: name.into(),
            value: LangfuseScoreValue::Boolean(value),
            comment: None,
            id: None,
        }
    }

    /// Scopes the score to a single observation (generation/span) id.
    pub fn on_observation(mut self, observation_id: impl Into<String>) -> Self {
        self.observation_id = Some(observation_id.into());
        self
    }

    /// Attaches a free-text comment to the score.
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Overrides the auto-derived score id (for a caller-controlled identity).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// Async Langfuse ingestion client.
#[derive(Clone, Debug)]
pub struct LangfuseClient {
    pub(super) endpoint: String,
    pub(super) auth: LangfuseAuth,
    pub(super) client: reqwest::Client,
}
