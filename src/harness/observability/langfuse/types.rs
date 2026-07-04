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

/// Async Langfuse ingestion client.
#[derive(Clone, Debug)]
pub struct LangfuseClient {
    pub(super) endpoint: String,
    pub(super) auth: LangfuseAuth,
    pub(super) client: reqwest::Client,
}
