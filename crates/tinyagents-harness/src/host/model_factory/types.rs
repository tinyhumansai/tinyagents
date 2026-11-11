use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tinyinference_llm::model::{ChatModel, ResolvedModelRoute};

use crate::error::Result;

/// Construction options passed through without interpretation by the harness.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BuildOpts {
    /// Whether the provider should expose native tool support.
    pub native_tools: bool,
    /// Optional provider temperature override.
    pub temperature: Option<f64>,
    /// Host-supplied egress accounting label.
    pub egress_tag: Option<String>,
}

/// Why a model is being constructed. Hosts can apply distinct policies to
/// primary, workload-tier, and summarization models without sharing policy
/// details with the harness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPurpose {
    /// Model selected for the primary turn.
    Primary,
    /// A workload route alias such as `hint:chat`.
    WorkloadTier(String),
    /// Model reserved for transcript summarization.
    Summarization,
}

/// Neutral facts a host may inspect before building one model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelBuildRequest {
    /// Host-defined role key.
    pub role: String,
    /// Optional host route alias, separate from the role and model id.
    pub route: Option<String>,
    /// Construction purpose.
    pub purpose: ModelPurpose,
    /// Optional requested provider model id or route alias. Hosts may resolve an
    /// alias to a concrete model; [`ResolvedModelRoute::model`] reports that result.
    pub model_pin: Option<String>,
    /// Provider construction options.
    pub opts: BuildOpts,
}

impl ModelBuildRequest {
    pub(super) fn new(role: impl Into<String>, purpose: ModelPurpose) -> Self {
        Self {
            role: role.into(),
            route: None,
            purpose,
            model_pin: None,
            opts: BuildOpts::default(),
        }
    }
}

/// Host policy hook called before every model construction.
///
/// Implementations are `Sync` so a turn-model assembly future may cross a
/// host task boundary while retaining the gate between model builds.
pub trait ModelBuildGate: Sync {
    /// Authorize constructing the model described by `request`.
    fn check(&self, request: &ModelBuildRequest) -> Result<()>;
}

/// Host implementation that selects and constructs a provider model.
///
/// Implementations are `Sync`, and [`build`](Self::build) returns a `Send`
/// future so the harness can assemble turn models inside sendable host tasks.
#[async_trait]
pub trait ModelFactory<State: Send + Sync>: Sync {
    /// Construct a model and report the concrete route selected by the host.
    async fn build(
        &self,
        request: &ModelBuildRequest,
    ) -> Result<(Arc<dyn ChatModel<State>>, ResolvedModelRoute)>;
}
