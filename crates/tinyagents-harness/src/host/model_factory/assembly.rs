//! Role-keyed model construction and turn model assembly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tinyinference_llm::model::workload::WORKLOAD_ROUTE_TIERS;
use tinyinference_llm::model::{
    ChatModel, ErrorSlotModel, ModelErrorSlot, ResolvedModelRoute, RouteRecordingModel,
};

use crate::error::Result;
use crate::host::model_resolver::PrebuiltModelResolver;

use super::types::{BuildOpts, ModelBuildGate, ModelBuildRequest, ModelFactory, ModelPurpose};
pub struct TurnModelSetBuilder<'a, State: Send + Sync, F: ?Sized, G: ?Sized> {
    factory: &'a F,
    gate: &'a G,
    primary: ModelBuildRequest,
    tiers: Vec<ModelBuildRequest>,
    summarization_role: Option<String>,
    summarization_pin: Option<String>,
    summarization_opts: Option<BuildOpts>,
    workload_opts: Option<BuildOpts>,
    workload_tier_opts: HashMap<String, BuildOpts>,
    marker: std::marker::PhantomData<fn() -> State>,
}

impl<'a, State: Send + Sync + 'static, F: ModelFactory<State> + ?Sized, G: ModelBuildGate + ?Sized>
    TurnModelSetBuilder<'a, State, F, G>
{
    /// Configures the default workload routes with `primary_role`.
    pub fn new(factory: &'a F, gate: &'a G, primary_role: impl Into<String>) -> Self {
        let primary = ModelBuildRequest::new(primary_role, ModelPurpose::Primary);
        let tiers = WORKLOAD_ROUTE_TIERS
            .iter()
            .map(|route| {
                let mut request = ModelBuildRequest::new(
                    route.strip_prefix("hint:").unwrap_or(route),
                    ModelPurpose::WorkloadTier((*route).to_owned()),
                );
                request.route = Some((*route).to_owned());
                request.model_pin = Some((*route).to_owned());
                request
            })
            .collect();
        Self {
            factory,
            gate,
            primary,
            tiers,
            summarization_role: None,
            summarization_pin: None,
            summarization_opts: None,
            workload_opts: None,
            workload_tier_opts: HashMap::new(),
            marker: std::marker::PhantomData,
        }
    }

    /// Begins a turn model set using the canonical workload routes.
    pub fn builder(factory: &'a F, gate: &'a G, primary_role: impl Into<String>) -> Self {
        Self::new(factory, gate, primary_role)
    }

    /// Pins only the primary model; workload-tier requests remain independent.
    #[must_use]
    pub fn primary_model_pin(mut self, model_pin: impl Into<String>) -> Self {
        self.primary.model_pin = Some(model_pin.into());
        self
    }

    /// Sets construction options for the primary model.
    #[must_use]
    pub fn primary_opts(mut self, opts: BuildOpts) -> Self {
        self.primary.opts = opts;
        self
    }

    /// Replaces default workload routes with caller-defined route aliases.
    /// A `hint:` prefix is removed for the default role key; use
    /// [`workload_requests`](Self::workload_requests) for other role keys.
    #[must_use]
    pub fn workload_tiers<I, S>(mut self, tiers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tiers = tiers
            .into_iter()
            .map(|route| {
                let route = route.into();
                let mut request = ModelBuildRequest::new(
                    route.strip_prefix("hint:").unwrap_or(&route),
                    ModelPurpose::WorkloadTier(route.clone()),
                );
                request.route = Some(route.clone());
                request.model_pin = Some(route);
                request
            })
            .collect();
        self
    }

    /// Replaces workload routes with fully specified neutral requests.
    #[must_use]
    pub fn workload_requests(mut self, tiers: Vec<ModelBuildRequest>) -> Self {
        self.tiers = tiers;
        self
    }

    /// Sets options shared by every workload-tier build. Use
    /// [`workload_tier_opts`](Self::workload_tier_opts) for a route-specific
    /// override.
    #[must_use]
    pub fn workload_opts(mut self, opts: BuildOpts) -> Self {
        self.workload_opts = Some(opts);
        self
    }

    /// Overrides shared workload options for one route alias.
    #[must_use]
    pub fn workload_tier_opts(mut self, route: impl Into<String>, opts: BuildOpts) -> Self {
        self.workload_tier_opts.insert(route.into(), opts);
        self
    }

    /// Omits workload routes, for example when the host supplies an injected
    /// primary model and should not construct unrelated external providers.
    #[must_use]
    pub fn without_workload_tiers(mut self) -> Self {
        self.tiers.clear();
        self
    }

    /// Selects an explicit summarization role and optional model pin.
    #[must_use]
    pub fn summarization(mut self, role: impl Into<String>, model_pin: Option<String>) -> Self {
        self.summarization_role = Some(role.into());
        self.summarization_pin = model_pin;
        self
    }

    /// Sets options for the summarization model. By default it inherits the
    /// primary model's options.
    #[must_use]
    pub fn summarization_opts(mut self, opts: BuildOpts) -> Self {
        self.summarization_opts = Some(opts);
        self
    }

    /// Builds all requested models. Primary and summarization failures are
    /// fatal; failed workload routes are logged and omitted.
    pub async fn build(self) -> Result<TurnModelSet<State>> {
        self.gate.check(&self.primary)?;
        let (primary_model, primary_route) = self.factory.build(&self.primary).await?;
        let primary = Arc::new(RouteRecordingModel::new(
            primary_model,
            primary_route.clone(),
        )) as Arc<dyn ChatModel<State>>;
        let primary_error_slot = Arc::new(Mutex::new(None));

        let mut tiers = Vec::new();
        let mut route_errors = HashMap::new();
        for mut request in self.tiers {
            if self
                .primary
                .model_pin
                .as_deref()
                .is_some_and(|pin| request.route.as_deref() == Some(pin))
            {
                tracing::debug!(route = ?request.route, "skipping workload model matching the primary pin");
                continue;
            }
            if let Some(opts) = &self.workload_opts {
                request.opts = opts.clone();
            }
            let route_name = request.route.as_deref().unwrap_or(&request.role);
            if let Some(opts) = self.workload_tier_opts.get(route_name) {
                request.opts = opts.clone();
            }
            if let Err(error) = self.gate.check(&request) {
                tracing::debug!(route = ?request.route, %error, "skipping gated workload model");
                continue;
            }
            match self.factory.build(&request).await {
                Ok((model, route)) => {
                    let slot = Arc::new(Mutex::new(None));
                    let recorded = Arc::new(RouteRecordingModel::new(model, route))
                        as Arc<dyn ChatModel<State>>;
                    let name = request
                        .route
                        .clone()
                        .unwrap_or_else(|| request.role.clone());
                    route_errors.insert(name.clone(), slot);
                    tiers.push((name, request.purpose, recorded));
                }
                Err(error) => {
                    tracing::debug!(route = ?request.route, %error, "skipping unavailable workload model");
                }
            }
        }

        let has_explicit_summary_role = self.summarization_role.is_some();
        let summary_request = ModelBuildRequest {
            role: self
                .summarization_role
                .unwrap_or_else(|| self.primary.role.clone()),
            route: None,
            purpose: ModelPurpose::Summarization,
            model_pin: self.summarization_pin.or_else(|| {
                (!has_explicit_summary_role)
                    .then_some(self.primary.model_pin)
                    .flatten()
            }),
            opts: self.summarization_opts.unwrap_or(self.primary.opts),
        };
        self.gate.check(&summary_request)?;
        let (summary_model, summary_route) = self.factory.build(&summary_request).await?;
        let summary_error_slot = Arc::new(Mutex::new(None));
        let summarizer = Arc::new(RouteRecordingModel::new(
            summary_model,
            summary_route.clone(),
        )) as Arc<dyn ChatModel<State>>;

        Ok(TurnModelSet {
            primary,
            tiers,
            primary_route,
            summary_route,
            summarizer,
            primary_error_slot,
            summary_error_slot,
            route_errors,
            _state: std::marker::PhantomData,
        })
    }
}

/// Built models for one turn configuration.
pub struct TurnModelSet<State: Send + Sync> {
    primary: Arc<dyn ChatModel<State>>,
    tiers: Vec<(String, ModelPurpose, Arc<dyn ChatModel<State>>)>,
    primary_route: ResolvedModelRoute,
    summary_route: ResolvedModelRoute,
    summarizer: Arc<dyn ChatModel<State>>,
    primary_error_slot: ModelErrorSlot,
    summary_error_slot: ModelErrorSlot,
    route_errors: HashMap<String, ModelErrorSlot>,
    _state: std::marker::PhantomData<fn() -> State>,
}

impl<State: Send + Sync + 'static> TurnModelSet<State> {
    /// Builds a turn's primary, canonical workload routes, and summarizer.
    pub async fn build<F, G>(factory: &F, gate: &G, primary_role: impl Into<String>) -> Result<Self>
    where
        F: ModelFactory<State> + ?Sized,
        G: ModelBuildGate + ?Sized,
    {
        Self::builder(factory, gate, primary_role).build().await
    }

    /// Starts building a model set using the standard workload aliases.
    pub fn builder<'a, F, G>(
        factory: &'a F,
        gate: &'a G,
        primary_role: impl Into<String>,
    ) -> TurnModelSetBuilder<'a, State, F, G>
    where
        F: ModelFactory<State> + ?Sized,
        G: ModelBuildGate + ?Sized,
    {
        TurnModelSetBuilder::new(factory, gate, primary_role)
    }

    /// Builds an owned route resolver from the already constructed and mapped models.
    pub fn resolver(&self) -> PrebuiltModelResolver<State> {
        let routes = self.tiers.iter().map(|(name, _, model)| {
            let slot = self
                .route_errors
                .get(name)
                .expect("built workload route has an error slot");
            (
                name.clone(),
                Arc::new(ErrorSlotModel::new(Arc::clone(model), Arc::clone(slot)))
                    as Arc<dyn ChatModel<State>>,
            )
        });
        PrebuiltModelResolver::new(Arc::clone(&self.primary), routes).with_lead_model(Arc::new(
            ErrorSlotModel::new(
                Arc::clone(&self.primary),
                Arc::clone(&self.primary_error_slot),
            ),
        ))
    }

    /// Applies a host-owned decorator once to the primary, every built
    /// workload route, and the summarizer before the resolver/error slots are
    /// assembled. Hosts use this for per-turn wrappers such as model budgets,
    /// then obtain `resolver`, `registrations`, and `summarizer` from the
    /// returned set. Any Arcs or resolver obtained earlier keep their prior
    /// model instances.
    #[must_use]
    pub fn map_models<F>(mut self, mut map: F) -> Self
    where
        F: FnMut(&ModelPurpose, Arc<dyn ChatModel<State>>) -> Arc<dyn ChatModel<State>>,
    {
        self.primary = map(&ModelPurpose::Primary, Arc::clone(&self.primary));
        for (_, purpose, model) in &mut self.tiers {
            *model = map(purpose, Arc::clone(model));
        }
        self.summarizer = map(&ModelPurpose::Summarization, Arc::clone(&self.summarizer));
        self
    }

    /// Primary model's concrete route.
    pub fn primary_route(&self) -> &ResolvedModelRoute {
        &self.primary_route
    }

    /// Summarization model's concrete route.
    pub fn summarizer_route(&self) -> &ResolvedModelRoute {
        &self.summary_route
    }

    /// Model reserved for summarization.
    pub fn summarizer(&self) -> Arc<dyn ChatModel<State>> {
        Arc::new(ErrorSlotModel::new(
            Arc::clone(&self.summarizer),
            Arc::clone(&self.summary_error_slot),
        ))
    }

    /// Returns primary and workload models in construction order for host
    /// turn planners that attach their own run-level error policy. The primary
    /// name is supplied by the host; workload names are their route aliases.
    /// Models are the already mapped instances, before this set's dedicated
    /// resolver error slots are applied.
    pub fn registrations(
        &self,
        primary_name: impl Into<String>,
    ) -> Vec<(String, Arc<dyn ChatModel<State>>)> {
        let mut models = Vec::with_capacity(1 + self.tiers.len());
        models.push((primary_name.into(), Arc::clone(&self.primary)));
        models.extend(
            self.tiers
                .iter()
                .map(|(name, _, model)| (name.clone(), Arc::clone(model))),
        );
        models
    }

    /// Error slot for team-lead calls routed through the primary model.
    pub fn primary_error_slot(&self) -> ModelErrorSlot {
        Arc::clone(&self.primary_error_slot)
    }

    /// Error slot for summarization calls.
    pub fn summarizer_error_slot(&self) -> ModelErrorSlot {
        Arc::clone(&self.summary_error_slot)
    }

    /// Error slot for a successfully built workload route.
    pub fn workload_error_slot(&self, route: &str) -> Option<ModelErrorSlot> {
        self.route_errors.get(route).cloned()
    }
}

#[cfg(test)]
#[path = "../model_factory_tests.rs"]
mod tests;
