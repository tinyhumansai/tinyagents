use super::*;

use std::cell::RefCell;

use crate::error::TinyAgentsError;
use crate::host::model_resolver::{ModelResolveRequest, ModelResolver};
use tinyinference_llm::model::budget::{Budget, CallBudget, SpendLimits};
use tinyinference_llm::model::{ModelProfile, ModelRequest, ModelResponse, ProfileOverrideModel};

#[path = "model_factory_assembly_tests.rs"]
mod assembly;
#[path = "model_factory_resolver_tests.rs"]
mod resolver;
#[path = "model_factory_subprocess_tests.rs"]
mod subprocess;

fn workload(role: &str, route: &str, model_pin: Option<&str>) -> ModelBuildRequest {
    ModelBuildRequest {
        role: role.to_owned(),
        route: Some(route.to_owned()),
        purpose: ModelPurpose::WorkloadTier(route.to_owned()),
        model_pin: model_pin.map(str::to_owned),
        opts: BuildOpts::default(),
    }
}

#[derive(Default)]
struct NonCloneContext {
    primary_provider_override: Option<String>,
}

struct RecordingFactory<'a> {
    context: &'a NonCloneContext,
    requests: std::sync::Mutex<Vec<ModelBuildRequest>>,
    failed_route: Option<String>,
    fail_primary: bool,
    fail_summary: bool,
    primary_call_fails: bool,
    provider_calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl<'a> RecordingFactory<'a> {
    fn new(context: &'a NonCloneContext) -> Self {
        Self {
            context,
            requests: std::sync::Mutex::new(Vec::new()),
            failed_route: None,
            fail_primary: false,
            fail_summary: false,
            primary_call_fails: false,
            provider_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    fn failing_route(mut self, route: Option<&str>) -> Self {
        self.failed_route = route.map(str::to_owned);
        self
    }

    fn failing_primary(mut self) -> Self {
        self.fail_primary = true;
        self
    }

    fn failing_summary(mut self) -> Self {
        self.fail_summary = true;
        self
    }

    fn failing_primary_call(mut self) -> Self {
        self.primary_call_fails = true;
        self
    }
}

#[async_trait::async_trait]
impl ModelFactory<()> for RecordingFactory<'_> {
    async fn build(
        &self,
        request: &ModelBuildRequest,
    ) -> Result<(Arc<dyn ChatModel<()>>, ResolvedModelRoute)> {
        tokio::task::yield_now().await;
        self.requests.lock().unwrap().push(request.clone());
        if (self.fail_primary && request.purpose == ModelPurpose::Primary)
            || (self.fail_summary && request.purpose == ModelPurpose::Summarization)
            || self
                .failed_route
                .as_deref()
                .is_some_and(|route| request.route.as_deref() == Some(route))
        {
            return Err(TinyAgentsError::Model("fixture failure".into()));
        }
        let provider = if matches!(
            request.purpose,
            ModelPurpose::Primary | ModelPurpose::Summarization
        ) {
            self.context
                .primary_provider_override
                .as_deref()
                .unwrap_or("factory")
        } else {
            "factory"
        };
        let model_id = request
            .model_pin
            .as_deref()
            .or(request.route.as_deref())
            .unwrap_or(&request.role);
        let model: Arc<dyn ChatModel<()>> =
            if self.primary_call_fails && request.purpose == ModelPurpose::Primary {
                Arc::new(FailingCallModel)
            } else if !self.fail_primary && !self.fail_summary && self.failed_route.is_none() {
                Arc::new(CountingModel(Arc::clone(&self.provider_calls)))
            } else {
                Arc::new(tinyinference_llm::MockModel::constant(model_id))
            };
        Ok((
            model,
            ResolvedModelRoute::new(
                provider,
                model_id,
                request.route.as_deref().unwrap_or(&request.role),
            ),
        ))
    }
}

struct CountingModel(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait::async_trait]
impl ChatModel<()> for CountingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ModelResponse::assistant("provider called"))
    }
}

struct FailingCallModel;

#[async_trait::async_trait]
impl ChatModel<()> for FailingCallModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Err(tinyinference_llm::Error::Model(
            "synthetic call failure".to_owned(),
        ))
    }
}

#[derive(Default)]
struct RecordingGate {
    requests: std::sync::Mutex<Vec<ModelBuildRequest>>,
    denied_role: Option<String>,
    denied_route: Option<String>,
    denied_purpose: Option<ModelPurpose>,
}

impl RecordingGate {
    fn denying_role(mut self, role: Option<&str>) -> Self {
        self.denied_role = role.map(str::to_owned);
        self
    }

    fn denying_route(mut self, route: Option<&str>) -> Self {
        self.denied_route = route.map(str::to_owned);
        self
    }

    fn denying_purpose(mut self, purpose: Option<ModelPurpose>) -> Self {
        self.denied_purpose = purpose;
        self
    }
}

impl ModelBuildGate for RecordingGate {
    fn check(&self, request: &ModelBuildRequest) -> Result<()> {
        self.requests.lock().unwrap().push(request.clone());
        if self.denied_role.as_deref() == Some(&request.role)
            || self
                .denied_route
                .as_deref()
                .is_some_and(|route| request.route.as_deref() == Some(route))
            || self.denied_purpose.as_ref() == Some(&request.purpose)
        {
            return Err(TinyAgentsError::Model("blocked by fixture gate".into()));
        }
        Ok(())
    }
}

struct SendContractFactory;

#[async_trait::async_trait]
impl ModelFactory<()> for SendContractFactory {
    async fn build(
        &self,
        _request: &ModelBuildRequest,
    ) -> Result<(Arc<dyn ChatModel<()>>, ResolvedModelRoute)> {
        Err(TinyAgentsError::Model("compile-time fixture".into()))
    }
}

struct SendContractGate;

impl ModelBuildGate for SendContractGate {
    fn check(&self, _request: &ModelBuildRequest) -> Result<()> {
        Ok(())
    }
}

#[test]
fn turn_model_set_build_future_is_send() {
    fn assert_send<T: Send>(_: &T) {}

    let factory = SendContractFactory;
    let gate = SendContractGate;
    let future = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .build();

    assert_send(&future);
}
