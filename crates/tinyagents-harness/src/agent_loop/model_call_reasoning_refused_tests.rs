//! A provider endpoint that cannot run without reasoning refuses a request
//! that turns it off. The call is sent again with the provider's own default
//! instead of failing the run.

use super::*;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ProviderError, ReasoningConfig, ReasoningEffort,
};

use crate::retry::RetryPolicy;
use crate::runtime::{AgentHarness, RunPolicy};

const REFUSAL: &str = "Reasoning is mandatory for this endpoint and cannot be disabled.";

/// Answers every request but one that turns reasoning off, which it refuses
/// with `refusal` as HTTP 400. Records each request it was sent.
struct ReasoningRequired {
    refusal: &'static str,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ReasoningRequired {
    fn refusing_with(refusal: &'static str) -> Arc<Self> {
        Arc::new(Self {
            refusal,
            requests: Mutex::new(Vec::new()),
        })
    }

    fn efforts(&self) -> Vec<Option<ReasoningEffort>> {
        let requests = self.requests.lock().unwrap();
        requests
            .iter()
            .map(|request| request.reasoning.as_ref().and_then(|r| r.effort))
            .collect()
    }
}

#[async_trait]
impl ChatModel<()> for ReasoningRequired {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let off = request
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| reasoning.effort == Some(ReasoningEffort::None));
        self.requests.lock().unwrap().push(request);
        if off {
            return Err(tinyinference_llm::Error::Provider(Box::new(
                ProviderError {
                    provider: "test".into(),
                    status: Some(400),
                    message: self.refusal.into(),
                    retryable: false,
                    ..ProviderError::default()
                },
            )));
        }
        Ok(ModelResponse::assistant("answered"))
    }
}

/// A harness on `model` alone that sends every call at `effort`, with no
/// retry of its own.
fn harness(model: &Arc<ReasoningRequired>, effort: ReasoningEffort) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_model_resolver(Arc::new(crate::host::FixedModelResolver::named(
        "primary",
        model.clone(),
    )));
    harness.with_policy(RunPolicy {
        default_reasoning: Some(ReasoningConfig::effort(effort)),
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn a_refused_reasoning_off_call_is_sent_again_with_the_providers_default() {
    let model = ReasoningRequired::refusing_with(REFUSAL);
    let run = harness(&model, ReasoningEffort::None)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("the refusal is answered by resending, not by failing the run");
    assert_eq!(run.text(), Some("answered".to_string()));
    assert_eq!(
        model.efforts(),
        [Some(ReasoningEffort::None), None],
        "sent once more, with no reasoning setting"
    );
}

#[tokio::test]
async fn another_refusal_of_a_reasoning_off_call_still_fails() {
    let model = ReasoningRequired::refusing_with("Invalid request: unknown parameter.");
    let error = harness(&model, ReasoningEffort::None)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("a 400 about something else is not resent");
    assert!(
        matches!(error, TinyAgentsError::Provider(_)),
        "got {error:?}"
    );
    assert_eq!(model.efforts(), [Some(ReasoningEffort::None)]);
}

#[tokio::test]
async fn a_call_that_keeps_its_reasoning_is_sent_once() {
    let model = ReasoningRequired::refusing_with(REFUSAL);
    let run = harness(&model, ReasoningEffort::Low)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a call that asks for reasoning is answered");
    assert_eq!(run.text(), Some("answered".to_string()));
    assert_eq!(model.efforts(), [Some(ReasoningEffort::Low)]);
}

#[test]
fn only_a_reasoning_off_request_refused_for_it_counts() {
    let off = ModelRequest {
        reasoning: Some(ReasoningConfig::effort(ReasoningEffort::None)),
        ..ModelRequest::default()
    };
    let on = ModelRequest {
        reasoning: Some(ReasoningConfig::effort(ReasoningEffort::High)),
        ..ModelRequest::default()
    };
    // As a gateway relays it in a plain model error, and as a provider error.
    let relayed = TinyAgentsError::Model(format!("ephemeral-route returned HTTP 400: {REFUSAL}"));
    let provider = TinyAgentsError::Provider(Box::new(ProviderError {
        status: Some(400),
        message: REFUSAL.into(),
        ..ProviderError::default()
    }));
    assert!(reasoning_off_refused(&off, &relayed));
    assert!(reasoning_off_refused(&off, &provider));
    assert!(
        !reasoning_off_refused(&on, &relayed),
        "it asked for reasoning"
    );
    assert!(!reasoning_off_refused(&ModelRequest::default(), &relayed));
    let other = TinyAgentsError::Model("HTTP 400: reasoning effort `none` is not a level".into());
    assert!(!reasoning_off_refused(&off, &other));
    assert!(!reasoning_off_refused(&off, &TinyAgentsError::Cancelled));
}
