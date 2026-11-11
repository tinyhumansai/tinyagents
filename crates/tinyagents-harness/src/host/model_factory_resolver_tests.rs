use super::*;

#[tokio::test]
async fn resolver_keeps_lead_pin_hint_and_fallback_precedence() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .workload_requests(vec![
            workload("pinned-role", "pinned-model", Some("pinned-model")),
            workload("hint-role", "hint:fast", None),
        ])
        .build()
        .await
        .expect("models should build")
        .map_models(|purpose, model| {
            Arc::new(ProfileOverrideModel::new(
                model,
                ModelProfile {
                    display_name: Some(format!("{purpose:?}")),
                    ..Default::default()
                },
            )) as Arc<dyn ChatModel<()>>
        });

    let resolver = models.resolver();
    for (request, expected) in [
        (ModelResolveRequest::new("lead").as_team_lead(), "Primary"),
        (
            ModelResolveRequest::new("pinned")
                .with_model_pin("pinned-model")
                .with_model_hint("fast"),
            "WorkloadTier(\"pinned-model\")",
        ),
        (
            ModelResolveRequest::new("hinted").with_model_hint("fast"),
            "WorkloadTier(\"hint:fast\")",
        ),
        (ModelResolveRequest::new("fallback"), "Primary"),
    ] {
        let model = resolver.resolve(&request).await.expect("route resolves");
        assert_eq!(profile_label(&model), expected);
    }
}

#[tokio::test]
async fn summarizer_uses_a_distinct_model_and_error_slot() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .build()
        .await
        .expect("models should build");

    let lead = models
        .resolver()
        .resolve(&ModelResolveRequest::new("lead").as_team_lead())
        .await
        .expect("primary resolver model");
    let summary = models.summarizer();
    assert!(!Arc::ptr_eq(&lead, &summary));
    assert!(!Arc::ptr_eq(
        &models.primary_error_slot(),
        &models.summarizer_error_slot()
    ));
}

#[tokio::test]
async fn owned_resolver_outlives_borrowed_factory_gate_context_and_model_set() {
    let resolver: Arc<dyn ModelResolver<()>> = {
        let context = NonCloneContext {
            primary_provider_override: None,
        };
        let factory = RecordingFactory::new(&context);
        let gate = RecordingGate::default();
        let factory_port: &dyn ModelFactory<()> = &factory;
        let gate_port: &dyn ModelBuildGate = &gate;
        let models = TurnModelSet::<()>::build(factory_port, gate_port, "chat")
            .await
            .expect("models should build");
        Arc::new(models.resolver())
    };

    let model = resolver
        .resolve(&ModelResolveRequest::new("lead").as_team_lead())
        .await
        .expect("owned resolver remains usable");
    let response = model
        .invoke(&(), ModelRequest::default())
        .await
        .expect("model remains usable");
    assert_eq!(response.text(), "provider called");
}

#[tokio::test]
async fn host_model_mapping_wraps_each_model_once_and_keeps_route_and_error_slots() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context).failing_primary_call();
    let gate = RecordingGate::default();
    let mut mapped = Vec::new();
    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .build()
        .await
        .expect("turn models should build")
        .map_models(|purpose, model| {
            mapped.push(purpose.clone());
            Arc::new(ProfileOverrideModel::new(
                model,
                ModelProfile {
                    display_name: Some(format!("{purpose:?}")),
                    ..Default::default()
                },
            )) as Arc<dyn ChatModel<()>>
        });

    assert_eq!(
        mapped,
        [
            ModelPurpose::Primary,
            ModelPurpose::WorkloadTier("hint:chat".into()),
            ModelPurpose::WorkloadTier("hint:reasoning".into()),
            ModelPurpose::WorkloadTier("hint:agentic".into()),
            ModelPurpose::WorkloadTier("hint:coding".into()),
            ModelPurpose::WorkloadTier("hint:burst".into()),
            ModelPurpose::WorkloadTier("hint:summarization".into()),
            ModelPurpose::WorkloadTier("hint:vision".into()),
            ModelPurpose::Summarization,
        ]
    );

    let registrations = models.registrations("lead");
    assert_eq!(
        registrations
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        [
            "lead",
            "hint:chat",
            "hint:reasoning",
            "hint:agentic",
            "hint:coding",
            "hint:burst",
            "hint:summarization",
            "hint:vision",
        ]
    );
    assert_eq!(profile_label(&registrations[0].1), "Primary");
    assert_eq!(
        profile_label(&registrations[1].1),
        "WorkloadTier(\"hint:chat\")"
    );

    let resolver = models.resolver();
    let lead = resolver
        .resolve(&ModelResolveRequest::new("lead").as_team_lead())
        .await
        .expect("primary resolves");
    assert_eq!(profile_label(&lead), "Primary");
    assert!(lead.invoke(&(), ModelRequest::default()).await.is_err());
    assert!(models.primary_error_slot().lock().unwrap().is_some());

    for (request, expected) in [
        (
            ModelResolveRequest::new("pinned").with_model_pin("hint:chat"),
            "WorkloadTier(\"hint:chat\")",
        ),
        (
            ModelResolveRequest::new("hinted").with_model_hint("coding"),
            "WorkloadTier(\"hint:coding\")",
        ),
        (ModelResolveRequest::new("fallback"), "Primary"),
    ] {
        let model = resolver.resolve(&request).await.expect("route resolves");
        assert_eq!(profile_label(&model), expected);
        if expected != "Primary" {
            model
                .invoke(&(), ModelRequest::default())
                .await
                .expect("tier wrapper returns normally");
        }
    }
    let summary = models.summarizer();
    assert_eq!(profile_label(&summary), "Summarization");
    summary
        .invoke(&(), ModelRequest::default())
        .await
        .expect("summary wrapper returns normally");
    assert!(models.summarizer_error_slot().lock().unwrap().is_none());
}

#[tokio::test]
async fn rejecting_host_wrapper_is_inside_primary_tier_and_summary_error_slots() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let provider_calls = Arc::clone(&factory.provider_calls);
    let gate = RecordingGate::default();
    let budget = crate::cost::ModelBudget::new(
        Budget::new(SpendLimits {
            tokens: Some(0),
            cost_micros: None,
        }),
        CallBudget {
            input_tokens: 1,
            output_tokens: 1,
            cost_micros: 1,
        },
    );
    let models = TurnModelSet::<()>::build(&factory, &gate, "chat")
        .await
        .expect("models should build")
        .map_models(|_, model| budget.wrap_model(model));

    let resolver = models.resolver();
    let primary = resolver
        .resolve(&ModelResolveRequest::new("lead").as_team_lead())
        .await
        .expect("primary resolves");
    assert!(primary.invoke(&(), ModelRequest::default()).await.is_err());
    assert!(models.primary_error_slot().lock().unwrap().is_some());
    assert!(
        models
            .workload_error_slot("hint:chat")
            .expect("tier slot")
            .lock()
            .unwrap()
            .is_none()
    );
    assert!(models.summarizer_error_slot().lock().unwrap().is_none());

    let tier = resolver
        .resolve(&ModelResolveRequest::new("worker").with_model_hint("chat"))
        .await
        .expect("tier resolves");
    assert!(tier.invoke(&(), ModelRequest::default()).await.is_err());
    assert!(
        models
            .workload_error_slot("hint:chat")
            .expect("tier slot")
            .lock()
            .unwrap()
            .is_some()
    );
    assert!(models.primary_error_slot().lock().unwrap().is_some());
    assert!(models.summarizer_error_slot().lock().unwrap().is_none());

    let summary = models.summarizer();
    assert!(summary.invoke(&(), ModelRequest::default()).await.is_err());
    assert!(models.summarizer_error_slot().lock().unwrap().is_some());
    assert!(models.primary_error_slot().lock().unwrap().is_some());
    assert!(
        models
            .workload_error_slot("hint:chat")
            .expect("tier slot")
            .lock()
            .unwrap()
            .is_some()
    );
    assert_eq!(provider_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(budget.ledger.snapshot().reserved, Default::default());
    assert_eq!(budget.ledger.snapshot().spent, Default::default());
}

fn profile_label(model: &Arc<dyn ChatModel<()>>) -> &str {
    model
        .profile()
        .and_then(|profile| profile.display_name.as_deref())
        .expect("host decorator profile")
}
