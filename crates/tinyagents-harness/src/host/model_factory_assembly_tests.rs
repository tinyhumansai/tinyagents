use super::*;

#[tokio::test]
async fn assembly_awaits_async_primary_and_summarization_factories() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();

    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .build()
        .await
        .expect("models should build after async factory work completes");

    assert_eq!(models.primary_route().model, "chat");
    assert_eq!(models.summarizer_route().model, "chat");
    assert_eq!(factory.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn model_build_requests_round_trip_through_serde() {
    let request = ModelBuildRequest {
        role: "summarization".into(),
        route: Some("hint:summarization".into()),
        purpose: ModelPurpose::Summarization,
        model_pin: Some("summary-model".into()),
        opts: BuildOpts {
            native_tools: false,
            temperature: Some(0.2),
            egress_tag: Some("summary".into()),
        },
    };
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: ModelBuildRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);
}

#[tokio::test]
async fn primary_override_stays_separate_from_default_workload_routes_and_summary() {
    let context = NonCloneContext {
        primary_provider_override: Some("provider-override".to_owned()),
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    let opts = BuildOpts {
        native_tools: true,
        temperature: Some(0.25),
        egress_tag: Some("turn".to_owned()),
    };

    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .primary_model_pin("primary-model")
        .primary_opts(opts.clone())
        .build()
        .await
        .expect("turn models should build");

    let requests = factory.requests.lock().unwrap();
    assert_eq!(requests[0].role, "chat");
    assert_eq!(requests[0].purpose, ModelPurpose::Primary);
    assert_eq!(requests[0].model_pin.as_deref(), Some("primary-model"));
    assert_eq!(requests[0].opts, opts);
    assert_eq!(
        requests[1].purpose,
        ModelPurpose::WorkloadTier("hint:chat".into())
    );
    assert_eq!(requests[1].role, "chat");
    assert_eq!(requests[1].route.as_deref(), Some("hint:chat"));
    assert_eq!(requests[1].model_pin.as_deref(), Some("hint:chat"));
    let default_tiers = requests
        .iter()
        .filter_map(|request| match &request.purpose {
            ModelPurpose::WorkloadTier(route) => Some((route.as_str(), request.role.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        default_tiers,
        [
            ("hint:chat", "chat"),
            ("hint:reasoning", "reasoning"),
            ("hint:agentic", "agentic"),
            ("hint:coding", "coding"),
            ("hint:burst", "burst"),
            ("hint:summarization", "summarization"),
            ("hint:vision", "vision"),
        ]
    );
    assert_eq!(requests.last().expect("summary").role, "chat");
    assert_eq!(
        requests.last().expect("summary").purpose,
        ModelPurpose::Summarization
    );
    assert_eq!(
        requests.last().expect("summary").model_pin.as_deref(),
        Some("primary-model")
    );
    assert_eq!(requests.last().expect("summary").opts, opts);
    assert_eq!(models.primary_route().provider, "provider-override");
    assert_eq!(models.summarizer_route().provider, "provider-override");
    assert_eq!(
        context.primary_provider_override.as_deref(),
        Some("provider-override")
    );
    assert_eq!(
        gate.requests.lock().unwrap().as_slice(),
        requests.as_slice()
    );
}

#[tokio::test]
async fn explicit_summary_role_pin_and_options_are_distinct() {
    let context = NonCloneContext {
        primary_provider_override: Some("primary-provider".to_owned()),
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    let summary_opts = BuildOpts {
        native_tools: false,
        temperature: Some(0.0),
        egress_tag: Some("summary".to_owned()),
    };
    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .primary_model_pin("chat-model")
        .summarization("summarization", Some("summary-model".to_owned()))
        .summarization_opts(summary_opts.clone())
        .build()
        .await
        .expect("summary model should build");

    let requests = factory.requests.lock().unwrap();
    let summary = requests.last().expect("summary request");
    assert_eq!(summary.role, "summarization");
    assert_eq!(summary.model_pin.as_deref(), Some("summary-model"));
    assert_eq!(summary.opts, summary_opts);
    assert_eq!(models.summarizer_route().model, "summary-model");
}

#[tokio::test]
async fn explicit_summary_role_does_not_inherit_the_primary_pin() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .without_workload_tiers()
        .primary_model_pin("pinned-chat")
        .summarization("summary-role", None)
        .build()
        .await
        .expect("default summary role route should build");

    let summary = factory.requests.lock().unwrap().last().unwrap().clone();
    assert_eq!(summary.role, "summary-role");
    assert_eq!(summary.model_pin, None);
}

#[tokio::test]
async fn custom_workload_requests_preserve_order_and_skip_failed_or_gated_routes() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context).failing_route(Some("hint:broken"));
    let gate = RecordingGate::default().denying_route(Some("hint:blocked"));
    let requests = vec![
        workload("planner", "hint:first", None),
        workload("coder", "hint:blocked", None),
        workload("fallback", "hint:broken", None),
        workload("scribe", "hint:last", None),
    ];

    let models = TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .workload_requests(requests)
        .build()
        .await
        .expect("failed tiers should not fail the set");

    let built_routes = factory
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter_map(|request| request.route.clone())
        .collect::<Vec<_>>();
    assert_eq!(built_routes, ["hint:first", "hint:broken", "hint:last"]);
    assert!(models.workload_error_slot("hint:first").is_some());
    assert!(models.workload_error_slot("hint:blocked").is_none());
    assert!(models.workload_error_slot("hint:broken").is_none());
    assert!(models.workload_error_slot("hint:last").is_some());
}

#[tokio::test]
async fn denied_primary_never_calls_factory_and_failed_primary_is_fatal() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default().denying_role(Some("chat"));
    assert!(
        TurnModelSet::<()>::builder(&factory, &gate, "chat")
            .build()
            .await
            .is_err()
    );
    assert!(factory.requests.lock().unwrap().is_empty());

    let factory = RecordingFactory::new(&context).failing_primary();
    let gate = RecordingGate::default();
    assert!(
        TurnModelSet::<()>::builder(&factory, &gate, "chat")
            .build()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn denied_or_failed_summary_is_fatal_after_primary_build() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let gate = RecordingGate::default().denying_purpose(Some(ModelPurpose::Summarization));
    let factory = RecordingFactory::new(&context);
    assert!(
        TurnModelSet::<()>::builder(&factory, &gate, "chat")
            .without_workload_tiers()
            .build()
            .await
            .is_err()
    );
    assert_eq!(factory.requests.lock().unwrap().len(), 1);

    let factory = RecordingFactory::new(&context).failing_summary();
    let gate = RecordingGate::default();
    assert!(
        TurnModelSet::<()>::builder(&factory, &gate, "chat")
            .without_workload_tiers()
            .build()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn different_workload_route_with_same_concrete_primary_pin_is_kept() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();

    TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .primary_model_pin("same-model")
        .workload_requests(vec![
            workload("other", "hint:other", Some("other-model")),
            workload("same", "hint:same", Some("same-model")),
        ])
        .build()
        .await
        .expect("turn models should build");

    let routes = factory
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter_map(|request| request.route.clone())
        .collect::<Vec<_>>();
    assert_eq!(routes, ["hint:other", "hint:same"]);
}

#[tokio::test]
async fn shared_workload_options_forward_to_every_default_tier_with_route_override() {
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    let common = BuildOpts {
        native_tools: true,
        temperature: Some(0.4),
        egress_tag: Some("workload".to_owned()),
    };
    let override_opts = BuildOpts {
        native_tools: false,
        temperature: Some(0.0),
        egress_tag: Some("vision".to_owned()),
    };

    TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .workload_opts(common.clone())
        .workload_tier_opts("hint:vision", override_opts.clone())
        .build()
        .await
        .expect("turn models should build");

    let tier_opts = factory
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|request| matches!(request.purpose, ModelPurpose::WorkloadTier(_)))
        .map(|request| {
            (
                request.route.clone().expect("workload route"),
                request.opts.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(tier_opts.len(), 7);
    for (route, opts) in tier_opts {
        if route == "hint:vision" {
            assert_eq!(opts, override_opts);
        } else {
            assert_eq!(opts, common);
        }
    }
}

#[tokio::test]
async fn default_workload_matching_primary_alias_is_skipped_but_other_alias_with_same_pin_is_kept()
{
    let context = NonCloneContext {
        primary_provider_override: None,
    };
    let factory = RecordingFactory::new(&context);
    let gate = RecordingGate::default();
    TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .primary_model_pin("hint:chat")
        .build()
        .await
        .expect("models should build");
    assert!(
        !factory
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|request| { request.purpose == ModelPurpose::WorkloadTier("hint:chat".into()) })
    );

    let factory = RecordingFactory::new(&context);
    TurnModelSet::<()>::builder(&factory, &gate, "chat")
        .primary_model_pin("shared-model")
        .workload_requests(vec![workload(
            "alternate",
            "hint:alternate",
            Some("shared-model"),
        )])
        .build()
        .await
        .expect("models should build");
    assert!(
        factory
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|request| { request.route.as_deref() == Some("hint:alternate") })
    );
}
