//! Unit tests for the run-context module.

use super::*;
use crate::events::{AgentEvent, RecordingListener};
use crate::steering::SteeringHandle;
use std::sync::Arc;
use tinytools::WorkspaceDescriptor;

#[test]
fn run_config_defaults_are_sensible() {
    let config = RunConfig::new("run-1");
    assert_eq!(config.run_id.as_str(), "run-1");
    assert!(config.thread_id.is_none());
    assert!(config.tags.is_empty());
    assert_eq!(config.metadata, serde_json::Value::Null);
    assert!(config.timeout_ms.is_none());
    // Unset by default, so a harness `RunPolicy` remains free to raise the cap
    // (see `RunConfig::max_model_calls`); the effective value is the crate
    // default.
    assert_eq!(config.max_model_calls, None);
    assert_eq!(config.max_tool_calls, None);
    assert_eq!(config.effective_max_model_calls(), 25);
    assert_eq!(config.effective_max_tool_calls(), 50);
}

#[test]
fn run_config_builders_compose() {
    let config = RunConfig::new("run-2")
        .with_thread("thread-9")
        .with_tag("a")
        .with_tag("b")
        .with_metadata(serde_json::json!({"k": "v"}))
        .with_timeout_ms(1234)
        .with_max_model_calls(3)
        .with_max_tool_calls(4);

    assert_eq!(config.thread_id.as_ref().unwrap().as_str(), "thread-9");
    assert_eq!(config.tags, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(config.metadata["k"], serde_json::json!("v"));
    assert_eq!(config.timeout_ms, Some(1234));
    assert_eq!(config.max_model_calls, Some(3));
    assert_eq!(config.max_tool_calls, Some(4));
}

#[test]
fn run_config_round_trips_through_json() {
    let config = RunConfig::new("run-3").with_thread("t1").with_tag("x");
    let json = serde_json::to_string(&config).unwrap();
    let back: RunConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.run_id.as_str(), "run-3");
    assert_eq!(back.thread_id.unwrap().as_str(), "t1");
    assert_eq!(back.tags, vec!["x".to_string()]);
}

#[test]
fn run_config_reads_legacy_depth_fields_into_lineage() {
    let config: RunConfig = serde_json::from_value(serde_json::json!({
        "run_id": "legacy",
        "depth": 3,
        "max_depth": 5
    }))
    .unwrap();
    assert_eq!(config.lineage.root_run_id.as_str(), "legacy");
    assert_eq!(config.depth(), 3);
    assert_eq!(config.max_depth(), 5);
}

#[test]
fn run_config_discovery_overrides_default_to_empty_and_round_trip() {
    use crate::tool::discover::DiscoveryOverrides;

    // An old persisted config has no such field and keeps today's behaviour,
    // and a default config does not write one.
    let old: RunConfig = serde_json::from_value(serde_json::json!({"run_id": "old"})).unwrap();
    assert!(old.discovery.is_empty());
    assert!(
        serde_json::to_value(RunConfig::new("fresh"))
            .unwrap()
            .get("discovery")
            .is_none()
    );

    let config = RunConfig::new("set").with_discovery_overrides(DiscoveryOverrides {
        promote_found_tools: Some(false),
        collapse_search_results: Some(true),
        repair_tool_call_wrapper: Some(false),
        ..DiscoveryOverrides::default()
    });
    let back: RunConfig = serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(back.discovery, config.discovery);
    assert_eq!(back.discovery.promote_found_tools, Some(false));
    assert_eq!(back.discovery.collapse_search_results, Some(true));
    assert_eq!(back.discovery.repair_tool_call_wrapper, Some(false));
}

#[test]
fn run_config_result_expiry_overrides_default_to_empty_and_round_trip() {
    use crate::result_expiry::ExpiryOverrides;

    // An old persisted config has no such field and keeps today's behaviour,
    // and a default config does not write one.
    let old: RunConfig = serde_json::from_value(serde_json::json!({"run_id": "old"})).unwrap();
    assert!(old.result_expiry.is_empty());
    assert!(
        serde_json::to_value(RunConfig::new("fresh"))
            .unwrap()
            .get("result_expiry")
            .is_none()
    );

    let config = RunConfig::new("set").with_result_expiry_overrides(ExpiryOverrides {
        enabled: Some(true),
        max_age_turns: Some(2),
        exempt_tools: Some(vec!["tool_search".to_string()]),
    });
    let back: RunConfig = serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(back.result_expiry, config.result_expiry);
    assert_eq!(back.result_expiry.max_age_turns, Some(2));
}

#[test]
fn run_config_replay_reasoning_defaults_to_unset_and_round_trips() {
    use crate::reasoning_replay::ReasoningReplay;

    // An old persisted config has no such field and keeps today's behaviour,
    // and a default config does not write one.
    let old: RunConfig = serde_json::from_value(serde_json::json!({"run_id": "old"})).unwrap();
    assert_eq!(old.replay_reasoning, None);
    assert!(
        serde_json::to_value(RunConfig::new("fresh"))
            .unwrap()
            .get("replay_reasoning")
            .is_none()
    );

    let config = RunConfig::new("set").with_replay_reasoning(ReasoningReplay::CurrentTurn);
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(json["replay_reasoning"], "current_turn");
    let back: RunConfig = serde_json::from_value(json).unwrap();
    assert_eq!(back.replay_reasoning, Some(ReasoningReplay::CurrentTurn));
}

#[test]
fn context_exposes_run_and_thread_ids() {
    let config = RunConfig::new("run-4").with_thread("thread-4");
    let ctx: RunContext = RunContext::new(config, ());
    assert_eq!(ctx.run_id().as_str(), "run-4");
    assert_eq!(ctx.thread_id().unwrap().as_str(), "thread-4");
}

#[test]
fn context_carries_generic_user_data() {
    let config = RunConfig::new("run-5");
    let ctx: RunContext<u32> = RunContext::new(config, 42);
    assert_eq!(ctx.data, 42);
}

#[test]
fn context_records_calls_and_enforces_limits() {
    let config = RunConfig::new("run-6")
        .with_max_model_calls(1)
        .with_max_tool_calls(1);
    let mut ctx: RunContext = RunContext::new(config, ());

    ctx.record_model_call().expect("first model call ok");
    assert_eq!(ctx.limits.model_calls(), 1);
    assert!(ctx.record_model_call().is_err(), "second exceeds cap");

    ctx.record_tool_call().expect("first tool call ok");
    assert_eq!(ctx.limits.tool_calls(), 1);
    assert!(ctx.record_tool_call().is_err(), "second exceeds cap");
}

#[test]
fn context_emit_delegates_to_event_sink() {
    let config = RunConfig::new("run-7");
    let ctx: RunContext = RunContext::new(config, ());

    let recorder = Arc::new(RecordingListener::new());
    ctx.events.subscribe(recorder.clone());

    let record = ctx.emit(AgentEvent::RunStarted {
        run_id: ctx.run_id().clone(),
        thread_id: None,
    });
    assert_eq!(record.offset, 0);
    assert_eq!(recorder.events().len(), 1);
}

#[test]
fn context_check_deadline_passes_without_timeout() {
    let config = RunConfig::new("run-8");
    let mut ctx: RunContext = RunContext::new(config, ());
    assert!(ctx.check_deadline().is_ok());
}

#[test]
fn with_events_shares_sink() {
    let shared = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    shared.subscribe(recorder.clone());

    let ctx: RunContext = RunContext::new(RunConfig::new("run-9"), ()).with_events(shared.clone());
    ctx.emit(AgentEvent::StateUpdate);
    assert_eq!(recorder.events().len(), 1);
    assert_eq!(shared.len(), 1);
}

#[test]
fn request_control_keeps_highest_precedence() {
    let ctx: RunContext = RunContext::new(RunConfig::new("run-ctrl"), ());

    // A stronger Interrupt is not downgraded by a later, weaker StopWithFinal.
    ctx.request_control(MiddlewareControl::Interrupt {
        node: "review".into(),
        message: "hold".into(),
    });
    ctx.request_control(MiddlewareControl::StopWithFinal("stop".into()));
    assert!(matches!(
        ctx.take_control(),
        Some(MiddlewareControl::Interrupt { .. })
    ));
    assert!(
        ctx.take_control().is_none(),
        "take_control clears the request"
    );

    // A stronger request does replace a weaker pending one.
    ctx.request_control(MiddlewareControl::StopWithFinal("stop".into()));
    ctx.request_control(MiddlewareControl::Interrupt {
        node: "review".into(),
        message: "hold".into(),
    });
    assert!(matches!(
        ctx.take_control(),
        Some(MiddlewareControl::Interrupt { .. })
    ));
}

#[test]
fn checked_child_depth_is_the_shared_depth_guard() {
    use crate::error::TinyAgentsError;

    // Below the cap: returns parent_depth + 1.
    assert_eq!(RunConfig::checked_child_depth(0, 8).unwrap(), 1);
    assert_eq!(RunConfig::checked_child_depth(7, 8).unwrap(), 8);

    // At the boundary (child would be max_depth + 1): fail closed with the cap.
    match RunConfig::checked_child_depth(8, 8) {
        Err(TinyAgentsError::SubAgentDepth(cap)) => assert_eq!(cap, 8),
        other => panic!("expected SubAgentDepth(8), got {other:?}"),
    }
}

#[test]
fn child_carries_explicit_lineage_and_rejects_the_depth_cap() {
    let parent: RunContext<()> = RunContext::new(
        RunConfig::new("root")
            .with_thread("thread")
            .with_max_depth(2)
            .with_max_turn_output_tokens(123),
        (),
    );
    let child = parent
        .child_with_data(RunConfig::new("child"), "child-data")
        .unwrap();
    let grandchild = child
        .child_with_data(RunConfig::new("grandchild"), ())
        .unwrap();

    assert_eq!(parent.lineage().root_run_id.as_str(), "root");
    assert_eq!(parent.lineage().parent_run_id, None);
    assert_eq!(child.lineage().root_run_id.as_str(), "root");
    assert_eq!(
        child.lineage().parent_run_id.as_ref().unwrap().as_str(),
        "root"
    );
    assert_eq!(child.depth(), 1);
    assert_eq!(
        grandchild
            .lineage()
            .parent_run_id
            .as_ref()
            .unwrap()
            .as_str(),
        "child"
    );
    assert_eq!(grandchild.depth(), 2);
    assert_eq!(grandchild.max_depth(), 2);
    assert_eq!(grandchild.thread_id().unwrap().as_str(), "thread");
    assert_eq!(grandchild.config.max_turn_output_tokens, Some(123));
    assert!(matches!(
        grandchild.child_with_data(RunConfig::new("too-deep"), ()),
        Err(crate::TinyAgentsError::SubAgentDepth(2))
    ));
}

#[test]
fn child_keeps_its_stricter_recursion_cap() {
    let parent: RunContext<()> = RunContext::new(RunConfig::new("parent").with_max_depth(4), ());
    let child = parent
        .child(RunConfig::new("child").with_max_depth(1), ())
        .expect("the first child is within both caps");

    assert_eq!(child.max_depth(), 1);
    assert!(matches!(
        child.child(RunConfig::new("grandchild"), ()),
        Err(crate::TinyAgentsError::SubAgentDepth(1))
    ));
}

#[test]
fn non_root_child_rejects_a_stricter_cap_already_below_its_next_depth() {
    let root: RunContext<()> = RunContext::new(RunConfig::new("root").with_max_depth(4), ());
    let parent = root
        .child(RunConfig::new("parent"), ())
        .expect("parent is within the root cap");

    assert!(matches!(
        parent.child(RunConfig::new("child").with_max_depth(1), ()),
        Err(crate::TinyAgentsError::SubAgentDepth(1))
    ));
}

#[test]
fn child_accepts_its_own_tags_timeout_and_call_caps() {
    let parent: RunContext<()> =
        RunContext::new(RunConfig::new("parent").with_thread("thread"), ());
    let child = parent
        .child(
            RunConfig::new("child")
                .with_tag("delegated")
                .with_timeout_ms(40)
                .with_max_model_calls(3)
                .with_max_tool_calls(4),
            (),
        )
        .unwrap();
    assert_eq!(child.config.tags, vec!["delegated"]);
    assert_eq!(child.config.timeout_ms, Some(40));
    assert_eq!(child.config.max_model_calls, Some(3));
    assert_eq!(child.config.max_tool_calls, Some(4));
    assert_eq!(child.thread_id().unwrap().as_str(), "thread");
}

#[test]
fn child_inherits_recursive_capabilities_but_not_mutable_run_state() {
    let cancellation = crate::CancellationToken::new();
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let steering = SteeringHandle::allow_all();
    let workspace = WorkspaceDescriptor::new("/workspace/child-contract");
    let mut parent: RunContext<()> = RunContext::new(
        RunConfig::new("parent")
            .with_metadata(serde_json::json!({"keep": true, "replace": "parent"}))
            .with_max_model_calls(2)
            .with_max_tool_calls(2),
        (),
    )
    .with_cancellation(cancellation.clone())
    .with_events(events.clone())
    .with_steering(steering.clone())
    .with_workspace(workspace.clone());
    parent.streaming = true;
    parent.record_model_call().unwrap();
    parent.request_control(MiddlewareControl::StopWithFinal("parent-only".into()));

    let child = parent
        .child(
            RunConfig::new("child")
                .with_metadata(serde_json::json!({"replace": "child", "new": 1})),
            (),
        )
        .unwrap();

    assert_eq!(
        child.config.metadata,
        serde_json::json!({"keep": true, "replace": "child", "new": 1})
    );
    assert_eq!(child.limits.model_calls(), 0, "limits begin independently");
    assert!(
        child.take_control().is_none(),
        "control slots are not shared"
    );
    assert_ne!(parent.instance_id(), child.instance_id());
    assert!(child.streaming);
    assert_eq!(child.workspace.as_ref(), Some(&workspace));
    assert!(child.steering.is_some());
    child.emit(AgentEvent::StateUpdate);
    assert_eq!(recorder.events().len(), 1, "child emits on the parent sink");
    cancellation.cancel();
    assert!(child.cancellation.is_cancelled());
}

#[test]
fn child_cancellation_is_independent_but_parent_cancel_cascades() {
    let parent: RunContext<()> = RunContext::new(RunConfig::new("root"), ());
    let first = parent.child(RunConfig::new("first"), ()).unwrap();
    let second = parent.child(RunConfig::new("second"), ()).unwrap();

    first.cancellation.cancel();
    assert!(first.cancellation.is_cancelled());
    assert!(!parent.cancellation.is_cancelled(), "no upward propagation");
    assert!(!second.cancellation.is_cancelled(), "siblings unaffected");

    parent.cancellation.cancel();
    assert!(second.cancellation.is_cancelled(), "parent cancel cascades");
}

#[test]
fn sibling_children_are_isolated_while_sharing_tree_signals() {
    let events = EventSink::new();
    let parent: RunContext<()> = RunContext::new(RunConfig::new("root"), ()).with_events(events);
    let mut first = parent.child(RunConfig::new("first"), ()).unwrap();
    let second = parent.child(RunConfig::new("second"), ()).unwrap();

    first.record_tool_call().unwrap();
    first.request_control(MiddlewareControl::StopWithFinal("first".into()));
    assert_eq!(second.limits.tool_calls(), 0);
    assert!(second.take_control().is_none());
    assert_ne!(first.instance_id(), second.instance_id());
    assert_eq!(first.lineage().root_run_id, second.lineage().root_run_id);
    assert_eq!(
        first.lineage().parent_run_id,
        second.lineage().parent_run_id
    );
    assert_ne!(first.run_id(), second.run_id());
}

#[tokio::test]
async fn child_store_values_are_shared_but_registry_membership_is_snapshotted() {
    use crate::store::InMemoryStore;

    let mut parent: RunContext<()> = RunContext::new(RunConfig::new("parent"), ());
    let child = parent.child(RunConfig::new("child"), ()).unwrap();
    parent
        .stores
        .register("added-later", Arc::new(InMemoryStore::new()));
    assert!(child.stores.get("added-later").is_none());

    parent
        .stores
        .default_store()
        .put("scope", "key", serde_json::json!("shared"))
        .await
        .unwrap();
    assert_eq!(
        child
            .stores
            .default_store()
            .get("scope", "key")
            .await
            .unwrap(),
        Some(serde_json::json!("shared"))
    );
}

#[test]
fn context_statistics_preserve_tool_request_result_pairing_and_image_counts() {
    use tinyinference_llm::message::{ContentBlock, ImageRef, Message, UserMessage};
    use tinyinference_llm::tool::ToolCall;

    let messages = vec![
        Message::Assistant(tinyinference_llm::message::AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("call it".into())],
            tool_calls: vec![ToolCall::new("call-1", "lookup", serde_json::json!({}))],
            usage: None,
            origin: None,
        }),
        Message::Tool(tinyinference_llm::message::ToolMessage {
            tool_call_id: "call-1".into(),
            content: vec![ContentBlock::Text("answer".into())],
            trusted_verbatim: false,
            artifact: None,
        }),
        Message::User(UserMessage {
            content: vec![ContentBlock::Image(ImageRef {
                url: "data:image/png;base64,AA==".into(),
                mime_type: Some("image/png".into()),
            })],
        }),
    ];

    assert_eq!(
        context_statistics(&messages),
        ContextStatistics {
            messages: 3,
            text_chars: 13,
            images: 1,
            media: 0,
            tool_calls: 1,
            tool_results: 1,
            paired_tool_results: 1,
        }
    );
}

#[test]
fn context_statistics_counts_audio_video_and_document_blocks_as_media() {
    use tinyinference_llm::message::{ContentBlock, MediaRef, Message, UserMessage};

    let messages = vec![Message::User(UserMessage {
        content: vec![
            ContentBlock::Audio(MediaRef::url("https://example.com/a.wav")),
            ContentBlock::Video(MediaRef::base64("AAAA", "video/mp4")),
            ContentBlock::Document(MediaRef::path("/tmp/doc.pdf")),
        ],
    })];

    let stats = context_statistics(&messages);
    assert_eq!(stats.media, 3);
    assert_eq!(stats.images, 0);
}

#[test]
fn token_estimation_uses_the_callers_tokenizer() {
    use tinyinference_llm::message::Message;
    let messages = vec![Message::system("one two"), Message::user("three")];
    assert_eq!(
        estimate_context_tokens(&messages, |text| text.split_whitespace().count()),
        3
    );
}

#[test]
fn token_estimation_includes_json_user_blocks() {
    use tinyinference_llm::message::{ContentBlock, Message, UserMessage};
    let messages = vec![Message::User(UserMessage {
        content: vec![ContentBlock::Json(
            serde_json::json!({"payload": "one two three"}),
        )],
    })];

    assert_eq!(
        estimate_context_tokens(&messages, |text| text.split_whitespace().count()),
        3
    );
}

#[test]
fn token_estimation_includes_structured_blocks_for_every_role() {
    use tinyinference_llm::message::{
        AssistantMessage, ContentBlock, Message, SystemMessage, ToolMessage, UserMessage,
    };
    let json = serde_json::json!({"payload": "one two"});
    let messages = vec![
        Message::System(SystemMessage {
            content: vec![ContentBlock::ProviderExtension(json.clone())],
            ..Default::default()
        }),
        Message::User(UserMessage {
            content: vec![ContentBlock::Json(json.clone())],
        }),
        Message::Assistant(AssistantMessage {
            id: None,
            content: vec![ContentBlock::ProviderExtension(json.clone())],
            tool_calls: vec![],
            usage: None,
            origin: None,
        }),
        Message::Tool(ToolMessage {
            tool_call_id: "call".into(),
            content: vec![ContentBlock::Json(json)],
            trusted_verbatim: false,
            artifact: None,
        }),
    ];
    assert_eq!(
        estimate_context_tokens(&messages, |text| text.split_whitespace().count()),
        8
    );
}

#[test]
fn token_estimation_includes_assistant_tool_names_and_arguments() {
    use tinyinference_llm::message::{AssistantMessage, Message};
    use tinyinference_llm::tool::ToolCall;

    let messages = vec![Message::Assistant(AssistantMessage {
        id: None,
        content: vec![],
        tool_calls: vec![ToolCall::new(
            "call-1",
            "search_docs",
            serde_json::json!({"query": "one two three"}),
        )],
        usage: None,
        origin: None,
    })];

    let rendered = std::cell::RefCell::new(String::new());
    assert_eq!(
        estimate_context_tokens(&messages, |text| {
            rendered.replace(text.to_string());
            1
        }),
        1
    );
    let rendered = rendered.into_inner();
    assert!(rendered.contains("search_docs"));
    assert!(rendered.contains("one two three"));
}

// ── RunContext::bounded (R-1) ───────────────────────────────────────────────

#[tokio::test]
async fn bounded_returns_the_futures_ok_value_with_no_deadline() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-bounded-ok"), ());
    let result: Result<u32> = ctx
        .bounded(None, async { Ok(42) }, || "unused".to_string())
        .await;
    assert_eq!(result.unwrap(), 42);
}

#[tokio::test]
async fn bounded_passes_through_the_futures_own_error() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-bounded-err"), ());
    let result: Result<u32> = ctx
        .bounded(
            None,
            async { Err(crate::error::TinyAgentsError::Model("boom".to_string())) },
            || "unused".to_string(),
        )
        .await;
    match result {
        Err(crate::error::TinyAgentsError::Model(message)) => assert_eq!(message, "boom"),
        other => panic!("expected a passthrough Model error, got {other:?}"),
    }
}

#[tokio::test]
async fn bounded_fires_the_timeout_message_only_when_the_deadline_elapses() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-bounded-timeout"), ());
    let mut message_built = false;
    let result: Result<u32> = ctx
        .bounded(
            Some(std::time::Duration::from_millis(5)),
            async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(1)
            },
            || {
                message_built = true;
                "call-specific timeout message".to_string()
            },
        )
        .await;
    assert!(message_built);
    match result {
        Err(crate::error::TinyAgentsError::Timeout(message)) => {
            assert_eq!(message, "call-specific timeout message");
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn bounded_does_not_build_the_timeout_message_on_the_success_path() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-bounded-no-timeout"), ());
    let result: Result<u32> = ctx
        .bounded(
            Some(std::time::Duration::from_secs(60)),
            async { Ok(7) },
            || panic!("timeout_message must not be called when the future finishes first"),
        )
        .await;
    assert_eq!(result.unwrap(), 7);
}

#[tokio::test]
async fn bounded_returns_cancelled_when_the_run_is_cancelled_before_the_future_resolves() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-bounded-cancel"), ());
    let cancellation = ctx.cancellation.clone();
    cancellation.cancel();
    let result: Result<u32> = ctx
        .bounded(
            None,
            async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(1)
            },
            || "unused".to_string(),
        )
        .await;
    assert!(matches!(
        result,
        Err(crate::error::TinyAgentsError::Cancelled)
    ));
}

#[test]
fn a_cleared_limit_kind_is_not_attributed_to_a_later_error() {
    use crate::events::{AgentEvent, LimitKind};
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("run-limit-cache"), ());
    ctx.emit(AgentEvent::LimitReached {
        kind: LimitKind::ToolCalls,
    });
    ctx.clear_last_limit();
    assert_eq!(ctx.take_last_limit(), None);
    ctx.emit(AgentEvent::LimitReached {
        kind: LimitKind::ModelCalls,
    });
    assert_eq!(ctx.take_last_limit(), Some(LimitKind::ModelCalls));
}

/// A transport round-trip preserves depth and refuses a fourth child.
#[tokio::test]
async fn serialized_run_depth_keeps_the_child_limit_across_a_transport_hop() {
    let config = RunConfig::new("parent").with_depth(3).with_max_depth(3);
    let depth = RunDepth::from(&config);
    let wire = serde_json::to_string(&depth).unwrap();
    let imported: RunDepth = serde_json::from_str(&wire).unwrap();
    assert_eq!(current_run_depth(), None);
    with_run_depth(imported, async {
        let current = current_run_depth().unwrap();
        assert_eq!(current.depth, 3);
        assert!(matches!(
            current.checked_child(),
            Err(crate::TinyAgentsError::SubAgentDepth(3))
        ));
    })
    .await;
    assert_eq!(current_run_depth(), None);
}

/// Nested task scopes restore their parent without leaking into spawned tasks.
#[tokio::test]
async fn nested_run_depth_scopes_restore_the_parent_and_do_not_leak_to_tasks() {
    let root = RunDepth {
        depth: 0,
        max_depth: 3,
    };
    with_run_depth(root, async {
        let child = current_run_depth().unwrap().checked_child().unwrap();
        with_run_depth(child, async {
            assert_eq!(current_run_depth().unwrap().depth, 1);
            assert_eq!(
                tokio::spawn(async { current_run_depth() }).await.unwrap(),
                None
            );
        })
        .await;
        assert_eq!(current_run_depth(), Some(root));
    })
    .await;
    assert_eq!(current_run_depth(), None);
}

/// Untrusted maximum integer depth returns a typed refusal rather than panicking.
#[test]
fn checked_child_depth_rejects_overflow_in_serialized_transport_facts() {
    assert!(matches!(
        RunConfig::checked_child_depth(usize::MAX, usize::MAX),
        Err(crate::TinyAgentsError::SubAgentDepth(usize::MAX))
    ));
}
