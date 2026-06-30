use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use tinyagents::harness::events::{
    AgentEvent, EventJournal, EventListener, EventRecord, EventSink, HarnessRunStatus, LimitKind,
    RecordingListener,
};
use tinyagents::harness::ids::{
    CallId, ComponentId as HarnessComponentId, EventId, ExecutionStatus, HarnessPhase, RunId,
    ThreadId,
};
use tinyagents::harness::observability::{
    AgentObservation, FanOutSink, HarnessEventJournal, HarnessStatusStore, InMemoryEventJournal,
    InMemoryStatusStore, JournalSink, JsonlSink, RedactingSink, StoreEventJournal,
};
use tinyagents::harness::providers::MockModel;
use tinyagents::harness::store::{AppendStore, InMemoryAppendStore, JsonlAppendStore};
use tinyagents::harness::testkit::{FakeTool, Trajectory};
use tinyagents::harness::tool::ToolCall;
use tinyagents::registry::{CapabilityRegistry, ComponentId, ComponentKind, ComponentMetadata};

#[tokio::test]
async fn capability_registry_resolves_aliases_and_hands_off_runtime_registries() {
    let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
    let model = Arc::new(MockModel::constant("ok"));
    let tool = Arc::new(FakeTool::returning("lookup", "answer"));

    registry.register_model("primary", model.clone()).unwrap();
    registry.register_tool(tool.clone()).unwrap();
    registry.register_router("route_by_score").unwrap();
    registry.register_reducer("append").unwrap();
    registry
        .register_descriptor(ComponentKind::Store, "kv")
        .unwrap();

    assert!(registry.has(ComponentKind::Model, "primary"));
    assert!(registry.has(ComponentKind::Tool, "lookup"));
    assert_eq!(registry.names(ComponentKind::Reducer), vec!["append"]);
    assert_eq!(
        registry
            .resolve_name(ComponentKind::Tool, "lookup")
            .as_deref(),
        Some("lookup")
    );

    registry
        .alias(ComponentKind::Model, "fast", "primary")
        .unwrap()
        .alias(ComponentKind::Tool, "search", "lookup")
        .unwrap()
        .alias(ComponentKind::Router, "router_alias", "route_by_score")
        .unwrap();

    assert!(registry.model("fast").is_some());
    assert!(registry.tool("search").is_some());
    assert_eq!(
        registry
            .metadata(ComponentKind::Model, "fast")
            .unwrap()
            .aliases,
        vec!["fast"]
    );
    assert_eq!(
        registry.names_including_aliases(ComponentKind::Tool),
        vec!["lookup", "search"]
    );

    let duplicate = registry
        .register_model("primary", model.clone())
        .unwrap_err();
    assert!(duplicate.to_string().contains("already registered"));
    let bad_alias = registry
        .alias(ComponentKind::Model, "ghost", "missing")
        .unwrap_err();
    assert!(bad_alias.to_string().contains("target is not registered"));
    let duplicate_alias = registry
        .alias(ComponentKind::Tool, "lookup", "lookup")
        .unwrap_err();
    assert!(
        duplicate_alias
            .to_string()
            .contains("already a registered component")
    );

    registry.replace_model("primary", Arc::new(MockModel::constant("new")));
    registry.replace_tool(Arc::new(FakeTool::returning("lookup", "new answer")));
    registry
        .replace_graph_blueprint(
            "notebook",
            tinyagents::language::Blueprint {
                graph_id: "notebook".into(),
                start: "start".into(),
                channels: Vec::new(),
                nodes: Vec::new(),
                edges: Vec::new(),
                ..tinyagents::language::Blueprint::default()
            },
        )
        .alias(ComponentKind::Graph, "nb", "notebook")
        .unwrap();

    assert!(registry.graph_blueprint("nb").is_some());

    let mut model_registry = registry.to_model_registry();
    model_registry.set_default("fast");
    assert_eq!(model_registry.names(), vec!["fast", "primary"]);
    assert!(model_registry.get("fast").is_some());

    let tool_registry = registry.to_tool_registry();
    assert_eq!(tool_registry.names(), vec!["lookup"]);
    assert_eq!(tool_registry.schemas()[0].name, "lookup");
    let result = tool_registry
        .get("lookup")
        .unwrap()
        .call(
            &(),
            ToolCall::new("call-1", "lookup", json!({ "query": "rust" })),
        )
        .await
        .unwrap();
    assert_eq!(result.content, "new answer");

    let resolver = registry.capability_resolver();
    assert!(resolver.model_allowed("fast"));
    assert!(resolver.tool_allowed("search"));
    assert!(resolver.subgraph_allowed("nb"));
    assert!(resolver.router_allowed("router_alias"));
    assert!(resolver.reducer_allowed("append"));
    assert!(resolver.node_kind_allowed("model"));

    let debug = format!("{registry:?}");
    assert!(debug.contains("primary"));
    assert!(debug.contains("lookup"));
}

#[test]
fn component_metadata_and_event_kinds_are_stable_serializable_contracts() {
    let id = ComponentId::new("researcher");
    assert_eq!(id.as_str(), "researcher");
    assert_eq!(id.to_string(), "researcher");
    assert_eq!(ComponentKind::ALL.len(), 7);
    assert_eq!(ComponentKind::Agent.as_str(), "agent");
    assert_eq!(ComponentKind::Tool.to_string(), "tool");

    let metadata = ComponentMetadata::new("researcher", ComponentKind::Agent)
        .with_description("research agent")
        .with_tag("fast")
        .with_tag("local");
    assert_eq!(metadata.name(), "researcher");
    assert_eq!(metadata.description.as_deref(), Some("research agent"));
    assert_eq!(metadata.tags, vec!["fast", "local"]);
    let encoded = serde_json::to_value(&metadata).unwrap();
    assert_eq!(encoded["kind"], "agent");

    let events = vec![
        AgentEvent::RunStarted {
            run_id: RunId::new("run-1"),
            thread_id: Some(ThreadId::new("thread-1")),
        },
        AgentEvent::ModelStarted {
            call_id: CallId::new("model-1"),
            model: "gpt".into(),
        },
        AgentEvent::ModelDelta {
            call_id: CallId::new("model-1"),
            delta: Default::default(),
        },
        AgentEvent::ModelCompleted {
            call_id: CallId::new("model-1"),
            usage: None,
        },
        AgentEvent::ToolStarted {
            call_id: CallId::new("tool-1"),
            tool_name: "lookup".into(),
        },
        AgentEvent::ToolCompleted {
            call_id: CallId::new("tool-1"),
            tool_name: "lookup".into(),
        },
        AgentEvent::StateUpdate,
        AgentEvent::MiddlewareStarted { name: "mw".into() },
        AgentEvent::MiddlewareCompleted { name: "mw".into() },
        AgentEvent::CacheHit {
            call_id: CallId::new("cache-1"),
            key: "secret-key".into(),
        },
        AgentEvent::CacheMiss {
            call_id: CallId::new("cache-2"),
            key: "miss-key".into(),
        },
        AgentEvent::RetryScheduled {
            call_id: CallId::new("retry-1"),
            attempt: 2,
        },
        AgentEvent::RateLimitWaited { waited_ms: 10 },
        AgentEvent::FallbackSelected {
            from: "large".into(),
            to: "small".into(),
        },
        AgentEvent::SubAgentStarted {
            name: "child".into(),
            depth: 1,
        },
        AgentEvent::SubAgentCompleted {
            name: "child".into(),
            depth: 1,
        },
        AgentEvent::SubAgentReused {
            name: "child".into(),
            turn: 1,
        },
        AgentEvent::Steered {
            command_kind: "cancel".into(),
            accepted: false,
        },
        AgentEvent::Compressed {
            from_tokens: 100,
            to_tokens: 25,
        },
        AgentEvent::RouteSelected {
            route: "done".into(),
        },
        AgentEvent::UsageRecorded {
            usage: tinyagents::harness::usage::Usage::new(1, 2),
        },
        AgentEvent::CostRecorded {
            cost: tinyagents::harness::cost::CostTotals::new(),
        },
        AgentEvent::LimitReached {
            kind: LimitKind::ModelCalls,
        },
        AgentEvent::MemoryLoaded,
        AgentEvent::MemorySaved,
        AgentEvent::ToolProgress {
            call_id: CallId::new("tool-1"),
            message: "halfway".into(),
        },
        AgentEvent::MiddlewareFailed {
            name: "mw".into(),
            error: "boom".into(),
        },
        AgentEvent::StreamClosed,
        AgentEvent::RunCompleted {
            run_id: RunId::new("run-1"),
        },
        AgentEvent::RunFailed {
            run_id: RunId::new("run-2"),
            error: "bad".into(),
        },
    ];
    let kinds: Vec<_> = events.iter().map(AgentEvent::kind).collect();
    assert!(kinds.contains(&"run.started"));
    assert!(kinds.contains(&"model.delta"));
    assert!(kinds.contains(&"middleware.failed"));
    assert!(kinds.contains(&"run.failed"));
    assert_eq!(LimitKind::WallClock.as_str(), "wall_clock");

    let trajectory = Trajectory::from_events(events);
    trajectory.assert_tool_called("lookup");
    trajectory.assert_model_called_times(1);
    trajectory
        .assert_order(&[
            "run.started",
            "model.started",
            "tool.started",
            "run.completed",
        ])
        .unwrap();
}

#[tokio::test]
async fn event_sinks_journals_and_status_stores_preserve_run_lineage() {
    let sink = EventSink::new();
    assert!(sink.is_empty());
    let recorder = Arc::new(RecordingListener::new());
    sink.subscribe(recorder.clone());
    assert_eq!(sink.len(), 1);

    let first = sink.emit(AgentEvent::RunStarted {
        run_id: RunId::new("run-1"),
        thread_id: Some(ThreadId::new("thread-1")),
    });
    let second = sink.emit(AgentEvent::RunCompleted {
        run_id: RunId::new("run-1"),
    });
    assert_eq!(first.offset, 0);
    assert_eq!(second.offset, 1);
    assert_eq!(recorder.len(), 2);
    assert_eq!(recorder.events()[0].event.kind(), "run.started");

    let journal = EventJournal::new();
    assert!(journal.is_empty());
    journal.append(AgentEvent::ToolStarted {
        call_id: CallId::new("tool-1"),
        tool_name: "lookup".into(),
    });
    journal.append(AgentEvent::ToolCompleted {
        call_id: CallId::new("tool-1"),
        tool_name: "lookup".into(),
    });
    assert_eq!(journal.len(), 2);
    assert_eq!(journal.replay_from(1)[0].event.kind(), "tool.completed");

    let run_id = RunId::new("child-run");
    let parent_id = RunId::new("parent-run");
    let root_id = RunId::new("root-run");
    let record = EventRecord {
        id: EventId::new("evt-99"),
        offset: 99,
        event: AgentEvent::StateUpdate,
    };
    let observation = AgentObservation::from_record(
        &record,
        run_id.clone(),
        Some(parent_id.clone()),
        root_id.clone(),
    );
    assert_eq!(observation.event_id.as_str(), "evt-99");
    assert_eq!(observation.run_id, run_id);
    assert_eq!(observation.parent_run_id, Some(parent_id.clone()));
    assert_eq!(observation.root_run_id, root_id);
    assert_eq!(observation.offset, 99);
    assert!(observation.ts_ms > 0);

    let memory_journal = InMemoryEventJournal::new();
    assert!(memory_journal.is_empty("child-run"));
    assert_eq!(memory_journal.append(observation.clone()).await.unwrap(), 0);
    assert_eq!(memory_journal.len("child-run"), 1);
    assert_eq!(
        memory_journal.read_from("child-run", 0).await.unwrap()[0]
            .event
            .kind(),
        "state.update"
    );
    assert!(
        memory_journal
            .read_from("missing", 0)
            .await
            .unwrap()
            .is_empty()
    );

    let append_store = InMemoryAppendStore::new();
    let store_journal = StoreEventJournal::new(append_store.clone());
    assert_eq!(store_journal.append(observation.clone()).await.unwrap(), 0);
    assert_eq!(store_journal.store().len("child-run").await.unwrap(), 1);
    assert_eq!(
        store_journal.read_from("child-run", 0).await.unwrap()[0]
            .event
            .kind(),
        "state.update"
    );

    let status_store = InMemoryStatusStore::new();
    assert!(status_store.is_empty());
    let mut status =
        HarnessRunStatus::new(RunId::new("run-status"), HarnessComponentId::new("agent"))
            .with_thread(ThreadId::new("thread-1"))
            .with_parent(RunId::new("parent-run"), RunId::new("root-run"));
    status.mark_running(HarnessPhase::Model);
    status.model_calls = 1;
    status.set_last_event(EventId::new("evt-1"));
    status.mark_interrupted();
    assert_eq!(status.status, ExecutionStatus::Interrupted);
    status.mark_failed("model failed");
    assert_eq!(status.current_phase, HarnessPhase::Done);
    assert_eq!(status.error.as_deref(), Some("model failed"));
    assert!(status.ended_at.is_some());
    status_store.put_status(status.clone()).await.unwrap();
    assert_eq!(status_store.len(), 1);
    assert_eq!(
        status_store
            .get_status("run-status")
            .await
            .unwrap()
            .unwrap()
            .parent_run_id
            .unwrap()
            .as_str(),
        "parent-run"
    );
    assert_eq!(
        status_store.list_by_thread("thread-1").await.unwrap().len(),
        1
    );

    let mut completed =
        HarnessRunStatus::new(RunId::new("run-done"), HarnessComponentId::new("agent"));
    completed.mark_running(HarnessPhase::Tools);
    completed.mark_completed();
    assert_eq!(completed.status, ExecutionStatus::Completed);
    assert!(completed.ended_at.is_some());
}

#[tokio::test]
async fn fanout_redaction_journal_and_jsonl_sinks_forward_best_effort_events() {
    let left = Arc::new(RecordingListener::new());
    let right = Arc::new(RecordingListener::new());
    let mut fanout = FanOutSink::new();
    assert!(fanout.is_empty());
    fanout.add(left.clone());
    let fanout = fanout.with(right.clone());
    assert_eq!(fanout.len(), 2);

    let secret_record = EventRecord {
        id: EventId::new("evt-secret"),
        offset: 0,
        event: AgentEvent::CacheHit {
            call_id: CallId::new("call-1"),
            key: "prefix-api-key-123-suffix".into(),
        },
    };
    fanout.on_event(&secret_record);
    assert_eq!(left.len(), 1);
    assert_eq!(right.len(), 1);

    let redacted_listener = Arc::new(RecordingListener::new());
    let redacting =
        RedactingSink::new(redacted_listener.clone(), vec!["api-key-123".into()]).with_mask("***");
    redacting.on_event(&secret_record);
    let redacted_value = serde_json::to_value(&redacted_listener.events()[0].event).unwrap();
    assert_eq!(redacted_value["key"], "prefix-***-suffix");

    let journal = Arc::new(InMemoryEventJournal::new());
    let journal_sink = JournalSink::new(journal.clone(), RunId::new("child-run"))
        .with_lineage(Some(RunId::new("parent-run")), RunId::new("root-run"));
    journal_sink.on_event(&secret_record);
    let observations = journal.read_from("child-run", 0).await.unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(
        observations[0].parent_run_id.as_ref().unwrap().as_str(),
        "parent-run"
    );
    assert_eq!(observations[0].root_run_id.as_str(), "root-run");

    let root = std::env::temp_dir().join(format!(
        "tinyagents-jsonl-sink-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let jsonl_store = JsonlAppendStore::new(root.clone());
    let jsonl_sink = JsonlSink::new(jsonl_store.clone(), "events");
    jsonl_sink.on_event(&secret_record);
    let rows = jsonl_store.read_from("events", 0).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, 0);
    assert_eq!(rows[0].1["event"]["key"], "prefix-api-key-123-suffix");

    std::fs::remove_dir_all(root).ok();
}
