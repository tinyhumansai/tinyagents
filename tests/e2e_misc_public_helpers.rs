use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::json;
use tinyagents::graph::observability::{
    GraphEventJournal, GraphStatusStore, InMemoryGraphEventJournal, InMemoryGraphStatusStore,
    JournalGraphSink, StoreGraphEventJournal,
};
use tinyagents::graph::reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
use tinyagents::graph::status::GraphRunStatus;
use tinyagents::graph::stream::{CollectingSink, GraphEvent, GraphEventSink, NoopSink, StreamMode};
use tinyagents::graph::{Command, Interrupt, NodeResult, Send};
use tinyagents::harness::ids::{
    CheckpointId, ExecutionStatus, GraphId, InterruptId, NodeId, RunId, ThreadId, new_call_id,
    new_cell_id, new_session_id, next_seq,
};
use tinyagents::harness::limits::{LimitTracker, RunLimits};
use tinyagents::harness::store::{AppendStore, InMemoryAppendStore};
use tinyagents::harness::tool::{ToolCall, ToolFormat, ToolResult, ToolSchema};
use tinyagents::repl::{CapabilityPolicy, ReplCommand, ReplOutcome, ReplSession, parse_command};

#[tokio::test]
async fn graph_reducers_streams_observability_and_status_helpers_work() {
    assert_eq!(OverwriteReducer.reduce(1, 2).unwrap(), 2);
    assert_eq!(
        AppendReducer.reduce(vec![1], vec![2, 3]).unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        SetUnionReducer.reduce(vec![1, 2], vec![2, 3]).unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(MinReducer.reduce(10, 2).unwrap(), 2);
    assert_eq!(MaxReducer.reduce(10, 20).unwrap(), 20);
    assert_eq!(
        ClosureReducer::new(|a: i32, b: i32| Ok(a * b))
            .reduce(3, 4)
            .unwrap(),
        12
    );
    assert_eq!(OverwriteStateReducer.apply("old", "new").unwrap(), "new");
    assert_eq!(
        ClosureStateReducer::new(|state: i32, update: i32| Ok(state + update))
            .apply(2, 5)
            .unwrap(),
        7
    );

    let send = Send {
        node: NodeId::new("worker"),
        arg: json!({ "item": 1 }),
    };
    let command = Command::send([send.clone()])
        .with_sends([send])
        .with_goto(["next"])
        .with_update(3)
        .with_resume(json!({ "resume": true }));
    assert_eq!(command.update, Some(3));
    assert_eq!(command.goto.len(), 3);
    assert_eq!(command.resume.as_ref().unwrap()["resume"], true);
    assert_eq!(Command::<i32>::update(9).update, Some(9));
    assert_eq!(
        Command::<i32>::resume(json!("ok")).resume,
        Some(json!("ok"))
    );
    let interrupt = Interrupt::new("node-a", json!({ "question": "continue?" }));
    assert!(interrupt.id.starts_with("interrupt-node-a-"));
    let explicit = Interrupt::with_id("int-1", "node-b", json!(1));
    assert_eq!(explicit.id, "int-1");

    let events = vec![
        GraphEvent::RunStarted {
            run_id: RunId::new("run"),
        },
        GraphEvent::RunCompleted {
            run_id: RunId::new("run"),
            steps: 2,
        },
        GraphEvent::RunFailed {
            run_id: RunId::new("run"),
            error: "bad".into(),
        },
        GraphEvent::StepStarted {
            step: 1,
            active: vec![NodeId::new("a")],
        },
        GraphEvent::StepCompleted { step: 1 },
        GraphEvent::TaskScheduled {
            node: NodeId::new("a"),
            step: 1,
        },
        GraphEvent::NodeStarted {
            node: NodeId::new("a"),
            step: 1,
        },
        GraphEvent::NodeCompleted {
            node: NodeId::new("a"),
            step: 1,
        },
        GraphEvent::NodeFailed {
            node: NodeId::new("a"),
            step: 1,
            error: "bad".into(),
        },
        GraphEvent::StateUpdated {
            node: NodeId::new("a"),
            step: 1,
        },
        GraphEvent::RouteSelected {
            node: NodeId::new("a"),
            target: NodeId::new("b"),
        },
        GraphEvent::CheckpointSaved {
            checkpoint_id: CheckpointId::new("cp-1"),
        },
        GraphEvent::InterruptEmitted {
            interrupt: explicit.clone(),
        },
        GraphEvent::SubgraphStarted {
            node: NodeId::new("a"),
            namespace: vec!["child".into()],
        },
        GraphEvent::SubgraphCompleted {
            node: NodeId::new("a"),
            namespace: vec!["child".into()],
        },
        GraphEvent::ContextForked {
            node: NodeId::new("a"),
            fork: 0,
            step: 1,
        },
        GraphEvent::RecursionDepthChanged { depth: 2 },
        GraphEvent::Custom {
            name: "custom".into(),
            data: json!({ "x": 1 }),
        },
    ];
    assert_eq!(events[0].kind(), "run.started");
    assert_eq!(events[3].step(), Some(1));
    assert_eq!(events[1].step(), Some(2));
    assert_eq!(events[10].step(), None);
    assert_ne!(StreamMode::Values, StreamMode::Debug);

    let sink = CollectingSink::new();
    assert!(sink.is_empty());
    for event in events.clone() {
        sink.emit(event);
    }
    assert_eq!(sink.len(), events.len());
    assert_eq!(sink.events()[0].kind(), "run.started");
    NoopSink.emit(GraphEvent::Custom {
        name: "drop".into(),
        data: json!(null),
    });

    let journal = Arc::new(InMemoryGraphEventJournal::new());
    assert!(journal.is_empty("run-g"));
    let journal_sink = JournalGraphSink::new(
        journal.clone(),
        RunId::new("run-g"),
        GraphId::new("graph-g"),
    )
    .with_lineage(Some(RunId::new("parent")), RunId::new("root"))
    .with_thread(Some(ThreadId::new("thread")))
    .with_namespace(vec!["child".into()])
    .with_inner(Arc::new(sink.clone()));
    journal_sink.emit(GraphEvent::StepStarted {
        step: 3,
        active: vec![NodeId::new("a")],
    });
    journal_sink.emit(GraphEvent::CheckpointSaved {
        checkpoint_id: CheckpointId::new("cp-3"),
    });
    assert_eq!(journal.len("run-g"), 2);
    let observations = journal.read_from("run-g", 0).await.unwrap();
    assert_eq!(observations[0].step, 3);
    assert_eq!(observations[1].step, 3);
    assert_eq!(
        observations[1].checkpoint_id.as_ref().unwrap().as_str(),
        "cp-3"
    );
    assert_eq!(
        observations[0].parent_run_id.as_ref().unwrap().as_str(),
        "parent"
    );
    assert_eq!(
        observations[0].thread_id.as_ref().unwrap().as_str(),
        "thread"
    );
    assert_eq!(observations[0].namespace, vec!["child"]);

    let append_store = InMemoryAppendStore::new();
    let store_journal = StoreGraphEventJournal::new(append_store.clone());
    assert_eq!(
        store_journal.append(observations[0].clone()).await.unwrap(),
        0
    );
    assert_eq!(store_journal.store().len("run-g").await.unwrap(), 1);
    assert_eq!(
        store_journal.read_from("run-g", 0).await.unwrap()[0]
            .event
            .kind(),
        "step.started"
    );

    let status_store = InMemoryGraphStatusStore::new();
    assert!(status_store.is_empty());
    let mut status = GraphRunStatus::new(
        RunId::new("run-status"),
        GraphId::new("graph-status"),
        ExecutionStatus::Running,
    );
    status.thread_id = Some(ThreadId::new("thread"));
    status.parent_run_id = Some(RunId::new("parent"));
    status.current_step = 4;
    status.active_nodes = vec![NodeId::new("a")];
    status.pending_interrupts = vec![InterruptId::new("int")];
    assert!(!status.is_terminal());
    status_store.put_status(status.clone()).await.unwrap();
    assert_eq!(status_store.len(), 1);
    assert_eq!(
        status_store
            .get_status("run-status")
            .await
            .unwrap()
            .unwrap()
            .current_step,
        4
    );
    assert_eq!(
        status_store.list_by_thread("thread").await.unwrap().len(),
        1
    );
    assert!(
        GraphRunStatus::new(
            RunId::new("done"),
            GraphId::new("g"),
            ExecutionStatus::Completed
        )
        .is_terminal()
    );

    let _ = NodeResult::<i32>::Update(1);
    let _ = NodeResult::<i32>::Command(Command::default());
    let _ = NodeResult::<i32>::Interrupt(interrupt);
}

#[test]
fn tool_schema_limits_ids_and_repl_contracts_cover_public_helpers() {
    let schema = ToolSchema::new(
        "make",
        "make a value",
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "enum": ["ok", "fine"] },
                "count": { "type": ["integer", "null"] },
                "items": { "type": "array", "items": { "type": "number" } }
            },
            "required": ["name"],
            "additionalProperties": false
        }),
    )
    .with_format(ToolFormat::Xml);
    assert!(!schema.format.is_json());
    schema
        .validate_call(&ToolCall::new(
            "call-1",
            "make",
            json!({ "name": "ok", "count": 1, "items": [1, 2.5] }),
        ))
        .unwrap();
    assert!(
        schema
            .validate_call(&ToolCall::new("call-2", "other", json!({ "name": "ok" })))
            .unwrap_err()
            .to_string()
            .contains("does not match")
    );
    assert!(
        schema
            .validate_call(&ToolCall::new("call-3", "make", json!({ "count": 1 })))
            .unwrap_err()
            .to_string()
            .contains("required")
    );
    assert!(
        schema
            .validate_call(&ToolCall::new("call-4", "make", json!({ "name": "bad" })))
            .unwrap_err()
            .to_string()
            .contains("enum")
    );
    assert!(
        schema
            .validate_call(&ToolCall::new(
                "call-5",
                "make",
                json!({ "name": "ok", "extra": true })
            ))
            .unwrap_err()
            .to_string()
            .contains("not allowed")
    );
    assert!(
        schema
            .validate_call(&ToolCall::new(
                "call-6",
                "make",
                json!({ "name": "ok", "items": ["x"] })
            ))
            .unwrap_err()
            .to_string()
            .contains("must be number")
    );

    let ok = ToolResult::text("call-1", "make", "done");
    assert!(!ok.is_error());
    let err = ToolResult::error("call-2", "make", "bad");
    assert!(err.is_error());

    let limits = RunLimits::default()
        .with_max_model_calls(1)
        .with_max_tool_calls(1)
        .with_max_wall_clock_ms(Some(1))
        .with_max_retries_per_call(3)
        .with_max_concurrency(Some(2))
        .with_max_depth(9);
    assert_eq!(limits.max_retries_per_call, 3);
    assert_eq!(limits.max_concurrency, Some(2));
    assert_eq!(limits.max_depth, 9);
    let mut tracker = LimitTracker::new(limits);
    assert_eq!(tracker.model_calls(), 0);
    assert_eq!(tracker.tool_calls(), 0);
    assert_eq!(tracker.remaining_model_calls(), 1);
    tracker.record_model_call().unwrap();
    tracker.record_tool_call().unwrap();
    assert!(tracker.record_model_call().is_err());
    assert!(tracker.record_tool_call().is_err());
    assert!(tracker.elapsed() < Duration::from_secs(10));
    assert!(tracker.remaining_wall_clock().is_some());
    thread::sleep(Duration::from_millis(2));
    assert!(tracker.check_wall_clock().is_err());
    assert_eq!(tracker.limits().max_model_calls, 1);

    let first_seq = next_seq();
    assert!(next_seq() > first_seq);
    assert!(new_session_id().as_str().starts_with("session-"));
    assert!(new_cell_id().as_str().starts_with("cell-"));
    assert!(new_call_id().as_str().starts_with("call-"));

    assert_eq!(parse_command("help").unwrap().name(), "help");
    assert_eq!(parse_command("?").unwrap(), ReplCommand::Help);
    assert_eq!(parse_command("q").unwrap(), ReplCommand::Quit);
    assert_eq!(
        parse_command(r#"set name "Ada Lovelace""#).unwrap(),
        ReplCommand::Set {
            key: "name".into(),
            value: "Ada Lovelace".into()
        }
    );
    assert_eq!(
        parse_command(r#"call tool {"x":1}"#).unwrap(),
        ReplCommand::Call {
            capability: "tool".into(),
            args: json!({ "x": 1 })
        }
    );
    assert!(parse_command("").is_err());
    assert!(parse_command("unknown").is_err());
    assert!(parse_command(r#"set x "unterminated"#).is_err());
    assert!(parse_command("call tool not-json").is_err());

    let mut policy = CapabilityPolicy::new();
    assert!(policy.is_empty());
    policy
        .allow("load")
        .allow("compile")
        .allow("run")
        .allow("tool");
    assert_eq!(policy.len(), 4);
    assert!(policy.is_allowed("tool"));
    let list_policy = CapabilityPolicy::from_list(["tool"]);
    assert!(list_policy.is_allowed("tool"));

    let mut session = ReplSession::new().with_policy(policy);
    session.set("direct", json!(42));
    assert_eq!(session.get("direct"), Some(&json!(42)));
    assert!(session.vars().contains_key("direct"));
    assert!(matches!(
        session.execute(ReplCommand::Help).unwrap(),
        ReplOutcome::Message(_)
    ));
    assert_eq!(
        session
            .execute(ReplCommand::Set {
                key: "name".into(),
                value: "Ada".into()
            })
            .unwrap(),
        ReplOutcome::Message("ok".into())
    );
    assert_eq!(
        session
            .execute(ReplCommand::Get { key: "name".into() })
            .unwrap(),
        ReplOutcome::Value(json!("Ada"))
    );
    assert!(matches!(
        session
            .execute(ReplCommand::Show {
                what: "vars".into()
            })
            .unwrap(),
        ReplOutcome::Value(_)
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Show {
                what: "graphs".into()
            })
            .unwrap(),
        ReplOutcome::Message(_)
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Show {
                what: "status".into()
            })
            .unwrap(),
        ReplOutcome::Value(_)
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Show { what: "bad".into() })
            .unwrap(),
        ReplOutcome::Message(_)
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Load {
                path: "x.rag".into()
            })
            .unwrap(),
        ReplOutcome::Planned { .. }
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Compile { name: "x".into() })
            .unwrap(),
        ReplOutcome::Planned { .. }
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Run {
                graph: "g".into(),
                input: "{}".into()
            })
            .unwrap(),
        ReplOutcome::Planned { .. }
    ));
    assert!(matches!(
        session
            .execute(ReplCommand::Call {
                capability: "tool".into(),
                args: json!({})
            })
            .unwrap(),
        ReplOutcome::Planned { .. }
    ));
    assert!(
        session
            .execute(ReplCommand::Call {
                capability: "blocked".into(),
                args: json!({})
            })
            .unwrap_err()
            .to_string()
            .contains("allowlist")
    );
    assert_eq!(
        session.execute(ReplCommand::Quit).unwrap(),
        ReplOutcome::Quit
    );
    assert!(!session.history.is_empty());
}
