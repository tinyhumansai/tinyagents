use super::super::DriverRunLedger;
use super::*;
use crate::run_ledger::types::*;
use crate::run_ledger::{RunLedgerStore, SessionHistoryStore};
use crate::testkit::MemoryStorage;
use crate::types::{SessionMessage, SessionRecord, SessionStatus, SessionToolCall};
use chrono::{DateTime, Duration, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tinyagents_storage_kit::{
    Item, MigrateOptions, Precondition, Registry, Scope, ScopedStorage, WriteOptions,
};
use tinystoragedrivers_api::DocumentStoreExt;
use tinystoragedrivers_api::StorageBackend;

fn agent_storage(backend: &Arc<dyn StorageBackend>) -> ScopedStorage {
    let scope = Scope::new("test-planner-ledger").unwrap();
    ScopedStorage::new(backend.for_scope(&scope).unwrap(), scope)
}

/// The v1 document id: a digest of the length-prefixed parts.
fn v1_id(parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part.as_bytes());
    }
    hex::encode(digest.finalize())
}

async fn put_v1(storage: &ScopedStorage, collection: &str, parts: &[&str], record: Value) {
    let envelope = json!({"schemaVersion": 1, "updatedSql": null, "record": record});
    let docs = storage.documents();
    docs.ensure_collection(&tinystoragedrivers_types::CollectionSpec::new(collection))
        .await
        .unwrap();
    docs.put_with(
        collection,
        &v1_id(parts),
        envelope,
        Precondition::None,
        WriteOptions::new().created_ms(1_700_000_000_000),
    )
    .await
    .unwrap();
}

async fn migrate(storage: &ScopedStorage) {
    for spec in tinyagents_storage_kit::primitives::collection_specs() {
        storage.documents().ensure_collection(&spec).await.unwrap();
    }
    for schema in [ledger_storage_schema(), history_storage_schema()] {
        for item in schema.items {
            if let Item::Collection { spec, .. } = item {
                storage.documents().ensure_collection(&spec).await.unwrap();
            }
        }
    }
    Registry::compose([ledger_storage_schema(), history_storage_schema()])
        .unwrap()
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap();
}

fn at(text: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(text)
        .unwrap()
        .with_timezone(&Utc)
}

fn workflow_v1() -> WorkflowRun {
    WorkflowRun {
        id: "wf/1".into(),
        definition_id: "def".into(),
        parent_thread_id: Some("thread".into()),
        input: json!({"camelCase": {"stays": 1}}),
        phase_states: json!({}),
        child_run_ids: vec!["child".into()],
        status: WorkflowRunStatus::Running,
        summary: None,
        started_at: at("2026-01-01T00:00:00.123456Z"),
        updated_at: at("2026-01-01T00:01:00Z"),
        completed_at: None,
        revision: 4,
        lease_owner: Some("old-driver".into()),
        lease_expires_at: Some(at("2026-01-01T01:00:00Z")),
    }
}

#[test]
fn the_schemas_name_what_each_owner_declares() {
    let ledger = ledger_storage_schema();
    assert_eq!(ledger.owner.as_str(), "ledger");
    let mut names: Vec<&str> = ledger.items.iter().map(|item| item.name()).collect();
    names.sort_unstable();
    let mut expected = [
        "ledger.agent_runs",
        "ledger.event_heads",
        "ledger.event_outbox",
        "ledger.events",
        "ledger.inputs",
        "ledger.members",
        "ledger.tasks",
        "ledger.teams",
        "ledger.telemetry",
        "ledger.tool_effects",
        "ledger.workflow_runs",
    ];
    expected.sort_unstable();
    assert_eq!(names, expected);
    let history = history_storage_schema();
    assert_eq!(history.owner.as_str(), "history");
    let mut names: Vec<&str> = history.items.iter().map(|item| item.name()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "history.messages",
            "history.search",
            "history.sessions",
            "history.tool_calls"
        ]
    );
    Registry::compose([ledger, history]).unwrap();
}

#[tokio::test]
async fn ledger_records_move_to_snake_case_bodies_with_ms_times_and_key_ids() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    let workflow = workflow_v1();
    put_v1(
        &storage,
        "run_ledger_workflow_runs",
        &[&workflow.id],
        serde_json::to_value(&workflow).unwrap(),
    )
    .await;
    let team = AgentTeam {
        id: "team".into(),
        parent_thread_id: None,
        lead_agent_id: "lead".into(),
        status: AgentTeamStatus::Active,
        summary: None,
        created_at: at("2026-01-02T00:00:00Z"),
        updated_at: at("2026-01-02T00:00:01Z"),
        closed_at: None,
    };
    put_v1(
        &storage,
        "run_ledger_teams",
        &["team"],
        serde_json::to_value(&team).unwrap(),
    )
    .await;
    put_v1(
        &storage,
        "run_ledger_tool_effects",
        &["run", "call"],
        json!({
            "runId": "run", "callId": "call", "tool": "shell", "status": "started",
            "idempotencyKey": null, "effectSummary": null,
            "startedAt": "2026-01-03T00:00:00Z", "settledAt": null
        }),
    )
    .await;
    migrate(&storage).await;

    let docs = storage.documents();
    let stored = docs
        .get("ledger.workflow_runs", "wf%2F1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.meta.schema, 2);
    assert!(
        stored.meta.created_ms > 0,
        "the engine assigns new row metadata"
    );
    assert_eq!(
        stored.doc,
        json!({
            "id": "wf/1", "definition_id": "def", "parent_thread_id": "thread",
            "input": {"camelCase": {"stays": 1}}, "phase_states": {},
            "child_run_ids": ["child"], "status": "running", "summary": null,
            "started_ms": workflow.started_at.timestamp_millis(),
            "updated_ms": workflow.updated_at.timestamp_millis(),
            "completed_ms": null, "revision": 4
        })
    );
    let team_doc = docs.get("ledger.teams", "team").await.unwrap().unwrap().doc;
    assert_eq!(team_doc["created_ms"], team.created_at.timestamp_millis());
    assert_eq!(team_doc["lead_agent_id"], "lead");
    let effect = docs
        .get("ledger.tool_effects", "run/call")
        .await
        .unwrap()
        .unwrap()
        .doc;
    assert_eq!(
        effect["started_ms"],
        at("2026-01-03T00:00:00Z").timestamp_millis()
    );
    assert_eq!(effect["settled_ms"], Value::Null);
    assert!(
        docs.query_all("run_ledger_teams", &Default::default())
            .await
            .unwrap()
            .is_empty()
    );

    // The typed store reads the migrated records back as the wire types.
    let ledger = DriverRunLedger::new(storage.clone());
    let read = ledger.get_workflow_run("wf/1").await.unwrap().unwrap();
    assert_eq!(read.started_at, at("2026-01-01T00:00:00.123Z"));
    assert_eq!(read.revision, 4);
    assert_eq!(read.lease_owner, None);
    assert_eq!(
        ledger
            .get_agent_team("team")
            .await
            .unwrap()
            .unwrap()
            .lead_agent_id,
        "lead"
    );
    assert_eq!(
        ledger
            .list_unresolved_tool_effects("run")
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn run_events_move_to_the_run_stream_and_sequences_continue() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    let mut events = Vec::new();
    for sequence in 1..=2u64 {
        let event = RunEvent {
            run_id: "run/a".into(),
            sequence,
            event_type: "step".into(),
            payload: json!({"n": sequence}),
            timestamp: at("2026-01-01T00:00:00Z") + Duration::seconds(sequence as i64),
        };
        put_v1(
            &storage,
            "run_ledger_event_outbox",
            &["run/a", &sequence.to_string()],
            serde_json::to_value(&event).unwrap(),
        )
        .await;
        storage
            .streams()
            .append(
                &format!("run_events:{}", v1_id(&["run/a"])),
                serde_json::to_value(&event).unwrap(),
            )
            .await
            .unwrap();
        events.push(event);
    }
    put_v1(
        &storage,
        "run_ledger_event_heads",
        &["run/a"],
        json!({"next": 3}),
    )
    .await;
    migrate(&storage).await;

    let stream = "ledger.events/run%2Fa";
    assert_eq!(storage.streams().len(stream).await.unwrap(), 2);
    let entry = &storage.streams().read_window(stream, 1, 1).await.unwrap()[0];
    assert_eq!(
        entry.value,
        json!({"sequence": 2, "event_type": "step", "payload": {"n": 2},
               "ts_ms": events[1].timestamp.timestamp_millis()})
    );
    assert!(
        storage
            .streams()
            .streams("run_events:")
            .await
            .unwrap()
            .is_empty()
    );

    let ledger = DriverRunLedger::new(storage.clone());
    assert_eq!(ledger.reconcile_run_events("run/a").await.unwrap(), 2);
    let listed = ledger
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run/a".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(listed.events, events);
    // The v1 head is dropped; the next sequence follows the last migrated event.
    let next = ledger
        .append_run_event(RunEventAppend {
            run_id: "run/a".into(),
            event_type: "after".into(),
            payload: json!({}),
        })
        .await
        .unwrap();
    assert_eq!(next.sequence, 3);
}

#[tokio::test]
async fn history_moves_to_ms_times_and_search_is_rebuilt() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    let session = SessionRecord {
        id: "s1".into(),
        agent_definition_id: "researcher".into(),
        agent_definition_name: "Researcher".into(),
        session_key: "s1".into(),
        parent_session_id: None,
        thread_id: None,
        source_channel: None,
        status: SessionStatus::Completed,
        model: None,
        turn_count: 1,
        input_tokens: 1,
        output_tokens: 2,
        cached_input_tokens: 0,
        cost_usd: 0.5,
        transcript_path: None,
        started_at: at("2026-02-01T00:00:00.000123Z"),
        ended_at: Some(at("2026-02-01T00:05:00Z")),
    };
    put_v1(
        &storage,
        "session_history_sessions",
        &["s1"],
        serde_json::to_value(&session).unwrap(),
    )
    .await;
    let message = SessionMessage {
        id: 7,
        session_id: "s1".into(),
        role: "user".into(),
        content: "needle in a haystack".into(),
        reasoning_content: None,
        model: None,
        input_tokens: None,
        output_tokens: None,
        cost_usd: None,
        created_at: at("2026-02-01T00:01:00Z"),
    };
    put_v1(
        &storage,
        "session_history_messages",
        &["message", "7"],
        serde_json::to_value(&message).unwrap(),
    )
    .await;
    let call = SessionToolCall {
        id: 3,
        session_id: "s1".into(),
        message_id: Some(7),
        tool_name: "grep".into(),
        tool_input: None,
        tool_output: None,
        status: "ok".into(),
        duration_ms: Some(12),
        created_at: at("2026-02-01T00:02:00Z"),
    };
    put_v1(
        &storage,
        "session_history_tool_calls",
        &["tool_call", "3"],
        serde_json::to_value(&call).unwrap(),
    )
    .await;
    put_v1(
        &storage,
        "session_history_fts",
        &["s", "s1"],
        json!({"sessionId": "s1", "agentName": "Researcher", "content": "", "toolName": ""}),
    )
    .await;
    migrate(&storage).await;

    let docs = storage.documents();
    let stored = docs
        .get("history.sessions", "s1")
        .await
        .unwrap()
        .unwrap()
        .doc;
    assert_eq!(
        stored["started_ms"],
        at("2026-02-01T00:00:00Z").timestamp_millis()
    );
    assert_eq!(
        stored["ended_ms"],
        at("2026-02-01T00:05:00Z").timestamp_millis()
    );
    assert!(stored.get("started_at").is_none());
    assert_eq!(
        docs.get("history.messages", "7")
            .await
            .unwrap()
            .unwrap()
            .doc["created_ms"],
        message.created_at.timestamp_millis()
    );
    assert_eq!(
        docs.get("history.tool_calls", "3")
            .await
            .unwrap()
            .unwrap()
            .doc["duration_ms"],
        12
    );
    assert!(
        docs.query_all("session_history_fts", &Default::default())
            .await
            .unwrap()
            .is_empty()
    );

    let ledger = DriverRunLedger::new(storage.clone());
    for query in ["needle", "grep", "Researcher"] {
        let found = ledger
            .search_sessions(&crate::types::SessionSearchParams {
                query: Some(query.into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(found.total, 1, "{query}");
    }
    assert_eq!(ledger.list_messages("s1", None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn migrating_twice_changes_nothing() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    let workflow = workflow_v1();
    put_v1(
        &storage,
        "run_ledger_workflow_runs",
        &[&workflow.id],
        serde_json::to_value(&workflow).unwrap(),
    )
    .await;
    migrate(&storage).await;
    migrate(&storage).await;
    let rows = storage
        .documents()
        .query_all("ledger.workflow_runs", &Default::default())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
}
