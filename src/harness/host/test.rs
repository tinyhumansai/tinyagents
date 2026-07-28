//! Unit tests for the host capability seams and their default implementations.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::harness::events::{AgentEvent, EventRecord};
use crate::harness::ids::{CallId, EventId, RunId, ThreadId};
use crate::harness::model::{ChatModel, ModelResponse};
use crate::harness::testkit::ScriptedModel;
use crate::harness::usage::Usage;
use crate::harness::workspace::WorkspaceDescriptor;

/// The unit application state every default implementation is exercised with.
const STATE: &() = &();

fn run_id() -> RunId {
    RunId::new("run-1")
}

fn thread_id(value: &str) -> ThreadId {
    ThreadId::new(value)
}

// ---------------------------------------------------------------------------
// MemoryProvider — InMemoryMemoryProvider
// ---------------------------------------------------------------------------

async fn seeded_memory() -> InMemoryMemoryProvider {
    let memory = InMemoryMemoryProvider::new();
    memory
        .write(
            STATE,
            MemoryWrite::new("preferences", "tone", "the user prefers terse answers")
                .with_category("style")
                .with_thread_id("t1"),
        )
        .await
        .unwrap();
    memory
        .write(
            STATE,
            MemoryWrite::new("preferences", "format", "answers should use bullet lists")
                .with_category("style")
                .with_thread_id("t2"),
        )
        .await
        .unwrap();
    memory
        .write(
            STATE,
            MemoryWrite::new("facts", "city", "the user lives in Lisbon").with_category("profile"),
        )
        .await
        .unwrap();
    memory
}

#[tokio::test]
async fn in_memory_provider_recalls_by_token_overlap_scored_and_sorted() {
    let memory = seeded_memory().await;

    let hits = memory
        .recall(STATE, &MemoryQuery::new("terse answers"))
        .await
        .unwrap();

    assert_eq!(hits.len(), 2, "both preference rows contain 'answers'");
    assert_eq!(hits[0].key, "tone");
    assert_eq!(hits[0].score, Some(1.0), "matched both query tokens");
    assert_eq!(hits[1].key, "format");
    assert_eq!(hits[1].score, Some(0.5), "matched one of two query tokens");
}

#[tokio::test]
async fn in_memory_provider_returns_empty_for_empty_query_without_erroring() {
    let memory = seeded_memory().await;

    // The contract explicitly promises `Ok(vec![])` rather than an error, and
    // the zero-token case must not reach the matched/total division.
    assert!(
        memory
            .recall(STATE, &MemoryQuery::new(""))
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        memory
            .recall(STATE, &MemoryQuery::new("   \t \n "))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn in_memory_provider_returns_empty_for_non_matching_query() {
    let memory = seeded_memory().await;

    let hits = memory
        .recall(STATE, &MemoryQuery::new("quantum chromodynamics"))
        .await
        .unwrap();

    assert!(hits.is_empty());
}

#[tokio::test]
async fn in_memory_provider_honours_namespace_limit_and_min_score() {
    let memory = seeded_memory().await;

    let namespaced = memory
        .recall(STATE, &MemoryQuery::new("user").with_namespace("facts"))
        .await
        .unwrap();
    assert_eq!(namespaced.len(), 1);
    assert_eq!(namespaced[0].key, "city");

    let limited = memory
        .recall(STATE, &MemoryQuery::new("terse answers").with_limit(1))
        .await
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].key, "tone");

    let floored = memory
        .recall(
            STATE,
            &MemoryQuery::new("terse answers").with_min_score(0.9),
        )
        .await
        .unwrap();
    assert_eq!(floored.len(), 1);
    assert_eq!(floored[0].key, "tone");
}

#[tokio::test]
async fn in_memory_provider_scopes_recall_to_a_thread_unless_cross_thread() {
    let memory = seeded_memory().await;
    let t1 = thread_id("t1");

    let same_thread = memory
        .recall(STATE, &MemoryQuery::new("answers").with_thread(&t1, false))
        .await
        .unwrap();
    assert_eq!(same_thread.len(), 1);
    assert_eq!(same_thread[0].key, "tone");

    let cross_thread = memory
        .recall(STATE, &MemoryQuery::new("answers").with_thread(&t1, true))
        .await
        .unwrap();
    assert_eq!(cross_thread.len(), 2);
}

#[tokio::test]
async fn in_memory_provider_lists_by_namespace_and_category_without_scores() {
    let memory = seeded_memory().await;

    let style = memory
        .list(
            STATE,
            &MemoryFilter::new()
                .with_namespace("preferences")
                .with_category("style"),
        )
        .await
        .unwrap();

    assert_eq!(style.len(), 2);
    assert_eq!(style[0].key, "format", "list sorts by key ascending");
    assert_eq!(style[1].key, "tone");
    assert!(style.iter().all(|record| record.score.is_none()));

    let missing = memory
        .list(STATE, &MemoryFilter::new().with_category("nonexistent"))
        .await
        .unwrap();
    assert!(missing.is_empty());
}

#[tokio::test]
async fn in_memory_provider_write_upserts_on_namespace_and_key() {
    let memory = InMemoryMemoryProvider::new();
    memory
        .write(STATE, MemoryWrite::new("facts", "city", "Lisbon"))
        .await
        .unwrap();
    memory
        .write(STATE, MemoryWrite::new("facts", "city", "Porto"))
        .await
        .unwrap();

    let all = memory.list(STATE, &MemoryFilter::new()).await.unwrap();
    assert_eq!(all.len(), 1, "same (namespace, key) overwrites");
    assert_eq!(all[0].content, "Porto");
    assert_eq!(all[0].id, "facts/city");
    assert_eq!(all[0].namespace.as_deref(), Some("facts"));
    assert_eq!(memory.len().unwrap(), 1);
}

#[tokio::test]
async fn in_memory_provider_namespace_digests_take_the_empty_trait_default() {
    let memory = seeded_memory().await;

    let digests = memory
        .namespace_digests(
            STATE,
            DigestCaps {
                per_namespace_max_chars: 512,
                total_max_chars: 2048,
            },
        )
        .await
        .unwrap();

    assert!(digests.is_empty(), "no rollup layer, so opt out not error");
}

#[test]
fn memory_record_round_trips_through_serde_omitting_absent_fields() {
    let record = MemoryRecord {
        id: "facts/city".into(),
        key: "city".into(),
        content: "Lisbon".into(),
        namespace: Some("facts".into()),
        category: None,
        thread_id: None,
        score: None,
        recorded_at: None,
        attributes: Value::Null,
    };

    let encoded = serde_json::to_value(&record).unwrap();
    assert!(encoded.get("category").is_none());
    assert!(encoded.get("score").is_none());

    let decoded: MemoryRecord = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, record);
}

// ---------------------------------------------------------------------------
// ContextComposer — PassthroughContextComposer
// ---------------------------------------------------------------------------

#[tokio::test]
async fn passthrough_composer_contributes_no_prompt_and_no_blocks() {
    let composer = PassthroughContextComposer::new();
    let run = run_id();
    let workspace = WorkspaceDescriptor::new(PathBuf::from("/tmp/workspace"));
    let visible = vec!["read_file".to_string()];

    let prompt = ContextComposer::compose_system_prompt(
        &composer,
        STATE,
        &SystemPromptRequest {
            run_id: &run,
            thread_id: None,
            agent_id: "agent",
            model_id: "model",
            tools: &[],
            visible_tool_names: &visible,
            tool_call_instructions: "",
            workspace: Some(&workspace),
        },
    )
    .unwrap();
    assert!(prompt.is_empty());

    let prepared = composer
        .prepare_turn(
            STATE,
            &TurnPreparationRequest {
                run_id: &run,
                thread_id: None,
                agent_id: "agent",
                input: "hello",
                turn_index: 0,
                first_turn: true,
                resumed: false,
            },
        )
        .await
        .unwrap();

    assert_eq!(prepared, TurnPreparation::default());
    assert!(prepared.blocks.is_empty());
    assert!(prepared.extras.is_null());
}

#[test]
fn context_block_defaults_to_the_turn_prefix_placement() {
    let block = ContextBlock::new("recall", "body text");
    assert_eq!(block.placement, ContextPlacement::TurnPrefix);
    assert_eq!(block.priority, 0);

    let system = block
        .clone()
        .with_placement(ContextPlacement::SystemPrefix)
        .with_priority(10);
    assert_eq!(system.placement, ContextPlacement::SystemPrefix);
    assert_eq!(system.priority, 10);
}

#[test]
fn turn_preparation_round_trips_and_omits_null_extras() {
    let prepared = TurnPreparation::new(vec![ContextBlock::new("goal", "finish the migration")])
        .with_extras(json!({ "host": "opaque" }));

    let encoded = serde_json::to_value(&prepared).unwrap();
    assert_eq!(encoded["extras"]["host"], "opaque");

    let bare = serde_json::to_value(TurnPreparation::default()).unwrap();
    assert!(bare.get("extras").is_none());

    let decoded: TurnPreparation = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, prepared);
}

// ---------------------------------------------------------------------------
// SecurityGate — RootContainedSecurityGate
// ---------------------------------------------------------------------------

fn resolve(path: &str, root: &str, intent: PathIntent) -> crate::error::Result<PathBuf> {
    let gate = RootContainedSecurityGate::new();
    let run = run_id();
    SecurityGate::resolve_path(
        &gate,
        STATE,
        &PathRequest {
            run_id: &run,
            path: Path::new(path),
            root: Path::new(root),
            intent,
        },
    )
}

#[test]
fn root_contained_gate_resolves_relative_paths_under_the_root() {
    assert_eq!(
        resolve("notes/today.md", "/work", PathIntent::Read).unwrap(),
        PathBuf::from("/work/notes/today.md")
    );
    assert_eq!(
        resolve("./notes/./today.md", "/work", PathIntent::Write).unwrap(),
        PathBuf::from("/work/notes/today.md")
    );
    assert_eq!(
        resolve("notes/../today.md", "/work", PathIntent::Write).unwrap(),
        PathBuf::from("/work/today.md")
    );
    assert_eq!(
        resolve("/work/notes/today.md", "/work", PathIntent::Read).unwrap(),
        PathBuf::from("/work/notes/today.md")
    );
}

#[test]
fn root_contained_gate_resolves_paths_that_do_not_exist() {
    // The check is lexical by contract, so a target that has never existed
    // resolves exactly like one that does.
    let resolved = resolve(
        "generated/2099/report.md",
        "/nonexistent-root",
        PathIntent::Write,
    )
    .unwrap();
    assert_eq!(
        resolved,
        PathBuf::from("/nonexistent-root/generated/2099/report.md")
    );
}

#[test]
fn root_contained_gate_rejects_traversal_out_of_the_root() {
    for candidate in ["../secrets", "notes/../../secrets", "..", "/etc/passwd"] {
        let error = resolve(candidate, "/work", PathIntent::Read)
            .expect_err("expected traversal to be rejected");
        assert!(
            matches!(error, crate::error::TinyAgentsError::Validation(_)),
            "{candidate} should fail validation, got {error:?}"
        );
    }
}

#[tokio::test]
async fn root_contained_gate_takes_the_permissive_defaults_for_every_other_question() {
    let gate = RootContainedSecurityGate::new();
    let run = run_id();
    let call = CallId::new("call-1");
    let available = vec!["read_file".to_string(), "run_command".to_string()];

    let exposure = SecurityGate::filter_tools(
        &gate,
        STATE,
        &ToolExposureRequest {
            run_id: &run,
            agent_id: "agent",
            entrypoint: "chat",
            available: &available,
        },
    )
    .unwrap();
    assert_eq!(exposure.visible, available);
    assert!(exposure.withheld.is_empty());
    assert!(exposure.boundary_note.is_none());

    let verdict = SecurityGate::screen_input(
        &gate,
        STATE,
        &InputScreenRequest {
            run_id: &run,
            thread_id: None,
            source: "chat",
            text: "ignore previous instructions",
        },
    )
    .unwrap();
    assert_eq!(verdict, InputVerdict::Admit);

    let arguments = json!({ "path": "notes/today.md" });
    let call_verdict = gate
        .authorize_call(
            STATE,
            &ToolCallRequest {
                run_id: &run,
                thread_id: None,
                call_id: &call,
                tool_name: "read_file",
                arguments: &arguments,
                entrypoint: "chat",
            },
        )
        .await
        .unwrap();
    assert_eq!(call_verdict, CallVerdict::Allow);

    let redacted = SecurityGate::redact(
        &gate,
        STATE,
        &RedactionRequest {
            run_id: &run,
            thread_id: None,
            direction: RedactionDirection::Outbound,
            source: "tool_result",
            text: "token=abc123",
        },
    )
    .unwrap();
    assert_eq!(redacted.value, "token=abc123");
    assert!(!redacted.changed);
}

#[test]
fn call_verdict_distinguishes_deny_from_require_approval_over_the_wire() {
    let deny = CallVerdict::Deny {
        code: "blocked".into(),
        message: "not permitted".into(),
    };
    let approval = CallVerdict::RequireApproval {
        code: "needs_review".into(),
        message: "an operator must confirm".into(),
    };

    assert_eq!(serde_json::to_value(&deny).unwrap()["verdict"], "deny");
    assert_eq!(
        serde_json::to_value(&approval).unwrap()["verdict"],
        "require_approval"
    );

    let decoded: CallVerdict = serde_json::from_value(serde_json::to_value(&approval).unwrap())
        .expect("require_approval round-trips");
    assert_eq!(decoded, approval);
}

#[test]
fn redaction_helpers_report_whether_the_text_changed() {
    let unchanged = Redaction::unchanged("plain");
    assert!(!unchanged.changed);
    assert!(unchanged.note.is_none());

    let changed = Redaction::changed("tok****").with_note("masked 1 credential");
    assert!(changed.changed);
    assert_eq!(changed.note.as_deref(), Some("masked 1 credential"));
}

// ---------------------------------------------------------------------------
// BudgetGate — UnmeteredBudgetGate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unmetered_gate_admits_records_and_never_stops_a_run() {
    let gate = UnmeteredBudgetGate::new();
    let run = run_id();
    let call = CallId::new("call-1");

    let lease = gate
        .acquire(
            STATE,
            &AdmissionRequest {
                run_id: &run,
                agent_id: "agent",
                workload: "chat",
                interactive: true,
            },
        )
        .await
        .unwrap();
    assert!(lease.is_some(), "unmetered admission always grants a lease");

    let usage = Usage {
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 120,
        ..Default::default()
    };
    assert_eq!(
        BudgetGate::estimate_cost(&gate, STATE, "model", &usage),
        crate::harness::cost::CostTotals::default()
    );

    gate.record_usage(
        STATE,
        &UsageEntry {
            run_id: &run,
            call_id: &call,
            model_id: "model",
            usage,
            cost: crate::harness::cost::CostTotals::default(),
        },
    )
    .await
    .unwrap();

    let verdict = gate
        .account_turn(
            STATE,
            &TurnCharge {
                run_id: &run,
                thread_id: None,
                usage,
                cost: crate::harness::cost::CostTotals::default(),
                elapsed_secs: 3,
            },
        )
        .await
        .unwrap();
    assert_eq!(verdict, BudgetVerdict::Continue);
}

#[test]
fn budget_lease_carries_an_opaque_host_guard_back_out() {
    let lease = BudgetLease::new(String::from("permit"));
    assert_eq!(format!("{lease:?}"), "BudgetLease(..)");

    let guard = lease.into_inner();
    assert_eq!(
        guard.downcast_ref::<String>().map(String::as_str),
        Some("permit")
    );

    // The unmetered lease holds nothing at all and is still droppable.
    drop(BudgetLease::unmetered());
}

#[test]
fn budget_verdict_stop_carries_a_host_authored_reason() {
    let stop = BudgetVerdict::Stop {
        reason: "goal budget exhausted".into(),
    };
    let encoded = serde_json::to_value(&stop).unwrap();
    assert_eq!(encoded["verdict"], "stop");
    assert_eq!(encoded["reason"], "goal budget exhausted");

    let decoded: BudgetVerdict = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, stop);
}

// ---------------------------------------------------------------------------
// DefinitionRegistry — InMemoryDefinitionRegistry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_memory_definition_registry_starts_empty() {
    let registry = InMemoryDefinitionRegistry::new();

    assert!(
        DefinitionRegistry::list(&registry, STATE)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        DefinitionRegistry::get(&registry, STATE, "anything")
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        DefinitionRegistry::default_id(&registry, STATE)
            .await
            .unwrap()
            .is_none(),
        "the crate ships no default agent identity"
    );
    assert!(registry.is_empty().unwrap());
}

#[tokio::test]
async fn in_memory_definition_registry_lists_in_insertion_order() {
    let registry = InMemoryDefinitionRegistry::new()
        .with_definition(AgentDefinition::new("beta"))
        .with_definition(AgentDefinition::new("alpha"))
        .with_default_id("beta");

    let listed = DefinitionRegistry::list(&registry, STATE).await.unwrap();
    let ids: Vec<&str> = listed.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, vec!["beta", "alpha"], "insertion order, not sorted");

    assert_eq!(
        DefinitionRegistry::default_id(&registry, STATE)
            .await
            .unwrap()
            .as_deref(),
        Some("beta")
    );
}

#[tokio::test]
async fn in_memory_definition_registry_replaces_in_place_without_reordering() {
    let registry = InMemoryDefinitionRegistry::new()
        .with_definition(AgentDefinition::new("first"))
        .with_definition(AgentDefinition::new("second"));

    registry
        .insert(AgentDefinition::new("first").with_description("updated"))
        .unwrap();

    let listed = DefinitionRegistry::list(&registry, STATE).await.unwrap();
    assert_eq!(listed.len(), 2, "replace, not append");
    assert_eq!(listed[0].id, "first");
    assert_eq!(listed[0].description.as_deref(), Some("updated"));
    assert_eq!(listed[1].id, "second");
}

#[test]
fn agent_definition_keeps_host_specific_data_in_opaque_extras() {
    let definition = AgentDefinition::new("assistant")
        .with_description("general purpose")
        .with_system_prompt("be helpful")
        .with_model("model-a")
        .with_tools(["read_file", "write_file"])
        .with_extras(json!({ "host_only": { "compaction": "aggressive" } }));

    let encoded = serde_json::to_value(&definition).unwrap();
    assert_eq!(encoded["extras"]["host_only"]["compaction"], "aggressive");

    let decoded: AgentDefinition = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, definition);

    // A bare definition serializes to just its id: every other field is
    // skipped when absent, so the wire shape stays minimal.
    let bare = serde_json::to_value(AgentDefinition::new("bare")).unwrap();
    assert_eq!(bare, json!({ "id": "bare" }));
}

// ---------------------------------------------------------------------------
// ExperienceStore — NoopExperienceStore
// ---------------------------------------------------------------------------

#[tokio::test]
async fn noop_experience_store_retrieves_nothing_and_retains_nothing() {
    let store = NoopExperienceStore::new();
    let tools = vec!["read_file".to_string()];

    let hits = store
        .retrieve(
            STATE,
            &ExperienceQuery {
                text: "migrate the database",
                agent_id: Some("agent"),
                entrypoint: Some("chat"),
                partition: Some("partition-a"),
                tool_names: &tools,
                max_hits: 5,
            },
        )
        .await
        .unwrap();
    assert!(hits.is_empty());

    store
        .record(
            STATE,
            vec![ExperienceEntry {
                id: "outcome-1".into(),
                agent_id: Some("agent".into()),
                partition: None,
                payload: json!({ "opaque": true }),
            }],
        )
        .await
        .unwrap();
}

#[test]
fn experience_hit_round_trips_with_its_match_reasons() {
    let hit = ExperienceHit {
        id: "outcome-1".into(),
        body: "previously solved by reading the manifest first".into(),
        score: Some(0.8),
        match_reasons: vec!["same_tool".into()],
    };

    let decoded: ExperienceHit =
        serde_json::from_value(serde_json::to_value(&hit).unwrap()).unwrap();
    assert_eq!(decoded, hit);

    let evidence_free = ExperienceHit {
        match_reasons: Vec::new(),
        ..hit
    };
    assert!(
        evidence_free.match_reasons.is_empty(),
        "callers may drop hits with no stated reason"
    );
}

// ---------------------------------------------------------------------------
// LearningSink — NoopLearningSink
// ---------------------------------------------------------------------------

fn sample_turn_record() -> TurnRecord {
    TurnRecord {
        run_id: run_id(),
        thread_id: Some(thread_id("t1")),
        agent_id: Some("agent".into()),
        entrypoint: Some("chat".into()),
        input: "hello".into(),
        output: "hi".into(),
        tool_calls: vec![ToolOutcomeRecord {
            name: "read_file".into(),
            arguments: json!({ "path": "notes.md" }),
            succeeded: true,
            summary: "read 1 file".into(),
            elapsed_ms: 12,
        }],
        model_calls: 2,
        elapsed_ms: 350,
    }
}

#[tokio::test]
async fn noop_learning_sink_accepts_turns_and_transcripts_without_effect() {
    let sink = NoopLearningSink::new();
    let record = sample_turn_record();
    let run = run_id();
    let thread = thread_id("t1");

    sink.on_turn_completed(STATE, &record).await.unwrap();
    sink.on_transcript_committed(
        STATE,
        &TranscriptCommit {
            run_id: &run,
            thread_id: Some(&thread),
            path: Path::new("/workspace/session_raw/t1.jsonl"),
            appended_messages: 4,
        },
    )
    .await
    .unwrap();
}

#[test]
fn turn_record_round_trips_through_serde() {
    let record = sample_turn_record();
    let decoded: TurnRecord =
        serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();
    assert_eq!(decoded, record);
}

// ---------------------------------------------------------------------------
// ProgressSink — NoopProgressSink
// ---------------------------------------------------------------------------

#[tokio::test]
async fn noop_progress_sink_reports_disconnected_but_still_accepts_delivery() {
    let sink = NoopProgressSink::new();
    assert!(
        !ProgressSink::is_connected(&sink, STATE),
        "callers should take the skip-expensive-payload path"
    );

    let record = EventRecord {
        id: EventId::new("event-1"),
        offset: 0,
        event: AgentEvent::RunStarted {
            run_id: run_id(),
            thread_id: None,
        },
    };
    sink.deliver(STATE, &record).await.unwrap();
}

// ---------------------------------------------------------------------------
// ToolOutcomeClassifier — NoopToolOutcomeClassifier
// ---------------------------------------------------------------------------

#[test]
fn noop_classifier_leaves_every_failure_unclassified() {
    let classifier = NoopToolOutcomeClassifier::new();
    let call = CallId::new("call-1");

    let classified = ToolOutcomeClassifier::classify(
        &classifier,
        STATE,
        &ToolFailureContext {
            call_id: &call,
            tool_name: "http_request",
            error: "connection reset by peer",
            timed_out: false,
        },
    );

    assert!(classified.is_none());
}

#[test]
fn retry_disposition_defaults_to_unknown_and_unknown_is_not_retryable() {
    assert_eq!(RetryDisposition::default(), RetryDisposition::Unknown);
    assert!(!RetryDisposition::Unknown.is_retryable());
    assert!(!RetryDisposition::Never.is_retryable());
    assert!(RetryDisposition::Immediate.is_retryable());
    assert!(RetryDisposition::Backoff.is_retryable());
}

#[test]
fn tool_failure_round_trips_and_defaults_its_retry_disposition() {
    let failure = ToolFailure {
        class: "host_timeout".into(),
        category: "recoverable".into(),
        cause: "the request exceeded its deadline".into(),
        next_action: "retry with a narrower query".into(),
        retry: RetryDisposition::Backoff,
    };

    let encoded = serde_json::to_value(&failure).unwrap();
    assert_eq!(encoded["retry"], "backoff");
    let decoded: ToolFailure = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, failure);

    let legacy: ToolFailure = serde_json::from_value(json!({
        "class": "x",
        "category": "y",
        "cause": "c",
        "next_action": "n"
    }))
    .unwrap();
    assert_eq!(legacy.retry, RetryDisposition::Unknown);
}

// ---------------------------------------------------------------------------
// ModelResolver — StaticModelResolver
// ---------------------------------------------------------------------------

#[test]
fn static_resolver_builds_a_registry_whose_default_is_the_configured_model() {
    let model: Arc<dyn ChatModel<()>> =
        Arc::new(ScriptedModel::new(vec![ModelResponse::assistant("hello")]));
    let resolver = StaticModelResolver::new("scripted", model);
    let run = run_id();

    let registry = resolver
        .resolve(
            STATE,
            &ModelResolution {
                run_id: &run,
                agent_id: "agent",
                workload: "chat",
                pinned_model: None,
                temperature: Some(0.2),
            },
        )
        .unwrap();

    assert!(registry.get("scripted").is_some());
    assert!(
        registry.default_model().is_some(),
        "the first registered model becomes the registry default"
    );
    assert_eq!(resolver.name(), "scripted");
}

#[test]
fn static_resolver_reports_unknown_capabilities_rather_than_guessing() {
    let model: Arc<dyn ChatModel<()>> = Arc::new(ScriptedModel::new(Vec::new()));
    let resolver = StaticModelResolver::new("scripted", model);

    assert!(resolver.profile(STATE, "scripted").unwrap().is_none());
    assert!(
        resolver
            .context_window(STATE, "scripted")
            .unwrap()
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// Object safety — every seam must be usable behind `Arc<dyn _>`
// ---------------------------------------------------------------------------

#[test]
fn every_seam_is_usable_as_a_trait_object() {
    let _memory: Arc<dyn MemoryProvider<()>> = Arc::new(InMemoryMemoryProvider::new());
    let _context: Arc<dyn ContextComposer<()>> = Arc::new(PassthroughContextComposer::new());
    let _security: Arc<dyn SecurityGate<()>> = Arc::new(RootContainedSecurityGate::new());
    let _budget: Arc<dyn BudgetGate<()>> = Arc::new(UnmeteredBudgetGate::new());
    let _definitions: Arc<dyn DefinitionRegistry<()>> = Arc::new(InMemoryDefinitionRegistry::new());
    let _experience: Arc<dyn ExperienceStore<()>> = Arc::new(NoopExperienceStore::new());
    let _learning: Arc<dyn LearningSink<()>> = Arc::new(NoopLearningSink::new());
    let _progress: Arc<dyn ProgressSink<()>> = Arc::new(NoopProgressSink::new());
    let _classifier: Arc<dyn ToolOutcomeClassifier<()>> = Arc::new(NoopToolOutcomeClassifier);

    let model: Arc<dyn ChatModel<()>> = Arc::new(ScriptedModel::new(Vec::new()));
    let _resolver: Arc<dyn ModelResolver<()>> = Arc::new(StaticModelResolver::new("m", model));
}

#[tokio::test]
async fn trait_object_futures_are_send_so_seams_can_be_spawned() {
    let memory: Arc<dyn MemoryProvider<()> + 'static> = Arc::new(InMemoryMemoryProvider::new());

    let handle = tokio::spawn(async move {
        memory
            .recall(&(), &MemoryQuery::new("anything"))
            .await
            .unwrap()
            .len()
    });

    assert_eq!(handle.await.unwrap(), 0);
}
