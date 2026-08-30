//! Unit tests for the delegation graph: the revision loop and its budget,
//! cancellation, durable checkpoint/resume classification, the human-approval
//! interrupt, and the on-disk shape of [`DelegationState`].

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::run::is_incompatible_checkpoint_error;
use super::*;
use crate::CancellationToken;
use crate::graph::Interrupt;
use crate::graph::checkpoint::{Checkpoint, Checkpointer};

/// A reviewer that rejects the first `reject_first` executions, then approves,
/// driving the execute⇄review revision loop.
fn flow_runner(
    reject_first: usize,
) -> impl Fn(
    DelegationStage,
    DelegationState,
) -> std::pin::Pin<Box<dyn Future<Output = Result<DelegationStageOutput, String>> + Send>>
+ Clone
+ Send
+ Sync
+ 'static {
    let reviews = Arc::new(AtomicUsize::new(0));
    move |stage, _state| {
        let reviews = reviews.clone();
        Box::pin(async move {
            match stage {
                DelegationStage::Plan => Ok(DelegationStageOutput::done("PLAN")),
                DelegationStage::Execute => Ok(DelegationStageOutput::done("EXEC")),
                DelegationStage::Review => {
                    let n = reviews.fetch_add(1, Ordering::SeqCst);
                    Ok(DelegationStageOutput {
                        text: format!("review-{n}"),
                        approved: n >= reject_first,
                        prompt: None,
                    })
                }
            }
        })
    }
}

#[tokio::test]
async fn approves_first_pass_no_revision() {
    let state = run_delegation(DelegationConfig::default(), flow_runner(0))
        .await
        .expect("runs");
    assert_eq!(state.plan.as_deref(), Some("PLAN"));
    assert_eq!(state.executions.len(), 1, "one execution, no revision");
    assert_eq!(state.executions[0].index, 0);
    assert_eq!(state.executions[0].result, "EXEC");
    assert_eq!(state.reviews.len(), 1);
    assert_eq!(state.revisions, 0);
    assert!(state.approved);
    assert_eq!(state.final_output.as_deref(), Some("EXEC"));
}

#[tokio::test]
async fn revises_then_approves() {
    // Reject the first review → one revision (a second execute+review).
    let state = run_delegation(DelegationConfig::default(), flow_runner(1))
        .await
        .expect("runs");
    assert_eq!(state.executions.len(), 2, "initial + one revised execution");
    assert_eq!(state.reviews.len(), 2);
    assert_eq!(state.revisions, 1);
    assert!(state.approved);
}

#[tokio::test]
async fn revision_budget_caps_a_never_approving_reviewer() {
    // Reviewer never approves on its own; the max_revisions cap forces finalize.
    let config = DelegationConfig {
        max_revisions: 2,
        ..DelegationConfig::default()
    };
    let state = run_delegation(config, flow_runner(999))
        .await
        .expect("runs");
    // revisions counted: 1st review (rev 1), 2nd review (rev 2), 3rd review
    // hits revisions>=2 → forced approve. So 3 executions, 3 reviews.
    assert_eq!(state.revisions, 2, "stops at the revision budget");
    assert!(state.approved, "forced-approved at the cap");
    assert_eq!(state.executions.len(), 3);
}

#[tokio::test]
async fn cancellation_short_circuits_to_finalize() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let ran = Arc::new(Mutex::new(Vec::<DelegationStage>::new()));
    let ran2 = ran.clone();
    let runner = move |stage: DelegationStage, _s: DelegationState| {
        let ran = ran2.clone();
        Box::pin(async move {
            ran.lock().unwrap().push(stage);
            Ok::<_, String>(DelegationStageOutput::done("X"))
        }) as std::pin::Pin<Box<dyn Future<Output = _> + Send>>
    };
    let config = DelegationConfig {
        cancel,
        ..DelegationConfig::default()
    };
    let state = run_delegation(config, runner).await.expect("runs");
    assert!(state.cancelled, "state flagged cancelled");
    assert!(state.final_output.is_some());
    assert!(
        ran.lock().unwrap().is_empty(),
        "no stage worker ran once cancelled at the plan boundary"
    );
}

#[tokio::test]
async fn human_gated_run_parks_on_interrupt_then_resume_approves() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let make_config = || DelegationConfig {
        require_review_approval: true,
        checkpointer: Some(cp.clone()),
        thread_id: Some("hg-approve".to_string()),
        ..DelegationConfig::default()
    };

    // First pass: reviewer approves on the first review, so the run reaches
    // the durable human-approval gate and parks on an interrupt.
    let outcome = run_delegation_durable(make_config(), flow_runner(0))
        .await
        .expect("runs");
    let pending = outcome.pending.expect("parked on the approval interrupt");
    assert_eq!(pending.node, "approval");
    assert_eq!(pending.thread_id, "hg-approve");
    assert!(
        outcome.state.final_output.is_none(),
        "must not finalize while paused for human approval"
    );

    // Simulated process restart: `resume_delegation` rebuilds a fresh graph
    // from the same checkpointer + thread and re-enters via Command::resume.
    let resumed = resume_delegation(make_config(), json!("approve_once"), flow_runner(0))
        .await
        .expect("resumes");
    assert!(resumed.pending.is_none(), "resume clears the pause");
    assert_eq!(resumed.state.human_approved, Some(true));
    assert!(!resumed.state.denied);
    assert!(
        resumed.state.final_output.is_some(),
        "resumes from checkpoint to finalize"
    );
}

#[tokio::test]
async fn human_gated_without_checkpointer_or_thread_id_is_rejected() {
    // `require_review_approval` documents that it needs both a checkpointer
    // and a thread_id (interrupts require durability). Without this guard the
    // run would still park on the approval interrupt with nothing persisted,
    // filling `PendingApproval::thread_id` with an empty string and stranding
    // the pause forever (`resume_delegation` has no checkpoint/thread to
    // resume from). Both misconfigurations must be rejected up front.
    let neither = DelegationConfig {
        require_review_approval: true,
        ..DelegationConfig::default()
    };
    let err = run_delegation_durable(neither, flow_runner(0))
        .await
        .expect_err("must reject: no checkpointer, no thread_id");
    assert!(
        err.contains("require_review_approval"),
        "error should name the misconfigured field: {err}"
    );

    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let checkpointer_only = DelegationConfig {
        require_review_approval: true,
        checkpointer: Some(cp),
        thread_id: None,
        ..DelegationConfig::default()
    };
    run_delegation_durable(checkpointer_only, flow_runner(0))
        .await
        .expect_err("must reject: checkpointer without a thread_id");

    let thread_only = DelegationConfig {
        require_review_approval: true,
        checkpointer: None,
        thread_id: Some("hg-no-cp".to_string()),
        ..DelegationConfig::default()
    };
    run_delegation_durable(thread_only, flow_runner(0))
        .await
        .expect_err("must reject: thread_id without a checkpointer");
}

#[tokio::test]
async fn ttl_expiry_resume_with_deny_blocks_the_result() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let make_config = || DelegationConfig {
        require_review_approval: true,
        checkpointer: Some(cp.clone()),
        thread_id: Some("hg-deny".to_string()),
        ..DelegationConfig::default()
    };

    let outcome = run_delegation_durable(make_config(), flow_runner(0))
        .await
        .expect("runs");
    assert!(outcome.pending.is_some(), "parks awaiting approval");

    // TTL expiry → resume-with-deny preserves the timeout-deny behavior.
    let resumed = resume_delegation(make_config(), deny_decision(), flow_runner(0))
        .await
        .expect("resumes");
    assert_eq!(resumed.state.human_approved, Some(false));
    assert!(resumed.state.denied, "deny is honoured as a blocked result");
    assert!(
        !resumed.state.approved,
        "human deny overrides the reviewer's in-graph approval"
    );
    assert!(
        resumed
            .state
            .final_output
            .as_deref()
            .unwrap_or_default()
            .contains("denied")
    );
}

#[tokio::test]
async fn durable_checkpointer_persists_thread_state() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp.clone()),
        thread_id: Some("run-1".to_string()),
        ..DelegationConfig::default()
    };
    let state = run_delegation(config, flow_runner(1)).await.expect("runs");
    assert!(state.approved);
    // The checkpointer recorded the run under its thread id.
    let threads = cp.list_threads().await.expect("list threads");
    assert!(
        threads.iter().any(|t| t == "run-1"),
        "thread persisted, saw {threads:?}"
    );
    let checkpoints = cp.list("run-1").await.expect("list checkpoints");
    assert!(
        !checkpoints.is_empty(),
        "at least one super-step boundary checkpoint persisted"
    );
}

/// The boxed-future type every inline test runner returns.
type BoxedStageFut =
    std::pin::Pin<Box<dyn Future<Output = Result<DelegationStageOutput, String>> + Send>>;

#[tokio::test]
async fn execution_records_capture_index_and_prompt() {
    // A worker that surfaces an execute prompt; assert the per-step record
    // carries index + prompt + result (issue #3884).
    let runner = move |stage: DelegationStage, _s: DelegationState| {
        Box::pin(async move {
            match stage {
                DelegationStage::Plan => Ok(DelegationStageOutput::done("PLAN")),
                DelegationStage::Execute => Ok(DelegationStageOutput {
                    text: "EXEC".to_string(),
                    approved: true,
                    prompt: Some("EXEC-PROMPT".to_string()),
                }),
                DelegationStage::Review => Ok(DelegationStageOutput {
                    text: "APPROVE".to_string(),
                    approved: true,
                    prompt: None,
                }),
            }
        }) as BoxedStageFut
    };
    let state = run_delegation(DelegationConfig::default(), runner)
        .await
        .expect("runs");
    assert_eq!(state.executions.len(), 1);
    assert_eq!(state.executions[0].index, 0);
    assert_eq!(state.executions[0].result, "EXEC");
    assert_eq!(state.executions[0].prompt, "EXEC-PROMPT");
}

#[test]
fn legacy_string_executions_do_not_deserialize_into_step_records() {
    // A pre-#3884 checkpoint stored `executions` as a flat string array. It
    // must fail to load into the new `Vec<StepRecord>` — which is exactly
    // what makes `run_or_resume_delegation` expire the stale checkpoint
    // instead of misreading it.
    let legacy = r#"{"plan":"P","executions":["raw a","raw b"],"reviews":[],"revisions":0,"approved":true,"final_output":null,"cancelled":false}"#;
    assert!(serde_json::from_str::<DelegationState>(legacy).is_err());
}

#[test]
fn schema_version_defaults_to_zero_and_step_records_round_trip() {
    // A pre-versioned record (no `schema_version`) loads as version 0.
    let unversioned = r#"{"plan":null,"executions":[],"reviews":[],"revisions":0,"approved":false,"final_output":null,"cancelled":false}"#;
    let s: DelegationState = serde_json::from_str(unversioned).expect("loads");
    assert_eq!(s.schema_version, 0);

    // A fresh run stamps the current version and round-trips step records.
    let mut fresh = DelegationState::new_run();
    assert_eq!(fresh.schema_version, CURRENT_SCHEMA_VERSION);
    fresh.executions.push(StepRecord {
        index: 0,
        prompt: "P".to_string(),
        result: "R".to_string(),
    });
    let json = serde_json::to_string(&fresh).expect("serializes");
    let back: DelegationState = serde_json::from_str(&json).expect("round-trips");
    assert_eq!(back.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(back.executions[0].result, "R");
    assert_eq!(back.executions[0].prompt, "P");
}

#[tokio::test]
async fn run_or_resume_starts_fresh_without_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp),
        thread_id: Some("fresh-1".to_string()),
        ..DelegationConfig::default()
    };
    let outcome = run_or_resume_delegation(config, flow_runner(0))
        .await
        .expect("runs fresh");
    assert!(outcome.state.final_output.is_some());
    assert_eq!(outcome.state.executions.len(), 1);
    assert_eq!(outcome.state.schema_version, CURRENT_SCHEMA_VERSION);
}

#[tokio::test]
async fn run_or_resume_continues_from_last_boundary_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    // Count stage invocations across BOTH the crashed run and the resume, to
    // prove plan/execute are NOT re-run on resume.
    let plan_runs = Arc::new(AtomicUsize::new(0));
    let exec_runs = Arc::new(AtomicUsize::new(0));
    let review_runs = Arc::new(AtomicUsize::new(0));
    let make_runner = |crash_first_review: bool| {
        let plan_runs = plan_runs.clone();
        let exec_runs = exec_runs.clone();
        let review_runs = review_runs.clone();
        move |stage: DelegationStage, _s: DelegationState| {
            let plan_runs = plan_runs.clone();
            let exec_runs = exec_runs.clone();
            let review_runs = review_runs.clone();
            Box::pin(async move {
                match stage {
                    DelegationStage::Plan => {
                        plan_runs.fetch_add(1, Ordering::SeqCst);
                        Ok(DelegationStageOutput::done("PLAN"))
                    }
                    DelegationStage::Execute => {
                        exec_runs.fetch_add(1, Ordering::SeqCst);
                        Ok(DelegationStageOutput {
                            text: "EXEC".to_string(),
                            approved: true,
                            prompt: Some("EXEC-PROMPT".to_string()),
                        })
                    }
                    DelegationStage::Review => {
                        let n = review_runs.fetch_add(1, Ordering::SeqCst);
                        if crash_first_review && n == 0 {
                            Err("simulated crash during review".to_string())
                        } else {
                            Ok(DelegationStageOutput {
                                text: "APPROVE".to_string(),
                                approved: true,
                                prompt: None,
                            })
                        }
                    }
                }
            }) as BoxedStageFut
        }
    };
    let make_config = || DelegationConfig {
        checkpointer: Some(cp.clone()),
        thread_id: Some("resume-1".to_string()),
        ..DelegationConfig::default()
    };

    // Run 1: crashes during the first review. plan + execute completed and
    // were checkpointed at their super-step boundaries; the run returns Err.
    let crashed = run_or_resume_delegation(make_config(), make_runner(true)).await;
    assert!(crashed.is_err(), "first run crashes at review");
    assert_eq!(plan_runs.load(Ordering::SeqCst), 1);
    assert_eq!(exec_runs.load(Ordering::SeqCst), 1);

    // Run 2 (resume): a resumable checkpoint exists → re-enter at the pending
    // node (review), NOT plan; plan/execute must not run again.
    let resumed = run_or_resume_delegation(make_config(), make_runner(false))
        .await
        .expect("resumes");
    assert!(
        resumed.state.final_output.is_some(),
        "resumed run finalizes"
    );
    assert_eq!(
        plan_runs.load(Ordering::SeqCst),
        1,
        "plan not re-run on resume"
    );
    assert_eq!(
        exec_runs.load(Ordering::SeqCst),
        1,
        "execute not re-run on resume"
    );
    assert_eq!(
        resumed.state.executions.len(),
        1,
        "the pre-crash execution survived; no duplicate"
    );
    assert_eq!(resumed.state.executions[0].result, "EXEC");
}

#[tokio::test]
async fn run_or_resume_is_idempotent_on_a_finalized_thread() {
    let dir = tempfile::tempdir().unwrap();
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::new(dir.path()));
    let stage_runs = Arc::new(AtomicUsize::new(0));
    let make_runner = || {
        let stage_runs = stage_runs.clone();
        move |stage: DelegationStage, _s: DelegationState| {
            let stage_runs = stage_runs.clone();
            Box::pin(async move {
                stage_runs.fetch_add(1, Ordering::SeqCst);
                match stage {
                    DelegationStage::Review => Ok(DelegationStageOutput {
                        text: "APPROVE".to_string(),
                        approved: true,
                        prompt: None,
                    }),
                    _ => Ok(DelegationStageOutput::done("X")),
                }
            }) as BoxedStageFut
        }
    };
    let make_config = || DelegationConfig {
        checkpointer: Some(cp.clone()),
        thread_id: Some("done-1".to_string()),
        ..DelegationConfig::default()
    };

    let first = run_or_resume_delegation(make_config(), make_runner())
        .await
        .expect("runs");
    assert!(first.state.final_output.is_some());
    let after_first = stage_runs.load(Ordering::SeqCst);
    assert!(after_first > 0, "the first run invoked stage workers");

    // Re-invoke the SAME thread: a terminal checkpoint → return the finalized
    // state without re-running any stage worker.
    let second = run_or_resume_delegation(make_config(), make_runner())
        .await
        .expect("idempotent");
    assert!(second.state.final_output.is_some());
    assert_eq!(
        stage_runs.load(Ordering::SeqCst),
        after_first,
        "no stage worker re-ran on a finalized thread"
    );
}

#[tokio::test]
async fn incompatible_checkpoint_expires_to_a_fresh_run() {
    // A pre-#3884 checkpoint (executions as `Vec<String>`) left in the store
    // must not crash a resume: `run_or_resume_delegation` expires it and runs
    // fresh, never panicking or surfacing the decode error.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct LegacyState {
        plan: Option<String>,
        executions: Vec<String>,
        reviews: Vec<String>,
        revisions: usize,
        approved: bool,
        final_output: Option<String>,
        cancelled: bool,
    }
    let dir = tempfile::tempdir().unwrap();
    // Seed the store as the OLD state type under the thread.
    let legacy_cp: crate::graph::checkpoint::FileCheckpointer<LegacyState> =
        crate::graph::checkpoint::FileCheckpointer::new(dir.path());
    let legacy = Checkpoint {
        thread_id: "legacy-1".to_string(),
        checkpoint_id: "cp-legacy".to_string(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state: LegacyState {
            plan: Some("old".to_string()),
            executions: vec!["a".to_string(), "b".to_string()],
            reviews: vec![],
            revisions: 0,
            approved: true,
            final_output: None,
            cancelled: false,
        },
        next_nodes: vec![],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: json!({}),
    };
    legacy_cp.put(legacy).await.expect("seed legacy checkpoint");

    // Reopen the SAME store as the current state type and resume: the
    // undecodable record is expired and a fresh run completes.
    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::<DelegationState>::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp),
        thread_id: Some("legacy-1".to_string()),
        ..DelegationConfig::default()
    };
    let outcome = run_or_resume_delegation(config, flow_runner(0))
        .await
        .expect("expires stale checkpoint and runs fresh");
    assert!(outcome.state.final_output.is_some(), "fresh run completed");
    assert_eq!(outcome.state.executions.len(), 1);
    assert_eq!(outcome.state.executions[0].result, "EXEC");
}

#[tokio::test]
async fn checkpoint_below_current_schema_version_expires_to_fresh_run() {
    // A decodable but OLD-schema checkpoint (schema_version defaults to 0 —
    // e.g. a pre-#3884 record whose executions happened to be empty) must be
    // expired, not resumed: `schema_version` is a real guard, not just a doc.
    let dir = tempfile::tempdir().unwrap();
    let seed: crate::graph::checkpoint::FileCheckpointer<DelegationState> =
        crate::graph::checkpoint::FileCheckpointer::new(dir.path());
    let checkpoint = Checkpoint {
        thread_id: "old-schema".to_string(),
        checkpoint_id: "cp-old".to_string(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state: DelegationState {
            plan: Some("stale".to_string()),
            ..Default::default()
        },
        next_nodes: vec![],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: json!({}),
    };
    assert_eq!(
        checkpoint.state.schema_version, 0,
        "an un-stamped record is version 0"
    );
    seed.put(checkpoint)
        .await
        .expect("seed old-schema checkpoint");

    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::<DelegationState>::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp),
        thread_id: Some("old-schema".to_string()),
        ..DelegationConfig::default()
    };
    let outcome = run_or_resume_delegation(config, flow_runner(0))
        .await
        .expect("expires + fresh");
    assert!(outcome.state.final_output.is_some(), "fresh run completed");
    assert_eq!(
        outcome.state.schema_version, CURRENT_SCHEMA_VERSION,
        "fresh run stamped the current version"
    );
    assert_eq!(
        outcome.state.plan.as_deref(),
        Some("PLAN"),
        "re-planned from scratch, not resumed with the stale plan"
    );
}

#[tokio::test]
async fn checkpoint_above_current_schema_version_also_expires_to_fresh_run() {
    // A checkpoint stamped with a schema version NEWER than this binary
    // understands (e.g. written by a newer binary during a rollback, or in a
    // mixed-version deployment) must be expired exactly like an older one, not
    // resumed. Serde can decode a newer record just fine by ignoring fields it
    // doesn't recognize, so an inequality check — not merely `<` — is what
    // catches this: resuming it under this binary's older semantics would be
    // silently wrong rather than loudly incompatible.
    let dir = tempfile::tempdir().unwrap();
    let seed: crate::graph::checkpoint::FileCheckpointer<DelegationState> =
        crate::graph::checkpoint::FileCheckpointer::new(dir.path());
    let checkpoint = Checkpoint {
        thread_id: "future-schema".to_string(),
        checkpoint_id: "cp-future".to_string(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state: DelegationState {
            plan: Some("from-the-future".to_string()),
            schema_version: CURRENT_SCHEMA_VERSION + 1,
            ..Default::default()
        },
        next_nodes: vec![],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: json!({}),
    };
    seed.put(checkpoint)
        .await
        .expect("seed future-schema checkpoint");

    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::<DelegationState>::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp),
        thread_id: Some("future-schema".to_string()),
        ..DelegationConfig::default()
    };
    let outcome = run_or_resume_delegation(config, flow_runner(0))
        .await
        .expect("expires + fresh");
    assert!(outcome.state.final_output.is_some(), "fresh run completed");
    assert_eq!(
        outcome.state.schema_version, CURRENT_SCHEMA_VERSION,
        "fresh run stamped the current version, not the future one"
    );
    assert_eq!(
        outcome.state.plan.as_deref(),
        Some("PLAN"),
        "re-planned from scratch, not resumed with the future-schema plan"
    );
}

#[test]
fn incompatible_checkpoint_error_matches_schema_not_corrupt_or_operational() {
    use crate::TinyAgentsError;
    // Positively identified schema mismatch → safe to expire.
    assert!(is_incompatible_checkpoint_error(
        &TinyAgentsError::Checkpoint(
            "sqlite checkpointer: decode [schema] record: invalid type: string".to_string()
        )
    ));
    assert!(is_incompatible_checkpoint_error(&TinyAgentsError::Checkpoint(
        "sqlite checkpointer: decode [schema] next_nodes: missing field `foo`".to_string()
    )));
    // Ambiguous/likely corruption must NOT be treated as a safe-to-expire
    // schema mismatch — deleting it would discard the only evidence of the
    // corruption and silently restart the delegation from `plan`, potentially
    // repeating already-completed execute-stage side effects.
    assert!(!is_incompatible_checkpoint_error(
        &TinyAgentsError::Checkpoint(
            "sqlite checkpointer: decode [corrupt] record: EOF while parsing a value".to_string()
        )
    ));
    assert!(!is_incompatible_checkpoint_error(&TinyAgentsError::Checkpoint(
        "file checkpointer: decode [corrupt] record: expected `,` or `}`".to_string()
    )));
    // Operational failures must NOT be treated as incompatible (they must
    // propagate, not silently restart durable work).
    assert!(!is_incompatible_checkpoint_error(
        &TinyAgentsError::Checkpoint(
            "sqlite checkpointer: query latest checkpoint: database is locked".to_string()
        )
    ));
    assert!(!is_incompatible_checkpoint_error(
        &TinyAgentsError::Checkpoint("sqlite checkpointer: connection lock poisoned".to_string())
    ));
    assert!(!is_incompatible_checkpoint_error(&TinyAgentsError::Resume(
        "no checkpoint".to_string()
    )));
}

#[test]
fn decode_json_err_classifies_data_errors_as_schema_and_others_as_corrupt() {
    use crate::graph::checkpoint::decode_json_err;

    // `Category::Data`: syntactically valid JSON that doesn't match the
    // target type — the shape a legacy or newer schema produces.
    let data_err = serde_json::from_str::<u32>("\"not a number\"").unwrap_err();
    assert_eq!(data_err.classify(), serde_json::error::Category::Data);
    let wrapped = decode_json_err("sqlite checkpointer", "record", data_err);
    assert!(
        format!("{wrapped}").contains("decode [schema] record"),
        "data-category decode errors must be tagged [schema]: {wrapped}"
    );

    // `Category::Eof`: the bytes were truncated — real corruption, not a
    // schema difference.
    let eof_err = serde_json::from_str::<serde_json::Value>("{\"a\":").unwrap_err();
    assert_eq!(eof_err.classify(), serde_json::error::Category::Eof);
    let wrapped = decode_json_err("file checkpointer", "record", eof_err);
    assert!(
        format!("{wrapped}").contains("decode [corrupt] record"),
        "eof-category decode errors must be tagged [corrupt]: {wrapped}"
    );

    // `Category::Syntax`: malformed bytes — also corruption, not a schema
    // difference.
    let syntax_err = serde_json::from_str::<serde_json::Value>("{not json}").unwrap_err();
    assert_eq!(syntax_err.classify(), serde_json::error::Category::Syntax);
    let wrapped = decode_json_err("sqlite checkpointer", "record", syntax_err);
    assert!(
        format!("{wrapped}").contains("decode [corrupt] record"),
        "syntax-category decode errors must be tagged [corrupt]: {wrapped}"
    );
}

#[tokio::test]
async fn terminal_checkpoint_with_a_pending_interrupt_surfaces_it() {
    // A terminal-classified checkpoint that still carries an interrupt must
    // surface it, not silently drop `pending`. (The live routing never
    // produces this shape; the terminal branch is defensive.)
    let dir = tempfile::tempdir().unwrap();
    let seed: crate::graph::checkpoint::FileCheckpointer<DelegationState> =
        crate::graph::checkpoint::FileCheckpointer::new(dir.path());
    let mut state = DelegationState::new_run();
    state.final_output = Some("done".to_string());
    let checkpoint = Checkpoint {
        thread_id: "terminal-interrupt".to_string(),
        checkpoint_id: "cp-ti".to_string(),
        run_id: None,
        parent_checkpoint_id: None,
        namespace: vec![],
        state,
        next_nodes: vec![],
        completed_tasks: vec![],
        pending_writes: vec![],
        interrupts: vec![Interrupt::with_id(
            "intr-1",
            "approval",
            json!({ "kind": "delegation_review" }),
        )],
        pending_activations: None,
        barrier_arrivals: vec![],
        metadata: json!({}),
    };
    seed.put(checkpoint).await.expect("seed terminal+interrupt");

    let cp: Arc<dyn Checkpointer<DelegationState>> =
        Arc::new(crate::graph::checkpoint::FileCheckpointer::<DelegationState>::new(dir.path()));
    let config = DelegationConfig {
        checkpointer: Some(cp),
        thread_id: Some("terminal-interrupt".to_string()),
        ..DelegationConfig::default()
    };
    let outcome = run_or_resume_delegation(config, flow_runner(0))
        .await
        .expect("terminal");
    let pending = outcome
        .pending
        .expect("the carried interrupt is surfaced, not dropped");
    assert_eq!(pending.node, "approval");
    assert_eq!(pending.interrupt_id, "intr-1");
    assert_eq!(outcome.state.final_output.as_deref(), Some("done"));
}

/// Pins the exact serialized shape of a fully-populated [`DelegationState`].
///
/// `DelegationState` is an **on-disk checkpoint format**: installations carry
/// persisted state written by an earlier release, and a resume decodes it with
/// whatever the current binary declares. A renamed field, a changed
/// `#[serde(...)]` attribute, a new field without `#[serde(default)]`, or a
/// reordering that changes the emitted JSON silently breaks resume for those
/// installations — the checkpoint decodes into something subtly different, or
/// fails to decode at all and is expired, discarding in-flight work.
///
/// The literal below is the byte-for-byte output of the implementation this
/// module was moved from, captured before the move. It is not merely a snapshot
/// of current behaviour: it is the compatibility contract. A change here must be
/// deliberate and paired with a [`CURRENT_SCHEMA_VERSION`] bump so
/// `run_or_resume_delegation` expires older checkpoints rather than misreading
/// them.
#[test]
fn serialized_state_shape_is_pinned() {
    let state = DelegationState {
        plan: Some("PLAN".into()),
        executions: vec![
            StepRecord {
                index: 0,
                prompt: "P0".into(),
                result: "R0".into(),
            },
            StepRecord {
                index: 1,
                prompt: "P1".into(),
                result: "R1".into(),
            },
        ],
        reviews: vec!["review-0".into(), "review-1".into()],
        revisions: 1,
        approved: true,
        final_output: Some("FINAL".into()),
        cancelled: false,
        human_approved: Some(true),
        denied: false,
        schema_version: CURRENT_SCHEMA_VERSION,
    };

    let json = serde_json::to_string(&state).expect("serializes");
    assert_eq!(
        json,
        r#"{"plan":"PLAN","executions":[{"index":0,"prompt":"P0","result":"R0"},{"index":1,"prompt":"P1","result":"R1"}],"reviews":["review-0","review-1"],"revisions":1,"approved":true,"final_output":"FINAL","cancelled":false,"human_approved":true,"denied":false,"schema_version":1}"#,
        "DelegationState's on-disk shape changed; see this test's doc comment"
    );

    // And it decodes back to an equal value, so the pin covers both directions.
    let back: DelegationState = serde_json::from_str(&json).expect("round-trips");
    assert_eq!(back.plan, state.plan);
    assert_eq!(back.executions, state.executions);
    assert_eq!(back.reviews, state.reviews);
    assert_eq!(back.revisions, state.revisions);
    assert_eq!(back.approved, state.approved);
    assert_eq!(back.final_output, state.final_output);
    assert_eq!(back.cancelled, state.cancelled);
    assert_eq!(back.human_approved, state.human_approved);
    assert_eq!(back.denied, state.denied);
    assert_eq!(back.schema_version, state.schema_version);
}

/// A pre-versioned checkpoint — one written before `schema_version`,
/// `human_approved` and `denied` existed — must still decode, taking the
/// documented defaults. This is the half of the contract that lets
/// `run_or_resume_delegation` *classify* a stale checkpoint (version `0`) rather
/// than failing to read it at all.
#[test]
fn pre_versioned_state_decodes_with_documented_defaults() {
    let legacy = r#"{"plan":"PLAN","executions":[],"reviews":[],"revisions":0,"approved":false,"final_output":null,"cancelled":false}"#;
    let state: DelegationState = serde_json::from_str(legacy).expect("decodes");
    assert_eq!(
        state.schema_version, 0,
        "defaults below CURRENT, so it expires"
    );
    assert_eq!(state.human_approved, None);
    assert!(!state.denied);
    assert!(state.schema_version < CURRENT_SCHEMA_VERSION);
}

/// The default state (a fresh, unstarted run) also has a pinned shape — it is
/// what the very first checkpoint of a run serializes from.
#[test]
fn default_state_shape_is_pinned() {
    let json = serde_json::to_string(&DelegationState::default()).expect("serializes");
    assert_eq!(
        json,
        r#"{"plan":null,"executions":[],"reviews":[],"revisions":0,"approved":false,"final_output":null,"cancelled":false,"human_approved":null,"denied":false,"schema_version":0}"#
    );
}
