//! Unit tests for shared-workspace claim arbitration.

use super::*;

fn p(s: &str) -> PathBuf {
    PathBuf::from(s)
}

// --- parse_relative_claim_paths -----------------------------------------

#[test]
fn parses_comma_and_newline_separated_entries() {
    let parsed = parse_relative_claim_paths("src/a.rs, src/b.rs\nsrc/c.rs").expect("parses");
    assert_eq!(parsed, vec![p("src/a.rs"), p("src/b.rs"), p("src/c.rs")]);
}

#[test]
fn strips_bullet_markers_and_surrounding_space() {
    let parsed = parse_relative_claim_paths("  - src/a.rs \n * src/b.rs").expect("parses");
    assert_eq!(parsed, vec![p("src/a.rs"), p("src/b.rs")]);
}

#[test]
fn sorts_and_deduplicates() {
    let parsed = parse_relative_claim_paths("src/b.rs, src/a.rs, src/b.rs").expect("parses");
    assert_eq!(parsed, vec![p("src/a.rs"), p("src/b.rs")]);
}

#[test]
fn skips_empty_entries_so_trailing_separators_are_harmless() {
    let parsed = parse_relative_claim_paths("src/a.rs,,\n  \n").expect("parses");
    assert_eq!(parsed, vec![p("src/a.rs")]);
}

#[test]
fn empty_spec_yields_no_paths() {
    assert_eq!(parse_relative_claim_paths("").expect("parses"), Vec::<PathBuf>::new());
}

#[test]
fn rejects_absolute_paths() {
    let err = parse_relative_claim_paths("/etc/passwd").expect_err("absolute is rejected");
    assert_eq!(err, ClaimPathError::Absolute { raw: "/etc/passwd".to_string() });
}

#[test]
fn rejects_parent_dir_escapes() {
    let err = parse_relative_claim_paths("../outside.rs").expect_err("escape is rejected");
    assert_eq!(err, ClaimPathError::Escaping { raw: "../outside.rs".to_string() });
}

#[test]
fn rejects_escape_hidden_mid_path() {
    let err = parse_relative_claim_paths("src/../../outside.rs").expect_err("escape is rejected");
    assert!(matches!(err, ClaimPathError::Escaping { .. }));
}

// --- paths_overlap -------------------------------------------------------

#[test]
fn identical_paths_overlap() {
    assert!(paths_overlap(&p("src/a.rs"), &p("src/a.rs")));
}

#[test]
fn a_directory_overlaps_a_file_beneath_it() {
    assert!(paths_overlap(&p("src"), &p("src/a.rs")));
    assert!(paths_overlap(&p("src/a.rs"), &p("src")));
}

#[test]
fn sibling_paths_sharing_a_textual_prefix_do_not_overlap() {
    // The classic bug: `starts_with` on strings would call these an overlap.
    // `Path::starts_with` is component-wise, and this pins that.
    assert!(!paths_overlap(&p("src/a"), &p("src/ab")));
    assert!(!paths_overlap(&p("src/ab"), &p("src/a")));
    assert!(!paths_overlap(&p("a.rs"), &p("ab.rs")));
}

#[test]
fn unrelated_paths_do_not_overlap() {
    assert!(!paths_overlap(&p("src/a.rs"), &p("docs/b.md")));
}

// --- writes_shared_workspace --------------------------------------------

#[test]
fn read_only_never_writes_even_when_other_flags_are_set() {
    let effects = ToolSideEffects {
        read_only: true,
        writes_files: true,
        destructive: true,
        ..Default::default()
    };
    assert!(!writes_shared_workspace(&effects));
}

#[test]
fn file_writes_installs_and_destructive_all_count_as_writing() {
    for effects in [
        ToolSideEffects { writes_files: true, ..Default::default() },
        ToolSideEffects { installs_dependencies: true, ..Default::default() },
        ToolSideEffects { destructive: true, ..Default::default() },
    ] {
        assert!(writes_shared_workspace(&effects), "{effects:?} should count as writing");
    }
}

#[test]
fn network_and_payment_alone_do_not_touch_the_workspace() {
    let effects = ToolSideEffects {
        network: true,
        payment: true,
        external_service: true,
        ..Default::default()
    };
    assert!(!writes_shared_workspace(&effects));
}

#[test]
fn default_effects_do_not_write() {
    assert!(!writes_shared_workspace(&ToolSideEffects::default()));
}

// --- plan_shared_workspace_dispatch -------------------------------------

#[test]
fn isolated_and_read_only_workers_all_run_in_parallel() {
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::isolated("w1"),
        WorkspaceClaim::read_only("w2"),
    ]);
    assert_eq!(
        plan.modes,
        vec![Some(DispatchMode::Parallel), Some(DispatchMode::Parallel)]
    );
    assert!(plan.conflicts.is_empty());
    assert!(!plan.has_serial_work());
    assert_eq!(plan.parallel_indices(), vec![0, 1]);
    assert!(plan.serial_indices().is_empty());
}

#[test]
fn an_isolated_worker_that_writes_still_runs_in_parallel() {
    // Isolation short-circuits the write check: its root is its own.
    let mut claim = WorkspaceClaim::isolated("w1");
    claim.writes = true;
    let plan = plan_shared_workspace_dispatch(&[claim]);
    assert_eq!(plan.modes, vec![Some(DispatchMode::Parallel)]);
    assert!(plan.conflicts.is_empty());
}

#[test]
fn a_shared_writer_with_a_claim_is_serialized() {
    let plan = plan_shared_workspace_dispatch(&[WorkspaceClaim::writing("w1", vec![p("src/a.rs")])]);
    assert_eq!(plan.modes, vec![Some(DispatchMode::Serial)]);
    assert!(plan.conflicts.is_empty());
    assert!(plan.has_serial_work());
    assert_eq!(plan.serial_indices(), vec![0]);
}

#[test]
fn a_shared_writer_without_a_claim_is_rejected_as_unbounded() {
    let plan = plan_shared_workspace_dispatch(&[WorkspaceClaim::writing("w1", vec![])]);
    assert_eq!(plan.modes, vec![None]);
    assert_eq!(
        plan.conflicts,
        vec![(0, ClaimConflict::UnboundedWrite { worker_id: "w1".to_string() })]
    );
}

#[test]
fn two_writers_with_disjoint_claims_are_both_serialized() {
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("w1", vec![p("src/a.rs")]),
        WorkspaceClaim::writing("w2", vec![p("src/b.rs")]),
    ]);
    assert_eq!(
        plan.modes,
        vec![Some(DispatchMode::Serial), Some(DispatchMode::Serial)]
    );
    assert!(plan.conflicts.is_empty());
    assert_eq!(plan.serial_indices(), vec![0, 1]);
}

#[test]
fn on_overlap_the_later_worker_is_rejected_and_the_earlier_keeps_its_claim() {
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("w1", vec![p("src/a.rs")]),
        WorkspaceClaim::writing("w2", vec![p("src/a.rs")]),
    ]);
    assert_eq!(plan.modes, vec![Some(DispatchMode::Serial), None]);
    assert_eq!(
        plan.conflicts,
        vec![(
            1,
            ClaimConflict::Overlap {
                worker_id: "w2".to_string(),
                other_worker_id: "w1".to_string(),
                path: p("src/a.rs"),
            }
        )]
    );
}

#[test]
fn overlap_is_detected_through_directory_containment() {
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("w1", vec![p("src")]),
        WorkspaceClaim::writing("w2", vec![p("src/a.rs")]),
    ]);
    assert_eq!(plan.modes, vec![Some(DispatchMode::Serial), None]);
    assert_eq!(plan.conflicts.len(), 1);
}

#[test]
fn a_rejected_worker_does_not_reserve_its_paths_for_later_workers() {
    // w2 is rejected against w1, so w2's *other* path stays free for w3.
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("w1", vec![p("src/a.rs")]),
        WorkspaceClaim::writing("w2", vec![p("src/a.rs"), p("src/z.rs")]),
        WorkspaceClaim::writing("w3", vec![p("src/z.rs")]),
    ]);
    assert_eq!(
        plan.modes,
        vec![Some(DispatchMode::Serial), None, Some(DispatchMode::Serial)]
    );
    assert_eq!(plan.conflicts.len(), 1);
    assert_eq!(plan.conflicts[0].0, 1);
}

#[test]
fn rejection_follows_input_order_not_claim_content() {
    // Swapping the inputs swaps who is rejected — the plan is a pure function
    // of input order, which is what makes it reproducible across runs.
    let forward = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("first", vec![p("shared.rs")]),
        WorkspaceClaim::writing("second", vec![p("shared.rs")]),
    ]);
    let reversed = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::writing("second", vec![p("shared.rs")]),
        WorkspaceClaim::writing("first", vec![p("shared.rs")]),
    ]);
    assert_eq!(forward.conflicts[0].1.worker_id(), "second");
    assert_eq!(reversed.conflicts[0].1.worker_id(), "first");
}

#[test]
fn mixed_batches_keep_every_index_aligned_with_the_input() {
    let plan = plan_shared_workspace_dispatch(&[
        WorkspaceClaim::isolated("isolated"),
        WorkspaceClaim::read_only("reader"),
        WorkspaceClaim::writing("writer", vec![p("src/a.rs")]),
        WorkspaceClaim::writing("clasher", vec![p("src/a.rs")]),
        WorkspaceClaim::writing("unbounded", vec![]),
    ]);
    assert_eq!(
        plan.modes,
        vec![
            Some(DispatchMode::Parallel),
            Some(DispatchMode::Parallel),
            Some(DispatchMode::Serial),
            None,
            None,
        ]
    );
    assert_eq!(plan.parallel_indices(), vec![0, 1]);
    assert_eq!(plan.serial_indices(), vec![2]);
    assert_eq!(
        plan.conflicts.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        vec![3, 4]
    );
}

#[test]
fn an_empty_batch_plans_nothing() {
    let plan = plan_shared_workspace_dispatch(&[]);
    assert_eq!(plan, DispatchPlan::default());
    assert!(!plan.has_serial_work());
}

#[test]
fn claim_conflict_reports_the_worker_it_rejected() {
    let unbounded = ClaimConflict::UnboundedWrite { worker_id: "w1".to_string() };
    assert_eq!(unbounded.worker_id(), "w1");
    let overlap = ClaimConflict::Overlap {
        worker_id: "w2".to_string(),
        other_worker_id: "w1".to_string(),
        path: p("src/a.rs"),
    };
    assert_eq!(overlap.worker_id(), "w2");
}
