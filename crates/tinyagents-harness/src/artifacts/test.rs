//! Tests for the artifact-offload convention.
//!
//! Covers the happy path (oversized result lands in `outputs/`, the parent gets
//! a path + abstract), the fallback path (offload refused, inline payload
//! survives for the host's backstop), and the fail-closed path hardening that
//! keeps offload inside the artifact root and out of host-internal state.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::*;

// ── Test policies ─────────────────────────────────────────────────────────────

/// Stands in for a host security policy: an internal state root plus a rule
/// naming specific internal state locations.
///
/// Mirrors the real shape — a host has both a root it owns wholesale and
/// individual state paths that may sit elsewhere — because the two are checked
/// separately and produce different errors.
#[derive(Debug)]
struct TestPolicy {
    internal_root: PathBuf,
    /// Path suffixes treated as internal state wherever they appear.
    internal_suffixes: Vec<String>,
}

impl TestPolicy {
    fn rooted_at(internal_root: PathBuf) -> Self {
        Self {
            internal_root,
            internal_suffixes: Vec::new(),
        }
    }

    fn with_internal_suffix(mut self, suffix: &str) -> Self {
        self.internal_suffixes.push(suffix.to_string());
        self
    }
}

impl ArtifactPathPolicy for TestPolicy {
    fn is_internal_state(&self, path: &Path) -> bool {
        let shown = path.to_string_lossy().replace('\\', "/");
        self.internal_suffixes
            .iter()
            .any(|suffix| shown.contains(suffix))
    }

    fn internal_root(&self) -> Option<&Path> {
        Some(&self.internal_root)
    }
}

/// Stands in for a host credential scrubber.
#[derive(Debug)]
struct TestRedactor;

const SECRET: &str = "sk-live-0123456789abcdef";
const REDACTION: &str = "[redacted]";

impl ArtifactRedactor for TestRedactor {
    fn redact(&self, content: &str) -> Redacted {
        if content.contains(SECRET) {
            Redacted::rewritten(content.replace(SECRET, REDACTION))
        } else {
            Redacted::unchanged(content)
        }
    }
}

const READ_TOOL: &str = "file_read";

fn offload_for(root: &Path, internal_root: &Path) -> ArtifactOffload {
    ArtifactOffload::new(root.to_path_buf(), "worker-agent", "task-1")
        .with_path_policy(Arc::new(TestPolicy::rooted_at(internal_root.to_path_buf())))
        .with_redactor(Arc::new(TestRedactor))
}

// ── Kinds and directories ─────────────────────────────────────────────────────

#[test]
fn kinds_map_to_the_documented_directories() {
    assert_eq!(ArtifactKind::Output.subdir(), OUTPUTS_DIR);
    assert_eq!(ArtifactKind::Scratch.subdir(), SCRATCH_DIR);
    assert_eq!(ArtifactKind::Output.as_str(), "output");
    assert_eq!(ArtifactKind::Scratch.as_str(), "scratch");
}

// ── Path resolution ───────────────────────────────────────────────────────────

#[test]
fn resolves_under_the_convention_directory() {
    let root = PathBuf::from("/action");
    let resolved =
        resolve_artifact_path(&root, None, ArtifactKind::Output, "report.md").expect("resolve");
    assert_eq!(resolved, root.join("outputs").join("report.md"));
}

#[test]
fn scratch_resolves_under_the_artifact_root_not_the_internal_root() {
    // The `workspace` subdir of the artifact root and the host's internal root
    // are different places that happen to share a word. Conflating them would
    // route every scratch write into host state.
    let root = PathBuf::from("/action");
    let internal = PathBuf::from("/internal/workspace");
    let policy = TestPolicy::rooted_at(internal.clone());
    let resolved = resolve_artifact_path(&root, Some(&policy), ArtifactKind::Scratch, "notes.md")
        .expect("resolve");
    assert_eq!(resolved, root.join("workspace").join("notes.md"));
    assert!(!resolved.starts_with(&internal));
}

#[test]
fn rejects_parent_traversal() {
    let err = resolve_artifact_path(
        Path::new("/action"),
        None,
        ArtifactKind::Output,
        "../../etc/passwd",
    )
    .expect_err("traversal must be refused");
    assert!(
        matches!(err, OffloadError::PathEscape { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_absolute_paths() {
    let err = resolve_artifact_path(
        Path::new("/action"),
        None,
        ArtifactKind::Output,
        "/etc/passwd",
    )
    .expect_err("absolute path must be refused");
    assert!(
        matches!(err, OffloadError::AbsolutePath { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_empty_and_whitespace_names() {
    for name in ["", "   ", "\t\n"] {
        let err = resolve_artifact_path(Path::new("/action"), None, ArtifactKind::Output, name)
            .expect_err("blank name must be refused");
        assert!(matches!(err, OffloadError::EmptyName), "got {err:?}");
    }
}

#[test]
fn accepts_leading_current_dir_segments() {
    // `./report.md` is how a model commonly writes a relative path; refusing it
    // would reject a correct request on a formatting technicality.
    let resolved = resolve_artifact_path(
        Path::new("/action"),
        None,
        ArtifactKind::Output,
        "./report.md",
    )
    .expect("resolve");
    assert_eq!(resolved, Path::new("/action/outputs/report.md"));
}

#[test]
fn rejects_targets_inside_the_internal_root_fail_closed() {
    // An artifact root configured inside the host's internal root: containment
    // passes, so only the policy check stands between an agent and host state.
    let root = PathBuf::from("/internal/action");
    let policy = TestPolicy::rooted_at(PathBuf::from("/internal"));
    let err = resolve_artifact_path(&root, Some(&policy), ArtifactKind::Output, "leak.md")
        .expect_err("internal root must be refused");
    assert!(
        matches!(err, OffloadError::InternalRoot { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_host_internal_state_paths_with_the_specific_error() {
    // The specific check runs first so its more useful message survives when a
    // path trips both rules.
    let root = PathBuf::from("/internal/action");
    let policy = TestPolicy::rooted_at(PathBuf::from("/internal")).with_internal_suffix("outputs");
    let err = resolve_artifact_path(&root, Some(&policy), ArtifactKind::Output, "leak.md")
        .expect_err("internal state must be refused");
    assert!(
        matches!(err, OffloadError::InternalState { .. }),
        "the specific rule must win over blanket containment, got {err:?}"
    );
}

#[test]
fn a_policy_free_resolve_still_enforces_containment() {
    // `policy: None` relaxes only the host checks. If it ever relaxed traversal
    // too, every host without a policy would gain an escape.
    let err = resolve_artifact_path(
        Path::new("/action"),
        None,
        ArtifactKind::Output,
        "../out.md",
    )
    .expect_err("traversal must still be refused without a policy");
    assert!(
        matches!(err, OffloadError::PathEscape { .. }),
        "got {err:?}"
    );
}

#[test]
fn sanitize_component_strips_separators_and_never_returns_empty() {
    assert_eq!(sanitize_component("sub-1a/2b"), "sub-1a_2b");
    assert_eq!(sanitize_component("../etc"), "___etc");
    assert_eq!(sanitize_component("ok_name-1"), "ok_name-1");
    // A component that sanitizes to nothing must still be a usable directory
    // name, or the resulting path is malformed rather than merely odd.
    assert_eq!(sanitize_component(""), "unknown");
    assert_eq!(sanitize_component("///"), "___");
    assert!(sanitize_component(&"x".repeat(500)).len() <= 80);
}

#[test]
fn relative_to_root_falls_back_to_display_for_outside_paths() {
    // An absolute fallback is correct here: a bogus relative path would resolve
    // against the reader's own root and silently miss the file.
    let rendered = relative_to_root(Path::new("/action"), Path::new("/elsewhere/outputs/x.md"));
    assert_eq!(rendered, "/elsewhere/outputs/x.md");
}

#[test]
fn relative_to_root_renders_slash_separated() {
    let rendered = relative_to_root(
        Path::new("/action"),
        &PathBuf::from("/action").join("outputs").join("x.md"),
    );
    assert_eq!(rendered, "outputs/x.md");
}

// ── Thresholds ────────────────────────────────────────────────────────────────

#[test]
fn should_offload_respects_threshold_and_the_zero_disable() {
    assert!(should_offload(100, 50));
    assert!(!should_offload(50, 50), "the threshold is exclusive");
    assert!(!should_offload(10, 50));
    // Zero is the documented opt-out, not "offload everything".
    assert!(!should_offload(usize::MAX, 0));
}

#[test]
fn offload_threshold_tightens_to_an_agents_own_result_cap() {
    // A cap below the default would truncate the result before offload fired,
    // so the artifact would never reach disk at all.
    assert_eq!(effective_offload_threshold(8_192, Some(4_000)), 4_000);
    assert_eq!(effective_offload_threshold(8_192, Some(16_000)), 8_192);
    assert_eq!(effective_offload_threshold(8_192, None), 8_192);
    // A zero cap means "no cap", not "offload nothing".
    assert_eq!(effective_offload_threshold(8_192, Some(0)), 8_192);
}

// ── Abstracts ─────────────────────────────────────────────────────────────────

#[test]
fn build_abstract_returns_short_content_unchanged() {
    assert_eq!(build_abstract("  short  ", 100), "short");
}

#[test]
fn build_abstract_cuts_at_a_line_boundary_when_one_is_available() {
    let content = format!("{}\n{}", "a".repeat(60), "b".repeat(60));
    let out = build_abstract(&content, 100);
    assert!(out.ends_with("..."));
    assert!(!out.contains('b'), "should have cut at the newline: {out}");
}

#[test]
fn build_abstract_cuts_at_a_word_boundary_when_there_is_no_line_break() {
    let content = format!("{} {}", "a".repeat(60), "b".repeat(60));
    let out = build_abstract(&content, 100);
    assert!(out.ends_with("..."));
    assert!(!out.contains('b'), "should have cut at the space: {out}");
}

#[test]
fn build_abstract_handles_a_zero_budget_and_boundary_free_text() {
    assert_eq!(build_abstract("anything", 0), "");
    // No line or word break in the back half — a hard cut is the only option.
    let out = build_abstract(&"a".repeat(200), 100);
    assert!(out.ends_with("..."));
}

#[test]
fn build_abstract_never_splits_a_multibyte_character() {
    // Budget counted in chars, truncation done on a String: a byte-indexed cut
    // here would panic rather than merely misformat.
    let content = "é".repeat(200);
    let out = build_abstract(&content, 50);
    assert!(out.ends_with("..."));
    assert!(out.is_char_boundary(out.len()));
}

// ── Pointers ──────────────────────────────────────────────────────────────────

fn sample_artifact(redacted: bool) -> OffloadedArtifact {
    OffloadedArtifact {
        kind: ArtifactKind::Output,
        relative_path: "outputs/agent/task-result.md".to_string(),
        absolute_path: PathBuf::from("/action/outputs/agent/task-result.md"),
        stored_bytes: 1234,
        original_bytes: 1300,
        redacted,
    }
}

#[test]
fn pointer_carries_path_size_and_a_read_call() {
    let rendered = render_artifact_pointer(&sample_artifact(false), "the abstract", READ_TOOL);
    assert!(rendered.starts_with(ARTIFACT_POINTER_PREFIX));
    assert!(rendered.contains("path=outputs/agent/task-result.md"));
    assert!(rendered.contains("bytes=1234"));
    assert!(rendered.contains("kind=output"));
    assert!(rendered.contains(READ_TOOL));
    assert!(rendered.contains("the abstract"));
}

#[test]
fn pointer_names_the_read_tool_it_was_given() {
    // The tool name is host vocabulary. Hard-coding one here would put a tool
    // the host may not have into its prompts.
    let rendered = render_artifact_pointer(&sample_artifact(false), "x", "read_file");
    assert!(rendered.contains("read_with: read_file"));
    assert!(!rendered.contains("file_read"));
}

#[test]
fn pointer_discloses_redaction_when_it_happened() {
    let clean = render_artifact_pointer(&sample_artifact(false), "x", READ_TOOL);
    assert!(!clean.contains("redaction"));
    let redacted = render_artifact_pointer(&sample_artifact(true), "x", READ_TOOL);
    assert!(redacted.contains("redaction was applied"));
}

#[test]
fn extract_artifact_paths_reads_pointers_out_of_a_handoff() {
    let handoff =
        format!("{ARTIFACT_POINTER_PREFIX} kind=output path=outputs/a.md bytes=10\nprose");
    assert_eq!(extract_artifact_paths(&handoff), vec!["outputs/a.md"]);
}

#[test]
fn extract_artifact_paths_dedupes_and_keeps_encounter_order() {
    let handoff = format!(
        "{ARTIFACT_POINTER_PREFIX} path=b.md bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=a.md bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=b.md bytes=1"
    );
    assert_eq!(extract_artifact_paths(&handoff), vec!["b.md", "a.md"]);
}

#[test]
fn extract_artifact_paths_ignores_non_pointer_and_malformed_lines() {
    let handoff = format!(
        "ordinary prose\n\
         {ARTIFACT_POINTER_PREFIX} no path field here\n\
         {ARTIFACT_POINTER_PREFIX} path= bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=good.md bytes=1"
    );
    // The empty `path=` case is the subtle one: splitting on whitespace rather
    // than the FIRST whitespace would yield `bytes=1` as the path.
    assert_eq!(extract_artifact_paths(&handoff), vec!["good.md"]);
}

#[test]
fn note_artifact_handoff_reports_how_many_paths_crossed() {
    let paths = vec!["a.md".to_string(), "b.md".to_string()];
    assert_eq!(
        note_artifact_handoff(HANDOFF_STAGE_RECORDED, "agent", "task", &paths),
        2
    );
    assert_eq!(
        note_artifact_handoff(HANDOFF_STAGE_CONSUMED, "agent", "task", &[]),
        0
    );
}

#[test]
fn handoff_stages_are_distinct() {
    // They exist to tell the two ends of one pointer apart in a journal; equal
    // values would render the same line twice.
    assert_ne!(HANDOFF_STAGE_RECORDED, HANDOFF_STAGE_CONSUMED);
}

// ── Writing ───────────────────────────────────────────────────────────────────

fn temp_roots() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("action");
    let internal = dir.path().join("internal");
    std::fs::create_dir_all(&root).expect("action dir");
    std::fs::create_dir_all(&internal).expect("internal dir");
    (dir, root, internal)
}

#[tokio::test]
async fn write_persists_under_outputs_and_reports_the_relative_path() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);

    let artifact = offload
        .write(ArtifactKind::Output, "report.md", "hello")
        .await
        .expect("write");

    assert_eq!(artifact.relative_path, "outputs/report.md");
    assert_eq!(artifact.stored_bytes, 5);
    assert_eq!(artifact.original_bytes, 5);
    assert!(!artifact.redacted);
    let on_disk = tokio::fs::read_to_string(&artifact.absolute_path)
        .await
        .expect("read back");
    assert_eq!(on_disk, "hello");
}

#[tokio::test]
async fn write_redacts_credentials_before_they_reach_disk() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);
    let body = format!("token: {SECRET}\n");

    let artifact = offload
        .write(ArtifactKind::Output, "creds.md", &body)
        .await
        .expect("write");

    assert!(artifact.redacted, "the redactor rewrote the body");
    let on_disk = tokio::fs::read_to_string(&artifact.absolute_path)
        .await
        .expect("read back");
    assert!(
        !on_disk.contains(SECRET),
        "the secret reached disk: {on_disk}"
    );
    assert!(on_disk.contains(REDACTION));
    // Original bytes describe the caller's payload, stored bytes the file.
    assert_eq!(artifact.original_bytes, body.len());
    assert_eq!(artifact.stored_bytes, on_disk.len());
}

#[tokio::test]
async fn write_without_a_redactor_stores_bytes_verbatim() {
    // The documented consequence of omitting a redactor. Pinned so the default
    // cannot quietly become "scrub something" and give false assurance.
    let (_dir, root, _internal) = temp_roots();
    let offload = ArtifactOffload::new(root.clone(), "agent", "task");
    let artifact = offload
        .write(ArtifactKind::Output, "raw.md", SECRET)
        .await
        .expect("write");
    assert!(!artifact.redacted);
    let on_disk = tokio::fs::read_to_string(&artifact.absolute_path)
        .await
        .expect("read back");
    assert_eq!(on_disk, SECRET);
}

#[tokio::test]
async fn write_refuses_a_traversal_target_without_touching_disk() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);

    let err = offload
        .write(ArtifactKind::Output, "../escape.md", "x")
        .await
        .expect_err("traversal must be refused");

    assert!(
        matches!(err, OffloadError::PathEscape { .. }),
        "got {err:?}"
    );
    assert!(
        !root.join("escape.md").exists(),
        "nothing may be written on a refused path"
    );
}

#[tokio::test]
async fn default_result_name_sanitizes_both_identifiers() {
    let (_dir, root, _internal) = temp_roots();
    let offload = ArtifactOffload::new(root, "agent/../x", "task id/1");
    // Each of `/`, `.`, `.`, `/` becomes its own underscore — the separators are
    // replaced, never collapsed, so no identifier can smuggle in a path level.
    assert_eq!(
        offload.default_result_name(),
        "agent____x/task_id_1-result.md"
    );
}

#[tokio::test]
async fn write_refuses_a_parent_that_symlinks_out_of_the_convention_root() {
    let (dir, root, internal) = temp_roots();
    let outside = dir.path().join("outside");
    std::fs::create_dir_all(&outside).expect("outside dir");
    std::fs::create_dir_all(root.join("outputs")).expect("outputs dir");

    // `outputs/escape -> ../../outside`. The lexical checks cannot see this:
    // the target does not exist when they run.
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, root.join("outputs").join("escape")).expect("symlink");
    #[cfg(not(unix))]
    return;

    let offload = offload_for(&root, &internal);
    let err = offload
        .write(ArtifactKind::Output, "escape/leak.md", "x")
        .await
        .expect_err("symlink escape must be refused");

    assert!(
        matches!(err, OffloadError::SymlinkEscape { .. }),
        "got {err:?}"
    );
    assert!(
        !outside.join("leak.md").exists(),
        "the write must not have followed the link"
    );
}

#[tokio::test]
async fn worktree_artifact_renders_a_path_the_parent_can_resolve() {
    // An isolated worker writes in its own checkout; the parent that receives
    // the pointer holds a different root. Rendering against the parent's root is
    // what keeps the path resolvable on the receiving side.
    let (dir, root, internal) = temp_roots();
    let parent_root = dir.path().to_path_buf();
    let offload = offload_for(&root, &internal).with_render_root(parent_root);

    let artifact = offload
        .write(ArtifactKind::Output, "report.md", "x")
        .await
        .expect("write");

    assert_eq!(artifact.relative_path, "action/outputs/report.md");
}

#[tokio::test]
async fn a_render_root_outside_the_write_root_falls_back_to_absolute() {
    // Better an absolute path than a relative one that resolves against the
    // wrong root and silently misses the file.
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal).with_render_root(PathBuf::from("/unrelated"));

    let artifact = offload
        .write(ArtifactKind::Output, "report.md", "x")
        .await
        .expect("write");

    assert!(
        Path::new(&artifact.relative_path).is_absolute(),
        "expected an absolute fallback, got {}",
        artifact.relative_path
    );
}

// ── The offload entry point ───────────────────────────────────────────────────

#[tokio::test]
async fn oversized_result_is_offloaded_and_the_parent_gets_a_path_plus_abstract() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 8_192, READ_TOOL).await;
    let artifact = artifact.expect("an artifact was written");

    assert!(text.starts_with(ARTIFACT_POINTER_PREFIX));
    assert!(
        text.len() < big.len(),
        "the pointer must be smaller than the payload it replaced"
    );
    let on_disk = tokio::fs::read_to_string(&artifact.absolute_path)
        .await
        .expect("read back");
    assert_eq!(on_disk, big, "full fidelity is preserved on disk");
}

#[tokio::test]
async fn small_result_stays_inline() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);

    let (text, artifact) =
        offload_oversized_result("small".to_string(), &offload, 8_192, READ_TOOL).await;

    assert_eq!(text, "small");
    assert!(artifact.is_none());
}

#[tokio::test]
async fn offload_is_disabled_by_a_zero_threshold() {
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 0, READ_TOOL).await;

    assert_eq!(text, big);
    assert!(artifact.is_none());
}

#[tokio::test]
async fn offload_failure_keeps_the_inline_payload_for_the_host_backstop() {
    // The load-bearing soft-failure contract: a refused offload must never cost
    // the caller its content, or a disk problem turns into data loss.
    let (_dir, _root, internal) = temp_roots();
    let unwritable = PathBuf::from("/proc/nonexistent-artifact-root");
    let offload = offload_for(&unwritable, &internal);
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 8_192, READ_TOOL).await;

    assert_eq!(
        text, big,
        "the inline payload must survive a failed offload"
    );
    assert!(artifact.is_none());
}

#[tokio::test]
async fn abstract_is_built_from_the_redacted_body_not_the_raw_output() {
    // The pointer goes straight into the parent's context. Building the
    // abstract from the raw text would re-expose the credential that was just
    // scrubbed out of the file — and the file would still look correct.
    let (_dir, root, internal) = temp_roots();
    let offload = offload_for(&root, &internal);
    let body = format!("{SECRET} {}", "x".repeat(10_000));

    let (text, artifact) = offload_oversized_result(body, &offload, 8_192, READ_TOOL).await;
    let artifact = artifact.expect("an artifact was written");

    assert!(artifact.redacted);
    assert!(
        !text.contains(SECRET),
        "the secret leaked into the pointer: {text}"
    );
    assert!(text.contains(REDACTION));
}
