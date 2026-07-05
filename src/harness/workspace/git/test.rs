//! Tests for the git-worktree isolation provider.

use super::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tinyagents-worktree-test-{}-{nanos}-{counter}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn run(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("git invocation");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_repo() -> (TempDir, PathBuf) {
    let tmp = TempDir::new();
    let root = tmp.path().to_path_buf();
    run(&root, &["init", "-b", "main"]);
    run(&root, &["config", "user.email", "test@example.com"]);
    run(&root, &["config", "user.name", "Test User"]);
    std::fs::write(root.join("README.md"), "hello\n").unwrap();
    run(&root, &["add", "README.md"]);
    run(&root, &["commit", "-m", "initial"]);
    (tmp, root)
}

#[test]
fn validate_repo_root_rejects_non_repo() {
    if !git_available() {
        return;
    }
    let tmp = TempDir::new();
    let err = create_git_worktree(tmp.path(), "run-1", GitWorktreeBaseRef::Head).unwrap_err();
    assert!(matches!(err, GitWorktreeError::NotAGitRepo(_)));
}

#[test]
fn create_then_status_reports_clean_worktree() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status =
        create_git_worktree(&root, "run-1", GitWorktreeBaseRef::Head).expect("create worktree");
    assert!(status.path.exists());
    assert_eq!(status.branch.as_deref(), Some("worker/run-1"));
    assert!(!status.is_dirty);
    assert!(status.changed_files.is_empty());
    assert!(status.path.ends_with(Path::new(".claude/worktrees/run-1")));
}

#[test]
fn list_includes_created_worktrees() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    create_git_worktree(&root, "run-a", GitWorktreeBaseRef::Head).expect("create a");
    create_git_worktree(&root, "run-b", GitWorktreeBaseRef::Fresh).expect("create b");

    let all = list_git_worktrees(&root).expect("list worktrees");
    assert!(all.len() >= 3, "expected main + two worktrees, got {all:?}");
    let branches: Vec<_> = all
        .iter()
        .filter_map(|worktree| worktree.branch.clone())
        .collect();
    assert!(branches.iter().any(|branch| branch == "worker/run-a"));
    assert!(branches.iter().any(|branch| branch == "worker/run-b"));
}

#[test]
fn status_detects_dirty_changes() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status = create_git_worktree(&root, "run-dirty", GitWorktreeBaseRef::Head).expect("create");
    std::fs::write(status.path.join("README.md"), "changed\n").unwrap();
    std::fs::write(status.path.join("new.txt"), "fresh\n").unwrap();

    let status = git_worktree_status(&root, &status.path).expect("status");
    assert!(status.is_dirty);
    let names: Vec<_> = status
        .changed_files
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    assert!(names.iter().any(|name| name.contains("README.md")));
    assert!(names.iter().any(|name| name.contains("new.txt")));
}

#[test]
fn diff_summary_lists_tracked_and_untracked() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status = create_git_worktree(&root, "run-diff", GitWorktreeBaseRef::Head).expect("create");
    std::fs::write(status.path.join("README.md"), "changed body\n").unwrap();
    std::fs::write(status.path.join("brand_new.txt"), "x\n").unwrap();

    let summary = git_worktree_diff_summary(&root, &status.path).expect("diff");
    assert!(summary.contains("README.md"));
    assert!(summary.contains("brand_new.txt") && summary.contains("untracked"));
}

#[test]
fn remove_refuses_dirty_without_force() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status = create_git_worktree(&root, "run-keep", GitWorktreeBaseRef::Head).expect("create");
    std::fs::write(status.path.join("README.md"), "dirty\n").unwrap();

    let err = remove_git_worktree(&root, &status.path, false).expect_err("must refuse dirty");
    assert!(matches!(err, GitWorktreeError::DirtyRefused(_)));
    assert!(status.path.exists());
}

#[test]
fn remove_force_deletes_dirty_worktree() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status = create_git_worktree(&root, "run-force", GitWorktreeBaseRef::Head).expect("create");
    std::fs::write(status.path.join("README.md"), "dirty\n").unwrap();

    remove_git_worktree(&root, &status.path, true).expect("force remove");
    assert!(!status.path.exists());
}

#[test]
fn remove_clean_worktree_succeeds() {
    if !git_available() {
        return;
    }
    let (_tmp, root) = init_repo();
    let status = create_git_worktree(&root, "run-clean", GitWorktreeBaseRef::Head).expect("create");

    remove_git_worktree(&root, &status.path, false).expect("clean remove");
    assert!(!status.path.exists());
}

#[test]
fn base_ref_parse_defaults_to_head() {
    assert_eq!(GitWorktreeBaseRef::parse(None), GitWorktreeBaseRef::Head);
    assert_eq!(
        GitWorktreeBaseRef::parse(Some("head")),
        GitWorktreeBaseRef::Head
    );
    assert_eq!(
        GitWorktreeBaseRef::parse(Some(" Fresh ")),
        GitWorktreeBaseRef::Fresh
    );
    assert_eq!(
        GitWorktreeBaseRef::parse(Some("garbage")),
        GitWorktreeBaseRef::Head
    );
}

#[test]
fn sanitize_run_id_strips_unsafe_chars() {
    assert_eq!(sanitize_run_id("sub-1234"), "sub-1234");
    assert_eq!(sanitize_run_id("a/b\\c"), "a-b-c");
    assert_eq!(sanitize_run_id("///"), "worker");
    assert_eq!(sanitize_run_id(""), "worker");
}

#[test]
fn detect_overlaps_flags_shared_files() {
    let per_worker = vec![
        (
            "w1".to_string(),
            vec![PathBuf::from("src/a.rs"), PathBuf::from("src/b.rs")],
        ),
        (
            "w2".to_string(),
            vec![PathBuf::from("src/b.rs"), PathBuf::from("src/c.rs")],
        ),
        ("w3".to_string(), vec![PathBuf::from("src/c.rs")]),
    ];

    let overlaps = detect_worktree_overlaps(&per_worker);
    assert_eq!(overlaps.len(), 2);
    assert_eq!(
        overlaps.get(&PathBuf::from("src/b.rs")).unwrap(),
        &vec!["w1".to_string(), "w2".to_string()]
    );
    assert_eq!(
        overlaps.get(&PathBuf::from("src/c.rs")).unwrap(),
        &vec!["w2".to_string(), "w3".to_string()]
    );
}

#[test]
fn detect_overlaps_empty_when_disjoint_or_duplicate_within_one_worker() {
    let disjoint = vec![
        ("w1".to_string(), vec![PathBuf::from("a.rs")]),
        ("w2".to_string(), vec![PathBuf::from("b.rs")]),
    ];
    assert!(detect_worktree_overlaps(&disjoint).is_empty());

    let duplicate = vec![(
        "w1".to_string(),
        vec![PathBuf::from("a.rs"), PathBuf::from("a.rs")],
    )];
    assert!(detect_worktree_overlaps(&duplicate).is_empty());
}
