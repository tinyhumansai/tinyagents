//! Git-worktree backed workspace isolation.
//!
//! This provider gives each agent run its own checkout under
//! `<repo>/.claude/worktrees/<run_id>`, so parallel edit-capable workers can
//! operate without sharing one mutable filesystem root. It intentionally owns
//! only generic git/workspace behavior; host applications remain responsible
//! for product-specific audit, cleanup policy, and merge UX.

use std::path::{Path, PathBuf};
use std::process::Command;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::harness::tool::SandboxMode;
use crate::harness::workspace::{WorkspaceDescriptor, WorkspaceIsolation};
use crate::{Result, TinyAgentsError};

/// Directory, relative to the repository root, where isolated worktrees are
/// created.
pub const GIT_WORKTREE_SUBDIR: &str = ".claude/worktrees";

/// Which ref a new worktree branches from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitWorktreeBaseRef {
    /// Branch off the repo's current `HEAD`.
    Head,
    /// Branch off the repository's default branch (`origin/HEAD`, then local
    /// `HEAD`, then `main`).
    Fresh,
}

impl GitWorktreeBaseRef {
    /// Parses a user/config string. Unknown or empty values default to
    /// [`GitWorktreeBaseRef::Head`].
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("fresh") => Self::Fresh,
            _ => Self::Head,
        }
    }

    /// Stable lowercase label for logs and policy ids.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Head => "head",
            Self::Fresh => "fresh",
        }
    }
}

/// Snapshot of a single git worktree's state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitWorktreeStatus {
    /// Absolute path to the worktree checkout.
    pub path: PathBuf,
    /// Checked-out branch, or `(detached HEAD)` for detached worktrees.
    pub branch: Option<String>,
    /// Whether staged, unstaged, or untracked changes are present.
    pub is_dirty: bool,
    /// Changed files relative to the worktree root.
    pub changed_files: Vec<PathBuf>,
}

/// [`WorkspaceIsolation`] implementation backed by `git worktree`.
#[derive(Debug, Clone)]
pub struct GitWorktreeIsolation {
    repo_root: PathBuf,
    base_ref: GitWorktreeBaseRef,
    sandbox: SandboxMode,
    trusted_roots: Vec<PathBuf>,
}

impl GitWorktreeIsolation {
    /// Creates an isolation provider rooted at a git repository.
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: repo_root.into(),
            base_ref: GitWorktreeBaseRef::Head,
            sandbox: SandboxMode::Inherit,
            trusted_roots: Vec::new(),
        }
    }

    /// Selects which ref newly prepared worktrees branch from.
    pub fn with_base_ref(mut self, base_ref: GitWorktreeBaseRef) -> Self {
        self.base_ref = base_ref;
        self
    }

    /// Advertises the sandbox expectation on prepared descriptors.
    pub fn with_sandbox(mut self, sandbox: SandboxMode) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Adds an extra root tools may touch alongside the isolated checkout.
    pub fn with_trusted_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.trusted_roots.push(root.into());
        self
    }
}

#[async_trait]
impl WorkspaceIsolation for GitWorktreeIsolation {
    async fn prepare(&self, run_id: &str, agent: Option<&str>) -> Result<WorkspaceDescriptor> {
        let status = create_git_worktree(&self.repo_root, run_id, self.base_ref)
            .map_err(|err| TinyAgentsError::Tool(err.to_string()))?;
        let policy_id = match agent {
            Some(agent) if !agent.is_empty() => format!("git.worktree:{agent}:{run_id}"),
            _ => format!("git.worktree:{run_id}"),
        };
        let mut descriptor = WorkspaceDescriptor::new(status.path)
            .with_policy_id(policy_id)
            .with_sandbox(self.sandbox);
        for root in &self.trusted_roots {
            descriptor = descriptor.with_trusted_root(root.clone());
        }
        Ok(descriptor)
    }

    async fn cleanup(&self, descriptor: &WorkspaceDescriptor) -> Result<()> {
        remove_git_worktree(&self.repo_root, &descriptor.root, false)
            .map_err(|err| TinyAgentsError::Tool(err.to_string()))
    }
}

/// Errors surfaced by the git worktree manager.
#[derive(Debug, thiserror::Error)]
pub enum GitWorktreeError {
    /// The supplied path is not inside a git work tree.
    #[error("path is not inside a git repository: {0}")]
    NotAGitRepo(PathBuf),

    /// Dirty worktrees are not removed unless `force = true`.
    #[error("worktree is dirty and force=false; refusing to remove: {0}")]
    DirtyRefused(PathBuf),

    /// A git command exited unsuccessfully.
    #[error("git command `{command}` failed: {stderr}")]
    GitFailed { command: String, stderr: String },

    /// Spawning git or creating the worktree parent directory failed.
    #[error("io error running git: {0}")]
    Io(#[from] std::io::Error),
}

type GitResult<T> = std::result::Result<T, GitWorktreeError>;

/// Creates an isolated worktree for `run_id` and returns its status snapshot.
pub fn create_git_worktree(
    repo_root: &Path,
    run_id: &str,
    base_ref: GitWorktreeBaseRef,
) -> GitResult<GitWorktreeStatus> {
    let repo_top = validate_repo_root(repo_root)?;
    let run_slug = sanitize_run_id(run_id);
    let worktree_path = repo_top.join(GIT_WORKTREE_SUBDIR).join(&run_slug);
    let branch = format!("worker/{run_slug}");
    let base = match base_ref {
        GitWorktreeBaseRef::Head => "HEAD".to_string(),
        GitWorktreeBaseRef::Fresh => resolve_fresh_base(&repo_top),
    };

    if let Some(parent) = worktree_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let worktree = worktree_path.to_string_lossy().to_string();
    git(
        &repo_top,
        &["worktree", "add", "-b", &branch, &worktree, &base],
    )?;
    git_worktree_status(&repo_top, &worktree_path)
}

/// Lists worktrees registered on the repository at `repo_root`.
pub fn list_git_worktrees(repo_root: &Path) -> GitResult<Vec<GitWorktreeStatus>> {
    let repo_top = validate_repo_root(repo_root)?;
    let porcelain = git(&repo_top, &["worktree", "list", "--porcelain"])?;
    let mut out = Vec::new();
    let mut cur_path: Option<PathBuf> = None;
    let mut cur_branch: Option<String> = None;

    let mut flush = |path: &mut Option<PathBuf>, branch: &mut Option<String>| {
        if let Some(path) = path.take() {
            let (is_dirty, changed_files) = dirty_state(&path).unwrap_or((false, Vec::new()));
            out.push(GitWorktreeStatus {
                path,
                branch: branch.take(),
                is_dirty,
                changed_files,
            });
        } else {
            *branch = None;
        }
    };

    for line in porcelain.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            flush(&mut cur_path, &mut cur_branch);
            cur_path = Some(PathBuf::from(rest.trim()));
        } else if let Some(rest) = line.strip_prefix("branch ") {
            let trimmed = rest.trim();
            cur_branch = Some(
                trimmed
                    .strip_prefix("refs/heads/")
                    .unwrap_or(trimmed)
                    .to_string(),
            );
        } else if line.trim() == "detached" {
            cur_branch = Some("(detached HEAD)".to_string());
        }
    }
    flush(&mut cur_path, &mut cur_branch);
    Ok(out)
}

/// Returns branch, dirty, and changed-file status for one worktree.
pub fn git_worktree_status(repo_root: &Path, worktree_path: &Path) -> GitResult<GitWorktreeStatus> {
    validate_repo_root(repo_root)?;
    if !worktree_path.exists() {
        return Err(GitWorktreeError::NotAGitRepo(worktree_path.to_path_buf()));
    }
    let branch = git(worktree_path, &["rev-parse", "--abbrev-ref", "HEAD"])
        .ok()
        .map(|branch| {
            if branch == "HEAD" {
                "(detached HEAD)".to_string()
            } else {
                branch
            }
        });
    let (is_dirty, changed_files) = dirty_state(worktree_path)?;
    Ok(GitWorktreeStatus {
        path: worktree_path.to_path_buf(),
        branch,
        is_dirty,
        changed_files,
    })
}

/// Human-readable diff stat of working changes vs `HEAD`, including untracked
/// files.
pub fn git_worktree_diff_summary(repo_root: &Path, worktree_path: &Path) -> GitResult<String> {
    validate_repo_root(repo_root)?;
    let stat = git(worktree_path, &["diff", "HEAD", "--stat"])?;
    let untracked = git(
        worktree_path,
        &["ls-files", "--others", "--exclude-standard"],
    )?;
    let mut parts = Vec::new();
    if !stat.is_empty() {
        parts.push(stat);
    }
    if !untracked.is_empty() {
        parts.push(
            untracked
                .lines()
                .map(|line| format!(" {line} (untracked)"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    Ok(parts.join("\n"))
}

/// Removes a worktree. Dirty worktrees are refused unless `force = true`.
pub fn remove_git_worktree(repo_root: &Path, worktree_path: &Path, force: bool) -> GitResult<()> {
    let repo_top = validate_repo_root(repo_root)?;
    let (is_dirty, _) = dirty_state(worktree_path).unwrap_or((false, Vec::new()));
    if is_dirty && !force {
        return Err(GitWorktreeError::DirtyRefused(worktree_path.to_path_buf()));
    }

    let worktree = worktree_path.to_string_lossy().to_string();
    let mut args = vec!["worktree", "remove", &worktree];
    if force {
        args.push("--force");
    }
    git(&repo_top, &args)?;
    Ok(())
}

/// Detects changed files touched by more than one sibling worker.
pub fn detect_worktree_overlaps(
    per_worker: &[(String, Vec<PathBuf>)],
) -> std::collections::BTreeMap<PathBuf, Vec<String>> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut by_file: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
    for (worker_id, files) in per_worker {
        let mut seen = BTreeSet::new();
        for file in files {
            if seen.insert(file.clone()) {
                by_file
                    .entry(file.clone())
                    .or_default()
                    .push(worker_id.clone());
            }
        }
    }

    by_file
        .into_iter()
        .filter_map(|(file, mut workers)| {
            workers.sort();
            workers.dedup();
            (workers.len() > 1).then_some((file, workers))
        })
        .collect()
}

fn git(cwd: &Path, args: &[&str]) -> GitResult<String> {
    let output = Command::new("git").current_dir(cwd).args(args).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(GitWorktreeError::GitFailed {
            command: format!("git {}", args.join(" ")),
            stderr,
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_raw(cwd: &Path, args: &[&str]) -> GitResult<String> {
    let output = Command::new("git").current_dir(cwd).args(args).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(GitWorktreeError::GitFailed {
            command: format!("git {}", args.join(" ")),
            stderr,
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn validate_repo_root(repo_root: &Path) -> GitResult<PathBuf> {
    if !repo_root.exists() {
        return Err(GitWorktreeError::NotAGitRepo(repo_root.to_path_buf()));
    }
    let inside = git(repo_root, &["rev-parse", "--is-inside-work-tree"])
        .map_err(|_| GitWorktreeError::NotAGitRepo(repo_root.to_path_buf()))?;
    if inside.trim() != "true" {
        return Err(GitWorktreeError::NotAGitRepo(repo_root.to_path_buf()));
    }
    let top = git(repo_root, &["rev-parse", "--show-toplevel"])
        .map_err(|_| GitWorktreeError::NotAGitRepo(repo_root.to_path_buf()))?;
    Ok(PathBuf::from(top.trim()))
}

fn resolve_fresh_base(repo_top: &Path) -> String {
    if let Ok(sym) = git(
        repo_top,
        &["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
    ) && !sym.is_empty()
    {
        return sym;
    }
    if let Ok(head) = git(repo_top, &["symbolic-ref", "--short", "HEAD"])
        && !head.is_empty()
    {
        return head;
    }
    "main".to_string()
}

fn dirty_state(worktree_path: &Path) -> GitResult<(bool, Vec<PathBuf>)> {
    let porcelain = git_raw(worktree_path, &["status", "--porcelain"])?;
    let mut changed = Vec::new();
    for line in porcelain.lines() {
        if line.len() > 3 {
            let path = line[3..].trim_end();
            let path = path.rsplit(" -> ").next().unwrap_or(path);
            changed.push(PathBuf::from(path));
        }
    }
    changed.sort();
    changed.dedup();
    Ok((!changed.is_empty(), changed))
}

fn sanitize_run_id(run_id: &str) -> String {
    let cleaned: String = run_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "worker".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod test;
