//! Shared configuration for the media generation tools.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tinyinference_image::MediaReference;

/// Sub-directory of the output root that generated media is written into.
pub const DEFAULT_MEDIA_SUBDIR: &str = "generated-media";

/// Decides whether a local reference path may be read and sent to a provider,
/// returning the path to read. Relative paths arrive already joined to the
/// workspace root.
pub type ReferencePathPolicy = Arc<dyn Fn(&Path) -> Result<PathBuf, String> + Send + Sync>;

/// Where a tool writes artifacts and which local references it may read.
#[derive(Clone)]
pub struct MediaOutput {
    /// Root used when the run context carries no workspace.
    pub fallback_root: PathBuf,
    /// Directory under the root that artifacts are written into.
    pub subdir: String,
    /// Local-reference admission policy; `None` confines references to the
    /// workspace (or fallback) root.
    pub reference_policy: Option<ReferencePathPolicy>,
}

impl MediaOutput {
    /// Writes under `fallback_root/generated-media` when no workspace is known.
    #[must_use]
    pub fn new(fallback_root: impl Into<PathBuf>) -> Self {
        Self {
            fallback_root: fallback_root.into(),
            subdir: DEFAULT_MEDIA_SUBDIR.to_owned(),
            reference_policy: None,
        }
    }

    /// Changes the artifact sub-directory.
    #[must_use]
    pub fn with_subdir(mut self, subdir: impl Into<String>) -> Self {
        self.subdir = subdir.into();
        self
    }

    /// Installs a host policy for local reference paths.
    #[must_use]
    pub fn with_reference_policy(mut self, policy: ReferencePathPolicy) -> Self {
        self.reference_policy = Some(policy);
        self
    }

    /// The root for this call: the run's workspace, else the fallback.
    pub(crate) fn root<'a>(&'a self, workspace: Option<&'a Path>) -> &'a Path {
        workspace.unwrap_or(&self.fallback_root)
    }

    /// Creates and returns the artifact directory for this call.
    ///
    /// Every pre-existing component is inspected without following symlinks,
    /// then the resolved directory is checked against the resolved root. This
    /// makes a scoped workspace reject an output directory redirected outside
    /// the workspace before a generation request can be billed.
    pub(crate) fn dir(&self, workspace: Option<&Path>) -> Result<PathBuf, String> {
        let subdir = Path::new(&self.subdir);
        if subdir
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!(
                "artifact subdirectory `{}` must be a relative path below the output root",
                self.subdir
            ));
        }
        let root = self.root(workspace);
        std::fs::create_dir_all(root).map_err(|error| {
            format!(
                "output root {} could not be created: {error}",
                root.display()
            )
        })?;
        let canonical_root = root.canonicalize().map_err(|error| {
            format!(
                "output root {} could not be resolved: {error}",
                root.display()
            )
        })?;

        let mut dir = canonical_root.clone();
        for component in subdir.components() {
            let Component::Normal(component) = component else {
                unreachable!("subdirectory components were validated above");
            };
            dir.push(component);
            match std::fs::symlink_metadata(&dir) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(format!(
                        "artifact directory {} must not contain symlinks",
                        dir.display()
                    ));
                }
                Ok(metadata) if !metadata.is_dir() => {
                    return Err(format!(
                        "artifact path {} is not a directory",
                        dir.display()
                    ));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    std::fs::create_dir(&dir).map_err(|error| {
                        format!(
                            "artifact directory {} could not be created: {error}",
                            dir.display()
                        )
                    })?;
                    let metadata = std::fs::symlink_metadata(&dir).map_err(|error| {
                        format!(
                            "artifact directory {} could not be inspected: {error}",
                            dir.display()
                        )
                    })?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        return Err(format!(
                            "artifact directory {} changed while being created",
                            dir.display()
                        ));
                    }
                }
                Err(error) => {
                    return Err(format!(
                        "artifact directory {} could not be inspected: {error}",
                        dir.display()
                    ));
                }
            }
        }
        let canonical_dir = dir.canonicalize().map_err(|error| {
            format!(
                "artifact directory {} could not be resolved: {error}",
                dir.display()
            )
        })?;
        if !canonical_dir.starts_with(&canonical_root) {
            return Err(format!(
                "artifact directory {} is outside the output root",
                canonical_dir.display()
            ));
        }
        Ok(canonical_dir)
    }

    /// Persists one artifact without following a final-path symlink or
    /// replacing an existing file.
    pub(crate) fn persist(
        &self,
        dir: &Path,
        stem: &str,
        extension: &str,
        bytes: &[u8],
    ) -> Result<PathBuf, String> {
        let path = dir.join(format!("{stem}.{extension}"));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // O_NOFOLLOW: refuse a final path that has been swapped for a symlink.
            options.custom_flags(0o400_000);
        }
        let mut file = options.open(&path).map_err(|error| {
            format!(
                "artifact {} could not be created safely: {error}",
                path.display()
            )
        })?;
        file.write_all(bytes).map_err(|error| {
            format!("artifact {} could not be written: {error}", path.display())
        })?;
        Ok(path)
    }

    /// Converts a model-supplied reference string into a [`MediaReference`],
    /// resolving and admitting local paths.
    pub(crate) fn reference(
        &self,
        raw: &str,
        workspace: Option<&Path>,
    ) -> Result<MediaReference, String> {
        match MediaReference::parse(raw) {
            MediaReference::Path(path) => {
                let root = self.root(workspace);
                let joined = if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                };
                // Canonicalize the path to resolve symlinks before checking confinement.
                // This prevents symlink attacks where a symlink inside the workspace
                // points to a location outside it.
                let canonical = joined.canonicalize().map_err(|e| {
                    format!(
                        "reference path {} could not be resolved: {}",
                        joined.display(),
                        e
                    )
                })?;
                // Canonicalize the root if it exists, for accurate symlink-safe
                // comparison. If the root is a fallback path that doesn't exist,
                // fall back to a lexical check after canonicalizing the path.
                let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
                let admitted = match &self.reference_policy {
                    Some(policy) => policy(&canonical)?,
                    None => confine(&canonical, &canonical_root)?,
                };
                Ok(MediaReference::Path(admitted))
            }
            other => Ok(other),
        }
    }
}

impl std::fmt::Debug for MediaOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaOutput")
            .field("fallback_root", &self.fallback_root)
            .field("subdir", &self.subdir)
            .field("reference_policy", &self.reference_policy.is_some())
            .finish()
    }
}

/// Default reference policy: the path must stay inside `root` when canonicalized,
/// so a model cannot read and upload arbitrary files via symlinks.
fn confine(path: &Path, root: &Path) -> Result<PathBuf, String> {
    // Both paths are already canonicalized by the caller, so we check the
    // resolved path against the resolved root.
    if !path.starts_with(root) {
        return Err(format!(
            "reference path {} is outside the workspace; use a URL or a file inside the workspace",
            path.display()
        ));
    }
    Ok(path.to_path_buf())
}

/// Reads the first present string among `keys`.
pub(crate) fn arg_str<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Rejects a present-but-malformed option instead of silently dropping it.
///
/// The schemas accept numeric and boolean options as strings too (models emit
/// both), so a value such as `"n": "two"` passes schema validation; without
/// this check it would be ignored and the call would run — and bill — with
/// the default instead of what was asked.
pub(crate) fn check_option_types(
    args: &Value,
    unsigned: &[&str],
    signed: &[&str],
    booleans: &[&str],
) -> Result<(), String> {
    for key in unsigned {
        match args.get(*key) {
            None | Some(Value::Null) => {}
            Some(Value::Number(number)) if number.is_u64() => {}
            Some(Value::String(text)) if text.trim().parse::<u64>().is_ok() => {}
            Some(other) => {
                return Err(format!(
                    "`{key}` must be a non-negative integer, got {other}"
                ));
            }
        }
    }
    for key in signed {
        match args.get(*key) {
            None | Some(Value::Null) => {}
            Some(Value::Number(number)) if number.is_i64() => {}
            Some(Value::String(text)) if text.trim().parse::<i64>().is_ok() => {}
            Some(other) => return Err(format!("`{key}` must be an integer, got {other}")),
        }
    }
    for key in booleans {
        match args.get(*key) {
            None | Some(Value::Null) | Some(Value::Bool(_)) => {}
            Some(Value::String(text)) if text.trim().parse::<bool>().is_ok() => {}
            Some(other) => return Err(format!("`{key}` must be true or false, got {other}")),
        }
    }
    Ok(())
}

/// Reads the first present unsigned integer among `keys` (numbers or numeric
/// strings, since models emit both).
pub(crate) fn arg_u64(args: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads the first present integer among `keys`.
pub(crate) fn arg_i64(args: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads the first present boolean among `keys`.
pub(crate) fn arg_bool(args: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads string lists among `keys`, accepting a single string too.
pub(crate) fn arg_list(args: &Value, keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for key in keys {
        match args.get(*key) {
            Some(Value::Array(items)) => out.extend(
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
            ),
            Some(Value::String(item)) if !item.trim().is_empty() => {
                out.push(item.trim().to_owned())
            }
            _ => {}
        }
    }
    out
}
