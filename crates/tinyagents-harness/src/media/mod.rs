//! Media generation tools (feature `media`).
//!
//! [`GenerateImageTool`] and [`GenerateVideoTool`] expose TinyInference's
//! provider-neutral [`tinyinference_image::ImageGenerator`] and
//! [`tinyinference_video::VideoGenerator`] as ordinary [`tinytools::Tool`]s.
//! The harness owns the tool contract — argument parsing (including the loose
//! spellings models emit), artifact persistence into the run's workspace, and
//! result wording. The host owns everything else: which generator (and so
//! which credential and endpoint), the tool's visible name, the output root,
//! and whether a local reference file may leave the machine.
//!
//! Both tools report a billed non-delivery as an error that says not to call
//! again. Video failures name the job id when it can be resumed, so a model
//! does not loop on a paid failure.

mod image_tool;
mod types;
mod video_tool;

pub use image_tool::{GENERATE_IMAGE_TOOL_NAME, GenerateImageTool};
pub use types::{DEFAULT_MEDIA_SUBDIR, MediaOutput, ReferencePathPolicy};
pub use video_tool::{GENERATE_VIDEO_TOOL_NAME, GenerateVideoTool};

use tinytools::{
    ToolAccess, ToolPolicy, ToolRuntime, ToolSideEffects, ToolTimeout, WorkspaceAccess,
};

/// Policy shared by both tools: network, a third-party service, a charge, and
/// file writes into the workspace. Never replayed after a crash, because a
/// replay is a second billed generation.
fn media_policy(timeout_ms: u64) -> ToolPolicy {
    ToolPolicy::classified()
        .with_side_effects(ToolSideEffects {
            read_only: false,
            writes_files: true,
            network: true,
            installs_dependencies: false,
            destructive: false,
            external_service: true,
            payment: true,
        })
        .with_runtime(ToolRuntime {
            timeout_ms: Some(timeout_ms),
            timeout: ToolTimeout::Millis(timeout_ms),
            idempotent: false,
            cancelable: true,
            ..ToolRuntime::default()
        })
        .with_access(ToolAccess {
            workspace: WorkspaceAccess::Scoped,
            ..ToolAccess::default()
        })
}

/// A unique, filesystem-safe artifact stem: `<kind>-<unix-millis>-<counter>`.
fn artifact_stem(kind: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    format!(
        "{kind}-{millis}-{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod test;
