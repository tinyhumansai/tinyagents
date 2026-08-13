//! Filesystem offload for oversized worker artifacts on long-horizon runs.
//!
//! ## The problem
//!
//! For minutes-to-hours runs, keeping compressed results *in context* still
//! accumulates summary text step after step, and it can never restore full
//! fidelity. Summarising one oversized payload at a time does not stop the
//! aggregate from growing.
//!
//! ## The convention
//!
//! Two directories under the agent's artifact root:
//!
//! | Directory     | Holds                                                    |
//! | ------------- | -------------------------------------------------------- |
//! | `outputs/`    | Deliverables. Handed between steps **by path**.          |
//! | `workspace/`  | Scratch. Intermediate files not meant to be handed back. |
//!
//! A worker that produces a large result writes it to `outputs/` and returns the
//! path plus a short abstract. Context stays lean and the full artifact is
//! recoverable with an ordinary file read.
//!
//! Two halves enforce it, and the split is the point:
//!
//! * **Prompt** — a host renders an offload contract into its worker prompts, so
//!   workers offload on purpose. That half is **host-owned**: it names host tools
//!   and is host prompt text, so it is deliberately not in this crate.
//! * **Harness** — [`offload_oversized_result`] runs on every worker outcome, so
//!   an oversized result is offloaded even when the worker inlined it anyway.
//!   That half needs no cooperation from the model, which is exactly why it
//!   exists.
//!
//! Whatever summarisation or truncation backstop the host already has stays
//! exactly as it is: it is the fallback for anything this convention does not
//! catch, and for every failure mode here — a refused path, a full disk — the
//! caller keeps its inline payload and falls through to it.
//!
//! ## Hardening
//!
//! [`resolve_artifact_path`] is fail-closed. Absolute paths, `..` traversal, and
//! anything escaping the convention root are refused; when an
//! [`ArtifactPathPolicy`] is supplied, so is anything reaching the host's
//! internal state.
//!
//! The lexical checks cannot see symlinks, because the target does not exist
//! when they run. So the real parent directory is re-validated after
//! `create_dir_all` and before the write — that is the first moment the
//! link-resolved location can be checked at all.
//!
//! ## What the host supplies
//!
//! Two policies, both of which a redistributed crate cannot decide for itself:
//! [`ArtifactPathPolicy`] (which paths are off limits) and [`ArtifactRedactor`]
//! (what is scrubbed before bytes hit disk). See [`policy`].
//!
//! ## Logging
//!
//! Every write emits `[artifact] wrote worker artifact under the artifact root`,
//! and every path a handoff carries emits `[artifact] handoff carried an
//! artifact path`, so a run journal shows both ends of a pointer.

mod ops;
mod paths;
pub mod policy;
mod types;

pub use ops::{
    ArtifactOffload, HANDOFF_STAGE_CONSUMED, HANDOFF_STAGE_RECORDED, build_abstract,
    effective_offload_threshold, extract_artifact_paths, note_artifact_handoff,
    offload_oversized_result, render_artifact_pointer, should_offload,
};
pub use paths::{relative_to_root, resolve_artifact_path, sanitize_component};
pub use policy::{ArtifactPathPolicy, ArtifactRedactor, NoRedaction, OpenPathPolicy, Redacted};
pub use types::{
    ABSTRACT_BUDGET_CHARS, ARTIFACT_POINTER_PREFIX, ArtifactKind, DEFAULT_OFFLOAD_THRESHOLD_BYTES,
    OUTPUTS_DIR, OffloadError, OffloadedArtifact, SCRATCH_DIR,
};

#[cfg(test)]
mod test;
