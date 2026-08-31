//! The two host policies an artifact write consults.
//!
//! Offloading a worker's result to disk is generic mechanics — thresholds,
//! path resolution, pointer rendering. Two decisions inside it are emphatically
//! not, and neither can be made correctly by a redistributed crate:
//!
//! * **Which paths are off limits.** A host keeps internal state somewhere and
//!   an agent write must never land in it. Only the host knows where that is.
//! * **What must be scrubbed before bytes touch disk.** Credential and PII
//!   patterns are a host's compliance surface, not a library's.
//!
//! Both are therefore *gates the runtime calls*, never behaviour the runtime is
//! trusted to have performed — RFC §2 rule 5.

use std::fmt;
use std::path::Path;

// ── ArtifactPathPolicy ────────────────────────────────────────────────────────

/// Host policy: which resolved paths an artifact write must refuse.
///
/// The crate already refuses absolute paths, `..` traversal and anything
/// escaping the artifact root — those are containment rules it can evaluate on
/// its own. This trait covers the rule it cannot: a host's *internal state*
/// location, which has no meaning here.
///
/// Both methods are consulted, and they are separate because they fail for
/// different reasons and a host wants to tell them apart in a log: a path may
/// sit under the internal root wholesale, or be a specific state location that
/// happens to live elsewhere.
pub trait ArtifactPathPolicy: Send + Sync + fmt::Debug {
    /// Whether `path` is a host-internal state location that an agent write may
    /// never reach.
    ///
    /// Evaluated on a lexically-resolved path before the write, and again on the
    /// real, symlink-resolved parent directory afterwards. An implementation
    /// must therefore be **pure and cheap** — it is called on a hot path and its
    /// answer must not depend on when it was asked.
    fn is_internal_state(&self, path: &Path) -> bool;

    /// Root of the host's internal state, when it has a single one.
    ///
    /// Anything under this is refused outright, independently of
    /// [`is_internal_state`](Self::is_internal_state). Returning `None` skips
    /// only that containment check; the crate's own traversal and root checks
    /// always run.
    fn internal_root(&self) -> Option<&Path>;
}

// ── ArtifactRedactor ──────────────────────────────────────────────────────────

/// The result of running a host's redactor over an artifact body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redacted {
    /// The body as it should be stored. This — never the caller's input — is
    /// what gets written and what any preview must be rendered from.
    pub text: String,
    /// Whether the redactor rewrote anything.
    pub changed: bool,
}

impl Redacted {
    /// A body the redactor left alone.
    pub fn unchanged(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            changed: false,
        }
    }

    /// A body the redactor rewrote.
    pub fn rewritten(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            changed: true,
        }
    }
}

/// Host policy: credential/PII scrubbing applied before an artifact is stored.
///
/// # This is not optional in the way the sinks are
///
/// A `None` redactor means **bytes are written exactly as the agent produced
/// them**. That is a legitimate configuration for a host with no secrets in
/// play, and it is the honest default for a crate that cannot know a host's
/// patterns — but it is a security decision, not an absence of one. Hosts
/// handling credentials must supply an implementation.
///
/// # The stored body is the only safe source for a preview
///
/// [`redact`](Self::redact) returns the text to store, and callers rendering any
/// part of an artifact back into a model's context must render it from that
/// value. Building a preview from the original input would re-expose precisely
/// the credentials this trait just removed from the file — the failure is
/// silent, because the file on disk looks correctly scrubbed.
pub trait ArtifactRedactor: Send + Sync + fmt::Debug {
    /// Scrubs `content` for storage.
    fn redact(&self, content: &str) -> Redacted;
}

// ── NoRedaction ───────────────────────────────────────────────────────────────

/// A redactor that stores content verbatim.
///
/// For tests, and for hosts that genuinely have nothing to scrub. Prefer
/// passing `None` in production code so the choice is visible at the call site
/// rather than hidden behind a type name that reads like a policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoRedaction;

impl ArtifactRedactor for NoRedaction {
    fn redact(&self, content: &str) -> Redacted {
        Redacted::unchanged(content)
    }
}

// ── OpenPathPolicy ────────────────────────────────────────────────────────────

/// A path policy that forbids nothing beyond the crate's own containment rules.
///
/// For tests and for hosts with no internal state under the artifact root.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenPathPolicy;

impl ArtifactPathPolicy for OpenPathPolicy {
    fn is_internal_state(&self, _path: &Path) -> bool {
        false
    }

    fn internal_root(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_redaction_reports_the_body_unchanged() {
        let out = NoRedaction.redact("hunter2");
        assert_eq!(out.text, "hunter2");
        // If this ever reported `changed`, every pointer would carry a
        // redaction note for a body nothing touched.
        assert!(!out.changed);
    }

    #[test]
    fn open_policy_forbids_nothing() {
        assert!(!OpenPathPolicy.is_internal_state(Path::new("/anywhere")));
        assert_eq!(OpenPathPolicy.internal_root(), None);
    }

    #[test]
    fn redacted_constructors_set_the_changed_flag() {
        assert!(!Redacted::unchanged("a").changed);
        assert!(Redacted::rewritten("b").changed);
    }
}
