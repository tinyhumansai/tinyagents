//! Guards the published host-capability surface against embedder vocabulary.
//!
//! `harness::host` exists so an embedding application can inject its own
//! behaviour without the crate learning anything about that application. That
//! guarantee is easy to state and easy to erode: a field name, a doc-comment
//! example, or an enum variant that encodes one host's internal concept ships
//! to docs.rs and becomes public API for everyone.
//!
//! This test is the mechanical check. It scans the seam module's source for
//! identifiers that belong to a specific embedding product rather than to a
//! general agent runtime, and fails with the offending file, line, and term.
//!
//! **Scope.** Today it covers `src/harness/host/` only, because that is the
//! whole published seam. As runtime code is relocated into this crate the
//! `SCANNED_DIRS` list must grow with it — a relocation that does not widen
//! this list has not been checked.
//!
//! **Adding a term.** Add anything that names a specific application, one of
//! its internal domains, one of its file conventions, or one of its named
//! third-party integrations. Do not add generic runtime vocabulary
//! ("orchestrator", "workspace", "session"); those are legitimately this
//! crate's own.

use std::fs;
use std::path::{Path, PathBuf};

/// Directories scanned for embedder vocabulary, relative to the crate root.
const SCANNED_DIRS: &[&str] = &["src/harness/host"];

/// Lowercased substrings that must not appear in the scanned sources.
///
/// Each entry is paired with why it is forbidden so a future contributor can
/// tell a real leak from an unlucky substring.
const FORBIDDEN_TERMS: &[(&str, &str)] = &[
    ("openhuman", "names a specific embedding application"),
    (
        "tinyhumans",
        "names a specific embedding application's vendor",
    ),
    (
        "integrations_agent",
        "names an application-internal agent id and its dispatcher override",
    ),
    (
        "tokenjuice",
        "names an application-internal compaction domain",
    ),
    (
        "subconscious",
        "names an application-internal routing workload",
    ),
    (
        "composio",
        "names a specific third-party integration provider",
    ),
    (
        "action_dir",
        "names an application config key with no crate-side meaning; use WorkspaceDescriptor",
    ),
    (
        "profile.md",
        "names an application file convention for prompt assembly",
    ),
    (
        "memory.md",
        "names an application file convention for prompt assembly",
    ),
    (
        "soul.md",
        "names an application file convention for prompt assembly",
    ),
    (
        "identity.md",
        "names an application file convention for prompt assembly",
    ),
];

/// Collects every `.rs` file under `dir`, recursively.
fn rust_sources(dir: &Path, into: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_sources(&path, into);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            into.push(path);
        }
    }
}

#[test]
fn host_seam_carries_no_embedder_vocabulary() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let mut sources = Vec::new();
    for dir in SCANNED_DIRS {
        let path = crate_root.join(dir);
        assert!(
            path.is_dir(),
            "SCANNED_DIRS names {dir}, which does not exist — update the list"
        );
        rust_sources(&path, &mut sources);
    }
    assert!(!sources.is_empty(), "scanned no sources; the glob is wrong");

    let mut violations = Vec::new();
    for source in &sources {
        let text =
            fs::read_to_string(source).unwrap_or_else(|e| panic!("read {}: {e}", source.display()));
        for (line_number, line) in text.lines().enumerate() {
            let haystack = line.to_lowercase();
            for (term, reason) in FORBIDDEN_TERMS {
                if haystack.contains(term) {
                    violations.push(format!(
                        "{}:{}: `{term}` — {reason}\n    {}",
                        source.strip_prefix(crate_root).unwrap_or(source).display(),
                        line_number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "embedder vocabulary reached the published host seam:\n{}",
        violations.join("\n")
    );
}
