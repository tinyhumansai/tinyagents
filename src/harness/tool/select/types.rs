//! Types for prompt-driven tool selection.

/// A borrowed view of one candidate tool, as the ranker sees it.
///
/// The ranker only ever reads a tool's name and its one-line description, so
/// a host adapts whatever tool descriptor it owns into this shape with a
/// borrowing `map` — no allocation beyond the `Vec` of views, and no
/// dependency from this crate on the host's tool type.
///
/// This is a named struct rather than a `(&str, &str)` tuple deliberately:
/// name hits are weighted three times as heavily as description hits, so a
/// transposed tuple would silently change the ranking with nothing to catch
/// it. Named fields make the transposition a compile error at the call site's
/// literal, and make the call site read for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectableTool<'a> {
    /// Tool name or action slug, e.g. `"GITHUB_CREATE_A_PULL_REQUEST"`.
    pub name: &'a str,
    /// One-line description of what the tool does.
    pub description: &'a str,
}

impl<'a> SelectableTool<'a> {
    /// Creates a candidate view over a tool's name and description.
    pub fn new(name: &'a str, description: &'a str) -> Self {
        Self { name, description }
    }
}

impl<'a> From<(&'a str, &'a str)> for SelectableTool<'a> {
    fn from((name, description): (&'a str, &'a str)) -> Self {
        Self::new(name, description)
    }
}

/// Detected query intent. A small, stable set — expanding it risks
/// over-matching (e.g. "open" is deliberately excluded because it appears in
/// both "open a PR" and "open PRs").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolVerb {
    /// Bring something into existence.
    Create,
    /// Transmit something to somebody.
    Send,
    /// Retrieve one known thing.
    Read,
    /// Enumerate or search over many things.
    List,
    /// Change something that exists.
    Update,
    /// Remove something that exists.
    Delete,
    /// Merge, approve, or accept.
    Merge,
}
