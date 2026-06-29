//! Source files and the source map: the substrate diagnostics render against.
//!
//! A [`SourceFile`] owns one `.rag` source string plus a precomputed line index
//! so it can map a byte offset back to a 1-based `(line, column)` and slice the
//! exact text of any line or [`crate::language::span::Span`]. A [`SourceMap`]
//! holds many such files, each addressed by a [`SourceId`], so a compiler that
//! ingests several sources (for example a graph plus its referenced subgraphs)
//! can resolve a span to the file it came from.
//!
//! This is the read-only counterpart to [`crate::language::span::Span`]: spans
//! carry coordinates, source files turn those coordinates back into text.

use crate::language::span::Span;

/// A handle identifying a [`SourceFile`] within a [`SourceMap`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub usize);

/// The default display name used for anonymous, in-memory source text.
pub const ANONYMOUS_NAME: &str = "<source>";

/// One source file: a display name, the full source text, and a precomputed
/// index of line-start byte offsets.
///
/// The line index makes offset → `(line, column)` lookups and line slicing
/// `O(log n)` / `O(1)` rather than rescanning the text per diagnostic.
#[derive(Clone, Debug)]
pub struct SourceFile {
    id: SourceId,
    name: String,
    text: String,
    /// Byte offset of the start of each line. `line_starts[0]` is always `0`.
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Builds a source file with the given display `name` and `text`, using
    /// [`SourceId`]`(0)`.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self::with_id(SourceId(0), name, text)
    }

    /// Builds an anonymous in-memory source file named [`ANONYMOUS_NAME`].
    pub fn anonymous(text: impl Into<String>) -> Self {
        Self::new(ANONYMOUS_NAME, text)
    }

    /// Builds a source file with an explicit [`SourceId`].
    pub fn with_id(id: SourceId, name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let line_starts = compute_line_starts(&text);
        Self {
            id,
            name: name.into(),
            text,
            line_starts,
        }
    }

    /// The file's identifier within its [`SourceMap`].
    pub fn id(&self) -> SourceId {
        self.id
    }

    /// The file's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full source text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The number of lines in the file (always at least one).
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Maps a byte `offset` to a 1-based `(line, column)`.
    ///
    /// `column` counts Unicode scalar values from the line start, so it matches
    /// the lexer's character-based column tracking. An offset past the end of
    /// the text clamps to the final position.
    pub fn location(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let line_start = self.line_starts[line_idx];
        let column = self.text[line_start..offset].chars().count() + 1;
        (line_idx + 1, column)
    }

    /// Returns the `start..end` byte range of the 1-based `line`, excluding the
    /// trailing newline, or `None` if the line does not exist.
    pub fn line_range(&self, line: usize) -> Option<(usize, usize)> {
        if line == 0 || line > self.line_starts.len() {
            return None;
        }
        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .map(|next| next.saturating_sub(1))
            .unwrap_or(self.text.len());
        // Trim a trailing carriage return for CRLF sources.
        let end = if end > start && self.text.as_bytes().get(end - 1) == Some(&b'\r') {
            end - 1
        } else {
            end
        };
        Some((start, end))
    }

    /// Returns the text of the 1-based `line` without its trailing newline.
    pub fn line_text(&self, line: usize) -> Option<&str> {
        let (start, end) = self.line_range(line)?;
        Some(&self.text[start..end])
    }

    /// Slices the exact source text a `span` covers (clamped to the text).
    pub fn snippet(&self, span: Span) -> &str {
        let start = span.start.min(self.text.len());
        let end = span.end.min(self.text.len()).max(start);
        &self.text[start..end]
    }
}

/// Computes the byte offset of every line start. The first entry is always `0`.
fn compute_line_starts(text: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// A collection of [`SourceFile`]s addressed by [`SourceId`].
///
/// Used when more than one source participates in a single compilation so a
/// diagnostic's span can be resolved back to the file it originated from.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    /// Creates an empty source map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a source file and returns its freshly assigned [`SourceId`].
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> SourceId {
        let id = SourceId(self.files.len());
        self.files.push(SourceFile::with_id(id, name, text));
        id
    }

    /// Returns the file for `id`, if present.
    pub fn get(&self, id: SourceId) -> Option<&SourceFile> {
        self.files.get(id.0)
    }

    /// The number of files in the map.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Returns true when the map holds no files.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Iterates over the files in insertion order.
    pub fn files(&self) -> impl Iterator<Item = &SourceFile> {
        self.files.iter()
    }
}
