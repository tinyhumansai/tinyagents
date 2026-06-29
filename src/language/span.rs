//! Source spans: a byte range plus a 1-based line/column anchor.
//!
//! A [`Span`] is the unit of source location that every `.rag` token, AST node,
//! and [`crate::language::diagnostic::Diagnostic`] carries. It records both the
//! byte range it covers (`start..end`, the source-of-truth used to slice
//! snippets through a [`crate::language::source::SourceFile`]) and the 1-based
//! `line`/`column` of its first character (a convenience anchor preserved for
//! error reporting and kept stable across every compiler phase).
//!
//! Spans compose with [`Span::merge`], which yields the smallest span covering
//! two inputs — used to widen a diagnostic from a single token to a whole
//! construct.

use serde::{Deserialize, Serialize};

/// A source location: a `start..end` byte range with a 1-based line/column
/// anchor at `start`.
///
/// The byte offsets are the authoritative coordinates (used by
/// [`crate::language::source::SourceFile`] to slice the offending source); the
/// `line`/`column` pair is a convenience anchor that survives every compiler
/// phase so diagnostics can be reported even without the original source text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Inclusive start byte offset into the source.
    pub start: usize,
    /// Exclusive end byte offset into the source.
    pub end: usize,
    /// 1-based line number of the first character.
    pub line: usize,
    /// 1-based column number of the first character.
    pub column: usize,
}

impl Span {
    /// Creates a zero-width span anchored at a 1-based `line`/`column` with
    /// unknown byte offsets.
    ///
    /// This is the back-compatible constructor used where only a line/column is
    /// available. Prefer [`Span::at`] when byte offsets are known so snippets
    /// can be sliced.
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            start: 0,
            end: 0,
            line,
            column,
        }
    }

    /// Creates a span covering `start..end` bytes, anchored at the 1-based
    /// `line`/`column` of its first character.
    pub fn at(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }

    /// The length of the span in bytes.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true when the span covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Returns the smallest span covering both `self` and `other`.
    ///
    /// The byte range spans from the earliest `start` to the latest `end`, and
    /// the line/column anchor is taken from whichever input begins earlier.
    pub fn merge(self, other: Span) -> Span {
        let (lo, _hi) = if self.start <= other.start {
            (self, other)
        } else {
            (other, self)
        };
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            line: lo.line,
            column: lo.column,
        }
    }
}
