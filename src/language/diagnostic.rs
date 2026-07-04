//! Structured diagnostics for the `.rag` language and a source-aware renderer.
//!
//! A [`Diagnostic`] is the structured form of a language error: a [`Severity`],
//! a headline message, a primary [`Span`] (optionally labelled), zero or more
//! labelled secondary spans, and an optional `help` line. The renderer turns a
//! diagnostic plus the originating [`SourceFile`] into the familiar caret-underline
//! presentation:
//!
//! ```text
//! error[E-rag-unknown-node]: route target `toolz` does not exist
//!   --> support.rag:11:20
//!    |
//! 11 |       tool_call -> toolz
//!    |                    ^^^^^ unknown node
//!    |
//! help: did you mean `tools`?
//! ```
//!
//! Diagnostics keep their structure all the way to the crate boundary: the
//! lexer and parser build a `Diagnostic`, then fold it into a
//! [`crate::error::TinyAgentsError::Parse`] through [`Diagnostic::into_parse_error`],
//! which preserves the primary span's `line`/`column` in the error variant and
//! stores the rendered presentation as the error message.

use std::fmt::Write as _;

use crate::error::TinyAgentsError;
use crate::language::source::SourceFile;
use crate::language::span::Span;

/// The severity of a [`Diagnostic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// A hard error: compilation cannot proceed.
    Error,
    /// A non-fatal warning.
    Warning,
    /// An informational note.
    Note,
}

impl Severity {
    /// The lowercase label used in rendered output (`error`, `warning`, `note`).
    pub fn label(&self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

/// A labelled secondary span attached to a [`Diagnostic`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// The span this label points at.
    pub span: Span,
    /// The message rendered beneath the caret.
    pub message: String,
}

impl Label {
    /// Creates a label pointing at `span` with `message`.
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

/// A structured language diagnostic.
///
/// The `primary` span is the offending location; `primary_label` is the text
/// drawn beneath its caret. Additional `labels` annotate related secondary
/// spans, and `help` carries an optional suggestion line. `code` is an optional
/// stable identifier rendered as `severity[code]:`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// The diagnostic severity.
    pub severity: Severity,
    /// An optional stable diagnostic code (e.g. `E-rag-unknown-node`).
    pub code: Option<String>,
    /// The headline message.
    pub message: String,
    /// The primary offending span.
    pub primary: Span,
    /// The label drawn beneath the primary span's caret, if any.
    pub primary_label: Option<String>,
    /// Labelled secondary spans.
    pub labels: Vec<Label>,
    /// An optional help/suggestion line.
    pub help: Option<String>,
}

impl Diagnostic {
    /// Creates a diagnostic with the given severity, message, and primary span.
    pub fn new(severity: Severity, message: impl Into<String>, primary: Span) -> Self {
        Self {
            severity,
            code: None,
            message: message.into(),
            primary,
            primary_label: None,
            labels: Vec::new(),
            help: None,
        }
    }

    /// Creates an [`Severity::Error`] diagnostic.
    pub fn error(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Error, message, primary)
    }

    /// Creates a [`Severity::Warning`] diagnostic.
    pub fn warning(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Warning, message, primary)
    }

    /// Creates a [`Severity::Note`] diagnostic.
    pub fn note(message: impl Into<String>, primary: Span) -> Self {
        Self::new(Severity::Note, message, primary)
    }

    /// Sets the stable diagnostic code. Returns `self` for chaining.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Sets the label drawn beneath the primary caret. Returns `self`.
    pub fn with_primary_label(mut self, label: impl Into<String>) -> Self {
        self.primary_label = Some(label.into());
        self
    }

    /// Adds a labelled secondary span. Returns `self` for chaining.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label::new(span, message));
        self
    }

    /// Sets the help/suggestion line. Returns `self` for chaining.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Renders the diagnostic against `source`, drawing each span's source line
    /// with a caret underline.
    pub fn render(&self, source: &SourceFile) -> String {
        let mut out = String::new();
        self.write_header(&mut out);

        let gutter = self.gutter_width(source);
        render_span_block(
            &mut out,
            source,
            gutter,
            self.primary,
            self.primary_label.as_deref(),
        );
        for label in &self.labels {
            render_span_block(&mut out, source, gutter, label.span, Some(&label.message));
        }
        if let Some(help) = &self.help {
            let _ = writeln!(out, "{:gutter$} = help: {help}", "");
        }
        out
    }

    /// Renders the diagnostic without source context, for callers that only
    /// hold a token stream. Includes the headline plus a `--> line:col` anchor.
    pub fn render_plain(&self) -> String {
        let mut out = String::new();
        self.write_header(&mut out);
        let _ = writeln!(out, "  --> {}:{}", self.primary.line, self.primary.column);
        if let Some(help) = &self.help {
            let _ = writeln!(out, "  = help: {help}");
        }
        out
    }

    /// Folds the diagnostic into a [`TinyAgentsError::Parse`].
    ///
    /// When `source` is provided *and* the primary span carries real byte
    /// offsets, the message is the full caret-underline rendering and the
    /// `line`/`column` are resolved from that byte offset. Otherwise (no
    /// `source`, or a back-compat span built with [`Span::new`] — which
    /// anchors only a `line`/`column` and leaves `start`/`end` at `0`) the
    /// message is the source-free rendering and the span's own stored
    /// `line`/`column` is used directly: resolving byte offset `0` against a
    /// real file would otherwise always render `1:1`, silently discarding
    /// whatever real position the caller anchored the span at.
    pub fn into_parse_error(self, source: Option<&SourceFile>) -> TinyAgentsError {
        let has_offsets = self.primary.start != 0 || self.primary.end != 0;
        let (line, column, message) = match source {
            Some(file) if has_offsets => {
                let (line, column) = file.location(self.primary.start);
                (line, column, self.render(file))
            }
            _ => (self.primary.line, self.primary.column, self.render_plain()),
        };
        TinyAgentsError::Parse {
            message,
            line,
            column,
        }
    }

    fn write_header(&self, out: &mut String) {
        match &self.code {
            Some(code) => {
                let _ = writeln!(out, "{}[{code}]: {}", self.severity.label(), self.message);
            }
            None => {
                let _ = writeln!(out, "{}: {}", self.severity.label(), self.message);
            }
        }
    }

    /// Width of the line-number gutter: the widest line number across every
    /// span the diagnostic references.
    fn gutter_width(&self, source: &SourceFile) -> usize {
        let mut max_line = source.location(self.primary.start).0;
        for label in &self.labels {
            max_line = max_line.max(source.location(label.span.start).0);
        }
        max_line.to_string().len()
    }
}

/// Renders one labelled span as a `-->`/source-line/caret block.
fn render_span_block(
    out: &mut String,
    source: &SourceFile,
    gutter: usize,
    span: Span,
    label: Option<&str>,
) {
    let (line, column) = source.location(span.start);
    let line_text = source.line_text(line).unwrap_or("");

    let _ = writeln!(out, "{:gutter$}--> {}:{line}:{column}", "", source.name());
    let _ = writeln!(out, "{:gutter$} |", "");
    let _ = writeln!(out, "{line:>gutter$} | {line_text}");

    // Caret width: the span's character length on this line, at least one.
    // Clamp the caret range into the line's byte range so a span that points
    // past the end of the source (or past this line) neither trips `clamp`'s
    // `min <= max` precondition nor slices out of bounds — the same defensive
    // clamping `SourceFile::snippet` applies.
    let text_len = source.text().len();
    let (line_start, line_end) = source.line_range(line).unwrap_or((text_len, text_len));
    let caret_start = span.start.clamp(line_start, line_end);
    let caret_end = span.end.clamp(caret_start, line_end);
    let caret_width = source.text()[caret_start..caret_end].chars().count().max(1);
    let indent = column.saturating_sub(1);

    let caret_line = format!("{:gutter$} | {:indent$}{}", "", "", "^".repeat(caret_width));
    match label {
        Some(text) if !text.is_empty() => {
            let _ = writeln!(out, "{caret_line} {text}");
        }
        _ => {
            let _ = writeln!(out, "{caret_line}");
        }
    }
    let _ = writeln!(out, "{:gutter$} |", "");
}
