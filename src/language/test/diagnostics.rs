//! Span, source-map, and diagnostic-rendering tests.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Spans, source map, and diagnostics
// ---------------------------------------------------------------------------

use crate::language::diagnostic::{Diagnostic, Severity};
use crate::language::source::{SourceFile, SourceMap};
use crate::language::span::Span;

#[test]
fn span_merge_covers_both_inputs() {
    let a = Span::at(2, 5, 1, 3);
    let b = Span::at(10, 14, 2, 1);
    let merged = a.merge(b);
    assert_eq!(merged.start, 2);
    assert_eq!(merged.end, 14);
    // Anchor comes from the earlier-starting span.
    assert_eq!((merged.line, merged.column), (1, 3));
    // Merge is commutative over the covered range.
    assert_eq!(b.merge(a).start, 2);
    assert_eq!(b.merge(a).end, 14);
}

#[test]
fn span_len_and_is_empty() {
    assert!(Span::new(1, 1).is_empty());
    let s = Span::at(4, 9, 1, 5);
    assert_eq!(s.len(), 5);
    assert!(!s.is_empty());
}

#[test]
fn source_file_maps_offsets_to_line_and_column() {
    let file = SourceFile::new("demo.rag", "graph g\n  node a\n");
    // `g` is on line 1.
    assert_eq!(file.location(6), (1, 7));
    // The `node` keyword starts at byte 10 on line 2, column 3.
    let node_byte = file.text().find("node").unwrap();
    assert_eq!(file.location(node_byte), (2, 3));
    assert_eq!(file.line_text(2), Some("  node a"));
    assert_eq!(
        file.snippet(Span::at(node_byte, node_byte + 4, 2, 3)),
        "node"
    );
}

#[test]
fn source_map_assigns_ids_and_resolves_files() {
    let mut map = SourceMap::new();
    assert!(map.is_empty());
    let a = map.add("a.rag", "graph a {}");
    let b = map.add("b.rag", "graph b {}");
    assert_eq!(map.len(), 2);
    assert_ne!(a, b);
    assert_eq!(map.get(a).unwrap().name(), "a.rag");
    assert_eq!(map.get(b).unwrap().text(), "graph b {}");
}

#[test]
fn diagnostic_renders_caret_under_primary_span() {
    let source = "graph g {\n  tool_call -> toolz\n}\n";
    let file = SourceFile::new("support.rag", source);
    let target = source.find("toolz").unwrap();
    let span = Span::at(target, target + "toolz".len(), 2, 16);
    let rendered = Diagnostic::error("route target `toolz` does not exist", span)
        .with_code("E-rag-unknown-node")
        .with_primary_label("unknown node")
        .with_help("did you mean `tools`?")
        .render(&file);

    assert!(
        rendered.contains("error[E-rag-unknown-node]: route target `toolz` does not exist"),
        "{rendered}"
    );
    assert!(rendered.contains("--> support.rag:2:16"), "{rendered}");
    assert!(rendered.contains("tool_call -> toolz"), "{rendered}");
    // Five carets under the five characters of `toolz`, plus the label.
    assert!(rendered.contains("^^^^^ unknown node"), "{rendered}");
    assert!(
        rendered.contains("help: did you mean `tools`?"),
        "{rendered}"
    );
}

#[test]
fn diagnostic_renders_span_past_end_of_source_without_panic() {
    // A span whose bytes extend past (or start past) the end of the source must
    // not panic when rendered — the caret range is clamped into the line.
    let source = "graph g {}\n";
    let file = SourceFile::new("plan.rag", source);
    let past = source.len() + 50;
    let span = Span::at(past, past + 10, 99, 1);
    let rendered = Diagnostic::error("dangling span", span)
        .with_primary_label("here")
        .render(&file);
    assert!(rendered.contains("error: dangling span"), "{rendered}");
    // At least one caret is emitted even for an empty clamped range.
    assert!(rendered.contains('^'), "{rendered}");
}

#[test]
fn into_parse_error_honors_stored_line_col_for_back_compat_spans_even_with_source() {
    // `Span::new(line, column)` is the back-compat constructor for callers
    // that only have a line/column, not a byte offset — `start`/`end` are
    // both left at 0. Even when a `SourceFile` is supplied, resolving byte
    // offset 0 against it would always yield 1:1, silently discarding the
    // real position the caller anchored the span at.
    let source = "graph g {\n  start missing\n}\n";
    let file = SourceFile::new("flow.rag", source);
    let span = Span::new(2, 9);
    let diagnostic = Diagnostic::error("unknown start node", span).with_primary_label("here");

    let err = diagnostic.into_parse_error(Some(&file));
    match err {
        crate::error::TinyAgentsError::Parse {
            line,
            column,
            message,
        } => {
            assert_eq!(
                (line, column),
                (2, 9),
                "must honor the span's stored anchor"
            );
            assert!(message.contains("unknown start node"), "{message}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn into_parse_error_uses_real_offsets_when_present() {
    // A span with real byte offsets (built via `Span::at`) must still resolve
    // its line/column from the source, not just echo the stored anchor —
    // this pins the happy path the previous test's fix must not regress.
    let source = "graph g {\n  start missing\n}\n";
    let file = SourceFile::new("flow.rag", source);
    let offset = source.find("missing").unwrap();
    let span = Span::at(offset, offset + "missing".len(), 2, 9);
    let diagnostic = Diagnostic::error("unknown start node", span).with_primary_label("here");

    let err = diagnostic.into_parse_error(Some(&file));
    match err {
        crate::error::TinyAgentsError::Parse { line, column, .. } => {
            assert_eq!((line, column), (2, 9));
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn severity_labels_are_lowercase() {
    assert_eq!(Severity::Error.label(), "error");
    assert_eq!(Severity::Warning.label(), "warning");
    assert_eq!(Severity::Note.label(), "note");
}

#[test]
fn parse_error_carries_rendered_caret_for_source() {
    // `bogus` is not a valid node item; `parse_str` has the source so the error
    // message should render a caret beneath the offending token.
    let err = parse_str("graph g {\n  node a { bogus x }\n}\n").unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse {
            message,
            line,
            column,
        } => {
            assert!(message.contains("unknown node item `bogus`"), "{message}");
            assert!(message.contains('^'), "{message}");
            assert!(message.contains("--> <source>:2:12"), "{message}");
            assert_eq!((line, column), (2, 12));
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_error_without_source_renders_plain() {
    // The token-only `parse` entry point has no source text, so the rendered
    // message falls back to the source-free presentation (no caret).
    let tokens = tokenize("graph { }").unwrap();
    let err = parse(&tokens).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("expected identifier"), "{message}");
            assert!(!message.contains('^'), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn parse_empty_token_slice_returns_error_not_panic() {
    // A well-formed token stream always ends with an `Eof` sentinel; an empty
    // slice violates that contract and previously underflowed `len() - 1`.
    // `parse` must return a parse error instead of panicking.
    let err = parse(&[]).unwrap_err();
    match err {
        crate::error::TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("empty token stream"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}
