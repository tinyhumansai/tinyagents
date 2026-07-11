//! Feature tests for the source/span/diagnostic substrate
//! ([`tinyagents::language::source`], [`span`], [`diagnostic`]).
//!
//! These cover the fiddly coordinate and rendering behaviours the existing
//! suite touches only lightly: CRLF line handling, Unicode-aware column
//! counting, snippet/caret clamping for spans that point past the end of the
//! text, [`Span::merge`] when the arguments are out of order, secondary-label
//! rendering, and the two [`Diagnostic::into_parse_error`] branches (byte-offset
//! spans versus back-compat line/column-only spans).

use tinyagents::{Diagnostic, Severity, SourceFile, SourceId, SourceMap, Span, TinyAgentsError};

#[test]
fn crlf_line_text_strips_the_trailing_carriage_return() {
    let file = SourceFile::new("crlf.rag", "ab\r\ncd");
    assert_eq!(file.line_count(), 2);
    assert_eq!(file.line_text(1), Some("ab"));
    assert_eq!(file.line_text(2), Some("cd"));
    // A line past the end has no text.
    assert_eq!(file.line_text(3), None);
    assert_eq!(file.line_range(3), None);
}

#[test]
fn location_columns_count_unicode_scalars_not_bytes() {
    // "αβ " is three characters but five bytes; `x` sits at byte offset 5.
    let file = SourceFile::new("u.rag", "αβ x");
    let x_offset = file.text().find('x').unwrap();
    assert_eq!(x_offset, 5);
    assert_eq!(file.location(x_offset), (1, 4));
}

#[test]
fn snippet_clamps_a_span_that_points_past_the_end() {
    let file = SourceFile::new("short.rag", "abc");
    let past_end = Span::at(100, 200, 1, 50);
    assert_eq!(file.snippet(past_end), "");
    // A location past the end clamps to the final position rather than panicking.
    assert_eq!(file.location(999), (1, 4));
}

#[test]
fn span_merge_is_order_independent() {
    let early = Span::at(2, 5, 1, 3);
    let late = Span::at(10, 14, 2, 1);

    let forward = early.merge(late);
    let backward = late.merge(early);

    assert_eq!(forward.start, 2);
    assert_eq!(forward.end, 14);
    // The line/column anchor comes from whichever input begins earlier, so both
    // orders agree.
    assert_eq!(forward.line, 1);
    assert_eq!(forward.column, 3);
    assert_eq!(forward, backward);
}

#[test]
fn a_line_column_only_span_is_empty_and_has_no_length() {
    let span = Span::new(4, 2);
    assert!(span.is_empty());
    assert_eq!(span.len(), 0);
    assert_eq!((span.line, span.column), (4, 2));
}

#[test]
fn render_draws_a_block_for_the_primary_and_each_secondary_label() {
    let source = "graph g {\n  start missing\n}\n";
    let file = SourceFile::new("flow.rag", source);
    let start = source.find("missing").unwrap();
    let primary = Span::at(start, start + "missing".len(), 2, 9);
    let secondary = Span::at(0, 5, 1, 1);

    let rendered = Diagnostic::error("unknown start node", primary)
        .with_code("E-rag-start")
        .with_primary_label("not declared")
        .with_label(secondary, "graph begins here")
        .render(&file);

    // One `-->` anchor for the primary span, one for the secondary label.
    assert_eq!(rendered.matches("-->").count(), 2, "{rendered}");
    assert!(rendered.contains("not declared"), "{rendered}");
    assert!(rendered.contains("graph begins here"), "{rendered}");
    assert!(rendered.contains("error[E-rag-start]"), "{rendered}");
}

#[test]
fn rendering_a_span_past_the_end_of_the_source_does_not_panic() {
    let file = SourceFile::new("tiny.rag", "graph g {}");
    let rendered = Diagnostic::error("out of range", Span::at(500, 600, 9, 9))
        .with_primary_label("here")
        .render(&file);
    assert!(rendered.contains("out of range"), "{rendered}");
}

#[test]
fn into_parse_error_resolves_line_and_column_from_a_byte_offset() {
    let file = SourceFile::new("f.rag", "ab\ncd\n");
    let c_offset = file.text().find('c').unwrap();
    // The stored 99/99 anchor is deliberately wrong; the byte offset must win.
    let span = Span::at(c_offset, c_offset + 1, 99, 99);
    let err = Diagnostic::error("boom", span)
        .with_primary_label("here")
        .into_parse_error(Some(&file));
    match err {
        TinyAgentsError::Parse { line, column, .. } => assert_eq!((line, column), (2, 1)),
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn into_parse_error_uses_a_line_column_only_span_verbatim() {
    let file = SourceFile::new("f.rag", "graph g {}");
    // A `Span::new` span has no byte offsets, so its stored line/column is used
    // directly and the source-free rendering is produced even with a file.
    let err = Diagnostic::error("headline", Span::new(3, 7))
        .with_help("try this")
        .into_parse_error(Some(&file));
    match err {
        TinyAgentsError::Parse {
            line,
            column,
            message,
        } => {
            assert_eq!((line, column), (3, 7));
            assert!(message.contains("--> 3:7"), "{message}");
            assert!(message.contains("help: try this"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn severity_labels_are_stable() {
    assert_eq!(Severity::Error.label(), "error");
    assert_eq!(Severity::Warning.label(), "warning");
    assert_eq!(Severity::Note.label(), "note");
}

#[test]
fn a_source_map_assigns_sequential_ids_and_looks_files_up() {
    let mut map = SourceMap::new();
    assert!(map.is_empty());
    let a = map.add("a.rag", "graph a {}");
    let b = map.add("b.rag", "graph b {}");
    assert_eq!(a, SourceId(0));
    assert_eq!(b, SourceId(1));
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(a).unwrap().name(), "a.rag");
    assert_eq!(map.get(b).unwrap().text(), "graph b {}");
    assert!(map.get(SourceId(5)).is_none());
    assert_eq!(
        map.files().map(SourceFile::name).collect::<Vec<_>>(),
        vec!["a.rag", "b.rag"]
    );
}
