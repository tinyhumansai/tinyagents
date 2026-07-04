//! Lexer tests (tokenization, escapes, spans, literal formatting).
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[test]
fn tokenizes_punctuation_and_arrow() {
    let tokens = tokenize("a -> b { } [ ] ,").unwrap();
    let kinds: Vec<_> = tokens.into_iter().map(|t| t.token).collect();
    assert_eq!(
        kinds,
        vec![
            Token::Ident("a".into()),
            Token::Arrow,
            Token::Ident("b".into()),
            Token::LBrace,
            Token::RBrace,
            Token::LBracket,
            Token::RBracket,
            Token::Comma,
            Token::Eof,
        ]
    );
}

#[test]
fn tokenizes_strings_numbers_and_comments() {
    let tokens = tokenize("// comment\n\"hi\\n\" 50 1.5 -3").unwrap();
    let kinds: Vec<_> = tokens.into_iter().map(|t| t.token).collect();
    assert_eq!(
        kinds,
        vec![
            Token::Str("hi\n".into()),
            Token::Num(50.0),
            Token::Num(1.5),
            Token::Num(-3.0),
            Token::Eof,
        ]
    );
}

#[test]
fn tracks_line_and_column_spans() {
    let tokens = tokenize("graph\n  foo").unwrap();
    assert_eq!(tokens[0].span.line, 1);
    assert_eq!(tokens[0].span.column, 1);
    assert_eq!(tokens[1].span.line, 2);
    assert_eq!(tokens[1].span.column, 3);
}

#[test]
fn unterminated_string_is_a_parse_error() {
    let err = tokenize("\"oops").unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Parse { .. }));
}

#[test]
fn invalid_escape_is_a_parse_error() {
    let err = tokenize("\"bad\\x\"").unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Parse { .. }));
}

#[test]
fn literal_as_display_does_not_saturate_huge_floats() {
    // A huge finite float must not be truncated to i64::MAX; it should render
    // using the float's own formatting instead.
    let huge = Literal::Num(1e30);
    assert_eq!(huge.as_display(), format!("{}", 1e30_f64));
    assert_ne!(huge.as_display(), format!("{}", i64::MAX));

    let nan = Literal::Num(f64::NAN);
    assert_eq!(nan.as_display(), "NaN");

    let inf = Literal::Num(f64::INFINITY);
    assert_eq!(inf.as_display(), "inf");
}
