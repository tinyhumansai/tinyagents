//! Feature tests for the `.rag` lexer surface
//! ([`tinyagents::language::lexer::tokenize`]).
//!
//! These focus on lexical edge cases that the existing suite does not exercise
//! directly: escape-sequence resolution, signed/decimal numbers, comment and
//! whitespace trivia, Unicode inside strings and Unicode column tracking, an
//! empty input, and every distinct lexical error (unterminated string, invalid
//! escape, a newline inside a string, a lone `-`, and an unexpected character).

use tinyagents::TinyAgentsError;
use tinyagents::language::lexer::tokenize;
use tinyagents::language::types::Token;

/// Collects just the token values (dropping spans) for compact assertions.
fn tokens(source: &str) -> Vec<Token> {
    tokenize(source)
        .expect("source should lex")
        .into_iter()
        .map(|t| t.token)
        .collect()
}

#[test]
fn empty_input_lexes_to_a_single_eof() {
    assert_eq!(tokens(""), vec![Token::Eof]);
}

#[test]
fn whitespace_only_input_lexes_to_a_single_eof() {
    assert_eq!(tokens("   \n\t  \r\n "), vec![Token::Eof]);
}

#[test]
fn all_punctuation_and_arrow_are_distinct_tokens() {
    assert_eq!(
        tokens("{ } [ ] , ->"),
        vec![
            Token::LBrace,
            Token::RBrace,
            Token::LBracket,
            Token::RBracket,
            Token::Comma,
            Token::Arrow,
            Token::Eof,
        ]
    );
}

#[test]
fn line_comments_are_skipped_entirely() {
    let toks = tokens("// leading comment\ngraph // trailing\n{ }");
    assert_eq!(
        toks,
        vec![
            Token::Ident("graph".into()),
            Token::LBrace,
            Token::RBrace,
            Token::Eof,
        ]
    );
}

#[test]
fn integers_decimals_and_signed_numbers_lex_as_f64() {
    assert_eq!(
        tokens("0 50 1.5 -3 -2.25"),
        vec![
            Token::Num(0.0),
            Token::Num(50.0),
            Token::Num(1.5),
            Token::Num(-3.0),
            Token::Num(-2.25),
            Token::Eof,
        ]
    );
}

#[test]
fn a_trailing_dot_does_not_join_the_number() {
    // `1.` has no fractional digit, so the dot is not consumed and is instead
    // an unexpected character error.
    let err = tokenize("1.").expect_err("a bare trailing dot is not lexable");
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn string_escape_sequences_are_resolved() {
    assert_eq!(
        tokens(r#""a\nb\tc\r\\\"d""#),
        vec![Token::Str("a\nb\tc\r\\\"d".into()), Token::Eof]
    );
}

#[test]
fn unicode_inside_strings_is_preserved() {
    assert_eq!(
        tokens("\"café — 世界 🌍\""),
        vec![Token::Str("café — 世界 🌍".into()), Token::Eof]
    );
}

#[test]
fn column_tracking_counts_unicode_scalars_not_bytes() {
    // A multi-byte scalar inside a preceding string must advance the column by
    // one character, not by its byte width. `"é"` is three characters plus a
    // space, so the following `[` sits at 1-based column 5, not column 6+ if
    // bytes were counted.
    let spanned = tokenize("\"é\" [").expect("lexes");
    let bracket = spanned
        .iter()
        .find(|t| t.token == Token::LBracket)
        .expect("bracket token present");
    assert_eq!(bracket.span.column, 5);
    assert_eq!(bracket.span.line, 1);
}

#[test]
fn unterminated_string_is_a_parse_error_with_position() {
    let err = tokenize("\"no closing quote").expect_err("unterminated string fails");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unterminated string"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn a_newline_inside_a_string_terminates_it_as_an_error() {
    let err = tokenize("\"line one\nline two\"").expect_err("newline in string fails");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unterminated string"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn invalid_escape_sequence_is_a_parse_error() {
    let err = tokenize("\"bad \\q escape\"").expect_err("invalid escape fails");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("invalid escape"), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn a_lone_dash_not_forming_arrow_or_number_is_rejected() {
    let err = tokenize("a - b").expect_err("a bare dash is not a token");
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn an_unexpected_character_is_rejected_with_the_offending_char() {
    let err = tokenize("graph @home {}").expect_err("`@` is not lexable");
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("unexpected character"), "{message}");
            assert!(message.contains('@'), "{message}");
        }
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[test]
fn identifiers_allow_underscores_and_digits_but_not_a_leading_digit() {
    assert_eq!(
        tokens("_a1 node_2"),
        vec![
            Token::Ident("_a1".into()),
            Token::Ident("node_2".into()),
            Token::Eof,
        ]
    );
    // A leading digit begins a number, so `1abc` lexes as a number followed by
    // an identifier rather than one identifier.
    assert_eq!(
        tokens("1abc"),
        vec![Token::Num(1.0), Token::Ident("abc".into()), Token::Eof]
    );
}
