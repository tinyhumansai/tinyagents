//! Lexer: turns `.rag` source text into a stream of [`SpannedToken`]s.
//!
//! First stage of the pipeline that lets the runtime ingest declarative plans —
//! including ones a model authored about itself. The lexer is deliberately tiny
//! and side-effect free: a small, auditable surface is part of what keeps
//! agent-authored source a safe input. It recognises:
//!
//! - identifiers / keywords: `[A-Za-z_][A-Za-z0-9_]*`
//! - numbers: integer or decimal, optionally signed (`50`, `1.5`, `-3`)
//! - double-quoted strings with `\n`, `\t`, `\r`, `\\`, `\"` escapes
//! - punctuation: `{ } [ ] ,` and the arrow `->`
//! - `//` line comments (skipped)
//!
//! Each [`SpannedToken`] carries a real byte range plus a 1-based line/column
//! anchor. Lexical errors are built as a structured
//! [`crate::language::diagnostic::Diagnostic`] and surfaced as
//! [`TinyAgentsError::Parse`] with the rendered caret and the offending line and
//! column.

use crate::error::{Result, TinyAgentsError};
use crate::language::diagnostic::Diagnostic;
use crate::language::source::SourceFile;
use crate::language::span::Span;
use crate::language::types::{SpannedToken, Token};

/// Tokenises `source` into a vector of [`SpannedToken`]s terminated by a
/// single [`Token::Eof`].
///
/// Line and column numbers are 1-based. Each returned span covers the token's
/// byte range, anchored at its first character.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Parse`] for an unterminated string, an invalid
/// escape sequence, a malformed number, a lone `-` not forming `->` or a
/// number, or any otherwise unrecognised character. The error message carries a
/// rendered caret pointing at the offending source.
pub fn tokenize(source: &str) -> Result<Vec<SpannedToken>> {
    Lexer::new(source).run()
}

struct Lexer<'a> {
    src: &'a str,
    chars: Vec<char>,
    /// Index into `chars`.
    pos: usize,
    /// Byte offset into `src`.
    byte: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            chars: src.chars().collect(),
            pos: 0,
            byte: 0,
            line: 1,
            column: 1,
        }
    }

    /// Builds a [`TinyAgentsError::Parse`] from a lexical [`Diagnostic`],
    /// rendered against the source.
    fn error(&self, message: impl Into<String>, span: Span) -> TinyAgentsError {
        let file = SourceFile::anonymous(self.src);
        Diagnostic::error(message, span)
            .with_primary_label("here")
            .into_parse_error(Some(&file))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    /// Advances one character, maintaining byte/line/column counters.
    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        self.byte += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn run(mut self) -> Result<Vec<SpannedToken>> {
        let mut tokens = Vec::new();
        loop {
            self.skip_trivia();
            let start_byte = self.byte;
            let start_line = self.line;
            let start_column = self.column;
            let start = Span::at(start_byte, start_byte, start_line, start_column);

            let Some(c) = self.peek() else {
                tokens.push(SpannedToken {
                    token: Token::Eof,
                    span: start,
                });
                return Ok(tokens);
            };

            let token = match c {
                '{' => {
                    self.bump();
                    Token::LBrace
                }
                '}' => {
                    self.bump();
                    Token::RBrace
                }
                '[' => {
                    self.bump();
                    Token::LBracket
                }
                ']' => {
                    self.bump();
                    Token::RBracket
                }
                ',' => {
                    self.bump();
                    Token::Comma
                }
                '-' if self.peek_at(1) == Some('>') => {
                    self.bump();
                    self.bump();
                    Token::Arrow
                }
                '"' => self.lex_string(start)?,
                c if c.is_ascii_digit() => self.lex_number(start)?,
                '-' if self.peek_at(1).is_some_and(|n| n.is_ascii_digit()) => {
                    self.lex_number(start)?
                }
                c if is_ident_start(c) => self.lex_ident(),
                other => {
                    self.bump();
                    let span = Span::at(start_byte, self.byte, start_line, start_column);
                    return Err(self.error(format!("unexpected character `{other}`"), span));
                }
            };

            let span = Span::at(start_byte, self.byte, start_line, start_column);
            tokens.push(SpannedToken { token, span });
        }
    }

    /// Skips whitespace and `//` line comments.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek_at(1) == Some('/') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    fn lex_ident(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Token::Ident(s)
    }

    fn lex_number(&mut self, start: Span) -> Result<Token> {
        let mut s = String::new();
        if self.peek() == Some('-') {
            s.push('-');
            self.bump();
        }
        let mut seen_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else if c == '.' && !seen_dot && self.peek_at(1).is_some_and(|n| n.is_ascii_digit()) {
                seen_dot = true;
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let span = Span::at(start.start, self.byte, start.line, start.column);
        s.parse::<f64>()
            .map(Token::Num)
            .map_err(|_| self.error(format!("invalid number `{s}`"), span))
    }

    fn lex_string(&mut self, start: Span) -> Result<Token> {
        // Consume the opening quote.
        self.bump();
        let mut s = String::new();
        loop {
            match self.peek() {
                None => {
                    let span = Span::at(start.start, self.byte, start.line, start.column);
                    return Err(self.error("unterminated string", span));
                }
                Some('"') => {
                    self.bump();
                    return Ok(Token::Str(s));
                }
                Some('\\') => {
                    let esc_start = self.byte;
                    let esc_line = self.line;
                    let esc_column = self.column;
                    self.bump();
                    match self.bump() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('r') => s.push('\r'),
                        Some('\\') => s.push('\\'),
                        Some('"') => s.push('"'),
                        Some(other) => {
                            let span = Span::at(esc_start, self.byte, esc_line, esc_column);
                            return Err(self.error(format!("invalid escape `\\{other}`"), span));
                        }
                        None => {
                            let span = Span::at(start.start, self.byte, start.line, start.column);
                            return Err(self.error("unterminated string", span));
                        }
                    }
                }
                Some('\n') => {
                    let span = Span::at(start.start, self.byte, start.line, start.column);
                    return Err(self.error("unterminated string", span));
                }
                Some(c) => {
                    s.push(c);
                    self.bump();
                }
            }
        }
    }
}

/// Returns true if `c` can begin an identifier.
fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

/// Returns true if `c` can continue an identifier.
fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}
