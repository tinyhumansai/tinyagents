//! Lexer: turns `.rag` source text into a stream of [`SpannedToken`]s.
//!
//! The lexer is deliberately tiny and side-effect free. It recognises:
//!
//! - identifiers / keywords: `[A-Za-z_][A-Za-z0-9_]*`
//! - numbers: integer or decimal, optionally signed (`50`, `1.5`, `-3`)
//! - double-quoted strings with `\n`, `\t`, `\r`, `\\`, `\"` escapes
//! - punctuation: `{ } [ ] ,` and the arrow `->`
//! - `//` line comments (skipped)
//!
//! Lexical errors surface as [`TinyAgentsError::Parse`] with the offending
//! line and column.

use crate::error::{Result, TinyAgentsError};
use crate::language::types::{Span, SpannedToken, Token};

/// Tokenises `source` into a vector of [`SpannedToken`]s terminated by a
/// single [`Token::Eof`].
///
/// Line and column numbers are 1-based. Each returned span points at the first
/// character of its token.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Parse`] for an unterminated string, an invalid
/// escape sequence, a malformed number, a lone `-` not forming `->` or a
/// number, or any otherwise unrecognised character.
pub fn tokenize(source: &str) -> Result<Vec<SpannedToken>> {
    Lexer::new(source).run()
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    fn span(&self) -> Span {
        Span::new(self.line, self.column)
    }

    fn error(&self, message: impl Into<String>, span: Span) -> TinyAgentsError {
        TinyAgentsError::Parse {
            message: message.into(),
            line: span.line,
            column: span.column,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    /// Advances one character, maintaining line/column counters.
    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
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
            let span = self.span();
            let Some(c) = self.peek() else {
                tokens.push(SpannedToken {
                    token: Token::Eof,
                    span,
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
                '"' => self.lex_string(span)?,
                c if c.is_ascii_digit() => self.lex_number(span)?,
                '-' if self.peek_at(1).is_some_and(|n| n.is_ascii_digit()) => {
                    self.lex_number(span)?
                }
                c if is_ident_start(c) => self.lex_ident(),
                other => {
                    return Err(self.error(format!("unexpected character `{other}`"), span));
                }
            };

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

    fn lex_number(&mut self, span: Span) -> Result<Token> {
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
        s.parse::<f64>()
            .map(Token::Num)
            .map_err(|_| self.error(format!("invalid number `{s}`"), span))
    }

    fn lex_string(&mut self, span: Span) -> Result<Token> {
        // Consume the opening quote.
        self.bump();
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated string", span)),
                Some('"') => {
                    self.bump();
                    return Ok(Token::Str(s));
                }
                Some('\\') => {
                    let esc_span = self.span();
                    self.bump();
                    match self.bump() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('r') => s.push('\r'),
                        Some('\\') => s.push('\\'),
                        Some('"') => s.push('"'),
                        Some(other) => {
                            return Err(self.error(format!("invalid escape `\\{other}`"), esc_span));
                        }
                        None => return Err(self.error("unterminated string", span)),
                    }
                }
                Some('\n') => return Err(self.error("unterminated string", span)),
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
