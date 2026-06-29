//! Parser: turns a [`SpannedToken`] stream into a [`Program`] AST.
//!
//! The parser is a small hand-written recursive-descent parser over the
//! grammar described in `docs/modules/expressive-language/README.md`. It
//! performs *structural* validation only (expected-token checks, well-formed
//! blocks); *semantic* validation (duplicate names, unknown targets, …) is the
//! job of [`crate::language::compiler`].

use crate::error::{Result, RustAgentsError};
use crate::language::types::{
    ChannelDecl, EdgeDecl, GraphDecl, Literal, NodeDecl, Program, RouteDecl, Span, SpannedToken,
    Token,
};

/// Tokenises and parses `source` in one step.
///
/// # Errors
///
/// Returns [`RustAgentsError::Parse`] for any lexical or structural error.
pub fn parse_str(source: &str) -> Result<Program> {
    let tokens = crate::language::lexer::tokenize(source)?;
    parse(&tokens)
}

/// Parses a token slice produced by [`crate::language::lexer::tokenize`].
///
/// # Errors
///
/// Returns [`RustAgentsError::Parse`] when the token stream does not match the
/// grammar, with the span of the offending token.
pub fn parse(tokens: &[SpannedToken]) -> Result<Program> {
    Parser { tokens, pos: 0 }.parse_program()
}

struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
}

impl Parser<'_> {
    // ---- token cursor helpers -------------------------------------------

    fn current(&self) -> &SpannedToken {
        // The lexer always appends an `Eof`, so the last token is a valid
        // sentinel even once `pos` reaches the end.
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn span(&self) -> Span {
        self.current().span
    }

    fn at_eof(&self) -> bool {
        matches!(self.current().token, Token::Eof)
    }

    fn advance(&mut self) -> SpannedToken {
        let tok = self.current().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn error(&self, message: impl Into<String>, span: Span) -> RustAgentsError {
        RustAgentsError::Parse {
            message: message.into(),
            line: span.line,
            column: span.column,
        }
    }

    /// Expects an exact punctuation/structural token.
    fn expect(&mut self, expected: &Token) -> Result<Span> {
        let tok = self.current().clone();
        if &tok.token == expected {
            self.advance();
            Ok(tok.span)
        } else {
            Err(self.error(
                format!(
                    "expected {}, found {}",
                    expected.describe(),
                    tok.token.describe()
                ),
                tok.span,
            ))
        }
    }

    /// Expects an identifier and returns its text.
    fn expect_ident(&mut self) -> Result<(String, Span)> {
        let tok = self.current().clone();
        match tok.token {
            Token::Ident(s) => {
                self.advance();
                Ok((s, tok.span))
            }
            other => Err(self.error(
                format!("expected identifier, found {}", other.describe()),
                tok.span,
            )),
        }
    }

    /// Expects a string literal and returns its value.
    fn expect_string(&mut self) -> Result<String> {
        let tok = self.current().clone();
        match tok.token {
            Token::Str(s) => {
                self.advance();
                Ok(s)
            }
            other => Err(self.error(
                format!("expected string, found {}", other.describe()),
                tok.span,
            )),
        }
    }

    /// Expects an identifier with a specific keyword spelling.
    fn expect_keyword(&mut self, keyword: &str) -> Result<Span> {
        let tok = self.current().clone();
        match &tok.token {
            Token::Ident(s) if s == keyword => {
                self.advance();
                Ok(tok.span)
            }
            other => Err(self.error(
                format!("expected `{keyword}`, found {}", other.describe()),
                tok.span,
            )),
        }
    }

    fn is_keyword(&self, keyword: &str) -> bool {
        matches!(&self.current().token, Token::Ident(s) if s == keyword)
    }

    // ---- grammar productions --------------------------------------------

    fn parse_program(&mut self) -> Result<Program> {
        let mut graphs = Vec::new();
        while !self.at_eof() {
            graphs.push(self.parse_graph()?);
        }
        Ok(Program { graphs })
    }

    fn parse_graph(&mut self) -> Result<GraphDecl> {
        let span = self.expect_keyword("graph")?;
        let (name, _) = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut graph = GraphDecl {
            name,
            span,
            start: None,
            defaults: Vec::new(),
            channels: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside graph body", self.span()));
            }
            self.parse_graph_item(&mut graph)?;
        }
        self.expect(&Token::RBrace)?;
        Ok(graph)
    }

    fn parse_graph_item(&mut self, graph: &mut GraphDecl) -> Result<()> {
        if self.is_keyword("start") {
            self.advance();
            let (name, _) = self.expect_ident()?;
            graph.start = Some(name);
        } else if self.is_keyword("defaults") {
            self.advance();
            graph.defaults = self.parse_defaults_block()?;
        } else if self.is_keyword("channel") {
            graph.channels.push(self.parse_channel()?);
        } else if self.is_keyword("node") {
            graph.nodes.push(self.parse_node()?);
        } else {
            // The only remaining production is an edge: `ident -> target`.
            graph.edges.push(self.parse_edge()?);
        }
        Ok(())
    }

    fn parse_defaults_block(&mut self) -> Result<Vec<(String, Literal)>> {
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside `defaults`", self.span()));
            }
            let (key, _) = self.expect_ident()?;
            let value = self.parse_literal()?;
            entries.push((key, value));
        }
        self.expect(&Token::RBrace)?;
        Ok(entries)
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        let tok = self.current().clone();
        match tok.token {
            Token::Str(s) => {
                self.advance();
                Ok(Literal::Str(s))
            }
            Token::Num(n) => {
                self.advance();
                Ok(Literal::Num(n))
            }
            Token::Ident(s) => {
                self.advance();
                Ok(Literal::Ident(s))
            }
            other => Err(self.error(
                format!("expected a literal value, found {}", other.describe()),
                tok.span,
            )),
        }
    }

    fn parse_channel(&mut self) -> Result<ChannelDecl> {
        let span = self.expect_keyword("channel")?;
        let (name, _) = self.expect_ident()?;
        let (reducer, _) = self.expect_ident()?;
        Ok(ChannelDecl {
            name,
            reducer,
            span,
        })
    }

    fn parse_edge(&mut self) -> Result<EdgeDecl> {
        let (from, span) = self.expect_ident()?;
        self.expect(&Token::Arrow)?;
        let to = self.parse_node_ref()?;
        Ok(EdgeDecl { from, to, span })
    }

    /// Parses a node reference: an identifier or the reserved `END`.
    fn parse_node_ref(&mut self) -> Result<String> {
        let (name, _) = self.expect_ident()?;
        Ok(name)
    }

    fn parse_node(&mut self) -> Result<NodeDecl> {
        let span = self.expect_keyword("node")?;
        let (name, _) = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut node = NodeDecl {
            name,
            kind: None,
            model: None,
            prompt: None,
            tools: Vec::new(),
            next: None,
            routes: Vec::new(),
            span,
        };

        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside node body", self.span()));
            }
            self.parse_node_item(&mut node)?;
        }
        self.expect(&Token::RBrace)?;
        Ok(node)
    }

    fn parse_node_item(&mut self, node: &mut NodeDecl) -> Result<()> {
        let tok = self.current().clone();
        let Token::Ident(keyword) = &tok.token else {
            return Err(self.error(
                format!(
                    "expected a node item keyword, found {}",
                    tok.token.describe()
                ),
                tok.span,
            ));
        };

        match keyword.as_str() {
            "kind" => {
                self.advance();
                let (k, _) = self.expect_ident()?;
                node.kind = Some(k);
            }
            "model" => {
                self.advance();
                node.model = Some(self.expect_string()?);
            }
            // `prompt` and `system` both populate the node prompt; `system`
            // is accepted as an alias for forward compatibility.
            "prompt" | "system" => {
                self.advance();
                node.prompt = Some(self.expect_string()?);
            }
            "tools" => {
                self.advance();
                node.tools = self.parse_string_list()?;
            }
            "next" => {
                self.advance();
                node.next = Some(self.parse_node_ref()?);
            }
            "routes" => {
                self.advance();
                node.routes = self.parse_routes_block()?;
            }
            other => {
                return Err(self.error(format!("unknown node item `{other}`"), tok.span));
            }
        }
        Ok(())
    }

    fn parse_string_list(&mut self) -> Result<Vec<String>> {
        self.expect(&Token::LBracket)?;
        let mut items = Vec::new();
        while !matches!(self.current().token, Token::RBracket) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside list", self.span()));
            }
            items.push(self.expect_string()?);
            // Optional comma separator.
            if matches!(self.current().token, Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(items)
    }

    fn parse_routes_block(&mut self) -> Result<Vec<RouteDecl>> {
        self.expect(&Token::LBrace)?;
        let mut routes = Vec::new();
        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside `routes`", self.span()));
            }
            let (label, span) = self.expect_ident()?;
            self.expect(&Token::Arrow)?;
            let target = self.parse_node_ref()?;
            routes.push(RouteDecl {
                label,
                target,
                span,
            });
        }
        self.expect(&Token::RBrace)?;
        Ok(routes)
    }
}
