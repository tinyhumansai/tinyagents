//! Parser: turns a [`SpannedToken`] stream into a [`Program`] AST.
//!
//! Middle stage of the `.rag` pipeline. It gives a declarative (possibly
//! self-authored) plan its tree shape without granting it any power: the grammar
//! admits only graph/node/route/capability *declarations*, so there is no place
//! for arbitrary code to hide — the structural counterpart to the registry
//! binding that [`crate::language::compiler`] enforces later.
//!
//! The parser is a small hand-written recursive-descent parser over the
//! grammar described in `docs/modules/expressive-language/README.md`. It
//! performs *structural* validation only (expected-token checks, well-formed
//! blocks); *semantic* validation (duplicate names, unknown targets, …) is the
//! job of [`crate::language::compiler`].

use crate::error::{Result, TinyAgentsError};
use crate::language::diagnostic::Diagnostic;
use crate::language::source::SourceFile;
use crate::language::types::{
    ChannelDecl, CommandDecl, EdgeDecl, GraphDecl, IoFieldDecl, JoinDecl, Literal, NodeDecl,
    Program, RouteDecl, SendDecl, Span, SpannedToken, Token,
};

/// Tokenises and parses `source` in one step.
///
/// Parse errors carry a caret-underline rendering of the offending source.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Parse`] for any lexical or structural error.
pub fn parse_str(source: &str) -> Result<Program> {
    let file = SourceFile::anonymous(source);
    let tokens = crate::language::lexer::tokenize(source)?;
    Parser {
        tokens: &tokens,
        pos: 0,
        source: Some(&file),
    }
    .parse_program()
}

/// Parses a token slice produced by [`crate::language::lexer::tokenize`].
///
/// Without the original source text, parse errors render a source-free
/// presentation (headline plus `line:column` anchor). Use [`parse_str`] to get
/// caret-underlined diagnostics.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Parse`] when the token stream does not match the
/// grammar, with the span of the offending token.
pub fn parse(tokens: &[SpannedToken]) -> Result<Program> {
    Parser {
        tokens,
        pos: 0,
        source: None,
    }
    .parse_program()
}

struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
    source: Option<&'a SourceFile>,
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

    fn error(&self, message: impl Into<String>, span: Span) -> TinyAgentsError {
        Diagnostic::error(message, span)
            .with_primary_label("here")
            .into_parse_error(self.source)
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
            input: Vec::new(),
            output: Vec::new(),
            checkpoint: None,
            interrupt: None,
            channels: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            joins: Vec::new(),
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
        } else if self.is_keyword("input") {
            self.advance();
            graph.input = self.parse_io_shape_block()?;
        } else if self.is_keyword("output") {
            self.advance();
            graph.output = self.parse_io_shape_block()?;
        } else if self.is_keyword("checkpoint") {
            self.advance();
            let (policy, _) = self.expect_ident()?;
            graph.checkpoint = Some(policy);
        } else if self.is_keyword("interrupt") {
            self.advance();
            let (policy, _) = self.expect_ident()?;
            graph.interrupt = Some(policy);
        } else if self.is_keyword("channel") {
            graph.channels.push(self.parse_channel()?);
        } else if self.is_keyword("join") {
            graph.joins.push(self.parse_join()?);
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
        // Optional reducer policy arguments. Restricted to string/number
        // literals so they cannot be confused with the next declaration's
        // leading keyword (e.g. `node`, `channel`).
        let mut args = Vec::new();
        loop {
            match &self.current().token {
                Token::Str(s) => {
                    let s = s.clone();
                    self.advance();
                    args.push(Literal::Str(s));
                }
                Token::Num(n) => {
                    let n = *n;
                    self.advance();
                    args.push(Literal::Num(n));
                }
                _ => break,
            }
        }
        Ok(ChannelDecl {
            name,
            reducer,
            args,
            span,
        })
    }

    /// Parses a graph `input`/`output` shape block: `{ name type … }`.
    fn parse_io_shape_block(&mut self) -> Result<Vec<IoFieldDecl>> {
        self.expect(&Token::LBrace)?;
        let mut fields = Vec::new();
        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside shape block", self.span()));
            }
            let (name, span) = self.expect_ident()?;
            let (ty, _) = self.expect_ident()?;
            fields.push(IoFieldDecl { name, ty, span });
        }
        self.expect(&Token::RBrace)?;
        Ok(fields)
    }

    /// Parses a top-level `join [a, b] -> c` declaration.
    fn parse_join(&mut self) -> Result<JoinDecl> {
        let span = self.expect_keyword("join")?;
        let sources = self.parse_ident_list()?;
        self.expect(&Token::Arrow)?;
        let target = self.parse_node_ref()?;
        Ok(JoinDecl {
            sources,
            target,
            span,
        })
    }

    /// Parses a bracketed, comma-separated identifier list: `[a, b, c]`.
    fn parse_ident_list(&mut self) -> Result<Vec<String>> {
        self.expect(&Token::LBracket)?;
        let mut items = Vec::new();
        while !matches!(self.current().token, Token::RBracket) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside list", self.span()));
            }
            let (name, _) = self.expect_ident()?;
            items.push(name);
            if matches!(self.current().token, Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(items)
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

        let mut node = NodeDecl::empty(name, span);

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
            "agent" => {
                self.advance();
                node.agent = Some(self.expect_string()?);
            }
            "graph" => {
                self.advance();
                node.graph = Some(self.expect_string()?);
            }
            "script" => {
                self.advance();
                node.script = Some(self.expect_string()?);
            }
            "input" => {
                self.advance();
                node.input = Some(self.expect_string()?);
            }
            "command" => {
                self.advance();
                node.command = Some(self.parse_command_block()?);
            }
            "sends" => {
                self.advance();
                node.sends = self.parse_sends_block()?;
            }
            "sources" => {
                self.advance();
                node.sources = self.parse_ident_list()?;
            }
            "options" => {
                self.advance();
                node.options = self.parse_string_list()?;
            }
            "checkpoint" => {
                self.advance();
                let (policy, _) = self.expect_ident()?;
                node.checkpoint = Some(policy);
            }
            "timeout" => {
                self.advance();
                node.timeout = Some(self.parse_literal()?);
            }
            "retry" => {
                self.advance();
                node.retry = self.parse_defaults_block()?;
            }
            "metadata" => {
                self.advance();
                node.metadata = self.parse_defaults_block()?;
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

    /// Parses a `command { goto <target> update { … } }` block. The `command`
    /// keyword has already been consumed.
    fn parse_command_block(&mut self) -> Result<CommandDecl> {
        let span = self.span();
        self.expect(&Token::LBrace)?;
        let mut goto = None;
        let mut update = Vec::new();
        while !matches!(self.current().token, Token::RBrace) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside `command`", self.span()));
            }
            if self.is_keyword("goto") {
                self.advance();
                goto = Some(self.parse_node_ref()?);
            } else if self.is_keyword("update") {
                self.advance();
                update = self.parse_defaults_block()?;
            } else {
                return Err(self.error("expected `goto` or `update` inside `command`", self.span()));
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(CommandDecl { goto, update, span })
    }

    /// Parses a `sends [ send <node> ["input"] … ]` block. The `sends` keyword
    /// has already been consumed.
    fn parse_sends_block(&mut self) -> Result<Vec<SendDecl>> {
        self.expect(&Token::LBracket)?;
        let mut sends = Vec::new();
        while !matches!(self.current().token, Token::RBracket) {
            if self.at_eof() {
                return Err(self.error("unexpected end of input inside `sends`", self.span()));
            }
            let span = self.expect_keyword("send")?;
            let target = self.parse_node_ref()?;
            let input = if matches!(self.current().token, Token::Str(_)) {
                Some(self.expect_string()?)
            } else {
                None
            };
            sends.push(SendDecl {
                target,
                input,
                span,
            });
            if matches!(self.current().token, Token::Comma) {
                self.advance();
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(sends)
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
