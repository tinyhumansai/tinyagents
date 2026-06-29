//! Token and AST types for the expressive language, plus the compiled
//! [`Blueprint`] artifact.
//!
//! This module holds *only* type definitions. The processing logic lives in the
//! sibling modules:
//!
//! - [`crate::language::lexer`] produces [`SpannedToken`]s from source text.
//! - [`crate::language::parser`] turns tokens into a [`Program`] AST.
//! - [`crate::language::compiler`] lowers a [`Program`] into one [`Blueprint`]
//!   per graph and wires blueprints into the runtime graph.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Lexical tokens
// ===========================================================================

/// A single lexical token produced by the lexer.
///
/// Keywords (`graph`, `node`, `start`, …) are not given dedicated variants;
/// they are lexed as [`Token::Ident`] and recognised contextually by the
/// parser. This keeps the token set small and lets identifiers that happen to
/// match a keyword be used as names where the grammar allows it.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// A bare identifier or keyword, e.g. `graph`, `agent`, `messages`.
    Ident(String),
    /// A double-quoted string literal with escapes already resolved.
    Str(String),
    /// A numeric literal, always stored as `f64`.
    Num(f64),
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `->`
    Arrow,
    /// `,`
    Comma,
    /// End of input.
    Eof,
}

impl Token {
    /// Returns a short human-readable description used in parser error
    /// messages (e.g. `"identifier"`, `` "`{`" ``).
    pub fn describe(&self) -> String {
        match self {
            Token::Ident(s) => format!("identifier `{s}`"),
            Token::Str(_) => "string".to_string(),
            Token::Num(_) => "number".to_string(),
            Token::LBrace => "`{`".to_string(),
            Token::RBrace => "`}`".to_string(),
            Token::LBracket => "`[`".to_string(),
            Token::RBracket => "`]`".to_string(),
            Token::Arrow => "`->`".to_string(),
            Token::Comma => "`,`".to_string(),
            Token::Eof => "end of input".to_string(),
        }
    }
}

/// A 1-based line/column source position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number.
    pub column: usize,
}

impl Span {
    /// Creates a new span at the given line and column.
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

/// A [`Token`] paired with the [`Span`] where it begins.
#[derive(Clone, Debug, PartialEq)]
pub struct SpannedToken {
    /// The token value.
    pub token: Token,
    /// The source position of the token's first character.
    pub span: Span,
}

// ===========================================================================
// AST
// ===========================================================================

/// A literal value used in `defaults` entries and similar key/value positions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Literal {
    /// A string literal (`"foo"`).
    Str(String),
    /// A numeric literal (`50`, `1.5`).
    Num(f64),
    /// A bare identifier literal (`inherit`, `exponential`).
    Ident(String),
}

/// The root of a parsed program: one or more graph declarations.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The graphs declared at the top level, in source order.
    pub graphs: Vec<GraphDecl>,
}

/// A `graph <name> { … }` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphDecl {
    /// The graph identifier.
    pub name: String,
    /// The position of the `graph` keyword.
    pub span: Span,
    /// The declared start node, if any (`start <ident>`).
    pub start: Option<String>,
    /// `defaults { key value … }` entries, in source order.
    pub defaults: Vec<(String, Literal)>,
    /// `channel <name> <reducer>` declarations.
    pub channels: Vec<ChannelDecl>,
    /// `node <name> { … }` declarations.
    pub nodes: Vec<NodeDecl>,
    /// Top-level `from -> to` edge declarations.
    pub edges: Vec<EdgeDecl>,
}

/// A `channel <name> <reducer>` declaration binding a state channel to a
/// named reducer.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDecl {
    /// The channel name (e.g. `messages`).
    pub name: String,
    /// The reducer reference (e.g. `append`, `overwrite`, `set_union`).
    pub reducer: String,
    /// Source position of the `channel` keyword.
    pub span: Span,
}

/// A `node <name> { … }` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDecl {
    /// The node name.
    pub name: String,
    /// The declared `kind`, if any (e.g. `agent`, `tool_executor`).
    pub kind: Option<String>,
    /// The bound model name, if any.
    pub model: Option<String>,
    /// The system/user prompt string, if any.
    pub prompt: Option<String>,
    /// Tool capability names referenced by this node.
    pub tools: Vec<String>,
    /// A static `next` successor, if declared.
    pub next: Option<String>,
    /// Conditional `routes { label -> target … }`.
    pub routes: Vec<RouteDecl>,
    /// Source position of the `node` keyword.
    pub span: Span,
}

/// A single `label -> target` route inside a node's `routes` block.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteDecl {
    /// The route label (a named outcome, e.g. `tool_call`).
    pub label: String,
    /// The target node name, or `END`.
    pub target: String,
    /// Source position of the route label.
    pub span: Span,
}

/// A top-level `from -> to` edge declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeDecl {
    /// The source node name.
    pub from: String,
    /// The target node name, or `END`.
    pub to: String,
    /// Source position of the source identifier.
    pub span: Span,
}

// ===========================================================================
// Blueprint (compiled, validated artifact)
// ===========================================================================

/// The reserved virtual terminal node name.
pub const END: &str = "END";

/// A compiled, semantically validated graph plan.
///
/// A `Blueprint` is the inspectable output of the compiler: it is fully
/// serializable so it can be stored, diffed, reviewed, and reloaded
/// independently of the source text. Runnable node *behaviour* is not part of
/// the blueprint — it is supplied later by a Rust-side
/// [`crate::language::compiler::NodeFactory`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Blueprint {
    /// The graph identifier.
    pub graph_id: String,
    /// The validated start node name.
    pub start: String,
    /// State channel specifications.
    pub channels: Vec<ChannelSpec>,
    /// Node specifications.
    pub nodes: Vec<NodeSpec>,
    /// Static edge specifications.
    pub edges: Vec<EdgeSpec>,
    /// Graph default key/value entries.
    pub defaults: Vec<(String, Literal)>,
}

/// A compiled state-channel binding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelSpec {
    /// The channel name.
    pub name: String,
    /// The reducer reference bound to the channel.
    pub reducer: String,
}

/// A compiled node specification with its resolved routing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec {
    /// The node name.
    pub name: String,
    /// The node kind (defaults to `model` when unspecified in source).
    pub kind: String,
    /// The bound model name, if any.
    pub model: Option<String>,
    /// The node prompt, if any.
    pub prompt: Option<String>,
    /// Tool capability names referenced by this node.
    pub tools: Vec<String>,
    /// How control leaves this node.
    pub routing: Routing,
}

/// How control flows out of a [`NodeSpec`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Routing {
    /// A single static successor node.
    Next(String),
    /// Conditional routing: `(label, target)` pairs in declaration order.
    Conditional(Vec<(String, String)>),
    /// The node terminates the run.
    Terminal,
}

/// A compiled static edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeSpec {
    /// The source node name.
    pub from: String,
    /// The target node name, or `END`.
    pub to: String,
}
