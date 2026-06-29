//! Token and AST types for the expressive language, plus the compiled
//! [`Blueprint`] artifact.
//!
//! These types are the intermediate representations along the `.rag` pipeline
//! that turns a declarative (possibly self-authored) plan into runnable
//! topology: [`Token`]s from the lexer, the [`Program`] AST from the parser, and
//! the fully serializable [`Blueprint`] from the compiler. The `Blueprint` is the
//! inspectable, diffable, checkpointable artifact at the heart of the recursive
//! architecture — the same shape whether a human or a model wrote the source.
//!
//! This module holds *only* type definitions. The processing logic lives in the
//! sibling modules:
//!
//! - [`crate::language::lexer`] produces [`SpannedToken`]s from source text.
//! - [`crate::language::parser`] turns tokens into a [`Program`] AST.
//! - [`crate::language::compiler`] lowers a [`Program`] into one [`Blueprint`]
//!   per graph and wires blueprints into the runtime graph.
//!
//! The source AST node types (`Program`, `GraphDecl`, `NodeDecl`, …) live in
//! [`crate::language::ast`] and are re-exported here for back-compatibility.

use serde::{Deserialize, Serialize};

// Re-export the source AST so existing `crate::language::types::{Program, …}`
// paths keep resolving after the AST moved to its own module.
pub use crate::language::ast::*;

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

// The source [`Span`] type lives in [`crate::language::span`] and is re-exported
// here so existing `crate::language::types::Span` paths keep resolving.
pub use crate::language::span::Span;

/// A [`Token`] paired with the [`Span`] where it begins.
#[derive(Clone, Debug, PartialEq)]
pub struct SpannedToken {
    /// The token value.
    pub token: Token,
    /// The source position of the token's first character.
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
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// The declared graph input shape (empty when unspecified).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<IoFieldSpec>,
    /// The declared graph output shape (empty when unspecified).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<IoFieldSpec>,
    /// Graph-level checkpoint policy (`inherit`, `always`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
    /// Graph-level interrupt policy (`manual`, `auto`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt: Option<String>,
    /// Compiled join/barrier declarations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joins: Vec<JoinSpec>,
    /// Source provenance: where each piece of this blueprint came from.
    ///
    /// Populated by [`crate::language::compiler::compile_with_provenance`] (and
    /// the [`crate::language::testkit`] helpers). The plain [`compile`] path
    /// leaves this `None` so its output is unchanged. Surface it through
    /// [`Blueprint::provenance`].
    ///
    /// [`compile`]: crate::language::compiler::compile
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<BlueprintProvenance>,
}

impl Blueprint {
    /// Returns this blueprint's source provenance, if it was compiled through a
    /// provenance-aware path.
    ///
    /// Provenance lets a UI, test, or review tool trace every node, channel, and
    /// edge back to the exact source span and origin (a file path or a
    /// generated plan) it came from.
    pub fn provenance(&self) -> Option<&BlueprintProvenance> {
        self.provenance.as_ref()
    }
}

// ===========================================================================
// Provenance (source traceability for a compiled Blueprint)
// ===========================================================================

/// Where a [`Blueprint`] (or one of its pieces) originated.
///
/// Origin is the trust-relevant half of provenance: a `File` blueprint was
/// authored by a human at a path, while a `Generated` blueprint was emitted by a
/// model running inside the harness. Review tooling treats the two differently
/// even though they compile through the same gate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Origin {
    /// Authored in a file at the given path.
    File(String),
    /// Emitted by a model/REPL session (self-authored), with an optional label
    /// identifying the producer.
    Generated(Option<String>),
}

impl Origin {
    /// A file origin at `path`.
    pub fn file(path: impl Into<String>) -> Self {
        Origin::File(path.into())
    }

    /// A generated origin with no producer label.
    pub fn generated() -> Self {
        Origin::Generated(None)
    }

    /// A generated origin labelled with the producer (e.g. a REPL session id).
    pub fn generated_by(label: impl Into<String>) -> Self {
        Origin::Generated(Some(label.into()))
    }

    /// A short human-readable description (`"plan.rag"`, `"generated"`,
    /// `"generated by repl-7"`).
    pub fn as_display(&self) -> String {
        match self {
            Origin::File(path) => path.clone(),
            Origin::Generated(None) => "generated".to_string(),
            Origin::Generated(Some(label)) => format!("generated by {label}"),
        }
    }
}

/// The source span of a single named piece (node, channel) of a blueprint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedSpan {
    /// The piece's name.
    pub name: String,
    /// The source span the piece was declared at.
    pub span: Span,
}

/// The source span of a single static edge of a blueprint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeSpan {
    /// The edge source node name.
    pub from: String,
    /// The edge target node name, or `END`.
    pub to: String,
    /// The source span the edge was declared at.
    pub span: Span,
}

/// Source traceability for a compiled [`Blueprint`].
///
/// Every node, channel, and edge is paired with the [`Span`] it was declared at,
/// and the whole blueprint records its [`Origin`]. This is the view returned by
/// [`Blueprint::provenance`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlueprintProvenance {
    /// Where this blueprint came from.
    pub origin: Origin,
    /// The span of the `graph` declaration.
    pub graph: Span,
    /// The span of each node, keyed by node name, in source order.
    pub nodes: Vec<NamedSpan>,
    /// The span of each channel, keyed by channel name, in source order.
    pub channels: Vec<NamedSpan>,
    /// The span of each static edge, in source order.
    pub edges: Vec<EdgeSpan>,
}

impl BlueprintProvenance {
    /// Returns the source span the node named `name` was declared at, if known.
    pub fn node_span(&self, name: &str) -> Option<Span> {
        self.nodes.iter().find(|n| n.name == name).map(|n| n.span)
    }

    /// Returns the source span the channel named `name` was declared at, if
    /// known.
    pub fn channel_span(&self, name: &str) -> Option<Span> {
        self.channels
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.span)
    }
}

/// A compiled state-channel binding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelSpec {
    /// The channel name.
    pub name: String,
    /// The reducer reference bound to the channel.
    pub reducer: String,
    /// Reducer policy arguments (e.g. a named aggregate reducer or barrier
    /// count). Empty for plain reducers like `append`/`overwrite`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<Literal>,
}

/// A compiled `name: type` field in a graph input/output shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IoFieldSpec {
    /// The field name.
    pub name: String,
    /// The declared field type.
    pub ty: String,
}

/// A compiled join/barrier: `target` resumes once every `source` has arrived.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JoinSpec {
    /// The upstream node names that must all arrive.
    pub sources: Vec<String>,
    /// The node to continue to once the barrier is satisfied.
    pub target: String,
}

/// A compiled `command`: a typed graph command that hands control to `goto`
/// while applying `update` writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandSpec {
    /// The `goto` target node name, or `END`, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goto: Option<String>,
    /// Channel/state updates the command applies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub update: Vec<(String, Literal)>,
}

/// A compiled fanout `send`: deliver `input` to a fresh `target` branch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SendSpec {
    /// The fanned-out target node name.
    pub target: String,
    /// An optional input-mapping name delivered to the target branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
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
    /// A registered agent name for a `subagent` node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// A registered subgraph name for a `subgraph` node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgraph: Option<String>,
    /// A registered REPL script name for a `repl_agent` node (declaration only;
    /// never inline code).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// An input-mapping name for sub-agent / subgraph nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// A typed `command` declaration, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSpec>,
    /// Fanout `send` declarations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sends: Vec<SendSpec>,
    /// Upstream node names a `join` node waits on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub join_sources: Vec<String>,
    /// Choices presented by an `interrupt` node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    /// Node-level checkpoint policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
    /// Node-level timeout policy (rendered literal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
    /// Node-level `retry { … }` policy entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retry: Vec<(String, Literal)>,
    /// Node-level `metadata { … }` entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metadata: Vec<(String, Literal)>,
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
