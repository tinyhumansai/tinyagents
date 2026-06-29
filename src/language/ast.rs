//! Source AST for the expressive language (`.rag`).
//!
//! These are the tree types the [`crate::language::parser`] produces from a
//! token stream and the [`crate::language::compiler`] lowers into a
//! [`crate::language::types::Blueprint`]. They describe a *declared* workflow —
//! graph defaults, input/output shape, state channels, nodes, routes, edges,
//! commands, fanout sends, joins, subgraphs, sub-agents, and REPL-backed nodes —
//! without carrying any executable behaviour. Every node retains a [`Span`] so a
//! diagnostic can point back at the offending source.
//!
//! The AST is re-exported from [`crate::language::types`] for back-compatibility,
//! so existing `crate::language::types::{Program, NodeDecl, …}` paths keep
//! resolving.

use serde::{Deserialize, Serialize};

// The source [`Span`] type lives in [`crate::language::span`].
pub use crate::language::span::Span;

/// A literal value used in `defaults` entries, channel arguments, and similar
/// key/value positions.
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

impl Literal {
    /// Renders the literal as a plain string (used when lowering single-value
    /// policy fields such as `timeout` into the blueprint).
    pub fn as_display(&self) -> String {
        match self {
            Literal::Str(s) | Literal::Ident(s) => s.clone(),
            Literal::Num(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
        }
    }
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
    /// The graph input shape (`input { field type … }`).
    pub input: Vec<IoFieldDecl>,
    /// The graph output shape (`output { field type … }`).
    pub output: Vec<IoFieldDecl>,
    /// Graph-level checkpoint policy (`checkpoint <ident>`).
    pub checkpoint: Option<String>,
    /// Graph-level interrupt policy (`interrupt <ident>`).
    pub interrupt: Option<String>,
    /// `channel <name> <reducer> <arg>*` declarations.
    pub channels: Vec<ChannelDecl>,
    /// `node <name> { … }` declarations.
    pub nodes: Vec<NodeDecl>,
    /// Top-level `from -> to` edge declarations.
    pub edges: Vec<EdgeDecl>,
    /// Top-level `join [a, b] -> c` declarations.
    pub joins: Vec<JoinDecl>,
}

/// A single `name type` entry inside a graph `input`/`output` shape block.
#[derive(Clone, Debug, PartialEq)]
pub struct IoFieldDecl {
    /// The field name.
    pub name: String,
    /// The declared field type (a bare identifier such as `string` or `messages`).
    pub ty: String,
    /// Source position of the field name.
    pub span: Span,
}

/// A `channel <name> <reducer> <arg>*` declaration binding a state channel to a
/// named reducer policy, with optional reducer arguments (e.g. a named
/// aggregate reducer or a barrier arrival count).
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDecl {
    /// The channel name (e.g. `messages`).
    pub name: String,
    /// The reducer reference (e.g. `append`, `overwrite`, `aggregate`).
    pub reducer: String,
    /// Reducer policy arguments (string/number literals), e.g. the name of a
    /// registered aggregate reducer or a barrier count.
    pub args: Vec<Literal>,
    /// Source position of the `channel` keyword.
    pub span: Span,
}

/// A `node <name> { … }` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDecl {
    /// The node name.
    pub name: String,
    /// The declared `kind`, if any (e.g. `agent`, `tool_executor`, `subgraph`).
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
    /// A registered agent name (`agent "researcher"`) for a `subagent` node.
    pub agent: Option<String>,
    /// A registered subgraph name (`graph "flow"`) for a `subgraph` node.
    pub graph: Option<String>,
    /// A registered REPL script name (`script "triage"`) for a `repl_agent`
    /// node. Names a script capability; it never inlines executable code.
    pub script: Option<String>,
    /// An input-mapping name (`input "split_a"`) for sub-agent / subgraph nodes.
    pub input: Option<String>,
    /// A `command { goto … update { … } }` declaration.
    pub command: Option<CommandDecl>,
    /// `sends [ send <node> "input" … ]` fanout declarations.
    pub sends: Vec<SendDecl>,
    /// `sources [a, b]` upstream node names for a `join` node.
    pub sources: Vec<String>,
    /// `options ["approve", "reject"]` choices for an `interrupt` node.
    pub options: Vec<String>,
    /// Node-level checkpoint policy (`checkpoint <ident>`).
    pub checkpoint: Option<String>,
    /// Node-level timeout policy (`timeout <literal>`).
    pub timeout: Option<Literal>,
    /// Node-level `retry { key value … }` policy.
    pub retry: Vec<(String, Literal)>,
    /// Node-level `metadata { key value … }` entries.
    pub metadata: Vec<(String, Literal)>,
    /// Source position of the `node` keyword.
    pub span: Span,
}

impl NodeDecl {
    /// Creates an empty node declaration with the given name and span and all
    /// optional fields unset. Keeps the parser's construction site concise as
    /// the grammar grows.
    pub fn empty(name: String, span: Span) -> Self {
        Self {
            name,
            kind: None,
            model: None,
            prompt: None,
            tools: Vec::new(),
            next: None,
            routes: Vec::new(),
            agent: None,
            graph: None,
            script: None,
            input: None,
            command: None,
            sends: Vec::new(),
            sources: Vec::new(),
            options: Vec::new(),
            checkpoint: None,
            timeout: None,
            retry: Vec::new(),
            metadata: Vec::new(),
            span,
        }
    }
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

/// A `command { goto <target> update { … } }` declaration: a typed graph
/// command that hands control to `goto` while applying channel updates.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandDecl {
    /// The declared `goto` target node name, or `END`, if any.
    pub goto: Option<String>,
    /// Channel/state updates the command applies (`update { key value … }`).
    pub update: Vec<(String, Literal)>,
    /// Source position of the `command` keyword.
    pub span: Span,
}

/// A single `send <node> ["input"]` fanout entry inside a `sends [ … ]` block.
#[derive(Clone, Debug, PartialEq)]
pub struct SendDecl {
    /// The fanned-out target node name.
    pub target: String,
    /// An optional input-mapping name delivered to the target branch.
    pub input: Option<String>,
    /// Source position of the `send` keyword.
    pub span: Span,
}

/// A top-level `join [a, b] -> c` declaration: a barrier that resumes `target`
/// once every named source has arrived.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinDecl {
    /// The upstream node names that must all arrive.
    pub sources: Vec<String>,
    /// The node to continue to once the barrier is satisfied.
    pub target: String,
    /// Source position of the `join` keyword.
    pub span: Span,
}
