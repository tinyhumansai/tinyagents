//! Expressive language (`.rag`) — the declarative blueprint surface of the
//! recursive runtime.
//!
//! In TinyAgents' recursive (RLM-style) architecture, a model can author the
//! very workflow it is standing inside. `.rag` is the *safe boundary* for that
//! self-authoring: a capability-by-name blueprint format that lowers into the
//! exact same [`crate::graph`] and [`crate::harness`] runtime as hand-written
//! Rust, yet can only *reference* capabilities — never define or execute code —
//! so an agent-emitted plan is parsed, validated, and bound against a registry
//! before it ever runs.
//!
//! The expressive language is a compact, side-effect-free way to describe an
//! agent graph: its state channels, nodes, routes, and capability references.
//! It compiles into the same [`crate::graph`] and [`crate::harness`] structures
//! as hand-written Rust through a fixed pipeline:
//!
//! ```text
//! source -> lexer -> tokens -> parser -> AST -> compiler -> Blueprint
//! ```
//!
//! The language deliberately cannot embed arbitrary code; it only references
//! capabilities (models, tools, routers) by name, which the compiler binds and
//! validates against a registry. This makes it the safe boundary for
//! agent-authored graph plans.
//!
//! Submodules:
//! - [`ast`] — source AST node types produced by the parser.
//! - [`span`] — byte+line/column source spans with merge.
//! - [`source`] — source files and the source map that resolve offsets to
//!   line/column and slice snippets.
//! - [`diagnostic`] — structured [`diagnostic::Diagnostic`]s and the caret
//!   renderer.
//! - [`types`] — token and AST type definitions plus the compiled [`types::Blueprint`].
//! - [`lexer`] — source text into tokens with source spans.
//! - [`parser`] — tokens into a validated AST.
//! - [`resolver`] — registry-backed binding of every reference in a plan,
//!   producing spanned diagnostics for unknown/disallowed capabilities.
//! - [`compiler`] — AST lowering into a [`types::Blueprint`], with an optional
//!   provenance-tagging path ([`compiler::compile_with_provenance`]).
//! - [`diff`] — structured, human-readable diffs of two blueprints
//!   ([`diff::blueprint_diff`]), backing generated-workflow review.
//! - [`testkit`] — deterministic helpers to compile source to a blueprint and
//!   assert on it.

pub mod ast;
pub mod diagnostic;
pub mod source;
pub mod span;
pub mod types;

pub mod compiler;
pub mod diff;
pub mod lexer;
pub mod parser;
pub mod resolver;
pub mod testkit;

pub use diagnostic::{Diagnostic, Label, Severity};
pub use diff::{BlueprintDiff, ChannelDiff, FieldChange, NodeDiff, blueprint_diff};
pub use resolver::{Resolver, resolve_source};
pub use source::{SourceFile, SourceId, SourceMap};
pub use span::Span;
pub use types::*;

#[cfg(test)]
mod test;
