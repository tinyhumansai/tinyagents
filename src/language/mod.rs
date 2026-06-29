//! Expressive language (`.rag`) for declarative agent-graph blueprints.
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
//! - [`types`] — token and AST type definitions plus the compiled [`types::Blueprint`].
//! - [`lexer`] — source text into tokens with source spans.
//! - [`parser`] — tokens into a validated AST.
//! - [`compiler`] — AST lowering into a [`types::Blueprint`].

pub mod types;

pub mod compiler;
pub mod lexer;
pub mod parser;

pub use types::*;

#[cfg(test)]
mod test;
