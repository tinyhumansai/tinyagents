//! REPL language (`.ragsh`) for capability-bound interactive orchestration.
//!
//! The REPL language is the interactive, capability-bound counterpart to the
//! declarative `.rag` expressive language. It lets an operator (human or a
//! parent orchestrator) drive a harness/graph session through typed, policy-
//! checked commands rather than free-form code.
//!
//! This module currently provides the skeleton: command grammar types, a
//! command parser, and a capability/policy boundary. (filled in by implementation pass)

pub mod types;

pub use types::*;

#[cfg(test)]
mod test;
