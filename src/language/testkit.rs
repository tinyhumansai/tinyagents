//! Deterministic helpers for compiling `.rag` source to a [`Blueprint`] and
//! asserting on the result.
//!
//! These helpers are scripted: they run the same `parse -> compile` pipeline the
//! real compiler uses, with no clocks, randomness, or registry I/O, so a test
//! gets the same [`Blueprint`] every run. They sit one level below
//! [`crate::language::compiler::compile_source`] — they do *not* bind against a
//! registry — so a test can assert on lowered topology without standing up a
//! [`CapabilityRegistry`](crate::registry::CapabilityRegistry).
//!
//! Two flavours are provided:
//!
//! - `try_*` functions return a [`Result`] so a test can assert on parse/compile
//!   errors.
//! - The remaining functions panic with a descriptive message on failure, for
//!   the common "this source must compile" case.

use crate::error::Result;
use crate::language::compiler::{compile, compile_with_provenance};
use crate::language::parser::parse_str;
use crate::language::types::{Blueprint, NodeSpec, Origin, Routing};

/// Parses and compiles `source`, returning every [`Blueprint`] it declares.
///
/// # Errors
///
/// Propagates [`crate::error::TinyAgentsError::Parse`] from the parser and
/// [`crate::error::TinyAgentsError::Compile`] from the compiler.
pub fn try_compile(source: &str) -> Result<Vec<Blueprint>> {
    let program = parse_str(source)?;
    compile(&program)
}

/// Parses and compiles `source`, returning every [`Blueprint`].
///
/// # Panics
///
/// Panics if the source fails to parse or compile.
pub fn compile_all(source: &str) -> Vec<Blueprint> {
    try_compile(source).expect("source should parse and compile")
}

/// Parses and compiles `source`, returning its single declared [`Blueprint`].
///
/// # Panics
///
/// Panics if the source fails to compile or does not declare exactly one graph.
pub fn blueprint(source: &str) -> Blueprint {
    let mut blueprints = compile_all(source);
    assert_eq!(
        blueprints.len(),
        1,
        "expected exactly one graph, found {}",
        blueprints.len()
    );
    blueprints.remove(0)
}

/// Parses and compiles `source` with provenance tagged by `origin`, returning
/// its single declared [`Blueprint`].
///
/// The returned blueprint has [`Blueprint::provenance`] populated so a test can
/// assert that each node/channel/edge points back at its source span.
///
/// # Panics
///
/// Panics if the source fails to compile or does not declare exactly one graph.
pub fn blueprint_with_provenance(source: &str, origin: Origin) -> Blueprint {
    let program = parse_str(source).expect("source should parse");
    let mut blueprints = compile_with_provenance(&program, origin).expect("source should compile");
    assert_eq!(
        blueprints.len(),
        1,
        "expected exactly one graph, found {}",
        blueprints.len()
    );
    blueprints.remove(0)
}

/// Returns the node named `name`, panicking with the available names if it is
/// absent.
pub fn node<'a>(blueprint: &'a Blueprint, name: &str) -> &'a NodeSpec {
    blueprint
        .nodes
        .iter()
        .find(|n| n.name == name)
        .unwrap_or_else(|| {
            let names: Vec<&str> = blueprint.nodes.iter().map(|n| n.name.as_str()).collect();
            panic!("no node `{name}`; nodes are {names:?}")
        })
}

/// Asserts that the node named `name` has the given `kind`.
pub fn assert_kind(blueprint: &Blueprint, name: &str, kind: &str) {
    let actual = &node(blueprint, name).kind;
    assert_eq!(actual, kind, "node `{name}` kind");
}

/// Asserts that the node named `name` continues to `target` over a static edge.
pub fn assert_next(blueprint: &Blueprint, name: &str, target: &str) {
    match &node(blueprint, name).routing {
        Routing::Next(actual) => assert_eq!(actual, target, "node `{name}` next"),
        other => panic!("node `{name}` routing is {other:?}, expected next -> {target}"),
    }
}

/// Asserts that the node named `name` is terminal (routes to `END`).
pub fn assert_terminal(blueprint: &Blueprint, name: &str) {
    match &node(blueprint, name).routing {
        Routing::Terminal => {}
        other => panic!("node `{name}` routing is {other:?}, expected terminal"),
    }
}

/// Asserts that the node named `name` has a conditional route `label -> target`.
pub fn assert_route(blueprint: &Blueprint, name: &str, label: &str, target: &str) {
    match &node(blueprint, name).routing {
        Routing::Conditional(routes) => {
            let found = routes.iter().any(|(l, t)| l == label && t == target);
            assert!(
                found,
                "node `{name}` has no route `{label} -> {target}`; routes are {routes:?}"
            );
        }
        other => panic!("node `{name}` routing is {other:?}, expected conditional routes"),
    }
}
