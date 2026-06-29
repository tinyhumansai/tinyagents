use super::*;
use crate::TinyAgentsError;
use crate::graph::command::NodeResult;

type S = i32;

#[test]
fn compile_requires_reducer() {
    let err = GraphBuilder::<S, S>::new()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[test]
fn compile_requires_start() {
    let err = GraphBuilder::<S, S>::overwrite()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .compile()
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::MissingStart));
}

#[test]
fn compile_rejects_missing_edge_target() {
    let err = GraphBuilder::<S, S>::overwrite()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .add_edge("a", "missing")
        .compile()
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::MissingNode(n) if n == "missing"));
}

#[test]
fn compile_rejects_command_routing_with_edges() {
    let err = GraphBuilder::<S, S>::overwrite()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("b", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .mark_command_routing("a")
        .compile()
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[test]
fn compile_rejects_static_and_conditional_on_same_node() {
    let err = GraphBuilder::<S, S>::overwrite()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("b", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .add_conditional_edges("a", |_s: &S| "x".to_string(), [("x", "b")])
        .set_finish("b")
        .compile()
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)));
}

#[test]
fn compile_succeeds_for_valid_graph() {
    let compiled = GraphBuilder::<S, S>::overwrite()
        .add_node("a", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .add_node("b", |s: S, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .set_finish("b")
        .compile();
    assert!(compiled.is_ok());
}
