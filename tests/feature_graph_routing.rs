//! Feature tests for the graph runtime's routing surface.
//!
//! These exercise the *executed* routing paths — conditional edges evaluated
//! against committed state at the superstep boundary and dynamic `Command`
//! `goto` routing — rather than the topology export. Complements
//! `tests/graph_durable.rs` (which only checks that an invalid command target
//! fails) and `tests/e2e_graph_export.rs` (which renders conditional edges
//! without running them).

use tinyagents::{Command, END, GraphBuilder, NodeContext, NodeResult, assert_graph, run_recorded};

/// A whole-state document that records the path of nodes it flowed through.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Doc {
    n: i64,
    path: Vec<String>,
}

impl Doc {
    fn seed(n: i64) -> Self {
        Doc {
            n,
            path: Vec::new(),
        }
    }

    fn stamped(mut self, label: &str) -> Self {
        self.path.push(label.to_string());
        self
    }
}

/// A classify node whose conditional router picks a terminal branch by parity,
/// routing the `zero` case straight to `END`.
fn parity_graph() -> tinyagents::CompiledGraph<Doc, Doc> {
    GraphBuilder::<Doc, Doc>::overwrite()
        .add_node("classify", |s: Doc, _c: NodeContext| async move {
            Ok(NodeResult::Update(s.stamped("classify")))
        })
        .add_node("even", |s: Doc, _c: NodeContext| async move {
            Ok(NodeResult::Update(s.stamped("even")))
        })
        .add_node("odd", |s: Doc, _c: NodeContext| async move {
            Ok(NodeResult::Update(s.stamped("odd")))
        })
        .set_entry("classify")
        .add_conditional_edges(
            "classify",
            |d: &Doc| {
                if d.n == 0 {
                    "zero"
                } else if d.n % 2 == 0 {
                    "even"
                } else {
                    "odd"
                }
            },
            [("even", "even"), ("odd", "odd"), ("zero", END)],
        )
        .set_finish("even")
        .set_finish("odd")
        .compile()
        .expect("parity graph compiles")
}

#[tokio::test]
async fn conditional_router_selects_even_branch_by_committed_state() {
    let run = run_recorded(&parity_graph(), None, Doc::seed(4))
        .await
        .expect("run succeeds");

    assert_graph(&run)
        .visited(["classify", "even"])
        .routed("classify", "even")
        .completed();
    assert_eq!(run.execution.state.path, vec!["classify", "even"]);
}

#[tokio::test]
async fn conditional_router_selects_odd_branch_by_committed_state() {
    let run = run_recorded(&parity_graph(), None, Doc::seed(7))
        .await
        .expect("run succeeds");

    assert_graph(&run)
        .visited(["classify", "odd"])
        .routed("classify", "odd")
        .completed();
    assert_eq!(run.execution.state.path, vec!["classify", "odd"]);
}

#[tokio::test]
async fn conditional_router_can_route_directly_to_end() {
    let run = run_recorded(&parity_graph(), None, Doc::seed(0))
        .await
        .expect("run succeeds");

    // The `zero` label resolves to END, so the run terminates at `classify`
    // without visiting a branch node.
    assert_graph(&run).visited(["classify"]).completed();
    assert_eq!(run.execution.state.path, vec!["classify"]);
}

#[tokio::test]
async fn conditional_router_is_re_evaluated_each_loop_iteration() {
    // A self-loop whose router terminates only once the counter crosses a
    // threshold — the router runs at every superstep boundary.
    let graph = GraphBuilder::<i64, i64>::overwrite()
        .with_recursion_limit(16)
        .add_node("inc", |s: i64, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("inc")
        .add_conditional_edges(
            "inc",
            |n: &i64| if *n >= 3 { "done" } else { "again" },
            [("again", "inc"), ("done", END)],
        )
        .compile()
        .expect("loop graph compiles");

    let run = run_recorded(&graph, None, 0).await.expect("run succeeds");

    assert_graph(&run)
        .visited(["inc", "inc", "inc"])
        .completed();
    assert_eq!(run.execution.state, 3);
    assert_eq!(run.execution.steps, 3);
}

#[tokio::test]
async fn command_goto_routes_dynamically_to_the_chosen_target() {
    // A command-routing node that resolves its successor at runtime from the
    // committed state, choosing between two declared destinations.
    fn threshold_graph() -> tinyagents::CompiledGraph<Doc, Doc> {
        GraphBuilder::<Doc, Doc>::overwrite()
            .add_node("route", |s: Doc, _c: NodeContext| async move {
                let target = if s.n >= 10 { "high" } else { "low" };
                Ok(NodeResult::Command(
                    Command::goto([target]).with_update(s.stamped("route")),
                ))
            })
            .add_node("low", |s: Doc, _c: NodeContext| async move {
                Ok(NodeResult::Update(s.stamped("low")))
            })
            .add_node("high", |s: Doc, _c: NodeContext| async move {
                Ok(NodeResult::Update(s.stamped("high")))
            })
            .set_entry("route")
            .with_command_destinations("route", ["low", "high"])
            .set_finish("low")
            .set_finish("high")
            .compile()
            .expect("threshold graph compiles")
    }

    let graph = threshold_graph();

    let low_run = run_recorded(&graph, None, Doc::seed(3))
        .await
        .expect("low run succeeds");
    assert_graph(&low_run)
        .visited(["route", "low"])
        .routed("route", "low")
        .completed();

    let high_run = run_recorded(&graph, None, Doc::seed(42))
        .await
        .expect("high run succeeds");
    assert_graph(&high_run)
        .visited(["route", "high"])
        .routed("route", "high")
        .completed();
    assert_eq!(high_run.execution.state.path, vec!["route", "high"]);
}
