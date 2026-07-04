//! Graph-materialisation tests: lowering a `Blueprint` into a runnable
//! `CompiledGraph` via `build_graph` and executing it.
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;

// ---------------------------------------------------------------------------
// Graph materialisation + execution
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TestState {
    trail: Vec<String>,
    agent_visits: u32,
}

/// Resolves a conditional route `label` to a *durable* target node id from the
/// blueprint's `(label, target)` table, translating the language `END` sentinel
/// (`"END"`) to the durable graph terminal ([`crate::graph::END`]). Unknown
/// labels fall back to the durable `END`.
fn resolve_durable_target(routes: &[(String, String)], label: &str) -> String {
    let target = routes
        .iter()
        .find(|(l, _)| l == label)
        .map(|(_, t)| t.as_str());
    match target {
        Some(t) if t != crate::language::types::END => t.to_string(),
        _ => crate::graph::END.to_string(),
    }
}

/// A trivial factory that materialises echo/route/end nodes purely from the
/// declarative [`NodeSpec`]. It demonstrates that runnable behaviour comes from
/// Rust, not the source: each node records its name; terminal/`next` nodes
/// commit a whole-state update (static edges route them), and conditional nodes
/// loop once before terminating by emitting an explicit `goto` command.
struct TestFactory;

impl NodeFactory<TestState> for TestFactory {
    fn make(&self, spec: &NodeSpec) -> crate::error::Result<BoxedNode<TestState>> {
        let name = spec.name.clone();
        let routing = spec.routing.clone();
        Ok(Arc::new(
            move |mut state: TestState, _ctx: NodeContext| -> NodeFuture<TestState> {
                let name = name.clone();
                let routing = routing.clone();
                Box::pin(async move {
                    state.trail.push(name.clone());
                    let result = match &routing {
                        // Static edges (Next/Terminal) handle routing; just
                        // commit the whole-state update.
                        Routing::Terminal | Routing::Next(_) => NodeResult::Update(state),
                        Routing::Conditional(routes) => {
                            state.agent_visits += 1;
                            // Take the `tool_call -> tools` route until the
                            // second visit, then take `final -> END`.
                            let label = if state.agent_visits >= 2 {
                                "final"
                            } else {
                                "tool_call"
                            };
                            let target = resolve_durable_target(routes, label);
                            NodeResult::Command(Command::goto([target]).with_update(state))
                        }
                    };
                    Ok(result)
                })
            },
        ))
    }
}

/// Collects the visited node ids into owned strings for comparison.
fn visited_names(run: &crate::graph::GraphExecution<TestState>) -> Vec<String> {
    run.visited.iter().map(ToString::to_string).collect()
}

#[tokio::test]
async fn build_graph_runs_to_end() {
    let bp = compile(&parse_str(SUPPORT_AGENT).unwrap())
        .unwrap()
        .remove(0);
    let graph = build_graph(&bp, &TestFactory).unwrap();

    let run = graph
        .run(TestState {
            trail: Vec::new(),
            agent_visits: 0,
        })
        .await
        .unwrap();

    // agent -> tools -> agent (ends on second visit).
    assert_eq!(visited_names(&run), vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.trail, vec!["agent", "tools", "agent"]);
    assert_eq!(run.state.agent_visits, 2);
}

#[tokio::test]
async fn build_graph_handles_linear_terminal() {
    let src = "graph g { start a node a { kind model next b } node b { kind model next END } }";
    let bp = compile(&parse_str(src).unwrap()).unwrap().remove(0);
    let graph = build_graph(&bp, &TestFactory).unwrap();
    let run = graph
        .run(TestState {
            trail: Vec::new(),
            agent_visits: 0,
        })
        .await
        .unwrap();
    assert_eq!(visited_names(&run), vec!["a", "b"]);
}
