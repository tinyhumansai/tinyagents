//! A complex durable-graph layout: a nested subgraph embedded inside a
//! parallel fan-out / join.
//!
//! Topology built below:
//!
//! ```text
//!                    ┌───────────────► branch_sub ──┐
//!   START ─► dispatch┤  (fan-out, goto both)        ├─► join ─► END
//!                    └──────────────► branch_plain ─┘
//!
//!   branch_sub embeds a 2-level subgraph:
//!       enrich_child:  +5  ─►  grandchild(+1)        // a subgraph inside a subgraph
//! ```
//!
//! `dispatch` fan-outs with a `Command::goto` into two branches that run
//! concurrently (`with_parallel(true)`). One branch (`branch_sub`) is a nested
//! subgraph embedded with [`adapter_subgraph_node`]; its child itself embeds a
//! grandchild via [`shared_subgraph_node`], so the value `0 -> +5 -> +1 = 6`
//! flows up through two subgraph levels. The branches' partial updates are folded
//! by the reducer (the deterministic join), then `join` finalizes the run.
//!
//! Run with:
//!
//! ```text
//! cargo run --example complex_graph
//! ```

use tinyagents::graph::{ClosureStateReducer, adapter_subgraph_node, shared_subgraph_node};
use tinyagents::{Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult, Result};

/// The pipeline state: an append-only path log and a running total.
#[derive(Clone, Debug, Default)]
struct Pipe {
    log: Vec<String>,
    total: i32,
}

/// A partial update merged by the reducer at each superstep boundary.
#[derive(Clone, Debug, Default)]
struct Patch {
    log: Vec<String>,
    add: i32,
}

/// Grandchild subgraph (level 3): adds 1 to a bare `i32`.
fn grandchild() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("g_add", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("g_add")
        .set_finish("g_add")
        .compile()
        .expect("grandchild compiles")
}

/// Enrich subgraph (level 2): adds 5, then runs the grandchild (a subgraph
/// embedded inside this subgraph).
fn enrich_child() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("c_add", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 5))
        })
        .add_node("grandchild", shared_subgraph_node(grandchild()))
        .set_entry("c_add")
        .add_edge("c_add", "grandchild")
        .set_finish("grandchild")
        .compile()
        .expect("enrich child compiles")
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = GraphBuilder::<Pipe, Patch>::new()
        .set_reducer(ClosureStateReducer::new(|mut s: Pipe, p: Patch| {
            s.log.extend(p.log);
            s.total += p.add;
            Ok(s)
        }))
        .with_parallel(true)
        // Fan-out into both branches in one superstep.
        .add_node("dispatch", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                Command::goto(["branch_sub", "branch_plain"]).with_update(Patch {
                    log: vec!["dispatch".to_string()],
                    add: 0,
                }),
            ))
        })
        .mark_command_routing("dispatch")
        // Branch 1: a nested subgraph (level 2 -> level 3) via an adapter.
        .add_node(
            "branch_sub",
            adapter_subgraph_node(
                enrich_child(),
                |p: &Pipe| p.total,
                |_p: &Pipe, child_value: i32| Patch {
                    log: vec![format!("branch_sub={child_value}")],
                    add: child_value,
                },
            ),
        )
        // Branch 2: a plain concurrent branch.
        .add_node("branch_plain", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Update(Patch {
                log: vec!["branch_plain".to_string()],
                add: 3,
            }))
        })
        // Join: both branches converge here (deduped to a single run).
        .add_node("join", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Update(Patch {
                log: vec!["join".to_string()],
                add: 0,
            }))
        })
        .set_entry("dispatch")
        .add_edge("branch_sub", "join")
        .add_edge("branch_plain", "join")
        .set_finish("join")
        .compile()?;

    let run = graph.run(Pipe::default()).await?;

    let path: Vec<&str> = run.visited.iter().map(|n| n.as_str()).collect();
    println!("path   : {}", path.join(" -> "));
    println!("log    : {}", run.state.log.join(" | "));
    println!("total  : {}", run.state.total); // 6 (nested subgraph) + 3 (plain) = 9
    println!("steps  : {}", run.steps);

    Ok(())
}
