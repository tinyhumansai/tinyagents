//! End-to-end coverage for COMPLEX durable-graph workflows.
//!
//! These offline tests exercise the harder compositions the durable graph
//! runtime is meant to support:
//!
//! - **Graph + subgraph nesting** using BOTH embedding modes in one parent:
//!   [`shared_subgraph_node`] (parent and child share the `State` channel) and
//!   [`adapter_subgraph_node`] (parent state is projected into a differently
//!   shaped child and folded back).
//! - **Deep nesting** (a subgraph that itself embeds a subgraph, >= 2 levels)
//!   running to completion with correctly merged state, while child checkpoints
//!   stay isolated and never collide with the parent's checkpoint thread.
//! - **A complex layout**: a `Command::goto` fan-out into a parallel branch set
//!   (`with_parallel`), a join node, a conditional route, and a loop-back edge
//!   bounded deterministically by the recursion limit.
//!
//! All scenarios are deterministic and run without network access.

use std::sync::Arc;

use tinyagents::graph::{ClosureStateReducer, adapter_subgraph_node, shared_subgraph_node};
use tinyagents::{
    Checkpointer, CompiledGraph, GraphBuilder, InMemoryCheckpointer, NodeContext, NodeResult,
    TinyAgentsError,
};

// ---------------------------------------------------------------------------
// Scenario 1: graph + subgraph nesting via BOTH shared and adapter embedding.
// ---------------------------------------------------------------------------

/// A document the parent graph carries end to end.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Doc {
    text: String,
    word_count: usize,
    sentiment: i32,
    log: Vec<String>,
}

/// A shared-state child subgraph that normalizes a [`Doc`] in two steps. It
/// uses the same `State`/`Update` channel as the parent (whole-state overwrite),
/// so it can be embedded with [`shared_subgraph_node`].
fn normalize_child() -> CompiledGraph<Doc, Doc> {
    GraphBuilder::<Doc, Doc>::overwrite()
        .add_node("upper", |mut d: Doc, _c: NodeContext| async move {
            d.text = d.text.to_uppercase();
            d.log.push("upper".to_string());
            Ok(NodeResult::Update(d))
        })
        .add_node("count", |mut d: Doc, _c: NodeContext| async move {
            d.word_count = d.text.split_whitespace().count();
            d.log.push("count".to_string());
            Ok(NodeResult::Update(d))
        })
        .set_entry("upper")
        .add_edge("upper", "count")
        .set_finish("count")
        .compile()
        .expect("normalize child compiles")
}

/// A differently-shaped child subgraph that scores a bare `i32`: doubles it then
/// adds one. It is embedded via [`adapter_subgraph_node`] because its state shape
/// (`i32`) differs from the parent's (`Doc`).
fn score_child() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("double", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s * 2))
        })
        .add_node("plus_one", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("double")
        .add_edge("double", "plus_one")
        .set_finish("plus_one")
        .compile()
        .expect("score child compiles")
}

#[tokio::test]
async fn parent_nests_shared_and_adapter_subgraphs() {
    let parent = GraphBuilder::<Doc, Doc>::overwrite()
        // Shared-state subgraph: runs over the whole parent Doc.
        .add_node("normalize", shared_subgraph_node(normalize_child()))
        // Adapter subgraph: project Doc -> i32, fold the i32 score back into Doc.
        .add_node(
            "score",
            adapter_subgraph_node(
                score_child(),
                |d: &Doc| d.text.chars().count() as i32,
                |d: &Doc, child_score: i32| {
                    let mut d = d.clone();
                    d.sentiment = child_score;
                    d.log.push("scored".to_string());
                    d
                },
            ),
        )
        .set_entry("normalize")
        .add_edge("normalize", "score")
        .set_finish("score")
        .compile()
        .expect("parent compiles");

    let run = parent
        .run(Doc {
            text: "hello world".to_string(),
            ..Doc::default()
        })
        .await
        .expect("run succeeds");

    // The shared child uppercased the text and counted words.
    assert_eq!(run.state.text, "HELLO WORLD");
    assert_eq!(run.state.word_count, 2);
    // The adapter child scored chars("HELLO WORLD") = 11 -> *2 + 1 = 23.
    assert_eq!(run.state.sentiment, 23);
    // Both children's work is reflected in the parent's merged log.
    assert_eq!(run.state.log, vec!["upper", "count", "scored"]);
    // The PARENT only sees its own two nodes; subgraph internals are private.
    let visited: Vec<&str> = run.visited.iter().map(|n| n.as_str()).collect();
    assert_eq!(visited, vec!["normalize", "score"]);
}

// ---------------------------------------------------------------------------
// Scenario 2: deep nesting (subgraph inside a subgraph) + checkpoint isolation.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deep_nested_subgraphs_merge_state_without_checkpoint_collision() {
    // Level 3 (innermost): +100, with its own checkpointer.
    let inner_ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let inner = GraphBuilder::<i32, i32>::overwrite()
        .add_node("add100", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 100))
        })
        .set_entry("add100")
        .set_finish("add100")
        .compile()
        .expect("inner compiles")
        .with_checkpointer(inner_ckpt.clone());

    // Level 2 (middle): +10, then run the inner subgraph.
    let mid_ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let middle = GraphBuilder::<i32, i32>::overwrite()
        .add_node("add10", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 10))
        })
        .add_node("inner", shared_subgraph_node(inner))
        .set_entry("add10")
        .add_edge("add10", "inner")
        .set_finish("inner")
        .compile()
        .expect("middle compiles")
        .with_checkpointer(mid_ckpt.clone());

    // Level 1 (parent/top): +1, then run the middle subgraph.
    let parent_ckpt = Arc::new(InMemoryCheckpointer::<i32>::new());
    let parent = GraphBuilder::<i32, i32>::overwrite()
        .add_node("add1", |s: i32, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("middle", shared_subgraph_node(middle))
        .set_entry("add1")
        .add_edge("add1", "middle")
        .set_finish("middle")
        .compile()
        .expect("parent compiles")
        .with_checkpointer(parent_ckpt.clone());

    let run = parent
        .run_with_thread("deep", 0)
        .await
        .expect("threaded run succeeds");

    // 0 +1 (parent) +10 (middle) +100 (inner) = 111, merged across all levels.
    assert_eq!(run.state, 111);
    assert!(!run.is_interrupted());

    // The top-level run persists one checkpoint per superstep boundary (add1,
    // middle), all under the empty top-level namespace.
    let list = parent_ckpt.list("deep").await.expect("list succeeds");
    assert_eq!(list.len(), 2);
    assert!(list.iter().all(|m| m.namespace.is_empty()));
    // Checkpoint ids are unique within the thread (parent chained correctly).
    assert_eq!(list[0].parent_checkpoint_id, None);
    assert_eq!(
        list[1].parent_checkpoint_id.as_deref(),
        Some(list[0].checkpoint_id.as_str())
    );

    // Embedded subgraphs run on their own (threadless) executions, so their
    // checkpoints NEVER leak into the parent's thread. The mechanism that keeps
    // them isolated — namespace extension — is unit-tested in
    // `src/graph/subgraph/test.rs`.
    assert_eq!(mid_ckpt.count("deep"), 0);
    assert_eq!(inner_ckpt.count("deep"), 0);
}

// ---------------------------------------------------------------------------
// Scenario 3: complex layout — fan-out -> parallel branches -> join, plus a
// conditional loop-back bounded by the recursion limit.
// ---------------------------------------------------------------------------

/// The pipeline state: an append-only audit log, two accumulators written by
/// the parallel branches, and a round counter incremented at each join.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Pipe {
    log: Vec<String>,
    a: i32,
    b: i32,
    rounds: i32,
}

/// A partial update folded by the reducer. Because the parallel branches each
/// write a different field, the reducer is the deterministic fan-in/join: every
/// branch's patch is applied in stable active-set order.
#[derive(Clone, Debug, Default)]
struct Patch {
    log: Vec<String>,
    add_a: i32,
    add_b: i32,
    inc_round: bool,
}

fn pipe_reducer() -> ClosureStateReducer<Pipe, Patch, impl Fn(Pipe, Patch) -> tinyagents::Result<Pipe>>
{
    ClosureStateReducer::new(|mut s: Pipe, p: Patch| {
        s.log.extend(p.log);
        s.a += p.add_a;
        s.b += p.add_b;
        if p.inc_round {
            s.rounds += 1;
        }
        Ok(s)
    })
}

/// Builds the fan-out/parallel/join/loop-back graph. `max_rounds` bounds the
/// loop via the join's conditional router; `recursion_limit` is the hard ceiling.
fn complex_graph(max_rounds: i32, recursion_limit: usize) -> CompiledGraph<Pipe, Patch> {
    GraphBuilder::<Pipe, Patch>::new()
        .set_reducer(pipe_reducer())
        .with_parallel(true)
        .with_recursion_limit(recursion_limit)
        // Fan-out: route to BOTH parallel branches via an explicit goto command.
        .add_node("dispatch", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Command(
                tinyagents::Command::goto(["work_a", "work_b"]).with_update(Patch {
                    log: vec!["dispatch".to_string()],
                    ..Patch::default()
                }),
            ))
        })
        .mark_command_routing("dispatch")
        // Two branches that run concurrently in one superstep.
        .add_node("work_a", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Update(Patch {
                log: vec!["a".to_string()],
                add_a: 1,
                ..Patch::default()
            }))
        })
        .add_node("work_b", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Update(Patch {
                log: vec!["b".to_string()],
                add_b: 10,
                ..Patch::default()
            }))
        })
        // Join: both branches' static edges converge here (deduped to one run).
        .add_node("join", |_s: Pipe, _c: NodeContext| async move {
            Ok(NodeResult::Update(Patch {
                log: vec!["join".to_string()],
                inc_round: true,
                ..Patch::default()
            }))
        })
        .set_entry("dispatch")
        .add_edge("work_a", "join")
        .add_edge("work_b", "join")
        // Conditional loop-back: keep looping until we hit `max_rounds`.
        .add_conditional_edges(
            "join",
            move |s: &Pipe| {
                if s.rounds < max_rounds {
                    "again".to_string()
                } else {
                    "done".to_string()
                }
            },
            [("again", "dispatch"), ("done", tinyagents::END)],
        )
        .compile()
        .expect("complex graph compiles")
}

#[tokio::test]
async fn fan_out_parallel_join_with_bounded_loop_back() {
    // Loop runs exactly two rounds, then routes to END.
    let graph = complex_graph(2, 50);
    let run = graph.run(Pipe::default()).await.expect("run succeeds");

    // Two rounds of (a += 1, b += 10); two joins increment rounds to 2.
    assert_eq!(run.state.a, 2);
    assert_eq!(run.state.b, 20);
    assert_eq!(run.state.rounds, 2);
    assert_eq!(
        run.state.log,
        vec!["dispatch", "a", "b", "join", "dispatch", "a", "b", "join"]
    );

    // Visited order is deterministic: parallel branches fold in active-set index
    // order (work_a before work_b) regardless of completion order.
    let visited: Vec<&str> = run.visited.iter().map(|n| n.as_str()).collect();
    assert_eq!(
        visited,
        vec![
            "dispatch", "work_a", "work_b", "join", "dispatch", "work_a", "work_b", "join",
        ]
    );

    // dispatch(1) -> {work_a,work_b}(2) -> join(3) -> dispatch(4) -> ... -> join(6).
    assert_eq!(run.steps, 6);
}

#[tokio::test]
async fn unbounded_loop_back_hits_recursion_limit_deterministically() {
    // `max_rounds` larger than the recursion limit allows means the router never
    // routes to END; the run must terminate on the deterministic ceiling.
    let graph = complex_graph(1_000, 4);
    let err = graph
        .run(Pipe::default())
        .await
        .expect_err("the loop never reaches `done`");
    assert!(
        matches!(err, TinyAgentsError::RecursionLimit(4)),
        "expected RecursionLimit(4), got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// DEFERRED: graph node that REPLs another graph (graph -> .ragsh REPL -> graph).
// ---------------------------------------------------------------------------

/// Placeholder tracking the graph->REPL->graph composition. The `.ragsh` live
/// execution engine that would let a node drive another graph through a REPL
/// session is not built yet (it lands in a later cluster). This test is
/// intentionally ignored — not faked — so the intent is tracked and this file
/// fails loudly to be wired up once the REPL engine exists.
#[tokio::test]
#[ignore = "pending the .ragsh REPL execution engine (later cluster); lands with that work"]
async fn graph_repls_another_graph_pending_repl_engine() {
    panic!("not implemented: requires the .ragsh REPL execution engine");
}
