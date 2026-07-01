# Graph Visualization, Introspection, And Testkit

The graph exports (implemented in `src/graph/export`):

- graph id and name — `GraphTopology::graph_id` / `GraphTopology::name`
  (set via `GraphBuilder::with_name`)
- node ids — `NodeInfo::id`
- node metadata — `NodeInfo::kind`, `NodeInfo::metadata`
  (`GraphBuilder::with_node_kind` / `with_node_metadata`)
- node policy summaries — `NodeInfo::policy` (`NodePolicySummary`: derived
  routing discipline plus barrier/interrupt/deferred/subgraph roles) and the
  graph-level `GraphTopology::policy` (`GraphPolicySummary`: recursion limit,
  parallel, max concurrency, node timeout)
- direct edges — `GraphTopology::edges`
- waiting/barrier edges — `GraphTopology::waiting_edges` (`WaitingEdgeInfo`,
  lifted out of the direct-edge set)
- conditional route labels — `GraphTopology::conditional_edges`
- command destination hints — `NodeInfo::command_destinations`
  (`GraphBuilder::with_command_destinations`)
- start and end paths — `GraphTopology::entry` / `GraphTopology::finish_nodes`
- interrupt markers — `NodeInfo::interrupt` (`GraphBuilder::mark_interrupt`)
- deferred-node markers — `NodeInfo::deferred` (`GraphBuilder::mark_deferred`)
- subgraph nodes — `NodeInfo::subgraph` (`GraphBuilder::mark_subgraph`)
- validation report — `GraphTopology::validation` (`ValidationReport`:
  structural errors for dangling references, warnings for unreachable/dead-end
  nodes)

Export formats:

- structured JSON — `to_json` / `from_json` (the structured `GraphTopology` is
  the single source of truth; tests snapshot it)
- Mermaid — `to_mermaid` (subgraph nodes use the subroutine shape; interrupt,
  deferred, and subgraph nodes get `classDef` markers; barrier and command
  `goto` edges render dotted)
- DOT — later

All three extraction sources — `CompiledGraph::topology`,
`GraphBuilder::topology`, and `blueprint_to_topology` — produce the same
`GraphTopology`, so visualization and test snapshots share one truth.

## Testkit

`graph::testkit` (implemented in `src/graph/testkit`) provides graph-test
building blocks distinct from the harness testkit. Each node helper returns a
closure ready for `GraphBuilder::add_node`:

- no-op node — `noop_node`
- scripted update node — `scripted_update_node`
- scripted route node — `scripted_route_node`
- send/fanout node — `fanout_node`
- failing node — `failing_node`
- retry-counting node — `RetryCountingNode` (`.handler(..)` / `.attempts()`)
- interrupting node — `interrupting_node`
- subgraph test node — `subgraph_test_node` (wraps `shared_subgraph_node`)
- sub-agent fake node — `subagent_fake_node` (records a `ChildRun`)
- in-memory checkpointer — reuse `graph::InMemoryCheckpointer`

Observation and assertion surfaces:

- event recorder — `GraphEventRecorder` (`.sink()`, `.events()`, `.kinds()`)
- stream collector — `StreamCollector` (`node_order`, `updates`, `routes`,
  `interrupts`, `checkpoint_count`, `custom`)
- run bundling — `run_recorded` produces a `GraphRun` (execution + recorded
  events + checkpoint history), the single test truth
- fluent assertions — `assert_graph(run)` → `GraphAssertions`: `.visited`,
  `.routed` (route assertion), `.checkpoint_count`, `.state_history` (state
  history assertion), `.checkpoint` (checkpoint assertion), `.completed`,
  `.interrupted`

`run_recorded` wires the recorder, runs the (cloned) graph, and — when a thread
id is supplied — collects the durable checkpoint history, so the export/event/
checkpoint truth all read from one `GraphRun`.

## Storage conformance {#conformance}

`graph::testkit::conformance` encodes durable-store **contracts** once so any
backend — the built-in ones or a caller-supplied adapter — behaves
interchangeably. Each function panics with a descriptive message on the first
violation, so call them from a `#[test]` / `#[tokio::test]`. Five contracts
ship:

| Contract | Asserts |
| --- | --- |
| `checkpointer_contract(cp)` | Single-threaded put/get (latest + specific), insertion-order listing, `list_threads`, `delete_thread`, `prune`. |
| `taskstore_contract(store)` | Full task lifecycle state machine, cooperative cancel, kill, deadline updates, status filtering, terminal-transition rejection. |
| `checkpointer_concurrent_contract(Arc<C>)` | Many concurrent tasks put distinct checkpoints on one shared instance; every write lands and is retrievable (no lost writes). |
| `taskstore_concurrent_contract(Arc<S>)` | Many threads insert and advance distinct tasks against one shared store; every write lands exactly once. |
| `taskstore_replay_contract(reopen)` | Durable state written through one handle survives re-opening the backing store (terminal status + transition history replay). |

The concurrent contracts take an `Arc<_>` (the store is shared across
threads/tasks); the replay contract takes a `Fn() -> S` that re-opens the same
durable backing so the same store can be closed and reconstructed. Wire them
across every backend in one test file:

```rust
use tinyagents::graph::checkpoint::{FileCheckpointer, InMemoryCheckpointer};
use tinyagents::graph::orchestration::{InMemoryTaskStore, JsonlTaskStore};
use tinyagents::graph::testkit::conformance::{
    checkpointer_concurrent_contract, taskstore_concurrent_contract, taskstore_replay_contract,
};

#[tokio::test]
async fn checkpointer_handles_concurrent_writes() {
    checkpointer_concurrent_contract(std::sync::Arc::new(InMemoryCheckpointer::<i32>::new())).await;
}

#[test]
fn task_store_handles_concurrent_writes() {
    taskstore_concurrent_contract(std::sync::Arc::new(InMemoryTaskStore::new()));
}

#[test]
fn jsonl_task_store_replays_after_restart() {
    let path = std::env::temp_dir().join("conformance-replay.jsonl");
    let _ = std::fs::remove_file(&path);
    taskstore_replay_contract(|| JsonlTaskStore::open(&path).unwrap());
    let _ = std::fs::remove_file(&path);
}
```

Only durable backends (`FileCheckpointer`, `JsonlTaskStore`, SQLite) can pass
the replay contract — an in-memory store drops state on reopen, so run it only
against the durable ones.

Example:

```rust
use tinyagents::graph::testkit::{assert_graph, run_recorded};

let run = run_recorded(&graph, Some("t1"), 0).await?;
assert_graph(&run)
    .visited(["agent", "tools", "agent"])
    .routed("agent", "tools")
    .checkpoint_count(3)
    .completed();
```
