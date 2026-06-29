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
