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

`graph::testkit` should include:

- no-op node
- scripted update node
- scripted route node
- send/fanout node
- failing node
- retry-counting node
- interrupting node
- subgraph test node
- sub-agent fake node
- in-memory checkpointer
- event recorder
- stream collector
- graph snapshot assertion
- checkpoint assertion
- route assertion
- state history assertion

Example:

```rust
assert_graph(run)
    .visited(["agent", "tools", "agent"])
    .routed("agent", "tool")
    .checkpoint_count(3)
    .completed();
```
