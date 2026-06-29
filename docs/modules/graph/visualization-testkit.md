# Graph Visualization, Introspection, And Testkit

The graph should export:

- graph id and name
- node ids
- node metadata
- node policy summaries
- direct edges
- waiting/barrier edges
- conditional route labels
- command destination hints
- start and end paths
- interrupt markers
- deferred-node markers
- subgraph nodes
- validation report

Export formats:

- structured JSON
- Mermaid
- DOT later

Test snapshots can use the same export so visualization and tests share one
truth.

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
