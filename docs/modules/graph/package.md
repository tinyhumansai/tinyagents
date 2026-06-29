# Graph Package And Core Types

## Package Shape

Target layout:

```text
src/graph/
  mod.rs
  builder.rs
  cache.rs
  channel.rs
  checkpoint.rs
  command.rs
  compile.rs
  edge.rs
  error.rs
  event.rs
  executor.rs
  interrupt.rs
  node.rs
  parallel.rs
  policy.rs
  reducer.rs
  recursion.rs
  state.rs
  stream.rs
  subagent.rs
  subgraph.rs
  testkit.rs
  visualize.rs
```

The current single-file `src/graph.rs` can remain through milestone 1. Split the
package when the API adds compile-time graph freezing, checkpointing, or typed
streaming; those features are large enough to deserve module boundaries.

## Core Types

```rust
pub struct GraphBuilder<State, Ctx = (), Input = State, Output = State> {
    graph_id: GraphId,
    nodes: IndexMap<NodeId, NodeSpec<State, Ctx>>,
    edges: EdgeSet,
    branches: BranchSet<State, Ctx>,
    channels: ChannelSet<State>,
    input_schema: SchemaRef<Input>,
    output_schema: SchemaRef<Output>,
    defaults: GraphDefaults,
}

pub struct CompiledGraph<State, Ctx = (), Input = State, Output = State> {
    graph_id: GraphId,
    nodes: Arc<IndexMap<NodeId, CompiledNode<State, Ctx>>>,
    edges: Arc<EdgeSet>,
    branches: Arc<BranchSet<State, Ctx>>,
    channels: Arc<ChannelSet<State>>,
    input_channels: ChannelSelection,
    output_channels: ChannelSelection,
    defaults: GraphDefaults,
}

pub struct GraphRun<Output> {
    pub run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub checkpoint_id: Option<CheckpointId>,
    pub output: Output,
    pub interrupts: Vec<Interrupt>,
    pub visited: Vec<NodeId>,
    pub steps: usize,
    pub max_depth: usize,
}
```

The builder is mutable and ergonomic. The compiled graph is immutable,
validated, cheap to clone, and safe to run concurrently.

`State`, `Input`, and `Output` should be separate generic concepts. Many real
graphs accept a narrow input shape, maintain richer internal state, and expose a
filtered output shape.
