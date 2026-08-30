# Graph Package And Core Types

## Package Shape

Package layout:

```text
crates/tinyagents-graph/src/
  lib.rs
  builder/
  channel/
  checkpoint/
  command/
  compiled/
  parallel/
  recursion/
  reducer/
  stream/
  subagent_node/
  subgraph/
  testkit/
```

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
