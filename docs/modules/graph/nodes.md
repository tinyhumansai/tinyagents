# Graph Node Model

Nodes are async units of work. They receive a state view plus runtime context
and return an update, command, interrupt, or no-op.

```rust
#[async_trait]
pub trait GraphNode<State, Ctx = ()>: Send + Sync {
    async fn run(
        &self,
        state: StateView<'_, State>,
        ctx: &mut GraphContext<Ctx>,
    ) -> Result<NodeResult>;
}

pub enum NodeResult {
    Update(StateUpdate),
    Command(Command),
    Interrupt(Interrupt),
    None,
}
```

Closure-backed nodes stay as an ergonomic adapter:

```rust
Node::new("agent", |state| async move {
    Ok(NodeOutput::continue_with(state))
})
```

Target node spec:

```rust
pub struct NodeSpec<State, Ctx = ()> {
    pub id: NodeId,
    pub node: Arc<dyn GraphNode<State, Ctx>>,
    pub input: ChannelSelection,
    pub destinations: Option<DestinationHints>,
    pub metadata: serde_json::Value,
    pub defer: bool,
    pub retry: Option<RetryPolicy>,
    pub cache: Option<CachePolicy>,
    pub timeout: Option<TimeoutPolicy>,
    pub error_handler: Option<NodeId>,
    pub is_error_handler: bool,
}
```

Important node kinds:

- closure node
- trait-backed node
- harness model/tool/agent-loop node
- sub-agent node
- subgraph node
- router node
- error-handler node
- test node

Deferred nodes run near graph termination, after normal active work is drained.
Use them for cleanup, final scoring, final summarization, or output shaping.
