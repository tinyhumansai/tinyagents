# Graph Runtime Context, Node Defaults, And Policies

Graph nodes receive a `GraphContext` that wraps run identity, graph policy,
runtime context, stores, streams, and child-call helpers.

```rust
pub struct GraphContext<Ctx = ()> {
    pub run: GraphRunContext,
    pub data: Ctx,
    pub stores: StoreRegistry,
    pub events: GraphEventSink,
    pub stream: StreamWriter,
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    pub cache: Option<Arc<dyn GraphCache>>,
}
```

The context must expose:

- `thread_id`
- `run_id`
- `root_run_id`
- `parent_run_id`
- `checkpoint_namespace`
- `step`
- `task_id`
- `node_id`
- `recursion_depth`
- immutable user context
- deadline and cancellation state
- custom stream writer
- local/fresh state read helper for branch logic

Graph context is not a global singleton. It is created per run and scoped per
task.

## Node Defaults And Policies

Graph-level defaults are applied at compile time. Per-node values always win.

```rust
pub struct GraphDefaults {
    pub retry: Option<RetryPolicy>,
    pub cache: Option<CachePolicy>,
    pub timeout: Option<TimeoutPolicy>,
    pub error_handler: Option<NodeId>,
    pub max_concurrency: Option<usize>,
}

pub struct TimeoutPolicy {
    pub run_timeout: Option<Duration>,
    pub idle_timeout: Option<Duration>,
    pub refresh_on: TimeoutRefresh,
}
```

Policy rules:

- retry and timeout defaults apply to normal nodes and error-handler nodes
- cache defaults apply only to normal nodes
- error-handler defaults apply only to normal nodes
- an error-handler node must not catch itself
- timeout cancellation is cooperative
- idle timeout can refresh on graph progress or explicit heartbeat
- cache keys must include node input, relevant context, code/config version, and
  namespace
- cached task writes are replayed as writes; they are not treated as opaque final
  state
