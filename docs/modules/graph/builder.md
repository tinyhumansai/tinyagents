# Graph Builder And Compile Contract

The builder supports:

- `add_node`
- `add_sequence`
- `add_edge`
- `add_waiting_edge`
- `add_conditional_edges`
- `set_entry_point`
- `set_conditional_entry_point`
- `set_finish_point`
- `set_node_defaults`
- `set_defaults`
- `with_max_concurrency`
- `with_node_timeout`
- `compile`

`add_conditional_edges` accepts route labels and a router return value that are
any `impl ToString`, so a user-defined route enum that implements `Display` (or
the `Route` newtype) can label edges directly; plain `&str`/`String` labels keep
working unchanged.

`add_sequence([a, b, c])` is convenience sugar for a chain of direct edges
(`add_edge(a, b).add_edge(b, c)`). `add_waiting_edge(from, to)` is a barrier edge:
`to` activates only once *all* of its registered predecessors have completed,
even across supersteps.

Graph defaults are settable in one call:

```rust
pub struct GraphDefaults {
    pub recursion_limit: Option<usize>,
    pub parallel: Option<bool>,
    pub max_concurrency: Option<usize>,
    pub node_timeout: Option<Duration>,
}
```

`set_defaults(GraphDefaults { .. })` applies only the `Some` fields.
`with_max_concurrency(n)` bounds the number of node handlers in flight per
parallel superstep (the active set runs in chunks of at most `n`).
`with_node_timeout(d)` fails the run with `TinyAgentsError::Timeout` if any node
handler does not resolve within `d`.

`CompiledGraph::with_run_deadline(d)` bounds the *whole run* by a wall-clock
`d`, checked at every super-step boundary: when the elapsed run time first
reaches `d` the run stops *between* super-steps with `TinyAgentsError::Timeout`,
leaving the last committed boundary checkpoint intact and resumable. Prefer this
over wrapping `run` in an external `tokio::time::timeout`, which aborts
mid-super-step and cannot leave a clean checkpoint. It bounds scheduling, not a
single in-flight node — pair it with `with_node_timeout` to also bound
individual handlers.

Compile-time options:

```rust
pub struct CompileOptions {
    pub name: Option<String>,
    pub checkpointer: CheckpointerChoice,
    pub cache: Option<Arc<dyn GraphCache>>,
    pub store: Option<StoreRegistry>,
    pub interrupt_before: InterruptSelector,
    pub interrupt_after: InterruptSelector,
    pub debug: bool,
    pub stream_transformers: Vec<StreamTransformerFactory>,
}
```

`CheckpointerChoice` mirrors the LangGraph subgraph model:

- `Inherit`: inherit the parent graph checkpointer when used as a subgraph.
- `Enabled(Arc<dyn Checkpointer>)`: use this saver.
- `Disabled`: do not checkpoint even if the parent has a saver.

Validation rules:

- graph must have at least one `START` path
- `START` cannot be an edge target
- `END` cannot be an edge source
- every edge source exists, except `START`
- every edge target exists, except `END`
- duplicate node ids are rejected
- duplicate branch names from a source are rejected
- conditional route targets are validated at compile time when known
- interrupt targets must exist
- waiting-edge sources and targets must exist
- command destinations used only for rendering are marked as such
- node additions after compile do not mutate an existing compiled graph
