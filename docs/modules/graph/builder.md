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
- `compile`

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
