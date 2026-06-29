# Graph Memory And Stores Boundary

Graph checkpointing is short-term execution persistence. Harness memory and
stores are separate concerns.

LangGraph makes this distinction directly:

- checkpointers store graph state by thread and checkpoint id
- stores provide long-term memory across threads and conversations, with
  hierarchical namespaces, metadata, and optional vector search

Graph owns:

- checkpoints
- channel values
- channel versions
- pending writes
- active tasks
- interrupts
- state history and forks

Harness/store owns:

- chat history storage
- semantic memory
- user/application records
- model/tool artifacts
- usage and cost records
- provider traces

The graph can pass a `StoreRegistry` through `GraphContext`, but graph execution
must not assume any specific store backend.
