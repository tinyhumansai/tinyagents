# Graph Subgraphs

Subgraphs are compiled graphs used as nodes.

Two state modes:

```text
shared-state subgraph
parent State == child State

adapter subgraph
parent State -> child Input -> child Output -> parent Update
```

Subgraph requirements:

- namespace checkpoint ids
- preserve `root_run_id`
- set child `parent_run_id`
- propagate thread id by default
- allow isolated child thread ids by explicit configuration
- inherit, override, or disable the parent checkpointer
- emit nested events with parent node id and namespace
- stream child values, updates, messages, tasks, and checkpoints when requested
- allow `Command::Parent` handoff from child graph to parent graph
- expose child state in parent checkpoint task metadata

Subgraph persistence must be explicit. Inherited checkpointing is convenient for
shared-state subgraphs; isolated checkpointing is safer for reusable child
graphs that may also run independently.
