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

## Run hierarchy

When a subgraph node runs its embedded `CompiledGraph`, the child run is wired
into the parent's recursion tree:

- the child gets its own `run_id`, preserves the enclosing run's `root_run_id`,
  and sets `parent_run_id` to the parent run;
- the child run extends the parent's recursion frames (seeded from
  `NodeContext::recursion_frames`) and its root frame names the embedding node,
  so depth tracking is correct without mutating the parent's live stack;
- the spawned child is reported back through `NodeContext::child_runs` and
  surfaces on `GraphExecution::child_runs` (a `ChildRun` list keyed by node) and
  in every boundary checkpoint's metadata under a `child_runs` array;
- callers read the parent/child lineage after a run via
  `GraphExecution::run_tree()` (a `RunTree`: this run's id, the shared root, the
  parent run, and the spawned children).
