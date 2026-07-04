# graph::subgraph

Subgraph node adapters — the graph-level recursion surface where a graph runs
another graph.

This is the structural counterpart to harness sub-agents (a model calling a
model): here an entire `CompiledGraph` is embedded *as a node* inside a parent
graph, so "graphs that run graphs" is just an ordinary node handler. Each
embedding extends the child's checkpoint namespace with the embedding node id,
which keeps every level of a recursively nested run durable and
collision-free, and the executor's recursion limit bounds how deep that
nesting can go.

## Embedding modes

- **`shared_subgraph_node(child)`** — parent and child share the same
  `State`/`Update` channel (`Update == State`). The child runs over the
  parent's state as passed to the node, and its final state becomes the
  parent update.
- **`adapter_subgraph_node(child, to_child, from_child)`** — parent and child
  use different state shapes. `to_child: Fn(&P) -> C` projects the parent
  state into the child's input; `from_child: Fn(&P, C) -> PU` folds the
  child's final state back into a parent update.

Both wrap a `CompiledGraph` into a node handler usable with
`GraphBuilder::add_node`.

## Interrupt propagation

A child that pauses on an interrupt must surface that interrupt to the
parent rather than have its partial state treated as a completed output —
both adapters check `execution.is_interrupted()` and return
`NodeResult::Interrupt(..)` instead of folding the paused child's state
through `from_child` (or returning it directly, for the shared-state case).

## Namespace and recursion bookkeeping

Internal helpers (not part of the public surface, but load-bearing for
correctness):

- `namespaced(child, ctx)` — clones `child` and extends its checkpoint
  namespace with the embedding node id, preventing parent/child checkpoint
  collisions when both share a `thread_id`.
- `child_for(child, ctx)` — prepares an embedded child for a run: applies
  `namespaced`, seeds it with the enclosing run's live recursion frames (so
  the child run extends the parent's recursion tree rather than starting a
  fresh one), and records the embedding node so the child's root frame names
  it.

## Public surface

- `shared_subgraph_node<State>(child: CompiledGraph<State, State>) -> Handler<State, State>`
- `adapter_subgraph_node<P, PU, C, CU, ToChild, FromChild>(child, to_child, from_child) -> Handler<P, PU>`

`types.rs` is documentation-only — the conceptual overview above — because the
adapter constructors return closures rather than named types; there is no
separate public type to document.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Documentation-only: the two embedding modes, conceptually. |
| `mod.rs` | `shared_subgraph_node`, `adapter_subgraph_node`, and the namespace/recursion-frame plumbing that makes nested checkpoints and recursion trees work. |
| `test.rs` | Unit tests (shared vs. adapter state mapping, checkpoint namespace isolation, interrupt propagation, nested recursion limits). |

## Operational constraints

- A parent and its embedded subgraph may share a `thread_id`, but they never
  share a checkpoint namespace — the embedding always appends the node id.
  A caller inspecting checkpoints directly (bypassing the node handler) must
  account for this namespace suffix or it will not find the child's records
  under the plain parent namespace.
- Deep subgraph nesting is bounded by the executor's recursion limit, which is
  seeded from the parent's live recursion frames on each embedding — a graph
  that embeds itself (directly or through a cycle of subgraphs) will hit
  `TinyAgentsError::RecursionLimit` rather than recurse unbounded.
