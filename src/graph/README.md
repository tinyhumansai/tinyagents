# graph

TinyAgents' durable workflow runtime (LangGraph-style), and one of the
load-bearing surfaces of the crate's recursive language-model (RLM)
architecture.

Because a node can embed another compiled graph (`subgraph`) or invoke a
sub-agent, **graphs run graphs** and orchestration recurses while every step
stays typed, checkpointed, and observable. A workflow authored from a `.rag`
blueprint or driven from the `.ragsh` REPL lowers into exactly these same
types, so a model can describe, compile, and re-enter the very runtime it is
executing inside.

Each submodule keeps type definitions in `types.rs`, behavior in `mod.rs`, and
unit tests in `test.rs` (per repo convention); complex submodules additionally
carry their own `README.md` — see the module map below.

## Module map

| Module | Concern |
| --- | --- |
| `builder` | Authoring/compile contract: `GraphBuilder` accumulates nodes, edges, conditional routing, and a reducer; `.compile()` validates topology and freezes it into an immutable `CompiledGraph`. |
| `channel` | Channel-per-field state model (additive) — per-field merge rules and concurrent-write conflict detection, running on the unmodified executor. See [`channel/README.md`](channel/README.md). |
| `checkpoint` | The `Checkpointer` trait and backends (file, sqlite, in-memory) — durability that makes runs resumable and time-travelable. See [`checkpoint/README.md`](checkpoint/README.md). |
| `command` | `Command`, `Interrupt`, `NodeResult`, `RouteTarget`, `Send` — the vocabulary a node handler returns to update state, route, interrupt, or fan out. |
| `compiled` | The superstep executor, `CompiledGraph` — sequential/parallel steps, node retry, resumable failure, run/resume/state APIs. See [`compiled/README.md`](compiled/README.md). |
| `export` | Graph introspection/visualization: topology extraction, Mermaid/JSON export, validation reports. |
| `goals` | A durable per-thread goal (single "completion contract"), continuation loop, and harness tools. See [`goals/README.md`](goals/README.md). |
| `observability` | Durable graph observability: journals, status stores, the journaling sink, latency/health rollups, Langfuse export. See [`observability/README.md`](observability/README.md). |
| `orchestration` | Managed child-work controls (`spawn`/`await`/`cancel`/... ) exposed as harness tools, backed by a `TaskStore`. See [`orchestration/README.md`](orchestration/README.md). |
| `parallel` | `map_reduce` — ordered, bounded-concurrency parallel map/reduce with a configurable failure policy, independent of the graph executor. |
| `recursion` | Recursion policy and depth tracking: `RecursionFrame`/`RecursionPolicy`/`RecursionStack`/`RunTree` bound and observe nested graph/subgraph/sub-agent recursion. |
| `reducer` | `StateReducer`/`Reducer` implementations (overwrite, append, min/max, set-union, closures) that fold branch updates into committed state at a superstep boundary. |
| `status` | `GraphRunStatus` — a compact run-status snapshot. |
| `stream` | `GraphEvent`, `GraphEventSink`, and streaming modes — the live, in-process event surface `observability` makes durable. |
| `subagent_node` | Embeds a harness agent as a graph node (the graph-level analogue of `subgraph`, but for agents instead of graphs). |
| `subgraph` | Embeds a `CompiledGraph` as a node (shared-state or adapter mode) — graph-level recursion. See [`subgraph/README.md`](subgraph/README.md). |
| `testkit` | Deterministic node doubles, event recorder, fluent run assertions, storage conformance suites. See [`testkit/README.md`](testkit/README.md). |
| `todos` | A per-thread kanban task board and harness tools. See [`todos/README.md`](todos/README.md). |

## How the pieces fit together

1. **Author**: `GraphBuilder` (in `builder`) accumulates nodes (`NodeHandler`),
   edges, conditional `Route`s, and a `StateReducer`, then `.compile()`
   validates the topology and produces an immutable `CompiledGraph` (in
   `compiled`).
2. **Run**: `CompiledGraph::run`/`run_with_thread` drives supersteps. Each
   step's node handlers return `NodeResult`s built from `command` vocabulary
   (`Command::Update`, `Command::Goto`, `Interrupt`, `Send` for fanout); the
   step boundary folds results through the configured reducer (`reducer`, or
   `channel` for per-field semantics).
3. **Persist**: when a `Checkpointer` (`checkpoint`) is configured, each
   boundary is checkpointed, making the run resumable, forkable, and
   time-travelable; `status::GraphRunStatus` gives a compact live snapshot.
4. **Observe**: `stream::GraphEvent`s fan out live through a `GraphEventSink`;
   `observability` makes that history durable (journals, status stores,
   Langfuse export) and derives latency/health rollups.
5. **Recurse**: a node can embed another compiled graph (`subgraph`) or a
   harness agent (`subagent_node`), or hand off named child work through
   `orchestration`'s tool surface; `recursion` tracks depth and the run tree
   across all three so nesting stays bounded and observable.
6. **Test**: `testkit` provides node doubles, a recorder, fluent assertions,
   and storage conformance suites so both hand-written graphs and new storage
   backends can be verified against the same contracts.

## Errors

Graph-specific failures surface through the crate's shared `Result`/
`TinyAgentsError` (see `crate::error`); the most graph-specific variants are
`TinyAgentsError::RecursionLimit` (nesting/loop depth exceeded) and
`TinyAgentsError::InvalidConcurrentUpdate` (a same-step concurrent write to a
channel that does not allow it — see `channel`).

## Operational constraints

- Checkpoints are written **only at superstep boundaries**, never mid-node —
  a node handler must be safe to re-run from scratch after a crash.
- Subgraph and sub-agent embedding always extends the child's checkpoint
  namespace with the embedding node id; inspecting a child's checkpoints
  directly requires accounting for that namespace suffix.
- Parallel-step determinism depends on branches never mutating shared state
  outside their own snapshot clone; the reducer, not lock ordering, is what
  resolves conflicting concurrent writes.
