# graph::testkit

Graph-test building blocks — deterministic node doubles, an event recorder, a
stream projector, a fluent run-assertion builder, and storage conformance
suites for the durable graph runtime.

This is the *graph-level* counterpart to `harness::testkit` (model/tool
doubles + trajectories): here the units under test are graph nodes and
supersteps, so the doubles are node handlers and the assertions read a run's
export/event/checkpoint truth. Because a node can recurse into a
`graph::subgraph` or a `graph::subagent_node`, the same recorder that observes
a top-level run also captures the events and child-run rollups of the nested
runs it spawns, so recursion stays observable in tests.

## Node doubles

Each returns a closure ready for `GraphBuilder::add_node`:

| Helper | Behavior |
|--------|----------|
| `noop_node` | Routes onward with no state update |
| `scripted_update_node` | Emits queued updates (saturating the last) |
| `scripted_route_node` | Emits queued `goto` route-sets |
| `fanout_node` | Emits one `Send` per arg (fanout) |
| `failing_node` | Always returns an error |
| `RetryCountingNode` | Counts activations, fails the first N |
| `interrupting_node` | Interrupts until resumed, then updates |
| `subgraph_test_node` | Embeds a child graph (shared state) |
| `subagent_fake_node` | Records a child run + updates (fake sub-agent) |

## Observation & assertions

- `GraphEventRecorder` — captures the `GraphEvent` stream from a run (`.sink()`
  to wire it, `.events()` / `.kinds()` to read back, `.collector()` to
  project).
- `StreamCollector` — projects a recorded event list into test-friendly views:
  `node_order()`, `updates()`, `routes()`, `interrupts()`,
  `checkpoint_count()`, `custom()`.
- `GraphRun<State>` — bundles a `GraphExecution<State>` with its recorded
  events and checkpoint history; built with `::new(execution)` +
  `.with_events(..)` / `.with_history(..)`. The single `GraphRun` is the test
  truth — execution, events, and checkpoints all read from it.
- `run_recorded(..)` — runs a graph with the recorder wired and returns the
  bundled `GraphRun`.
- `assert_graph(&run) -> GraphAssertions<'_, State>` — opens a fluent
  assertion builder: `.visited(..)`, `.routed(from, to)`,
  `.checkpoint_count(n)`, `.state_history(f)`, `.checkpoint(f)`,
  `.completed()`, `.interrupted()` — each panics with a descriptive message on
  failure and returns `&Self` for chaining.

## Storage conformance suites (`conformance.rs`)

Durable graph stores are hard to migrate safely without a shared contract:
two backends that both implement a trait should behave identically. These
functions encode that contract once so any backend — built-in or a
caller-supplied adapter — can be certified by running the same assertions:

- `taskstore_contract(store)` — basic CRUD/lifecycle contract for a
  `graph::orchestration::TaskStore` implementation.
- `taskstore_concurrent_contract(store: Arc<S>)` — concurrent-access contract
  (no lost updates, no corrupted records under concurrent writers).
- `taskstore_replay_contract(reopen)` — durability contract: a store reopened
  via `reopen` must replay to the same state.

Each function **panics** with a descriptive message on the first violation —
call them from a `#[tokio::test]` / `#[test]` in the crate implementing a new
backend.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `GraphEventRecorder`, `StreamCollector`, `GraphRun`, `GraphAssertions`, `RetryCountingNode`. |
| `mod.rs` | Node doubles, `run_recorded`, `assert_graph`. |
| `conformance.rs` | `taskstore_contract`, `taskstore_concurrent_contract`, `taskstore_replay_contract`. |
| `test.rs` | Unit tests for the testkit itself (each double, recorder, assertion). |

## Operational constraints

- These helpers are intended for tests; none of them are optimized for
  production workloads (e.g. `GraphEventRecorder` buffers every event
  unbounded in memory).
- `RetryCountingNode` and `interrupting_node` hold interior mutable counters —
  construct a fresh instance per test rather than sharing one across
  independent test cases, or activation counts will leak between them.
- Conformance suites assume the store under test starts empty (or, for
  `taskstore_replay_contract`, that `reopen` yields an equivalent fresh
  handle to the same underlying storage) — seed/clean state accordingly in
  the calling test.
