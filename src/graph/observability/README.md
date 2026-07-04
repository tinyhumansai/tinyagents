# graph::observability

Durable observability for the graph runtime — journals, status stores, and
the journaling event sink.

The live `graph::stream` layer emits transient `GraphEvent`s into an
in-process `GraphEventSink`. This module makes that history **durable and
correlatable** so a UI, supervisor, or test can reconstruct a recursive graph
run tree after the fact.

`CompiledGraph` wires into this module through builder-style
`with_status_store` and `with_event_journal`; both are opt-in and default off
so existing runs are unchanged.

## Public surface

- `GraphObservation` — a durable envelope pairing a `GraphEvent` with its run
  lineage (`run_id` / `parent_run_id` / `root_run_id`), `graph_id`,
  `checkpoint_id`, subgraph `namespace`, `step`, `offset`, and timestamp. This
  is the unit everything else in the module is built from.
- `GraphEventJournal` (trait) — an append-only, offset-addressable journal of
  observations.
  - `InMemoryGraphEventJournal` — in-process implementation for tests.
  - `StoreGraphEventJournal<A>` — store-backed implementation; stream key is
    the run id, so a `harness::store::AppendStore` durably persists a run's
    observations.
- `GraphStatusStore` (trait) — a compact "what is running now?" surface over
  `graph::GraphRunStatus`.
  - `InMemoryGraphStatusStore` — in-process implementation.
- `JournalGraphSink` — a `GraphEventSink` that wraps each emitted event into a
  `GraphObservation` and appends it to a journal, optionally also forwarding
  to a live `inner` sink (`with_lineage`, `with_thread`, `with_namespace`,
  `with_inner` builders).
- `GraphLatencyMetrics` — per-step/per-node timing rollups derived from a
  run's observations (`from_observations`, `average_step_ms`,
  `average_node_ms`).
- `GraphHealthSummary` — per-node success/failure counts derived from
  observations — node-level **tool health** telemetry (`from_observations`,
  `from_status`).
- `GraphLangfuseExporter` (`langfuse/`) — exports a run's observations to
  Langfuse, turning supersteps and nodes into timed spans (failures promoted
  to `ERROR`) and attaching the health summary to the trace. It shares the
  harness `LangfuseClient` transport and defaults its `traceId` to the run's
  `root_run_id`, so a graph run and the agent/tool runs its nodes spawn land
  under one trace.

## Persistence bridge

`JournalGraphSink` bridges the synchronous `GraphEventSink::emit` hook to the
async journal API with `futures::executor::block_on`, and treats persistence
as **best-effort**: a backend error never aborts the run. Do not rely on the
sink for delivery guarantees stronger than "usually persisted, never
run-blocking."

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: `GraphObservation`, journal/status-store traits and in-memory impls, `JournalGraphSink`, latency/health rollups. |
| `mod.rs` | Behavioral code: rollup computation, journal/store/sink impls. |
| `langfuse/` | `GraphLangfuseExporter` and its span-construction logic. |
| `test.rs` | Unit tests (journal round-trips, sink lineage, latency/health rollups). |

## Operational constraints

- `StoreGraphEventJournal` keys the stream by run id; reusing a run id across
  logically distinct runs interleaves their observations.
- The health summary's failure counts are node-scoped, not step-scoped — a
  node retried and eventually succeeded still contributes its earlier
  failures to `GraphHealthSummary`, by design (it is a *tool health* signal,
  not a final-status signal).
- The Langfuse exporter shares transport code with
  `harness::observability::LangfuseClient` (via `harness::observability`'s
  crate-visible `clean_nulls`/`iso_ms` helpers); keep timestamp/null-pruning
  behavior in sync between the two if either changes.
