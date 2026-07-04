# harness::observability

Durable observability for the harness — journals, status stores, and sinks.

The live `harness::events` layer fans typed `AgentEvent`s out to in-process
listeners for the duration of a run. This module makes that history
**durable and correlatable** so a UI, supervisor, or test can reconstruct a
recursive run tree after the fact, including across process restarts.

See `docs/modules/harness/observability.md` for the design rationale
(inspiration, responsibilities, cross-cutting requirements shared with
`graph::observability`). This README documents the module's current public
surface and operational constraints.

## Public surface

- `AgentObservation` — a durable envelope pairing an `AgentEvent` with its run
  lineage (`run_id` / `parent_run_id` / `root_run_id`), a stream `offset`, and
  a `ts_ms` timestamp. This is the unit everything else in the module is built
  from.
- `HarnessEventJournal` (trait) — an append-only, offset-addressable journal of
  observations.
  - `InMemoryEventJournal` — in-process implementation for tests and
    short-lived processes.
  - `StoreEventJournal<A: AppendStore>` — store-backed implementation; the
    stream key is the run id, so a `harness::store::AppendStore` (e.g. a
    JSONL file or Sqlite-backed store) durably persists observations per run.
- `HarnessStatusStore` (trait) — a compact "what is running now?" surface
  (phase, counters, last-updated) distinct from the full journal.
  - `InMemoryStatusStore` — in-process implementation.
- **Sinks**, each implementing `harness::events::EventListener`:
  - `FanOutSink` — broadcasts one event to multiple inner listeners.
  - `RedactingSink` — masks secrets in event payloads before forwarding.
  - `JournalSink` — persists observations into a `HarnessEventJournal`.
  - `JsonlSink` — appends records to a JSONL stream via
    `harness::store::JsonlAppendStore`.
- `AgentObservation`-derived metrics:
  - `AgentCallLatency` — start/end/elapsed for a single model or tool call.
  - `AgentLatencyMetrics` — latency rollups for one run, built with
    `AgentLatencyMetrics::from_observations(&[AgentObservation])`.
- `LangfuseAuth`, `LangfuseClient`, `LangfuseTraceConfig` (re-exported from the
  private `langfuse` submodule) — the Langfuse exporter used to ship harness
  (and, via shared helpers, graph) traces to Langfuse.

## Persistence bridge

Persisting sinks (`JournalSink`, `JsonlSink`) bridge the synchronous
`EventListener::on_event` hook to the async journal/store APIs with
`futures::executor::block_on`, and treat persistence as **best-effort**: a
backend error is logged/dropped and never aborts the run. Do not rely on a
sink for delivery guarantees stronger than "usually persisted, never
run-blocking."

## Latency metrics semantics

`AgentLatencyMetrics::from_observations` tolerates redacted payload strings
(structural ids must be preserved by any redaction upstream) but **ignores
incomplete calls** — a `ModelStarted`/`ToolStarted` with no matching
`*Completed`/`*Failed` has no terminal timestamp to measure against and is
silently excluded from the rollup rather than reported with a bogus duration.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: `AgentObservation`, journal/status-store traits and in-memory impls, sink structs, latency types. |
| `mod.rs` | Behavioral code: latency rollups, journal/store/sink impls. |
| `langfuse.rs` | `LangfuseClient` and payload helpers (`clean_nulls`, `iso_ms`) shared with `graph::observability::langfuse`. |
| `test.rs` | Unit tests (journal round-trips, redaction, latency rollups, sink fan-out). |

## Operational constraints

- `StoreEventJournal` keys the stream by run id; using the same `AppendStore`
  for unrelated runs is safe (streams are namespaced) but reusing a run id
  across logically distinct runs will interleave their observations.
- `RedactingSink` must be composed *outside* any sink that persists or
  exports off-process (e.g. wrap before `JournalSink`/`JsonlSink`/Langfuse) —
  it only redacts what passes through it, not what already landed elsewhere.
- The Langfuse client performs network I/O; failures there follow the same
  best-effort, non-aborting policy as the other persisting sinks.
