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
    Stored entries omit what the stream already says (`run_id`, `root_run_id`
    unless it differs, and the `{run_id}-evt-{offset}` event id); readers
    rebuild the `AgentObservation` wire type unchanged. `seal_run(run_id)`
    closes a finished run's journal (the host calls it when a run ends;
    `reap_orphaned_runs_and_seal` does it for orphans) without changing any
    read; with `DriverAppendStore::journal` the stream is
    `harness.journal/<run id>`, sealed on finish and kept 30 days.
- `HarnessStatusStore` (trait) — a compact "what is running now?" surface
  (phase, counters, last-updated) distinct from the full journal.
  - `InMemoryStatusStore` — in-process implementation.
- **Sinks**, each implementing `harness::events::EventListener`:
  - `FanOutSink` — broadcasts one event to multiple inner listeners.
  - `RedactingSink` — masks secrets in event payloads before forwarding.
  - `JournalSink` — persists observations into a `HarnessEventJournal`.
  - `JsonlSink` — appends records to a JSONL stream via
    a `harness::store::AppendStore` such as `DriverAppendStore`.
  - `SinkHealth` — queue-drop and backend-append-failure totals returned by
    `JournalSink::health` and `JsonlSink::health`.
- `AgentObservation`-derived metrics:
  - `AgentCallLatency` — start/end/elapsed for a single model or tool call.
  - `AgentLatencyMetrics` — latency rollups for one run, built with
    `AgentLatencyMetrics::from_observations(&[AgentObservation])`.
- Process profiling:
  - `ProcessSnapshot` — point-in-time process RSS and user/system CPU counters.
  - `ProcessProfiler` — a scoped RSS sampler around a workload.
  - `ProcessProfile` — the serializable wall-time, CPU utilization, and
    baseline/peak RSS report returned by `ProcessProfiler::finish`.
- `LangfuseAuth`, `LangfuseClient`, `LangfuseTraceConfig` (re-exported from the
  private `langfuse` submodule) — shared Langfuse credentials, HTTP transport,
  trace config, and the exporter for durable harness observations. The agent
  exporter emits a
  per-run span (`{trace}:run:{run_id}`, `agent`/`sub-agent`) and nests every
  generation, tool span, and event under it, so a run — and sub-agent recursion
  across separately-exported batches — surfaces as a tree, matching LangChain's
  callback run hierarchy.
- `spans::Span`, `spans::SpanState`, and `spans::SpanExporter` (behind the
  `langfuse` feature) — a host-neutral model for already-mapped run, step,
  generation, tool, and event observations. It builds the Langfuse ingestion
  shape and delegates v4 OTLP conversion and transport to the shared helpers
  and client. Hosts retain event mapping, identity, and content-capture policy.
- `LangfuseScore` / `LangfuseScoreValue` — an evaluation score (numeric,
  categorical, or boolean; trace- or observation-scoped) attached to an exported
  trace via `LangfuseClient::create_score`, mirroring Langfuse's `createScore`.

## Persistence bridge

Persisting sinks (`JournalSink`, `JsonlSink`) bridge the synchronous
`EventListener::on_event` hook to the async journal/store APIs through a
background `AppendWorker`: `on_event` hands the observation to a **bounded**
queue drained by a task on the one process-wide `tinyagents-drain` thread
(started on first use, shared by every sink), so it never blocks the run on
I/O and a new sink costs a task, not an OS thread and a runtime. Appends of
different sinks interleave at their `.await`s; an append that blocks its
thread without awaiting holds up every sink's drain while it runs.
Persistence is **best-effort**: if the queue is full the observation is dropped
(and counted in `AppendWorker::dropped`), backend errors are counted (in
`AppendWorker::append_failures`) and reported through `tracing` rather than
propagated, and neither ever aborts the run. Call `JournalSink::flush` /
`JsonlSink::flush` to block until the durable log has caught up (for example
before reading it back or shutting down). Do not rely on a sink for delivery
guarantees stronger than "usually persisted, never run-blocking."

A persistent backend failure (read-only volume, full disk) fails every queued
observation identically, so reporting is **rate-limited** rather than
per-observation: the first failure of a run logs at `ERROR` on the
`tinyagents::observability` target, further failures are counted silently with a
`WARN` reminder at most once per `APPEND_REPORT_COOLDOWN` (5 minutes), and the
first success afterwards logs one `WARN` recovery summary carrying how many
observations were lost. The worker keeps attempting every item while degraded —
the attempt is what detects recovery. Because reporting goes through `tracing`,
an embedder with no subscriber installed sees nothing — install a subscriber to
observe durable-log loss. Host applications can inspect both loss paths through
`JournalSink::health` / `JsonlSink::health` without relying on a tracing
subscriber.

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
| `profile.rs` | Scoped process CPU/RSS sampling for stress tests and host-specific workloads. |
| `mod.rs` | Behavioral code: latency rollups, journal/store/sink impls. |
| `langfuse/` | `LangfuseClient` and payload helpers (`clean_nulls`, `iso_ms`), split into `mod.rs` (impl + helpers), `types.rs` (`LangfuseAuth`/`LangfuseClient`/`LangfuseTraceConfig`), and `mod_tests.rs`. |
| `spans.rs`, `spans/` | Host-neutral span records and Langfuse batch/OTLP exporter; module README documents lifecycle and shape. |
| `mod_tests.rs` | Unit tests (journal round-trips, redaction, latency rollups, sink fan-out). |

## Operational constraints

- `StoreEventJournal` keys the stream by run id; using the same `AppendStore`
  for unrelated runs is safe (streams are namespaced) but reusing a run id
  across logically distinct runs will interleave their observations.
- `RedactingSink` must be composed *outside* any sink that persists or
  exports off-process (e.g. wrap before `JournalSink`/`JsonlSink`/Langfuse) —
  it only redacts what passes through it, not what already landed elsewhere.
- The Langfuse client performs network I/O; failures there follow the same
  best-effort, non-aborting policy as the other persisting sinks.
