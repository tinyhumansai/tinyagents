# Subconscious Loop Example

This folder contains a runnable, fully offline example of an autonomous
closed-loop agent harness with a dedicated subconscious layer. It translates the
LangGraph-style reference design into TinyAgents' typed graph runtime while
keeping every step deterministic enough for normal `cargo test` coverage.

Run it with:

```text
cargo run --example subconscious_loop
```

Run the integration coverage with:

```text
cargo test --test e2e_subconscious_loop_example
```

## Files

- `main.rs`: small executable entrypoint that seeds an initial channel packet,
  runs the graph, and prints the final state.
- `autonomous_loop.rs`: the reusable graph, state types, node implementations,
  routing functions, reducer, mock retrieval, diff summarization, and
  subconscious steering logic.
- `../../tests/e2e_subconscious_loop_example.rs`: integration tests that import
  the same implementation file used by the example binary, so documented and
  tested behavior stay aligned.

## Architecture

The example models three cognitive tiers:

- Quick layer: `frontend_agent` turns raw channel payloads into macro
  instructions, then later compiles the final channel response.
- Reasoning layer: `agent_execution` retrieves from long-term memory, simulates
  sub-agent execution, performs semantic extraction, emits a sequential
  world-state diff, and decides whether to force a subconscious evaluation.
- Subconscious layer: `subconscious_eval` consumes gated world summaries and
  returns a short steering directive, then resets the trigger state to avoid an
  infinite evaluation loop.

The graph nodes are:

```text
channel_ingestion
  -> frontend_agent
  -> agent_execution
  -> summarization_gate
  -> frontend_agent
  -> context_manager_hook
  -> subconscious_eval | END
```

The second `frontend_agent` pass is intentional. The first pass defers work to
the Reasoning layer by writing `agent_instructions`; the second pass sees
`agent_reply` and writes `channel_response`.

## State Model

`SystemState` is the committed graph state. It contains the surface channel
fields, operational memory fields, routing flags, and an `event_log` used by the
example output and tests.

Important fields:

- `messages`: channel-visible messages.
- `agent_instructions`: macro instruction produced by the Quick layer.
- `agent_reply`: Reasoning-layer result consumed by the Quick layer.
- `semantic_history`: entity-centric extracted traces from execution.
- `sequential_diffs`: small world-state mutations accumulated across cycles.
- `gated_world_summary`: consolidated package forwarded to the subconscious
  layer.
- `context_utilization`: simulated context pressure.
- `trigger_subconscious`: event escalation flag set by the Reasoning layer.
- `cron_due`: scheduled consolidation flag.
- `long_term_memory`: mock vector database records.
- `retrieved_context`: memory traces retrieved for the current run.
- `subconscious_steering`: final steering directive.

Nodes do not overwrite `SystemState` directly. They return a `StatePatch`, and
`apply_patch` merges partial updates through `ClosureStateReducer`. This mirrors
the durable graph contract: nodes emit small updates, and the reducer is the
single deterministic fan-in point.

## Memory Lifecycle

The Reasoning layer performs a mock retrieval step against `long_term_memory`.
The implementation uses string matching rather than embeddings so the example
has no network calls, API keys, or nondeterministic model output.

When `context_utilization` reaches the eviction threshold, the
`context_manager_hook` moves semantic traces into `long_term_memory`, replaces
the active `semantic_history` with an eviction marker, and lowers utilization.
This demonstrates the bi-directional memory loop:

1. retrieve historical traces before execution;
2. execute and extract semantic history;
3. evict dense traces into long-term memory under pressure;
4. retrieve them again on future runs.

## Summarization Gate

The graph does not forward every diff to the subconscious layer. The
`summarization_gate` forwards only when either condition is true:

- at least three sequential diffs are queued;
- `trigger_subconscious` is set by the Reasoning layer.

Forwarding converts the queued `WorldDiff` values into one `GatedWorldSummary`
containing a macro trend, critical event count, and total magnitude. After
forwarding, the diff queue is cleared. If the threshold is not met, the gate
holds the diffs and records that decision in `event_log`.

## Hybrid Triggers

The example supports both event and cron-style triggers:

- Event trigger: `agent_execution` sets `trigger_subconscious` when the
  instruction contains critical anomaly terms or when context pressure is very
  high.
- Cron trigger: callers can seed `SystemState::cron_due = true` before running
  the graph to force a scheduled subconscious review.

`subconscious_eval` always clears both `trigger_subconscious` and `cron_due`.
That reset is part of the circuit breaker that prevents repeated asynchronous
evaluation loops.

## Offline Simulation Boundaries

This example intentionally avoids real LLM and vector database calls. It is a
reference architecture sample, not a hosted provider integration. The simulated
parts are isolated behind small functions:

- `retrieve_context`: stand-in for vector DB retrieval.
- `agent_execution_node`: stand-in for Reasoning LLM orchestration and
  sub-agent spawning.
- `summarize_diffs`: stand-in for the summarization gate model.
- `subconscious_eval_node`: stand-in for deep reflection and steering.

A production version would replace those functions with provider-backed model
calls and a real `Retriever`, while keeping the same graph topology, state
fields, routing flags, and reducer shape.

## Test Coverage

The integration test covers the main architectural guardrails:

- normal execution retrieves memory and holds a single diff below the gate
  threshold;
- critical Reasoning escalation forces the summarization gate and routes through
  `subconscious_eval`;
- sequential diff accumulation triggers the subconscious layer without an
  anomaly;
- context pressure evicts semantic history into long-term memory;
- the public `run_subconscious_loop` helper used by `main.rs` stays executable.

These tests are offline and run as part of the default suite.
