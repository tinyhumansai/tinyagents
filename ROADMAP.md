# Roadmap

TinyAgents is at v1.5.0. The roadmap favors small, well-tested modules that
build toward a production-grade Rust agent runtime.

## Shipped Foundation

- typed harness model calls, tools, middleware, structured output, streaming,
  usage/cost tracking, retry/limits, cache, memory/embeddings, sub-agents,
  and steering (`harness/`)
- durable typed state graph runtime: `START`/`END`, nodes, conditional
  routing, `Command`s, fan-out, reducers/channels, checkpoints, interrupts,
  subgraphs, streaming, and topology export (`graph/`)
- per-thread `ThreadGoal` and `TaskBoard` productivity primitives, exposed as
  harness tools
- named capability registry (models, tools, agents, graphs, stores,
  middleware, policy) bound by name (`registry/`)
- the declarative `.rag` blueprint language: lexer, parser, compiler, and
  registry-backed binding (`language/`)
- the imperative `.ragsh` REPL language for capability-bound interactive
  orchestration (`repl/`)
- an optional SQLite-backed checkpointer (`sqlite` feature) and an optional
  Rhai-backed `.ragsh` session runtime (`repl` feature)
- an embedded Langfuse client and graph exporter for observability

## Near-Term Work

- broaden `.rag`/`.ragsh` example coverage for less-common routing and
  parallel-fanout shapes
- continue splitting any module or doc that grows past the 500-line limit
  into focused files
- expand live (network-gated) provider contract tests as new
  OpenAI-compatible endpoints are added
- track and close the internal SDK feature-parity backlog in
  [`docs/sdk-gaps.md`](docs/sdk-gaps.md)

## Parallel Agents And Sub-Agents

Shipped: forked child contexts, shared caches with explicit isolation policy,
child event namespaces, parent/child run ids, deterministic reducer-based
merges, optional/blocking/race/quorum/fallback/compare policies, and
resumable checkpoints across parallel branches. Ongoing work focuses on
hardening edge cases surfaced by the `e2e_parallel_*` and `live_subagent_*`
test suites.

## Stability

The public API is versioned via semver starting at 1.0. Breaking changes are
documented in release notes, tested, and shaped by real examples rather than
speculative abstraction.
