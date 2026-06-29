# Roadmap

TinyAgents is pre-1.0. The roadmap favors small, well-tested modules that build
toward a production-grade Rust agent runtime.

## Current Foundation

- chat message primitives
- model request and response types
- async chat model trait
- async tool trait
- executable state graph
- direct and conditional routing
- graph validation and recursion limits
- basic examples and serialization tests

## Near-Term Work

- split broad modules into focused module directories with `types.rs` and
  `test.rs`
- expand graph tests for routing, recursion, validation, and error behavior
- strengthen harness model, tool, prompt, context, middleware, and usage APIs
- add more examples for model calls, tools, and graph composition
- define reducer and state-channel APIs for parallel writes
- document stable public API boundaries as modules mature

## Declarative Workflow Language

The `.rag` language should let humans and LLMs describe agent workflows without
embedding arbitrary host code.

Planned capabilities:

- graph topology declarations
- allowed models, tools, agents, stores, middleware, and subgraphs
- state channels and reducers
- direct routes, conditional routes, commands, sends, joins, and barriers
- parallel sub-agent fanout
- blocking and optional child-agent policies
- checkpoint, interrupt, timeout, retry, budget, and concurrency policies
- source spans and diagnostics
- blueprint review before execution

## Parallel Agents And Sub-Agents

TinyAgents should support workflow-native parallelism:

- forked child contexts
- shared caches with explicit isolation policy
- child event namespaces
- parent and child run ids
- deterministic reducer-based merges
- optional, blocking, race, quorum, fallback, and compare policies
- resumable checkpoints across parallel branches

## REPL And Agent-Authored Workflows

The `.ragsh` REPL layer should let agents and humans inspect, script, and
control graph runs through capability-bound functions. It should be able to
propose `.rag` workflows, but those workflows must pass through parser,
registry, policy, and compiler checks before execution.

## Pre-1.0 Stability

APIs may change before 1.0. Changes should be documented, tested, and shaped by
real examples rather than speculative abstraction.
