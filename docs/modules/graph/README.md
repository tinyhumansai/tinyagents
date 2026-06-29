# Graph Module Specification

The graph module is RustAgents' workflow runtime. It owns topology, state
transition semantics, routing, execution history, checkpointing, interrupts,
streaming, parallel execution, subgraph invocation, sub-agent nodes, recursive
calls, and graph-level observability.

The graph module must be usable without the expressive language. The expressive
language compiles into graph structures; the graph runtime must not know or care
where a graph came from.

The current implementation in `src/graph.rs` is a milestone-1 scaffold:
closure-backed nodes, whole-state node outputs, direct and conditional edges,
sequential execution, and a recursion limit. The feature specifications below
describe the target contract that the scaffold should grow into.

## Source Inspiration

Primary references:

- LangGraph repository: <https://github.com/langchain-ai/langgraph>
- LangGraph graph API:
  <https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph persistence:
  <https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph durable execution:
  <https://docs.langchain.com/oss/python/langgraph/durable-execution>
- LangGraph checkpointers:
  <https://docs.langchain.com/oss/python/langgraph/checkpointers>
- LangGraph interrupts:
  <https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph streaming:
  <https://docs.langchain.com/oss/python/langgraph/streaming>
- LangGraph event streaming:
  <https://docs.langchain.com/oss/python/langgraph/event-streaming>
- LangGraph subgraphs:
  <https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
- LangGraph fault tolerance:
  <https://docs.langchain.com/oss/python/langgraph/fault-tolerance>

Useful upstream code references:

- `libs/langgraph/langgraph/graph/state.py`: `StateGraph`, channels,
  conditional edges, compile-time validation, node defaults, and subgraph
  attachment.
- `libs/langgraph/langgraph/pregel/main.py`: executable graph runtime,
  superstep loop, streaming, state update APIs, durability, recursion errors,
  and subgraph stream propagation.
- `libs/langgraph/langgraph/types.py`: `Command`, `Send`, `Interrupt`,
  `StateSnapshot`, stream part types, retry/cache/timeout policy, and durability
  modes.
- `libs/langgraph/langgraph/channels/`: reducer/channel implementations such as
  last-value, binary operator aggregate, topic, ephemeral, named barrier, and
  delta channels.
- `libs/checkpoint/langgraph/checkpoint/base/__init__.py`: checkpoint tuple,
  pending writes, thread operations, copy/prune semantics, and delta-channel
  history.

## Responsibilities

- Build named node graphs with direct, conditional, barrier, and command-based
  routing.
- Validate topology before execution and freeze it into an immutable compiled
  graph.
- Execute async nodes under graph and node policies.
- Apply partial state updates through typed channel/reducer policies.
- Execute multiple active nodes in a superstep.
- Support dynamic fanout through `Send`-style packets.
- Enforce recursion, step, timeout, concurrency, retry, and cache policy.
- Persist checkpoints, pending writes, task outcomes, and interrupt state at
  execution boundaries.
- Support human-in-the-loop interrupts and resumable commands.
- Support manual state inspection, state history, state update, forks, and time
  travel when checkpointing is enabled.
- Emit typed graph, task, checkpoint, interrupt, and streamed-output events.
- Represent subgraphs as executable nodes with namespaced checkpoints and nested
  streams.
- Represent harness sub-agents as graph nodes with child run identity.
- Export graph structure for visualization, tests, and generated docs.

## Non-Responsibilities

- It does not own chat model provider logic.
- It does not own tool schema validation or tool dispatch.
- It does not implement prompt templating.
- It does not manage long-term application memory.
- It does not own model/tool usage accounting, though it must forward and roll
  up child events from harness nodes.
- It does not parse the expressive language.

## Feature Specifications

- [Package and core types](package.md)
- [Builder and compile contract](builder.md)
- [Node model](nodes.md)
- [State, channels, and updates](state-channels.md)
- [Edges, routing, commands, and sends](routing.md)
- [Execution model and parallelization](execution.md)
- [Parallel agents and context forking](parallel-agents-forking.md)
- [Checkpointing, durability, state inspection, and time travel](checkpointing.md)
- [Interrupts and resume](interrupts.md)
- [Streaming and events](streaming.md)
- [Observability and tracing](observability.md)
- [Runtime context, node defaults, and policies](runtime-policy.md)
- [Error handling and fault tolerance](fault-tolerance.md)
- [Subgraphs](subgraphs.md)
- [Sub-agents, recursion, and depth tracking](subagents-recursion.md)
- [Memory and stores boundary](memory-boundary.md)
- [Visualization, introspection, and testkit](visualization-testkit.md)
- [Implementation milestones](milestones.md)
