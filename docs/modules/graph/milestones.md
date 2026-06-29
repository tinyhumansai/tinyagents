# Graph Implementation Milestones

## G1: Current Sequential Runtime

- `Node`
- `NodeOutput`
- `StateGraph`
- direct edges
- conditional edges
- recursion limit

## G2: Builder And Compile Step

- introduce `GraphBuilder`
- introduce `CompiledGraph`
- move validation to `compile`
- add `START` and `END`
- add input/output schema concepts
- preserve current closure-node ergonomics

## G3: Commands, Sends, And Typed Events

- add `Command`
- add `Send`
- add `GraphEvent`
- add event recorder
- add destination hints for command-rendered edges

## G4: Channels And Partial Updates

- add `Channel` trait
- add state update type
- add last-value, append, aggregate, topic, and message reducers
- add invalid concurrent update errors
- add partial update examples

## G5: Supersteps And Parallel Execution

- add multi-active-task executor
- apply writes at step boundaries
- add waiting/barrier edges
- add concurrency limits
- add task stream events

## G6: Checkpointing, Interrupts, And Time Travel

- add `Checkpointer`
- add in-memory backend
- add checkpoint tuples and pending writes
- add interrupt/resume API
- add `get_state`, `get_state_history`, and `update_state`
- add durability modes

## G7: Policies, Cache, And Error Handlers

- add graph and node defaults
- add retry, timeout, cache, and error-handler policies
- replay cached writes
- add cooperative drain

## G8: Subgraphs And Namespaces

- add subgraph node
- add checkpoint namespaces
- add parent command routing
- stream nested subgraph events
- expose child state in task metadata

## G9: Sub-Agents And Recursion

- add `SubAgentNode`
- add child run hierarchy
- add context forking for parallel sub-agents
- add shared-cache fork policy
- add recursion stack
- add depth events
- add max-depth policy
- roll up child usage and cost events

## G10: Full Graph Streaming And Introspection

- expose async graph event streams
- add stream projections for values, updates, messages, custom data,
  checkpoints, tasks, and debug
- forward harness streams with node context
- add stream transformers
- add JSON and Mermaid graph exports
