# RustAgents System Specification

RustAgents is a Rust-native LLM application framework inspired by LangChain and
LangGraph. The system is organized around three modules:

1. the harness
2. the graph
3. the expressive language

The goal is to make agent systems easy to define, inspect, run, test, and
eventually serialize without hiding the Rust types that make production systems
reliable.

## Design Goals

- Make simple agent workflows concise.
- Make complex workflows explicit, inspectable, and testable.
- Treat graph execution as a first-class runtime, not an incidental callback
  chain.
- Keep model providers, tools, memory, and tracing behind stable traits.
- Support both Rust builder APIs and a compact expressive language for workflow
  definitions.
- Prefer deterministic state transitions around inherently nondeterministic LLM
  calls.

## Module 1: Harness

The harness is the outer runtime for LLM applications. It owns the pieces needed
to run an agent workflow in a repeatable way: models, tools, state, memory,
callbacks, tracing, retries, and execution policy.

### Responsibilities

- Register chat model providers.
- Register tools and validate tool calls.
- Build model requests from state.
- Apply prompt and message templates.
- Manage per-run context such as run ids, metadata, tags, deadlines, and
  cancellation.
- Provide callback hooks for observability.
- Enforce retry, timeout, and recursion policies.
- Normalize model and tool errors into framework errors.
- Provide test doubles for models and tools.

### Core Types

```rust
pub struct Harness<State> {
    models: ModelRegistry<State>,
    tools: ToolRegistry<State>,
    callbacks: CallbackRegistry<State>,
    policy: RunPolicy,
}

pub struct RunContext {
    pub run_id: String,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
}

pub struct RunPolicy {
    pub recursion_limit: usize,
    pub timeout_ms: Option<u64>,
    pub max_retries: usize,
}
```

### Model Abstraction

Models should be provider-agnostic at the graph layer. The harness binds a model
name to an implementation.

```rust
#[async_trait]
pub trait ChatModel<State>: Send + Sync {
    async fn invoke(
        &self,
        state: &State,
        request: ModelRequest,
    ) -> Result<ModelResponse>;
}
```

Initial provider implementations should be optional feature flags:

- `openai`
- `anthropic`
- `ollama`
- `mock`

### Tool Abstraction

Tools are typed capabilities exposed to agents. The initial surface can accept
JSON arguments, then later add typed schema helpers.

```rust
#[async_trait]
pub trait Tool<State>: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, state: &State, call: ToolCall) -> Result<ToolResult>;
}
```

Tool calls must be observable and replayable. Each call should record:

- tool name
- arguments
- result content
- raw provider result when available
- elapsed time
- error details

### Memory

Memory should be a harness capability, not a graph primitive. Graph nodes may
read or write memory through the harness context.

Memory backends should start with:

- in-memory store for tests
- file-backed store for local development
- trait boundary for external stores

### Observability

Every run should be traceable. At minimum, the harness should emit events for:

- run started
- node started
- node completed
- model requested
- model responded
- tool requested
- tool responded
- route selected
- run completed
- run failed

The event stream should be structured data so it can later feed logs, OpenTelemetry,
or a custom UI.

## Module 2: Graph

The graph is the workflow runtime. It executes stateful nodes, follows direct or
conditional edges, records visited nodes, and returns a final state.

### Responsibilities

- Store named nodes.
- Store direct and conditional edges.
- Validate graph structure before execution.
- Run async node handlers.
- Route based on node output.
- Enforce recursion limits.
- Return final state and execution history.
- Support graph visualization and serialization later.

### Core Concepts

`State` is user-owned application state. RustAgents should never require a
specific state shape.

`Node<State>` is an async unit of work.

`NodeOutput<State>` controls execution:

- `Continue(State)` follows a direct edge.
- `Route { state, route }` follows a conditional edge.
- `End(State)` stops execution.

`StateGraph<State>` owns graph structure and runtime policy.

### Graph Lifecycle

1. Define state.
2. Create nodes.
3. Add edges.
4. Set start node.
5. Validate graph.
6. Run graph with initial state.
7. Inspect final state and visited nodes.

### Routing Semantics

Direct routing:

```text
agent -> summarize -> end
```

Conditional routing:

```text
agent --tool--> tool
agent --final--> end
tool  --------> agent
```

Conditional routes should be explicit strings first. Later versions may support
typed route enums.

### Execution Guarantees

The graph runtime should guarantee:

- every visited node existed at validation time
- every configured edge points to an existing node
- conditional routes fail clearly when missing
- recursion limit failures are deterministic
- final state is returned exactly once

The graph runtime should not guarantee:

- deterministic LLM output
- tool idempotency
- provider-specific retry behavior
- persistence across process restarts unless configured through the harness

### Future Graph Features

- graph serialization to JSON
- Mermaid export
- checkpointing
- resume from checkpoint
- parallel branches
- joins
- subgraphs
- streaming node output
- typed route enums
- static graph analysis

## Module 3: Expressive Language

The expressive language is a compact way to define agent workflows without
writing all builder calls manually. It should compile into the same graph and
harness types as Rust code.

This language is not meant to replace Rust. It is a workflow definition layer for
fast iteration, examples, documentation, and eventually user-authored agent
plans.

### Goals

- Make common agent graphs readable at a glance.
- Keep syntax close to graph intent.
- Compile into explicit RustAgents structures.
- Preserve source locations for helpful errors.
- Avoid embedding arbitrary code in the first version.

### Non-Goals

- It is not a general-purpose programming language.
- It is not a prompt templating language by itself.
- It should not execute untrusted code.
- It should not bypass Rust type checks for stateful logic.

### Initial Syntax Sketch

```rustagents
graph support_agent {
  start agent

  node agent {
    model "default"
    prompt "You are a concise support agent."
    routes {
      tool_call -> tools
      final -> end
    }
  }

  node tools {
    tool_choice auto
    next agent
  }
}
```

### Minimal Grammar

```text
program       = graph_decl*
graph_decl    = "graph" ident "{" graph_item* "}"
graph_item    = start_decl | node_decl | edge_decl
start_decl    = "start" ident
node_decl     = "node" ident "{" node_item* "}"
node_item     = model_decl | prompt_decl | next_decl | routes_decl | tool_decl
model_decl    = "model" string
prompt_decl   = "prompt" string
next_decl     = "next" ident
routes_decl   = "routes" "{" route_decl* "}"
route_decl    = ident "->" (ident | "end")
tool_decl     = "tool_choice" ident
edge_decl     = ident "->" ident
```

### Compilation Pipeline

1. Parse source into an AST.
2. Validate identifiers and route targets.
3. Lower AST into graph builder calls.
4. Bind model and tool references through the harness.
5. Return a compiled workflow object.

### Error Requirements

Errors should include:

- file name when available
- line and column
- invalid token or missing token
- unknown node name
- duplicate node name
- missing start node
- route target that does not exist
- model or tool reference that is not registered in the harness

### Runtime Relationship

The expressive language should produce the same runtime structures as hand-written
Rust:

```text
source -> parser -> AST -> compiler -> StateGraph<State> + Harness bindings
```

The graph runtime should not know whether a graph came from Rust builders or the
expressive language.

## Package Layout

Target module layout:

```text
src/
  chat.rs
  error.rs
  graph.rs
  harness.rs
  language/
    ast.rs
    lexer.rs
    parser.rs
    compiler.rs
    mod.rs
  model.rs
  tool.rs
```

Provider implementations should live behind feature flags:

```text
src/providers/
  openai.rs
  anthropic.rs
  ollama.rs
  mock.rs
```

## Milestones

### Milestone 1: Core Runtime

- Chat message primitives.
- Model trait.
- Tool trait.
- State graph with direct and conditional edges.
- Basic tests and examples.

### Milestone 2: Harness

- Harness type.
- Model registry.
- Tool registry.
- Run context.
- Callback events.
- Mock model and mock tool utilities.

### Milestone 3: Expressive Language Preview

- AST.
- Parser for a small graph definition language.
- Compiler into `StateGraph`.
- Helpful parse and validation errors.
- Example `.rag` or `.rustagents` workflow file.

### Milestone 4: Provider Integrations

- OpenAI chat model provider.
- Anthropic chat model provider.
- Local/mock provider.
- Provider feature flags.

### Milestone 5: Production Runtime Features

- Streaming events.
- Checkpointing.
- Resume support.
- Graph export.
- Tracing integration.

## Open Questions

- Should the expressive language file extension be `.rag`, `.rustagents`, or
  something shorter?
- Should state schemas be declared in the language, or should state remain purely
  Rust-owned?
- Should graph nodes support typed route enums before serialization support?
- Should provider crates live in this crate behind feature flags or in separate
  crates?
- Should memory be synchronous, async, or both?
