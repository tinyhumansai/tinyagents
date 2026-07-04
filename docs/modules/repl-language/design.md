# REPL Language Module Specification

Parent module: [REPL language](README.md).

The REPL language is an interactive orchestration layer for TinyAgents. It is
inspired by Recursive Language Models (`rlm`) and CodeAct-style agents, where a
model can write small programs, inspect their output, call sub-models, and
iterate until it has a final answer.

This module is separate from the expressive language:

- the expressive language (`.rag`) is a declarative graph definition format
- the REPL language (`.ragsh`) is an imperative session language for inspecting,
  scripting, and recursively orchestrating harness and graph runs

Both layers compile or lower into the same harness and graph runtime. Neither
layer should bypass the model registry, tool registry, graph registry, event
system, recursion policy, or run limits.

## Source Inspiration

Primary references:

- `alexzhang13/rlm`: <https://github.com/alexzhang13/rlm>
- RLM paper and docs linked from that repository
- Rhai book: <https://rhai.rs/book/>
- Rhai sandboxing: <https://rhai.rs/book/safety/sandbox.html>
- Rhai operation limits: <https://rhai.rs/book/safety/max-operations.html>
- Rhai API docs: <https://docs.rs/rhai/latest/rhai/>

The useful idea from `rlm` is not Python itself. The useful idea is that context
and intermediate state live in a persistent REPL namespace, while language-model
calls, recursive sub-calls, and tools are exposed as functions inside that
namespace.

TinyAgents should preserve that programming model while making every capability
explicit and typed at the Rust boundary.

## Responsibilities

- Provide an interactive session runtime over harness and graph primitives.
- Execute small scripts with a persistent namespace.
- Expose registered models, agents, graphs, tools, stores, and context as
  capability-bound functions.
- Let sessions draft, validate, inspect, diff, compile, and optionally register
  graph blueprints through the expressive-language compiler.
- Support model-driven CodeAct loops where model output contains fenced REPL
  blocks.
- Capture stdout, return values, state changes, model calls, tool calls, graph
  calls, errors, and final answers as typed events.
- Support recursive sub-model, sub-agent, and sub-graph calls with depth
  tracking.
- Support batched model, agent, and graph calls with bounded concurrency.
- Preserve source spans and session history for diagnostics and replay.
- Provide deterministic test utilities for scripted sessions.

## Non-Responsibilities

- It is not a replacement for the declarative graph language.
- It is not a general-purpose unsafe host-code execution layer.
- It does not provide direct filesystem, network, environment variable, or
  process access.
- It does not own model provider logic.
- It does not own graph topology or checkpointing.
- It does not allow scripts to call unregistered tools or models.
- It does not install model-generated graph topology directly into the runtime;
  generated graphs must pass through the `.rag` compiler and policy checks.

## Why Rhai First

The `rlm` repository uses Python as its default local REPL. Python is effective
for long-context programming because models already know it well, but embedding
Python in Rust would make TinyAgents depend on a large runtime, a separate
sandbox story, and a weaker capability boundary.

Rhai is a better first fit for TinyAgents because it is an embedded scripting
language for Rust with a small host API. It lets TinyAgents register exactly the
functions and values a script may use. The Rhai book describes Rhai as
sandboxed from the host environment by default, with external access provided by
registered functions. It also supports operation limits through
`Engine::set_max_operations`, which gives TinyAgents a direct way to fail closed
on runaway scripts.

Rhai tradeoffs:

- Pros:
  - Rust-native embedding.
  - No Python interpreter dependency.
  - Host-controlled function registration.
  - Familiar JavaScript/Rust-like syntax.
  - Resource limits such as operation counts.
  - Suitable for WASM and other Rust deployment targets.
- Cons:
  - Models know Python better than Rhai.
  - Rhai is dynamically typed, so TinyAgents must validate values at capability
    boundaries.
  - Async host functions require an adapter design.
  - The default `Engine` is not `Send + Sync` unless configured with the Rhai
    `sync` feature, so runtime ownership must be explicit.

Recommendation: use Rhai for the first in-process REPL runtime and document a
future Python compatibility sandbox as a separate environment backend.

## Language Extension

Recommended extension: `.ragsh`.

Reasoning:

- pairs naturally with `.rag`
- reads like "TinyAgents shell"
- avoids implying that the syntax is Rust
- leaves room for future non-Rhai backends

Examples:

```text
support.rag     declarative graph definition
support.ragsh   interactive orchestration script
```

## Runtime Model

The REPL runtime is a session around a capability registry.

```rust
pub struct ReplSession<State, Ctx = ()> {
    pub session_id: SessionId,
    pub run_context: RunContext<Ctx>,
    pub variables: ReplVariables,
    pub capabilities: ReplCapabilities<State, Ctx>,
    pub policy: ReplPolicy,
    pub events: EventSink,
}

pub struct ReplCapabilities<State, Ctx = ()> {
    pub models: ModelRegistry<State, Ctx>,
    pub tools: ToolRegistry<State, Ctx>,
    pub graphs: GraphRegistry<State, Ctx>,
    pub agents: AgentRegistry<State, Ctx>,
    pub stores: StoreRegistry,
    pub language: Option<LanguageCompiler<State, Ctx>>,
}

pub struct ReplPolicy {
    pub max_operations: u64,
    pub max_iterations: usize,
    pub max_script_bytes: usize,
    pub max_output_bytes: usize,
    pub max_model_calls: usize,
    pub max_tool_calls: usize,
    pub max_graph_calls: usize,
    pub max_graph_definitions: usize,
    pub max_depth: usize,
    pub timeout: Option<Duration>,
    pub max_concurrency: usize,
    pub generated_graphs_require_review: bool,
}
```

The session namespace persists across cells. Each cell produces a `ReplResult`.

```rust
pub struct ReplResult {
    pub stdout: String,
    pub value: Option<ReplValue>,
    pub variables_changed: Vec<String>,
    pub calls: Vec<ReplCallRecord>,
    pub final_answer: Option<String>,
    pub elapsed: Duration,
}
```

## Built-In Variables

Initial variables:

- `context`: user input or context payload
- `state`: current graph or agent state when the REPL is used inside a node
- `messages`: current message list when available
- `history`: prior REPL cells and compacted summaries
- `run`: run metadata such as run id, thread id, tags, and depth
- `answer`: final-answer object or helper function

The runtime should restore reserved names after each cell, similar to `rlm`.
Scripts may create local variables, but they may not permanently replace core
capabilities such as `model_query` or `graph_run`.

Reserved names:

- `model_query`
- `model_query_batched`
- `agent_query`
- `agent_query_batched`
- `graph_run`
- `graph_run_batched`
- `graph_define`
- `graph_validate`
- `graph_compile`
- `graph_diff`
- `graph_register`
- `tool_call`
- `tool_call_batched`
- `emit`
- `show_vars`
- `answer`
- `context`
- `state`
- `messages`
- `history`
- `run`

## Built-In Functions

The REPL should expose a small, stable surface. These functions are host
capabilities, not script-native side effects.

### `model_query`

Single provider-neutral model call through the harness.

```rhai
let summary = model_query(#{
  model: "default",
  prompt: "Summarize the relevant facts:\n" + context
});
```

Lowering:

```text
model_query(...) -> ModelRegistry -> ChatModel::invoke -> ModelResponse
```

Requirements:

- validates model alias
- applies harness middleware
- records usage and cost
- emits model events
- increments model-call limits
- returns text by default and structured metadata on request

### `model_query_batched`

Bounded concurrent model calls.

```rhai
let prompts = chunks.map(|chunk| #{
  model: "default",
  prompt: "Extract relevant names:\n" + chunk
});

let answers = model_query_batched(prompts);
```

Requirements:

- preserves input order
- records per-item failures without losing successful results when policy allows
- respects `max_concurrency`
- rolls usage and cost into the parent run

### `agent_query`

Run a registered harness agent loop.

```rhai
let result = agent_query(#{
  agent: "support_agent",
  input: #{
    messages: messages,
    notes: notes
  }
});
```

Lowering:

```text
agent_query(...) -> AgentHarness::run -> AgentRun
```

Use this when the subtask should have model-tool iteration but does not need a
full graph.

### `graph_run`

Run a registered compiled graph.

```rhai
let run = graph_run(#{
  graph: "approval_flow",
  input: state,
  thread_id: run.thread_id
});
```

Lowering:

```text
graph_run(...) -> CompiledGraph::run/resume -> GraphRun
```

Use this when the subtask has explicit topology, routing, interrupts, or
checkpointing.

### `graph_define`

Create a graph blueprint from `.rag` source without installing it.

```rhai
let draft = graph_define(#{
  name: "candidate_support_flow",
  source: `
graph candidate_support_flow {
  start agent

  node agent {
    kind agent
    model "default"
    tools ["lookup_user"]
    routes {
      tool_call -> tools
      final -> END
    }
  }

  node tools {
    kind tool_executor
    next agent
  }
}
`
});
```

Lowering:

```text
graph_define(...) -> LanguageCompiler::parse -> GraphBlueprint
```

Requirements:

- preserves source spans
- records generated-by provenance
- does not compile, register, or run the graph
- counts against `max_graph_definitions`
- rejects source that exceeds policy limits

### `graph_validate`

Parse and resolve a graph blueprint against the current capability allowlist.

```rhai
let diagnostics = graph_validate(draft);
```

Requirements:

- validates syntax, duplicate ids, routes, node kinds, and policies
- checks model, tool, agent, graph, reducer, store, middleware, and script
  references against registries
- returns structured diagnostics that can be shown to the model or user
- does not mutate graph registry state

### `graph_compile`

Compile a validated blueprint into a `CompiledGraph` value under policy.

```rhai
let compiled = graph_compile(draft);
```

Requirements:

- uses the same expressive-language compiler as file-backed `.rag` source
- applies parent run capability allowlists
- marks generated graphs as untrusted unless policy says otherwise
- requires review when `generated_graphs_require_review` is true
- emits compiler and graph blueprint events

### `graph_diff`

Compare two graph blueprints or a blueprint and a registered graph.

```rhai
let diff = graph_diff("support_flow", draft);
```

Requirements:

- reports node, edge, channel, policy, capability, and metadata differences
- preserves source locations where available
- redacts prompt or metadata fields according to event policy
- is deterministic for tests and review UIs

### `graph_register`

Register a compiled graph under a name only when policy permits it.

```rhai
graph_register(#{
  name: "candidate_support_flow",
  graph: compiled,
  review_id: "approval_123"
});
```

Requirements:

- never accepts raw source directly
- requires a compiled graph
- requires a review token when policy says generated graphs need approval
- emits registry events
- does not grant capabilities beyond the compiled graph's validated bindings

### `tool_call`

Call a registered tool by name.

```rhai
let user = tool_call(#{
  tool: "lookup_user",
  arguments: #{ user_id: "usr_123" }
});
```

Requirements:

- validates the tool exists
- validates arguments against the tool schema
- applies middleware
- emits tool events
- records raw and normalized result values

### `emit`

Emit a custom event for tracing and tests.

```rhai
emit("candidate_selected", #{ id: candidate.id, score: candidate.score });
```

### `answer`

Mark the session complete.

```rhai
answer("The account should be escalated to human review.");
```

or, if an object style is preferred:

```rhai
answer.content = "The account should be escalated to human review.";
answer.ready = true;
```

The function style should be the default because it is harder to accidentally
partially mutate.


---

Continues in [`operations.md`](operations.md) (RLM feature map,
CodeAct loop, examples, Rhai embedding plan, Python compatibility
backend, safety, events, diagnostics, testkit, milestones).
