# REPL Language Module Specification

Parent module: [REPL language](README.md).

The REPL language is an interactive orchestration layer for RustAgents. It is
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

RustAgents should preserve that programming model while making every capability
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
Python in Rust would make RustAgents depend on a large runtime, a separate
sandbox story, and a weaker capability boundary.

Rhai is a better first fit for RustAgents because it is an embedded scripting
language for Rust with a small host API. It lets RustAgents register exactly the
functions and values a script may use. The Rhai book describes Rhai as
sandboxed from the host environment by default, with external access provided by
registered functions. It also supports operation limits through
`Engine::set_max_operations`, which gives RustAgents a direct way to fail closed
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
  - Rhai is dynamically typed, so RustAgents must validate values at capability
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
- reads like "RustAgents shell"
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

## RLM Feature Map

The goal is to port the useful `rlm` behavior into RustAgents without porting
Python's unsafe local execution model.

| `rlm` feature | RustAgents REPL equivalent |
| --- | --- |
| Python `context` variable | Rhai `context` variable |
| Python persistent locals | `ReplSession::variables` |
| fenced `repl` blocks | fenced `ragsh` blocks |
| `llm_query` | `model_query` |
| `llm_query_batched` | `model_query_batched` |
| `rlm_query` | `agent_query` or `repl_query` |
| `rlm_query_batched` | `agent_query_batched` or `repl_query_batched` |
| custom Python tools | registered Rust tool capabilities |
| generated Python programs | `.ragsh` cells plus generated `.rag` graph blueprints |
| `SHOW_VARS()` | `show_vars()` |
| `answer["ready"] = True` | `answer(...)` |
| max iterations | `ReplPolicy::max_iterations` for CodeAct loops |
| max depth | graph/harness recursion policy |
| max budget | harness cost policy |
| token compaction | harness summarization feature |
| JSONL trajectory logger | typed event stream plus store backend |
| Docker/cloud REPL isolation | future `PythonSandboxRepl` backend |

## CodeAct Loop

A model-driven REPL agent has this lifecycle:

1. Create `ReplSession`.
2. Load `context`, `state`, `messages`, `history`, and `run` variables.
3. Build a model request explaining the available REPL functions.
4. Invoke the model through the harness.
5. Extract fenced `ragsh` blocks from the assistant message.
6. Execute each block in the REPL session.
7. Capture stdout, changed variables, call records, events, and errors.
8. Append a compact execution result as the next user message.
9. Repeat until `answer(...)` is called or limits are reached.
10. Persist events, usage, cost, and final answer.

This loop is a harness feature. When used inside a graph node, the graph still
owns node routing, checkpointing, interrupts, recursion depth, and failure
policy.

If the model writes `.rag` source, the loop should treat it as a graph proposal.
The REPL may validate, diff, compile, and run that proposal only through the
expressive-language compiler and the graph registry policy. This is how an
agent can define its own graph without acquiring arbitrary topology mutation or
host-code execution privileges.

## Example Session

```rhai
let lines = context.split("\n");
let candidates = [];

for line in lines {
  if line.contains("SECRET_NUMBER=") {
    candidates.push(line);
  }
}

emit("candidates_found", #{ count: candidates.len() });

let result = model_query(#{
  model: "default",
  prompt: "Return only the digits from this candidate line:\n" + candidates[0]
});

answer(result);
```

## Example Graph Node

```rustagents
graph support_repl {
  start investigate

  node investigate {
    kind repl_agent
    model "default"
    script "support-investigation.ragsh"
    tools ["lookup_user", "create_ticket"]
    routes {
      final -> END
      needs_review -> review
    }
  }

  node review {
    kind interrupt
    prompt "Approve escalation?"
    routes {
      approved -> END
      rejected -> investigate
    }
  }
}
```

The `repl_agent` node is a harness-backed node template. It may execute a fixed
script, a model-driven CodeAct loop, or a combination where a fixed prologue
sets up variables before the model starts writing cells.

## Rhai Embedding Plan

The Rhai runtime should be isolated behind an interface so future Python or WASM
backends can reuse the same RustAgents semantics.

```rust
#[async_trait]
pub trait ReplBackend<State, Ctx = ()>: Send {
    async fn execute_cell(
        &mut self,
        session: &mut ReplSession<State, Ctx>,
        source: SourceCell,
    ) -> Result<ReplResult>;
}

pub struct RhaiReplBackend {
    engine: rhai::Engine,
    ast_cache: AstCache,
}
```

Rhai-specific requirements:

- configure `Engine::set_max_operations`
- disable or avoid unneeded packages
- register only RustAgents capability functions
- expose data through `Dynamic`, maps, and arrays with explicit conversion
- compile and cache ASTs for repeated scripts
- keep each session's `Scope` separate
- restore reserved names after each cell
- truncate stdout and returned values according to policy
- convert Rhai errors into structured diagnostics with spans

Async adapter requirement:

Rhai host functions are easiest to expose as synchronous functions. RustAgents
model, tool, and graph calls are async. The backend should not hide blocking in
unbounded threads. Use one of these designs:

1. command recording: Rhai functions create `ReplCommand` values, then the
   Rust async runtime executes those commands after the cell
2. blocking bridge: host functions call into a bounded runtime handle with
   strict timeouts
3. staged syntax: `let x = model_query(...)` is transformed before evaluation
   into host-executed calls

Recommendation for v1: use a blocking bridge only in examples and tests, but
design the public API around command recording. Command recording is easier to
make deterministic and safer under async graph execution.

## Python Compatibility Backend

Python should be a compatibility backend, not the default embedded runtime.

```rust
pub struct PythonSandboxReplBackend {
    sandbox: SandboxClient,
}
```

Potential use cases:

- training model behavior that already expects Python
- local research workflows
- compatibility with RLM-style prompts
- data-heavy scripts where Python libraries are explicitly useful

Requirements:

- must run out of process
- must have no direct host filesystem access by default
- must communicate through a framed JSON protocol
- must expose the same RustAgents capability functions
- must enforce the same `ReplPolicy`
- must emit the same `ReplEvent` stream

This lets RustAgents support Python-like RLM ergonomics without making Python a
trusted in-process extension language.

## Safety

Safety rules:

- no arbitrary filesystem access in the default Rhai backend
- no environment variable interpolation from scripts
- no direct network access
- no process spawning
- no unregistered native functions
- bounded script size
- bounded operation count
- bounded output size
- bounded model/tool/graph calls
- bounded recursion depth
- bounded concurrency
- typed conversion at every capability boundary
- redaction before event and store writes

The REPL is an orchestration surface, not a privilege escalation surface.

## Events

The REPL event stream should compose with graph and harness events.

```rust
pub enum ReplEvent {
    SessionStarted { session_id: SessionId, run_id: RunId },
    CellStarted { cell_id: CellId, source_name: String },
    CellStdout { cell_id: CellId, chunk: String },
    CellCompleted { cell_id: CellId, elapsed: Duration },
    CellFailed { cell_id: CellId, diagnostic: Diagnostic },
    VariableChanged { cell_id: CellId, name: String },
    CapabilityCallStarted { cell_id: CellId, call_id: CallId, name: String },
    CapabilityCallCompleted { cell_id: CellId, call_id: CallId },
    GraphBlueprintDefined { cell_id: CellId, graph_name: String },
    GraphBlueprintValidated { cell_id: CellId, graph_name: String },
    GraphBlueprintCompiled { cell_id: CellId, graph_name: String },
    GraphBlueprintRegistered { cell_id: CellId, graph_name: String },
    FinalAnswer { cell_id: CellId, content: String },
    SessionCompleted { session_id: SessionId },
    SessionFailed { session_id: SessionId, error: String },
}
```

When the REPL calls a model, tool, agent, or graph, the child harness/graph
events should preserve:

- root run id
- parent run id
- cell id
- node id when used inside a graph
- recursion depth
- capability name

## Diagnostics

Diagnostics should preserve source spans from scripts and model-generated cells.

Required errors:

- invalid script syntax
- unknown capability
- unknown model
- unknown tool
- unknown graph
- invalid graph source
- graph compilation failed
- generated graph review required
- graph registration denied
- invalid arguments
- unsupported value type
- operation limit exceeded
- timeout exceeded
- output limit exceeded
- call limit exceeded
- recursion limit exceeded
- unsafe backend requested
- reserved name overwrite

Example:

```text
error[E-ragsh-unknown-tool]: tool `lookup_usr` is not registered
  --> support.ragsh:8:18
   |
8  | let user = tool_call(#{ tool: "lookup_usr", arguments: #{ id: id } });
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^^ unknown tool
   |
help: did you mean `lookup_user`?
```

## Testkit

`repl::testkit` should include:

- fake model capability
- fake tool capability
- fake graph capability
- deterministic event recorder
- script execution helper
- CodeAct loop helper with scripted model responses
- operation-limit assertion
- output-limit assertion
- recursive-call assertion
- batched-call ordering assertion
- golden trajectory fixtures

## Implementation Milestones

### R1: Documentation And Types

- add this module doc
- add `repl` package shape to the spec
- define `ReplSession`, `ReplPolicy`, `ReplResult`, and `ReplEvent`
- no Rhai dependency yet

### R2: Rhai Prototype

- add optional `repl-rhai` feature
- embed Rhai behind `ReplBackend`
- support persistent variables
- support `show_vars`, `emit`, and `answer`
- enforce operation and output limits

### R3: Harness Capabilities

- add `model_query`
- add `model_query_batched`
- add fake-model tests
- forward harness events through REPL events

### R4: Tool And Agent Capabilities

- add `tool_call`
- add `agent_query`
- validate schemas and limits
- record usage and cost rollups

### R5: Graph Capability

- add `graph_run`
- add `graph_define`, `graph_validate`, `graph_compile`, `graph_diff`, and
  `graph_register`
- support graph-node `kind repl_agent`
- preserve node id, parent run id, and depth in child events
- require generated-graph review gates when policy enables them

### R6: CodeAct Loop

- parse fenced `ragsh` blocks from assistant messages
- execute cells iteratively
- append compact execution feedback to model history
- stop on `answer(...)`
- add trajectory logging and tests

### R7: Python Sandbox Backend

- add optional out-of-process backend
- expose the same capability protocol
- run RLM-compatible Python scripts under explicit policy
- keep it disabled by default
