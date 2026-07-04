# REPL Language: RLM Feature Map, Embedding, Safety, Events, Testkit

Continues from [`design.md`](design.md): RLM feature map, CodeAct
loop, example session/graph node, Rhai embedding plan, Python
compatibility backend, safety, events, diagnostics, testkit, and
implementation milestones.

## RLM Feature Map

The goal is to port the useful `rlm` behavior into TinyAgents without porting
Python's unsafe local execution model.

| `rlm` feature               | TinyAgents REPL equivalent                            |
| --------------------------- | ----------------------------------------------------- |
| Python `context` variable   | Rhai `context` variable                               |
| Python persistent locals    | `ReplSession::variables`                              |
| fenced `repl` blocks        | fenced `ragsh` blocks                                 |
| `llm_query`                 | `model_query`                                         |
| `llm_query_batched`         | `model_query_batched`                                 |
| `rlm_query`                 | `agent_query` or `repl_query`                         |
| `rlm_query_batched`         | `agent_query_batched` or `repl_query_batched`         |
| custom Python tools         | registered Rust tool capabilities                     |
| generated Python programs   | `.ragsh` cells plus generated `.rag` graph blueprints |
| `SHOW_VARS()`               | `show_vars()`                                         |
| `answer["ready"] = True`    | `answer(...)`                                         |
| max iterations              | `ReplPolicy::max_iterations` for CodeAct loops        |
| max depth                   | graph/harness recursion policy                        |
| max budget                  | harness cost policy                                   |
| token compaction            | harness summarization feature                         |
| JSONL trajectory logger     | typed event stream plus store backend                 |
| Docker/cloud REPL isolation | future `PythonSandboxRepl` backend                    |

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

```tinyagents
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
backends can reuse the same TinyAgents semantics.

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
- register only TinyAgents capability functions
- expose data through `Dynamic`, maps, and arrays with explicit conversion
- compile and cache ASTs for repeated scripts
- keep each session's `Scope` separate
- restore reserved names after each cell
- truncate stdout and returned values according to policy
- convert Rhai errors into structured diagnostics with spans

Async adapter requirement:

Rhai host functions are easiest to expose as synchronous functions. TinyAgents
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
- must expose the same TinyAgents capability functions
- must enforce the same `ReplPolicy`
- must emit the same `ReplEvent` stream

This lets TinyAgents support Python-like RLM ergonomics without making Python a
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
