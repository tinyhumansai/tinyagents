# Expressive Language Module Specification

The expressive language is a compact workflow definition layer. It should make
common agent graphs readable without replacing Rust as the implementation
language.

The language compiles into the same harness and graph runtime structures as
hand-written Rust. The graph runtime should not know whether a graph came from
Rust builders or a source file.

The language is also the safe serialization boundary for agent-authored graph
plans. If a REPL agent proposes a new workflow, that proposal should become
`.rag` source or an equivalent AST, then pass through the same parser, resolver,
registry binding, policy checks, and graph compiler as a human-authored file.
Generated topology must never be installed directly into the runtime.

This module is intentionally declarative. Interactive scripting and
CodeAct-style recursive execution belong to the
[REPL language module](../repl-language/README.md). A `.rag` file defines graph topology
and bindings; a `.ragsh` session inspects, scripts, and orchestrates harness or
graph calls through capability-bound functions.

## Responsibilities

- Parse workflow source into an AST.
- Validate syntax, names, routes, and references.
- Compile graph topology into graph builder calls.
- Bind model and tool references through the harness.
- Bind agents, subgraphs, route functions, reducers, stores, middleware, and
  node templates through registries.
- Declare graph input/output shape, state channels, reducer policies, and
  checkpoint/interrupt policy when the compiled graph supports them.
- Declare commands, fanout sends, joins, subgraphs, sub-agents, and REPL-backed
  nodes without embedding arbitrary executable code.
- Produce inspectable blueprints for registries, UIs, documentation, tests, and
  generated workflow review.
- Preserve source spans for clear errors.
- Provide a safe declarative subset for agent workflows.
- Accept both file-backed source and model-generated source through the same
  validation path.
- Support examples, docs, and eventually user-authored workflows.

## Non-Responsibilities

- It is not a general-purpose programming language.
- It does not execute arbitrary Rust.
- It does not replace typed Rust state logic.
- It does not implement model providers.
- It does not own graph execution.
- It does not grant generated source new tools, models, stores, routes, or
  subgraphs that were not already registered and allowed by policy.
- It does not make model-generated graph source trusted merely because a model
  produced it.

## Package Shape

Target layout:

```text
src/language/
  mod.rs
  ast.rs
  compiler.rs
  diagnostic.rs
  lexer.rs
  parser.rs
  resolver.rs
  source.rs
  span.rs
  testkit.rs
```

## File Extension

Candidate extensions:

- `.rustagents`
- `.rag`
- `.agent`

Recommended default: `.rag`.

Reasoning:

- short
- specific to RustAgents Graphs
- easier to use in examples
- does not imply general Rust syntax

The docs can still describe the language as RustAgents source.

## Design Principles

- The syntax should show graph intent first.
- Every named reference should validate before execution.
- Runtime behavior should compile into explicit graph and harness structures.
- Source spans should survive every compiler phase.
- The first version should avoid arbitrary expressions.
- Any future expression support should be pure and deterministic.
- Generated source should be reviewable as a blueprint before it is compiled,
  registered, or run.
- The language should prefer declarative graph primitives over callback names
  whenever the graph runtime has a typed primitive, such as `command`, `send`,
  `interrupt`, `join`, `subgraph`, or `repl_agent`.
- Escape hatches should bind to named Rust capabilities, never inline host code.

## Expressiveness Targets

The long-term language should cover the graph concepts proven useful in
LangGraph, LangChain agent graphs, OpenHuman's state-machine harness, and RLM
style orchestration:

- graph defaults: recursion limits, timeouts, checkpointing, durability,
  streaming modes, cache policy, steering policy, and concurrency
- capabilities: allowed models, tools, agents, graphs, stores, middleware,
  retrievers, route functions, node templates, and REPL scripts
- state channels: messages, scratch state, tool calls, artifacts, candidates,
  usage/cost deltas, interrupt payloads, and custom app fields
- reducers: last value, append, aggregate, topic, messages-by-id, barrier,
  named barrier, delta, and custom registered reducers
- routing: direct edges, conditional routes, typed route labels, command goto,
  `Send` fanout, joins/barriers, parent graph handoff, and terminal output
- execution nodes: model, agent loop, tool executor, subgraph, sub-agent,
  interrupt, router, map/fanout, join, and REPL agent
- observability: source name, graph id, node ids, tags, metadata, event stream
  projections, generated-by provenance, and blueprint version
- safety: source size limits, policy allowlists, review gates for generated
  graphs, and deterministic diagnostics

The first parser does not need to implement all of this at once. The syntax and
AST should leave room for these primitives so early examples do not paint the
runtime into a callback-only design.

## Initial Syntax

```rustagents
graph support_agent {
  metadata {
    description: "Support workflow with tool loop and optional review."
  }

  defaults {
    recursion_limit 50
    timeout 60s
    checkpoint inherit
  }

  start agent

  channel messages messages
  channel tool_calls append
  channel review overwrite

  node agent {
    kind agent
    model "default"
    system "You are a concise support agent."
    tools ["lookup_user", "create_ticket"]

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
```

## Core Concepts

`graph` declares a workflow.

`start` declares the first node after `START`.

`node` declares a named unit of work.

`kind` selects a known node template.

`model` binds to a harness model registry entry.

`tools` selects harness tool registry entries.

`routes` declares conditional routing.

`next` declares a direct edge.

`END` is a reserved graph terminal.

## Grammar Sketch

```text
program        = graph_decl*
graph_decl     = "graph" ident graph_body
graph_body     = "{" graph_item* "}"
graph_item     = start_decl
               | defaults_decl
               | capability_decl
               | channel_decl
               | node_decl
               | edge_decl
               | join_decl
               | metadata_decl

start_decl     = "start" ident
defaults_decl  = "defaults" object
capability_decl = "allow" capability_kind "[" string_list? "]"
channel_decl   = "channel" ident reducer_ref
edge_decl      = ident "->" node_ref
join_decl      = "join" "[" ident_list "]" "->" ident
metadata_decl  = "metadata" object

node_decl      = "node" ident node_body
node_body      = "{" node_item* "}"
node_item      = kind_decl
               | model_decl
               | system_decl
               | prompt_decl
               | tools_decl
               | next_decl
               | routes_decl
               | command_decl
               | sends_decl
               | retry_decl
               | timeout_decl
               | steering_decl
               | checkpoint_decl
               | metadata_decl

kind_decl      = "kind" ident
model_decl     = "model" string
system_decl    = "system" string
prompt_decl    = "prompt" string
tools_decl     = "tools" "[" string_list? "]"
next_decl      = "next" node_ref
routes_decl    = "routes" "{" route_decl* "}"
route_decl     = ident "->" node_ref
command_decl   = "command" object
sends_decl     = "sends" "[" send_decl* "]"
retry_decl     = "retry" object
timeout_decl   = "timeout" duration
steering_decl  = "steering" object
checkpoint_decl = "checkpoint" ident

node_ref       = ident | "END"
```

## AST

```rust
pub struct Program {
    pub graphs: Vec<GraphDecl>,
}

pub struct GraphDecl {
    pub name: Ident,
    pub items: Vec<GraphItem>,
    pub span: Span,
}

pub enum GraphItem {
    Start(StartDecl),
    Defaults(DefaultsDecl),
    Capability(CapabilityDecl),
    Channel(ChannelDecl),
    Node(NodeDecl),
    Edge(EdgeDecl),
    Join(JoinDecl),
    Metadata(MetadataDecl),
}

pub struct NodeDecl {
    pub name: Ident,
    pub items: Vec<NodeItem>,
    pub span: Span,
}

pub enum NodeItem {
    Kind(Ident),
    Model(StringLit),
    System(StringLit),
    Prompt(StringLit),
    Tools(Vec<StringLit>),
    Next(NodeRef),
    Routes(Vec<RouteDecl>),
    Command(ObjectLit),
    Sends(Vec<SendDecl>),
    Retry(ObjectLit),
    Timeout(DurationLit),
    Steering(ObjectLit),
    Checkpoint(Ident),
    Metadata(ObjectLit),
}
```

Every AST node carries a `Span`.

## Compiler Pipeline

```text
source
  -> lexer
  -> parser
  -> AST
  -> resolver
  -> validated graph plan
  -> compiler
  -> GraphBuilder + Harness bindings
  -> CompiledGraph
```

Phases:

1. Lex tokens.
2. Parse tokens into AST.
3. Resolve graph names.
4. Resolve node names.
5. Validate duplicate declarations.
6. Validate `START` and `END` use.
7. Validate route targets.
8. Validate model references against `ModelRegistry`.
9. Validate tool references against `ToolRegistry`.
10. Lower node templates into graph nodes.
11. Compile graph topology.
12. Return compiled workflow and diagnostics.

## Diagnostics

Diagnostics should be structured:

```rust
pub struct Diagnostic {
    pub severity: Severity,
    pub code: DiagnosticCode,
    pub message: String,
    pub span: Span,
    pub labels: Vec<Label>,
    pub help: Option<String>,
}
```

Required errors:

- invalid token
- unterminated string
- unexpected token
- duplicate graph
- duplicate node
- missing start node
- unknown node
- unknown route target
- unknown model
- unknown tool
- invalid node kind
- invalid timeout
- duplicate route
- incompatible node items
- unknown graph
- unknown agent
- unknown route function
- unknown reducer
- unknown store
- disallowed generated graph capability
- generated graph requires review
- checkpoint policy incompatible with interrupts
- state channel missing reducer
- send target missing input mapping
- steering target not allowed
- steering policy references unknown actor or capability

Example diagnostic:

```text
error[E-rag-unknown-node]: route target `toolz` does not exist
  --> support.rag:11:20
   |
11 |       tool_call -> toolz
   |                    ^^^^^ unknown node
   |
help: did you mean `tools`?
```

## Node Kinds

Initial built-in node kinds:

### `agent`

Uses the harness agent loop or one model call depending on config.

Supported fields:

- `model`
- `system`
- `prompt`
- `tools`
- `routes`
- `retry`
- `timeout`

### `model`

Single model invocation. Does not automatically execute tools.

Supported fields:

- `model`
- `system`
- `prompt`
- `routes`
- `retry`
- `timeout`

### `tool_executor`

Executes tool calls already present in state.

Supported fields:

- `tools`
- `next`
- `retry`
- `timeout`

### `router`

Routes based on a named route function provided from Rust.

Supported fields:

- `routes`
- `metadata`

### `subgraph`

Calls another compiled graph.

Supported fields:

- `graph`
- `next`
- `routes`

### `subagent`

Calls a registered harness agent as a graph node.

Supported fields:

- `agent`
- `input`
- `next`
- `routes`
- `retry`
- `timeout`
- `steering`

Example:

```rustagents
node research {
  kind subagent
  agent "researcher"
  steering {
    parent allow ["add_instruction", "request_status", "cancel"]
    human allow ["add_instruction", "pause", "resume", "cancel"]
    delivery "safe_boundary"
  }
  next synthesize
}
```

Steering policies lower into harness steering policy and graph task policy. They
can narrow a child agent's model/tool/runtime limits but cannot grant
capabilities absent from the registry or parent run policy.

### `repl_agent`

Runs a registered REPL script or model-driven CodeAct loop through the harness
REPL runtime.

Supported fields:

- `model`
- `script`
- `tools`
- `routes`
- `retry`
- `timeout`

### `interrupt`

Emits a resumable human-in-the-loop interrupt.

Supported fields:

- `prompt`
- `options`
- `routes`
- `metadata`

### `join`

Waits for named upstream nodes or barrier channels before continuing.

Supported fields:

- `sources`
- `next`
- `timeout`

## Binding To Rust

The language should not define arbitrary Rust closures. Instead, it should bind
to Rust-provided registries:

```rust
let workflow = LanguageCompiler::new()
    .with_models(models)
    .with_tools(tools)
    .with_node_templates(templates)
    .compile_source("support.rag", source)?;
```

Registries:

- model registry
- tool registry
- agent registry
- node template registry
- route function registry
- reducer registry
- graph registry for subgraphs
- store registry
- middleware registry
- REPL script registry

When a graph is generated by a REPL session, the session may call the compiler
with source text or an AST, but the compiler must use the same registries and
policy checks. Generated source can request capabilities only from the allowed
set attached to the parent run or registry namespace.

This keeps source files declarative and prevents unsafe dynamic execution.

## State Model

Version 1 should keep state Rust-owned. The language can refer to standard
channels by convention and bind them to registered reducers:

- `messages`
- `tool_calls`
- `structured_response`
- `metadata`
- `artifacts`
- `candidates`
- `usage`
- `interrupts`

Example:

```rustagents
channel messages messages
channel candidates append
channel usage aggregate "usage_delta"
channel review overwrite
```

Future versions may add state schema declarations:

```rustagents
state SupportState {
  messages: messages append
  customer_id: string overwrite
  ticket_id: string? overwrite
}
```

State schemas should be delayed until reducer-based graph execution exists.

## Routes

Routes are named outcomes.

```rustagents
routes {
  tool_call -> tools
  final -> END
  escalate -> human_review
}
```

Rules:

- route names are unique per node
- route targets must exist or be `END`
- route names are ASCII identifiers
- a node may use `routes` or `next`, not both
- `END` is reserved

Future typed route support can generate Rust enums from route declarations.

## Policies

Node-level policies:

```rustagents
node agent {
  timeout 30s
  retry {
    max_attempts: 3
    backoff: "exponential"
  }
}
```

Graph-level defaults:

```rustagents
graph support_agent {
  defaults {
    timeout 60s
    recursion_limit 50
  }
}
```

Policies lower into graph node policies and harness request policies.

## Comments And Strings

Comments:

```rustagents
// line comment
```

Strings:

```rustagents
system "single line"

prompt """
multi-line prompt
"""
```

Multi-line strings should preserve content exactly except for one predictable
dedent rule.

## Safety

The language must be safe to parse and validate from untrusted text.

Safety rules:

- no arbitrary code execution
- no filesystem access from language source
- no network access from language source
- no dynamic provider lookup without registry binding
- no environment variable interpolation in v1
- bounded parser recursion
- bounded source size

## Examples

### Minimal Model Graph

```rustagents
graph summarize {
  start model

  node model {
    kind model
    model "default"
    system "Summarize the user request."
    next END
  }
}
```

### Agent With Tools

```rustagents
graph support_agent {
  start agent

  node agent {
    kind agent
    model "default"
    system "Resolve support requests using tools when useful."
    tools ["lookup_user", "create_ticket"]
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
```

### Human Review

```rustagents
graph approval_flow {
  start draft

  node draft {
    kind model
    model "default"
    system "Draft a response."
    next review
  }

  node review {
    kind interrupt
    prompt "Approve this response?"
    routes {
      approved -> send
      rejected -> draft
    }
  }

  node send {
    kind tool_executor
    tools ["send_email"]
    next END
  }
}
```

## Formatting

Formatter goals:

- stable ordering within declarations
- preserve comments
- normalize indentation to two spaces
- one item per line for lists longer than one entry
- avoid rewriting prompt body content

The formatter can come after parser and diagnostics.

## Testkit

`language::testkit` should include:

- parse snapshot helper
- diagnostic snapshot helper
- compile helper with fake registries
- golden source fixtures
- round-trip formatter tests once formatter exists

## Implementation Milestones

### L1: Parser Skeleton

- token model
- spans
- lexer
- parser for graph, start, node, next, routes

### L2: Diagnostics

- structured diagnostics
- duplicate node validation
- unknown route target validation

### L3: Compiler Preview

- compile topology into `GraphBuilder`
- support `kind model`
- support `kind agent`
- bind model names

### L4: Tool Binding

- validate tool names
- compile `tool_executor`
- add agent/tool graph example

### L5: Policies And Subgraphs

- parse timeout and retry
- bind subgraphs
- compile node policies

### L6: Channels, Commands, And Fanout

- parse channel declarations
- bind reducer registry entries
- lower `command` and `sends`
- compile join/barrier nodes

### L7: Agent-Authored Graphs

- compile generated source under parent run policy
- require review gates when policy marks generated graphs as untrusted
- store generated blueprint provenance
- expose graph diff and preview diagnostics

### L8: Formatter

- stable formatting
- golden tests
