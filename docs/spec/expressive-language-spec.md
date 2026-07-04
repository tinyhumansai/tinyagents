# Expressive Language Module Specification

The expressive language is a compact way to define agent workflows without
writing all builder calls manually. It should compile into the same graph and
harness types as Rust code.

This language is not meant to replace Rust. It is a workflow definition layer for
fast iteration, examples, documentation, and eventually user-authored agent
plans.

It is also the safe boundary for agent-authored graph plans. A REPL or model may
propose `.rag` source, but that source must pass through the same parser,
diagnostics, registry binding, allowlist checks, review gates, and graph
compiler as human-authored source before it can run.

### Goals

- Make common agent graphs readable at a glance.
- Keep syntax close to graph intent.
- Compile into explicit TinyAgents structures.
- Preserve source locations for helpful errors.
- Avoid embedding arbitrary code in the first version.
- Describe state channels, reducers, policies, subgraphs, sub-agents,
  interrupts, joins, and fanout as declarative graph primitives.
- Produce inspectable blueprints that can be reviewed, diffed, registered, and
  tested.

### Non-Goals

- It is not a general-purpose programming language.
- It is not a prompt templating language by itself.
- It should not execute untrusted code.
- It should not bypass Rust type checks for stateful logic.
- It should not install model-generated topology directly into the graph
  runtime.

### Initial Syntax Sketch

```tinyagents
graph support_agent {
  defaults {
    recursion_limit 50
    checkpoint inherit
  }

  start agent

  channel messages messages
  channel tool_calls append

  node agent {
    kind agent
    model "default"
    prompt "You are a concise support agent."
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

### Minimal Grammar

```text
program       = graph_decl*
graph_decl    = "graph" ident "{" graph_item* "}"
graph_item    = start_decl | defaults_decl | channel_decl | node_decl | edge_decl
start_decl    = "start" ident
defaults_decl = "defaults" object
channel_decl  = "channel" ident reducer_ref
node_decl     = "node" ident "{" node_item* "}"
node_item     = kind_decl | model_decl | prompt_decl | tools_decl | next_decl | routes_decl
kind_decl     = "kind" ident
model_decl    = "model" string
prompt_decl   = "prompt" string
tools_decl    = "tools" "[" string_list? "]"
next_decl     = "next" ident
routes_decl   = "routes" "{" route_decl* "}"
route_decl    = ident "->" (ident | "END")
edge_decl     = ident "->" ident
```

The full language target is broader than this minimal grammar. It should grow
toward commands, `Send` fanout, joins/barriers, subgraphs, sub-agents,
`repl_agent` nodes, interrupts, registered route functions, graph defaults,
capability allowlists, blueprint provenance, and deterministic graph diffs. See
[the expressive language module](../modules/expressive-language/README.md) for
the canonical target.

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

For generated source, the runtime relationship is:

```text
REPL/model proposal -> .rag source or AST -> parser -> diagnostics -> resolver
  -> policy/review gate -> compiler -> GraphBuilder + Harness bindings
  -> CompiledGraph -> optional registry registration
```

