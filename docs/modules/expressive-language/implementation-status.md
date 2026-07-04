# Implementation Status

This tracks what the `src/language/` pipeline implements today against the
grammar and AST sketched in [README.md](README.md). The runtime stays
declarative: the compiler captures topology and policy in an inspectable
[`Blueprint`], while runnable behaviour is supplied by a Rust-side `NodeFactory`.

## Package shape

Flat files under `src/language/`:

| file | role |
| --- | --- |
| `span.rs` | byte+line/column source spans |
| `source.rs` | source files and the source map |
| `diagnostic.rs` | structured diagnostics and the caret renderer |
| `ast.rs` | source AST node types (`Program`, `GraphDecl`, `NodeDecl`, …) |
| `lexer.rs` | source text into spanned tokens |
| `parser.rs` | tokens into the AST |
| `compiler.rs` | AST into one `Blueprint` per graph, capability binding, graph build |
| `types.rs` | tokens + compiled `Blueprint`/`*Spec` types (re-exports `ast`) |

`ast.rs` is re-exported from `types.rs`, so existing
`crate::language::types::{Program, NodeDecl, …}` paths keep resolving.

## Implemented grammar

### Graph-level items

- `start <ident>` — entry node.
- `defaults { key value … }` — graph defaults.
- `input { name type … }` / `output { name type … }` — graph I/O shape
  (lowered to `Blueprint::input` / `Blueprint::output` as `IoFieldSpec`s).
- `checkpoint <ident>` / `interrupt <ident>` — graph-level checkpoint and
  interrupt policies (`Blueprint::checkpoint` / `Blueprint::interrupt`).
- `channel <name> <reducer> <arg>*` — state channel bound to a reducer policy.
  Arguments are string/number literals (e.g. a named aggregate reducer or a
  barrier arrival count) captured in `ChannelSpec::args`.
- `node <name> { … }` — node declarations (see below).
- `from -> to` — static edges.
- `join [a, b] -> c` — top-level barrier (`Blueprint::joins`, `JoinSpec`).

### Node-level items

Common: `kind`, `model`, `system`/`prompt`, `tools [..]`, `next`,
`routes { label -> target }`.

Extended (H2):

- `agent "name"` — sub-agent reference for a `subagent` node (`NodeSpec::agent`).
- `graph "name"` — subgraph reference for a `subgraph` node
  (`NodeSpec::subgraph`; binding prefers it over the legacy `model` field).
- `script "name"` — REPL script capability for a `repl_agent` node
  (`NodeSpec::script`). Declaration only — never inline code.
- `input "mapping"` — input mapping for sub-agent / subgraph nodes.
- `command { goto <target> update { key value … } }` — typed command
  (`NodeSpec::command`, `CommandSpec`). A bare `goto` also lowers into the
  node's routing (precedence: `routes` > `next` > command `goto` > edge >
  terminal).
- `sends [ send <node> ["input"] … ]` — fanout (`NodeSpec::sends`, `SendSpec`).
- `sources [a, b]` — upstream nodes for a `join` node (`NodeSpec::join_sources`).
- `options ["approve", "reject"]` — choices for an `interrupt` node.
- `checkpoint <ident>`, `timeout <literal>`, `retry { … }`, `metadata { … }`
  — node-level policies.

### Node kinds

The registry-backed binding path (`DEFAULT_NODE_KINDS`) accepts `agent`,
`model`, `tool_executor`, `subgraph`, `graph`, `subagent`, `repl_agent`,
`router`, `interrupt`, `join`, and `human`.

## Validation

`compile` rejects: duplicate nodes, missing/undefined `start`, unknown
`next`/`route`/`edge`/`command goto`/`send`/`join` targets, duplicate route
labels, and mixing static routing with `routes`. Registry binding additionally
checks model/tool/subgraph/router/agent/script/reducer references and node
kinds. A single shared policy (`CapabilityResolver::classify_reference`) maps
each node kind to the reference it must resolve, so the compiler blueprint gate
and both `Resolver` paths cannot drift: `subagent` binds its `agent` reference
against the registered agents and `repl_agent` binds its `script` reference
against the registered scripts.

## Not yet implemented

- State-schema declarations (`state Name { … }`).
- Steering policy lowering for `subagent` nodes (parsed shape only is partial).
- Duration literals like `60s` (write timeouts as a number or quoted string).
- Formatter and round-trip golden tests (milestone L8).
- Agent-authored review gates and blueprint provenance (milestone L7).
- An agent-name allowlist on `CapabilityResolver` (sub-agent names are carried
  in the blueprint but not yet registry-validated).
