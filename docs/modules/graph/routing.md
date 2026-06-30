# Graph Edges, Routing, Commands, And Sends

Reserved virtual nodes:

- `START`
- `END`

Direct edge:

```text
START -> agent
agent -> summarize
summarize -> END
```

Conditional edge:

```text
agent --tool--> tools
agent --final--> END
tools ---------> agent
```

Waiting/barrier edge:

```text
retrieve_docs --\
lookup_user  ----> synthesize
score_risk   --/
```

Command routing:

```rust
Command::new()
    .update(update)
    .goto(["tools"])
```

Typed routes should be supported after string routes:

```rust
enum AgentRoute {
    Tool,
    Final,
}
```

Routing outputs:

- node name
- `END`
- one or more node names
- `Send` packet
- one or more `Send` packets
- `Command`

Branch metadata should include an optional path map and typed route labels so
visualization does not have to assume every conditional edge can reach every
node.

## Commands And Send Packets

Commands combine state update, routing, parent/subgraph targeting, and interrupt
resume values.

```rust
pub struct Command {
    pub graph: CommandGraphTarget,
    pub update: Option<StateUpdate>,
    pub resume: Option<ResumeValue>,
    pub goto: Vec<RouteTarget>,
}

pub enum CommandGraphTarget {
    Current,
    Parent,
    Graph(GraphId),
}

pub enum RouteTarget {
    Node(NodeId),
    Send(Send),
}

pub struct Send {
    pub node: NodeId,
    pub arg: serde_json::Value,
}
```

`Command::goto([..])` (plain node activations) and `Command::send([Send::new(node,
arg), ..])` (per-invocation fanout) both populate `Command::goto:
Vec<RouteTarget>`; `with_goto`/`with_sends` append to it. A `Send`-scheduled node
receives its `arg` on `NodeContext::send_arg` (`None` for normal activations);
many `Send`s may target the *same* node, producing one parallel activation each
(map-reduce). Plain activations are deduplicated by node; `Send` activations are
not.

Runtime command targets are validated before they become the next active set or
are persisted in a checkpoint:

- `END` is allowed and terminates that branch.
- `START` is rejected because it is a virtual entry marker, not an executable
  runtime target.
- every other target must name a compiled node.
- invalid runtime targets fail the run before the boundary checkpoint is
  written, so durable state cannot be poisoned with missing next nodes.

Use `Command` for:

- dynamic routing
- node-local state update plus routing
- human approval resume values
- parent graph handoff from subgraphs
- supervisor/worker handoff

Use `Send` for dynamic fanout where each target invocation receives custom
input that can differ from the graph's main state. This is the primitive for
map-reduce, search fanout, parallel tool calls, and per-item scoring.

## External Run Inputs

Most runs enter through the compiled `START -> entry` edge:

```rust
graph.run(state).await?;
```

When the caller needs to seed multiple graph loops at once, use
`GraphInput`:

```rust
graph
    .run_with_inputs(
        state,
        [
            GraphInput::start(json!({ "message": "hello" })),
            GraphInput::new("tool_loop", json!({ "tool": "search" })),
        ],
    )
    .await?;
```

`GraphInput::start(..)` resolves to the compiled entry node and delivers its
payload through `NodeContext::send_arg`. `GraphInput::new(node, ..)` targets a
real node directly in the first superstep. Inputs are not deduplicated, so two
inputs aimed at the same node run two activations with distinct payloads.
