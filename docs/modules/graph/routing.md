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
    pub timeout: Option<TimeoutPolicy>,
}
```

Use `Command` for:

- dynamic routing
- node-local state update plus routing
- human approval resume values
- parent graph handoff from subgraphs
- supervisor/worker handoff

Use `Send` for dynamic fanout where each target invocation receives custom
input that can differ from the graph's main state. This is the primitive for
map-reduce, search fanout, parallel tool calls, and per-item scoring.
