//! TinyAgents graph runtime.
//!
//! The graph module is TinyAgents' durable workflow runtime (LangGraph-style):
//! partial updates and reducers ([`reducer`]), commands and interrupts
//! ([`command`]), a builder/compile contract ([`builder`]), a superstep executor
//! ([`compiled`]), checkpointing ([`checkpoint`]), streaming/events
//! ([`stream`]), run-status snapshots ([`status`]), graph export/visualization
//! ([`export`]), and subgraph embedding ([`subgraph`]).
//!
//! Each concern lives in its own submodule with `types.rs` (definitions),
//! `mod.rs` (implementations), and `test.rs` (unit tests).

pub mod builder;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod export;
pub mod reducer;
pub mod status;
pub mod stream;
pub mod subgraph;

// --- Durable execution model ---
pub use builder::{
    END, ForkId, GraphBuilder, NodeContext, NodeFuture, NodeHandler, RouterFn, START,
};
pub use checkpoint::{
    Checkpoint, CheckpointMetadata, Checkpointer, InMemoryCheckpointer, PendingWrite,
};
pub use command::{Command, Interrupt, NodeResult};
pub use compiled::{CompiledGraph, GraphExecution};
pub use export::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphTopology, NodeInfo, RouteInfo,
    blueprint_to_json, blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
pub use reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
pub use status::GraphRunStatus;
pub use stream::{CollectingSink, GraphEvent, GraphEventSink, NoopSink, StreamMode};
pub use subgraph::{adapter_subgraph_node, shared_subgraph_node};
