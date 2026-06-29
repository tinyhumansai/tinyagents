//! TinyAgents graph runtime.
//!
//! The graph module is TinyAgents' durable workflow runtime (LangGraph-style)
//! and one of the load-bearing surfaces of the crate's recursive language-model
//! (RLM) architecture: because a node can embed another compiled graph
//! ([`subgraph`]) or invoke a sub-agent, **graphs run graphs** and orchestration
//! recurses while every step stays typed, checkpointed, and observable. A
//! workflow authored from a `.rag` blueprint or driven from the `.ragsh` REPL
//! lowers into exactly these same types, so a model can describe, compile, and
//! re-enter the very runtime it is executing inside.
//!
//! The pieces: partial updates and reducers ([`reducer`]), commands and
//! interrupts ([`command`]), a builder/compile contract ([`builder`]), a
//! superstep executor ([`compiled`]), checkpointing ([`checkpoint`]),
//! streaming/events ([`stream`]), run-status snapshots ([`status`]), graph
//! export/visualization ([`export`]), and subgraph embedding ([`subgraph`]).
//!
//! Each concern lives in its own submodule with `types.rs` (definitions),
//! `mod.rs` (implementations), and `test.rs` (unit tests).

pub mod builder;
pub mod channel;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod export;
pub mod observability;
pub mod reducer;
pub mod status;
pub mod stream;
pub mod subgraph;

// --- Durable execution model ---
pub use builder::{
    END, ForkId, GraphBuilder, GraphDefaults, NodeContext, NodeFuture, NodeHandler, Route,
    RouterFn, START,
};
pub use channel::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, Delta, Ephemeral,
    LastValue, Messages, NamedBarrier, Topic, Untracked,
};
#[cfg(feature = "sqlite")]
pub use checkpoint::SqliteCheckpointer;
pub use checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSource, CheckpointTuple,
    Checkpointer, DurabilityMode, FileCheckpointer, InMemoryCheckpointer, PendingWrite,
};
pub use command::{Command, Interrupt, NodeResult, RouteTarget, Send};
pub use compiled::{CompiledGraph, GraphExecution, ResumeTarget, StateSnapshot};
pub use export::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphTopology, NodeInfo, RouteInfo,
    blueprint_to_json, blueprint_to_mermaid, blueprint_to_topology, from_json, to_json, to_mermaid,
};
pub use observability::{
    GraphEventJournal, GraphObservation, GraphStatusStore, InMemoryGraphEventJournal,
    InMemoryGraphStatusStore, JournalGraphSink, StoreGraphEventJournal,
};
pub use reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
pub use status::GraphRunStatus;
pub use stream::{CollectingSink, GraphEvent, GraphEventSink, NoopSink, StreamMode};
pub use subgraph::{adapter_subgraph_node, shared_subgraph_node};
