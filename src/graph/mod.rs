//! RustAgents graph runtime.
//!
//! The graph module is RustAgents' workflow runtime. It hosts two layers that
//! share the crate error type but are otherwise independent:
//!
//! - The **legacy** sequential, whole-state [`StateGraph`] (milestone-1). It is
//!   preserved verbatim — [`Edge`], [`GraphRun`], [`Node`], [`NodeOutput`], and
//!   [`StateGraph`] keep their exact original semantics so existing code, the
//!   `basic_graph` example, and the serialization tests compile unchanged.
//! - The **durable** execution model (LangGraph-style): partial updates and
//!   reducers ([`reducer`]), commands and interrupts ([`command`]), a
//!   builder/compile contract ([`builder`]), a superstep executor
//!   ([`compiled`]), checkpointing ([`checkpoint`]), streaming/events
//!   ([`stream`]), run-status snapshots ([`status`]), and subgraph embedding
//!   ([`subgraph`]).
//!
//! Each concern lives in its own submodule with `types.rs` (definitions),
//! `mod.rs` (implementations), and `test.rs` (unit tests).

pub mod builder;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod legacy;
pub mod reducer;
pub mod status;
pub mod stream;
pub mod subgraph;

// --- Legacy milestone-1 API (preserved, re-exported by the crate root) ---
pub use legacy::{BoxNodeFuture, Edge, GraphRun, Node, NodeFn, NodeOutput, StateGraph};

// --- Durable execution model ---
pub use builder::{END, GraphBuilder, NodeContext, NodeFuture, NodeHandler, RouterFn, START};
pub use checkpoint::{
    Checkpoint, CheckpointMetadata, Checkpointer, InMemoryCheckpointer, PendingWrite,
};
pub use command::{Command, Interrupt, NodeResult};
pub use compiled::{CompiledGraph, GraphExecution};
pub use reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
pub use status::GraphRunStatus;
pub use stream::{CollectingSink, GraphEvent, GraphEventSink, NoopSink, StreamMode};
pub use subgraph::{adapter_subgraph_node, shared_subgraph_node};
