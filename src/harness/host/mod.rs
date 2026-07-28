//! Host capability seams — the extension points an embedding application fills.
//!
//! Every other harness module names a concern the crate itself implements:
//! [`memory`][crate::harness::memory] ships [`InMemoryChatHistory`][crate::harness::memory::InMemoryChatHistory],
//! [`model`][crate::harness::model] ships real providers,
//! [`store`][crate::harness::store] ships a file-backed store. This module is
//! the opposite axis: points where the crate deliberately has **no** opinion
//! and ships only inert defaults, so an embedder can supply retrieval, policy,
//! budgets, routing, and delivery without the runtime importing any of it.
//!
//! The traits follow the same generic-over-`State` pattern as
//! [`Tool`][crate::harness::tool::Tool] and
//! [`ChatModel`][crate::harness::model::ChatModel]: the embedder's application
//! state is threaded to every call, so an implementation reaches its own
//! services through `&State` rather than through crate-visible globals.
//!
//! | Trait | Question it answers |
//! |-------|---------------------|
//! | [`MemoryProvider`] | What does the host know that is relevant to this text? |
//! | [`ContextComposer`] | What text precedes the model request this turn? |
//! | [`SecurityGate`] | Is this tool, path, input, or call permitted? |
//! | [`BudgetGate`] | May this work start, what did it cost, may it continue? |
//! | [`DefinitionRegistry`] | Which agents exist and which is the default? |
//! | [`ExperienceStore`] | What did prior runs of this shape learn? |
//! | [`LearningSink`] | Here is completed work — derive what you like from it. |
//! | [`ProgressSink`] | Deliver run events to an out-of-process consumer. |
//! | [`ToolOutcomeClassifier`] | What kind of failure was that? |
//! | [`ModelResolver`] | Which models should this unit of work use? |
//!
//! # Async only where the work is async
//!
//! Methods are `async` only when a realistic implementation performs I/O.
//! Lookups, lexical path checks, pricing tables, and failure classification are
//! plain `fn`, mirroring the sync [`ChatModel::profile`][crate::harness::model::ChatModel::profile]
//! next to the async [`ChatModel::invoke`][crate::harness::model::ChatModel::invoke].
//! Making a seam `async` forces every caller up the stack to become `async`
//! too, which is a real cost paid by the embedder, not a stylistic choice.
//!
//! # Configuration is not a seam here
//!
//! There is no `ConfigProvider`. Crate-side configuration is expressed as
//! ordinary crate-owned structs the embedder populates when it builds a run.
//! Where a host genuinely needs a *live* value re-read mid-session (a toggle
//! that must take effect without rebuilding), the vehicle is `State`: every
//! method on every trait here receives `&State`, so a host that parks a
//! reloadable handle in its state type keeps read-at-call-time semantics with
//! no crate surface at all.
//!
//! # Scope boundaries worth stating once
//!
//! - [`MemoryProvider`] is **not** conversation history. Thread message logs
//!   are [`ChatHistory`][crate::harness::memory::ChatHistory]; opaque key/value
//!   and append-only streams are [`Store`][crate::harness::store::Store] and
//!   [`AppendStore`][crate::harness::store::AppendStore]. Nothing in this
//!   module reads, rewrites, or compacts a transcript.
//! - [`SecurityGate`] decides what is permitted *inside* an environment;
//!   [`WorkspaceIsolation`][crate::harness::workspace::WorkspaceIsolation]
//!   prepares the environment itself.
//! - [`ProgressSink`] carries the crate's own
//!   [`EventRecord`][crate::harness::events::EventRecord]. Projecting that into
//!   a presentation model is the embedder's job and stays outside the crate.
//! - [`BudgetGate`] is cost and admission only. Context-window compaction is
//!   [`Summarizer`][crate::harness::summarization::Summarizer] plus
//!   [`RunLimits`][crate::harness::limits::RunLimits]; a host compaction
//!   *profile* is host data and rides in [`AgentDefinition::extras`].
//! - [`DefinitionRegistry`] supplies agent definitions. Workspace layout is
//!   [`WorkspaceIsolation`][crate::harness::workspace::WorkspaceIsolation],
//!   prompt personality is [`ContextComposer`], and profile-scoped retrieval is
//!   [`MemoryQuery::namespace`] / [`ExperienceQuery::partition`]. A host
//!   "profile" concept typically decomposes across those three rather than
//!   living here.
//!
//! # Example
//!
//! ```
//! use tinyagents::harness::host::{MemoryProvider, InMemoryMemoryProvider, MemoryQuery, MemoryWrite};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let memory = InMemoryMemoryProvider::new();
//! memory
//!     .write(
//!         &(),
//!         MemoryWrite::new("preferences", "tone", "prefers terse answers"),
//!     )
//!     .await
//!     .unwrap();
//!
//! let hits = memory
//!     .recall(&(), &MemoryQuery::new("terse answers"))
//!     .await
//!     .unwrap();
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].key, "tone");
//! # });
//! ```

mod budget;
mod context;
mod definition;
mod experience;
mod learning;
mod memory;
mod model;
mod progress;
mod security;
mod tool_outcome;

pub use budget::*;
pub use context::*;
pub use definition::*;
pub use experience::*;
pub use learning::*;
pub use memory::*;
pub use model::*;
pub use progress::*;
pub use security::*;
pub use tool_outcome::*;

#[cfg(test)]
mod test;
