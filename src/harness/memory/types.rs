//! Type definitions for the harness memory module.
//!
//! Memory is the conversation- and application-state capability of the harness,
//! distinct from graph checkpoints. It has two conceptual layers:
//!
//! - **Short-term** memory is thread-scoped conversation state (the messages of
//!   one [`ThreadId`]-keyed thread).
//! - **Long-term** memory is cross-thread application state, persisted through
//!   the harness [`Store`][crate::harness::store::Store].
//!
//! All public items are re-exported through [`super`]. Implementations and
//! tests live in the sibling `mod.rs` and `test.rs`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::harness::message::Message;
use crate::harness::store::Store;

/// Distinguishes the two conceptual layers of harness memory.
///
/// This enum is primarily documentary: it lets callers and stored records label
/// which layer a piece of memory belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Thread-scoped conversation state, keyed by a single thread id.
    ShortTerm,
    /// Cross-thread application state, persisted through a long-term store.
    LongTerm,
}

/// Thread-scoped conversation history.
///
/// A `ChatHistory` stores an ordered list of [`Message`]s per `thread_id`.
/// Implementations may be ephemeral ([`InMemoryChatHistory`]) or backed by a
/// durable [`Store`] ([`StoreChatHistory`]). They must be `Send + Sync` so they
/// can be shared across async task boundaries.
#[async_trait]
pub trait ChatHistory: Send + Sync {
    /// Returns the ordered messages for `thread_id`.
    ///
    /// Returns an empty `Vec` if the thread has no history yet.
    async fn messages(&self, thread_id: &str) -> Result<Vec<Message>>;

    /// Appends `message` to the end of `thread_id`'s history.
    async fn append(&self, thread_id: &str, message: Message) -> Result<()>;

    /// Removes all history for `thread_id`.
    ///
    /// This is a no-op if the thread has no history; it does not error.
    async fn clear(&self, thread_id: &str) -> Result<()>;
}

/// Ephemeral, in-process [`ChatHistory`] backed by a shared map.
///
/// Useful for unit tests, examples, and local prototyping. Clones share the
/// same underlying data through the inner [`Arc`]; there is no durability.
#[derive(Clone, Default)]
pub struct InMemoryChatHistory {
    /// `thread_id → messages` map protected by a standard mutex.
    pub(crate) threads: Arc<Mutex<HashMap<String, Vec<Message>>>>,
}

/// A [`ChatHistory`] backed by a long-term [`Store`].
///
/// Each thread's full message list is serialized to JSON and stored as a single
/// value under namespace [`StoreChatHistory::NAMESPACE`] with the `thread_id` as
/// the key. This shows the store boundary working end-to-end: history survives
/// as long as the backing store does.
pub struct StoreChatHistory<S: Store> {
    /// The backing long-term store.
    pub(crate) store: S,
}

/// A thin thread-scoped wrapper over a [`ChatHistory`] with an optional
/// trimming hook.
///
/// `ShortTermMemory` loads and saves the messages for one fixed `thread_id`.
/// When a trimming function is configured it is applied to the loaded messages
/// (for example to cap context length) before they are returned, and to the
/// full message list before it is saved.
pub struct ShortTermMemory<H: ChatHistory> {
    /// The underlying conversation history backend.
    pub(crate) history: H,
    /// The thread this memory is scoped to.
    pub(crate) thread_id: String,
    /// Optional trimming hook applied on load and save.
    #[allow(clippy::type_complexity)]
    pub(crate) trim: Option<Box<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>>,
}
