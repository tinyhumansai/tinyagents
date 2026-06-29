//! Short-term conversation memory and its store boundary.
//!
//! This module provides the harness memory capability: thread-scoped
//! conversation history ([`ChatHistory`]) with both an ephemeral
//! ([`InMemoryChatHistory`]) and a store-backed ([`StoreChatHistory`])
//! implementation, plus a thin thread-scoped wrapper ([`ShortTermMemory`]) that
//! applies an optional trimming policy. Long-term, cross-thread memory is the
//! harness [`Store`][crate::harness::store::Store] itself; the [`MemoryScope`]
//! enum labels which layer a record belongs to.
//!
//! See [`types`] for the definitions.
//!
//! # Example
//!
//! ```
//! use rustagents::harness::memory::{ChatHistory, InMemoryChatHistory};
//! use rustagents::harness::message::Message;
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let history = InMemoryChatHistory::new();
//! history.append("t1", Message::user("hello")).await.unwrap();
//! let msgs = history.messages("t1").await.unwrap();
//! assert_eq!(msgs.len(), 1);
//! assert_eq!(msgs[0].text(), "hello");
//! # });
//! ```

mod types;

pub use types::*;

use async_trait::async_trait;

use crate::error::{Result, RustAgentsError};
use crate::harness::message::Message;
use crate::harness::store::Store;

// ── InMemoryChatHistory ───────────────────────────────────────────────────────

impl InMemoryChatHistory {
    /// Creates a new, empty in-memory chat history.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ChatHistory for InMemoryChatHistory {
    async fn messages(&self, thread_id: &str) -> Result<Vec<Message>> {
        let threads = self
            .threads
            .lock()
            .map_err(|e| RustAgentsError::Memory(format!("chat history lock poisoned: {e}")))?;
        Ok(threads.get(thread_id).cloned().unwrap_or_default())
    }

    async fn append(&self, thread_id: &str, message: Message) -> Result<()> {
        let mut threads = self
            .threads
            .lock()
            .map_err(|e| RustAgentsError::Memory(format!("chat history lock poisoned: {e}")))?;
        threads
            .entry(thread_id.to_string())
            .or_default()
            .push(message);
        Ok(())
    }

    async fn clear(&self, thread_id: &str) -> Result<()> {
        let mut threads = self
            .threads
            .lock()
            .map_err(|e| RustAgentsError::Memory(format!("chat history lock poisoned: {e}")))?;
        threads.remove(thread_id);
        Ok(())
    }
}

// ── StoreChatHistory ──────────────────────────────────────────────────────────

impl<S: Store> StoreChatHistory<S> {
    /// Namespace under which thread histories are persisted in the store.
    pub const NAMESPACE: &'static str = "chat_history";

    /// Wraps `store` as a chat-history backend.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// Returns a reference to the backing store.
    pub fn store(&self) -> &S {
        &self.store
    }
}

#[async_trait]
impl<S: Store> ChatHistory for StoreChatHistory<S> {
    async fn messages(&self, thread_id: &str) -> Result<Vec<Message>> {
        match self.store.get(Self::NAMESPACE, thread_id).await? {
            Some(value) => {
                let messages: Vec<Message> = serde_json::from_value(value)?;
                Ok(messages)
            }
            None => Ok(Vec::new()),
        }
    }

    async fn append(&self, thread_id: &str, message: Message) -> Result<()> {
        let mut messages = self.messages(thread_id).await?;
        messages.push(message);
        let value = serde_json::to_value(&messages)?;
        self.store.put(Self::NAMESPACE, thread_id, value).await
    }

    async fn clear(&self, thread_id: &str) -> Result<()> {
        self.store.delete(Self::NAMESPACE, thread_id).await
    }
}

// ── ShortTermMemory ───────────────────────────────────────────────────────────

impl<H: ChatHistory> ShortTermMemory<H> {
    /// Scopes `history` to a single `thread_id` with no trimming.
    pub fn new(history: H, thread_id: impl Into<String>) -> Self {
        Self {
            history,
            thread_id: thread_id.into(),
            trim: None,
        }
    }

    /// Installs a trimming hook applied on [`load`](Self::load) and
    /// [`save`](Self::save).
    ///
    /// The hook receives the messages and returns the (possibly shorter) list to
    /// use, allowing context-window capping or summarization-driven trimming.
    pub fn with_trim(
        mut self,
        trim: impl Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.trim = Some(Box::new(trim));
        self
    }

    /// The thread this memory is scoped to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// Loads the thread's messages, applying the trimming hook if configured.
    pub async fn load(&self) -> Result<Vec<Message>> {
        let messages = self.history.messages(&self.thread_id).await?;
        Ok(self.apply_trim(messages))
    }

    /// Appends `message` to the thread's history.
    pub async fn append(&self, message: Message) -> Result<()> {
        self.history.append(&self.thread_id, message).await
    }

    /// Replaces the thread's history with `messages` (trimmed first).
    ///
    /// Clears the existing history then re-appends the trimmed list, so the
    /// stored state matches what [`load`](Self::load) would return.
    pub async fn save(&self, messages: Vec<Message>) -> Result<()> {
        let trimmed = self.apply_trim(messages);
        self.history.clear(&self.thread_id).await?;
        for message in trimmed {
            self.history.append(&self.thread_id, message).await?;
        }
        Ok(())
    }

    /// Clears the thread's history.
    pub async fn clear(&self) -> Result<()> {
        self.history.clear(&self.thread_id).await
    }

    /// Applies the trimming hook to `messages`, returning them unchanged when no
    /// hook is configured.
    fn apply_trim(&self, messages: Vec<Message>) -> Vec<Message> {
        match &self.trim {
            Some(trim) => trim(messages),
            None => messages,
        }
    }
}

#[cfg(test)]
mod test;
