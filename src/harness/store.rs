//! Durable and ephemeral storage backends.
//!
//! Owns store traits and backend adapters for JSONL, local files, MongoDB,
//! in-memory tests, event journals, conversation records, tool artifacts, and
//! long-term application data. Conversation memory should use this module for
//! persistence instead of owning backend-specific code directly.
