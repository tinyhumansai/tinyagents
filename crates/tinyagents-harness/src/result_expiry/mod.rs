//! Expires old tool results by age, independent of the context window.
//!
//! See [`ToolResultExpiryPolicy`] and [`expire_tool_results`].

mod expire;
mod types;

pub use expire::expire_tool_results;
pub(crate) use expire::{call_for_result, effective_call_name};
pub use types::{
    DEFAULT_EXPIRY_MAX_AGE_TURNS, EXPIRY_NOTE_CHARS, ExpiryOverrides, ToolResultExpiryPolicy,
};
