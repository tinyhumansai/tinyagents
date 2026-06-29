//! Run limits and policy enforcement.
//!
//! Owns max model calls, max tool calls, max concurrency, wall-clock deadlines,
//! per-call deadlines, recursion limits delegated from graph runs, and fail-closed
//! limit errors.
