//! Middleware stack.
//!
//! Owns before/after/wrap hooks around agent, model, and tool execution.
//! Cross-cutting behavior such as tracing, guardrails, dynamic prompts, tool
//! filtering, rate limiting, and summarization should live here.
