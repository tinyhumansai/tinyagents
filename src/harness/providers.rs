//! Feature-gated model provider integrations.
//!
//! Owns optional adapters for hosted and local models. Provider modules should
//! translate between RustAgents' provider-neutral request types and
//! provider-specific APIs without leaking provider shape into core harness code.
