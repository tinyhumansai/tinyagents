//! Optional builtin harness tools.
//!
//! These tools are generic implementations that depend only on TinyAgents'
//! [`Tool`][crate::harness::tool::Tool] interface. They are behind the `tools`
//! Cargo feature so applications that provide their own tool surface do not pull
//! in extra dependencies by default.

mod time;

pub use time::{CurrentTimeTool, ResolveTimeTool, register_time_tools, time_tools};

#[cfg(test)]
mod time_test;
