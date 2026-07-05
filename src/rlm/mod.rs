//! Recursive-language-model (RLM) runtime: a driver model writes code cells
//! executed in a sandboxed interpreter whose only host surface is capability
//! calls back into the [`CapabilityRegistry`](crate::registry::CapabilityRegistry)
//! — sub-LLM queries, tools, and sub-agent delegation. Scripts can therefore
//! *recursively* call language models, which is what turns a code sandbox
//! into an RLM.
//!
//! ## The three layers
//!
//! 1. **[`RlmInterpreter`]** — the pluggable execution API ("the interpreter
//!    exposed as an API"). Built-ins: the embedded Rhai engine (hermetic
//!    sandbox, the default) and external Python / JavaScript processes
//!    (binary + args are configuration; they speak a line-delimited JSON
//!    wire protocol, see [`interpreter::external`]).
//! 2. **[`RlmSession`]** — one interpreter bound to one [`RlmHost`], with
//!    every [`RlmPolicy`] limit enforced fail-closed per cell. Drive it
//!    directly when embedding your own loop.
//! 3. **[`RlmRunner`]** — the model-driven loop: render a template into a
//!    system prompt, let the driver model emit fenced code cells, execute,
//!    feed observations back, stop on `final_answer(...)`.
//!
//! Everything a run needs is describable as one serde document
//! ([`RlmConfig`]), so external harnesses can define RLM behaviors as
//! configuration rather than code.
//!
//! ```no_run
//! use std::sync::Arc;
//! use tinyagents::registry::CapabilityRegistry;
//! use tinyagents::rlm::{RlmConfig, RlmRunner};
//!
//! # async fn demo() -> tinyagents::Result<()> {
//! let mut registry: CapabilityRegistry<()> = CapabilityRegistry::new();
//! // registry.register_model("openai", Arc::new(model))?; …
//! let config = RlmConfig::from_json(
//!     r#"{ "interpreter": {"kind": "rhai"}, "template": "general" }"#,
//! )?;
//! let mut runner = RlmRunner::from_config(config, Arc::new(registry), Arc::new(()))?;
//! let outcome = runner.run("How many primes are there below 1000?").await?;
//! println!("{:?}", outcome.answer);
//! # Ok(()) }
//! ```

mod host;
pub mod interpreter;
mod runner;
mod session;
pub mod templates;
mod types;

#[cfg(test)]
mod test;

pub use host::{CapabilityListing, RlmHost, RlmHostApi, is_fatal};
pub use interpreter::{CellEval, RlmInterpreter, build_interpreter};
pub use runner::{RlmRunner, extract_code_cell};
pub use session::RlmSession;
pub use types::{
    CellOutcome, HostCall, InterpreterSpec, RlmCallKind, RlmCallRecord, RlmCancelFlag, RlmConfig,
    RlmOutcome, RlmPolicy, RlmStep, RlmStopReason, RlmTemplate, TemplateSpec,
};
