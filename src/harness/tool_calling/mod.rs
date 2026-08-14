//! Recovering tool calls from model output.
//!
//! A model that supports native tool use hands back structured calls and none
//! of this is needed. Everything else — prompt-guided models, local models,
//! providers whose native mode is unavailable or disabled — emits tool calls as
//! *text*, in whatever shape the model was trained to produce. This module
//! turns that text back into calls.
//!
//! ## Why it is this forgiving
//!
//! Each accommodation here exists because a model actually produced it and the
//! alternative was dropping a well-formed call and burning an agent iteration.
//! Concretely, the parsers accept `<tool_call>` tags in several spellings,
//! fenced `tool_call` blocks, bare JSON objects, Anthropic-style
//! `<invoke name="…"><parameter name="…">` XML, and the compact positional
//! [`pformat`] syntax.
//!
//! The permissiveness is bounded on purpose, and the boundary is worth knowing
//! before widening anything:
//!
//! * **Argument keys are aliased; tool names are not.** A model drifting from
//!   `arguments` to `args`/`parameters`/`params`/`input` still yields a usable
//!   call. The *name* stays strict, because loosening it risks reading a plain
//!   JSON answer as a tool call in the whole-response path — turning an ordinary
//!   reply into a phantom invocation.
//! * **The generic `input` alias is only honoured behind an explicit marker**
//!   (a `tool_calls` array, a `<tool_call>` tag, a fenced block). Untagged text
//!   does not get it.
//! * **[`pformat`] refuses to invent argument names for an unknown tool**, so a
//!   model cannot tunnel arbitrary JSON through by guessing a tool name that
//!   does not exist.
//!
//! ## What the host still owns
//!
//! This module takes **schemas**, never a tool trait object. A host's tool type
//! is its own vocabulary, and depending on it here would defeat the point — so
//! [`pformat::build_registry`] takes `(name, schema)` pairs and the host keeps a
//! one-line adapter over its own tool slice. Dispatch and execution stay host-side
//! too: this module answers "what did the model ask for", never "what happens next".

pub(crate) mod parse;
pub(crate) mod pformat;

// The two entry points, plus the building blocks a host legitimately reaches
// for on its own. `extract_json_values` in particular is not a test helper:
// pulling the first JSON object out of model prose is how a host checks a
// required-output contract, which has nothing to do with tool calls.
pub use parse::{
    ParsedToolCall, extract_json_values, parse_arguments_value, parse_glm_style_tool_calls,
    parse_tool_call_value, parse_tool_calls, parse_tool_calls_from_json_value,
    parse_tool_calls_with_pformat,
};
pub use pformat::{
    PFormatParamType, PFormatRegistry, PFormatToolParams, build_registry, parse_call,
    render_signature, render_signature_from_schema,
};
