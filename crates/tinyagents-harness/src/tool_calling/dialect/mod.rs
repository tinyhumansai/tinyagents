//! End-to-end tool calling for a host that drives its own model loop.
//!
//! [`super`] answers *"what did the model ask for"*. This module answers
//! everything around it: how the model is **told** to ask, how results are
//! **rendered** back, and how a transcript is **replayed** onto the provider
//! wire so the next iteration is well-formed. Together those are one thing —
//! a **dialect** — and they have to move together, because a catalogue that
//! advertises one grammar and a parser that expects another is a silent
//! whole-turn failure with no error anywhere.
//!
//! Three dialects ship: [`XmlDialect`] (JSON in a tag), [`PFormatDialect`]
//! (compact positional), and [`NativeDialect`] (the provider's structured
//! channel).
//!
//! # Which surface to use
//!
//! There are two, and picking the wrong one costs a rewrite:
//!
//! * **Driving the crate's [`agent_loop`](crate::agent_loop)** —
//!   use [`crate::tool::prompt`]. It speaks
//!   [`Message`](tinyinference::message::Message) and slots straight into the
//!   loop, which owns the iteration for you.
//! * **Driving your own loop over your own durable transcript** — use this
//!   module. It speaks [`TranscriptEntry`], a deliberately thin record shape a
//!   host maps onto in a few `From` impls, and it does not assume anything
//!   about how or when you call the model.
//!
//! The second case is not a lesser one. A host with years of persisted
//! transcripts, per-turn security policy, and provider quirks encoded in its
//! own storage cannot adopt a foreign message model just to stop maintaining a
//! tool-call parser — and the parser is the part that is genuinely universal.
//! This module is the seam that lets it hand over the universal part alone.
//!
//! # What stays with the host
//!
//! Executing a tool: permission checks, sandboxing, approval gates, timeouts,
//! progress events. A dialect decides what the model *reads and writes*; it
//! never decides what is allowed to *happen*. That boundary is what keeps a
//! host's security policy in the host, where it can be audited.

mod catalogue;
mod native;
mod pairing;
mod pformat;
mod text;
mod types;
mod xml;

pub use catalogue::{CATALOGUE_HEADING, render_json_catalogue, render_pformat_catalogue};
pub use native::NativeDialect;
pub use pairing::pair_tool_cycles;
pub use pformat::PFormatDialect;
pub use text::TOOL_RESULTS_PREFIX;
pub use types::{
    DialectMessage, DialectResponse, DialectRole, NativeToolCall, ToolCallFormat, ToolOutcome,
    ToolResultEntry, TranscriptEntry,
};
pub use xml::XmlDialect;

use crate::tool::ToolSchema;
use crate::tool_calling::ParsedToolCall;

/// One complete way of speaking tools to a model.
///
/// Every method is a face of the same contract, which is why they are one
/// trait: the catalogue the model reads, the calls it writes, the results it
/// gets back, and the history it is re-shown all have to describe the same
/// protocol. Splitting them into separate configurable pieces is how they drift.
pub trait ToolDialect: Send + Sync {
    /// Split a model response into narrative text and the calls it requested.
    fn parse_response(&self, response: &DialectResponse) -> (String, Vec<ParsedToolCall>);

    /// Render executed outcomes into the transcript records that follow the
    /// assistant turn.
    ///
    /// Usually one record. It is a `Vec` because a
    /// [`trusted_verbatim`](ToolOutcome::trusted_verbatim) outcome must reach
    /// the model at byte 0 of its own message, which a batch cannot provide: a
    /// text dialect closes the batch before such an outcome and reopens it
    /// after. A dialect that never reshapes output — the native one — always
    /// returns exactly one record.
    fn format_results(&self, results: &[ToolOutcome]) -> Vec<TranscriptEntry>;

    /// The protocol block for the system prompt.
    ///
    /// Whether `tools` is used depends on the dialect: only a dialect that
    /// [embeds its own catalogue](Self::embeds_tool_catalogue) reads it. The
    /// others get their catalogue from the prompt's tool section, or need none
    /// at all.
    fn prompt_instructions(&self, tools: &[ToolSchema]) -> String;

    /// Replay a transcript as flat provider messages.
    fn to_provider_messages(&self, history: &[TranscriptEntry]) -> Vec<DialectMessage>;

    /// Whether structured tool specs belong in the API request.
    ///
    /// `false` for the text dialects: sending specs a dialect cannot read back
    /// invites the model onto a channel nothing is listening on.
    fn should_send_tool_specs(&self) -> bool;

    /// Whether [`Self::prompt_instructions`] already lists the tools.
    ///
    /// A host uses this to avoid rendering the catalogue twice. Defaults to
    /// `false` — the common case, where the prompt's tool section owns it.
    fn embeds_tool_catalogue(&self) -> bool {
        false
    }

    /// How the tool catalogue should spell each entry, so the prompt and this
    /// dialect's parser cannot disagree.
    fn tool_call_format(&self) -> ToolCallFormat;
}

#[cfg(test)]
#[path = "test.rs"]
mod test;
