//! The slot-indexed dialect: `<tool_call>read_file[0|src/main.rs]</tool_call>`.
//!
//! Roughly an 80% token saving over the JSON form on the call side, and more
//! than that on the catalogue side, since a signature replaces a schema. See
//! [`crate::tool_calling::pformat`] for the grammar itself.
//!
//! The interesting property is that it degrades rather than fails: a body that
//! is not a well-formed p-format call falls through to the JSON parser per tag,
//! so a model that mixes the two forms in one response — or ignores the protocol
//! entirely and emits JSON — is still understood. That fallback is also what
//! makes the parser's strictness affordable: a call with a miscounted or
//! non-numeric index is refused here and retried as JSON, rather than being
//! bound to whichever parameters it happens to line up with.

use std::sync::Arc;

use super::ToolDialect;
use super::text;
use super::types::{DialectMessage, DialectResponse, ToolCallFormat, ToolOutcome, TranscriptEntry};
use crate::tool::ToolSchema;
use crate::tool_calling::{PFormatRegistry, ParsedToolCall, parse_tool_calls_with_pformat};

/// Positional tool calling, driven by a registry of parameter layouts.
#[derive(Debug, Clone)]
pub struct PFormatDialect {
    /// Name → parameter layout, built once from the agent's real tools.
    ///
    /// This is the safety boundary the grammar depends on, not just a lookup:
    /// the parser refuses to invent argument names for a tool it does not know,
    /// so a model cannot tunnel arbitrary JSON through by guessing a name. A
    /// registry built from anything but the agent's own tools would widen that.
    registry: Arc<PFormatRegistry>,
}

impl PFormatDialect {
    /// Build the dialect over a prepared registry.
    pub fn new(registry: PFormatRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }

    /// Share an already-`Arc`'d registry rather than cloning the map.
    pub fn from_shared(registry: Arc<PFormatRegistry>) -> Self {
        Self { registry }
    }

    /// The registry backing this dialect.
    pub fn registry(&self) -> &PFormatRegistry {
        self.registry.as_ref()
    }

    /// The protocol block — **protocol only**, no catalogue.
    ///
    /// The signatures live in the prompt's tool section, rendered by
    /// [`super::catalogue::render_pformat_catalogue`] from the same schemas
    /// this dialect parses against. Repeating them here is the "tools listed
    /// twice" pattern the JSON dialect is stuck with, and it means adding a
    /// tool changes the prompt in one place instead of two.
    pub fn instructions() -> String {
        let mut instructions = String::new();
        instructions.push_str("## Tool Use Protocol\n\n");
        instructions.push_str(
            "Tool calls use **P-Format** (Parameter-Format): compact, slot-indexed, \
             pipe-delimited syntax wrapped in `<tool_call>` tags. ~80% cheaper on tokens \
             than JSON.\n\n",
        );
        instructions
            .push_str("```\n<tool_call>\nget_weather[0|London|1|metric]\n</tool_call>\n```\n\n");
        instructions.push_str(
            "**Rules:**\n\
             - Form: `name[index|value|index|value|...]`. Each value is preceded by the slot \
               number it fills, taken from that tool's `Call as:` signature in the `## Tools` \
               section above.\n\
             - **Send only the arguments you mean to send.** To pass just the third slot, \
               write `name[2|value]` — there are no empty slots to count.\n\
             - The signature shows each slot as `index|<name>`, e.g. \
               `search[0|<query>|1|<limit>]`. `<name>` is a placeholder: replace it with the \
               value, and do not send the name itself.\n\
             - Empty calls: `name[]` for zero-arg tools, or for a call sending no arguments.\n\
             - A call whose indices are missing, non-numeric, or not in the signature is \
               **rejected** — it will not run. Copy the numbers from the signature.\n\
             - Escapes inside argument values: `\\|` → `|`, `\\]` → `]`, `\\\\` → `\\`.\n\
             - You may emit multiple `<tool_call>` blocks in a single response. Each tag holds \
               exactly one call.\n\
             - After tool execution, results appear in `<tool_result>` tags. Continue reasoning \
               with the results until you can give a final answer.\n\
             - If you genuinely need a complex nested argument that p-format can't express, \
               you may fall back to the JSON form: \
               `<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>`. Prefer p-format \
               for everything else.\n\n",
        );
        instructions
    }
}

impl ToolDialect for PFormatDialect {
    fn parse_response(&self, response: &DialectResponse) -> (String, Vec<ParsedToolCall>) {
        let (text, calls) =
            parse_tool_calls_with_pformat(response.text_or_empty(), self.registry.as_ref());
        tinyagents_tracing::debug!(
            parse_mode = "pformat_combined",
            parsed_tool_calls = calls.len(),
            "pformat dialect parsed response"
        );
        (text, calls)
    }

    fn format_results(&self, results: &[ToolOutcome]) -> TranscriptEntry {
        text::format_results(results)
    }

    fn prompt_instructions(&self, _tools: &[ToolSchema]) -> String {
        Self::instructions()
    }

    fn to_provider_messages(&self, history: &[TranscriptEntry]) -> Vec<DialectMessage> {
        text::to_provider_messages(history)
    }

    fn should_send_tool_specs(&self) -> bool {
        // Text protocol: the model never sees a structured spec, only the
        // catalogue in the system prompt.
        false
    }

    fn tool_call_format(&self) -> ToolCallFormat {
        ToolCallFormat::PFormat
    }
}
