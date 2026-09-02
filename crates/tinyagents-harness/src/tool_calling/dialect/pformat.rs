//! The positional dialect: `<tool_call>read_file[src/main.rs]</tool_call>`.
//!
//! Roughly an 80% token saving over the JSON form on the call side, and more
//! than that on the catalogue side, since a signature replaces a schema. See
//! [`crate::tool_calling::pformat`] for the grammar itself.
//!
//! The interesting property is that it degrades rather than fails: a body that
//! is not a well-formed positional call falls through to the JSON parser per
//! tag, so a model that mixes the two forms in one response — or ignores the
//! protocol entirely and emits JSON — is still understood.

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
            "Tool calls use **P-Format** (Parameter-Format): compact, positional, \
             pipe-delimited syntax wrapped in `<tool_call>` tags. ~80% cheaper on tokens \
             than JSON.\n\n",
        );
        instructions
            .push_str("```\n<tool_call>\nget_weather[London|metric]\n</tool_call>\n```\n\n");
        instructions.push_str(
            "**Rules:**\n\
             - Form: `name[arg1|arg2|...|argN]`. Arguments are positional and must match the \
               order shown in each tool's `Call as:` signature in the `## Tools` section above \
               (alphabetical by parameter name).\n\
             - Empty calls: `name[]` for zero-arg tools.\n\
             - Empty argument: `name[||value]` is three positional values, the first two empty.\n\
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

    fn format_results(&self, results: &[ToolOutcome]) -> Vec<TranscriptEntry> {
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
