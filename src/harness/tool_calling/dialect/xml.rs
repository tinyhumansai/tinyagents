//! The JSON-in-tag dialect: `<tool_call>{"name":…,"arguments":{…}}</tool_call>`.
//!
//! The baseline for any model that was never trained to call tools but can
//! follow an instruction. It costs the most tokens of the three — every call
//! spells out its argument names, and the catalogue carries full schemas — and
//! it is the one that works everywhere, which is why it stays the fallback
//! rather than being retired.

use super::catalogue::render_json_catalogue;
use super::text;
use super::types::{
    DialectMessage, DialectResponse, ToolCallFormat, ToolOutcome, TranscriptEntry,
};
use super::ToolDialect;
use crate::harness::tool::ToolSchema;
use crate::harness::tool_calling::{parse_tool_calls, ParsedToolCall};

/// JSON-in-tag tool calling.
#[derive(Debug, Default, Clone, Copy)]
pub struct XmlDialect;

impl XmlDialect {
    /// Recover tool calls from raw model text.
    ///
    /// Shared with the other two dialects: p-format falls back to it per tag,
    /// and the native dialect uses it to recover calls a model narrated as text
    /// despite having a structured channel available.
    pub fn parse_text(text: &str) -> (String, Vec<ParsedToolCall>) {
        parse_tool_calls(text)
    }

    /// The protocol block plus the full-schema catalogue.
    ///
    /// This dialect embeds its own catalogue rather than leaving it to the
    /// prompt's tool section, because the schemas it needs are the protocol:
    /// a model writing `{"arguments": {…}}` by hand has to know the argument
    /// names, and there is nowhere else in the prompt that tells it.
    pub fn instructions(tools: &[ToolSchema]) -> String {
        let mut instructions = String::new();
        instructions.push_str("## Tool Use Protocol\n\n");
        instructions
            .push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
        instructions.push_str(
            "```\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n```\n\n",
        );
        instructions.push_str("### Available Tools\n\n");
        instructions.push_str(&render_json_catalogue(tools));
        instructions
    }
}

impl ToolDialect for XmlDialect {
    fn parse_response(&self, response: &DialectResponse) -> (String, Vec<ParsedToolCall>) {
        let (text, calls) = Self::parse_text(response.text_or_empty());
        tracing::debug!(
            parse_mode = "text_fallback",
            parsed_tool_calls = calls.len(),
            "xml dialect parsed response"
        );
        (text, calls)
    }

    fn format_results(&self, results: &[ToolOutcome]) -> TranscriptEntry {
        text::format_results(results)
    }

    fn prompt_instructions(&self, tools: &[ToolSchema]) -> String {
        Self::instructions(tools)
    }

    fn to_provider_messages(&self, history: &[TranscriptEntry]) -> Vec<DialectMessage> {
        text::to_provider_messages(history)
    }

    fn should_send_tool_specs(&self) -> bool {
        // The schemas are already in the prompt; sending them again as native
        // specs would double the cost and invite the model to use a channel
        // this dialect cannot read.
        false
    }

    fn embeds_tool_catalogue(&self) -> bool {
        true
    }

    fn tool_call_format(&self) -> ToolCallFormat {
        ToolCallFormat::Json
    }
}
