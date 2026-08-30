//! Rendering a tool catalogue into a system prompt.
//!
//! Two shapes, because there are two things a model can be told:
//!
//! * [`render_pformat_catalogue`] gives it a **call signature** —
//!   `read_file[path]` — which is all a positional dialect needs and costs a
//!   handful of tokens per tool.
//! * [`render_json_catalogue`] gives it the **whole parameter schema**, which
//!   is what a JSON-in-tag dialect needs because the model has to name each
//!   argument itself.
//!
//! Neither is the "informational" case: when the provider receives real tool
//! specs in the request, repeating them in the prompt is pure token bloat, so
//! the native dialect renders no catalogue at all.

use std::fmt::Write as _;

use crate::tool::ToolSchema;
use crate::tool_calling::pformat::render_signature_from_schema;

/// Heading the catalogue is rendered under.
pub const CATALOGUE_HEADING: &str = "## Tools\n\n";

/// Render the compact positional catalogue: one line per tool carrying its
/// description and its p-format call signature.
///
/// The signature comes straight from the parameter schema, through the same
/// [`render_signature_from_schema`] the parser reconstructs arguments with — so
/// the order the model reads is by construction the order the parser expects.
/// Rendering it any other way is how a catalogue and a parser drift apart.
pub fn render_pformat_catalogue(tools: &[ToolSchema]) -> String {
    let mut out = String::from(CATALOGUE_HEADING);
    for tool in tools {
        let signature = render_signature_from_schema(&tool.name, &tool.parameters);
        let _ = writeln!(
            out,
            "- **{}**: {}\n  Call as: `{}`",
            tool.name, tool.description, signature
        );
    }
    out
}

/// Render the full-schema catalogue: one line per tool carrying its description
/// and its complete JSON-Schema parameter object.
///
/// Used by the JSON-in-tag dialect, where the model writes argument names out
/// itself and therefore has to see them.
pub fn render_json_catalogue(tools: &[ToolSchema]) -> String {
    let mut out = String::new();
    for tool in tools {
        let _ = writeln!(
            out,
            "- **{}**: {}\n  Parameters: `{}`",
            tool.name, tool.description, tool.parameters
        );
    }
    out
}
