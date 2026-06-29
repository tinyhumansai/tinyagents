//! Public types for the prompt assembly module.
//!
//! All user-visible structs and enums live here so that [`super`] can provide
//! clean implementations without mixing type definitions and method bodies.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::harness::message::Message;
use crate::harness::model::{ModelRequest, PromptSegment, ResponseFormat};
use crate::harness::tool::ToolSchema;

/// The role a rendered message will take in the conversation.
///
/// Used by [`PromptTemplate::render_message`] and [`MessagesTemplate`] to
/// determine which [`Message`] variant to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateRole {
    /// System / developer instructions.
    System,
    /// User / human turn.
    User,
    /// Assistant / model turn.
    Assistant,
}

/// A string template that supports `{placeholder}` substitution.
///
/// # Syntax
///
/// * `{name}` – replaced with the string value of `name` from the variable map.
/// * `{{` – literal `{`.
/// * `}}` – literal `}`.
///
/// Rendering fails with [`crate::error::RustAgentsError::Validation`] if a
/// `{name}` placeholder is present in the template but `name` is not found in
/// the provided variable map, or if a placeholder is left unclosed.
///
/// # Examples
///
/// ```rust
/// use rustagents::harness::prompt::PromptTemplate;
/// use serde_json::{json, Map};
///
/// let t = PromptTemplate::new("Hello, {name}!");
/// let mut vars = Map::new();
/// vars.insert("name".to_string(), json!("world"));
/// assert_eq!(t.render(&vars).unwrap(), "Hello, world!");
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// The raw template string.
    pub template: String,
}

/// An ordered sequence of (role, template) pairs that renders to a
/// [`Vec<Message>`].
///
/// Each entry produces one [`Message`] when [`MessagesTemplate::render`] is
/// called with a variable map.  The entries are rendered in declaration order.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MessagesTemplate {
    /// Ordered (role, template) entries.
    pub entries: Vec<(TemplateRole, PromptTemplate)>,
}

/// Assembles a [`ModelRequest`] while tracking prompt-cache segments.
///
/// Callers push segments in logical order — system, tools, instructions (all
/// cacheable), then history and volatile context (not cacheable).  The stable
/// prefix is kept at the head so providers can apply KV-cache reuse.
///
/// Call [`PromptBuilder::build`] to finalize the request, optionally appending
/// extra tail messages (e.g. the current user turn).
///
/// # Cache-segment ordering
///
/// | Push method              | [`SegmentRole`]  | cacheable |
/// |--------------------------|------------------|-----------|
/// | [`push_system`]          | `System`         | `true`    |
/// | [`push_tools_segment`]   | `Tools`          | `true`    |
/// | [`push_instructions`]    | `Instructions`   | `true`    |
/// | [`push_history`]         | `History`        | `false`   |
/// | [`push_volatile`]        | `Volatile`       | `false`   |
///
/// [`push_system`]: PromptBuilder::push_system
/// [`push_tools_segment`]: PromptBuilder::push_tools_segment
/// [`push_instructions`]: PromptBuilder::push_instructions
/// [`push_history`]: PromptBuilder::push_history
/// [`push_volatile`]: PromptBuilder::push_volatile
/// [`SegmentRole`]: crate::harness::model::SegmentRole
#[derive(Clone, Debug, Default)]
pub struct PromptBuilder {
    /// Accumulated segments, in push order.
    pub(crate) segments: Vec<BuiltSegment>,
    /// Tool declarations gathered from [`push_tools_segment`] calls.
    ///
    /// [`push_tools_segment`]: PromptBuilder::push_tools_segment
    pub(crate) tools: Vec<ToolSchema>,
    /// Optional response-format override applied in [`build`].
    ///
    /// [`build`]: PromptBuilder::build
    pub(crate) response_format: Option<ResponseFormat>,
}

// ---------------------------------------------------------------------------
// Private internals
// ---------------------------------------------------------------------------

/// A single assembled segment held by [`PromptBuilder`].
#[derive(Clone, Debug)]
pub(crate) struct BuiltSegment {
    /// The messages that belong to this segment (empty for a tools segment).
    pub(crate) messages: Vec<Message>,
    /// Segment metadata propagated to [`ModelRequest::cache_segments`].
    pub(crate) meta: PromptSegment,
}
