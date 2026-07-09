//! Prompt assembly.
//!
//! In the recursive harness a prompt is data that is built, not a hard-coded
//! string: templates and the [`PromptBuilder`] turn runtime values into the
//! [`ModelRequest`] that drives a model call, which is exactly what lets a model
//! (via the REPL or a self-authored blueprint) shape the prompt of a nested
//! sub-model call. Segment-aware assembly also keeps the cacheable prefix stable
//! across the repeated calls a deep recursion makes.
//!
//! Owns prompt templates, system message construction, dynamic context
//! injection, message rendering, prompt variables, and prompt validation.
//!
//! # Quick start
//!
//! ```rust
//! use tinyagents::harness::prompt::{PromptTemplate, PromptBuilder};
//! use serde_json::{json, Map};
//!
//! let mut vars = Map::new();
//! vars.insert("task".to_string(), json!("Summarise the text below."));
//!
//! let system_tpl = PromptTemplate::new("You are a helpful assistant. Task: {task}");
//! let system_msg = system_tpl.render_message(
//!     tinyagents::harness::prompt::TemplateRole::System,
//!     &vars,
//! ).unwrap();
//!
//! let mut builder = PromptBuilder::new();
//! builder.push_system("system", vec![system_msg]);
//! let request = builder.build(vec![]);
//! assert!(!request.cache_segments.is_empty());
//! ```

mod types;

pub use types::*;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::error::{Result, TinyAgentsError};
use crate::harness::message::Message;
use crate::harness::model::{ModelRequest, PromptSegment, ResponseFormat, SegmentRole};
use crate::harness::tool::ToolSchema;

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

impl PromptTemplate {
    /// Creates a new template from any string-like value.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Renders the template by substituting `{name}` placeholders with values
    /// from `vars`.
    ///
    /// `{{` is an escaped `{` and `}}` is an escaped `}`.
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Validation`] when:
    ///
    /// * a `{name}` references a key absent from `vars`, or
    /// * a placeholder is opened but never closed.
    pub fn render(&self, vars: &Map<String, Value>) -> Result<String> {
        render_template(&self.template, vars)
    }

    /// Renders the template and wraps the result in a [`Message`] of the given
    /// role.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`render`](Self::render).
    pub fn render_message(&self, role: TemplateRole, vars: &Map<String, Value>) -> Result<Message> {
        let text = self.render(vars)?;
        Ok(match role {
            TemplateRole::System => Message::system(text),
            TemplateRole::User => Message::user(text),
            TemplateRole::Assistant => Message::assistant(text),
        })
    }

    /// Convenience wrapper: renders as a [`TemplateRole::System`] message.
    pub fn render_system(&self, vars: &Map<String, Value>) -> Result<Message> {
        self.render_message(TemplateRole::System, vars)
    }

    /// Convenience wrapper: renders as a [`TemplateRole::User`] message.
    pub fn render_user(&self, vars: &Map<String, Value>) -> Result<Message> {
        self.render_message(TemplateRole::User, vars)
    }

    /// Convenience wrapper: renders as a [`TemplateRole::Assistant`] message.
    pub fn render_assistant(&self, vars: &Map<String, Value>) -> Result<Message> {
        self.render_message(TemplateRole::Assistant, vars)
    }
}

// ---------------------------------------------------------------------------
// MessagesTemplate
// ---------------------------------------------------------------------------

impl MessagesTemplate {
    /// Creates an empty template sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a (role, template) pair to the sequence.
    pub fn push(&mut self, role: TemplateRole, template: PromptTemplate) -> &mut Self {
        self.entries.push((role, template));
        self
    }

    /// Renders every entry in declaration order into a [`Vec<Message>`].
    ///
    /// # Errors
    ///
    /// Returns the first rendering error encountered.
    pub fn render(&self, vars: &Map<String, Value>) -> Result<Vec<Message>> {
        self.entries
            .iter()
            .map(|(role, tpl)| tpl.render_message(*role, vars))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------

impl PromptBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a cacheable **system** segment.
    ///
    /// The segment is labelled with `id` and receives
    /// [`SegmentRole::System`].
    pub fn push_system(&mut self, id: impl Into<String>, messages: Vec<Message>) -> &mut Self {
        self.segments.push(BuiltSegment {
            messages,
            meta: PromptSegment {
                id: id.into(),
                role: SegmentRole::System,
                cacheable: true,
            },
        });
        self
    }

    /// Appends a cacheable **tools** segment and accumulates `tools` into the
    /// request's tool list.
    ///
    /// The segment carries [`SegmentRole::Tools`] and has no messages of its
    /// own; the schemas are stored in [`PromptBuilder`]'s tool list and passed
    /// to [`ModelRequest::with_tools`] at build time.
    pub fn push_tools_segment(
        &mut self,
        id: impl Into<String>,
        tools: Vec<ToolSchema>,
    ) -> &mut Self {
        self.tools.extend(tools);
        self.segments.push(BuiltSegment {
            messages: vec![],
            meta: PromptSegment {
                id: id.into(),
                role: SegmentRole::Tools,
                cacheable: true,
            },
        });
        self
    }

    /// Appends a cacheable **instructions** segment.
    ///
    /// Use this for stable, per-deployment instruction messages that should
    /// participate in KV-cache reuse. The segment receives
    /// [`SegmentRole::Instructions`].
    pub fn push_instructions(
        &mut self,
        id: impl Into<String>,
        messages: Vec<Message>,
    ) -> &mut Self {
        self.segments.push(BuiltSegment {
            messages,
            meta: PromptSegment {
                id: id.into(),
                role: SegmentRole::Instructions,
                cacheable: true,
            },
        });
        self
    }

    /// Appends a **non-cacheable history** segment.
    ///
    /// Conversation history changes between turns and must not be part of the
    /// stable prefix.  The segment receives [`SegmentRole::History`].
    pub fn push_history(&mut self, id: impl Into<String>, messages: Vec<Message>) -> &mut Self {
        self.segments.push(BuiltSegment {
            messages,
            meta: PromptSegment {
                id: id.into(),
                role: SegmentRole::History,
                cacheable: false,
            },
        });
        self
    }

    /// Appends a **non-cacheable volatile** segment.
    ///
    /// Volatile content (e.g. retrieved context, the current user turn) must
    /// always appear at the tail and never enter the cacheable prefix.  The
    /// segment receives [`SegmentRole::Volatile`].
    pub fn push_volatile(&mut self, id: impl Into<String>, messages: Vec<Message>) -> &mut Self {
        self.segments.push(BuiltSegment {
            messages,
            meta: PromptSegment {
                id: id.into(),
                role: SegmentRole::Volatile,
                cacheable: false,
            },
        });
        self
    }

    /// Overrides the response format applied to the built request.
    pub fn with_response_format(&mut self, format: ResponseFormat) -> &mut Self {
        self.response_format = Some(format);
        self
    }

    /// Finalises the builder into a [`ModelRequest`].
    ///
    /// Messages from all segments are concatenated in push order; `tail` is
    /// appended last (use it for the current user turn).  Cache-segment
    /// metadata is propagated verbatim so middleware and providers can reason
    /// about the stable prefix.
    ///
    /// The fingerprint of the stable prefix is stored in
    /// [`ModelRequest::prompt_fingerprint`].
    pub fn build(&self, tail: Vec<Message>) -> ModelRequest {
        let mut messages: Vec<Message> = self
            .segments
            .iter()
            .flat_map(|s| s.messages.iter().cloned())
            .collect();
        messages.extend(tail);

        let cache_segments: Vec<PromptSegment> =
            self.segments.iter().map(|s| s.meta.clone()).collect();

        let fp = self.fingerprint();

        let mut req = ModelRequest::new(messages)
            .with_tools(self.tools.clone())
            .with_cache_segments(cache_segments);

        req.prompt_fingerprint = Some(fp);

        if let Some(fmt) = &self.response_format {
            req = req.with_response_format(fmt.clone());
        }

        req
    }

    /// Returns a 64-character lowercase hex SHA-256 fingerprint of the stable
    /// (cacheable) prefix.
    ///
    /// The fingerprint is derived from the canonical JSON serialization of
    /// every cacheable segment — its id, its [`SegmentRole`], and its full
    /// messages (roles, text, JSON blocks, image URLs/data, and provider
    /// extensions) — plus the complete [`ToolSchema`] list (names,
    /// descriptions, and parameter schemas). Any change to the stable prefix,
    /// including a tool-schema edit, an image swap, or a role change with
    /// identical text, produces a different fingerprint.
    ///
    /// Because the hash is SHA-256 over serde's deterministic (sorted-key)
    /// JSON, the value is stable across processes, Rust versions, and
    /// platforms — safe to persist and compare across restarts. It only
    /// changes when the serialized shape of [`Message`]/[`ToolSchema`]
    /// changes, which is a versioned public-API change.
    pub fn fingerprint(&self) -> String {
        let segments: Vec<Value> = self
            .segments
            .iter()
            .filter(|s| s.meta.cacheable)
            .map(|seg| {
                json!({
                    "id": seg.meta.id,
                    "role": seg.meta.role,
                    "messages": seg.messages,
                })
            })
            .collect();
        let payload = json!({
            "segments": segments,
            "tools": self.tools,
        });

        // Serialize the canonical JSON, then feed the bytes into the digest.
        // (`Sha256` no longer implements `std::io::Write`, so hash the
        // serialized bytes directly instead of streaming via `to_writer`.)
        let mut hasher = Sha256::new();
        if let Ok(bytes) = serde_json::to_vec(&payload) {
            hasher.update(&bytes);
        }
        // Serializing an already-materialized `Value` does not fail in
        // practice; on the unexpected error path the digest hashes empty
        // data for a stable result.
        let digest = hasher.finalize();
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

// ---------------------------------------------------------------------------
// Template rendering engine
// ---------------------------------------------------------------------------

/// Performs `{name}` placeholder substitution on `template`.
///
/// * `{{` → literal `{`
/// * `}}` → literal `}`
/// * `{name}` → `vars["name"]` (JSON string coerced; other types via Display)
///
/// Returns [`TinyAgentsError::Validation`] on an unknown or unclosed
/// placeholder.
fn render_template(template: &str, vars: &Map<String, Value>) -> Result<String> {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '{' => match chars.peek() {
                Some('{') => {
                    chars.next();
                    result.push('{');
                }
                _ => {
                    // Collect placeholder name up to '}'.
                    let mut name = String::new();
                    let mut closed = false;
                    for nc in chars.by_ref() {
                        if nc == '}' {
                            closed = true;
                            break;
                        }
                        name.push(nc);
                    }
                    if !closed {
                        return Err(TinyAgentsError::Validation(format!(
                            "unclosed placeholder '{{{name}'"
                        )));
                    }
                    match vars.get(&name) {
                        Some(Value::String(s)) => result.push_str(s),
                        Some(v) => result.push_str(&v.to_string()),
                        None => {
                            return Err(TinyAgentsError::Validation(format!(
                                "unknown placeholder '{{{name}}}'"
                            )));
                        }
                    }
                }
            },
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    result.push('}');
                } else {
                    result.push('}');
                }
            }
            _ => result.push(c),
        }
    }

    Ok(result)
}

#[cfg(test)]
mod test;
