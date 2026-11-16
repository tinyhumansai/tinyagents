//! Prompt assembly.
//!
//! In the recursive harness a prompt is data that is built, not a hard-coded
//! string: templates and the [`PromptBuilder`] turn runtime values into the
//! [`ModelRequest`] that drives a model call, which is exactly what lets a model
//! shape the prompt of a nested
//! sub-model call. Segment-aware assembly also keeps the cacheable prefix stable
//! across the repeated calls a deep recursion makes.
//!
//! Owns prompt templates, system message construction, dynamic context
//! injection, message rendering, prompt variables, and prompt validation.
//!
//! # Quick start
//!
//! ```rust
//! use tinyagents_harness::prompt::{PromptTemplate, PromptBuilder};
//! use serde_json::{json, Map};
//!
//! let mut vars = Map::new();
//! vars.insert("task".to_string(), json!("Summarise the text below."));
//!
//! let system_tpl = PromptTemplate::new("You are a helpful assistant. Task: {task}");
//! let system_msg = system_tpl.render_message(
//!     tinyagents_harness::prompt::TemplateRole::System,
//!     &vars,
//! ).unwrap();
//!
//! let mut builder = PromptBuilder::new();
//! builder.push_system("system", vec![system_msg]);
//! let request = builder.build(vec![]);
//! assert!(!request.cache_segments.is_empty());
//! ```

mod model_guidance;
mod types;
pub mod workspace_files;

pub use model_guidance::{
    EXECUTION_DISCIPLINE, NEEDS_EXECUTION_DISCIPLINE, execution_discipline_for,
    execution_discipline_for_profile, needs_execution_discipline,
};
pub use types::*;
pub use workspace_files::{
    builtin_hash, builtin_hash_sidecar_name, inject_inline_content, inject_workspace_file_capped,
    sync_workspace_file,
};

/// Marker that changes a dynamic prompt body's following text to the context tier.
pub const PROMPT_TIER_CONTEXT_MARKER: &str = "<!--prompt-tier:context-->";
/// Marker that changes a dynamic prompt body's following text to the volatile tier.
pub const PROMPT_TIER_VOLATILE_MARKER: &str = "<!--prompt-tier:volatile-->";

/// Split a dynamic prompt body into typed cache-tier parts.
///
/// Marker lines are recognized after trimming for comparison and are omitted
/// from the returned content. Markers may repeat or appear in either order;
/// each changes the tier of following text. All other line bytes are preserved
/// exactly, except whitespace-only slices between markers, which are
/// discarded. An unmarked body stays in `default_tier`.
#[must_use]
pub fn split_prompt_tiers(body: &str, default_tier: PromptTier) -> Vec<TieredPromptPart> {
    let mut parts = Vec::new();
    let mut tier = default_tier;
    let mut current = String::new();
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim();
        let next = if trimmed == PROMPT_TIER_CONTEXT_MARKER {
            Some(PromptTier::Context)
        } else if trimmed == PROMPT_TIER_VOLATILE_MARKER {
            Some(PromptTier::Volatile)
        } else {
            None
        };
        match next {
            Some(next_tier) => {
                if !current.trim().is_empty() {
                    parts.push(TieredPromptPart::new(tier, std::mem::take(&mut current)));
                } else {
                    current.clear();
                }
                tier = next_tier;
            }
            None => current.push_str(line),
        }
    }
    if !current.trim().is_empty() {
        parts.push(TieredPromptPart::new(tier, current));
    }
    parts
}

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::error::{Result, TinyAgentsError};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelRequest, PromptSegment, ResponseFormat, SegmentRole};
use tinyinference_llm::tool::ToolSchema;

/// Collects context-dependent prompt sections and lays them out by cache tier.
pub struct PromptComposer<C> {
    sections: Vec<Box<dyn DynamicPromptSection<C>>>,
}

impl<C> Default for PromptComposer<C> {
    fn default() -> Self {
        Self {
            sections: Vec::new(),
        }
    }
}

impl<C> PromptComposer<C> {
    /// Creates an empty prompt composer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a section. Sections render once per call to [`Self::build`].
    pub fn push(&mut self, section: impl DynamicPromptSection<C> + 'static) -> &mut Self {
        self.sections.push(Box::new(section));
        self
    }

    /// Renders all sections, then emits stable, context, and volatile slices
    /// in that order while preserving declaration order within each tier.
    pub fn build(&self, context: &C) -> Result<TieredPrompt> {
        let mut buckets: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for section in &self.sections {
            for part in section.render(context)? {
                if !part.content.trim().is_empty() {
                    buckets[tier_index(part.tier)].push(part.content.trim_end().to_string());
                }
            }
        }
        let mut text = String::new();
        let mut breakpoints = Vec::new();
        let mut parts = Vec::new();
        for tier in [
            PromptTier::Stable,
            PromptTier::Context,
            PromptTier::Volatile,
        ] {
            let bucket = &buckets[tier_index(tier)];
            if bucket.is_empty() {
                continue;
            }
            let rendered = format!("{}\n\n", bucket.join("\n\n"));
            text.push_str(&rendered);
            parts.push((tier, rendered));
            if tier != PromptTier::Volatile {
                breakpoints.push(text.len());
            }
        }
        if breakpoints.last() == Some(&text.len()) {
            breakpoints.pop();
        }
        if !text.is_empty() {
            text = format!("{}\n", text.trim_end());
        }
        Ok(TieredPrompt {
            text,
            breakpoints,
            parts,
        })
    }
}

fn tier_index(tier: PromptTier) -> usize {
    match tier {
        PromptTier::Stable => 0,
        PromptTier::Context => 1,
        PromptTier::Volatile => 2,
    }
}

/// Renders supplied sections in order without product prompt selection.
///
/// `max_bytes` is a caller-owned budget. The first section that does not fit
/// is clipped at a UTF-8 boundary and recorded; later sections are omitted.
pub fn assemble_sections(sections: &[PromptSection], max_bytes: usize) -> PromptAssembly {
    let mut output = PromptAssembly::default();
    for section in sections {
        let separator = if output.text.is_empty() { "" } else { "\n\n" };
        let available = max_bytes.saturating_sub(output.text.len());
        let required = separator.len() + section.content.len();
        if required <= available {
            output.text.push_str(separator);
            output.text.push_str(&section.content);
            output.included_sections.push(section.name.clone());
            continue;
        }
        if available > separator.len() {
            output.text.push_str(separator);
            let content_cap = available - separator.len();
            let cut = utf8_prefix(&section.content, content_cap);
            output.text.push_str(&section.content[..cut]);
            output.truncation = Some(PromptTruncation {
                section: section.name.clone(),
                omitted_bytes: section.content.len().saturating_sub(cut),
            });
        } else {
            output.truncation = Some(PromptTruncation {
                section: section.name.clone(),
                omitted_bytes: section.content.len(),
            });
        }
        break;
    }
    output
}

/// Assembles sections under both caller-supplied byte and token budgets.
pub fn assemble_sections_with_budget(
    sections: &[PromptSection],
    budget: PromptBudget,
    tokenize: impl Fn(&str) -> usize,
) -> PromptAssembly {
    let mut assembled = PromptAssembly::default();
    for section in sections {
        // Tokenize the exact text the caller receives. In particular, the
        // separator belongs to the candidate rather than being charged after
        // the section's content, because tokenizers may assign it a cost.
        let mut candidate = assembled.text.clone();
        if !candidate.is_empty() {
            candidate.push_str("\n\n");
        }
        candidate.push_str(&section.content);
        if candidate.len() > budget.max_bytes || tokenize(&candidate) > budget.max_tokens {
            assembled.truncation = Some(PromptTruncation {
                section: section.name.clone(),
                omitted_bytes: section.content.len(),
            });
            break;
        }
        assembled.text = candidate;
        assembled.included_sections.push(section.name.clone());
    }
    assembled
}

/// Renders a stable Markdown heading.
/// Segment id of the first leading system message (see
/// [`PromptBuilder::push_system_messages`]).
pub const SYSTEM_SEGMENT_ID: &str = "system";

/// Segment id for the `index`-th leading system message: `"system"` for the
/// first, `"system.1"`, `"system.2"`, ... for the following ones.
#[must_use]
pub fn system_segment_id(index: usize) -> String {
    if index == 0 {
        SYSTEM_SEGMENT_ID.to_string()
    } else {
        format!("{SYSTEM_SEGMENT_ID}.{index}")
    }
}

/// Whether `id` is one the harness assigns to a leading system message.
#[must_use]
pub fn is_system_segment_id(id: &str) -> bool {
    id == SYSTEM_SEGMENT_ID
        || id
            .strip_prefix(SYSTEM_SEGMENT_ID)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|suffix| {
                suffix
                    .parse::<usize>()
                    .is_ok_and(|index| index > 0 && suffix == index.to_string())
            })
}

pub fn render_heading(title: &str) -> String {
    format!("## {title}")
}

/// Renders an optional named section; blank bodies produce no bytes.
pub fn render_optional_section(title: &str, body: Option<&str>) -> String {
    body.filter(|body| !body.trim().is_empty())
        .map(|body| format!("{}\n\n{}", render_heading(title), body))
        .unwrap_or_default()
}

/// Renders a stable, name-sorted catalogue from caller-owned rows.
pub fn render_tool_catalogue(rows: &[(String, String)]) -> String {
    let mut rows = rows.to_vec();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows.into_iter()
        .map(|(name, description)| format!("- `{name}`: {description}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn utf8_prefix(text: &str, max_bytes: usize) -> usize {
    if text.len() <= max_bytes {
        return text.len();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}

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
    /// [`SegmentRole::System`]. Canonical `system` / `system.N` ids mean exactly
    /// one message each. A grouped segment requested under one of those ids
    /// is labelled as custom so cache guards use the conservative full-request
    /// path rather than mistaking its segment count for a message boundary.
    pub fn push_system(&mut self, id: impl Into<String>, messages: Vec<Message>) -> &mut Self {
        let mut id = id.into();
        if messages.len() != 1 && is_system_segment_id(&id) {
            id.push_str(":grouped");
        }
        self.segments.push(BuiltSegment {
            messages,
            meta: PromptSegment {
                id,
                role: SegmentRole::System,
                cacheable: true,
            },
        });
        self
    }

    /// Appends a composed system prompt with explicit stable/context/volatile
    /// segment markers. Stable and context segments are cacheable; volatile
    /// content is placed last and excluded from the reusable prefix.
    pub fn push_tiered_system_prompt(&mut self, prompt: &TieredPrompt) -> &mut Self {
        for (tier, content) in &prompt.parts {
            let marker = match tier {
                PromptTier::Stable => "stable",
                PromptTier::Context => "context",
                PromptTier::Volatile => "volatile",
            };
            self.segments.push(BuiltSegment {
                messages: vec![Message::system(content.trim_end())],
                meta: PromptSegment {
                    id: format!("prompt-tier:{marker}"),
                    role: SegmentRole::System,
                    cacheable: *tier != PromptTier::Volatile,
                },
            });
        }
        self
    }

    /// Appends the leading system messages of a transcript as **one cacheable
    /// segment per message**.
    ///
    /// A host that renders its system prompt in tiers (identity and rules
    /// that never change, then per-deployment context, then per-session
    /// material such as connected services) sends them as consecutive
    /// `Message::System` values. Keeping each one its own segment lets the
    /// cache layout guard tell a rewritten stable tier from a volatile one
    /// and lets provider adapters place a breakpoint at the tier boundary
    /// instead of only at the end of the whole prompt. Segment ids follow
    /// [`SYSTEM_SEGMENT_ID`] for the first message and
    /// `"{SYSTEM_SEGMENT_ID}.{n}"` for the rest, which
    /// [`is_system_segment_id`] recognises.
    pub fn push_system_messages(&mut self, system_messages: &[Message]) -> &mut Self {
        let mut index = self
            .segments
            .iter()
            .filter(|segment| {
                segment.meta.role == SegmentRole::System && is_system_segment_id(&segment.meta.id)
            })
            .count();
        for message in system_messages {
            while self
                .segments
                .iter()
                .any(|segment| segment.meta.id == system_segment_id(index))
            {
                index += 1;
            }
            self.push_system(system_segment_id(index), vec![message.clone()]);
            index += 1;
        }
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

        // Recursively sort object keys independently of serde_json features;
        // array order and the default fingerprint bytes stay unchanged.
        // Serialize the canonical JSON, then feed it to the digest via
        // `Digest::update` (digest 0.11 dropped the `io::Write` impl the previous
        // `to_writer` path relied on). Serializing an already-materialized
        // `Value` does not fail in practice; on the impossible error we hash empty
        // data for a stable result.
        let mut hasher = Sha256::new();
        if let Ok(bytes) = crate::canonical_json::to_vec(&payload) {
            hasher.update(&bytes);
        }
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
#[path = "mod_tests.rs"]
mod test;
