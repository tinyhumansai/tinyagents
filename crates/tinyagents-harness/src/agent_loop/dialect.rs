//! Host-side selection and application of the tool dialect for one run.
//!
//! The protocol itself — how a call is rendered, parsed, repaired, and
//! scrubbed from a stream — is owned by `tinytools-agent`. What is decided
//! *here* is the host's part: which dialect a run speaks
//! ([`RunPolicy::tool_dialect`]), the rewrite of a request onto a text
//! protocol when one is forced, the minting of call ids for calls recovered
//! from text, and the fallback that reads a native model's narrated call out
//! of its visible text.
//!
//! [`RunPolicy::tool_dialect`]: crate::runtime::RunPolicy::tool_dialect

use std::sync::Arc;

use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::{
    ModelRequest, ModelResponse, PromptSegment, SegmentRole, ToolChoice,
};
use tinyinference_llm::tool::{ToolCall, ToolSchema};
use tinytools_agent::dialect::{CodeDialect, CodeStyle, PFormatDialect};
use tinytools_agent::types::{ParseOptions, ParsedToolCall};
use tinytools_agent::{PFormatRegistry, StreamScrubber};

use crate::config::ToolDispatcher;
use crate::ids::CallId;

#[cfg(test)]
mod test;

/// The dialect a run speaks, resolved once from policy.
#[derive(Debug, Clone)]
pub(super) enum RunDialect {
    /// Schemas on the wire; the provider adapter owns any text fallback.
    Native,
    /// JSON-in-tag, rendered into the system prompt by the host.
    Xml,
    /// Positional P-Format, rendered into the system prompt by the host.
    PFormat(Arc<PFormatRegistry>),
    /// Code-style calls against Python or TypeScript signatures, rendered
    /// into the system prompt by the host. Shares P-Format's registry: both
    /// bind positional arguments against the same layout.
    Code(CodeStyle, Arc<PFormatRegistry>),
}

impl RunDialect {
    /// Resolves the policy against the tools this run offers.
    pub(super) fn resolve(
        dispatcher: ToolDispatcher,
        tools: &[ToolSchema],
        native_tool_calling: Option<bool>,
    ) -> Self {
        match dispatcher {
            ToolDispatcher::Auto if native_tool_calling == Some(false) => Self::Xml,
            ToolDispatcher::Auto | ToolDispatcher::Native => Self::Native,
            ToolDispatcher::Xml => Self::Xml,
            ToolDispatcher::Pformat => Self::PFormat(Arc::new(registry_from(tools))),
            ToolDispatcher::Python => Self::Code(CodeStyle::Python, Arc::new(registry_from(tools))),
            ToolDispatcher::Typescript => {
                Self::Code(CodeStyle::TypeScript, Arc::new(registry_from(tools)))
            }
        }
    }

    /// Whether the host renders the protocol and parses the answer itself.
    pub(super) fn is_text(&self) -> bool {
        !matches!(self, Self::Native)
    }

    /// The P-Format registry for one call, extended with any tool in `tools`
    /// beyond the run-level set the registry was built from.
    ///
    /// The run-level registry is built once from the schemas offered at the
    /// start of the run (see [`Self::resolve`]); a synthetic per-turn tool —
    /// the structured-output fallback schema pushed onto `request.tools`
    /// after that — is advertised in the P-Format catalogue (rendered fresh
    /// from the final tool list on every call) but would otherwise have no
    /// positional layout to decode a call against. Extending here, rather
    /// than rebuilding from scratch every call, keeps the common case (no
    /// new tool this turn) a cheap `Arc::clone`.
    pub(super) fn registry_for(&self, tools: &[ToolSchema]) -> Option<Arc<PFormatRegistry>> {
        match self {
            Self::PFormat(registry) | Self::Code(_, registry) => {
                let extra: Vec<&ToolSchema> = tools
                    .iter()
                    .filter(|schema| !registry.contains_key(&schema.name))
                    .collect();
                if extra.is_empty() {
                    return Some(Arc::clone(registry));
                }
                let mut merged = (**registry).clone();
                merged.extend(tinytools_agent::build_registry(
                    extra
                        .into_iter()
                        .map(|schema| (schema.name.clone(), schema.parameters.clone())),
                ));
                Some(Arc::new(merged))
            }
            _ => None,
        }
    }

    /// Rewrites `request` onto this dialect's text protocol: the transcript
    /// is folded into forms a prompt-guided model can read, the protocol block
    /// and catalogue go into the system prompt, and no schema goes on the
    /// wire. A no-op for [`Self::Native`] or when no tools are offered.
    ///
    /// With `host_renders_catalogue` the schemas still leave the wire (the
    /// registry built from them before this call is what parses the answer),
    /// and nothing from the run's ordinary catalogue is appended: the host's
    /// own prompt already carries the protocol block and the catalogue for
    /// this dialect. `synthesized` is the exception — tool schemas minted
    /// *this turn*, after the host's static prompt was already composed (the
    /// structured-output fallback tool `StructuredStrategy::ToolCall` /
    /// `ToolCallUnion` push onto `request.tools`). The host cannot have
    /// rendered a schema it did not know about yet, so their catalogue
    /// entries are appended here even in the host-rendered case, or the
    /// model never learns the shape it is being forced to call.
    pub(super) fn apply_to_request(
        &self,
        request: &mut ModelRequest,
        host_renders_catalogue: bool,
        synthesized: &[ToolSchema],
    ) {
        if !self.is_text() || request.tools.is_empty() || request.tool_choice == ToolChoice::None {
            return;
        }
        use tinyinference_llm::prompt_tools;

        let tools = std::mem::take(&mut request.tools);
        let messages = prompt_tools::coalesce_tool_results(&request.messages);
        let messages = prompt_tools::ensure_resolvable_user_turn(&messages);
        let messages = prompt_tools::anchor_user_request_after_tool_result(&messages);
        let had_leading_system = matches!(messages.first(), Some(Message::System(_)));
        if host_renders_catalogue {
            let mut block = String::new();
            if !synthesized.is_empty() {
                block.push_str(&self.render_catalogue(synthesized));
            }
            // Only a forced choice still has to be said, since the host's
            // prompt was composed before the choice was known.
            match &request.tool_choice {
                ToolChoice::Required => block.push_str("You must emit at least one tool call.\n"),
                ToolChoice::Tool(name) => {
                    block.push_str(&format!("You must call the `{name}` tool.\n"));
                }
                ToolChoice::Auto | ToolChoice::None => {}
            }
            request.messages = if block.is_empty() {
                // Nothing to say: no synthesized tool to advertise and no
                // forced choice, so this rewrite leaves `messages` — and in
                // particular whether a leading system message exists —
                // exactly as it already was.
                messages
            } else {
                prompt_tools::append_system_block(&messages, &block)
            };
            request.tool_choice = ToolChoice::Auto;
            sync_stripped_tools_cache_segment(request, had_leading_system);
            return;
        }
        request.messages = match self {
            Self::Xml | Self::Native => {
                prompt_tools::with_tool_instructions(&messages, &tools, &request.tool_choice)
            }
            Self::PFormat(_) | Self::Code(..) => {
                let specs: Vec<tinytools_agent::tinytools::ToolSpec> = tools
                    .iter()
                    .map(|schema| tinytools_agent::tinytools::ToolSpec {
                        name: schema.name.clone(),
                        description: schema.description.clone(),
                        parameters: schema.parameters.clone(),
                    })
                    .collect();
                let mut block = match self {
                    Self::Code(style, _) => {
                        let mut block = CodeDialect::instructions(*style);
                        block.push_str(&tinytools_agent::render::render_code_catalogue(
                            &specs, *style,
                        ));
                        block
                    }
                    _ => {
                        let mut block = PFormatDialect::instructions();
                        block.push_str(&tinytools_agent::render::render_pformat_catalogue(&specs));
                        block
                    }
                };
                // The XML branch renders `tool_choice` into its instructions
                // via `prompt_tools::tool_instructions`; P-Format has no
                // schema on the wire either (the wire choice is reset to
                // `Auto` below), so a forced choice has to be said in plain
                // English here too or `Required`/`Tool(name)` silently loses
                // its meaning — in particular the sole synthetic
                // structured-output tool would no longer be forced, and a
                // plain-text response would make extraction fail.
                match &request.tool_choice {
                    ToolChoice::Required => {
                        block.push_str("\nYou must emit at least one tool call.\n");
                    }
                    ToolChoice::Tool(name) => {
                        block.push_str(&format!("\nYou must call the `{name}` tool.\n"));
                    }
                    ToolChoice::Auto | ToolChoice::None => {}
                }
                prompt_tools::append_system_block(&messages, &block)
            }
        };
        request.tool_choice = ToolChoice::Auto;
        sync_stripped_tools_cache_segment(request, had_leading_system);
    }

    /// Renders `tools` into this dialect's catalogue shape alone (no
    /// protocol instructions): the `Self::Xml` full-schema form, the
    /// `Self::PFormat` positional-signature form, or the `Self::Code`
    /// function-signature form. Used to advertise a schema the host's own
    /// static catalogue could not have carried — see
    /// [`Self::apply_to_request`]'s `synthesized` parameter.
    fn render_catalogue(&self, tools: &[ToolSchema]) -> String {
        let specs: Vec<tinytools_agent::tinytools::ToolSpec> = tools
            .iter()
            .map(|schema| tinytools_agent::tinytools::ToolSpec {
                name: schema.name.clone(),
                description: schema.description.clone(),
                parameters: schema.parameters.clone(),
            })
            .collect();
        match self {
            Self::Xml | Self::Native => tinytools_agent::render::render_json_catalogue(&specs),
            Self::PFormat(_) => tinytools_agent::render::render_pformat_catalogue(&specs),
            Self::Code(style, _) => tinytools_agent::render::render_code_catalogue(&specs, *style),
        }
    }
}

/// Keeps a harness-declared `cache_segments` layout in sync with a text
/// dialect's rewrite, using the one thing only this call site still knows
/// for certain: whether `pre_rewrite_messages` already had a leading system
/// message *before* the protocol block gets folded in below.
///
/// `request.cache_segments` may declare a trailing canonical tools segment
/// (`{id: "tools", role: Tools, cacheable: true}`) that is about to
/// disappear once `request.tools` is cleared. When a leading system message
/// already existed, dropping that trailing segment is all that is needed —
/// the declared head still names the same messages it always did, and later
/// fingerprinting (`refresh_prompt_cache_fingerprint`) can verify that by
/// simple equality. But when none existed yet,
/// `tinyinference_llm::prompt_tools::append_system_block` (used by both the
/// host-rendered and ordinary rewrite paths below) synthesizes exactly one
/// new leading system message for the protocol block — a segment no
/// declaration could have named in advance. That case is resolved *here*,
/// with certain knowledge of the pre-rewrite shape, rather than left for
/// `refresh_prompt_cache_fingerprint` to guess from the rewritten request
/// alone: reconstructing it after the fact from the post-rewrite shape alone
/// cannot tell an actually-synthesized segment apart from a custom
/// declaration that deliberately left an already-present system message out
/// of the cache key, and conflating the two would silently widen what a
/// middleware asked to keep out of the stable prefix.
///
/// A declaration that is not exactly `[.., tools_segment]` — anything with a
/// head that does not otherwise account for the messages, or no declaration
/// at all — is left untouched, so `refresh_prompt_cache_fingerprint` keeps
/// taking the conservative whole-request digest for it.
fn sync_stripped_tools_cache_segment(request: &mut ModelRequest, had_leading_system: bool) {
    let canonical_tools_segment = PromptSegment {
        id: "tools".to_string(),
        role: SegmentRole::Tools,
        cacheable: true,
    };
    let Some((last, head)) = request.cache_segments.split_last() else {
        return;
    };
    if *last != canonical_tools_segment {
        return;
    }
    // The real, post-rewrite leading-system-message count — the exact same
    // thing `refresh_prompt_cache_fingerprint` will independently derive
    // from `request.messages` a moment later. Deriving the comparison
    // against this, rather than against `head`'s own declared shape, is
    // what lets every case below be a plain equality check instead of a
    // guess: whatever the rewrite actually did to the messages is the one
    // fact this function can trust.
    let final_system_end = request
        .messages
        .iter()
        .take_while(|message| matches!(message, Message::System(_)))
        .count();
    let canonical_head: Vec<PromptSegment> = (0..final_system_end)
        .map(|index| PromptSegment {
            id: crate::prompt::system_segment_id(index),
            role: SegmentRole::System,
            cacheable: true,
        })
        .collect();
    if head == canonical_head {
        // The declared head already names exactly the messages that are
        // really there (including the trivial `final_system_end == 0`
        // case, where both sides are empty because this rewrite turned out
        // to touch nothing — e.g. a host-rendered request with no
        // synthesized tool and an `Auto` choice leaves `messages`
        // untouched); only the now-gone trailing tools segment is stale.
        request.cache_segments = canonical_head;
    } else if head.is_empty() && !had_leading_system && final_system_end == 1 {
        // No declared head and no existing leading system message before
        // this call: the rewrite is the sole source of the new leading
        // segment (`prompt_tools::append_system_block` inserts exactly one
        // when none exists), so this is unambiguously the harness's own
        // synthesis rather than something a declaration could have named in
        // advance.
        request.cache_segments = canonical_head;
    }
    // Every other combination is left completely untouched, including the
    // trailing tools segment: `head.is_empty() && had_leading_system` is a
    // declaration that deliberately named nothing ahead of the tools
    // segment even though a system message already existed (dropping to
    // `head` would leave an *empty* `cache_segments`, which
    // `refresh_prompt_cache_fingerprint` reads as "nothing declared yet"
    // and promotes just the same); a non-empty `head` that does not match
    // `canonical_head` is a custom declaration (a middleware-owned id, or a
    // stale count) that must not be silently rewritten out from under it,
    // partially or otherwise.
}

/// Builds the positional layout registry the P-Format and code dialects
/// bind against, from the schemas offered this run.
fn registry_from(tools: &[ToolSchema]) -> PFormatRegistry {
    tinytools_agent::build_registry(
        tools
            .iter()
            .map(|schema| (schema.name.clone(), schema.parameters.clone())),
    )
}

/// What a model call needs in order to recover text-dialect calls: the
/// tools that were offered (which a text-dialect request no longer carries
/// on the wire) and the P-Format registry, when there is one.
#[derive(Debug, Clone, Default)]
pub(super) struct TextRecovery {
    /// The tools offered this turn, before any dialect rewrite.
    pub(super) offered: Arc<Vec<ToolSchema>>,
    /// The P-Format layouts, for [`RunDialect::PFormat`].
    pub(super) registry: Option<Arc<PFormatRegistry>>,
}

impl TextRecovery {
    /// A scrubber for one streamed model call, or `None` when no tools were
    /// offered and there is nothing to recover.
    pub(super) fn scrubber(&self, model_call_id: &CallId) -> Option<DeltaScrubber> {
        (!self.offered.is_empty()).then(|| {
            DeltaScrubber::new(model_call_id.clone(), &self.offered, self.registry.clone())
        })
    }
}

/// How one model call is made: streamed or unary, and what it needs to
/// recover text-dialect calls from the answer.
#[derive(Debug, Clone, Default)]
pub(super) struct CallShape {
    /// Whether the provider's streaming path is used.
    pub(super) streaming: bool,
    /// Offered tools and P-Format registry for text recovery.
    pub(super) recovery: TextRecovery,
    /// Whether this call's final response is eligible for the opt-in retry
    /// when it has no visible answer. Structured-output plans are excluded.
    pub(super) retry_empty_final: bool,
}

/// Converts a recovered call into the harness's [`ToolCall`], minting an id
/// scoped to the model call it came from.
///
/// `{model_call_id}-tool-{n}` is unique per run by construction — model call
/// ids already are — and visibly distinct from any provider's, so a
/// recovered call can never be confused with a native one in a transcript.
fn to_tool_call(call: ParsedToolCall, model_call_id: &CallId, slot: usize) -> ToolCall {
    // `call.id` is intentionally never used, even when a grammar or a future
    // change to `tinytools-agent` happens to populate one: this function's
    // whole contract (see its doc comment) is that a text-recovered call's id
    // is always host-minted and unique per run, so it can never collide with
    // another recovered call or be confused with a native provider one. A
    // parser-supplied id would be model-controlled input; trusting it here
    // would let two calls collide on an id the model chose, or let a
    // narrated call impersonate a specific native one.
    let id = format!("{model_call_id}-tool-{slot}");
    ToolCall::new(id, call.name, call.arguments)
}

/// Reads text-dialect calls out of a response that carries no structured
/// ones, through every grammar `tinytools-agent` knows, with the offered
/// tools enabling name repair. Non-text content blocks (reasoning) survive.
pub(super) fn recover_text_calls(
    response: &mut ModelResponse,
    model_call_id: &CallId,
    offered: &[ToolSchema],
    registry: Option<&PFormatRegistry>,
) {
    if offered.is_empty() {
        return;
    }
    let known: Vec<String> = offered.iter().map(|tool| tool.name.clone()).collect();
    let mut options = ParseOptions::new().with_known_tools(&known);
    if let Some(registry) = registry {
        options = options.with_registry(registry);
    }
    let text = response.text();
    let outcome = tinytools_agent::parse_text(&text, &options);
    if outcome.calls.is_empty() {
        return;
    }
    for diagnostic in &outcome.diagnostics {
        tracing::debug!(?diagnostic, "[agent_loop] text-dialect recovery");
    }
    // Appended, not assigned: a provider can legitimately return one native
    // structured call *and* narrate a second one as text in the same
    // response (this is deliberately parsed even when `tool_calls` was
    // already non-empty — see above), and overwriting the collection here
    // used to silently drop whichever set ran second.
    let recovered = outcome
        .calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| to_tool_call(call, model_call_id, index + 1));
    response.message.tool_calls.extend(recovered);
    response.message.content =
        replace_text_blocks(std::mem::take(&mut response.message.content), outcome.text);
}

/// Keeps every non-text block in place and substitutes one cleaned text at
/// the first text block's position; an empty `cleaned` emits no text block.
fn replace_text_blocks(content: Vec<ContentBlock>, cleaned: String) -> Vec<ContentBlock> {
    let mut out = Vec::with_capacity(content.len());
    let mut inserted = false;
    for block in content {
        match block {
            ContentBlock::Text(_) => {
                if !inserted {
                    if !cleaned.is_empty() {
                        out.push(ContentBlock::Text(cleaned.clone()));
                    }
                    inserted = true;
                }
            }
            other => out.push(other),
        }
    }
    if !inserted && !cleaned.is_empty() {
        out.push(ContentBlock::Text(cleaned));
    }
    out
}

/// Scrubs tool-call markup from streamed visible text and collects the
/// calls it completes, minting harness ids for them.
///
/// Consumers of [`AgentEvent::ModelDelta`](crate::events::AgentEvent::ModelDelta)
/// never see a partial `<tool_call>`; the calls surface on the terminal
/// response instead, exactly once.
pub(super) struct DeltaScrubber {
    inner: StreamScrubber,
    model_call_id: CallId,
    calls: Vec<ToolCall>,
}

impl DeltaScrubber {
    /// A scrubber for one model call, knowing the tools it offered.
    pub(super) fn new(
        model_call_id: CallId,
        offered: &[ToolSchema],
        registry: Option<Arc<PFormatRegistry>>,
    ) -> Self {
        let known = offered.iter().map(|tool| tool.name.clone()).collect();
        let mut inner = StreamScrubber::new().with_known_tools(known);
        if let Some(registry) = registry {
            inner = inner.with_registry(registry);
        }
        Self {
            inner,
            model_call_id,
            calls: Vec::new(),
        }
    }

    /// Feeds one text delta; returns the text safe to forward.
    pub(super) fn feed(&mut self, text: &str) -> String {
        let step = self.inner.feed(text);
        self.collect(step.calls);
        step.text
    }

    /// Drains the remainder at end of stream.
    pub(super) fn flush(&mut self) -> String {
        let step = self.inner.flush();
        self.collect(step.calls);
        step.text
    }

    fn collect(&mut self, calls: Vec<ParsedToolCall>) {
        for call in calls {
            let slot = self.calls.len() + 1;
            self.calls
                .push(to_tool_call(call, &self.model_call_id, slot));
        }
    }

    /// The calls completed during the stream, in order.
    pub(super) fn into_calls(self) -> Vec<ToolCall> {
        self.calls
    }

    /// Whether any call was completed during the stream.
    pub(super) fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }
}
