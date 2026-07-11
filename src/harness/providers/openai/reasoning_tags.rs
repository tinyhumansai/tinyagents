//! Streaming-safe extraction of inline `<think>…</think>` reasoning tags.
//!
//! Reasoning models served through OpenAI-compatible local runtimes (qwen3 and
//! deepseek-r1 distills via Ollama `/v1`, LM Studio, llama.cpp) frequently emit
//! their chain-of-thought **inline** in the normal `content` string, wrapped in
//! `<think>…</think>`, instead of on the `reasoning_content` / `reasoning`
//! side-channel the adapter already normalizes (see
//! [`reasoning_value_text`](super::reasoning_value_text)). Left untouched, that
//! chain-of-thought leaks straight into the visible assistant text.
//!
//! This module moves the tagged text onto the reasoning channel
//! ([`ContentBlock::Thinking`](crate::harness::message::ContentBlock::Thinking))
//! and strips the tags from the visible text. It provides two entry points that
//! share the same tag-matching logic so the streamed deltas and the final
//! response agree on what is reasoning and what is visible:
//!
//! * [`ReasoningTagStream`] — an incremental state machine for the SSE path. It
//!   buffers and holds back any trailing bytes that *could* be the prefix of an
//!   opening or closing tag ([`potential_start_index`], the Vercel AI SDK's
//!   `getPotentialStartIndex` trick) so a tag split across deltas is neither
//!   leaked as visible text nor mangled.
//! * [`extract_reasoning`] — a whole-string extractor for the non-streaming
//!   path (and the authoritative terminal streamed response).
//!
//! # Side-channel interaction
//!
//! Inline-tag reasoning and side-channel reasoning both feed the single leading
//! `Thinking` block. When a response carries both, side-channel reasoning leads
//! and inline-extracted reasoning follows, joined by the configured separator.
//! In practice a given model uses one convention or the other, not both.

/// Options controlling inline `<tag>…</tag>` reasoning extraction on the
/// OpenAI-compatible provider. Construct with [`ReasoningTagExtraction::new`] or
/// [`ReasoningTagExtraction::default`] (the `think` tag) and pass to
/// [`OpenAiModel::with_reasoning_tag_extraction`](super::OpenAiModel::with_reasoning_tag_extraction).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningTagExtraction {
    /// Tag name without angle brackets, e.g. `think` → `<think>` / `</think>`.
    tag_name: String,
    /// Separator inserted between multiple extracted reasoning sections (and
    /// between side-channel and inline reasoning). Defaults to a newline.
    separator: String,
    /// When `true`, the output BEGINS mid-reasoning with no opening tag and only
    /// a closing `</tag>` appears — the DeepSeek-R1 template mode (the AI SDK's
    /// `startWithReasoning`). Everything before the first closing tag is
    /// reasoning. Defaults to `false`.
    start_with_reasoning: bool,
}

impl Default for ReasoningTagExtraction {
    fn default() -> Self {
        Self {
            tag_name: "think".to_string(),
            separator: "\n".to_string(),
            start_with_reasoning: false,
        }
    }
}

impl ReasoningTagExtraction {
    /// Extraction for a custom tag name (no angle brackets), newline separator,
    /// opening-tag-gated (not DeepSeek mode).
    pub fn new(tag_name: impl Into<String>) -> Self {
        Self {
            tag_name: tag_name.into(),
            ..Self::default()
        }
    }

    /// Overrides the separator joining multiple reasoning sections.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Enables DeepSeek-R1 template mode: the stream begins mid-reasoning with no
    /// opening tag, and only a closing `</tag>` marks the end of reasoning.
    pub fn with_start_with_reasoning(mut self, enabled: bool) -> Self {
        self.start_with_reasoning = enabled;
        self
    }

    /// The literal opening tag, e.g. `<think>`.
    fn opening_tag(&self) -> String {
        format!("<{}>", self.tag_name)
    }

    /// The literal closing tag, e.g. `</think>`.
    fn closing_tag(&self) -> String {
        format!("</{}>", self.tag_name)
    }

    /// The separator joining reasoning sections.
    pub(super) fn separator(&self) -> &str {
        &self.separator
    }
}

/// Finds where `searched` begins within `text`, as either a full occurrence or a
/// partial prefix at the very end of `text`.
///
/// Returns the byte offset of the earliest position at which `searched` could
/// start:
/// * `Some(idx)` where `text[idx..]` fully contains `searched` (a complete tag),
///   or where `text[idx..]` is a non-empty proper prefix of `searched` (a
///   partial tag straddling the end that must be held back for more input);
/// * `None` when no suffix of `text` could begin `searched` — the whole `text`
///   is safe to release.
///
/// This is the Rust port of the Vercel AI SDK's `getPotentialStartIndex`. It
/// only ever returns positions on UTF-8 character boundaries (the tags are
/// ASCII, and it scans `char_indices`), so callers may slice `text` at the
/// result safely.
pub(super) fn potential_start_index(text: &str, searched: &str) -> Option<usize> {
    if searched.is_empty() {
        return None;
    }
    if let Some(idx) = text.find(searched) {
        return Some(idx);
    }
    // No full occurrence: look for a suffix of `text` that is a prefix of
    // `searched`, scanning shortest-suffix-first (end inward) to match the AI
    // SDK. Iterate char-boundary starts so any returned index is sliceable.
    let starts: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    for &i in starts.iter().rev() {
        let suffix = &text[i..];
        if searched.starts_with(suffix) {
            return Some(i);
        }
    }
    None
}

/// Incremental extractor for the streamed content path.
///
/// Feed each `content` delta to [`push`](Self::push); it appends any resolved
/// visible / reasoning text to the supplied buffers and retains any trailing
/// partial-tag bytes internally until the next delta (or [`finish`](Self::finish)
/// at end of stream) resolves them.
#[derive(Clone, Debug)]
pub(super) struct ReasoningTagStream {
    opening: String,
    closing: String,
    /// Bytes received but not yet released (may end in a partial tag).
    buffer: String,
    /// Whether the machine is currently inside a reasoning section.
    in_reasoning: bool,
}

impl ReasoningTagStream {
    /// Builds a stream extractor from the configured tags. In DeepSeek mode the
    /// machine starts already inside a reasoning section.
    pub(super) fn new(config: &ReasoningTagExtraction) -> Self {
        Self {
            opening: config.opening_tag(),
            closing: config.closing_tag(),
            buffer: String::new(),
            in_reasoning: config.start_with_reasoning,
        }
    }

    /// Folds one content delta into the machine, appending resolved text to
    /// `visible` / `reasoning`. A trailing partial tag is held in `buffer`.
    pub(super) fn push(&mut self, delta: &str, visible: &mut String, reasoning: &mut String) {
        self.buffer.push_str(delta);
        loop {
            let tag = if self.in_reasoning {
                &self.closing
            } else {
                &self.opening
            };
            match potential_start_index(&self.buffer, tag) {
                // Neither a complete tag nor a partial-tag suffix: everything
                // buffered is safe to release.
                None => {
                    let released = std::mem::take(&mut self.buffer);
                    self.emit(&released, visible, reasoning);
                    break;
                }
                Some(idx) => {
                    // Text before the (partial or full) tag belongs to the
                    // current channel.
                    let before = self.buffer[..idx].to_string();
                    self.emit(&before, visible, reasoning);
                    if idx + tag.len() <= self.buffer.len() {
                        // Complete tag: drop it, toggle channel, keep scanning
                        // the remainder for the opposite tag.
                        self.buffer = self.buffer[idx + tag.len()..].to_string();
                        self.in_reasoning = !self.in_reasoning;
                    } else {
                        // Partial tag at the buffer tail: hold it for more input.
                        self.buffer = self.buffer[idx..].to_string();
                        break;
                    }
                }
            }
        }
    }

    /// Releases any buffered partial-tag tail at end of stream into the current
    /// channel — a partial tag that never completed is real content.
    ///
    /// The streaming provider path does not call this: it recomputes the
    /// authoritative split from the raw accumulated content in
    /// [`into_response`](super::OpenAiStreamAcc), which also applies the
    /// non-streaming trimming so the terminal response matches exactly. This is
    /// the state machine's flush contract, exercised by the unit tests.
    #[cfg(test)]
    pub(super) fn finish(&mut self, visible: &mut String, reasoning: &mut String) {
        let released = std::mem::take(&mut self.buffer);
        self.emit(&released, visible, reasoning);
    }

    fn emit(&self, text: &str, visible: &mut String, reasoning: &mut String) {
        if text.is_empty() {
            return;
        }
        if self.in_reasoning {
            reasoning.push_str(text);
        } else {
            visible.push_str(text);
        }
    }
}

/// Extracts inline reasoning from a complete `content` string.
///
/// Returns `(visible_text, reasoning_text)`. Reasoning sections are trimmed and
/// joined with the configured separator; the visible text has the tagged
/// sections (and the whitespace that surrounded them) removed. When the content
/// contains no reasoning at all, it is returned verbatim as the visible text so
/// plain responses are preserved byte-for-byte.
pub(super) fn extract_reasoning(
    config: &ReasoningTagExtraction,
    content: &str,
) -> (String, String) {
    let opening = config.opening_tag();
    let closing = config.closing_tag();
    let mut in_reasoning = config.start_with_reasoning;
    let mut rest = content;
    let mut visible_parts: Vec<&str> = Vec::new();
    let mut reasoning_parts: Vec<&str> = Vec::new();

    loop {
        let tag = if in_reasoning { &closing } else { &opening };
        match rest.find(tag.as_str()) {
            Some(idx) => {
                let (before, after) = rest.split_at(idx);
                if in_reasoning {
                    reasoning_parts.push(before);
                } else {
                    visible_parts.push(before);
                }
                rest = &after[tag.len()..];
                in_reasoning = !in_reasoning;
            }
            None => {
                if in_reasoning {
                    reasoning_parts.push(rest);
                } else {
                    visible_parts.push(rest);
                }
                break;
            }
        }
    }

    // No reasoning was ever entered: return the content untouched so plain text
    // is preserved exactly (no whitespace normalization).
    if !config.start_with_reasoning && reasoning_parts.is_empty() {
        return (content.to_string(), String::new());
    }

    // Visible text: concatenate the surviving segments (matching the streaming
    // deltas, which carry no separator) and trim the whitespace that bordered
    // the removed sections at the ends.
    let visible = visible_parts.concat().trim().to_string();
    // Reasoning: join distinct sections with the configured separator, trimming
    // each so tag-adjacent whitespace does not bloat the thinking block.
    let reasoning = reasoning_parts
        .iter()
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(&config.separator);

    (visible, reasoning)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a sequence of content deltas through the stream machine and
    /// returns the accumulated `(visible, reasoning)`.
    fn run_stream(config: &ReasoningTagExtraction, deltas: &[&str]) -> (String, String) {
        let mut machine = ReasoningTagStream::new(config);
        let mut visible = String::new();
        let mut reasoning = String::new();
        for delta in deltas {
            machine.push(delta, &mut visible, &mut reasoning);
        }
        machine.finish(&mut visible, &mut reasoning);
        (visible, reasoning)
    }

    #[test]
    fn potential_start_index_finds_full_and_partial_matches() {
        assert_eq!(potential_start_index("ab<think>c", "<think>"), Some(2));
        // Partial tag straddling the end is held from its first byte.
        assert_eq!(potential_start_index("hello<thi", "<think>"), Some(5));
        assert_eq!(potential_start_index("<", "<think>"), Some(0));
        // A lone trailing `<` with a non-matching continuation is not held.
        assert_eq!(potential_start_index("<h", "<think>"), None);
        assert_eq!(potential_start_index("plain text", "<think>"), None);
        // `<thinker>` is not `<think>` and must not be held back.
        assert_eq!(potential_start_index("<thinker>", "<think>"), None);
    }

    #[test]
    fn tag_fully_inside_one_delta() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = run_stream(&cfg, &["Hi<think>ponder</think>there"]);
        assert_eq!(visible, "Hithere");
        assert_eq!(reasoning, "ponder");
    }

    #[test]
    fn tag_split_across_deltas_at_every_boundary() {
        let cfg = ReasoningTagExtraction::default();
        let source = "before<think>secret</think>after";
        let open = "<think>";
        // Split the opening tag at every interior byte boundary.
        for cut in 0..=open.len() {
            let prefix = format!("before{}", &open[..cut]);
            let suffix = format!("{}secret</think>after", &open[cut..]);
            let (visible, reasoning) = run_stream(&cfg, &[&prefix, &suffix]);
            assert_eq!(visible, "beforeafter", "cut at {cut}");
            assert_eq!(reasoning, "secret", "cut at {cut}");
        }
        // Sanity: the whole thing in one delta agrees.
        let (visible, reasoning) = run_stream(&cfg, &[source]);
        assert_eq!(visible, "beforeafter");
        assert_eq!(reasoning, "secret");
    }

    #[test]
    fn closing_tag_split_across_deltas() {
        let cfg = ReasoningTagExtraction::default();
        let close = "</think>";
        for cut in 0..=close.len() {
            let first = format!("<think>reason{}", &close[..cut]);
            let second = format!("{}visible", &close[cut..]);
            let (visible, reasoning) = run_stream(&cfg, &[&first, &second]);
            assert_eq!(visible, "visible", "cut at {cut}");
            assert_eq!(reasoning, "reason", "cut at {cut}");
        }
    }

    #[test]
    fn lone_angle_bracket_is_not_held_forever() {
        let cfg = ReasoningTagExtraction::default();
        // A `<` that turns out not to open a tag is released once disproven.
        let (visible, reasoning) = run_stream(&cfg, &["a < b < c"]);
        assert_eq!(visible, "a < b < c");
        assert_eq!(reasoning, "");

        // `<` held at a delta boundary, then disproven by the next delta.
        let (visible, reasoning) = run_stream(&cfg, &["value <", "= 3"]);
        assert_eq!(visible, "value <= 3");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn thinker_lookalike_tag_is_not_extracted() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = run_stream(&cfg, &["use <thinker> here"]);
        assert_eq!(visible, "use <thinker> here");
        assert_eq!(reasoning, "");

        // Even when split so `<think` is a real prefix mid-stream.
        let (visible, reasoning) = run_stream(&cfg, &["use <think", "er> here"]);
        assert_eq!(visible, "use <thinker> here");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn think_section_spanning_many_deltas() {
        let cfg = ReasoningTagExtraction::default();
        let deltas = [
            "Answer: ",
            "<think>",
            "step one, ",
            "step two, ",
            "step three",
            "</think>",
            "42",
        ];
        let (visible, reasoning) = run_stream(&cfg, &deltas);
        assert_eq!(visible, "Answer: 42");
        assert_eq!(reasoning, "step one, step two, step three");
    }

    #[test]
    fn text_after_think_is_visible() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = run_stream(&cfg, &["<think>hmm</think>", "the final answer"]);
        assert_eq!(visible, "the final answer");
        assert_eq!(reasoning, "hmm");
    }

    #[test]
    fn start_with_reasoning_mode_stream() {
        // DeepSeek-R1 template: output begins mid-reasoning, only a closing tag.
        let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
        let (visible, reasoning) =
            run_stream(&cfg, &["chain of ", "thought</think>", "final answer"]);
        assert_eq!(visible, "final answer");
        assert_eq!(reasoning, "chain of thought");
    }

    #[test]
    fn start_with_reasoning_without_closing_tag_is_all_reasoning() {
        let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
        let (visible, reasoning) = run_stream(&cfg, &["still ", "thinking"]);
        assert_eq!(visible, "");
        assert_eq!(reasoning, "still thinking");
    }

    #[test]
    fn unclosed_think_streams_remainder_as_reasoning() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = run_stream(&cfg, &["ok <think>never closed"]);
        assert_eq!(visible, "ok ");
        assert_eq!(reasoning, "never closed");
    }

    #[test]
    fn non_streaming_extraction_strips_surrounding_whitespace() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) =
            extract_reasoning(&cfg, "<think>deliberate</think>\n\nThe answer is 42.");
        assert_eq!(visible, "The answer is 42.");
        assert_eq!(reasoning, "deliberate");
    }

    #[test]
    fn non_streaming_plain_text_is_preserved_verbatim() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = extract_reasoning(&cfg, "line one\nline two");
        assert_eq!(visible, "line one\nline two");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn non_streaming_multiple_sections_join_reasoning_with_separator() {
        let cfg = ReasoningTagExtraction::default();
        let (visible, reasoning) = extract_reasoning(&cfg, "a<think>r1</think>b<think>r2</think>c");
        // Visible segments concatenate (consistent with the streamed deltas);
        // reasoning sections join with the separator.
        assert_eq!(visible, "abc");
        assert_eq!(reasoning, "r1\nr2");
    }

    #[test]
    fn non_streaming_start_with_reasoning() {
        let cfg = ReasoningTagExtraction::default().with_start_with_reasoning(true);
        let (visible, reasoning) =
            extract_reasoning(&cfg, "reasoning here</think>\n\nvisible answer");
        assert_eq!(visible, "visible answer");
        assert_eq!(reasoning, "reasoning here");
    }

    #[test]
    fn custom_tag_name_is_honored() {
        let cfg = ReasoningTagExtraction::new("reasoning");
        let (visible, reasoning) = run_stream(&cfg, &["a<reasoning>", "b</reasoning>c"]);
        assert_eq!(visible, "ac");
        assert_eq!(reasoning, "b");
    }
}
