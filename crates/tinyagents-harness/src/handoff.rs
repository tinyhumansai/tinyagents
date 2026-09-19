//! Progressive-disclosure handoff cache for oversized tool results.
//!
//! Sub-agents regularly call tools
//! that return megabyte-scale payloads — `GMAIL_LIST_MESSAGES`,
//! `NOTION_GET_PAGE`, `GOOGLEDRIVE_LIST_FILES`. The default behaviour pushes
//! that raw blob into the sub-agent's history as a tool-result message, and
//! the NEXT iteration ships the bloated history back to the provider where
//! it hits the model's context-length ceiling.
//!
//! Progressive disclosure fixes this: when a tool returns too much data we
//! stash the full payload here, replace it in history with a short
//! placeholder (size + preview + `result_id` + how to query it), and expose
//! an extraction tool — host-side, since a tool is host vocabulary — that the
//! sub-agent can call with a targeted query. The extractor only runs when
//! the sub-agent actually asks for a narrower view.
//!
//! This module owns:
//! * the thresholds and limits (token cut-off, preview size, max entries);
//! * the [`ResultHandoffCache`] store itself (FIFO-evicting, `Arc`-shared);
//! * the [`build_handoff_placeholder`] renderer used when rewriting tool
//!   results into history.
//!
//! # Not on the agent loop path (M-10)
//!
//! This is a host utility, not something [`crate::agent_loop`] calls on its
//! own: nothing in the built-in loop invokes [`apply_handoff`] or registers
//! the extraction tool it references. A host wires this in itself — calling
//! `apply_handoff` on each tool result before it is appended to history, and
//! registering an extraction tool (named per [`HandoffConfig::extractor_tool_name`])
//! that reads from the same [`ResultHandoffCache`].

use std::collections::HashMap;
use std::sync::Mutex as StdMutex;

// ── Tunables ───────────────────────────────────────────────────────────────

/// Token threshold above which a tool result is routed to the handoff
/// cache instead of being pushed into history raw. Token count is
/// estimated at ~4 chars/token.
///
/// Set at `50_000` so the clean Gmail / Notion envelopes emitted by provider
/// post-processing fit through unchanged for normal workloads — only
/// genuinely oversized results (bulk fetches, raw thread dumps) are routed
/// through the `extract_from_result` path.
pub const HANDOFF_OVERSIZE_THRESHOLD_TOKENS: usize = 50_000;

/// Characters of the raw payload to surface in the placeholder preview.
/// Enough for the sub-agent to recognise the shape (JSON keys, first
/// record) and often small enough to answer trivial questions without a
/// follow-up `extract_from_result` call.
pub const HANDOFF_PREVIEW_CHARS: usize = 1500;

/// Maximum entries per session. Bounded to keep memory use predictable on
/// long-running sub-agents that might call many large tools. When over
/// capacity we evict the oldest entry (FIFO); callers see "no cached
/// result" for evicted ids and can either re-run the tool or ask the
/// user/orchestrator to narrow the request.
pub const HANDOFF_MAX_ENTRIES: usize = 8;

// ── Host configuration ───────────────────────────────────────────────────────

/// Host-specific naming this module needs but does not own: which tool name
/// is the extractor (so its own output passes through uncleaned/unstashed),
/// and which literal prefixes mark a result as already an error.
///
/// Both were previously hardcoded to one particular host's conventions
/// (`extract_from_result`, a bare `result_text.starts_with("Error")`) even
/// though the rest of this module is host-agnostic (M-9). A different host
/// — a different extractor tool name, or error-carrying results that do not
/// start with the literal word "Error" — could not use this cache without
/// forking the module. [`HandoffConfig::default`] reproduces the historical
/// behaviour exactly, so existing callers are unaffected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandoffConfig {
    /// The tool name whose own output skips cleaning/stashing (it is already
    /// a narrowed, host-curated response to a targeted query).
    pub extractor_tool_name: String,
    /// Literal prefixes that mark `result_text` as already an error, which
    /// also skips cleaning/stashing (an error message should reach the model
    /// verbatim, not truncated or placeholder-replaced).
    pub error_prefixes: Vec<String>,
}

impl Default for HandoffConfig {
    fn default() -> Self {
        Self {
            extractor_tool_name: "extract_from_result".to_string(),
            error_prefixes: vec!["Error".to_string()],
        }
    }
}

impl HandoffConfig {
    /// `true` when `result_text` starts with any of
    /// [`HandoffConfig::error_prefixes`].
    fn is_error_result(&self, result_text: &str) -> bool {
        self.error_prefixes
            .iter()
            .any(|prefix| result_text.starts_with(prefix.as_str()))
    }
}

// ── Store ──────────────────────────────────────────────────────────────────

/// Per-spawn cache of oversized tool payloads. One instance is built at
/// the top of `run_typed_mode` and shared (via `Arc`) with both the inner
/// tool-call loop (writes) and the `extract_from_result` tool (reads).
#[derive(Default)]
pub struct ResultHandoffCache {
    inner: StdMutex<HandoffInner>,
}

#[derive(Default)]
struct HandoffInner {
    /// FIFO of inserted ids, used for eviction.
    order: Vec<String>,
    /// Content by id.
    entries: HashMap<String, CachedResult>,
    /// Monotonic counter for id generation within the session.
    next_id: u64,
}

/// One stashed tool result: the raw content plus the name of the tool that
/// produced it, returned by [`ResultHandoffCache::get`] and stored on
/// [`ResultHandoffCache::store`].
pub struct CachedResult {
    /// Name of the tool whose result was stashed.
    pub tool_name: String,
    /// The (cleaned) payload the sub-agent can query with
    /// `extract_from_result`.
    pub content: String,
}

impl ResultHandoffCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stash a payload and return a stable, short, grep-friendly id.
    pub fn store(&self, tool_name: String, content: String) -> String {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        g.next_id = g.next_id.saturating_add(1);
        let id = format!("res_{:x}", g.next_id);
        g.order.push(id.clone());
        g.entries
            .insert(id.clone(), CachedResult { tool_name, content });
        while g.order.len() > HANDOFF_MAX_ENTRIES {
            let evicted = g.order.remove(0);
            g.entries.remove(&evicted);
        }
        id
    }

    /// Looks up a previously stashed result by id, returning `None` if it was
    /// never stored or has since been evicted (FIFO, past
    /// [`HANDOFF_MAX_ENTRIES`]).
    pub fn get(&self, result_id: &str) -> Option<CachedResult> {
        let g = self.inner.lock().ok()?;
        g.entries.get(result_id).map(|r| CachedResult {
            tool_name: r.tool_name.clone(),
            content: r.content.clone(),
        })
    }
}

/// Apply the progressive-disclosure handoff to a tool result. If a cache is
/// present and the cleaned result is large enough, stash the raw payload and
/// substitute a short placeholder the sub-agent can drill into with
/// `extract_from_result`. Errors and already-extracted output pass through
/// unchanged.
///
/// `threshold_tokens` is the size above which a payload is stashed rather than
/// inlined. It is a parameter rather than a constant read here because a host
/// legitimately varies it — by agent, by model context window, or to exercise
/// the path in tests — and because the alternative this replaced was an
/// environment-variable backdoor named after one particular host. Pass
/// [`HANDOFF_OVERSIZE_THRESHOLD_TOKENS`] for the default.
///
/// `config` supplies the host's extractor tool name and error-prefix
/// heuristics (M-9); pass [`HandoffConfig::default`] to reproduce the
/// historical hardcoded behaviour.
pub fn apply_handoff(
    cache: &ResultHandoffCache,
    config: &HandoffConfig,
    tool_name: &str,
    task_id: &str,
    agent_id: &str,
    result_text: String,
    threshold_tokens: usize,
) -> String {
    let skip_cleaning =
        tool_name == config.extractor_tool_name || config.is_error_result(&result_text);
    let cleaned = if skip_cleaning {
        result_text
    } else {
        let pre_len = result_text.len();
        let cleaned = clean_tool_output(&result_text);
        if cleaned.len() < pre_len {
            tracing::debug!(
                tool = %tool_name,
                before_bytes = pre_len,
                after_bytes = cleaned.len(),
                saved_pct = ((pre_len - cleaned.len()) * 100) / pre_len.max(1),
                "[handoff] cleaned tool output (stripped markup/data-uris/whitespace)"
            );
        }
        cleaned
    };
    let tokens = cleaned.len().div_ceil(4);
    if !skip_cleaning && tokens > threshold_tokens {
        let id = cache.store(tool_name.to_string(), cleaned.clone());
        let placeholder = build_handoff_placeholder(config, tool_name, &id, &cleaned);
        tracing::info!(
            task_id = %task_id,
            agent_id = %agent_id,
            tool = %tool_name,
            raw_tokens = tokens,
            raw_bytes = cleaned.len(),
            threshold_tokens,
            result_id = %id,
            "[handoff] stashed oversized tool output; substituted placeholder into history"
        );
        placeholder
    } else {
        cleaned
    }
}

// ── Placeholder renderer ───────────────────────────────────────────────────

/// Build the placeholder text that replaces an oversized tool result in
/// the sub-agent's history. Shows the payload size (estimated tokens and
/// raw bytes), a preview, and a call shape for the configured extractor
/// tool ([`HandoffConfig::extractor_tool_name`]). The sub-agent decides
/// whether to answer from the preview or dispatch the extractor.
///
/// Token count is estimated at ~4 chars/token (same heuristic as the
/// trigger threshold in [`HANDOFF_OVERSIZE_THRESHOLD_TOKENS`]), so the
/// unit the sub-agent sees matches the unit the runtime used to decide
/// to hand off in the first place.
pub fn build_handoff_placeholder(
    config: &HandoffConfig,
    tool_name: &str,
    result_id: &str,
    raw: &str,
) -> String {
    let preview: String = raw.chars().take(HANDOFF_PREVIEW_CHARS).collect();
    let raw_tokens = raw.len().div_ceil(4);
    let extractor = &config.extractor_tool_name;
    format!(
        "[oversized tool output: {raw_tokens} tokens ({raw_bytes} bytes) — stashed as result_id=\"{result_id}\"]\n\
         Preview (first {preview_chars} chars):\n{preview}\n\n\
         If the preview does not answer your task, call:\n\
         {extractor}(result_id=\"{result_id}\", query=\"<specific question>\")\n\
         Good queries name the exact fields/identifiers you need \
         (e.g. \"subject and sender of the 5 most recent messages\"). \
         Tool: {tool_name}",
        raw_bytes = raw.len(),
        preview_chars = preview.chars().count(),
    )
}

// ── Content hygiene helpers (used by the extract path) ─────────────────────

use regex::Regex;
use std::sync::LazyLock;

/// Strip common noise from tool outputs before they're stashed or chunked.
///
/// Agent tools frequently return raw HTML email bodies, inline SVG, base64
/// data URIs, CSS/JS blocks, and collapsed whitespace — all of which bloat
/// the handoff cache and waste summarizer context on tokens that carry
/// zero semantic value for most extraction queries. Cleaning before the
/// oversize check means (a) some payloads drop below threshold entirely
/// and skip the extract pipeline, (b) chunked payloads fit more real
/// content per chunk, and (c) summarizers see clean text instead of
/// parsing around markup.
pub fn clean_tool_output(content: &str) -> String {
    static SCRIPT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<script\b[^>]*>.*?</script\s*>").unwrap());
    static STYLE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<style\b[^>]*>.*?</style\s*>").unwrap());
    static SVG_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<svg\b[^>]*>.*?</svg\s*>").unwrap());
    static HTML_COMMENT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());
    static DATA_URI_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)data:[a-z0-9.+\-/]+;base64,[A-Za-z0-9+/=]+").unwrap());
    static HTML_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
    static WS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\f\v]+").unwrap());
    static BLANK_LINE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

    let cleaned = SCRIPT_RE.replace_all(content, "");
    let cleaned = STYLE_RE.replace_all(&cleaned, "");
    let cleaned = SVG_RE.replace_all(&cleaned, "[svg]");
    let cleaned = HTML_COMMENT_RE.replace_all(&cleaned, "");
    let cleaned = DATA_URI_RE.replace_all(&cleaned, "[data-uri]");
    let cleaned = HTML_TAG_RE.replace_all(&cleaned, "");
    let cleaned = WS_RE.replace_all(&cleaned, " ");
    let cleaned = BLANK_LINE_RE.replace_all(&cleaned, "\n\n");
    cleaned.trim().to_string()
}

/// Split `content` into chunks no larger than `budget` bytes, breaking
/// at natural boundaries (blank lines, then single newlines) so the
/// extraction LLM rarely sees a structure torn mid-record. Falls back to
/// char-safe slicing for pathological single-line inputs.
pub fn chunk_content(content: &str, budget: usize) -> Vec<String> {
    if content.len() <= budget {
        return vec![content.to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::with_capacity(budget.min(content.len()));

    let flush = |current: &mut String, chunks: &mut Vec<String>| {
        if !current.is_empty() {
            chunks.push(std::mem::take(current));
        }
    };

    for line in content.lines() {
        let projected = current.len() + line.len() + 1;
        if projected > budget && !current.is_empty() {
            flush(&mut current, &mut chunks);
        }
        if line.len() > budget {
            // Single line exceeds budget (e.g. JSON with no formatting).
            // Emit any pending content, then slice the line at char
            // boundaries so we don't panic on multi-byte chars.
            flush(&mut current, &mut chunks);
            let mut remaining = line;
            while !remaining.is_empty() {
                let mut cut = budget.min(remaining.len());
                while cut > 0 && !remaining.is_char_boundary(cut) {
                    cut -= 1;
                }
                if cut == 0 {
                    // Degenerate case — shouldn't happen for normal
                    // text. Take the entire remaining line to avoid
                    // an infinite loop.
                    chunks.push(remaining.to_string());
                    break;
                }
                chunks.push(remaining[..cut].to_string());
                remaining = &remaining[cut..];
            }
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }
    flush(&mut current, &mut chunks);

    chunks
}

#[cfg(test)]
#[path = "handoff_test.rs"]
mod tests;
