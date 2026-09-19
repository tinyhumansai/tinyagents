//! Legacy HTML-comment `.md` transcript reader (one-release migration
//! compat for sessions written before the JSONL source of truth).

use super::types::TranscriptMessage;
use super::types::{SessionTranscript, TranscriptMeta};
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Read a legacy HTML-comment `.md` transcript. Used as a fallback when
/// only a `.md` exists (no `.jsonl` sibling).
///
/// Returns a `SessionTranscript` with whatever fields the `.md` tracked;
/// fields the old format didn't carry are defaulted.
pub fn read_transcript_legacy_md(path: &Path) -> Result<SessionTranscript> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read legacy transcript {}", path.display()))?;

    let meta = parse_legacy_meta(&raw)
        .with_context(|| format!("parse legacy transcript meta in {}", path.display()))?;

    let messages = parse_legacy_messages(&raw)
        .with_context(|| format!("parse legacy transcript messages in {}", path.display()))?;

    tracing::debug!(
        "[transcript] loaded {} messages (legacy md) from {}",
        messages.len(),
        path.display()
    );

    Ok(SessionTranscript { meta, messages })
}

const LEGACY_MSG_OPEN_PREFIX: &str = "<!--MSG role=\"";
const LEGACY_MSG_OPEN_SUFFIX: &str = "\"-->";
const LEGACY_MSG_CLOSE: &str = "<!--/MSG-->";
const LEGACY_MSG_CLOSE_ESCAPED: &str = "<!--\\/MSG-->";

fn parse_legacy_meta(raw: &str) -> Result<TranscriptMeta> {
    let header_start = raw
        .find("<!-- session_transcript")
        .context("missing session_transcript header")?;
    let header_end = raw[header_start..]
        .find("-->")
        .context("unclosed session_transcript header")?;
    let header = &raw[header_start..header_start + header_end + 3];

    let get = |key: &str| -> Option<String> {
        header.lines().find_map(|line| {
            let line = line.trim();
            if line.starts_with(&format!("{key}:")) {
                Some(line[key.len() + 1..].trim().to_string())
            } else {
                None
            }
        })
    };

    Ok(TranscriptMeta {
        agent_name: get("agent").unwrap_or_else(|| "unknown".into()),
        dispatcher: get("dispatcher").unwrap_or_else(|| "native".into()),
        agent_id: None,
        agent_type: None,
        provider: None,
        model: None,
        created: get("created").unwrap_or_default(),
        updated: get("updated").unwrap_or_default(),
        turn_count: get("turn_count").and_then(|s| s.parse().ok()).unwrap_or(0),
        input_tokens: get("input_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        output_tokens: get("output_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        cached_input_tokens: get("cached_input_tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        charged_amount_usd: get("charged_usd")
            .and_then(|s| s.trim_start_matches('$').parse().ok())
            .unwrap_or(0.0),
        thread_id: get("thread_id").filter(|s| !s.is_empty()),
        task_id: None,
    })
}

fn parse_legacy_messages(raw: &str) -> Result<Vec<TranscriptMessage>> {
    let mut messages = Vec::new();
    let mut search_from = 0;

    while let Some(open_start) = raw[search_from..].find(LEGACY_MSG_OPEN_PREFIX) {
        let open_start = search_from + open_start;
        let after_prefix = open_start + LEGACY_MSG_OPEN_PREFIX.len();

        let Some(role_end) = raw[after_prefix..].find(LEGACY_MSG_OPEN_SUFFIX) else {
            break;
        };
        let role = raw[after_prefix..after_prefix + role_end].to_string();

        let content_start = after_prefix + role_end + LEGACY_MSG_OPEN_SUFFIX.len();
        let content_start = if raw[content_start..].starts_with('\n') {
            content_start + 1
        } else {
            content_start
        };

        let close_tag = format!("\n{LEGACY_MSG_CLOSE}");
        let Some(content_end_rel) = raw[content_start..].find(&close_tag) else {
            let Some(content_end_rel) = raw[content_start..].find(LEGACY_MSG_CLOSE) else {
                break;
            };
            let content = &raw[content_start..content_start + content_end_rel];
            messages.push(TranscriptMessage {
                id: None,
                role,
                content: content.replace(LEGACY_MSG_CLOSE_ESCAPED, LEGACY_MSG_CLOSE),
                extra_metadata: None,
                cache_breakpoints: Vec::new(),
                turn_usage: None,
                request_id: None,
                preserve_request_id: false,
                interrupted: false,
                tool_failure: None,
            });
            search_from = content_start + content_end_rel + LEGACY_MSG_CLOSE.len();
            continue;
        };

        let content = &raw[content_start..content_start + content_end_rel];
        messages.push(TranscriptMessage {
            id: None,
            role,
            content: content.replace(LEGACY_MSG_CLOSE_ESCAPED, LEGACY_MSG_CLOSE),
            extra_metadata: None,
            cache_breakpoints: Vec::new(),
            turn_usage: None,
            request_id: None,
            preserve_request_id: false,
            interrupted: false,
            tool_failure: None,
        });

        search_from = content_start + content_end_rel + close_tag.len();
    }

    Ok(messages)
}
