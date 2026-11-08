//! Shrinks a finished run's `tool_search` results to what the run used.
//!
//! A search answer carries the full schema of every match, and it stays in the
//! history, re-sent on every later call. Most searches then use one match, so
//! once the run ends [`collapse_search_results`] rewrites each answer to the
//! query, the matches the run went on to call, and only their schemas. A later
//! turn can still call a kept tool directly: admission and argument validation
//! read the registered tool, not this text.
//!
//! The rewrite touches one message in place. Everything before it is byte for
//! byte what it was, so a provider's cached prefix stays valid up to that
//! message and only the messages after it are read again, once.

use std::collections::BTreeMap;

use serde_json::Value;
use tinyinference_llm::message::{ContentBlock, Message};

use super::bridge::{TOOL_SEARCH_NAME, parse_search_hits, search_query};
use super::types::PriorSearchResults;

impl PriorSearchResults {
    /// The search results `messages` already hold.
    #[must_use]
    pub fn of(messages: &[Message]) -> Self {
        let mut counts = BTreeMap::new();
        for message in messages {
            if let Some((key, _)) = parsed_answer(message) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        Self(counts)
    }
}

/// Rewrites every un-collapsed `tool_search` answer in `messages` except those
/// in `prior`, and returns how many it rewrote.
///
/// An answer becomes `tool_search "<query>" → used <names>` followed by the
/// compact JSON of the used tools' hits, or, when no match was called after
/// it, the one line `tool_search "<query>" → none used; matched <names>`. A
/// match counts as used when a later assistant message calls it by name.
///
/// Idempotent: a collapsed answer no longer parses as a search answer. Anything
/// that is not an answer the bridge rendered is left alone: a no-match line, an
/// error, a host's own `tool_search`, a result marked verbatim, or one with no
/// preceding call to read the query from.
pub fn collapse_search_results(messages: &mut [Message], prior: &PriorSearchResults) -> usize {
    let mut unclaimed = prior.0.clone();
    let mut rewritten = 0;
    for index in 0..messages.len() {
        let Some(collapsed) = collapsed_text(messages, index, &mut unclaimed) else {
            continue;
        };
        if let Message::Tool(tool) = &mut messages[index] {
            tool.content = vec![ContentBlock::Text(collapsed)];
            rewritten += 1;
        }
    }
    rewritten
}

/// A tool message that holds an un-collapsed search answer: its
/// `(tool_call_id, text)` key and the parsed hits.
fn parsed_answer(message: &Message) -> Option<((String, String), Vec<Value>)> {
    let Message::Tool(tool) = message else {
        return None;
    };
    let text = plain_text(&tool.content)?;
    let hits = parse_search_hits(&text)?;
    Some(((tool.tool_call_id.clone(), text), hits))
}

fn collapsed_text(
    messages: &[Message],
    index: usize,
    unclaimed: &mut BTreeMap<(String, String), usize>,
) -> Option<String> {
    let (key, hits) = parsed_answer(&messages[index])?;
    // Claim one prior occurrence before anything else can skip this message,
    // so the counts stay aligned with the order `PriorSearchResults::of` saw.
    if let Some(left) = unclaimed.get_mut(&key).filter(|left| **left > 0) {
        *left -= 1;
        return None;
    }
    let Message::Tool(tool) = &messages[index] else {
        return None;
    };
    if tool.trusted_verbatim {
        return None;
    }
    let call = crate::result_expiry::call_for_result(messages, index)?;
    // A search wrapped in a pseudo `tool_call` is read as the search it ran.
    let (name, arguments) = crate::result_expiry::effective_call(call);
    if name != TOOL_SEARCH_NAME {
        return None;
    }
    let query = serde_json::to_string(search_query(&arguments)).ok()?;

    let called = names_called(&messages[index + 1..]);
    let used: Vec<&Value> = hits
        .iter()
        .filter(|hit| {
            hit["name"]
                .as_str()
                .is_some_and(|name| called.iter().any(|called| called == name))
        })
        .collect();
    if used.is_empty() {
        let matched = hit_names(hits.iter());
        return Some(format!(
            "{TOOL_SEARCH_NAME} {query} \u{2192} none used; matched {matched}"
        ));
    }
    let payload = crate::canonical_json::to_string(&Value::Array(
        used.iter().map(|hit| (*hit).clone()).collect(),
    ))
    .ok()?;
    Some(format!(
        "{TOOL_SEARCH_NAME} {query} \u{2192} used {}\n{payload}",
        hit_names(used.into_iter())
    ))
}

/// Every tool name the assistant messages in `messages` called, with a call
/// wrapped in a pseudo `tool_call` counted under the tool it was repaired to.
fn names_called(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant.tool_calls.iter()),
            _ => None,
        })
        .flatten()
        .flat_map(|call| {
            let direct = call.name.clone();
            let repaired = crate::result_expiry::effective_call_name(call);
            [
                Some(direct.clone()),
                (repaired != direct).then_some(repaired),
            ]
        })
        .flatten()
        .collect()
}

fn hit_names<'a>(hits: impl Iterator<Item = &'a Value>) -> String {
    hits.filter_map(|hit| hit["name"].as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The text of `content` when every block is plain text.
fn plain_text(content: &[ContentBlock]) -> Option<String> {
    content
        .iter()
        .map(ContentBlock::as_text)
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.concat())
}

#[cfg(test)]
#[path = "collapse_tests.rs"]
mod test;
