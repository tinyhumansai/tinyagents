//! Structural repair of transcript cut points so compaction never orphans a
//! tool call or a tool result.
//!
//! # The failure this prevents
//!
//! Every provider enforces the same structural invariant: a `role:"tool"`
//! message must be preceded by the assistant message whose `tool_calls`
//! declared its id, and an assistant `tool_calls` entry must be answered. Cut a
//! transcript at a blind index and you routinely break both halves:
//!
//! ```text
//! [system, user, assistant(tool_calls=[c1]), tool(c1), assistant("done")]
//!                                      ^ keep_last = 2 cuts here
//! ```
//!
//! The rebuilt request opens with `tool(c1)` and no preceding `tool_calls`.
//! OpenAI answers `400`; Anthropic rejects a `tool_result` with no matching
//! `tool_use`. This fires only on long tool-driven runs — which is to say, only
//! on the runs that ever reach a compaction threshold, and only after the run
//! has already done expensive work.
//!
//! # The two repairs
//!
//! Which direction is correct depends on what the cut is *for*:
//!
//! - [`find_safe_cutoff_point`] moves the cut **backward** to swallow the
//!   assistant message that owns the orphaned results. Use it when the goal is
//!   a message *count* ("keep the last N") or a summarize/keep split, where
//!   keeping one extra message is free. This is a port of LangChain's
//!   `SummarizationMiddleware._find_safe_cutoff_point`.
//! - [`advance_past_orphan_tools`] moves the cut **forward**, discarding the
//!   orphaned tool results instead. Use it when the cut enforces a *token
//!   budget*, where moving backward would re-admit the very tokens the trim
//!   was trying to shed.
//! - [`retract_orphan_tool_calls`] repairs the *other* end: an assistant
//!   message left at the tail of a retained prefix whose results were cut is an
//!   orphaned tool call, and just as fatal. LangChain's summarization
//!   middleware never hits this because it only ever keeps a suffix;
//!   [`TrimStrategy::KeepFirstAndLast`][crate::harness::summarization::TrimStrategy::KeepFirstAndLast]
//!   does keep a prefix, so the crate needs the mirrored repair.
//!
//! All three take and return indices into a slice that contains **no system
//! messages** (callers partition those out first); a system message can never
//! sit between an assistant and its tool results, so the partitioning does not
//! affect pairing.

use std::collections::HashSet;

use crate::harness::message::Message;

/// Returns the tool-call ids declared by an assistant message, or an empty set
/// for every other message kind.
fn declared_call_ids(message: &Message) -> HashSet<&str> {
    match message {
        Message::Assistant(assistant) => assistant
            .tool_calls
            .iter()
            .map(|call| call.id.as_str())
            .filter(|id| !id.is_empty())
            .collect(),
        _ => HashSet::new(),
    }
}

/// Returns `true` when `message` is an assistant turn that requested tools.
pub fn is_tool_calling_assistant(message: &Message) -> bool {
    matches!(message, Message::Assistant(a) if !a.tool_calls.is_empty())
}

/// Moves `cutoff_index` to a point that does not split an assistant tool-call
/// message from the tool results answering it, preferring to keep *more*.
///
/// `cutoff_index` is the index at which the retained suffix begins (everything
/// before it is dropped or summarized). Semantics, in order:
///
/// 1. If the message at the cutoff is not a tool result, the cutoff is already
///    safe and is returned unchanged.
/// 2. Otherwise the consecutive run of tool results at the cutoff is collected
///    and the slice is scanned **backward** for the assistant message whose
///    `tool_calls` ids intersect that run; the cutoff moves back to that
///    assistant's index so the pair stays together.
/// 3. If no such assistant exists (a truncated or imported transcript), the
///    cutoff falls **forward** past the whole tool run, discarding results that
///    can never be paired.
///
/// Port of LangChain's `SummarizationMiddleware._find_safe_cutoff_point`.
pub fn find_safe_cutoff_point(messages: &[Message], cutoff_index: usize) -> usize {
    if cutoff_index >= messages.len() || !matches!(messages[cutoff_index], Message::Tool(_)) {
        return cutoff_index;
    }

    // Collect the ids of the consecutive tool-result run starting at the cutoff.
    let mut orphan_ids: HashSet<&str> = HashSet::new();
    let mut past_run = cutoff_index;
    while past_run < messages.len()
        && let Message::Tool(tool) = &messages[past_run]
    {
        if !tool.tool_call_id.is_empty() {
            orphan_ids.insert(tool.tool_call_id.as_str());
        }
        past_run += 1;
    }

    // Scan backward for the assistant turn that declared any of those ids.
    for index in (0..cutoff_index).rev() {
        let declared = declared_call_ids(&messages[index]);
        if !declared.is_empty() && declared.intersection(&orphan_ids).next().is_some() {
            tracing::debug!(
                "[summarization::pairing] cutoff {cutoff_index} split a tool pair; moving back to {index} to keep the assistant tool-call turn"
            );
            return index;
        }
    }

    tracing::debug!(
        "[summarization::pairing] cutoff {cutoff_index} has no matching assistant tool-call turn; advancing to {past_run} to drop unpairable tool results"
    );
    past_run
}

/// Moves `cutoff_index` **forward** past any leading tool results whose
/// assistant tool-call turn was dropped, discarding them.
///
/// This is the budget-preserving counterpart to [`find_safe_cutoff_point`]:
/// where that function keeps more to repair the pair, this one keeps less, so a
/// token-bounded trim cannot re-admit the tokens it just shed. A tool result at
/// the head of the retained slice is orphaned by definition — its assistant
/// turn necessarily preceded it and was therefore already dropped.
pub fn advance_past_orphan_tools(messages: &[Message], cutoff_index: usize) -> usize {
    let mut index = cutoff_index;
    while index < messages.len() && matches!(messages[index], Message::Tool(_)) {
        index += 1;
    }
    if index != cutoff_index {
        tracing::debug!(
            "[summarization::pairing] dropped {} leading orphan tool result(s) at cutoff {cutoff_index}",
            index - cutoff_index
        );
    }
    index
}

/// Pulls an exclusive `end_index` **backward** so the retained prefix does not
/// end on an assistant message whose tool results were cut.
///
/// An assistant `tool_calls` entry with no answering tool message is as fatal
/// as the reverse orphan: OpenAI rejects the request, and Anthropic rejects a
/// `tool_use` with no `tool_result`. Because the results always follow their
/// call, an assistant tool-call turn sitting at the very end of a kept prefix
/// is unanswerable by construction, so it is removed along with any tool-call
/// turns it uncovers.
pub fn retract_orphan_tool_calls(messages: &[Message], end_index: usize) -> usize {
    let mut end = end_index.min(messages.len());
    while end > 0 && is_tool_calling_assistant(&messages[end - 1]) {
        end -= 1;
    }
    if end != end_index.min(messages.len()) {
        tracing::debug!(
            "[summarization::pairing] retracted retained prefix end from {end_index} to {end} to drop unanswered assistant tool call(s)"
        );
    }
    end
}

/// Returns `true` when `messages` satisfies the provider pairing invariant:
/// every tool result is preceded by an assistant turn declaring its id, and
/// every declared tool call is answered later in the slice.
///
/// Exposed for tests and for hosts that want to assert the invariant before
/// sending a request they assembled themselves.
pub fn tool_pairing_is_intact(messages: &[Message]) -> bool {
    let mut declared: HashSet<&str> = HashSet::new();
    let mut answered: HashSet<&str> = HashSet::new();

    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                for call in &assistant.tool_calls {
                    declared.insert(call.id.as_str());
                }
            }
            Message::Tool(tool) => {
                if !declared.contains(tool.tool_call_id.as_str()) {
                    return false;
                }
                answered.insert(tool.tool_call_id.as_str());
            }
            _ => {}
        }
    }

    declared.is_subset(&answered)
}
