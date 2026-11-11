//! Drops reasoning from completed turns in an outgoing request.

use tinyinference_llm::message::{ContentBlock, Message};

/// Removes reasoning blocks from every assistant message that precedes the
/// last user message and returns how many blocks it removed.
///
/// A user message in a run of user messages that directly follows a tool
/// result is a tool's follow-up content (a batch can append several), not a new
/// user turn, so it does not end the current turn.
///
/// Known limit: a user prompt that directly follows a tool row (a turn a
/// terminating tool ended without an assistant message) is read as a follow-up
/// too. Messages carry no provenance to tell them apart, and wrongly dropping
/// an active loop's signed reasoning fails the request, while keeping a stale
/// turn's reasoning only costs tokens, so the check errs toward keeping.
///
/// The current turn, the messages after the last user message, is left
/// untouched, because a provider that signs thinking (Anthropic) or requires
/// `reasoning_content` back (DeepSeek) validates it across a tool loop. An
/// assistant message left with no content and no tool calls (it held only
/// reasoning) is removed, since providers reject an empty assistant message.
pub fn drop_past_turn_reasoning(messages: &mut Vec<Message>) -> usize {
    // With no user message the whole history is the current turn.
    let Some(current_turn_start) = current_turn_start(messages) else {
        return 0;
    };
    let mut dropped = 0;
    let mut index = 0;
    messages.retain_mut(|message| {
        let in_past_turn = index < current_turn_start;
        index += 1;
        let Message::Assistant(assistant) = message else {
            return true;
        };
        if !in_past_turn {
            return true;
        }
        let before = assistant.content.len();
        assistant.content.retain(|block| !is_reasoning(block));
        let removed = before - assistant.content.len();
        dropped += removed;
        // Only a message this pass emptied goes; one that was already empty
        // is not ours to remove.
        removed == 0 || !assistant.content.is_empty() || !assistant.tool_calls.is_empty()
    });
    if dropped > 0 {
        tracing::debug!(
            target: "tinyagents::reasoning_replay",
            dropped,
            "[reasoning_replay] omitted completed turns' reasoning from the request"
        );
    }
    dropped
}

/// The index of the last user message that starts a turn, found in one
/// reverse pass. A run of user messages directly after a tool result is the
/// tool's follow-ups, so the whole run is skipped; any other run ends at the
/// turn's first message.
fn current_turn_start(messages: &[Message]) -> Option<usize> {
    let mut end = messages.len();
    while end > 0 {
        if !matches!(messages[end - 1], Message::User(_)) {
            end -= 1;
            continue;
        }
        let last = end - 1;
        let mut start = last;
        while start > 0 && matches!(messages[start - 1], Message::User(_)) {
            start -= 1;
        }
        if start > 0 && matches!(messages[start - 1], Message::Tool(_)) {
            end = start;
        } else {
            return Some(last);
        }
    }
    None
}

fn is_reasoning(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
    )
}

#[cfg(test)]
#[path = "strip_tests.rs"]
mod test;
