//! Drops reasoning from completed turns in an outgoing request.

use tinyinference_llm::message::{ContentBlock, Message};

/// Removes reasoning blocks from every assistant message that precedes the
/// last user message and returns how many blocks it removed.
///
/// A user message that directly follows a tool result is a tool's follow-up
/// content, not a new user turn, so it does not end the current turn.
///
/// The current turn, the messages after the last user message, is left
/// untouched, because a provider that signs thinking (Anthropic) or requires
/// `reasoning_content` back (DeepSeek) validates it across a tool loop. An
/// assistant message left with no content and no tool calls (it held only
/// reasoning) is removed, since providers reject an empty assistant message.
pub fn drop_past_turn_reasoning(messages: &mut Vec<Message>) -> usize {
    // With no user message the whole history is the current turn.
    let Some(current_turn_start) = messages.iter().enumerate().rposition(|(at, message)| {
        matches!(message, Message::User(_))
            && !(at > 0 && matches!(messages[at - 1], Message::Tool(_)))
    }) else {
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

fn is_reasoning(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
    )
}

#[cfg(test)]
#[path = "strip_tests.rs"]
mod test;
