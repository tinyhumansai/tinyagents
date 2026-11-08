//! Replaces old tool results with a one-line note.
//!
//! A tool result is re-sent verbatim on every later call, long after the model
//! stopped needing it. [`expire_tool_results`] rewrites each result older than
//! the policy allows to `[used <tool>: <first 120 chars> …]`, so the stored
//! transcript, and every prompt built from it, stops carrying the body.
//!
//! The rewrite touches only the message content, in place. Everything before
//! the first rewritten message is byte for byte what it was, so a provider's
//! cached prefix stays valid up to it.

use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::tool::ToolCall;

use super::types::{EXPIRY_NOTE_CHARS, ToolResultExpiryPolicy};

/// The tool named in a note when no assistant message made the call.
const UNKNOWN_TOOL: &str = "tool";

/// Rewrites every tool result in `messages` that is older than
/// `policy.max_age_turns` user turns and returns how many it rewrote. Does
/// nothing when the policy is not enabled.
///
/// A result's age is the number of user messages after it. A result is left
/// alone when its tool is in `policy.exempt_tools`, when it is marked
/// `trusted_verbatim`, or when it is already a note, so the rewrite is
/// idempotent.
pub fn expire_tool_results(messages: &mut [Message], policy: &ToolResultExpiryPolicy) -> usize {
    if !policy.enabled {
        return 0;
    }
    let mut expired = 0;
    let mut age = 0usize;
    let calls = call_index(messages);
    // Walk newest to oldest so the age is a running count of user messages.
    for index in (0..messages.len()).rev() {
        if matches!(messages[index], Message::User(_)) {
            age += 1;
            continue;
        }
        if age <= policy.max_age_turns {
            continue;
        }
        let Message::Tool(tool) = &messages[index] else {
            continue;
        };
        if tool.trusted_verbatim {
            continue;
        }
        let body = messages[index].text();
        let call = call_for_result(messages, &calls, index);
        let name = call
            .map(effective_call_name)
            .unwrap_or_else(|| UNKNOWN_TOOL.to_string());
        if is_note(&body, &name) {
            continue;
        }
        // A registered tool that is itself named like a wrapper keeps its own
        // name, and a repaired call runs as its target, so either name exempts.
        if policy
            .exempt_tools
            .iter()
            .any(|exempt| exempt == &name || call.is_some_and(|call| exempt == &call.name))
        {
            continue;
        }
        let note = note_for(&name, &body);
        if let Message::Tool(tool) = &mut messages[index] {
            tool.content = vec![ContentBlock::Text(note)];
            expired += 1;
        }
    }
    if expired > 0 {
        tracing::debug!(
            target: "tinyagents::result_expiry",
            expired,
            max_age_turns = policy.max_age_turns,
            "[result_expiry] replaced old tool results with notes"
        );
    }
    expired
}

/// For every message, where the call that produced it sits: the index of the
/// assistant message and of the call within it. `None` for any message that is
/// not a tool result with a call. Built in one forward pass.
///
/// A result pairs with the nearest earlier assistant message holding a call
/// with its id. When that message repeats the id, the n-th result carrying the
/// id answers the n-th call carrying it (the last call once they run out), so
/// duplicate provider ids cannot point two results at one call.
pub(crate) fn call_index(messages: &[Message]) -> Vec<Option<(usize, usize)>> {
    use std::collections::HashMap;
    // id -> (assistant index, call positions with that id, results seen so far)
    let mut open: HashMap<&str, (usize, Vec<usize>, usize)> = HashMap::new();
    let mut index = vec![None; messages.len()];
    for (at, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant(assistant) => {
                let mut fresh: HashMap<&str, Vec<usize>> = HashMap::new();
                for (position, call) in assistant.tool_calls.iter().enumerate() {
                    fresh.entry(call.id.as_str()).or_default().push(position);
                }
                for (id, positions) in fresh {
                    open.insert(id, (at, positions, 0));
                }
            }
            Message::Tool(result) => {
                if let Some((assistant_at, positions, seen)) =
                    open.get_mut(result.tool_call_id.as_str())
                {
                    let position = positions[(*seen).min(positions.len() - 1)];
                    *seen += 1;
                    index[at] = Some((*assistant_at, position));
                }
            }
            _ => {}
        }
    }
    index
}

/// The assistant call that produced the tool result at `index`, using the
/// pairing [`call_index`] built.
pub(crate) fn call_for_result<'a>(
    messages: &'a [Message],
    calls: &[Option<(usize, usize)>],
    index: usize,
) -> Option<&'a ToolCall> {
    let (assistant_at, position) = (*calls.get(index)?)?;
    match messages.get(assistant_at)? {
        Message::Assistant(assistant) => assistant.tool_calls.get(position),
        _ => None,
    }
}

/// The name of the tool a call ran as: the target a pseudo `tool_call` wrapper
/// resolves to, else the called name.
pub(crate) fn effective_call_name(call: &ToolCall) -> String {
    effective_call(call).0
}

/// The tool name and arguments a call ran with: a pseudo `tool_call` wrapper is
/// read as the call it wraps, any other call as itself.
pub(crate) fn effective_call(call: &ToolCall) -> (String, serde_json::Value) {
    if crate::agent_loop::tool_call_wrapper::is_wrapper_name(&call.name)
        && let Some(inner) =
            crate::agent_loop::tool_call_wrapper::unwrap_wrapped_call(&call.arguments, &|_| true)
    {
        return inner;
    }
    (call.name.clone(), call.arguments.clone())
}

/// `[used <tool>: <first 120 chars of body on one line>]`, with ` …` before the
/// bracket when the body was cut.
fn note_for(tool: &str, body: &str) -> String {
    let tool = bounded_name(tool);
    let flat = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = flat.chars();
    let head: String = chars.by_ref().take(EXPIRY_NOTE_CHARS).collect();
    let cut = if chars.next().is_some() {
        " \u{2026}"
    } else {
        ""
    };
    format!("[used {tool}: {head}{cut}]")
}

/// The longest tool name a note carries, so a model-supplied wrapper target
/// cannot make a note larger than the result it replaces.
const NOTE_NAME_CHARS: usize = 64;

fn bounded_name(tool: &str) -> String {
    tool.chars().take(NOTE_NAME_CHARS).collect()
}

/// Whether `body` is a note this module wrote for `tool`: it opens
/// `[used <tool>: `, is one line and is no longer than a note can be, so a
/// large body that merely starts with `[used ` and ends with `]`, or a tool's
/// own `[used other: ...]` output, is still expired.
fn is_note(body: &str, tool: &str) -> bool {
    let prefix = format!("[used {}: ", bounded_name(tool));
    // prefix + head + ` …]`.
    let longest = prefix.chars().count() + EXPIRY_NOTE_CHARS + 3;
    body.starts_with(&prefix)
        && body.ends_with(']')
        && !body.contains('\n')
        && body.chars().count() <= longest
}

#[cfg(test)]
#[path = "expire_tests.rs"]
mod test;
