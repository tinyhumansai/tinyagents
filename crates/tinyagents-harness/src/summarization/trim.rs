//! Synchronous, LLM-free transcript trimming.
//!
//! Every strategy here is **pairing-safe by default**: the cut point produced by
//! the strategy is repaired by [`super::pairing`] before the slice is rebuilt,
//! so a trimmed transcript never opens on an orphaned tool result or ends a
//! retained prefix on an unanswered assistant tool call. See
//! [`TrimOptions::repair_tool_pairs`] for the escape hatch and why you almost
//! certainly do not want it.
//!
//! Two mechanisms are provided, mirroring the two LangChain has:
//!
//! - **structural repair** — the pairing scan described above, which is
//!   automatic (LangChain implements this only inside its summarization
//!   middleware, not in `trim_messages`);
//! - **role boundaries** — [`TrimOptions::start_on`] / [`TrimOptions::end_on`],
//!   the caller-driven predicates LangChain core's `trim_messages` exposes,
//!   whose docstring makes honouring provider pairing rules the caller's job.

use super::pairing::{
    advance_past_orphan_tools, find_safe_cutoff_point, retract_orphan_tool_calls,
};
use super::types::{MessageRole, TokenTrimPolicy, TrimOptions, TrimStrategy};
use tinyinference::message::{Message, estimate_message_tokens};

/// Partition `messages` into system and non-system messages, preserving order.
///
/// Returns `(system, non_system)`. Pairing repair operates on the non-system
/// half alone: a system message can never sit between an assistant tool-call
/// turn and the tool results answering it, so removing them from consideration
/// cannot change a pairing decision.
pub(super) fn partition_system(messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
    let system = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .cloned()
        .collect();
    let non_system = messages
        .iter()
        .filter(|m| !matches!(m, Message::System(_)))
        .cloned()
        .collect();
    (system, non_system)
}

/// Trim a message slice according to `strategy`, returning the retained subset.
///
/// System messages are preserved by default:
///
/// - [`TrimStrategy::KeepLast`] and [`TrimStrategy::KeepFirstAndLast`] always
///   keep all system messages and apply the rule only to non-system messages.
/// - [`TrimStrategy::MaxTokens`] drops non-system messages first (from the
///   front) and only starts dropping system messages if the budget still
///   cannot be met after all non-system messages are removed.
///
/// Tool-call pairing is repaired automatically — see [`trim_messages_with`] for
/// the details and for the role-boundary knobs. The returned `Vec<Message>`
/// preserves the relative order of messages as they appeared in the input.
pub fn trim_messages(messages: &[Message], strategy: &TrimStrategy) -> Vec<Message> {
    trim_messages_with(messages, strategy, &TrimOptions::default())
}

/// Trim a message slice according to `strategy` under explicit [`TrimOptions`].
///
/// # Pairing repair
///
/// With [`TrimOptions::repair_tool_pairs`] set (the default), each strategy's
/// raw cut point is adjusted so the result is a slice a provider will accept:
///
/// | Strategy | Repair |
/// | -------- | ------ |
/// | [`TrimStrategy::KeepLast`] | cut moves **backward** to include the assistant turn owning any leading tool results |
/// | [`TrimStrategy::KeepFirstAndLast`] | suffix cut moves backward as above; the retained prefix's end moves backward past any unanswered assistant tool call |
/// | [`TrimStrategy::MaxTokens`] | cut moves **forward**, dropping orphaned leading tool results — moving backward would re-admit the tokens the budget was shedding |
///
/// # Role boundaries
///
/// [`TrimOptions::start_on`] and [`TrimOptions::end_on`] drop further messages
/// from the front / back until the retained slice begins / ends on one of the
/// listed roles, matching LangChain core's `trim_messages(start_on=…,
/// end_on=…)`. They are applied *after* the strategy, and pairing repair runs
/// once more afterwards so a role boundary cannot reintroduce an orphan.
pub fn trim_messages_with(
    messages: &[Message],
    strategy: &TrimStrategy,
    options: &TrimOptions,
) -> Vec<Message> {
    let (system, non_system) = partition_system(messages);
    let (retained_system, mut retained) = apply_strategy(&system, &non_system, strategy, options);
    apply_role_boundaries(&mut retained, options);

    if options.repair_tool_pairs {
        let start = advance_past_orphan_tools(&retained, 0);
        if start > 0 {
            retained.drain(..start);
        }
    }

    let mut result = retained_system;
    result.extend(retained);
    tinyagents_tracing::debug!(
        "[summarization::trim] strategy={strategy:?} input={} retained={}",
        messages.len(),
        result.len()
    );
    result
}

/// Applies `strategy`, returning `(retained_system, retained_non_system)`.
///
/// System messages are returned separately (rather than re-attached here)
/// because [`TrimStrategy::MaxTokens`] is the one strategy allowed to shed
/// them, and only after every other message is gone.
fn apply_strategy(
    system: &[Message],
    non_system: &[Message],
    strategy: &TrimStrategy,
    options: &TrimOptions,
) -> (Vec<Message>, Vec<Message>) {
    match strategy {
        TrimStrategy::KeepLast(n) => {
            let mut keep_start = non_system.len().saturating_sub(*n);
            if options.repair_tool_pairs {
                keep_start = find_safe_cutoff_point(non_system, keep_start);
            }
            (system.to_vec(), non_system[keep_start..].to_vec())
        }

        TrimStrategy::KeepFirstAndLast { first, last } => {
            let len = non_system.len();
            let (first, last) = (*first, *last);

            if first + last >= len {
                // No room to drop anything: keep every non-system message.
                return (system.to_vec(), non_system.to_vec());
            }

            let mut prefix_end = first;
            let mut suffix_start = len - last;
            if options.repair_tool_pairs {
                prefix_end = retract_orphan_tool_calls(non_system, prefix_end);
                suffix_start = find_safe_cutoff_point(non_system, suffix_start);
                // Backward repair of the suffix can reach into (or past) the
                // retained prefix. Clamping keeps the two ranges contiguous and
                // non-overlapping, which in that case means keeping everything.
                suffix_start = suffix_start.max(prefix_end);
            }

            let mut result = non_system[..prefix_end].to_vec();
            result.extend_from_slice(&non_system[suffix_start..]);
            (system.to_vec(), result)
        }

        TrimStrategy::MaxTokens(limit) => {
            let limit = *limit;

            // Precompute each message's token estimate once: re-summing the
            // slice per dropped message (and using `remove(0)`) would make this
            // O(n^2).
            let sys_tokens: Vec<u64> = system.iter().map(estimate_message_tokens).collect();
            let non_sys_tokens: Vec<u64> = non_system.iter().map(estimate_message_tokens).collect();
            let sys_total: u64 = sys_tokens.iter().sum();

            let mut non_sys_start = 0;
            let mut non_sys_total: u64 = non_sys_tokens.iter().sum();
            while non_sys_start < non_system.len() && sys_total + non_sys_total > limit {
                non_sys_total -= non_sys_tokens[non_sys_start];
                non_sys_start += 1;
            }

            if options.repair_tool_pairs {
                // Forward-only: a budget-bound trim must not grow again.
                non_sys_start = advance_past_orphan_tools(non_system, non_sys_start);
            }

            // Still over budget with every non-system message gone: shed
            // system instructions from the front as a last resort.
            let mut sys_start = 0;
            let mut sys_running = sys_total;
            while sys_start < system.len() && sys_running + non_sys_total > limit {
                sys_running -= sys_tokens[sys_start];
                sys_start += 1;
            }

            (
                system[sys_start..].to_vec(),
                non_system[non_sys_start..].to_vec(),
            )
        }
    }
}

/// Drops messages from either end until the slice begins / ends on an allowed
/// role, per [`TrimOptions::start_on`] / [`TrimOptions::end_on`].
fn apply_role_boundaries(retained: &mut Vec<Message>, options: &TrimOptions) {
    if let Some(end_on) = options.end_on.as_ref()
        && !end_on.is_empty()
    {
        while let Some(last) = retained.last() {
            if end_on.contains(&MessageRole::of(last)) {
                break;
            }
            retained.pop();
        }
        if options.repair_tool_pairs {
            let end = retract_orphan_tool_calls(retained, retained.len());
            retained.truncate(end);
        }
    }

    if let Some(start_on) = options.start_on.as_ref()
        && !start_on.is_empty()
    {
        let mut start = 0;
        while start < retained.len() && !start_on.contains(&MessageRole::of(&retained[start])) {
            start += 1;
        }
        retained.drain(..start);
    }
}

/// Trim messages to a token budget while preserving their original order.
///
/// `estimate` lets callers account for provider-specific payloads without
/// teaching the harness about their wire representation. The built-in
/// [`Message::estimated_char_weight`] already assigns a flat weight to native
/// image blocks; hosts may additionally recognize inline image markers or use a
/// real tokenizer. Oldest non-system messages are evicted first. System
/// messages are either retained unconditionally or evicted oldest-first only
/// after all other messages, according to [`TokenTrimPolicy::preserve_system`].
///
/// When `drop_leading_orphan_tools` is enabled, leading tool-result messages are
/// removed after budget eviction so the retained transcript starts on a valid
/// provider turn boundary. The returned messages always retain their original
/// relative order.
pub fn trim_messages_to_token_budget_with(
    messages: &[Message],
    policy: TokenTrimPolicy,
    estimate: impl Fn(&Message) -> u64,
) -> Vec<Message> {
    let estimates: Vec<u64> = messages.iter().map(estimate).collect();
    let mut total: u64 = estimates.iter().copied().sum();
    if total <= policy.limit && !policy.drop_leading_orphan_tools {
        return messages.to_vec();
    }

    let mut retained = vec![true; messages.len()];
    for (index, message) in messages.iter().enumerate() {
        if total <= policy.limit {
            break;
        }
        if !matches!(message, Message::System(_)) {
            retained[index] = false;
            total = total.saturating_sub(estimates[index]);
        }
    }

    if !policy.preserve_system && total > policy.limit {
        for (index, message) in messages.iter().enumerate() {
            if total <= policy.limit {
                break;
            }
            if matches!(message, Message::System(_)) && retained[index] {
                retained[index] = false;
                total = total.saturating_sub(estimates[index]);
            }
        }
    }

    let mut result: Vec<Message> = messages
        .iter()
        .zip(retained)
        .filter(|(_, keep)| *keep)
        .map(|(message, _)| message.clone())
        .collect();

    if policy.drop_leading_orphan_tools {
        while let Some(index) = result
            .iter()
            .position(|message| !matches!(message, Message::System(_)))
        {
            if matches!(result[index], Message::Tool(_)) {
                result.remove(index);
            } else {
                break;
            }
        }
    }
    result
}
