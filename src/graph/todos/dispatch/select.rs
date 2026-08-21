//! Which card runs next, whether it needs approval first, and how often to
//! look — the scheduling policy of a task board, as pure functions.
//!
//! None of this touches a [`Store`](crate::harness::store::Store): a caller
//! reads a board snapshot, asks these functions what to do, and performs the
//! claim itself. That keeps the policy trivially testable and lets a host apply
//! it to boards it holds in memory.

use std::time::Duration;

use crate::graph::todos::types::{TaskApprovalMode, TaskBoardCard, TaskCardStatus};

/// A card's urgency, read from `source_metadata.urgency`. Cards without one
/// sort as `0.0`.
pub fn card_urgency(card: &TaskBoardCard) -> f64 {
    card.source_metadata
        .as_ref()
        .and_then(|meta| meta.get("urgency"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
}

/// Whether the board already has a card being worked.
///
/// The board store caps a board at one `InProgress` card, so this answers
/// "is there anything for a dispatcher to claim right now?".
pub fn has_card_in_progress(cards: &[TaskBoardCard]) -> bool {
    cards
        .iter()
        .any(|card| card.status == TaskCardStatus::InProgress)
}

/// The highest-urgency dispatchable card, or `None` when the board has none.
///
/// Dispatchable means `Todo` (not yet triaged) or `Ready` (approved). Ties on
/// urgency break toward the lower board `order`, so equal-priority work runs in
/// the order it was planned.
///
/// `agent_assigned_only` restricts the pick to cards with an `assigned_agent`.
/// A host uses it for boards that mix human-authored and agent-authored cards,
/// so an autonomous sweep never picks up a card a person wrote for themselves.
pub fn pick_next_card(
    cards: &[TaskBoardCard],
    agent_assigned_only: bool,
) -> Option<TaskBoardCard> {
    cards
        .iter()
        .filter(|card| matches!(card.status, TaskCardStatus::Todo | TaskCardStatus::Ready))
        .filter(|card| !agent_assigned_only || is_agent_assigned(card))
        .max_by(|a, b| {
            card_urgency(a)
                .partial_cmp(&card_urgency(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                // Reversed, so the *lower* order wins an urgency tie.
                .then(b.order.cmp(&a.order))
        })
        .cloned()
}

fn is_agent_assigned(card: &TaskBoardCard) -> bool {
    card.assigned_agent
        .as_deref()
        .is_some_and(|agent| !agent.trim().is_empty())
}

/// Whether a card must be parked at
/// [`AwaitingApproval`](TaskCardStatus::AwaitingApproval) before it runs.
///
/// The card's own [`TaskApprovalMode`] is authoritative when set; `global_required`
/// is only the fallback for cards that express no preference:
///
/// - [`Required`](TaskApprovalMode::Required) → always park, *even when the
///   global default is off*. A card stamped by an interactive plan review must
///   still be reviewed, or the plan would execute before anyone saw it.
/// - [`NotRequired`](TaskApprovalMode::NotRequired) → never park; the card has
///   already cleared review.
/// - unset → `global_required`.
pub fn requires_plan_approval(
    global_required: bool,
    approval_mode: Option<&TaskApprovalMode>,
) -> bool {
    match approval_mode {
        Some(TaskApprovalMode::Required) => true,
        Some(TaskApprovalMode::NotRequired) => false,
        None => global_required,
    }
}

/// Diminishing-returns polling cadence for a board sweep.
///
/// A dispatcher that sweeps a board on a timer should not keep sweeping an idle
/// board at full rate forever. [`PollCadence::next_delay`] holds the base
/// interval while there is work (and for a short grace period after it dries
/// up), then doubles per idle tick up to a ceiling — an effective self-suspend
/// that still rechecks often enough to pick up newly-arrived work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollCadence {
    /// Interval used while the board has work.
    pub base: Duration,
    /// Ceiling the backoff saturates at.
    pub max_backoff: Duration,
    /// Consecutive idle ticks tolerated at `base` before backing off, so a
    /// briefly-empty board does not immediately slow down.
    pub grace_ticks: u32,
}

impl Default for PollCadence {
    fn default() -> Self {
        Self {
            base: Duration::from_secs(60),
            max_backoff: Duration::from_secs(15 * 60),
            grace_ticks: 2,
        }
    }
}

impl PollCadence {
    /// How long to wait before the next sweep, given the number of consecutive
    /// idle ticks so far (`0` right after a dispatch).
    ///
    /// Monotonic non-decreasing in `idle_ticks`, never above `max_backoff`, and
    /// overflow-free for any `u32` streak.
    pub fn next_delay(&self, idle_ticks: u32) -> Duration {
        let over = idle_ticks.saturating_sub(self.grace_ticks);
        if over == 0 {
            return self.base;
        }
        // Doubling per idle tick past the grace window. The shift is clamped so
        // a long streak saturates instead of wrapping to a tiny delay.
        let factor = 1u64.checked_shl(over.min(20)).unwrap_or(u64::MAX);
        let secs = self
            .base
            .as_secs()
            .saturating_mul(factor)
            .min(self.max_backoff.as_secs());
        Duration::from_secs(secs)
    }
}
