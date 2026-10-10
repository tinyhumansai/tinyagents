//! Per-turn recovery bookkeeping for the superstep loop.
//!
//! One logical turn can be re-issued several times before the model produces a
//! usable reply: a length-truncated empty reply is retried with a larger output
//! cap, a dropped or withheld tool call is re-prompted, an empty completion is
//! retried. Each of those recoveries is bounded by a counter on
//! [`crate::runtime::RunPolicy`], and the counters (plus the boosted output cap)
//! must be cleared the moment the turn *resolves* so a spent budget or a stale
//! cap never leaks into a later, unrelated turn.
//!
//! [`TurnRecovery`] owns all of that state and exposes one named reset per
//! kind of turn boundary, replacing the field-by-field resets that used to be
//! repeated at each exit of the loop body.

use super::types::TurnRecovery;

/// Share of the remaining wall clock a truncated-empty retry may take. A
/// retry that would run longer than this leaves no time to act on whatever
/// it returns, so the loop nudges instead.
pub(super) const TRUNCATED_RETRY_CLOCK_SHARE: f64 = 0.5;

/// Lowest output cap a repeated nudge drives the call down to.
pub(super) const TRUNCATED_NUDGE_CAP_FLOOR: u32 = 2048;

/// Most nudges one turn may spend on a model that keeps dying at the cap,
/// however much clock is left: beyond this the run is better closed than
/// prolonged.
pub(super) const TRUNCATED_CLOCK_NUDGE_LIMIT: u32 = 6;

/// Whether retrying a call that died at its output cap with nothing to show
/// can help, judged from that call's own rate and the run's clock.
///
/// Two patterns of dead call exist and both have to work. Most are a long
/// think the model does not repeat: in one run five of seven dead calls were
/// followed by a live call of about 3k tokens, so the doubled cap was never
/// needed, and a cap raised for them made every later dead call cost two to
/// four times as long. Some are a step that genuinely no longer fits the cap
/// (23k tokens after a 16k death). So the first retry re-sends at the same
/// cap, growth waits for a second death at that cap, and a retry that would
/// run past half the remaining clock, or that cannot grow the cap at all, is
/// not made: the loop nudges instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TruncatedRetryPlan {
    /// The cap the dead call ran under.
    pub(super) current: Option<u32>,
    /// The cap a retry would run under.
    pub(super) next: Option<u32>,
    /// The cap the first call of the turn ran under: the floor for any cap
    /// the clock pins.
    pub(super) base: Option<u32>,
    /// Output tokens the dead call emitted (its hidden reasoning included).
    pub(super) dead_tokens: u64,
    /// How long the dead call took.
    pub(super) dead_ms: u64,
    /// Wall clock left for the run, when it has one.
    pub(super) remaining: Option<std::time::Duration>,
    /// The first retry of this turn re-sends at the same cap on purpose.
    pub(super) first_retry: bool,
    /// The retry goes out with reasoning switched off, so it is a different
    /// call from the one that died even at the same cap.
    pub(super) reasoning_off: bool,
}

impl TruncatedRetryPlan {
    /// Milliseconds the dead call spent per output token, when both are known.
    fn ms_per_token(&self) -> Option<f64> {
        (self.dead_tokens > 0 && self.dead_ms > 0)
            .then(|| self.dead_ms as f64 / self.dead_tokens as f64)
    }

    /// The retry can grow the cap, or the call never had one (a plain retry of
    /// an uncapped call is still worth one attempt: the failure is stochastic).
    pub(super) fn cap_grows(&self) -> bool {
        match (self.current, self.next) {
            (Some(current), Some(next)) => next > current,
            _ => true,
        }
    }

    /// How long the retry would run if it dies at its cap the same way.
    fn expected_ms(&self) -> u64 {
        match (self.ms_per_token(), self.next) {
            (Some(rate), Some(next)) => (rate * next as f64) as u64,
            (None, Some(next)) if self.current.is_some_and(|current| current > 0) => {
                let current = self.current.unwrap_or_default();
                (self.dead_ms as f64 * next as f64 / current as f64) as u64
            }
            _ => self.dead_ms,
        }
    }

    /// The retry finishes inside its share of the remaining clock, or there is
    /// no clock to keep.
    pub(super) fn fits_clock(&self) -> bool {
        match self.remaining {
            Some(remaining) => {
                self.expected_ms() as f64
                    <= remaining.as_millis() as f64 * TRUNCATED_RETRY_CLOCK_SHARE
            }
            None => true,
        }
    }

    pub(super) fn worth_it(&self) -> bool {
        (self.first_retry || self.reasoning_off || self.cap_grows()) && self.fits_clock()
    }

    /// The retry would re-send the transcript that just died, at the cap it
    /// died at and with the same reasoning: nothing about it can go
    /// differently.
    pub(super) fn at_ceiling(&self) -> bool {
        !self.first_retry && !self.reasoning_off && !self.cap_grows()
    }

    pub(super) fn skip_reason(&self) -> String {
        if self.at_ceiling() {
            format!(
                "the output cap is already at its ceiling ({}); re-sending the same transcript at the same cap fails the same way",
                self.current.map_or("unset".to_string(), |c| c.to_string())
            )
        } else {
            format!(
                "a retry at {} tokens would run about {}s at the rate of the call that just died, more than half the {}s left",
                self.next
                    .map_or("the same cap".to_string(), |c| c.to_string()),
                self.expected_ms() / 1000,
                self.remaining.map_or(0, |d| d.as_secs())
            )
        }
    }

    /// The largest cap, between the first cap and the current one, whose call
    /// fits the clock share at the dead call's rate: the cap the nudged call
    /// should run under so it cannot spend the rest of the run the same way.
    /// `None` when there is nothing to learn from (no cap, no rate, no clock).
    pub(super) fn affordable_cap(&self) -> Option<u32> {
        let current = self.current?;
        let rate = self.ms_per_token()?;
        let remaining = self.remaining?;
        let affordable = (remaining.as_millis() as f64 * TRUNCATED_RETRY_CLOCK_SHARE / rate) as u64;
        let floor = self.base.unwrap_or(current).min(current).max(1);
        (affordable >= u64::from(floor)).then_some(affordable.min(u64::from(current)) as u32)
    }

    /// The cap the next nudged call runs under: half the current one, never
    /// below [`TRUNCATED_NUDGE_CAP_FLOOR`]. `None` when the call had no cap.
    pub(super) fn halved_cap(&self) -> Option<u32> {
        let current = self.current?;
        Some((current / 2).max(TRUNCATED_NUDGE_CAP_FLOOR).min(current))
    }

    /// The nudge cap is the smaller of the halved cap and the cap the clock
    /// can afford. If the known rate says even the original cap cannot fit,
    /// there is no affordable nudge cap.
    pub(super) fn nudge_cap(&self) -> Option<u32> {
        let halved = self.halved_cap()?;
        match (self.ms_per_token(), self.remaining) {
            (Some(rate), Some(remaining)) => {
                let affordable =
                    (remaining.as_millis() as f64 * TRUNCATED_RETRY_CLOCK_SHARE / rate) as u64;
                let minimum = halved.min(TRUNCATED_NUDGE_CAP_FLOOR);
                (affordable >= u64::from(minimum))
                    .then_some(halved.min(affordable.min(u64::from(u32::MAX)) as u32))
            }
            _ => Some(halved),
        }
    }

    /// A nudged call at the halved cap, at the dead call's rate, fits inside
    /// its share of the remaining clock. Without a clock there is nothing to
    /// spend, so the answer is no: the policy's own nudge count applies.
    pub(super) fn another_nudge_fits(&self) -> bool {
        let Some(remaining) = self.remaining else {
            return false;
        };
        let expected_ms = match (self.current, self.ms_per_token(), self.nudge_cap()) {
            (None, _, _) => {
                // An uncapped call cannot be shortened by changing its output
                // limit. Still allow a clock-bounded nudge when the previous
                // call's duration leaves enough room for another attempt.
                self.dead_ms
            }
            (_, Some(rate), Some(cap)) => (rate * cap as f64) as u64,
            // A capped call with no affordable nudge cap cannot safely repeat.
            (_, _, None) => return false,
            (_, None, Some(_)) => self.dead_ms,
        };
        expected_ms as f64 <= remaining.as_millis() as f64 * TRUNCATED_RETRY_CLOCK_SHARE
    }
}

impl TurnRecovery {
    /// What a retry of the dead call that just returned would cost and
    /// whether it can change anything (see [`TruncatedRetryPlan`]).
    /// `reasoning_off` says the retry goes out without reasoning: such a call
    /// needs no larger cap, since the cap was for the deliberation that is
    /// now switched off.
    pub(super) fn truncated_retry_plan(
        &self,
        attempt_max_tokens: Option<u32>,
        dead_tokens: u64,
        dead_ms: u64,
        remaining: Option<std::time::Duration>,
        reasoning_off: bool,
    ) -> TruncatedRetryPlan {
        let first_retry = self.truncated_empty_retries_used == 0;
        let current = self.boosted_max_tokens.or(attempt_max_tokens);
        let base = self.truncation_base.or(attempt_max_tokens);
        let next = attempt_max_tokens.map(|sent| {
            let base = self.truncation_base.unwrap_or(sent);
            let current = self.boosted_max_tokens.unwrap_or(sent);
            if first_retry || reasoning_off {
                current
            } else {
                current.saturating_mul(2).min(base.saturating_mul(4))
            }
        });
        TruncatedRetryPlan {
            current,
            next,
            base,
            dead_tokens,
            dead_ms,
            remaining,
            first_retry,
            reasoning_off,
        }
    }

    /// Take the retry a plan describes: remember the original cap and set the
    /// cap the retry runs under. An unset cap stays unset.
    pub(super) fn take_truncated_retry(
        &mut self,
        attempt_max_tokens: Option<u32>,
        plan: &TruncatedRetryPlan,
    ) {
        self.truncated_empty_retries_used += 1;
        if let Some(sent) = attempt_max_tokens {
            self.truncation_base.get_or_insert(sent);
            self.boosted_max_tokens = plan.next;
        }
    }

    /// Grows the next request's output cap after a length-truncated reply:
    /// double the cap last sent, clamped at 4x the original. An unset cap
    /// stays unset (a plain retry is still worthwhile — the failure is
    /// stochastic).
    pub(super) fn boost_max_tokens(&mut self, attempt_max_tokens: Option<u32>) {
        if let Some(sent) = attempt_max_tokens {
            let base = *self.truncation_base.get_or_insert(sent);
            self.boosted_max_tokens = Some(
                self.boosted_max_tokens
                    .unwrap_or(sent)
                    .saturating_mul(2)
                    .min(base.saturating_mul(4)),
            );
        }
    }

    /// Clears the truncated-empty recovery state: both counters, the boosted
    /// output cap and the cap it grows from. The state is scoped to a single
    /// logical turn, so it must not carry into the turns after a recovered one.
    pub(super) fn reset_truncated_empty(&mut self) {
        self.truncated_empty_retries_used = 0;
        self.truncated_empty_nudges_used = 0;
        self.boosted_max_tokens = None;
        self.truncation_base = None;
    }

    /// Clears the re-prompt counters (dropped, withheld, empty-response).
    fn reset_nudges(&mut self) {
        self.dropped_tool_call_nudges_used = 0;
        self.withheld_call_nudges_used = 0;
        self.empty_response_retries_used = 0;
    }

    /// Clears everything that tracks a truncated turn: the truncated-tool-call
    /// retry counter and the truncated-empty state (including the boosted cap).
    fn reset_truncation(&mut self) {
        self.truncated_tool_call_retries_used = 0;
        self.truncated_tool_names.clear();
        self.truncated_repeat_names.clear();
        self.reset_truncated_empty();
    }

    /// A tool-calling turn resolved (a plain one, or a mixed structured-output
    /// plus tool-call one): clear the re-prompt counters so a spent counter
    /// cannot leak into a later dropped-call turn.
    ///
    /// A turn whose call was cut off by the output limit
    /// (`turn_had_truncated_calls`) keeps its truncated-tool-call retry budget
    /// and boosted output cap for the retry.
    ///
    /// A tool call is a live reply: it spends one call of the reasoning
    /// fallback's hold-off.
    pub(super) fn reset_after_tool_turn(&mut self, turn_had_truncated_calls: bool) {
        self.reset_nudges();
        if !turn_had_truncated_calls {
            self.reset_truncation();
        }
        self.reasoning_fallback.on_live_reply();
    }

    /// The turn resolved without a tool call and without scheduling another
    /// retry (it is about to be taken as the answer): clear every counter and
    /// the boosted cap, which would otherwise override the caller's per-turn
    /// cap on every later call.
    ///
    /// An answer is a live reply: it spends one call of the reasoning
    /// fallback's hold-off.
    pub(super) fn reset_after_final(&mut self) {
        self.reset_nudges();
        self.reset_truncation();
        self.reasoning_fallback.on_live_reply();
    }
}

#[cfg(test)]
#[path = "turn_recovery_tests.rs"]
mod tests;
