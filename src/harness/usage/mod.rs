//! Token usage accounting.
//!
//! [`Usage`] records per-call token counts and supports `+`/`+=` accumulation.
//! [`UsageTotals`] aggregates many calls, also tracking the call count.

mod types;

use std::ops::{Add, AddAssign};

pub use types::*;

impl Usage {
    /// Creates a usage record from input and output token counts, setting
    /// `total_tokens` to their sum.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            ..Self::default()
        }
    }

    /// Returns the total tokens, falling back to `input + output` when the
    /// provider did not report an explicit total.
    pub fn effective_total(&self) -> u64 {
        if self.total_tokens > 0 {
            self.total_tokens
        } else {
            self.input_tokens + self.output_tokens
        }
    }
}

impl Add for Usage {
    type Output = Usage;

    fn add(mut self, rhs: Usage) -> Usage {
        self += rhs;
        self
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.total_tokens += rhs.total_tokens;
        self.cache_read_tokens += rhs.cache_read_tokens;
        self.cache_creation_tokens += rhs.cache_creation_tokens;
        self.reasoning_tokens += rhs.reasoning_tokens;
    }
}

impl UsageTotals {
    /// Creates an empty totals accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one call's usage, incrementing the call count.
    pub fn record(&mut self, usage: Usage) {
        self.calls += 1;
        self.usage += usage;
    }
}

impl Add<Usage> for UsageTotals {
    type Output = UsageTotals;

    fn add(mut self, rhs: Usage) -> UsageTotals {
        self.record(rhs);
        self
    }
}

impl AddAssign<Usage> for UsageTotals {
    fn add_assign(&mut self, rhs: Usage) {
        self.record(rhs);
    }
}

#[cfg(test)]
mod test;
