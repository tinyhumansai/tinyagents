//! Shared, dynamically updateable tool-timeout resolution.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::ToolTimeout;

#[derive(Debug)]
struct ToolTimeoutSettingsInner {
    inherited_ms: AtomicU64,
    min_ms: u64,
    max_ms: u64,
    grace_ms: u64,
}

/// Runtime tool-timeout settings shared by every settings clone installed on a
/// harness.
///
/// The inherited timeout is stored atomically, so a host can apply a config or
/// operator override without rebuilding active harnesses. A value of `0`
/// disables the inherited per-tool deadline; explicit [`ToolTimeout::Millis`]
/// policies remain bounded. Explicit budgets are clamped to `min_ms..=max_ms`
/// and receive `grace_ms` of scheduling slack on the enforced deadline.
#[derive(Clone)]
pub struct ToolTimeoutSettings {
    inner: Arc<ToolTimeoutSettingsInner>,
}

impl ToolTimeoutSettings {
    /// Creates shared timeout settings.
    ///
    /// Bounds are normalized defensively: `min_ms` is at least one and
    /// `max_ms` is at least the normalized minimum. A non-zero inherited value
    /// is clamped into those bounds.
    pub fn new(inherited_ms: u64, min_ms: u64, max_ms: u64, grace_ms: u64) -> Self {
        let min_ms = min_ms.max(1);
        let max_ms = max_ms.max(min_ms);
        let inherited_ms = clamp_or_disabled(inherited_ms, min_ms, max_ms);
        Self {
            inner: Arc::new(ToolTimeoutSettingsInner {
                inherited_ms: AtomicU64::new(inherited_ms),
                min_ms,
                max_ms,
                grace_ms,
            }),
        }
    }

    /// Replaces the inherited timeout for all settings clones.
    ///
    /// `0` disables inherited per-tool deadlines. Other values are clamped to
    /// the configured bounds. Returns the stored value.
    pub fn set_inherited_timeout_ms(&self, timeout_ms: u64) -> u64 {
        let timeout_ms = clamp_or_disabled(timeout_ms, self.inner.min_ms, self.inner.max_ms);
        self.inner.inherited_ms.store(timeout_ms, Ordering::Relaxed);
        timeout_ms
    }

    /// Returns the current inherited timeout, or `None` when it is disabled.
    pub fn inherited_timeout(&self) -> Option<Duration> {
        match self.inner.inherited_ms.load(Ordering::Relaxed) {
            0 => None,
            timeout_ms => Some(Duration::from_millis(timeout_ms)),
        }
    }

    /// Resolves a tool policy into an enforced deadline and reported budget.
    pub fn resolve(&self, policy: ToolTimeout) -> ResolvedToolTimeout {
        match policy {
            ToolTimeout::Inherit => {
                let budget_ms = self.inner.inherited_ms.load(Ordering::Relaxed);
                ResolvedToolTimeout {
                    deadline: (budget_ms != 0).then(|| Duration::from_millis(budget_ms)),
                    budget_ms,
                }
            }
            ToolTimeout::Unbounded => ResolvedToolTimeout {
                deadline: None,
                budget_ms: 0,
            },
            ToolTimeout::Millis(requested_ms) => {
                let budget_ms = requested_ms.clamp(self.inner.min_ms, self.inner.max_ms);
                ResolvedToolTimeout {
                    deadline: Some(Duration::from_millis(
                        budget_ms.saturating_add(self.inner.grace_ms),
                    )),
                    budget_ms,
                }
            }
        }
    }
}

impl fmt::Debug for ToolTimeoutSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolTimeoutSettings")
            .field(
                "inherited_ms",
                &self.inner.inherited_ms.load(Ordering::Relaxed),
            )
            .field("min_ms", &self.inner.min_ms)
            .field("max_ms", &self.inner.max_ms)
            .field("grace_ms", &self.inner.grace_ms)
            .finish()
    }
}

impl PartialEq for ToolTimeoutSettings {
    fn eq(&self, other: &Self) -> bool {
        self.inner.inherited_ms.load(Ordering::Relaxed)
            == other.inner.inherited_ms.load(Ordering::Relaxed)
            && self.inner.min_ms == other.inner.min_ms
            && self.inner.max_ms == other.inner.max_ms
            && self.inner.grace_ms == other.inner.grace_ms
    }
}

/// A resolved tool-call timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedToolTimeout {
    /// Deadline enforced around the call, including explicit-budget grace.
    pub deadline: Option<Duration>,
    /// Unpadded budget reported to callers and observability surfaces.
    pub budget_ms: u64,
}

fn clamp_or_disabled(timeout_ms: u64, min_ms: u64, max_ms: u64) -> u64 {
    if timeout_ms == 0 {
        0
    } else {
        timeout_ms.clamp(min_ms, max_ms)
    }
}
