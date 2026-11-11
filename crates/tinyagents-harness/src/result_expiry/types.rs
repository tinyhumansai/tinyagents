//! Types for expiring old tool results by age.

use serde::{Deserialize, Serialize};

/// How many user turns a tool result survives when expiry is switched on and
/// no age is given.
pub const DEFAULT_EXPIRY_MAX_AGE_TURNS: usize = 6;

/// How many characters of the original result an expiry note keeps.
pub const EXPIRY_NOTE_CHARS: usize = 120;

/// Whether, and how, a run replaces old tool results in its stored transcript
/// with a one-line note.
///
/// A result's age is the number of user messages after it. At the start of a
/// run, every result older than [`Self::max_age_turns`] is rewritten, whatever
/// the context window has left (micro-compaction only acts near the window).
/// Defaults to off, so a transcript is byte-identical unless the host opts in.
/// A session overrides it per run with [`ExpiryOverrides`] on
/// [`crate::context::RunConfig::result_expiry`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolResultExpiryPolicy {
    /// Whether old results expire. Defaults to `false`.
    pub enabled: bool,
    /// A result with more than this many user messages after it expires.
    /// Defaults to [`DEFAULT_EXPIRY_MAX_AGE_TURNS`].
    pub max_age_turns: usize,
    /// Tools whose results never expire, by exact name (for example a collapsed
    /// `tool_search` result the model keeps calling from).
    pub exempt_tools: Vec<String>,
}

impl Default for ToolResultExpiryPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_age_turns: DEFAULT_EXPIRY_MAX_AGE_TURNS,
            exempt_tools: Vec::new(),
        }
    }
}

/// Per-session overrides of [`ToolResultExpiryPolicy`]. An option left `None`
/// keeps the harness-wide value.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpiryOverrides {
    /// Overrides [`ToolResultExpiryPolicy::enabled`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Overrides [`ToolResultExpiryPolicy::max_age_turns`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_age_turns: Option<usize>,
    /// Replaces [`ToolResultExpiryPolicy::exempt_tools`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exempt_tools: Option<Vec<String>>,
}

impl ExpiryOverrides {
    /// Whether no option is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl ToolResultExpiryPolicy {
    /// This policy with a session's `overrides` applied.
    #[must_use]
    pub fn with_overrides(&self, overrides: &ExpiryOverrides) -> Self {
        let mut effective = self.clone();
        if let Some(enabled) = overrides.enabled {
            effective.enabled = enabled;
        }
        if let Some(max_age_turns) = overrides.max_age_turns {
            effective.max_age_turns = max_age_turns;
        }
        if let Some(exempt_tools) = &overrides.exempt_tools {
            effective.exempt_tools = exempt_tools.clone();
        }
        effective
    }
}
