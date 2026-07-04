//! Context-management middleware: message trimming, summarization-based
//! compression, and prompt-cache-layout guarding.
//!
//! Split out of `middleware/mod.rs`; see that module's doc comment for the
//! full middleware pipeline overview.

use super::*;
use crate::harness::cache::{CacheLayoutEvent, PromptCacheLayout};
use crate::harness::middleware::{
    ContextCompressionMiddleware, DEFAULT_CACHE_GUARD_EVENT_CAP, DEFAULT_COMPRESSION_RECORD_CAP,
    MessageTrimMiddleware, PromptCacheGuardMiddleware,
};
use crate::harness::summarization::{
    ConcatSummarizer, SummarizationPolicy, Summarizer, SummaryRecord, TrimStrategy,
    estimate_tokens, trim_messages,
};

// ── MessageTrimMiddleware ─────────────────────────────────────────────────────

impl MessageTrimMiddleware {
    /// Creates a trim middleware using the given [`TrimStrategy`].
    pub fn new(strategy: TrimStrategy) -> Self {
        Self { strategy }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for MessageTrimMiddleware {
    fn name(&self) -> &str {
        "message_trim"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.messages = trim_messages(&request.messages, &self.strategy);
        Ok(())
    }
}

// ── ContextCompressionMiddleware ──────────────────────────────────────────────

/// Estimate the total tokens of a message slice using the same per-message
/// heuristic the [`SummarizationPolicy`] uses internally.
fn total_message_tokens(messages: &[crate::harness::message::Message]) -> u64 {
    messages.iter().map(|m| estimate_tokens(&m.text())).sum()
}

impl ContextCompressionMiddleware {
    /// Creates a compression middleware backed by the default
    /// [`ConcatSummarizer`].
    pub fn new(policy: SummarizationPolicy) -> Self {
        Self::with_summarizer(policy, Box::new(ConcatSummarizer))
    }

    /// Creates a compression middleware with a custom [`Summarizer`].
    pub fn with_summarizer(policy: SummarizationPolicy, summarizer: Box<dyn Summarizer>) -> Self {
        Self {
            label: "context_compression",
            policy,
            summarizer,
            records: std::sync::Mutex::new(std::collections::VecDeque::new()),
            max_records: DEFAULT_COMPRESSION_RECORD_CAP,
        }
    }

    /// Sets the maximum number of [`SummaryRecord`]s retained before the
    /// oldest is evicted. `0` disables recording entirely.
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        let mut records = self.records.lock().expect("records mutex poisoned");
        while records.len() > max_records {
            records.pop_front();
        }
        drop(records);
        self
    }

    /// Returns the configured [`SummarizationPolicy`].
    pub fn policy(&self) -> &SummarizationPolicy {
        &self.policy
    }

    /// Returns the [`SummaryRecord`]s produced so far, in order. Bounded to
    /// at most [`ContextCompressionMiddleware::with_max_records`] entries
    /// (default [`DEFAULT_COMPRESSION_RECORD_CAP`]); older records are
    /// evicted first.
    pub fn records(&self) -> Vec<SummaryRecord> {
        self.records
            .lock()
            .expect("records mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ContextCompressionMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Below the window threshold: pass through untouched (no-op, no event).
        if !self.policy.should_summarize(&request.messages) {
            return Ok(());
        }

        let (to_summarize, mut to_keep) = self.policy.plan(&request.messages);
        // Nothing old enough to compress (e.g. keep_last covers everything):
        // leave the transcript untouched rather than summarizing an empty set.
        if to_summarize.is_empty() {
            return Ok(());
        }

        let from_tokens = total_message_tokens(&request.messages);
        let record = self.summarizer.summarize(&to_summarize).await?;

        // `plan` returns `to_keep` as `[system prompts..., recent turns...]`.
        // Insert the summary *after* the leading system prompts, not at index 0:
        // a system prompt must stay first so its persistent instructions keep
        // priority and the cacheable prefix is not churned. The summary of the
        // elided older turns then sits between the system prompt and the kept
        // recent turns, in chronological position.
        let system_prefix = to_keep
            .iter()
            .take_while(|m| matches!(m, crate::harness::message::Message::System(_)))
            .count();
        let recent = to_keep.split_off(system_prefix);
        let mut new_messages = Vec::with_capacity(to_keep.len() + recent.len() + 1);
        new_messages.append(&mut to_keep);
        new_messages.push(record.summary.clone());
        new_messages.extend(recent);
        let to_tokens = total_message_tokens(&new_messages);

        {
            let mut records = self.records.lock().expect("records mutex poisoned");
            if self.max_records > 0 {
                if records.len() >= self.max_records {
                    records.pop_front();
                }
                records.push_back(record);
            }
        }
        request.messages = new_messages;

        ctx.emit(AgentEvent::Compressed {
            from_tokens,
            to_tokens,
        });
        Ok(())
    }
}

// ── PromptCacheGuardMiddleware ────────────────────────────────────────────────

impl PromptCacheGuardMiddleware {
    /// Creates a cache-guard middleware with the default label
    /// `"prompt_cache_guard"`.
    pub fn new() -> Self {
        Self {
            label: "prompt_cache_guard",
            previous: std::sync::Mutex::new(None),
            events: std::sync::Mutex::new(std::collections::VecDeque::new()),
            max_events: DEFAULT_CACHE_GUARD_EVENT_CAP,
        }
    }

    /// Sets the maximum number of [`CacheLayoutEvent`]s retained before the
    /// oldest is evicted. `0` disables recording entirely.
    pub fn with_max_events(mut self, max_events: usize) -> Self {
        self.max_events = max_events;
        let mut events = self.events.lock().expect("events mutex poisoned");
        while events.len() > max_events {
            events.pop_front();
        }
        drop(events);
        self
    }

    /// Returns the cache-layout change events recorded so far, in order.
    /// Bounded to at most [`PromptCacheGuardMiddleware::with_max_events`]
    /// entries (default [`DEFAULT_CACHE_GUARD_EVENT_CAP`]); older events are
    /// evicted first.
    pub fn layout_events(&self) -> Vec<CacheLayoutEvent> {
        self.events
            .lock()
            .expect("events mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

impl Default for PromptCacheGuardMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for PromptCacheGuardMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let layout = PromptCacheLayout::from_request(request);
        let mut previous = self.previous.lock().expect("previous mutex poisoned");
        if let Some(prev) = previous.as_ref()
            && !prev.is_prefix_stable_against(&layout)
        {
            let event = CacheLayoutEvent::new(prev, &layout);
            let mut events = self.events.lock().expect("events mutex poisoned");
            if self.max_events > 0 {
                if events.len() >= self.max_events {
                    events.pop_front();
                }
                events.push_back(event);
            }
        }
        *previous = Some(layout);
        Ok(())
    }
}
