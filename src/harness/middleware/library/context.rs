//! Context-management middleware: message trimming, summarization-based
//! compression, and prompt-cache-layout guarding.
//!
//! Split out of `middleware/mod.rs`; see that module's doc comment for the
//! full middleware pipeline overview.

use super::*;
use crate::harness::cache::{CacheLayoutEvent, PromptCacheLayout};
use crate::harness::middleware::{
    CompressionFailurePolicy, ContextCompressionMiddleware, DEFAULT_CACHE_GUARD_EVENT_CAP,
    DEFAULT_COMPRESSION_RECORD_CAP, MessageTrimMiddleware, MicrocompactMiddleware,
    PromptCacheGuardMiddleware,
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
    ///
    /// The failure policy defaults to
    /// [`CompressionFailurePolicy::FallbackTrim`]; override it with
    /// [`with_failure_policy`](Self::with_failure_policy).
    pub fn with_summarizer(policy: SummarizationPolicy, summarizer: Box<dyn Summarizer>) -> Self {
        Self {
            label: "context_compression",
            policy,
            summarizer,
            records: std::sync::Mutex::new(std::collections::VecDeque::new()),
            max_records: DEFAULT_COMPRESSION_RECORD_CAP,
            on_failure: CompressionFailurePolicy::default(),
        }
    }

    /// Sets the [`CompressionFailurePolicy`] applied when the [`Summarizer`]
    /// returns an `Err`. Defaults to
    /// [`CompressionFailurePolicy::FallbackTrim`].
    pub fn with_failure_policy(mut self, on_failure: CompressionFailurePolicy) -> Self {
        self.on_failure = on_failure;
        self
    }

    /// Returns the configured [`CompressionFailurePolicy`].
    pub fn failure_policy(&self) -> CompressionFailurePolicy {
        self.on_failure
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
        let record = match self.summarizer.summarize(&to_summarize).await {
            Ok(record) => record,
            Err(err) => {
                // A summarizer failure hits precisely the longest, most valuable
                // transcripts (the ones that reached the compaction threshold).
                // Emit a diagnostic and recover per the configured
                // `CompressionFailurePolicy` instead of aborting the whole run.
                ctx.emit(AgentEvent::MiddlewareFailed {
                    name: self.label.to_string(),
                    error: err.to_string(),
                });
                match self.on_failure {
                    // Legacy behaviour: propagate and let the run fail.
                    CompressionFailurePolicy::Abort => return Err(err),
                    // Keep the (over-threshold) transcript verbatim and continue.
                    CompressionFailurePolicy::PassThrough => return Ok(()),
                    // Deterministic front-drop to the policy's trigger budget,
                    // preserving system messages (see `TrimStrategy::MaxTokens`).
                    CompressionFailurePolicy::FallbackTrim => {
                        let trimmed = trim_messages(
                            &request.messages,
                            &TrimStrategy::MaxTokens(self.policy.trigger_budget()),
                        );
                        let to_tokens = total_message_tokens(&trimmed);
                        request.messages = trimmed;
                        ctx.emit(AgentEvent::Compressed {
                            from_tokens,
                            to_tokens,
                        });
                        return Ok(());
                    }
                }
            }
        };

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

// ── MicrocompactMiddleware ────────────────────────────────────────────────────

impl MicrocompactMiddleware {
    /// Creates a micro-compaction middleware that keeps the newest `keep_recent`
    /// tool-result bodies verbatim and blanks older ones with `placeholder`.
    /// Event emission is off by default; enable it with
    /// [`MicrocompactMiddleware::with_events`].
    pub fn new(keep_recent: usize, placeholder: impl Into<String>) -> Self {
        Self {
            label: "microcompact",
            keep_recent,
            placeholder: placeholder.into(),
            emit_events: false,
            token_budget: None,
        }
    }

    /// Enable or disable emitting an
    /// [`AgentEvent::Compressed`][crate::harness::events::AgentEvent::Compressed]
    /// event whenever at least one tool body is cleared. Off by default so the
    /// middleware can be a silent transcript rewrite.
    pub fn with_events(mut self, emit_events: bool) -> Self {
        self.emit_events = emit_events;
        self
    }

    /// Only blank stale tool bodies once the transcript's estimated tokens
    /// exceed `budget`; below it the middleware is a no-op so the request stays
    /// append-only and the provider KV-cache prefix is preserved (issue
    /// tinyhumansai/openhuman#4755).
    ///
    /// Set this below the model's context window (leaving headroom for the reply
    /// and the `keep_recent` verbatim results) so a run that fits the window is
    /// never compacted — compaction only kicks in when it is actually needed to
    /// stay under the window, which is the only time the cache-invalidation cost
    /// of blanking an already-sent tool body pays for itself. `budget == 0`
    /// disables the gate (equivalent to leaving it unset).
    pub fn with_token_budget(mut self, budget: u64) -> Self {
        self.token_budget = (budget > 0).then_some(budget);
        self
    }

    /// The number of most-recent tool-result bodies kept verbatim.
    pub fn keep_recent(&self) -> usize {
        self.keep_recent
    }

    /// The placeholder text swapped in for cleared tool-result bodies.
    pub fn placeholder(&self) -> &str {
        &self.placeholder
    }

    /// The estimated-token floor below which blanking is skipped, if configured.
    pub fn token_budget(&self) -> Option<u64> {
        self.token_budget
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for MicrocompactMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        // Indices of every tool-result message, oldest → newest.
        let tool_idxs: Vec<usize> = request
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m, Message::Tool(_)))
            .map(|(i, _)| i)
            .collect();
        if tool_idxs.len() <= self.keep_recent {
            return Ok(());
        }

        // Prompt-cache preservation (issue tinyhumansai/openhuman#4755): when a
        // token budget is configured, skip blanking while the transcript still
        // fits within it. Blanking a tool body that was sent verbatim on an
        // earlier iteration mutates an already-transmitted prefix position and
        // invalidates the provider KV-cache from there on; doing that every call
        // (the boundary moves by one each turn) churns the cache to reclaim
        // tokens the model still has room for. Gating on the *pre-blank* estimate
        // — which grows monotonically as the run appends — keeps the request
        // append-only (fully cache-eligible) below budget and can't oscillate.
        // Reuse `from_tokens` when events are on so we never estimate twice.
        let from_tokens = if self.emit_events {
            total_message_tokens(&request.messages)
        } else {
            0
        };
        if let Some(budget) = self.token_budget {
            let tokens = if self.emit_events {
                from_tokens
            } else {
                total_message_tokens(&request.messages)
            };
            if tokens <= budget {
                return Ok(());
            }
        }

        let cut = tool_idxs.len() - self.keep_recent;
        let mut cleared = 0usize;
        for &i in &tool_idxs[..cut] {
            // Skip messages already reduced to the placeholder; otherwise swap the
            // body for it (idempotent, preserves the tool_call_id).
            if request.messages[i].text() == self.placeholder {
                continue;
            }
            if let Message::Tool(t) = &request.messages[i] {
                let id = t.tool_call_id.clone();
                request.messages[i] = Message::tool(id, self.placeholder.clone());
                cleared += 1;
            }
        }

        if self.emit_events && cleared > 0 {
            let to_tokens = total_message_tokens(&request.messages);
            ctx.emit(AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            });
        }
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
