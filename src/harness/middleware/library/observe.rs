//! Observation middleware: structured-output validation, dynamic prompt
//! injection, redaction, and tracing.
//!
//! Split out of `library/mod.rs`; see that module's doc comment for the
//! full built-in middleware library overview.

use serde_json::Value;

use super::*;
use crate::harness::middleware::{
    AgentRun, HookCounts, LoggingMiddleware, UsageAccountingMiddleware,
};
use crate::harness::usage::UsageTotals;

// ── StructuredOutputValidatorMiddleware ───────────────────────────────────────

impl StructuredOutputValidatorMiddleware {
    /// Creates a validator middleware checking responses against `format`.
    pub fn new(format: ResponseFormat) -> Self {
        Self {
            label: "structured_output_validator",
            format,
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for StructuredOutputValidatorMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        match &self.format {
            ResponseFormat::Text => Ok(()),
            ResponseFormat::JsonObject => {
                let text = response.text();
                serde_json::from_str::<serde_json::Value>(&text).map_err(|e| {
                    TinyAgentsError::StructuredOutput(format!(
                        "response text is not valid JSON: {e}"
                    ))
                })?;
                Ok(())
            }
            ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
                let extractor = StructuredExtractor::new(
                    StructuredStrategy::ProviderSchema,
                    name.clone(),
                    schema.clone(),
                );
                extractor.extract(response)?;
                Ok(())
            }
        }
    }
}

// ── DynamicPromptMiddleware ───────────────────────────────────────────────────

impl<State, Ctx> DynamicPromptMiddleware<State, Ctx> {
    /// Creates a dynamic-prompt middleware deriving a system message from
    /// `prompt`.
    pub fn new(prompt: PromptFn<State>) -> Self {
        Self {
            label: "dynamic_prompt",
            prompt,
            _marker: PhantomData,
        }
    }

    /// Creates a dynamic-prompt middleware from a closure over the shared state
    /// and the run's [`RunConfig`].
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(&State, &RunConfig) -> Option<String> + Send + Sync + 'static,
    {
        Self::new(Arc::new(f))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for DynamicPromptMiddleware<State, Ctx>
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        if let Some(text) = (self.prompt)(state, &ctx.config) {
            request.messages.insert(0, Message::system(text));
        }
        Ok(())
    }
}

// ── RedactionMiddleware ───────────────────────────────────────────────────────

impl RedactionMiddleware {
    /// Creates a redaction middleware replacing each pattern with `"[REDACTED]"`.
    pub fn new(patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::with_mask(patterns, "[REDACTED]")
    }

    /// Creates a redaction middleware replacing each pattern with `mask`.
    pub fn with_mask(
        patterns: impl IntoIterator<Item = impl Into<String>>,
        mask: impl Into<String>,
    ) -> Self {
        Self {
            label: "redaction",
            patterns: patterns
                .into_iter()
                .map(Into::into)
                .filter(|p| !p.is_empty())
                .collect(),
            mask: mask.into(),
            redactions: Mutex::new(0),
        }
    }

    /// Returns the total number of pattern occurrences redacted so far.
    pub fn redactions(&self) -> usize {
        *self.redactions.lock().expect("redactions mutex poisoned")
    }

    /// Replaces every configured pattern in `text`, returning the redacted
    /// string and the number of occurrences replaced.
    ///
    /// The scan is a single left-to-right pass over the *original* text, so a
    /// pattern can never match inside mask text introduced by an earlier
    /// replacement. Existing occurrences of the full mask string in the input
    /// are skipped as opaque, which makes redaction idempotent: re-running the
    /// middleware over already-redacted text is a no-op even when a pattern
    /// happens to be a substring of the mask. When several patterns match at
    /// the same position, the first-configured pattern wins.
    fn redact(&self, text: &str) -> (String, usize) {
        let mut out = String::with_capacity(text.len());
        let mut hits = 0usize;
        let mut rest = text;
        'scan: while !rest.is_empty() {
            // Treat prior mask output as opaque so redaction is idempotent.
            if !self.mask.is_empty() && rest.starts_with(self.mask.as_str()) {
                out.push_str(&self.mask);
                rest = &rest[self.mask.len()..];
                continue;
            }
            for pattern in &self.patterns {
                if rest.starts_with(pattern.as_str()) {
                    out.push_str(&self.mask);
                    hits += 1;
                    rest = &rest[pattern.len()..];
                    continue 'scan;
                }
            }
            let ch = rest.chars().next().expect("non-empty remainder");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
        (out, hits)
    }

    /// Recursively redacts every string value inside a JSON `value` (array
    /// elements and object values; object keys are left untouched), returning
    /// the number of occurrences replaced.
    fn redact_value(&self, value: &mut Value) -> usize {
        match value {
            Value::String(s) => {
                let (redacted, hits) = self.redact(s);
                if hits > 0 {
                    *s = redacted;
                }
                hits
            }
            Value::Array(items) => items.iter_mut().map(|v| self.redact_value(v)).sum(),
            Value::Object(map) => map.values_mut().map(|v| self.redact_value(v)).sum(),
            _ => 0,
        }
    }

    /// Records `hits` redactions against the running total.
    fn record(&self, hits: usize) {
        if hits > 0 {
            *self.redactions.lock().expect("redactions mutex poisoned") += hits;
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for RedactionMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        let mut hits = 0usize;
        for block in &mut response.message.content {
            match block {
                ContentBlock::Text(text) => {
                    let (redacted, n) = self.redact(text);
                    if n > 0 {
                        *text = redacted;
                        hits += n;
                    }
                }
                ContentBlock::Json(value) => hits += self.redact_value(value),
                _ => {}
            }
        }
        // Model-authored tool-call arguments leave the harness (they are
        // echoed into transcripts and journals), so scrub them too.
        for call in &mut response.message.tool_calls {
            hits += self.redact_value(&mut call.arguments);
        }
        // The raw provider payload carries the same text verbatim.
        if let Some(raw) = &mut response.raw {
            hits += self.redact_value(raw);
        }
        self.record(hits);
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        let hits = self.redact_value(&mut call.arguments);
        self.record(hits);
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        let (redacted, mut hits) = self.redact(&result.content);
        if hits > 0 {
            result.content = redacted;
        }
        // The structured `raw` payload duplicates (or extends) the content, so
        // it must be scrubbed as well.
        if let Some(raw) = &mut result.raw {
            hits += self.redact_value(raw);
        }
        if let Some(error) = &mut result.error {
            let (redacted, n) = self.redact(error);
            if n > 0 {
                *error = redacted;
                hits += n;
            }
        }
        self.record(hits);
        Ok(())
    }
}

// ── TracingMiddleware ─────────────────────────────────────────────────────────

impl TracingMiddleware {
    /// Creates a tracing middleware with the default label `"tracing"`.
    pub fn new() -> Self {
        Self::with_label("tracing")
    }

    /// Creates a tracing middleware with a custom static label.
    pub fn with_label(label: &'static str) -> Self {
        Self {
            label,
            records: Mutex::new(VecDeque::new()),
            counts: Mutex::new(TraceCounts::default()),
            max_records: DEFAULT_TRACE_RECORD_CAP,
        }
    }

    /// Sets the maximum number of [`PhaseTrace`] entries retained before the
    /// oldest is evicted. `0` disables recording entirely (counts are still
    /// tracked).
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        let mut records = self.records.lock().expect("records mutex poisoned");
        while records.len() > max_records {
            records.pop_front();
        }
        drop(records);
        self
    }

    /// Returns the structured begin/end traces recorded so far, in order.
    /// Bounded to at most [`TracingMiddleware::with_max_records`] entries
    /// (default [`DEFAULT_TRACE_RECORD_CAP`]); older traces are evicted first.
    pub fn records(&self) -> Vec<PhaseTrace> {
        self.records
            .lock()
            .expect("records mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// Returns a snapshot of the per-phase begin counts.
    pub fn counts(&self) -> TraceCounts {
        self.counts.lock().expect("counts mutex poisoned").clone()
    }

    fn push(&self, phase: &'static str, boundary: TraceBoundary) {
        let mut records = self.records.lock().expect("records mutex poisoned");
        if self.max_records == 0 {
            return;
        }
        if records.len() >= self.max_records {
            records.pop_front();
        }
        records.push_back(PhaseTrace { phase, boundary });
    }
}

impl Default for TracingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for TracingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").agent += 1;
        self.push("agent", TraceBoundary::Begin);
        Ok(())
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut crate::harness::middleware::AgentRun,
    ) -> Result<()> {
        self.push("agent", TraceBoundary::End);
        Ok(())
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").model += 1;
        self.push("model", TraceBoundary::Begin);
        Ok(())
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").delta += 1;
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.push("model", TraceBoundary::End);
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").tool += 1;
        self.push("tool", TraceBoundary::Begin);
        Ok(())
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").delta += 1;
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.push("tool", TraceBoundary::End);
        Ok(())
    }

    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").error += 1;
        Ok(())
    }
}

// ── LoggingMiddleware ─────────────────────────────────────────────────────────

// ── LoggingMiddleware ─────────────────────────────────────────────────────────

impl LoggingMiddleware {
    /// Creates a logging middleware with the default label `"logging"`.
    pub fn new() -> Self {
        Self::with_label("logging")
    }

    /// Creates a logging middleware with a custom static label.
    pub fn with_label(label: &'static str) -> Self {
        Self {
            label,
            counts: std::sync::Mutex::new(HookCounts::default()),
        }
    }

    /// Returns a snapshot of the per-hook invocation counts recorded so far.
    pub fn counts(&self) -> HookCounts {
        self.counts.lock().expect("counts mutex poisoned").clone()
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for LoggingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_agent(&self, _ctx: &mut RunContext<Ctx>, _state: &State) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_agent += 1;
        Ok(())
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _run: &mut AgentRun,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_agent += 1;
        Ok(())
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_model += 1;
        Ok(())
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ModelDelta,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .on_model_delta += 1;
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_model += 1;
        Ok(())
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _call: &mut ToolCall,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .before_tool += 1;
        Ok(())
    }

    async fn on_tool_delta(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _delta: &mut ToolDelta,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .on_tool_delta += 1;
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        _result: &mut ToolResult,
    ) -> Result<()> {
        self.counts
            .lock()
            .expect("counts mutex poisoned")
            .after_tool += 1;
        Ok(())
    }

    async fn on_error(&self, _ctx: &mut RunContext<Ctx>, _error: &TinyAgentsError) -> Result<()> {
        self.counts.lock().expect("counts mutex poisoned").on_error += 1;
        Ok(())
    }
}

// ── UsageAccountingMiddleware ─────────────────────────────────────────────────

// ── UsageAccountingMiddleware ─────────────────────────────────────────────────

impl UsageAccountingMiddleware {
    /// Creates a usage-accounting middleware with the default label
    /// `"usage_accounting"`.
    pub fn new() -> Self {
        Self {
            label: "usage_accounting",
            totals: std::sync::Mutex::new(UsageTotals::new()),
        }
    }

    /// Returns a snapshot of the accumulated usage totals.
    pub fn totals(&self) -> UsageTotals {
        *self.totals.lock().expect("totals mutex poisoned")
    }
}

impl Default for UsageAccountingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for UsageAccountingMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        response: &mut ModelResponse,
    ) -> Result<()> {
        if let Some(usage) = response.usage {
            self.totals
                .lock()
                .expect("totals mutex poisoned")
                .record(usage);
        }
        Ok(())
    }
}
