//! Tests for the middleware stack and built-in middleware.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::error::{Result, TinyAgentsError};
use crate::harness::context::{RunConfig, RunContext};
use crate::harness::events::{AgentEvent, RecordingListener};
use crate::harness::message::{AssistantMessage, ContentBlock, Message, UserMessage};
use crate::harness::model::{ModelRequest, ModelResponse, PromptSegment, SegmentRole};
use crate::harness::summarization::{SummarizationPolicy, Summarizer, SummaryRecord, TrimStrategy};
use crate::harness::tool::{ToolCall, ToolResult};
use crate::harness::usage::Usage;

// ── helpers ───────────────────────────────────────────────────────────────────

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

fn response_with_usage(usage: Usage) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
        },
        usage: Some(usage),
        finish_reason: None,
        raw: None,
        resolved_model: None,
    }
}

fn segment(id: &str, role: SegmentRole, cacheable: bool) -> PromptSegment {
    PromptSegment {
        id: id.to_string(),
        role,
        cacheable,
    }
}

/// Records hook firing order into a shared log for ordering assertions.
struct OrderRecorder {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<()> for OrderRecorder {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:before", self.label));
        Ok(())
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _response: &mut ModelResponse,
    ) -> Result<()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:after", self.label));
        Ok(())
    }
}

/// Always fails its `before_model` hook to exercise short-circuiting.
struct FailingMiddleware;

#[async_trait]
impl Middleware<()> for FailingMiddleware {
    fn name(&self) -> &str {
        "failing"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        Err(TinyAgentsError::Middleware("boom".to_string()))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn before_runs_forward_after_runs_reverse() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(OrderRecorder {
        label: "a",
        log: log.clone(),
    }));
    stack.push(Arc::new(OrderRecorder {
        label: "b",
        log: log.clone(),
    }));

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let mut response = response_with_usage(Usage::new(1, 1));

    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    stack
        .run_after_model(&mut c, &(), &mut response)
        .await
        .unwrap();

    let order = log.lock().unwrap().clone();
    assert_eq!(
        order,
        vec!["a:before", "b:before", "b:after", "a:after"],
        "before runs in registration order, after runs reversed"
    );
}

#[tokio::test]
async fn error_short_circuits_and_invokes_on_error() {
    let logging = Arc::new(LoggingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(logging.clone());
    stack.push(Arc::new(FailingMiddleware));
    // This third middleware must never run because the second one fails first.
    let never = Arc::new(LoggingMiddleware::with_label("never"));
    stack.push(never.clone());

    let mut c = ctx();
    let mut request = ModelRequest::default();
    let result = stack.run_before_model(&mut c, &(), &mut request).await;

    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));
    // on_error fanned out to the whole stack, so the first logging mw saw it.
    assert_eq!(logging.counts().on_error, 1);
    // The first logging mw's before_model ran; the one after the failure did not.
    assert_eq!(logging.counts().before_model, 1);
    assert_eq!(never.counts().before_model, 0);
}

#[tokio::test]
async fn emits_started_and_completed_events() {
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let kinds: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert_eq!(
        kinds,
        vec![
            AgentEvent::MiddlewareStarted {
                name: "logging".to_string()
            },
            AgentEvent::MiddlewareCompleted {
                name: "logging".to_string()
            },
        ]
    );
}

#[tokio::test]
async fn failing_hook_still_emits_balanced_completed_event() {
    // A hook that returns `Err` must still close its `MiddlewareStarted` with a
    // matching `MiddlewareCompleted`, so downstream observers never see a
    // dangling, unbalanced `Started`.
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(FailingMiddleware));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest::default();
    let result = stack.run_before_model(&mut c, &(), &mut request).await;
    assert!(matches!(result, Err(TinyAgentsError::Middleware(_))));

    let brackets: Vec<AgentEvent> = recorder
        .events()
        .into_iter()
        .map(|r| r.event)
        .filter(|e| {
            matches!(
                e,
                AgentEvent::MiddlewareStarted { .. } | AgentEvent::MiddlewareCompleted { .. }
            )
        })
        .collect();
    assert_eq!(
        brackets,
        vec![
            AgentEvent::MiddlewareStarted {
                name: "failing".to_string()
            },
            AgentEvent::MiddlewareCompleted {
                name: "failing".to_string()
            },
        ],
        "a failing hook must emit a balanced Started/Completed pair"
    );
}

#[tokio::test]
async fn on_model_delta_hook_emits_no_bracketing_events() {
    // The per-delta hook runs on the streaming hot path, so it must NOT emit
    // `MiddlewareStarted`/`MiddlewareCompleted` events the way the other stack
    // runners do — those two events per middleware per token dominated the
    // stream loop for no observability value.
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(LoggingMiddleware::new()));

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut delta = ModelDelta {
        call_id: "call-1".to_string(),
        content: "tok".to_string(),
        reasoning: String::new(),
        tool_call: None,
    };
    stack
        .run_on_model_delta(&mut c, &(), &mut delta)
        .await
        .unwrap();

    let bracketing = recorder
        .events()
        .into_iter()
        .filter(|r| {
            matches!(
                r.event,
                AgentEvent::MiddlewareStarted { .. } | AgentEvent::MiddlewareCompleted { .. }
            )
        })
        .count();
    assert_eq!(
        bracketing, 0,
        "the delta hook must not bracket middleware with events"
    );
}

#[tokio::test]
async fn message_trim_middleware_shrinks_request() {
    let mw = MessageTrimMiddleware::new(TrimStrategy::KeepLast(1));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(Arc::new(mw));

    let mut request = ModelRequest {
        messages: vec![user("one"), user("two"), user("three")],
        ..Default::default()
    };
    let mut c = ctx();
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0], user("three"));
}

#[tokio::test]
async fn context_compression_is_noop_below_window_threshold() {
    // 1000-token window, 0.9 threshold → 900-token budget. A tiny transcript
    // stays far below it, so the middleware must leave messages untouched and
    // emit no Compressed event.
    let policy = SummarizationPolicy::default()
        .with_context_window(1000)
        .with_threshold_fraction(0.9);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let before = vec![user("one"), user("two"), user("three")];
    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Messages unchanged.
    assert_eq!(request.messages, before);
    // No record produced.
    assert!(mw.records().is_empty());
    // No Compressed event emitted (only the stack's started/completed events).
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
    );
}

#[tokio::test]
async fn context_compression_compresses_at_or_above_threshold() {
    // 100-token window, 0.5 threshold → 50-token budget. keep_last=1 keeps the
    // newest message verbatim; everything older is summarized.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    // Three ~50-token (200-char) messages → ~150 tokens, well above the 50 budget.
    let big = "a".repeat(200);
    let mut request = ModelRequest {
        messages: vec![
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Compressed to: one summary message + the single kept recent message.
    assert_eq!(request.messages.len(), 2);
    assert!(matches!(request.messages[0], Message::System(_)));
    assert_eq!(request.messages[1].text(), format!("{big}-3"));

    // Provenance recorded: the two oldest messages were the source.
    let records = mw.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(records[0].provenance.original_token_estimate > 0);

    // A single Compressed event was emitted. (ConcatSummarizer keeps text
    // verbatim, so the guarantee is fewer messages, not fewer tokens; the event
    // still reports the before/after token estimates.)
    let compressed: Vec<(u64, u64)> = recorder
        .events()
        .into_iter()
        .filter_map(|r| match r.event {
            AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            } => Some((from_tokens, to_tokens)),
            _ => None,
        })
        .collect();
    assert_eq!(compressed.len(), 1);
    assert!(compressed[0].0 > 0);
    assert!(compressed[0].1 > 0);
}

/// A [`Summarizer`] that always fails — models the real gap the
/// `ConcatSummarizer` can never exhibit (a model-backed summarizer whose
/// provider call is rejected).
struct FailingSummarizer;

#[async_trait]
impl Summarizer for FailingSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        Err(TinyAgentsError::Model("summarizer boom".to_string()))
    }
}

/// Build an over-threshold transcript `[system, big-1, big-2, big-3]` under a
/// 100-token window / 0.5 threshold (→ 50-token trigger budget), so the
/// compression middleware is guaranteed to fire.
fn over_threshold_request() -> (SummarizationPolicy, Vec<Message>) {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let big = "a".repeat(200);
    let messages = vec![
        Message::system("You are a helpful assistant."),
        user(&format!("{big}-1")),
        user(&format!("{big}-2")),
        user(&format!("{big}-3")),
    ];
    (policy, messages)
}

#[tokio::test]
async fn context_compression_falls_back_to_trim_when_summarizer_errors() {
    // Regression for the "summarizer failure aborts the run" gap: under the
    // default FallbackTrim policy a failing summarizer must NOT abort. The
    // middleware front-drops the transcript to the trigger budget (keeping the
    // leading system prompt) and continues, emitting a MiddlewareFailed
    // diagnostic plus a Compressed event for the trim.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(FailingSummarizer),
    ));
    // Default failure policy is the trim fallback.
    assert_eq!(mw.failure_policy(), CompressionFailurePolicy::FallbackTrim);

    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .expect("summarizer failure must not abort the run under FallbackTrim");

    // Trimmed from the front — strictly fewer messages — with the system prompt
    // preserved (MaxTokens drops system messages last).
    assert!(request.messages.len() < before.len());
    assert!(matches!(request.messages[0], Message::System(_)));
    // No summary record could be produced (the summarizer failed).
    assert!(mw.records().is_empty());

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, error }
                if name == "context_compression" && error.contains("summarizer boom")
        )),
        "a MiddlewareFailed diagnostic naming the failure must be emitted: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "the fallback trim must emit a Compressed event: {events:?}"
    );
}

#[tokio::test]
async fn context_compression_abort_policy_propagates_summarizer_error() {
    // Opt back into the legacy behaviour: Abort propagates the error so the run
    // fails, but still emits the MiddlewareFailed diagnostic first.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(FailingSummarizer))
            .with_failure_policy(CompressionFailurePolicy::Abort),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before,
        ..Default::default()
    };
    let result = stack.run_before_model(&mut c, &(), &mut request).await;
    assert!(
        result.is_err(),
        "Abort must propagate the summarizer error and fail the run"
    );

    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, .. } if name == "context_compression"
        )),
        "Abort must still emit the MiddlewareFailed diagnostic: {events:?}"
    );
}

#[tokio::test]
async fn context_compression_pass_through_policy_keeps_transcript_and_continues() {
    // PassThrough leaves the (over-threshold) transcript untouched and lets the
    // run continue, emitting only the MiddlewareFailed diagnostic.
    let (policy, before) = over_threshold_request();
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(FailingSummarizer))
            .with_failure_policy(CompressionFailurePolicy::PassThrough),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: before.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .expect("PassThrough must not abort the run");

    // Transcript is untouched.
    assert_eq!(request.messages, before);
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MiddlewareFailed { name, .. } if name == "context_compression"
        )),
        "PassThrough must emit the MiddlewareFailed diagnostic: {events:?}"
    );
    // No compression happened, so no Compressed event.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "PassThrough must not emit a Compressed event: {events:?}"
    );
}

#[tokio::test]
async fn context_compression_records_are_bounded_by_max_records() {
    // A long-running loop that compresses repeatedly must not grow the
    // recorder without bound; cap it and confirm eviction happens.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy).with_max_records(2));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let big = "a".repeat(200);
    let mut c = ctx();
    for _ in 0..10 {
        let mut request = ModelRequest {
            messages: vec![
                user(&format!("{big}-1")),
                user(&format!("{big}-2")),
                user(&format!("{big}-3")),
            ],
            ..Default::default()
        };
        stack
            .run_before_model(&mut c, &(), &mut request)
            .await
            .unwrap();
    }

    assert_eq!(mw.records().len(), 2);
}

#[tokio::test]
async fn context_compression_keeps_system_prompt_before_summary() {
    // A leading system prompt carries persistent instructions and anchors the
    // cacheable prefix; the summary of elided older turns must be inserted
    // *after* it, never at position 0.
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let big = "a".repeat(200);
    let system_prompt = "You are a helpful assistant. Always follow these rules.";
    let mut request = ModelRequest {
        messages: vec![
            Message::system(system_prompt),
            user(&format!("{big}-1")),
            user(&format!("{big}-2")),
            user(&format!("{big}-3")),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Result is [real system prompt, summary, kept recent turn].
    assert_eq!(request.messages.len(), 3);
    assert!(matches!(request.messages[0], Message::System(_)));
    assert_eq!(
        request.messages[0].text(),
        system_prompt,
        "the real system prompt must stay at position 0"
    );
    assert!(
        matches!(request.messages[1], Message::System(_)),
        "the summary follows the system prompt"
    );
    assert_ne!(
        request.messages[1].text(),
        system_prompt,
        "position 1 is the summary, not a duplicated system prompt"
    );
    assert_eq!(request.messages[2].text(), format!("{big}-3"));
}

#[tokio::test]
async fn context_compression_none_window_falls_back_to_trigger_tokens() {
    // No context window → raw trigger_tokens gate (strict `>`). Trigger at 2
    // tokens, keep_last=1.
    let policy = SummarizationPolicy {
        trigger_tokens: 2,
        keep_last: 1,
        ..SummarizationPolicy::default()
    };
    assert_eq!(policy.context_window, None);
    let mw = Arc::new(ContextCompressionMiddleware::new(policy));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    // Two ~4-token (16-char) messages → ~8 tokens > 2 trigger.
    let mut request = ModelRequest {
        messages: vec![user("aaaaaaaaaaaaaaaa"), user("bbbbbbbbbbbbbbbb")],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    // Summary + the one kept recent message.
    assert_eq!(request.messages.len(), 2);
    assert!(matches!(request.messages[0], Message::System(_)));
    assert_eq!(request.messages[1].text(), "bbbbbbbbbbbbbbbb");
    assert_eq!(mw.records().len(), 1);
}

// ── MicrocompactMiddleware ────────────────────────────────────────────────────
//
// These cases are the byte-for-byte parity contract ported from the OpenHuman
// in-house `MicrocompactMiddleware` this type replaces: older tool-result bodies
// are blanked to the caller's placeholder, the newest `keep_recent` are kept
// verbatim, non-tool messages are never touched, and the pass is idempotent.

const CLEARED: &str = "[Old tool result content cleared]";

#[tokio::test]
async fn microcompact_clears_older_tool_bodies_and_keeps_recent() {
    let mw = MicrocompactMiddleware::new(1, CLEARED);
    let mut request = ModelRequest {
        messages: vec![
            Message::system("sys"),
            Message::user("hello"),
            Message::tool("t1", "FIRST_BODY"),
            Message::assistant("thinking"),
            Message::tool("t2", "SECOND_BODY"),
            Message::tool("t3", "THIRD_BODY"),
        ],
        ..Default::default()
    };

    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();

    // 3 tool messages, keep_recent=1 → the two oldest cleared, newest kept.
    assert_eq!(request.messages[2].text(), CLEARED);
    assert_eq!(request.messages[4].text(), CLEARED);
    assert_eq!(request.messages[5].text(), "THIRD_BODY");
    // Non-tool messages are never touched.
    assert_eq!(request.messages[0].text(), "sys");
    assert_eq!(request.messages[1].text(), "hello");
    assert_eq!(request.messages[3].text(), "thinking");
    // Cleared tool messages keep their tool_call_id.
    match &request.messages[2] {
        Message::Tool(t) => assert_eq!(t.tool_call_id, "t1"),
        other => panic!("expected tool message, got {other:?}"),
    }
}

#[tokio::test]
async fn microcompact_is_a_noop_when_within_keep_recent() {
    let mw = MicrocompactMiddleware::new(5, CLEARED);
    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "A"), Message::tool("t2", "B")],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), "A");
    assert_eq!(request.messages[1].text(), "B");
}

#[tokio::test]
async fn microcompact_is_idempotent() {
    let mw = MicrocompactMiddleware::new(1, CLEARED);
    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "FIRST"), Message::tool("t2", "SECOND")],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    // Second pass leaves the already-cleared body as the placeholder.
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), "SECOND");
}

// ── token-budget gate (issue tinyhumansai/openhuman#4755) ──────────────────────

#[tokio::test]
async fn microcompact_with_token_budget_is_a_noop_below_budget() {
    // More than `keep_recent` tool results, but the whole transcript fits the
    // configured budget: NOTHING is blanked, so the request stays byte-stable
    // across iterations and the provider KV-cache prefix is preserved. This is
    // the fix — the ungated middleware would have blanked the two oldest here
    // (see `microcompact_clears_older_tool_bodies_and_keeps_recent`), churning
    // the cache to reclaim tokens the model had ample room for.
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(100_000);
    assert_eq!(mw.token_budget(), Some(100_000));
    let mut request = ModelRequest {
        messages: vec![
            Message::system("sys"),
            Message::tool("t1", "FIRST_BODY"),
            Message::tool("t2", "SECOND_BODY"),
            Message::tool("t3", "THIRD_BODY"),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[1].text(), "FIRST_BODY");
    assert_eq!(request.messages[2].text(), "SECOND_BODY");
    assert_eq!(request.messages[3].text(), "THIRD_BODY");
}

#[tokio::test]
async fn microcompact_with_token_budget_blanks_once_over_budget() {
    // Same shape, but the (un-blanked) transcript exceeds the tiny budget, so
    // compaction is genuinely needed to stay under the window: the gate lets the
    // usual keep-recent blanking run and reclaims tokens.
    let body = "x".repeat(400); // ~100 estimated tokens each (chars / 4)
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(10);
    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", body.clone()),
            Message::tool("t2", body.clone()),
            Message::tool("t3", body.clone()),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), CLEARED);
    assert_eq!(request.messages[2].text(), body);
}

#[tokio::test]
async fn microcompact_with_token_budget_zero_disables_the_gate() {
    // `budget == 0` is an explicit opt-out: the gate is `None`, so blanking
    // reverts to the legacy count-only behaviour even on a tiny transcript.
    let mw = MicrocompactMiddleware::new(1, CLEARED).with_token_budget(0);
    assert_eq!(mw.token_budget(), None);
    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", "A"),
            Message::tool("t2", "B"),
            Message::tool("t3", "C"),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), CLEARED);
    assert_eq!(request.messages[1].text(), CLEARED);
    assert_eq!(request.messages[2].text(), "C");
}

#[tokio::test]
async fn microcompact_emits_no_event_by_default() {
    // Default construction is silent: bodies are still cleared, but no
    // Compressed event is emitted (parity with the OpenHuman in-house version).
    let mw = Arc::new(MicrocompactMiddleware::new(1, CLEARED));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: vec![Message::tool("t1", "FIRST"), Message::tool("t2", "SECOND")],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    assert_eq!(request.messages[0].text(), CLEARED);
    let events: Vec<AgentEvent> = recorder.events().into_iter().map(|r| r.event).collect();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Compressed { .. })),
        "no Compressed event should be emitted when events are off"
    );
}

#[tokio::test]
async fn microcompact_emits_compressed_event_when_enabled() {
    let mw = Arc::new(MicrocompactMiddleware::new(1, CLEARED).with_events(true));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let recorder = Arc::new(RecordingListener::new());
    let mut c = ctx();
    c.events.subscribe(recorder.clone());

    let mut request = ModelRequest {
        messages: vec![
            Message::tool("t1", "x".repeat(400)),
            Message::tool("t2", "y".repeat(400)),
        ],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();

    let compressed: Vec<(u64, u64)> = recorder
        .events()
        .into_iter()
        .filter_map(|r| match r.event {
            AgentEvent::Compressed {
                from_tokens,
                to_tokens,
            } => Some((from_tokens, to_tokens)),
            _ => None,
        })
        .collect();
    assert_eq!(
        compressed.len(),
        1,
        "one Compressed event when a body cleared"
    );
    assert!(
        compressed[0].0 > compressed[0].1,
        "tokens dropped after clear"
    );
}

#[tokio::test]
async fn usage_accounting_accumulates_across_calls() {
    let mw = Arc::new(UsageAccountingMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    let mut r1 = response_with_usage(Usage::new(10, 5));
    let mut r2 = response_with_usage(Usage::new(3, 2));
    stack.run_after_model(&mut c, &(), &mut r1).await.unwrap();
    stack.run_after_model(&mut c, &(), &mut r2).await.unwrap();

    let totals = mw.totals();
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.input_tokens, 13);
    assert_eq!(totals.usage.output_tokens, 7);
    assert_eq!(totals.usage.total_tokens, 20);
}

#[tokio::test]
async fn prompt_cache_guard_detects_prefix_change() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new());
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();

    // First call establishes a cacheable prefix [sys].
    let mut req1 = ModelRequest {
        cache_segments: vec![segment("sys", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req1)
        .await
        .unwrap();
    assert!(mw.layout_events().is_empty(), "no prior layout to compare");

    // Second call changes the stable prefix -> a layout event is recorded.
    let mut req2 = ModelRequest {
        cache_segments: vec![segment("sys2", SegmentRole::System, true)],
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut req2)
        .await
        .unwrap();

    let events = mw.layout_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].changed_prefix);
    assert_eq!(events[0].segment_ids_before, vec!["sys".to_string()]);
    assert_eq!(events[0].segment_ids_after, vec!["sys2".to_string()]);
}

#[tokio::test]
async fn prompt_cache_guard_events_are_bounded_by_max_events() {
    let mw = Arc::new(PromptCacheGuardMiddleware::new().with_max_events(2));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());

    let mut c = ctx();
    // Each call after the first changes the stable prefix, producing an
    // event; run far more iterations than the cap and confirm eviction.
    for i in 0..10 {
        let mut req = ModelRequest {
            cache_segments: vec![segment(&format!("sys{i}"), SegmentRole::System, true)],
            ..Default::default()
        };
        stack.run_before_model(&mut c, &(), &mut req).await.unwrap();
    }

    assert_eq!(mw.layout_events().len(), 2);
}

// ── wrap middleware: model ────────────────────────────────────────────────────

/// Builds a `ModelResponse` whose single text block is `text`.
fn response_text(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: None,
        },
        usage: None,
        finish_reason: None,
        raw: None,
        resolved_model: None,
    }
}

/// A model base that records how many times it was invoked and fails its first
/// `fail_times` calls (with a retryable-looking error) before succeeding.
struct CountingModelBase {
    calls: Arc<Mutex<usize>>,
    fail_times: usize,
    text: &'static str,
}

impl ModelBaseCall<(), ()> for CountingModelBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        _request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let attempt = {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                *n
            };
            if attempt <= self.fail_times {
                Err(TinyAgentsError::Middleware("transient".to_string()))
            } else {
                Ok(response_text(self.text))
            }
        })
    }
}

/// Wrap middleware that returns a canned response without calling `next`.
struct ShortCircuitModel {
    text: &'static str,
}

#[async_trait]
impl ModelMiddleware<()> for ShortCircuitModel {
    fn name(&self) -> &str {
        "short_circuit_model"
    }

    async fn wrap_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        _request: ModelRequest,
        _next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        Ok(MiddlewareModelOutcome::Response(response_text(self.text)))
    }
}

/// Wrap middleware that calls `next` then mutates the resulting response.
struct MutateAfterModel;

#[async_trait]
impl ModelMiddleware<()> for MutateAfterModel {
    fn name(&self) -> &str {
        "mutate_after_model"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut response = next.run(ctx, state, request).await?.into_response();
        response.finish_reason = Some("mutated".to_string());
        Ok(response.into())
    }
}

/// Wrap middleware that retries `next` up to `max` times until it succeeds.
struct RetryModel {
    max: usize,
}

#[async_trait]
impl ModelMiddleware<()> for RetryModel {
    fn name(&self) -> &str {
        "retry_model"
    }

    async fn wrap_model(
        &self,
        ctx: &mut RunContext,
        state: &(),
        request: ModelRequest,
        next: ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match next.run(ctx, state, request.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(_) if attempt < self.max => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

#[tokio::test]
async fn wrap_model_short_circuits_without_calling_base() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(ShortCircuitModel { text: "canned" }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "canned");
    // Base was never invoked because the wrap middleware short-circuited.
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn wrap_model_calls_next_then_mutates_response() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(MutateAfterModel));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    // Forwarded the base response, then mutated it.
    assert_eq!(response.text(), "from-base");
    assert_eq!(response.finish_reason.as_deref(), Some("mutated"));
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn wrap_model_retries_next_until_success() {
    let calls = Arc::new(Mutex::new(0));
    // Fails twice, succeeds on the third attempt.
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 2,
        text: "eventually",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(RetryModel { max: 5 }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "eventually");
    // Two failures + one success = three base invocations.
    assert_eq!(*calls.lock().unwrap(), 3);
}

#[tokio::test]
async fn wrap_model_onion_orders_outer_to_inner() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingModelBase {
        calls: calls.clone(),
        fail_times: 0,
        text: "base",
    };
    // Outer = mutate-after (sees the canned response from the inner layer and
    // stamps finish_reason); inner = short-circuit (never reaches base).
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_model_middleware(Arc::new(MutateAfterModel));
    stack.push_model_middleware(Arc::new(ShortCircuitModel { text: "canned" }));

    let mut c = ctx();
    let response = stack
        .run_wrapped_model(&mut c, &(), ModelRequest::default(), &base)
        .await
        .unwrap()
        .into_response();

    assert_eq!(response.text(), "canned");
    assert_eq!(response.finish_reason.as_deref(), Some("mutated"));
    // Inner layer short-circuited, so the base call never ran.
    assert_eq!(*calls.lock().unwrap(), 0);
    assert_eq!(stack.model_middleware_len(), 2);
}

// ── wrap middleware: tool ───────────────────────────────────────────────────--

/// A tool base that records invocations and fails its first `fail_times` calls.
struct CountingToolBase {
    calls: Arc<Mutex<usize>>,
    fail_times: usize,
    content: &'static str,
}

impl ToolBaseCall<(), ()> for CountingToolBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let attempt = {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                *n
            };
            if attempt <= self.fail_times {
                Err(TinyAgentsError::Middleware("transient".to_string()))
            } else {
                Ok(ToolResult {
                    call_id: call.id,
                    name: call.name,
                    content: self.content.to_string(),
                    raw: None,
                    error: None,
                    elapsed_ms: 0,
                })
            }
        })
    }
}

fn tool_call() -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        name: "fake".to_string(),
        arguments: serde_json::Value::Null,
        invalid: None,
    }
}

/// Wrap middleware that returns a canned result without calling `next`.
struct ShortCircuitTool {
    content: &'static str,
}

#[async_trait]
impl ToolMiddleware<()> for ShortCircuitTool {
    fn name(&self) -> &str {
        "short_circuit_tool"
    }

    async fn wrap_tool(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        call: ToolCall,
        _next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        Ok(MiddlewareToolOutcome::Result(ToolResult {
            call_id: call.id,
            name: call.name,
            content: self.content.to_string(),
            raw: None,
            error: None,
            elapsed_ms: 0,
        }))
    }
}

/// Wrap middleware that calls `next` then mutates the resulting result.
struct MutateAfterTool;

#[async_trait]
impl ToolMiddleware<()> for MutateAfterTool {
    fn name(&self) -> &str {
        "mutate_after_tool"
    }

    async fn wrap_tool(
        &self,
        ctx: &mut RunContext,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        let mut result = next.run(ctx, state, call).await?.into_result();
        result.content = format!("{}!", result.content);
        Ok(result.into())
    }
}

/// Wrap middleware that retries `next` up to `max` times until it succeeds.
struct RetryTool {
    max: usize,
}

#[async_trait]
impl ToolMiddleware<()> for RetryTool {
    fn name(&self) -> &str {
        "retry_tool"
    }

    async fn wrap_tool(
        &self,
        ctx: &mut RunContext,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match next.run(ctx, state, call.clone()).await {
                Ok(outcome) => return Ok(outcome),
                Err(_) if attempt < self.max => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

#[tokio::test]
async fn wrap_tool_short_circuits_without_calling_base() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 0,
        content: "from-base",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(ShortCircuitTool { content: "canned" }));

    let mut c = ctx();
    let result = stack
        .run_wrapped_tool(&mut c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.content, "canned");
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn wrap_tool_calls_next_then_mutates_result() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 0,
        content: "ok",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(MutateAfterTool));

    let mut c = ctx();
    let result = stack
        .run_wrapped_tool(&mut c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.content, "ok!");
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn wrap_tool_retries_next_until_success() {
    let calls = Arc::new(Mutex::new(0));
    let base = CountingToolBase {
        calls: calls.clone(),
        fail_times: 2,
        content: "eventually",
    };
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push_tool_middleware(Arc::new(RetryTool { max: 5 }));

    let mut c = ctx();
    let result = stack
        .run_wrapped_tool(&mut c, &(), tool_call(), &base)
        .await
        .unwrap()
        .into_result();

    assert_eq!(result.content, "eventually");
    assert_eq!(*calls.lock().unwrap(), 3);
    assert_eq!(stack.tool_middleware_len(), 1);
}

#[tokio::test]
async fn agent_run_text_reflects_final_response() {
    let mut run = AgentRun::new();
    assert_eq!(run.text(), None);
    run.final_response = Some(response_with_usage(Usage::new(1, 1)));
    assert_eq!(run.text(), Some("ok".to_string()));
}
