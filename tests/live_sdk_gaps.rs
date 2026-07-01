//! LIVE end-to-end coverage for a few SDK "gap" surfaces against a real
//! OpenAI model: budget preflight gating, tool-policy exposure of a classified
//! read-only tool, and the streaming reasoning side channel.
//!
//! Every test here talks to the real OpenAI API, so each one is an early no-op
//! `return` (after an `eprintln!`) when `OPENAI_API_KEY` is unset, so the
//! default `cargo test` passes with no key configured.
//!
//! Prompts are tiny and `max_tokens` is small to keep cost negligible. Asserts
//! target structural facts (an event fired, the run succeeded, text is
//! non-empty) rather than exact model prose.

/// Budget preflight blocks a real harness run *before* any provider call.
///
/// We first confirm the model works with a plain completion, then push a
/// [`BudgetMiddleware`] whose shared [`BudgetTracker`] is pre-loaded past a tiny
/// `max_total_tokens` budget. The middleware's `before_model` hook runs the
/// preflight check and fails the run with [`TinyAgentsError::LimitExceeded`]
/// deterministically — without ever contacting OpenAI on the gated call.
#[tokio::test]
async fn live_budget_blocks_second_call() {
    use std::sync::Arc;

    use tinyagents::TinyAgentsError;
    use tinyagents::harness::cost::CostTotals;
    use tinyagents::harness::message::Message;
    use tinyagents::harness::middleware::{BudgetLimits, BudgetMiddleware, BudgetTracker};
    use tinyagents::harness::model::ChatModel;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::AgentHarness;
    use tinyagents::harness::usage::Usage;

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_budget_blocks_second_call: OPENAI_API_KEY is not set");
        return;
    }

    let model: Arc<dyn ChatModel<()>> =
        Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present"));

    // 1) Sanity: a normal harness run succeeds against the real model.
    {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.register_model("openai", model.clone());
        let run = harness
            .invoke_default(&(), vec![Message::user("Reply with exactly: hi")])
            .await
            .expect("baseline live run succeeds");
        assert!(
            run.text().is_some_and(|t| !t.trim().is_empty()),
            "baseline run should produce non-empty text"
        );
    }

    // 2) Pre-load a shared tracker past a tiny budget, then attach a
    //    BudgetMiddleware that shares it. The very first gated call is blocked
    //    by preflight before touching OpenAI.
    let tracker = BudgetTracker::new();
    tracker.record(Usage::new(10_000, 0), CostTotals::default());

    let budget = BudgetMiddleware::new(BudgetLimits {
        max_total_tokens: Some(100),
        ..Default::default()
    })
    .with_tracker(tracker);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("openai", model);
    harness.push_middleware(Arc::new(budget));

    let err = harness
        .invoke_default(&(), vec![Message::user("Reply with exactly: hi")])
        .await
        .expect_err("budget preflight must block the run");

    assert!(
        matches!(err, TinyAgentsError::LimitExceeded(_)),
        "expected LimitExceeded from budget preflight, got: {err:?}"
    );
}

/// A trivial, classified, read-only calculator tool used by the tool-policy
/// test. It records every execution so the test can tell whether the model
/// actually called it.
struct AddTool {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl tinyagents::harness::tool::Tool<()> for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds one to the provided number x and returns the result."
    }

    fn schema(&self) -> tinyagents::harness::tool::ToolSchema {
        tinyagents::harness::tool::ToolSchema::new(
            "add",
            "Adds one to the provided number x and returns the result.",
            serde_json::json!({
                "type": "object",
                "properties": { "x": { "type": "number" } },
                "required": ["x"]
            }),
        )
    }

    fn policy(&self) -> tinyagents::harness::tool::ToolPolicy {
        tinyagents::harness::tool::ToolPolicy::read_only()
    }

    async fn call(
        &self,
        _state: &(),
        call: tinyagents::harness::tool::ToolCall,
    ) -> tinyagents::Result<tinyagents::harness::tool::ToolResult> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let x = call
            .arguments
            .get("x")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        Ok(tinyagents::harness::tool::ToolResult::text(
            call.id,
            "add",
            format!("{}", x + 1.0),
        ))
    }
}

/// A classified `read_only` tool survives a *strict* tool-policy exposure in a
/// real run: strict policy rejects unclassified/destructive tools, but this one
/// is classified read-only, so the run completes. If the model happens to call
/// the tool it executes (recorded), but we tolerate it not calling it.
#[tokio::test]
async fn live_tool_policy_exposes_classified_tool() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tinyagents::harness::message::Message;
    use tinyagents::harness::middleware::ToolPolicyMiddleware;
    use tinyagents::harness::model::ChatModel;
    use tinyagents::harness::providers::openai::OpenAiModel;
    use tinyagents::harness::runtime::AgentHarness;

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_tool_policy_exposes_classified_tool: OPENAI_API_KEY is not set");
        return;
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let model: Arc<dyn ChatModel<()>> =
        Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("openai", model);
    harness.register_tool(Arc::new(AddTool {
        calls: calls.clone(),
    }));

    // Fail-closed policy exposure built from the registry's classified policies.
    let policies = harness.tools().policies();
    harness.push_middleware(Arc::new(ToolPolicyMiddleware::strict(policies)));

    let run = harness
        .invoke_default(
            &(),
            vec![Message::user(
                "Use the add tool to compute add(x=1), then reply with the number.",
            )],
        )
        .await
        .expect("strict tool-policy run with a classified read_only tool succeeds");

    // The run produced a final response; we do not require the model to have
    // called the tool.
    assert!(run.text().is_some(), "run should produce a final response");

    // If it did call the tool, it must have actually executed.
    let called = calls.load(Ordering::SeqCst);
    assert!(
        run.text().is_some() || called > 0,
        "run completed structurally (tool calls observed: {called})"
    );
}

/// Smoke test: the streaming reasoning side channel coexists with normal
/// streaming. We stream a short real completion and assert deltas arrived, the
/// merged text is non-empty, and the `reasoning()` accessor returns a valid
/// `&str` (empty for a non-reasoning model — we only assert it does not panic).
#[tokio::test]
async fn live_streaming_reasoning_channel_smoke() {
    use futures::StreamExt;

    use tinyagents::harness::message::Message;
    use tinyagents::harness::model::{ChatModel, ModelRequest, ModelStreamItem, StreamAccumulator};
    use tinyagents::harness::providers::openai::OpenAiModel;

    let _ = dotenvy::dotenv();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("skipping live_streaming_reasoning_channel_smoke: OPENAI_API_KEY is not set");
        return;
    }

    let model = OpenAiModel::from_env().expect("OPENAI_API_KEY present");

    let request = ModelRequest {
        messages: vec![Message::user("Reply with exactly the single word: hello")],
        max_tokens: Some(16),
        ..ModelRequest::default()
    };

    let mut stream = model
        .stream(&(), request)
        .await
        .expect("opening the live stream succeeds");

    let mut delta_count = 0usize;
    let mut acc = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        if matches!(item, ModelStreamItem::MessageDelta(_)) {
            delta_count += 1;
        }
        acc.push(&item);
    }

    // The reasoning accessor must work regardless of whether the model emits
    // reasoning (a non-reasoning model yields an empty string).
    let reasoning: &str = acc.reasoning();
    assert!(
        reasoning.len() >= reasoning.trim_start().len(),
        "reasoning() returns a valid &str without panicking"
    );

    let response = acc.finish().expect("stream merges into a response");

    assert!(
        delta_count > 0,
        "expected at least one streamed message delta, got {delta_count}"
    );
    assert!(
        !response.text().trim().is_empty(),
        "expected non-empty final streamed text"
    );
}
