//! Tests for [`FinalCallWrapUpMiddleware`].

use std::sync::Arc;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::Middleware;
use tinyinference_llm::message::Message as TaMessage;
use tinyinference_llm::model::ModelRequest;
use tinyinference_llm::tool::ToolSchema;

/// A fixed set of captured outcomes, in the shape the wrap-up reads them.
struct Sink(Vec<(String, String)>);

impl CapturedOutcomes for Sink {
    fn content_for(
        &self,
        call_id: &str,
    ) -> std::result::Result<Option<String>, OutcomesUnavailable> {
        Ok(self
            .0
            .iter()
            .find(|(id, _)| id == call_id)
            .map(|(_, content)| content.clone()))
    }
}

fn sink_with(entries: &[(&str, &str)]) -> Arc<dyn CapturedOutcomes> {
    Arc::new(Sink(
        entries
            .iter()
            .map(|(id, content)| ((*id).to_string(), (*content).to_string()))
            .collect(),
    ))
}

/// The tools the OpenHuman host keeps for the penultimate call.
fn deliverables() -> [&'static str; 2] {
    ["file_write", "apply_patch"]
}

/// A run with calls left is untouched: no instruction, and the tool belt intact.
#[tokio::test]
async fn wrap_up_leaves_a_call_with_budget_remaining_alone() {
    let mw = FinalCallWrapUpMiddleware::new("CONCLUDE NOW", "WRITE NOW", sink_with(&[]), 0);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(5), ());
    ctx.limits.record_model_call().unwrap();
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: vec![ToolSchema::new("echo", "echo", serde_json::json!({}))],
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(
        request.messages.len(),
        1,
        "no instruction should be appended"
    );
    assert_eq!(request.tools.len(), 1, "the belt must stay intact mid-turn");
}

/// On the last permitted call the tools are withdrawn — structurally, not by
/// asking — and the wrap-up instruction is appended as the final turn.
#[tokio::test]
async fn wrap_up_withdraws_tools_and_appends_the_instruction_on_the_last_call() {
    let mw = FinalCallWrapUpMiddleware::new("CONCLUDE NOW", "WRITE NOW", sink_with(&[]), 0);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(2), ());
    ctx.limits.record_model_call().unwrap();
    ctx.limits.record_model_call().unwrap(); // now the final call
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: vec![ToolSchema::new("echo", "echo", serde_json::json!({}))],
        tool_choice: tinyinference_llm::model::ToolChoice::Required,
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert!(
        request.tools.is_empty(),
        "tools must be withdrawn, not discouraged"
    );
    assert!(
        matches!(
            request.tool_choice,
            tinyinference_llm::model::ToolChoice::None
        ),
        "a Required choice with no tools is a provider 400"
    );
    assert_eq!(
        request.messages.last().map(|m| m.text()),
        Some("CONCLUDE NOW".to_string()),
        "the instruction must be the final turn of the request"
    );
    assert!(mw.fired(&ctx));
    assert!(mw.fired_for(ctx.instance_id()));
}

/// The concluding call gets back the results microcompact blanked — otherwise
/// it is asked to report findings it cannot read, which is the same
/// empty-handed answer the whole mechanism exists to prevent.
#[tokio::test]
async fn wrap_up_restores_tool_results_microcompact_cleared() {
    let mw = FinalCallWrapUpMiddleware::new(
        "CONCLUDE NOW",
        "WRITE NOW",
        sink_with(&[
            ("call-old", "issue #41: auth bypass"),
            ("call-new", "issue #42: leak"),
        ]),
        // Unbounded: this case is about restoring what was cleared, not about
        // the budget that stops it (covered by its own test below).
        0,
    );
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(2), ());
    ctx.limits.record_model_call().unwrap();
    ctx.limits.record_model_call().unwrap();
    let mut request = ModelRequest {
        messages: vec![
            // The shape microcompact leaves behind: an older result blanked to
            // the placeholder, a recent one kept verbatim.
            TaMessage::tool("call-old", DEFAULT_CLEARED_PLACEHOLDER),
            TaMessage::tool("call-new", "issue #42: leak"),
        ],
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    let bodies: Vec<String> = request.messages.iter().map(|m| m.text()).collect();
    assert!(
        bodies.iter().any(|b| b.contains("issue #41: auth bypass")),
        "the cleared result should be restored from the capture sink: {bodies:?}"
    );
    assert!(
        !bodies
            .iter()
            .any(|b| b.trim() == DEFAULT_CLEARED_PLACEHOLDER),
        "no placeholder should survive into the concluding call: {bodies:?}"
    );
}

/// A result the model legitimately saw in full is never rewritten, even when
/// the sink holds a different (e.g. later-truncated) copy for that id.
#[tokio::test]
async fn wrap_up_does_not_rewrite_a_result_that_was_never_cleared() {
    let mw = FinalCallWrapUpMiddleware::new(
        "CONCLUDE NOW",
        "WRITE NOW",
        sink_with(&[("call-1", "FROM SINK")]),
        0,
    );
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(2), ());
    ctx.limits.record_model_call().unwrap();
    ctx.limits.record_model_call().unwrap();
    let mut request = ModelRequest {
        messages: vec![TaMessage::tool("call-1", "IN THE TRANSCRIPT")],
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(
        request.messages[0].text(),
        "IN THE TRANSCRIPT",
        "only a placeholder body may be replaced"
    );
}

fn mw(sink: Arc<dyn CapturedOutcomes>) -> FinalCallWrapUpMiddleware {
    FinalCallWrapUpMiddleware::new("CONCLUDE NOW", "WRITE NOW", sink, 0)
        .with_deliverable_tools(deliverables())
}

/// A context sitting on the Nth call of an N-call budget, minus `back`.
fn ctx_at(max: usize, back: usize) -> RunContext {
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(max), ());
    for _ in 0..(max - back) {
        ctx.limits.record_model_call().unwrap();
    }
    ctx
}

/// A mixed belt: two gatherers and the two writers.
fn mixed_belt() -> Vec<ToolSchema> {
    ["web_fetch", "file_write", "shell", "apply_patch"]
        .iter()
        .map(|n| ToolSchema::new(*n, *n, serde_json::json!({})))
        .collect()
}

fn names(request: &ModelRequest) -> Vec<String> {
    request.tools.iter().map(|t| t.name.clone()).collect()
}

/// The defect this exists for: on the call before the conclusion the writers
/// survive, so a turn that owes a file can still write it.
#[tokio::test]
async fn penultimate_call_keeps_the_writers_and_drops_the_gatherers() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(15, 1);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("summarise the baggage rules into a file")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(
        names(&request),
        vec!["file_write", "apply_patch"],
        "only the tools that can emit a deliverable may survive"
    );
    assert_eq!(
        request.messages.last().map(|m| m.text()),
        Some("WRITE NOW".to_string()),
        "the write instruction must be the final turn of the request"
    );
    assert!(
        !mw.fired(&ctx),
        "this is not the conclusion: the turn has one more call and must not \
         be reported as capped yet"
    );
}

/// `Auto`, never `Required`. A turn that has already written its file, or was
/// only ever asked for an answer, must be free to spend this call on text —
/// forcing a call would make it invent a write.
#[tokio::test]
async fn penultimate_call_leaves_the_model_free_not_to_write() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(15, 1);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        tool_choice: tinyinference_llm::model::ToolChoice::Required,
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert!(matches!(
        request.tool_choice,
        tinyinference_llm::model::ToolChoice::Auto
    ));
}

/// Asked to write its findings into a file, this call has to be able to read
/// them — so it gets the same restoration the conclusion does.
#[tokio::test]
async fn penultimate_call_restores_what_microcompact_cleared() {
    let mw = mw(sink_with(&[("call-old", "carry-on max 22 x 14 x 9 in")]));
    let mut ctx = ctx_at(15, 1);
    let mut request = ModelRequest {
        messages: vec![
            TaMessage::tool("call-old", DEFAULT_CLEARED_PLACEHOLDER),
            TaMessage::tool("call-new", "checked bag 50 lb"),
        ],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    let bodies: Vec<String> = request.messages.iter().map(|m| m.text()).collect();
    assert!(
        bodies.iter().any(|b| b.contains("22 x 14 x 9")),
        "a call asked to write findings down must be able to read them: {bodies:?}"
    );
}

/// A belt with no writer on it gets no narrowing and no instruction: telling a
/// read-only agent that "the only tools left are the ones that write files"
/// would simply be false.
#[tokio::test]
async fn a_belt_without_a_writer_is_left_alone() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(15, 1);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: vec![
            ToolSchema::new("web_fetch", "web_fetch", serde_json::json!({})),
            ToolSchema::new("shell", "shell", serde_json::json!({})),
        ],
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(names(&request), vec!["web_fetch", "shell"]);
    assert_eq!(
        request.messages.len(),
        1,
        "no instruction should be appended"
    );
}

/// Reserving out of a two-call budget would leave no round in which anything
/// could be gathered to write, so it is skipped: one tool round, then the
/// conclusion, exactly as before.
#[tokio::test]
async fn a_two_call_budget_keeps_its_one_gathering_round() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(2, 1);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(
        names(&request),
        vec!["web_fetch", "file_write", "shell", "apply_patch"],
        "the belt must stay whole when there is no room to reserve a call"
    );
    assert_eq!(
        request.messages.len(),
        1,
        "no instruction should be appended"
    );
}

/// Two calls earlier nothing has happened yet — the reservation is for the
/// penultimate call alone, not for the tail of the turn.
#[tokio::test]
async fn the_call_before_the_penultimate_one_is_untouched() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(15, 2);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(request.tools.len(), 4, "the belt must stay intact mid-turn");
    assert_eq!(request.messages.len(), 1);
}

/// And the call after it still clears the belt outright: narrowing buys the
/// artifact, the conclusion still has to be text.
#[tokio::test]
async fn the_conclusion_still_withdraws_everything() {
    let mw = mw(sink_with(&[]));
    let mut ctx = ctx_at(15, 0);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert!(
        request.tools.is_empty(),
        "the final call keeps nothing, writers included"
    );
    assert_eq!(
        request.messages.last().map(|m| m.text()),
        Some("CONCLUDE NOW".to_string())
    );
    assert!(mw.fired(&ctx));
}

// ── the bounds CodeRabbit asked for on #6068 ────────────────────────────────

/// Restoration stops at the input allowance instead of overflowing it.
///
/// It runs after compression and before the trim, so an unbounded restore
/// pushes the request over and the trim evicts whole messages — strictly more
/// destructive than the blanking being undone, and able to discard the very
/// results just restored. Preventing the overflow beats repairing it.
#[tokio::test]
async fn wrap_up_stops_restoring_at_the_input_budget() {
    let big = "x".repeat(4_000);
    let mw = FinalCallWrapUpMiddleware::new(
        "CONCLUDE NOW",
        "WRITE NOW",
        sink_with(&[("call-1", &big), ("call-2", &big), ("call-3", &big)]),
        // Room for roughly one of them, not three.
        1_200,
    );
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(2), ());
    ctx.limits.record_model_call().unwrap();
    ctx.limits.record_model_call().unwrap();
    let mut request = ModelRequest {
        messages: vec![
            TaMessage::tool("call-1", DEFAULT_CLEARED_PLACEHOLDER),
            TaMessage::tool("call-2", DEFAULT_CLEARED_PLACEHOLDER),
            TaMessage::tool("call-3", DEFAULT_CLEARED_PLACEHOLDER),
        ],
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    let restored = request
        .messages
        .iter()
        .filter(|m| m.text().starts_with("xxx"))
        .count();
    assert!(
        (1..3).contains(&restored),
        "some restored, not all — budget 1200 cannot hold three 4k bodies, got {restored}"
    );
    // Newest-first: the last tool result is the one the model never saw (the
    // cap is checked before the request is built). Not `messages.last()` —
    // that is the wrap-up instruction this middleware just appended.
    let newest_tool = request
        .messages
        .iter()
        .rev()
        .find(|m| matches!(m, TaMessage::Tool(_)))
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        newest_tool.starts_with("xxx"),
        "the newest cleared result is restored first, got: {}",
        &newest_tool[..newest_tool.len().min(60)]
    );
}

/// The other half of the no-window split: restoration must stop too. Without a
/// window there is no trim middleware behind it, so an unbounded wrap-up would
/// hand the provider a request it rejects — losing the in-loop conclusion the
/// whole cap-checkpoint path exists to deliver (CodeRabbit on #6068).
#[tokio::test]
async fn wrap_up_restoration_stays_bounded_when_no_window_is_advertised() {
    let big = "x".repeat(8_000);
    let entries: Vec<(String, String)> = (0..40)
        .map(|i| (format!("call-{i}"), big.clone()))
        .collect();
    let borrowed: Vec<(&str, &str)> = entries
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let (_toc, restore) = split_input_allowance(0);
    let mw =
        FinalCallWrapUpMiddleware::new("CONCLUDE NOW", "WRITE NOW", sink_with(&borrowed), restore);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(2), ());
    ctx.limits.record_model_call().unwrap();
    ctx.limits.record_model_call().unwrap();
    let mut request = ModelRequest {
        messages: borrowed
            .iter()
            .map(|(id, _)| TaMessage::tool(*id, DEFAULT_CLEARED_PLACEHOLDER))
            .collect(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    let used: u64 = request.messages.iter().map(estimate_message_tokens).sum();
    assert!(
        used <= NO_WINDOW_ALLOWANCE,
        "the no-window fallback must bound the whole request, not just the list: {used}"
    );
    let restored = request
        .messages
        .iter()
        .filter(|m| m.text().starts_with("xxx"))
        .count();
    assert!(
        (1..40).contains(&restored),
        "some restored, not all 40 — got {restored}"
    );
}

// ── the budget notice (openhuman#6958) ──────────────────────────────────────

/// Drive one run of `max` model calls through the middleware and return, per
/// call number (1-based), the text of whatever it appended to a fresh request.
async fn appended_per_call(
    mw: &FinalCallWrapUpMiddleware,
    ctx: &mut RunContext,
    max: usize,
) -> Vec<(usize, String)> {
    let mut appended = Vec::new();
    for call in 1..=max {
        ctx.limits.record_model_call().unwrap();
        let mut request = ModelRequest {
            messages: vec![TaMessage::user("fix the bug")],
            tools: mixed_belt(),
            ..Default::default()
        };
        mw.before_model(ctx, &(), &mut request).await.unwrap();
        if request.messages.len() > 1 {
            appended.push((call, request.messages.last().unwrap().text()));
        }
    }
    appended
}

/// Without opting in, nothing is said about the budget until the wrap-up.
#[tokio::test]
async fn no_budget_notice_unless_configured() {
    let mw = mw(sink_with(&[]));
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(20), ());

    let appended = appended_per_call(&mw, &mut ctx, 20).await;

    let calls: Vec<usize> = appended.iter().map(|(call, _)| *call).collect();
    assert_eq!(calls, vec![19, 20], "only the wrap-up pair may speak");
}

/// The defect: the model first heard of its budget on the penultimate call.
/// With the notice on it hears once at half the budget and once at 80%, each
/// stating how many calls are left — and never again for the same threshold.
#[tokio::test]
async fn budget_notice_fires_once_at_each_threshold() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5, 0.8]);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(20), ());

    let appended = appended_per_call(&mw, &mut ctx, 20).await;

    let calls: Vec<usize> = appended.iter().map(|(call, _)| *call).collect();
    assert_eq!(
        calls,
        vec![10, 16, 19, 20],
        "a notice at 50% and 80%, then the wrap-up pair: {appended:?}"
    );
    assert!(
        appended[0].1.contains("10 model calls left"),
        "the 50% notice states the remaining budget: {}",
        appended[0].1
    );
    assert!(
        appended[1].1.contains("4 model calls left"),
        "the 80% notice states the remaining budget: {}",
        appended[1].1
    );
}

/// The notice says how many calls are left and to finish, but names no kind
/// of task: a browsing or shopping agent has no files to edit.
#[tokio::test]
async fn budget_notice_names_no_kind_of_task() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5]);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(20), ());

    let appended = appended_per_call(&mw, &mut ctx, 18).await;

    let (call, notice) = &appended[0];
    assert_eq!(*call, 10, "{appended:?}");
    assert!(notice.contains("10 model calls left"), "{notice}");
    assert!(notice.contains("finishing the task now"), "{notice}");
    for word in ["file", "edit", "deliverable"] {
        assert!(!notice.contains(word), "{word:?} in {notice}");
    }
}

/// A notice must not report the turn as capped: only the conclusion does.
#[tokio::test]
async fn budget_notice_does_not_mark_the_turn_capped() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5]);
    let mut ctx = ctx_at(20, 10);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(request.messages.len(), 2, "the 50% notice is appended");
    assert_eq!(request.tools.len(), 4, "a notice never narrows the belt");
    assert!(!mw.fired(&ctx));
}

/// Two thresholds crossed by the same call produce one notice, not two
/// stacked messages, and neither fires again.
#[tokio::test]
async fn thresholds_crossed_together_produce_one_notice() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5, 0.6]);
    // The first call this middleware sees is already past both thresholds
    // (6 of 10 used), as on a run whose earlier calls it did not observe.
    let mut ctx = ctx_at(10, 4);
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };

    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();

    assert_eq!(request.messages.len(), 2, "exactly one notice for both");
    assert!(request.messages[1].text().contains("4 model calls left"));

    ctx.limits.record_model_call().unwrap();
    let mut next = ModelRequest {
        messages: vec![TaMessage::user("hi")],
        tools: mixed_belt(),
        ..Default::default()
    };
    mw.before_model(&mut ctx, &(), &mut next).await.unwrap();
    assert_eq!(next.messages.len(), 1, "neither threshold fires again");
}

/// A notice never lands on the penultimate or final call: those carry their
/// own instruction, and a second voice there only competes with it.
#[tokio::test]
async fn budget_notice_yields_to_the_wrap_up_calls() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.8, 0.95]);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(5), ());

    let appended = appended_per_call(&mw, &mut ctx, 5).await;

    assert_eq!(
        appended,
        vec![
            (4, "WRITE NOW".to_string()),
            (5, "CONCLUDE NOW".to_string())
        ],
        "80% of 5 is the penultimate call and 95% the last: no notice fits"
    );
}

/// Each run gets its own notices: the bookkeeping is per run context, not
/// per middleware instance.
#[tokio::test]
async fn budget_notice_is_tracked_per_run() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5]);
    let mut first = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(10), ());
    let mut second = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(10), ());

    let a = appended_per_call(&mw, &mut first, 8).await;
    let b = appended_per_call(&mw, &mut second, 8).await;

    assert_eq!(a.len(), 1, "{a:?}");
    assert_eq!(b.len(), 1, "a second run hears its own notice: {b:?}");
}

/// Thresholds outside (0, 1) mean nothing as a fraction of a budget and are
/// ignored rather than firing on the first call or never.
#[tokio::test]
async fn out_of_range_thresholds_are_ignored() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.0, -1.0, 1.0, 2.5, f64::NAN]);
    let mut ctx = RunContext::new(RunConfig::new("mw-test").with_max_model_calls(10), ());

    let appended = appended_per_call(&mw, &mut ctx, 10).await;

    let calls: Vec<usize> = appended.iter().map(|(call, _)| *call).collect();
    assert_eq!(calls, vec![9, 10], "{appended:?}");
}

// ── nudges ride the ephemeral-instruction placement (#6962) ─────────────────

fn hoisting() -> tinyinference_llm::model::ModelProfile {
    tinyinference_llm::model::ModelProfile {
        hoists_system_messages: true,
        ..Default::default()
    }
}

/// On a hoisting model a nudge folds into the tail user turn, never a second
/// consecutive user turn.
async fn assert_folds_into_tail(
    mw: &FinalCallWrapUpMiddleware,
    max: usize,
    back: usize,
    needle: &str,
) {
    let mut ctx = ctx_at(max, back);
    ctx.model_profile = Some(hoisting());
    let mut request = ModelRequest {
        messages: vec![TaMessage::user("fix the bug")],
        tools: mixed_belt(),
        ..Default::default()
    };
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    assert_eq!(
        request.messages.len(),
        1,
        "no extra turn: {:?}",
        request.messages
    );
    let text = request.messages[0].text();
    assert!(text.starts_with("fix the bug"), "{text}");
    assert!(text.contains(needle), "{text}");
}

#[tokio::test]
async fn budget_notice_folds_into_the_tail_turn_on_hoisting_models() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5]);
    assert_folds_into_tail(&mw, 20, 10, "model calls left").await;
}

#[tokio::test]
async fn final_write_instruction_folds_into_the_tail_turn_on_hoisting_models() {
    assert_folds_into_tail(&mw(sink_with(&[])), 20, 1, "WRITE NOW").await;
}

#[tokio::test]
async fn conclusion_instruction_folds_into_the_tail_turn_on_hoisting_models() {
    assert_folds_into_tail(&mw(sink_with(&[])), 20, 0, "CONCLUDE NOW").await;
}

/// The check and the budget notice must not both speak in one run.
#[tokio::test]
async fn announced_budget_notice_is_visible_to_the_run_that_got_it() {
    let mw = mw(sink_with(&[])).with_budget_notice([0.5]);
    let mut ctx = ctx_at(20, 10);
    assert!(!mw.budget_notice_announced(&ctx));
    let mut request = ModelRequest::new(vec![TaMessage::user("x")]);
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    assert!(mw.budget_notice_announced(&ctx));
    assert!(!mw.budget_notice_announced(&ctx_at(20, 10)));
}
