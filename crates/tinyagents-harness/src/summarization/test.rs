//! Tests for trimming, summarization, and compression policies.
//!
//! Cover the [`estimate_tokens`] heuristic (clamping and ~4-chars-per-token
//! scaling), every [`TrimStrategy`] variant (including system-message retention
//! and the last-resort drop), [`ConcatSummarizer`] output and provenance, and
//! [`SummarizationPolicy`] gating — both the raw `trigger_tokens` path and the
//! context-window-aware `threshold_fraction` path, plus `plan` splitting.

#[cfg(test)]
mod smoke {
    use crate::message::Message;
    use crate::summarization::{
        ConcatSummarizer, SummarizationPolicy, Summarizer, TokenTrimPolicy, TrimStrategy,
        estimate_tokens, trim_messages, trim_messages_to_token_budget_with,
    };

    /// Verify that `estimate_tokens` produces a non-zero value for a non-empty
    /// string and zero for an empty string.
    #[test]
    fn estimate_tokens_basic() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("hello world") > 0);
    }

    /// `trim_messages` with `KeepLast(1)` retains the last non-system message
    /// and all system messages.
    #[test]
    fn trim_keep_last_preserves_system() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("first"),
            Message::user("second"),
        ];
        let trimmed = trim_messages(&msgs, &TrimStrategy::KeepLast(1));
        // system + last user
        assert_eq!(trimmed.len(), 2);
        assert!(matches!(trimmed[0], Message::System(_)));
        assert_eq!(trimmed[1].text(), "second");
    }

    /// `SummarizationPolicy::should_summarize` returns false when messages are short.
    #[test]
    fn policy_should_not_summarize_short_messages() {
        let policy = SummarizationPolicy {
            trigger_tokens: 10_000,
            keep_last: 4,
            ..Default::default()
        };
        let msgs = vec![Message::user("hi"), Message::assistant("hello")];
        assert!(!policy.should_summarize(&msgs));
    }

    /// `ConcatSummarizer` produces a non-empty system summary with provenance.
    #[tokio::test]
    async fn concat_summarizer_produces_record() {
        let summarizer = ConcatSummarizer;
        let msgs = vec![Message::user("a"), Message::assistant("b")];
        let record = summarizer.summarize(&msgs).await.expect("summarize failed");
        assert!(!record.summary.text().is_empty());
        assert_eq!(record.provenance.source_ids, vec!["msg-0", "msg-1"]);
        assert!(record.provenance.original_token_estimate > 0);
    }

    // ── estimate_tokens edge cases ────────────────────────────────────────────

    #[test]
    fn estimate_tokens_clamps_short_and_scales_long() {
        // Empty → 0.
        assert_eq!(estimate_tokens(""), 0);
        // Any non-empty short string clamps to at least 1.
        assert_eq!(estimate_tokens("x"), 1);
        assert_eq!(estimate_tokens("abc"), 1);
        // ~4 chars per token for longer text.
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        let long = "a".repeat(400);
        assert_eq!(estimate_tokens(&long), 100);
    }

    // ── TrimStrategy variants ─────────────────────────────────────────────────

    #[test]
    fn trim_keep_first_and_last() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("u1"),
            Message::user("u2"),
            Message::user("u3"),
            Message::user("u4"),
        ];
        let trimmed = trim_messages(&msgs, &TrimStrategy::KeepFirstAndLast { first: 1, last: 1 });
        // system + first non-system + last non-system.
        assert_eq!(trimmed.len(), 3);
        assert!(matches!(trimmed[0], Message::System(_)));
        assert_eq!(trimmed[1].text(), "u1");
        assert_eq!(trimmed[2].text(), "u4");
    }

    #[test]
    fn trim_keep_first_and_last_no_overlap_keeps_all() {
        let msgs = vec![Message::user("u1"), Message::user("u2")];
        // first + last >= len → keep everything.
        let trimmed = trim_messages(&msgs, &TrimStrategy::KeepFirstAndLast { first: 2, last: 2 });
        assert_eq!(trimmed.len(), 2);
    }

    #[test]
    fn trim_keep_last_more_than_available() {
        let msgs = vec![Message::user("only")];
        let trimmed = trim_messages(&msgs, &TrimStrategy::KeepLast(5));
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].text(), "only");
    }

    #[test]
    fn token_budget_trim_preserves_order_and_system_messages() {
        let messages = vec![
            Message::user("old"),
            Message::system("late-system"),
            Message::assistant("new"),
        ];
        let policy = TokenTrimPolicy::strict(2).preserve_system();
        let trimmed = trim_messages_to_token_budget_with(&messages, policy, |_| 2);

        assert_eq!(trimmed, vec![Message::system("late-system")]);
    }

    #[test]
    fn token_budget_trim_uses_caller_estimator() {
        let messages = vec![Message::user("large-image"), Message::user("small")];
        let policy = TokenTrimPolicy::strict(2);
        let trimmed = trim_messages_to_token_budget_with(&messages, policy, |message| {
            if message.text() == "large-image" {
                10
            } else {
                2
            }
        });

        assert_eq!(trimmed, vec![Message::user("small")]);
    }

    #[test]
    fn token_budget_trim_drops_leading_orphan_tool_results() {
        let messages = vec![
            Message::assistant("old"),
            Message::tool("call-1", "result"),
            Message::user("new"),
        ];
        let policy = TokenTrimPolicy::strict(2).drop_leading_orphan_tools();
        let trimmed = trim_messages_to_token_budget_with(&messages, policy, |_| 1);

        assert_eq!(trimmed, vec![Message::user("new")]);
    }

    #[test]
    fn token_budget_trim_drops_orphan_tool_results_when_under_budget() {
        let messages = vec![
            Message::system("policy"),
            Message::tool("call-1", "orphan"),
            Message::user("new"),
        ];
        let policy = TokenTrimPolicy::strict(100).drop_leading_orphan_tools();
        let trimmed = trim_messages_to_token_budget_with(&messages, policy, |_| 1);

        assert_eq!(
            trimmed,
            vec![Message::system("policy"), Message::user("new")]
        );
    }

    #[test]
    fn trim_max_tokens_drops_oldest_non_system_first() {
        // Each user message ~ "aaaaaaaa" (8 chars) → 2 tokens. System "ssss" (4) → 1 token.
        let msgs = vec![
            Message::system("ssss"),
            Message::user("aaaaaaaa"),
            Message::user("bbbbbbbb"),
            Message::user("cccccccc"),
        ];
        // Budget allows system (1) + at most ~2 user messages (4) = 5 tokens.
        let trimmed = trim_messages(&msgs, &TrimStrategy::MaxTokens(5));
        // System always kept; oldest user dropped from the front.
        assert!(matches!(trimmed[0], Message::System(_)));
        let texts: Vec<String> = trimmed.iter().map(|m| m.text()).collect();
        assert!(texts.contains(&"ssss".to_string()));
        assert!(texts.contains(&"cccccccc".to_string()));
        assert!(!texts.contains(&"aaaaaaaa".to_string()));
    }

    #[test]
    fn trim_max_tokens_drops_system_as_last_resort() {
        // Tiny budget that cannot fit even one message: system is dropped too.
        let msgs = vec![
            Message::system("a very long system instruction string here"),
            Message::user("a very long user message string here too ok"),
        ];
        let trimmed = trim_messages(&msgs, &TrimStrategy::MaxTokens(1));
        // Everything is shed to meet the impossible budget.
        assert!(trimmed.is_empty());
    }

    // ── ConcatSummarizer provenance detail ────────────────────────────────────

    #[tokio::test]
    async fn concat_summarizer_empty_is_error() {
        let summarizer = ConcatSummarizer;
        assert!(summarizer.summarize(&[]).await.is_err());
    }

    #[tokio::test]
    async fn concat_summarizer_provenance_fields() {
        let summarizer = ConcatSummarizer;
        let msgs = vec![
            Message::system("sys"),
            Message::user("hello there"),
            Message::assistant("general kenobi"),
        ];
        let record = summarizer.summarize(&msgs).await.unwrap();

        // Summary is a system message.
        assert!(matches!(record.summary, Message::System(_)));
        // One synthetic id per source message, in order.
        assert_eq!(
            record.provenance.source_ids,
            vec!["msg-0", "msg-1", "msg-2"]
        );
        // Reason names the summarizer.
        assert!(record.provenance.reason.contains("ConcatSummarizer"));
        // Token estimates are populated.
        assert!(record.provenance.original_token_estimate > 0);
        assert!(record.provenance.summary_token_estimate > 0);
        // Role labels appear in the rendered summary.
        let text = record.summary.text();
        assert!(text.contains("system:"));
        assert!(text.contains("user:"));
        assert!(text.contains("assistant:"));
    }

    // ── SummarizationPolicy ───────────────────────────────────────────────────

    #[test]
    fn policy_should_summarize_over_trigger() {
        let policy = SummarizationPolicy {
            trigger_tokens: 2,
            keep_last: 1,
            ..Default::default()
        };
        // ~16 chars → 4 tokens > trigger 2.
        let msgs = vec![Message::user("aaaaaaaaaaaaaaaa")];
        assert!(policy.should_summarize(&msgs));
    }

    #[test]
    fn policy_plan_splits_keeping_system_and_recent() {
        let policy = SummarizationPolicy {
            trigger_tokens: 0,
            keep_last: 2,
            ..Default::default()
        };
        let msgs = vec![
            Message::system("sys"),
            Message::user("old1"),
            Message::user("old2"),
            Message::user("recent1"),
            Message::assistant("recent2"),
        ];
        let (to_summarize, to_keep) = policy.plan(&msgs);

        // Oldest two non-system messages are summarized.
        let sum_texts: Vec<String> = to_summarize.iter().map(|m| m.text()).collect();
        assert_eq!(sum_texts, vec!["old1", "old2"]);

        // System is kept verbatim plus the last `keep_last` non-system messages.
        assert!(matches!(to_keep[0], Message::System(_)));
        let keep_texts: Vec<String> = to_keep.iter().map(|m| m.text()).collect();
        assert_eq!(keep_texts, vec!["sys", "recent1", "recent2"]);
    }

    // ── context-window-aware triggering ───────────────────────────────────────

    #[test]
    fn policy_below_window_threshold_does_not_summarize() {
        // 1000-token window, 0.9 threshold → budget 900 tokens.
        let policy = SummarizationPolicy::default()
            .with_context_window(1000)
            .with_threshold_fraction(0.9);
        assert_eq!(policy.trigger_budget(), 900);

        // ~400 chars → ~100 tokens, far below the 900-token budget.
        let msgs = vec![Message::user("a".repeat(400))];
        assert!(!policy.should_summarize(&msgs));
    }

    #[test]
    fn policy_at_or_above_window_threshold_summarizes() {
        // 100-token window, 0.5 threshold → budget 50 tokens.
        let policy = SummarizationPolicy::default()
            .with_context_window(100)
            .with_threshold_fraction(0.5);
        assert_eq!(policy.trigger_budget(), 50);

        // Exactly at the budget: 200 chars → 50 tokens (>= 50 triggers).
        let at = vec![Message::user("a".repeat(200))];
        assert!(policy.should_summarize(&at));

        // Above the budget triggers too.
        let above = vec![Message::user("a".repeat(400))];
        assert!(policy.should_summarize(&above));

        // Below the budget does not.
        let below = vec![Message::user("a".repeat(100))];
        assert!(!policy.should_summarize(&below));
    }

    #[test]
    fn policy_none_window_falls_back_to_trigger_tokens() {
        // No context window → use raw trigger_tokens with strict `>` semantics.
        let policy = SummarizationPolicy {
            trigger_tokens: 2,
            keep_last: 1,
            ..Default::default()
        };
        assert_eq!(policy.context_window, None);
        assert_eq!(policy.trigger_budget(), 2);

        // ~16 chars → 4 tokens > 2.
        let over = vec![Message::user("aaaaaaaaaaaaaaaa")];
        assert!(policy.should_summarize(&over));

        // ~4 chars → 1 token, not > 2.
        let under = vec![Message::user("aaaa")];
        assert!(!policy.should_summarize(&under));
    }

    #[test]
    fn policy_default_threshold_is_ninety_percent() {
        let policy = SummarizationPolicy::default();
        assert!((policy.threshold_fraction - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn policy_from_profile_reads_max_input_tokens() {
        use crate::model::ModelProfile;

        let profile = ModelProfile {
            max_input_tokens: Some(1000),
            ..Default::default()
        };
        let policy = SummarizationPolicy::from_profile(&profile, 0.8);
        assert_eq!(policy.context_window, Some(1000));
        assert!((policy.threshold_fraction - 0.8).abs() < f64::EPSILON);
        assert_eq!(policy.trigger_budget(), 800);

        // A profile without max_input_tokens leaves the window None (fallback).
        let bare = ModelProfile::default();
        let fallback = SummarizationPolicy::from_profile(&bare, 0.9);
        assert_eq!(fallback.context_window, None);
    }

    #[test]
    fn policy_plan_keeps_everything_when_few_messages() {
        let policy = SummarizationPolicy {
            trigger_tokens: 0,
            keep_last: 5,
            ..Default::default()
        };
        let msgs = vec![Message::system("sys"), Message::user("u")];
        let (to_summarize, to_keep) = policy.plan(&msgs);
        assert!(to_summarize.is_empty());
        assert_eq!(to_keep.len(), 2);
    }
}

/// Regression tests for the structural repair of transcript cut points.
///
/// Every test here is written against the concrete provider failure it
/// prevents: a `role:"tool"` message with no preceding assistant `tool_calls`
/// (OpenAI `400`, Anthropic "tool_result with no matching tool_use"), or the
/// mirror image, an assistant `tool_calls` entry nothing ever answers.
///
/// Before the repair landed, `plan` and all three [`TrimStrategy`] variants cut
/// at a blind index and produced exactly those shapes.
#[cfg(test)]
mod pairing {
    use crate::message::{AssistantMessage, ContentBlock, Message};
    use crate::summarization::{
        MessageRole, SummarizationPolicy, TrimOptions, TrimStrategy, tool_pairing_is_intact,
        trim_messages, trim_messages_with,
    };
    use crate::tool::ToolCall;
    use serde_json::json;

    /// An assistant turn that only calls tools: no visible text at all, which
    /// is precisely the shape that used to estimate to zero tokens and to be
    /// severed from its results by a blind cut.
    fn assistant_calling(ids: &[&str]) -> Message {
        Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: ids
                .iter()
                .map(|id| ToolCall::new(*id, "lookup", json!({"q": "rust"})))
                .collect(),
            usage: None,
        })
    }

    /// `[system, user, assistant(tool_calls=[c1]), tool(c1), assistant("done")]`
    /// — the canonical transcript that a `keep_last = 2` cut splits.
    fn tool_transcript() -> Vec<Message> {
        vec![
            Message::system("sys"),
            Message::user("weather?"),
            assistant_calling(&["c1"]),
            Message::tool("c1", "sunny"),
            Message::assistant("done"),
        ]
    }

    #[test]
    fn plan_does_not_orphan_a_tool_result() {
        let policy = SummarizationPolicy {
            keep_last: 2,
            ..Default::default()
        };
        let (to_summarize, to_keep) = policy.plan(&tool_transcript());

        // The assistant tool-call turn must travel with its result.
        assert!(
            tool_pairing_is_intact(&to_keep),
            "kept slice orphans a tool result: {to_keep:?}"
        );
        assert!(
            !to_summarize
                .iter()
                .any(|m| matches!(m, Message::Assistant(a) if !a.tool_calls.is_empty())),
            "the assistant tool-call turn was summarized away from its result"
        );
        // keep_last is a minimum: the repair kept one extra message.
        assert_eq!(to_keep.len(), 4);
    }

    #[test]
    fn keep_last_does_not_orphan_a_tool_result() {
        let trimmed = trim_messages(&tool_transcript(), &TrimStrategy::KeepLast(2));
        assert!(
            tool_pairing_is_intact(&trimmed),
            "KeepLast orphaned a tool result: {trimmed:?}"
        );
    }

    #[test]
    fn keep_first_and_last_does_not_orphan_either_end() {
        // Head block ends on an assistant tool-call turn; tail block starts on
        // a tool result. Both ends are broken without repair.
        let messages = vec![
            Message::user("one"),
            assistant_calling(&["c1"]),
            Message::tool("c1", "r1"),
            Message::user("two"),
            assistant_calling(&["c2"]),
            Message::tool("c2", "r2"),
            Message::assistant("done"),
        ];
        let trimmed = trim_messages(
            &messages,
            &TrimStrategy::KeepFirstAndLast { first: 2, last: 2 },
        );
        assert!(
            tool_pairing_is_intact(&trimmed),
            "KeepFirstAndLast produced an unpaired slice: {trimmed:?}"
        );
    }

    #[test]
    fn max_tokens_drops_orphan_tool_results_rather_than_readmitting_tokens() {
        let messages = vec![
            Message::user("a".repeat(400)),
            assistant_calling(&["c1"]),
            Message::tool("c1", "r1"),
            Message::assistant("done"),
        ];
        let trimmed = trim_messages(&messages, &TrimStrategy::MaxTokens(4));
        assert!(
            tool_pairing_is_intact(&trimmed),
            "MaxTokens orphaned a tool result: {trimmed:?}"
        );
        // Forward repair: the assistant tool-call turn is NOT re-admitted, so
        // the budget-bound trim cannot grow back.
        assert!(
            !trimmed
                .iter()
                .any(|m| matches!(m, Message::Assistant(a) if !a.tool_calls.is_empty()))
        );
    }

    #[test]
    fn unpairable_tool_results_are_dropped_when_no_assistant_exists() {
        // An imported/truncated transcript whose assistant turn is already
        // gone: there is nothing to move back to, so the results are shed.
        let messages = vec![
            Message::tool("ghost", "r1"),
            Message::tool("ghost2", "r2"),
            Message::assistant("done"),
        ];
        let trimmed = trim_messages(&messages, &TrimStrategy::KeepLast(2));
        assert!(tool_pairing_is_intact(&trimmed), "{trimmed:?}");
        assert_eq!(trimmed.len(), 1);
    }

    #[test]
    fn opting_out_of_repair_restores_the_unsafe_cut() {
        // Pins that the repair is what makes the difference, not some other
        // change of behaviour: without it the old orphaning cut comes back.
        let trimmed = trim_messages_with(
            &tool_transcript(),
            &TrimStrategy::KeepLast(2),
            &TrimOptions::default().without_pair_repair(),
        );
        assert!(
            !tool_pairing_is_intact(&trimmed),
            "expected the unrepaired cut to orphan a tool result"
        );
    }

    #[test]
    fn role_boundaries_trim_to_the_requested_roles() {
        let messages = vec![
            Message::assistant("lead-in"),
            Message::user("question"),
            Message::assistant("answer"),
            Message::user("trailing"),
        ];
        let trimmed = trim_messages_with(
            &messages,
            &TrimStrategy::KeepLast(4),
            &TrimOptions::default()
                .starting_on([MessageRole::User])
                .ending_on([MessageRole::Assistant]),
        );
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[0].text(), "question");
        assert_eq!(trimmed[1].text(), "answer");
    }

    #[test]
    fn end_on_boundary_does_not_leave_an_unanswered_tool_call() {
        let messages = vec![
            Message::user("go"),
            Message::assistant("thinking"),
            assistant_calling(&["c1"]),
            Message::tool("c1", "r1"),
        ];
        // Ending on an assistant turn would naively stop on the tool-call turn
        // whose result was just dropped.
        let trimmed = trim_messages_with(
            &messages,
            &TrimStrategy::KeepLast(4),
            &TrimOptions::default().ending_on([MessageRole::Assistant]),
        );
        assert!(tool_pairing_is_intact(&trimmed), "{trimmed:?}");
        assert_eq!(trimmed.last().map(Message::text), Some("thinking".into()));
    }

    #[test]
    fn tool_pairing_is_intact_detects_both_orphan_shapes() {
        assert!(tool_pairing_is_intact(&tool_transcript()));
        // Orphaned result.
        assert!(!tool_pairing_is_intact(&[Message::tool("c1", "r")]));
        // Unanswered call.
        assert!(!tool_pairing_is_intact(&[assistant_calling(&["c1"])]));
    }

    #[test]
    fn a_tool_only_assistant_turn_is_not_free() {
        // REASON-3: an assistant message with no content but a 2 KB argument
        // blob used to weigh zero, so no compaction gate could ever fire.
        let heavy = Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(
                "c1",
                "search",
                json!({"query": "x".repeat(2000)}),
            )],
            usage: None,
        });
        assert!(
            heavy.estimated_char_weight() > 2000,
            "tool-call arguments must be counted, got {}",
            heavy.estimated_char_weight()
        );

        let policy = SummarizationPolicy {
            trigger_tokens: 100,
            ..Default::default()
        };
        assert!(
            policy.should_summarize(&[heavy]),
            "a 2 KB tool-call turn must trip a 100-token trigger"
        );
    }

    #[test]
    fn a_tool_result_id_is_counted() {
        let bare = Message::Tool(crate::message::ToolMessage {
            tool_call_id: "call_abcdefghijklmnop".into(),
            content: Vec::new(),
            trusted_verbatim: false,
            artifact: None,
        });
        assert_eq!(bare.estimated_char_weight(), "call_abcdefghijklmnop".len());
    }

    #[test]
    fn reasoning_only_turns_still_weigh() {
        let msg = Message::Assistant(AssistantMessage {
            id: None,
            content: vec![ContentBlock::thinking("z".repeat(120))],
            tool_calls: Vec::new(),
            usage: None,
        });
        assert_eq!(msg.estimated_char_weight(), 120);
    }
}

/// Tests for [`render_message_for_summary`] and the default summarizer built on
/// it.
#[cfg(test)]
mod rendering {
    use crate::message::{AssistantMessage, Message};
    use crate::summarization::{ConcatSummarizer, Summarizer, render_message_for_summary};
    use crate::tool::ToolCall;
    use serde_json::json;

    #[tokio::test]
    async fn default_summarizer_keeps_tool_history() {
        // REASON-8: every one of these messages rendered to a bare role label
        // under `Message::text()`, so the default summarizer replaced real
        // history with nothing.
        let messages = vec![
            Message::Assistant(AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![ToolCall::new("c1", "get_weather", json!({"city": "Paris"}))],
                usage: None,
            }),
            Message::tool("c1", r#"{"temp_c":21}"#),
        ];

        let record = ConcatSummarizer.summarize(&messages).await.unwrap();
        let text = record.summary.text();

        assert!(text.contains("get_weather"), "tool name lost: {text}");
        assert!(text.contains("Paris"), "tool arguments lost: {text}");
        assert!(text.contains("temp_c"), "tool result lost: {text}");
        assert!(text.contains("c1"), "correlation id lost: {text}");
    }

    #[test]
    fn reasoning_and_json_are_rendered() {
        let msg = Message::Assistant(AssistantMessage {
            id: None,
            content: vec![
                crate::message::ContentBlock::thinking("weighing options"),
                crate::message::ContentBlock::Json(json!({"k": "v"})),
            ],
            tool_calls: Vec::new(),
            usage: None,
        });
        let rendered = render_message_for_summary(&msg);
        assert!(rendered.contains("weighing options"), "{rendered}");
        assert!(rendered.contains("\"k\""), "{rendered}");
    }

    #[test]
    fn oversized_payloads_are_elided_not_reproduced() {
        let msg = Message::tool("c1", "y".repeat(9_000));
        let rendered = render_message_for_summary(&msg);
        assert!(rendered.contains("chars elided"), "no elision marker");
        assert!(
            rendered.chars().count() < 3_000,
            "summary reproduced the payload"
        );
    }
}
