//! Tests for trimming, summarization, and compression policies.
//!
//! Cover the [`estimate_tokens`] heuristic (clamping and ~4-chars-per-token
//! scaling), every [`TrimStrategy`] variant (including system-message retention
//! and the last-resort drop), [`ConcatSummarizer`] output and provenance, and
//! [`SummarizationPolicy`] gating — both the raw `trigger_tokens` path and the
//! context-window-aware `threshold_fraction` path, plus `plan` splitting.

#[cfg(test)]
mod smoke {
    use crate::harness::message::Message;
    use crate::harness::summarization::{
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
        use crate::harness::model::ModelProfile;

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
