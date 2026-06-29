//! Tests added in a later pass.
//!
//! This file is a placeholder.  Comprehensive tests covering trimming,
//! summarization, and policy logic will be added when the integration test
//! suite is extended.

#[cfg(test)]
mod smoke {
    use crate::harness::message::Message;
    use crate::harness::summarization::{
        ConcatSummarizer, SummarizationPolicy, Summarizer, TrimStrategy, estimate_tokens,
        trim_messages,
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

    #[test]
    fn policy_plan_keeps_everything_when_few_messages() {
        let policy = SummarizationPolicy {
            trigger_tokens: 0,
            keep_last: 5,
        };
        let msgs = vec![Message::system("sys"), Message::user("u")];
        let (to_summarize, to_keep) = policy.plan(&msgs);
        assert!(to_summarize.is_empty());
        assert_eq!(to_keep.len(), 2);
    }
}
