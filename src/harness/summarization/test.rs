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
}
