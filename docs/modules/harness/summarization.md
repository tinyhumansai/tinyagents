# Harness Summarization Feature

Summarization keeps long-running conversations and agent loops inside model
context limits while preserving useful provenance.

## Responsibilities

- Summarize old messages.
- Summarize large tool outputs.
- Maintain rolling conversation summaries.
- Store summary provenance.
- Decide when summaries are stale.
- Emit summary events and usage/cost records.
- Distinguish summaries used for prompt compaction from summaries stored for
  user-facing history.
- Preserve source ids and policy metadata for auditability.

## Source Inspiration

LangChain v1 includes summarization middleware, and LangChain docs discuss
short-term memory and message trimming:

- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/summarization.py>
- <https://docs.langchain.com/oss/python/langchain/short-term-memory>
- <https://docs.langchain.com/oss/python/langchain/middleware/built-in>

## Summary Record

```rust
pub struct SummaryRecord {
    pub id: SummaryId,
    pub thread_id: ThreadId,
    pub source_message_ids: Vec<MessageId>,
    pub model: ModelName,
    pub content: String,
    pub token_count_before: usize,
    pub token_count_after: usize,
    pub created_at: SystemTime,
    pub policy: SummaryPolicyName,
    pub prompt_version: Option<String>,
}
```

## Policies

- summarize when prompt estimate exceeds threshold
- summarize after N messages
- summarize tool outputs above byte/token threshold
- never summarize system messages unless explicit
- preserve latest user and assistant turns verbatim

Summaries should be stored through `harness::store` and linked to source message
ids so users can audit what was compressed.

## Staleness

A summary is stale when:

- one of its source messages changed
- the summarization policy changed
- the summarization prompt changed
- the selected summary model changed and policy requires regeneration
- the summary is older than a configured TTL

Stale summaries should not silently replace raw messages. The harness should
either regenerate, fall back to trimming, or fail with a context error according
to policy.
