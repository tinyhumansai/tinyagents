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
