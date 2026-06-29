# Harness Usage Feature

Usage accounting normalizes token and cache usage across providers.

## Responsibilities

- Track prompt tokens.
- Track completion tokens.
- Track cached input tokens.
- Track reasoning tokens when providers expose them.
- Track embedding input tokens, characters, vector counts, dimensions, and
  batch counts when providers expose them.
- Track tool/model call counts.
- Roll usage up by call, node, agent, graph, thread, and run.
- Emit usage events.
- Preserve provider-specific token details.
- Mark provider-reported versus locally estimated usage.
- Normalize streamed cumulative usage into consistent records.

## Source Inspiration

LangChain standardizes usage metadata on `AIMessage` and has usage callback
helpers:

- <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/ai.py>
- <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- provider usage conversion such as OpenAI:
  <https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py>

## Core Types

```rust
pub struct UsageRecord {
    pub run_id: RunId,
    pub component: ComponentId,
    pub call_id: CallId,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub reasoning_tokens: usize,
    pub total_tokens: usize,
    pub input_details: TokenDetails,
    pub output_details: TokenDetails,
    pub source: UsageSource,
}
```

Provider adapters should map provider-specific usage into this normalized shape.
When providers do not return usage, the harness may estimate tokens and mark the
record as estimated.

## Token Details

Usage should allow provider-specific details without requiring every provider to
fill every field:

- embedding input characters
- embedding vector count
- embedding dimensions
- audio input tokens
- image input tokens
- cache creation tokens
- cache read tokens
- reasoning output tokens
- audio output tokens
- provider-specific extras

Details do not need to sum to the total because providers expose different
breakdowns. The normalized totals remain required when known.
