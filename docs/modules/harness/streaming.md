# Harness Streaming Feature

The harness must support streaming independently from the graph. Graph streaming
can forward harness events, but direct harness users should also be able to
consume streams.

## Responsibilities

- Stream model token/message deltas.
- Stream tool progress.
- Stream usage and cost updates.
- Stream cache hit/miss events.
- Stream summary events.
- Stream final outputs.
- Forward events into the registry event bus.
- Merge provider chunks into a final assistant message.
- Preserve streamed tool-call chunks where providers support them.
- Support cancellation and backpressure.
- Support stream replay from event stores.

## Source Inspiration

LangChain exposes model streams, runnable event streams, callback streaming, and
tracer streams:

- chat model stream:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_model_stream.py>
- runnable event streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/runnables>
- streaming tracers:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- provider streaming adapters such as OpenAI:
  <https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py>

## Stream Modes

- `messages`: model deltas and final messages
- `tools`: tool lifecycle and progress
- `usage`: token updates
- `cost`: price updates
- `events`: all harness events
- `final`: final result only

Every stream item should carry run ids and component ids so web UIs can merge
harness streams with graph streams.

## Stream Items

```rust
pub enum HarnessStreamItem {
    Event(HarnessEvent),
    MessageDelta(MessageDelta),
    ToolCallDelta(ToolCallDelta),
    ToolProgress(ToolProgress),
    Usage(UsageRecord),
    Cost(CostRecord),
    Final(AgentRun),
}
```

A stream consumer should be able to subscribe to a subset of modes without
changing execution. Dropping a consumer must not cancel the run unless the
consumer owns the run cancellation token.

## Chunk Merging

Streaming adapters must merge chunks deterministically:

- text chunks preserve order
- content block indexes are respected
- tool-call chunks are correlated by id or index
- cumulative usage is converted into deltas or clearly marked cumulative
- final message equals the merged stream
- invalid partial tool calls are surfaced as repairable parse errors
