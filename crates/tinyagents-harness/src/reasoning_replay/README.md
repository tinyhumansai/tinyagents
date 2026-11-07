# reasoning_replay

Keeps earlier turns' reasoning out of the requests a run sends, while the
stored transcript keeps all of it.

## Why

An assistant message keeps its reasoning blocks (`ContentBlock::Thinking`,
`RedactedThinking`) in the transcript, and every later request is built from
that transcript. Reasoning from finished turns is rarely useful to the model
and can be several percent of the input tokens of a long thread.

## Surface

- `ReasoningReplay` (`RunPolicy::replay_reasoning`): `All` (default) or
  `CurrentTurn`.
- `RunConfig::replay_reasoning` / `with_replay_reasoning`: a per-session
  `Option<ReasoningReplay>` override. `None` by default and not serialized when
  `None`, so an older thread never changes until a host opts in.
- `drop_past_turn_reasoning(messages)`: the pure rewrite of a request's
  history.

## Behaviour

- A turn is the span after a user message. The current turn is everything
  after the last user message; every assistant message before it is a past
  turn.
- With `CurrentTurn`, each request is built from the history with the
  reasoning blocks of past-turn assistant messages removed. The current turn's
  reasoning is untouched, byte for byte (signature included), for every call of
  the tool loop.
- The rewrite is on the outgoing request only. The working messages, the
  transcript the run returns and anything a resume replays keep every block.
- Text and tool calls of past turns are kept. A past assistant message that
  held only reasoning is removed, since providers reject an empty assistant
  message.
- `All` builds exactly the request it built before this option existed.

## Provider caveats

Whether past reasoning reaches the model at all depends on the provider path,
so the measured saving depends on it too.

- **OpenAI-compatible routes** drop reasoning blocks when they build the wire
  message, so `CurrentTurn` saves nothing on the wire there. Any share of
  reasoning seen in traces that record the harness history (for example
  Langfuse inputs) is not what the provider was billed for.
- **DeepSeek** replays reasoning as `reasoning_content`. It is needed on the
  assistant messages of the current turn's tool loop, which `CurrentTurn`
  keeps, and not for finished turns.
- **Anthropic** signed thinking must be replayed unmodified inside the current
  tool loop, which `CurrentTurn` does. Thinking from earlier turns is not
  validated and the API ignores it, so omitting it is safe.

## Cache trade-off

Dropping a finished turn's reasoning changes the middle of the request, not
its front. The system prompt and tool list are never touched, so the stable
prefix, a frozen system prefix and any sealed generation stay valid. The
messages from the previous turn's first reasoning block onward change once, at
the moment a new user message arrives, and a provider's prefix cache re-reads
that span once. Within a turn the request prefix only grows, so the cache holds
as it did before. Turn on the option for threads whose reasoning is large
enough to repay that one re-read.
