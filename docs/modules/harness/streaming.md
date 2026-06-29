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

## Stream Modes

- `messages`: model deltas and final messages
- `tools`: tool lifecycle and progress
- `usage`: token updates
- `cost`: price updates
- `events`: all harness events
- `final`: final result only

Every stream item should carry run ids and component ids so web UIs can merge
harness streams with graph streams.
