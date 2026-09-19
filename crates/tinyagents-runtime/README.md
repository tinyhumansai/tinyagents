# tinyagents-runtime

`tinyagents-runtime` provides the stateful session layer that sits between a
host's turn policy and TinyAgents' provider-neutral harness. A `Session` owns
the mutable model history, a stable prefix, and append-only transcript state.
Tool declarations are prepared afresh for every driver call. It has no host configuration, credentials,
prompt construction, tool authorization, model selection, or event system.

## Host responsibilities

The host supplies three narrow seams:

- `SessionDriver<C>` executes one prepared history snapshot. `HarnessDriver`
  adapts an `AgentHarness<State, C>` and passes the host's explicit `C` through
  unchanged. It fails closed unless the harness registry exactly matches the
  frozen request tool snapshot.
- `TranscriptCodec<C>` decodes the host's durable dialect and reconciles the
  previous durable rows with a model-history transition. The codec, rather
  than the runtime, retains fields inference messages cannot express. `C` is
  `Clone` so reconciliation receives the current host context plus request,
  thread, stream, and resume options after the live `RunContext` moves into the
  driver. Its optional `turn_usage` hook reads host sidecars after driver
  execution and returns a `TurnUsage` for the same atomic transcript append;
  the default returns `None` for generic codecs.
- `SessionHooks<C>` prepares a request and mutable `TurnOptions<C>` in two
  stages. `before_resume` lazily chooses a `TranscriptTarget`, then the runtime
  binds it and loads any requested transcript. `before_turn` sees that decoded
  history and raw rows plus `SessionStateView::resumed`; its `TurnPreparation`
  can install or replace the prefix before the first commit and selects the
  one immutable `ToolSnapshot` for that request. `before_commit` validates the candidate; `after_commit`
  receives an exactly-once `CommitReceipt<C>` containing the explicit context
  snapshot and neutral transcript path/delta receipt. `on_terminal` receives
  one truthful terminal state. It does not make policy decisions.

```rust,no_run
use std::sync::Arc;
use tinyagents_runtime::{SessionBuilder, SessionDriver, TranscriptCodec};

# fn build(driver: Arc<dyn SessionDriver<()>>, codec: Arc<dyn TranscriptCodec<()>>) {
let session = SessionBuilder::new(driver)
    .codec(codec)
    .build();
# let _ = session;
# }
```

To supply a default lazy destination, add `SessionBuilder::transcript(locator, stem, meta)`.
It does not open a transcript while building: a selected target binds only on
resume/first append and cannot be redirected after that. `TranscriptTarget` can
use a distinct `resume_agent` for `LatestForAgent` lookup; writes always use its
`stem`. The runtime
uses `tinyagents-session`'s `TranscriptHistory::append_turn_with_partial`, so a normal
extension appends only the new tail and a reduced context writes one compaction
record. A supplied partial driver outcome is represented through that single
history operation: logical history is replayable and interrupted display text
is not. A codec-provided `TurnUsage` is attached to that append's final
assistant row for both successful and recoverable-partial transitions.
Histories that cannot provide the combined operation reject a partial rather
than risk a two-step write. A reconciliation or usage-hook failure happens
before the append and leaves the session's in-memory history and persisted
snapshot unchanged.

Every turn receives explicit `TurnOptions`, including its cancellation token
and `RunContext<C>`; no task-local data crosses the runtime boundary. The
stable prefix is reconciled after resume and driver compaction without
duplication, including a prefix supplied by `before_turn` after a resumed
history. `Session::seed_history(history, raw)` is the explicit, lossless
resume/seed boundary; a host must not keep a second shadow history. Cancellation before the commit point leaves no durable mutation;
once it succeeds, the turn remains successful. `after_commit` and terminal
hooks get the committed outcome, but their error or a cooperative cancellation
cannot relabel it.
