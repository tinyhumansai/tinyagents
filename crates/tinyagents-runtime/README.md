# tinyagents-runtime

`tinyagents-runtime` provides the stateful session layer that sits between a
host's turn policy and TinyAgents' provider-neutral harness. A `Session` owns
the mutable model history, a stable prefix, and append-only transcript state.
Tool declarations are prepared afresh for every driver call. It has no host configuration, credentials,
prompt construction, tool authorization, model selection, or event system.

Sub-agent turns are leaf turns by default: registration refuses spawn tools
even when the child request or tool allowlist includes them. A host may opt a
turn into nested spawning by setting `ToolAdmission::nested_spawn_depth` to a
validated `NestedSpawnDepth`. The runtime admits a spawn tool only when it is
also present in the ordinary tool allowlist and another child fits within the
supplied depth ceiling. The host remains responsible for enforcing that same
ceiling when it executes the spawn.

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
history. Each committed transcript generation records the frozen prefix's
message count so a cold resume strips that generation's stored tiers rather
than guessing from every leading System row; older files fall back to the
sealed generation when it can establish the boundary.
Legacy files without an exact boundary reconcile overlapping configured
few-shot messages on every resume. A compacted head may recover an exact count
from its sealed root when that full prefix matches the head. An inferred System
count is not cached as authoritative; the next successful persisted turn
records the complete configured prefix count. Recorded boundaries preserve
equal messages that belong to the conversation after the prefix.
When all frozen tiers are System messages, the session also carries their
count into the harness run context, so a later System compaction summary stays
in history when the next driver invocation rebuilds its request.
`Session::seed_history(history, raw)` is the explicit, lossless
resume/seed boundary; a host must not keep a second shadow history. Cancellation before the commit point leaves no durable mutation,
with one exception: a cancelled turn whose driver returns a `DriverFailure`
partial within a short grace (2 s) persists that partial exactly as a failed
turn would, so a stopped turn keeps its prompt and completed rounds. The turn
still ends as `RuntimeError::Cancelled` and no `after_commit` runs. A driver
that does not answer in the grace is dropped and nothing is written. Once a
commit succeeds, the turn remains successful. `after_commit` and terminal
hooks get the committed outcome, but their error or a cooperative cancellation
cannot relabel it.

`Session::reset_history()` seals the current generation and starts an empty
successor for the same durable session. It preserves the sealed history for
recovery, records an explicit prompt-reset marker, and is a no-op for an absent
or already-empty session. A `SessionStateView` reports `prefix_is_frozen` so a
host can distinguish imported history that still needs its first prompt from
a session whose prompt boundary is already authoritative.

## Explicit prefix refresh

Prefixes are frozen by default. A host deliberately changing a managed system
section may return `PrefixSnapshot::new(messages).refreshing()` from
`before_turn`. This opt-in preserves conversation rows while allowing a new
prefix after committed turns. The replacement stays tentative until the turn
commits, including a successfully persisted partial outcome. Every changed committed
prefix seals the existing transcript and creates a successor generation for session-bound
targets. Stem targets record the changed rows through in-place compaction instead.
This replacement is explicit even when the refreshed prefix starts with all
prior raw rows; the old generation's conversation and prefix boundary remain intact.
An identical prefix is a no-op. Cold resume restores the committed prefix. Preserve the original system messages when only
one managed section needs updating. Codecs receive the history before the
refresh beside its original raw rows, so unchanged durable metadata remains
associated with the correct messages even when prefix length changes.
