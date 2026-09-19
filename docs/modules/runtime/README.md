# Session runtime module

`tinyagents-runtime` provides one host-neutral, stateful conversation session
above `tinyagents-harness` and `tinyagents-session`. It is not a policy layer:
the host owns prompts, model selection, authorization, tool admission,
credentials, product events, and its durable transcript dialect.

## Turn boundary

A `Session<C>` holds model-visible history and an immutable stable prefix.
Every `turn` takes explicit `TurnOptions`:
request and thread identifiers, streaming mode, resume mode, cancellation, and
the live `RunContext<C>`. The driver consumes the live context. Because the
codec must reconcile after that driver call, `C: Clone` and the codec receives
the cloned host context together with the relevant per-turn options.

`SessionDriver<C>` is the execution seam. `HarnessDriver` adapts the pinned
harness partial-run entry points and retains their accumulated history on an
error. Those harness entry points do not currently expose a separate streamed
text delta, partial reasoning, or iteration; the adapter can only use the last
accumulated assistant text as a display partial when it exists.

Before driver handoff, `SessionHooks<C>::before_turn` receives the mutable
request and options plus a read-only `SessionStateView`. Its `TurnPreparation`
may set a prefix only for an empty, uncommitted session; it may select one
immutable tool snapshot for this call; and it may select a lazy
`TranscriptTarget`. A returned tool snapshot is not stored for the following
turn. The target is bound only when resume/append needs it and cannot change
after binding. `Session::seed_history` accepts a model history and lossless raw
rows before any commit, replacing host-side shadow history.

## Durability and projections

With a `TranscriptHistory`, a successful turn first passes `before_commit`, the
pre-commit validation hook. The runtime then submits its logical transcript
delta, metadata, and any supplied `TranscriptPartial` to one
`append_turn_with_partial` operation. `FileTranscriptHistory` serializes those
records into one buffer before one file write. The interrupted partial remains
in the display projection and is excluded from model-context replay.

Custom history implementations that cannot make this combined operation reject
a supplied partial, so the runtime does not fall back to two independent
writes. This is an operation-level guarantee; it does not claim crash-safe
filesystem transactions beyond the underlying storage implementation.

After a successful append and in-memory state update, `after_commit` receives
a `CommitReceipt<C>` with the committed outcome, the explicit context/options
snapshot, and neutral transcript path/delta data. `on_terminal` follows with a
completed terminal outcome. Both are observational: their errors, or a
cooperative cancellation that they observe, cannot change durable success.
Failed persistence and cancelled/failed turns never invoke `after_commit`.
