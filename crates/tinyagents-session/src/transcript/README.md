# `session::transcript` — durable, provider-neutral session transcripts

The store-port seam hosts persist conversations through. A transcript lives in
whatever store the host installed (`crate::port`: the `tinystoragedrivers`
backed `DriverSessionStores`, or the in-memory double); this module defines
the traits and types, and nothing in it reads or writes a workspace directory.
Workspace discovery, import and rewrite of classic `session_raw/{stem}.jsonl`
files belong to the host. The session crate exposes typed transcript ports and
keeps no workspace path or SQLite reader. See the module doc in
`../transcript.rs` for the append-only log model and the classic line schema.

## Why this exists, and how it differs from `crate::ops` / `crate::run_ledger`

This module persists the literal message stream a model saw, so a resumed run
gets exactly the same context (and, for providers with prompt caching, hits
the same cache) it would have if the process never restarted. `crate::ops`
records a *searchable summary* of a session (messages, tool calls, cost) for
history/search; `crate::run_ledger` records *execution state* (run status,
checkpoints, coordination). All three can describe the same conversation from
different angles and are written independently.

## Public surface

Re-exported from `transcript::` (see `../transcript.rs`).

### Types (`types.rs`)

- `TranscriptMessage` / `TranscriptToolCall` — the durable, provider-neutral
  message and tool-call shapes. Deliberately not an inference message: this
  is a compatibility boundary, and hosts convert at their own edge.
- `TranscriptMeta` / `SessionTranscript` — the `_meta` header and the
  model-context transcript (`meta` + exact message array). The optional
  `_meta.prefix_message_count` records how many leading messages were frozen
  prompt tiers in this generation. `_meta.prefix_reset` distinguishes an
  intentional context reset from a legacy transcript with no recorded boundary.
- `MessageUsage` / `TurnUsage` — per-turn usage and provenance attached to
  the last assistant message of a turn. `input` / `output` / `cached_input`
  sum every provider call of the turn (its spend); `last_call_input` /
  `last_call_output` are the final call alone, the figure a context-window
  gauge needs. `spend::TranscriptSpend::last_context_tokens` reads them,
  with `last_context_input_tokens` and `last_context_output_tokens` exposing
  the two components separately,
  falling back to the mean request size for records written before them.
- `DisplayMessage` / `CompactionMarker` / `DisplayRecord` /
  `DisplaySessionTranscript` — the display projection, which (unlike the
  model-context view) preserves every record in log order.
- `ToolFailure` — provider-neutral failure status for a tool-result row.

### Store seam (`history.rs`)

- `TranscriptLocator` — resolves transcripts by thread, agent or
  `SessionRef`, lists a thread's transcripts (`thread_transcripts`: the roots
  with session chains oldest first and pre-identity roots dropped once a
  session root exists, plus sub-agent transcripts), binds a session's write
  handle (`open_session` / `open_stem`), and opens compaction generations
  (`begin_generation`, `begin_generation_from_baseline`,
  `truncate_into_next_generation`). `destination_key()` names the durable
  destination so rebuilt locators over one store compare equal.
  `adopt_messages_if_absent` atomically imports model-context history only
  when the session has never been written.
- `TranscriptHistory` / `TranscriptRead` — the write and read handles a turn
  path holds. `TranscriptRead` has no write methods, so a discovered
  transcript cannot be appended to by construction. `read_session()` is the
  model-context replay (compactions replace the accumulator, interrupted
  partials are skipped); `read_display()` is the display projection (every
  record, including pre-compaction history, compaction markers and interrupted
  partials). `adopt_if_empty` is the storage-level atomic primitive used by
  session import.
- `TranscriptTurn` — the borrowed argument bundle for
  `TranscriptHistory::append_turn`.

### Background appends (`background.rs`, `turn_lock.rs`)

- `append_background_message_in(&dyn TranscriptLocator, &SessionRef,
  TranscriptMessage, BackgroundAppend) -> Result<BackgroundAppendOutcome>`
  (async) — appends one plain assistant message to a session's **head**
  generation from outside any turn (e.g. a scheduled job reporting back into
  the chat that created it). Outcomes: `Appended { generation }`,
  `Duplicate { generation }` (the idempotency key is already held by the head
  generation's *live* context: a key dropped by a compaction, in the same
  generation or a successor, is free again), `StaleGeneration { expected,
  head }` (a pinned `expected_generation` no longer matches), `NoSession` (no
  transcript; none is created). The provenance is kept in the row's
  `extra_metadata` under `BACKGROUND_METADATA_KEY`, which the display read
  lifts back out as `DisplayMessage::background`.
- `lock_session_turn(&dyn TranscriptLocator, &SessionRef) ->
  Option<SessionTurnGuard>` (async) — the per-session, in-process turn lock,
  keyed by `destination_key()` + the generation-0 stem. A live turn holds it
  from its resume read through its persist (`tinyagents-runtime`'s
  `Session::turn` does so for session-bound targets); the background append
  awaits it, so it always lands between turns. Not reentrant: never await a
  background append into a session from inside that session's own turn.

### Spend and cost provenance (`spend.rs`)

- `transcript_spend(&SessionTranscript)` sums usage in the logical context.
  Compaction removes older rows from that view; use
  `thread_spend_in(locator, thread_id)` for the full append history.
- `thread_spend_in` returns `ThreadSpend`: `root` contains the orchestrator's
  own spend, and `subagents` groups descendant runs by archetype, including
  children whose headers name a different worker thread. Retained root rows
  across generations are counted once.
- Each `TranscriptSpend` exposes `cost_split: CostSplit`. The existing
  `cost_usd` remains the sum of all recorded costs for compatibility; it may
  include legacy guesses and incomplete costs. Use the split for presentation.

`MessageUsage::cost_source` serializes as `charged`, `estimated`, or
`unknown`. `Charged` means every call's cost came from provider billing;
`Estimated` means at least one call was priced from published list rates;
`Unknown` means at least one call had no known cost, so the recorded cost is
incomplete. An omitted source deserializes as `None` for legacy records and
remains omitted when written back. Hosts must set provenance when producing
usage; a nonzero cost alone does not establish its source.

`CostSplit::priced_cost_usd` includes only `Charged` and `Estimated` turns.
`priced_source` is `Estimated` if any included turn was estimated, `Charged`
if all were charged, and `None` when no priced turns exist. `Unknown` and
legacy `None` turns instead contribute to `unpriced_turns`,
`unpriced_input_tokens`, `unpriced_output_tokens`, and
`unpriced_cached_input_tokens`, even when their recorded cost is nonzero.
`CostSplit::merge` preserves this contract across transcripts.

For example, a charged $0.25 turn and a legacy $4.25 guess yield
`priced_cost_usd == 0.25`, `priced_source == Some(UsageCostSource::Charged)`,
and `unpriced_turns == 1`; the compatibility `cost_usd` is still $4.50.
Present the known priced amount with its source and separately re-price the
unpriced tokens at host-selected rates, or report their cost as unavailable.
Do not describe an estimated amount as a provider charge.

### View (`view/`)

`view::get_page_in(locator, thread_id, agent_id, cursor, limit)` projects a
thread's listed transcripts and returns one newest-first page;
`view::project_thread_in` returns the whole projection (optionally placing
sub-agents by a `SpawnCallLookup`). See `view/README.md`.

## File-by-file map

| File | Role |
| --- | --- |
| `types.rs` | Public domain types: messages, meta, usage, display projection. |
| `history.rs` | `TranscriptHistory` / `TranscriptRead` / `TranscriptLocator` seam. |
| `background.rs` | Out-of-band, idempotent assistant appends into a session's head. |
| `turn_lock.rs` | Per-session async turn lock shared by live turns and background appends. |
| `spend.rs` | Full-history thread spend and cost provenance splits. |
| `session.rs` | `SessionRef` identity and its deterministic stem. |
| `display.rs` | Row ⇄ display projection helpers. |
| `jsonl.rs` | Internal classic JSONL line shapes and line ⇄ message conversions. |
| `reader.rs` | Classic-file model-context replay and display projection retained for host migration tooling. |
| `writer.rs` | Legacy JSONL write helpers used by host migration tooling and tests. |
| `paths.rs`, `markdown.rs` | Stem sanitisation and `.md` compatibility helpers for host migration tooling. |
| `legacy_md.rs` | Legacy HTML-comment `.md` reader used by host migration tooling. |
| `test_locator.rs` | Test fixtures: classic files loaded into a test locator. |
| `*_tests.rs` | Module-local unit tests. |

## Operational constraints

**Compaction never erases.** A reduction is recorded as a compaction (or a
successor generation), and the earlier rows stay in the log: the model reads
the reduced context, the display read shows everything.

**Interrupted partials are display-only.** They are skipped by the
model-context replay so a resumed context never carries a truncated assistant
answer, but they are included by the display read for the UI timeline.

**Root vs. sub-agent transcripts are distinguished by stem shape**: a stem
containing `__` is a sub-agent transcript (`{parent_chain}__{unix_ts}_{agent_id}`);
a stem without it is a root session. Listing and spend rely on this to avoid
folding delegated sub-agent history into a thread's main timeline.

**Classic-file readers must take the *last* `_meta` line, not the first.**
Cumulative totals were kept fresh by appending a fresh `_meta` line each
turn; line 1 remains a valid fallback only for the oldest files.
