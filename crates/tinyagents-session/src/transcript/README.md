# `session::transcript` — durable, provider-neutral session transcripts

An append-only, JSONL-backed transcript store used for KV-cache-stable
resume: `{workspace}/session_raw/{stem}.jsonl` is the source of truth, with a
human-readable `.md` companion re-rendered on every write. See the extensive
module doc in `../transcript.rs` for the on-disk schema, the append-only log
format (pure extension vs. compaction records), and backward compatibility
with the legacy date-grouped layout — this README covers the module's shape
and how its files relate rather than repeating that.

## Why this exists, and how it differs from `crate::ops` / `crate::run_ledger`

This module persists the literal message stream a model saw, byte-for-byte,
so a resumed run gets exactly the same context (and, for providers with
prompt caching, hits the same cache) it would have if the process never
restarted. `crate::ops` records a *searchable summary* of a session
(messages, tool calls, cost) for history/search; `crate::run_ledger` records
*execution state* (run status, checkpoints, coordination). All three can
describe the same conversation from different angles and are written
independently — a host that only wants search doesn't need transcripts, and
vice versa.

## Public surface

Re-exported from `transcript::` (see `../transcript.rs`); reachable as
`session::transcript::`.

### Types (`types.rs`)

- `TranscriptMessage` / `TranscriptToolCall` — the durable, provider-neutral
  message and tool-call shapes. Deliberately not an inference message: this
  is an on-disk compatibility boundary, and hosts convert at their own edge.
- `TranscriptMeta` / `SessionTranscript` — the `_meta` header and the parsed
  model-context transcript (`meta` + exact message array). The optional
  `_meta.prefix_message_count` records how many leading messages were frozen
  prompt tiers in this generation; older files omit it.
- `MessageUsage` / `TurnUsage` — per-turn usage and provenance attached to
  the last assistant message of a turn. `input` / `output` / `cached_input`
  sum every provider call of the turn (its spend); `last_call_input` /
  `last_call_output` are the final call alone, the figure a context-window
  gauge needs. `spend::TranscriptSpend::last_context_tokens` reads them,
  falling back to the mean request size for records written before them.
- `DisplayMessage` / `CompactionMarker` / `DisplayRecord` /
  `DisplaySessionTranscript` — the display projection, which (unlike the
  model-context view) preserves every record in file order.
- `ToolFailure` — provider-neutral failure status for a tool-result row.

### Writing (`writer.rs`)

- `write_transcript` — one-shot **full rewrite**; used by migrations,
  sub-agent runners, and tests.
- `append_transcript_turn` — the incremental, append-only path: appends new
  tail lines on pure extension, or a `compaction` record on context
  reduction. Never rewrites existing lines.
- `append_interrupted_partial` — appends a display-only partial assistant
  answer when a stream was cancelled mid-turn; skipped by the model-context
  reader.

### Background appends (`background.rs`, `turn_lock.rs`)

- `append_background_message(&FileTranscriptLocator, &SessionRef,
  TranscriptMessage, BackgroundAppend) -> Result<BackgroundAppendOutcome>`
  (async) — appends one plain assistant message to a session's **head**
  generation from outside any turn (e.g. a scheduled job reporting back into
  the chat that created it). Outcomes: `Appended { generation }`,
  `Duplicate { generation }` (the idempotency key is already in the head),
  `StaleGeneration { expected, head }` (a pinned `expected_generation` no
  longer matches), `NoSession` (no transcript; none is created). The line is
  an ordinary message line plus an additive top-level
  `"background": {"idempotency_key", "provenance"}` field, so the
  model-context reader resumes it as an assistant row and the display reader
  exposes the origin as `DisplayMessage::background`.
- `lock_session_turn(&dyn TranscriptLocator, &SessionRef) ->
  Option<SessionTurnGuard>` (async) — the per-session, in-process turn lock,
  keyed by `destination_key()` + the generation-0 stem. A live turn holds it
  from its resume read through its persist (`tinyagents-runtime`'s
  `Session::turn` does so for session-bound targets); the background append
  awaits it, so it always lands between turns. Not reentrant: never await a
  background append into a session from inside that session's own turn.

### Reading (`reader.rs`)

- `read_transcript` — model-context replay: compaction records replace the
  accumulator, interrupted partials are skipped. Falls back to the legacy
  `.md` reader when no `.jsonl` exists.
- `read_transcript_display` — display projection: every record in file
  order, including pre-compaction history and interrupted partials.

### Paths and resume (`paths.rs`)

- `resolve_keyed_transcript_path` — resolves/creates
  `session_raw/{stem}.jsonl` for a given stem.
- `find_latest_transcript` — newest-transcript scan for an agent, with a
  fallback to the legacy `session_raw/DDMMYYYY/` layout.

### Thread lookups (`thread_lookup.rs`)

- `find_root_transcript_for_thread` / `find_root_transcript_for_thread_scoped`
  / `find_root_transcripts_for_thread` — locate a thread's root transcript(s)
  by `_meta.thread_id`, optionally scoped by `_meta.agent_id` to avoid
  splicing one agent's history into another's when a `thread_id` is reused.
- `read_thread_usage_summary` — aggregated token/cost usage for a thread
  across all its root transcripts, plus a per-archetype sub-agent breakdown.

### Legacy compatibility (`legacy_md.rs`)

- `read_transcript_legacy_md` — parses the pre-JSONL HTML-comment `.md`
  format; one-release migration compat.

### History/locator seam (`history.rs`)

- `TranscriptHistory` / `TranscriptRead` / `TranscriptLocator` — the traits a
  host turn path holds (`Arc<dyn ...>`) instead of calling the free functions
  directly, so a discovered transcript that turns out to be a legacy `.md`
  file cannot be appended to by construction (`TranscriptRead` has no write
  methods).
- `FileTranscriptHistory` / `FileTranscriptLocator` — the default
  implementations, each a thin wrapper over exactly one free function per
  method.
- `TranscriptTurn` — the borrowed argument bundle for
  `TranscriptHistory::append_turn`, shaped to mirror
  `append_transcript_turn`'s signature one-for-one.

## File-by-file map

| File | Role |
| --- | --- |
| `types.rs` | Public domain types: messages, meta, usage, display projection. |
| `jsonl.rs` | Internal JSONL line shapes and line ⇄ message conversions. |
| `writer.rs` | Full rewrite, append-only turn delta, interrupted partials. |
| `reader.rs` | Model-context replay, display projection, meta-only/usage scans. |
| `paths.rs` | Path resolution, sanitization, and the newest-transcript resume scan. |
| `thread_lookup.rs` | Thread → root transcript lookup and usage summaries. |
| `markdown.rs` | Human-readable `.md` companion rendering (never read back). |
| `legacy_md.rs` | Legacy HTML-comment `.md` reader (migration compat). |
| `history.rs` | `TranscriptHistory` / `TranscriptRead` / `TranscriptLocator` seam. |
| `background.rs` | Out-of-band, idempotent assistant appends into a session's head. |
| `turn_lock.rs` | Per-session async turn lock shared by live turns and background appends. |
| `*_tests.rs` | Module-local unit tests (`transcript_tests.rs`, `session_tests.rs`, ...). |

`../transcript.rs` (the module root, one level up) wires these together,
re-exports the public surface, and carries the full format specification in
its module doc.

## Operational constraints

**The JSONL is the only source of truth; the `.md` companion is never read
back.** All round-trip and resume logic reads `.jsonl`. A companion write
failure is logged and swallowed (`writer.rs::render_md_companion`) so it can
never take down state persistence.

**`append_transcript_turn` never rewrites existing lines.** It diffs the
incoming logical message set against what was previously persisted
(`common_prefix_len`, compared on `role`/`content`/`id`): a prefix match
appends only the new tail; anything else appends a full `compaction` record
carrying the reduced set, leaving earlier lines untouched.

**Readers must take the *last* `_meta` line, not the first.** Cumulative
totals are kept fresh by appending a fresh `_meta` line each turn rather than
rewriting line 1; line 1 remains a valid fallback only for old cores that
never learned to look further.

**Interrupted partials are display-only.** `interrupted: true` lines are
skipped by `read_transcript`'s model-context replay so a resumed context
never carries a truncated assistant answer, but they are included by
`read_transcript_display` for the UI timeline.

**Root vs. sub-agent transcripts are distinguished by stem shape**, not a
separate directory: a stem containing `__` is a sub-agent transcript
(`{parent_chain}__{unix_ts}_{agent_id}`); a stem without it is a root
session (`{unix_ts}_{agent_id}`). `thread_lookup.rs` and `paths.rs` both rely
on this to avoid folding delegated sub-agent history into a thread's main
timeline.

**`FileTranscriptHistory`'s generic trait path always re-reads the file
before writing** (`write_logical_set`), unlike the turn path proper, which
tracks its previously-persisted set in memory. This is deliberate: messages
reconstructed from disk have had turn-usage fields hoisted and
`failure`/`failure_detail` lifted out of `extra_metadata`, so feeding that
back in as `prev` for an in-memory-tracked caller would mismatch at the
first such message and force a full compaction record every turn.

**File history mutations are coordinated across processes.** A persistent
advisory lock under the workspace's `.transcript-locks/` directory serializes
each file mutation. `begin_generation` keeps the successor's lock from its
existence check through its first successful write, so a crash releases the
reservation and leaves no phantom generation. Turn appends also compare the
caller baseline with the current logical transcript while holding that lock;
a stale session receives a persistence error instead of writing a compaction
that hides another session's newer turns. Lock files are deliberately
retained: removing a lock inode while another process waits could split future
writers across different locks.
