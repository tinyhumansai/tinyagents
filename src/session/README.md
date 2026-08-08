# `session` — durable session history and run ledger

SQLite-backed history for agent sessions, and a restart-survivable ledger for
background agent/workflow execution. Requires the `sqlite` feature.

## Why this is a top-level module

Session history is a persistence domain in its own right, not part of the agent
loop. Nothing in `harness` reads from it, and a host can use it without running
a harness at all — indexing sessions produced elsewhere, or recovering
orchestration state at boot before any agent exists. Filing it under `harness::`
would imply a dependency that exists in neither direction.

## How it differs from the other persistence layers

| Layer | Question it answers | Lifetime |
| --- | --- | --- |
| `harness::store` | "what is this run working with right now?" | during a run |
| `graph::checkpoint` | "how do I resume this interrupted run?" | until resumed |
| **`session`** | "what happened, what did it cost, how did runs nest?" | indefinitely |

Nothing resumes from this module. It is queryable history: cross-session search,
cost attribution, and orchestration recovery.

## Layout

Every entry point takes the workspace root and derives the path itself, so a
host chooses only where its workspace lives:

```text
{workspace_dir}/session_db/sessions.db
```

## Public surface

Re-exported from the crate root (see `src/lib.rs`); the full surface stays
reachable under `session::` and `session::run_ledger::`.

- **Recording** — `record_session_start`, `record_message`, `record_tool_call`,
  `record_session_end`
- **Querying** — `get_session`, `list_sessions`, `search_sessions`,
  `list_messages`, `list_tool_calls`, `list_children`
- **Recovery** — `mark_interrupted`
- **Run ledger** — agent runs, workflow runs, teams, members, tasks, run events,
  and telemetry, with the claim/completion coordination primitives
- **Connections** — `with_connection` (autocommit) and `with_transaction`
  (`BEGIN IMMEDIATE`)

## Schema

Six tables plus one FTS5 virtual table, created on demand and idempotently:

| Table | Holds |
| --- | --- |
| `sessions` | one row per session; lineage via `parent_session_id` |
| `session_messages` | per-message content, model, tokens, cost |
| `session_tool_calls` | tool name, input, bounded output, status, duration |
| `sessions_fts` | FTS5 index over session name, message content, tool name |
| `agent_runs` / `workflow_runs` | background execution state |
| `run_events` / `run_telemetry` | per-run event stream and rollups |
| `agent_teams` / `agent_team_members` / `agent_team_tasks` | team coordination |

WAL journaling and `foreign_keys = ON`. FTS5 comes from `rusqlite`'s `bundled`
build — there is no separate `fts5` cargo feature at 0.40, so do not add one.

## Operational constraints

These are the non-obvious rules; each is pinned by a test in `test.rs`.

**Search input is plain text, not FTS5 syntax.** `SessionSearchParams::query` is
translated to a quoted FTS5 expression before it reaches `MATCH`. Binding raw
user input made ordinary strings (`C++`, `foo-bar`, `file.rs`) fail with a
syntax or `no such column` error instead of searching.

**Indexed content is truncated on a character boundary.** Slicing at a raw byte
offset panics on multi-byte input, and because the message insert has already
committed, the row would survive with no FTS entry — silently unsearchable.

**Tool output is bounded** to `MAX_TOOL_OUTPUT_BYTES`, truncated on a character
boundary with a marker appended.

**Telemetry counters are `Option` for partial updates.** The columns are
`NOT NULL DEFAULT`, and SQLite does not apply a column default to an explicitly
supplied `NULL`, so the insert path coalesces to the default while the update
path coalesces to the stored value. `excluded.*` cannot serve the update side —
it observes the already-coalesced row, so `None` would read as `0` and clobber a
stored counter.

**Run-event sequences are allocated by the INSERT itself.** Reading
`MAX(sequence) + 1` and then inserting is a read-modify-write race; the loser
fails the primary key and the event is lost.

**Coordination operations need `with_transaction`, not `with_connection`.**
`with_connection` is autocommit, which gives ordering but no isolation. Claim
and completion read state and then act on it, so they take the write lock up
front with `BEGIN IMMEDIATE` — racing claims serialize at `BEGIN` rather than
failing at `COMMIT` after one has already decided it won.

**A claim is meaningful only while a task is `in_progress`.** An upsert that
moves a task off that status clears `claimed_by_member_id` and `claim_token`;
leaving them set strands the task, since a new claim sees `AlreadyClaimed`,
completion sees `NotClaimed`, and release/shutdown skip it.

**Evidence accumulates across completion attempts,** including attempts whose
gate fails — otherwise a retry after fixing an unrelated gate would fail
`require_evidence` on evidence already submitted.

## Layout of this module

| File | Role |
| --- | --- |
| `mod.rs` | module docs and public surface |
| `types.rs` | serde record types |
| `store.rs` | connection/transaction helpers and schema init |
| `ops.rs` | recording and querying |
| `context.rs` | `StorageContext`, the error-context shim |
| `run_ledger/` | background run + team coordination |
| `test.rs` | module-local unit tests |
