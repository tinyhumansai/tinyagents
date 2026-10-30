# Changelog

## 3.0.0 — 2026-10-22

**Breaking.** This release completes the TinyAgents slim-down: generic tool
types and tool-call parsing live in TinyTools, while host-specific execution
policy and adapters remain with their hosts. Public crates and modules removed
during the 2.x migration are no longer part of the TinyAgents workspace.

- Bump the surviving TinyAgents workspace crates to 3.0.0.
- Update the nested TinyTools pin to the 0.6-compatible epic revision.

The unreleased entries below describe the changes accumulated before this
major release.

## Unreleased: tool hook denials answer the model

**Breaking (behaviour).** A `tinyagents-hooks` tool-hook `Deny` no longer
aborts the run. `ToolHooksMiddleware` now returns
`TinyAgentsError::ToolFailed`, so the agent loop answers the call with a failed
tool result carrying `tool hook '<hook>' denied <tool>: <reason>` and the run
continues; the tool never runs. This matches `HookPermission::Deny`'s
contract ("The agent is told why") and covers blank or invented tool names a
host denies, which used to end the turn before the unknown-tool policy ran.

Hosts that relied on a deny ending the run must set `continue_: Some(false)`
on the decision. That form stops the run with the new
`TinyAgentsError::ToolRefused`, classified as `TerminalReason::ToolRefused` and,
for hosted callers, `HostedErrorKind::ToolRefused` with the message
`hosted agent invocation stopped: host policy refused a tool call` instead of
the generic `hosted agent invocation failed`. The refusal text never carries
the call's arguments. `Ask` is unchanged and still fails the run with
`TinyAgentsError::Tool`.

An answered deny releases its tool-call slot like every other `before_tool`
refusal, so a model that keeps calling a denied tool is bounded by
`RunLimits::max_model_calls`. `TinyAgentsError`, `TerminalReason` and
`HostedErrorKind` are `#[non_exhaustive]`, so the new variants are additive.
See `docs/modules/hooks/README.md`.

## Unreleased: model observations report who served a call and its first token

**Breaking.** `TurnObservation::Model` and `TurnObservation::ModelDispatch`
gain `served_by`, `response_id` and `first_token_ms`, and
`PhysicalDispatchOutcome` gains `served_by`, `response_id` and
`first_output`. Code that builds these values must set the new fields; code
that matches them with `..` is unaffected. `served_by` is a routing gateway's
top-level `provider` (OpenRouter's upstream provider), `response_id` is the
provider's response id (`gen-…` on OpenRouter) and `first_token_ms` is the
time to the first streamed text, reasoning or tool-call output. See
`docs/modules/harness/observability.md`.

OTLP export (`otlp_requests`) now splits a generation's reasoning tokens
(`gen_ai.usage.reasoning_tokens`) out of `output` into Langfuse's
`output_reasoning_tokens` usage bucket. `total` is unchanged.

## Unreleased: `AgentProgress::ContextCompacting`

**Breaking.** Added `AgentProgress::ContextCompacting` to the public
`AgentProgress` enum. Downstream exhaustive matches must handle the new
variant.

## Unreleased: settled completions drop their result (tinyagents#354)

`CompletionRouter` now empties a record's `result` when the record settles:
`mark_delivered`, `tombstone` and `cancel_parent` store the record with its
id, parent key, status, state, attempts and timestamps but no text, artifact
reference or omitted-character count, so a delivered child's output or a
deleted parent's withdrawn completions do not sit in the store until
`compact(retain)`. `mark_failed` still returns the full record to the host's
give-up writer and stores the emptied one. Dedupe, tombstones and reopen
behaviour are unchanged.

## Unreleased: dedicated hooks crate (tinyagents#470)

**Breaking.** Stop-hook and lifecycle-hook APIs now live in
`tinyagents_hooks::{stop,lifecycle}`. `tinyagents-runtime::command_hooks`
remains as a one-release compatibility re-export; new integrations should
depend on `tinyagents-hooks` directly. The workspace version is bumped to
`2.2.0` for this package-boundary change.

- Added `tinyagents-hooks` with typed tool hooks, command hooks, stop policies,
  lifecycle middleware, per-scope hook lists, and safe tool-output summaries.
- Added `BudgetStopHook`, the optional `goals`-feature
  `GoalBudgetStopHook`, and `StopHookMiddleware::with_halt_summary`.
- Added per-tool outcome records to completed-turn summaries.

## Unreleased: Langfuse v4 trace export over OTLP

Langfuse v4 ingests traces only over OTLP, and Langfuse Cloud removes the
legacy `/api/public/ingestion` trace path on 2026-11-16.

- `LangfuseClient::send_otlp` sends `x-langfuse-ingestion-version: 4` and
  reports a 2xx answer with rejected spans (`partialSuccess.rejectedSpans`)
  as an error.
- New `trace_export::ingestion_otlp` (`ingestion_batch_to_otlp_spans`,
  `ingestion_batch_otlp_requests`): converts a locally built ingestion batch
  into v4 OTLP spans, with trace name, user, session, tags, metadata,
  environment, release and version on every span and the trace input/output
  on the root observation.
- New `LangfuseClient::send_observations_otlp` / `send_batch_otlp` and
  `GraphLangfuseExporter::send_observations_otlp` / `build_otlp_requests`.
  The batch-path `send_observations` methods stay for legacy deployments.
- `trace_export::otlp::chunk_otlp_spans` exposes the request chunking that
  `otlp_requests` uses.

## Unreleased: the path-based transcript API is removed (tinyagents#453)

**Breaking.** `tinyagents-session` no longer reads or writes a workspace
directory for transcripts. Everything goes through the store port
(`TranscriptLocator`/`TranscriptHistory`/`TranscriptRead`, served by
`DriverSessionStores`, `WorkspaceStores` or `InMemorySessionStores`); the only
reader of classic `session_raw/*.jsonl` files is the one-time, read-only
`legacy::import_transcripts_once` / `WorkspaceStores`, and
`legacy::rewrite_legacy_transcripts` for pre-import fixups.

Removed from `tinyagents_session::transcript`: `FileTranscriptLocator`,
`FileTranscriptHistory`, `AgentTranscriptFiles`, `agent_transcript_root`,
`write_transcript*`, `append_transcript_turn*`, `append_interrupted_partial`,
`append_tools_record`, `read_transcript`, `read_transcript_display`,
`read_transcript_legacy_md`, `adopt_legacy_session_transcripts`,
`migrate_layout_if_needed`, `find_latest_transcript`,
`find_root_transcript_for_thread*`, `find_root_transcripts_for_thread`,
`read_thread_usage_summary` (and `ThreadUsageSummary`/`SubagentArchetypeUsage`),
`resolve_keyed_transcript_path*`, `append_background_message` and its
lock-notification variant, `TranscriptLocator::file_backing`, `view::get_page`,
`view::get_page_scoped`, `view::project_thread*`, `view::resolve_files_scoped`,
the view's projection cache, and `thread_spend`. Replacements:
`append_background_message_in`, `view::get_page_in`,
`view::project_thread_in(locator, thread_id, agent_id, lookup)`,
`spend::thread_spend_in`, `TranscriptLocator::thread_transcripts` /
`root_for_thread_scoped` / `read_session_transcript`, and
`TranscriptLocator::adopt_legacy` (driver stores).
`SessionAdoption`, `SessionTurnGuard`, `lock_session_turn` and the rest of the
locator-based surface are unchanged. The `fs2` dependency is gone.

Fixes (the two deferred tinyagents#463 findings):

- A background append's idempotency check considered every row of the head
  generation's log, so a key dropped by a compaction *inside the same
  generation* still reported `Duplicate` and the delivery was lost. It now
  looks at the generation's live context only (rows after the last compaction
  plus that compaction's replacement), which is what the contract always said:
  a key a compaction dropped is appended again.
- The driver-backed thread listing read the index alone, so a root whose
  index refresh failed stayed invisible until something read or wrote that
  transcript again. A listing now compares written index documents with
  written transcript logs and re-indexes any missing stem before it answers.
- `legacy::import_transcripts` now carries a classic line's top-level
  `background` provenance onto the imported row, so an out-of-band delivery
  keeps its origin and idempotency key after import.

The Release workflow derives the next version from the highest of the
workspace version and the latest `v*` tag instead of the (stale) workspace
version alone, so it no longer proposes a tag that already exists.
## Unreleased: locator-based transcript reads and writes (tinyagents#453)

Hosts can now do through a `TranscriptLocator` everything they used the
path-taking transcript API for, so the same code serves JSONL files and a
driver store:

- `TranscriptRead::read_display()` (default: from `read_session`; file
  handles read the JSONL display log; driver handles replay their entry log
  with compactions and interrupted partials in place).
- `TranscriptLocator::thread_transcripts(thread_id, agent_id) ->
  Option<ThreadTranscripts>` lists a thread's roots in projection order and
  its sub-agent transcripts (`ThreadTranscript { stem, transcript }`).
- `view::get_page_in(locator, thread_id, agent_id, cursor, limit)` and
  `view::project_transcripts(...)`; `spend::thread_spend_in(locator,
  thread_id)`.
- `append_background_message_in(locator, session, message, options)`; for a
  non-file store the provenance rides in `extra_metadata` under
  `BACKGROUND_METADATA_KEY`. `TranscriptLocator::file_backing()` names the
  file locator behind a file-backed one.
- `DriverTranscriptLocator::adopt_legacy` folds a thread's pre-identity
  roots (for example the imported `session_raw/` timestamped stems of a
  `WorkspaceStores` workspace) into the session once, with the file adopter's
  semantics; the write reserves the session stem, so a concurrent first turn
  wins and the call reports `Ok(None)`.
- `legacy::rewrite_legacy_transcripts(workspace_dir, rewrite) ->
  LegacyRewriteReport` for pre-import fixups of classic `session_raw/` files.

(The path API was still present at this point and was removed in the entry above.)

Review follow-ups: `rewrite_legacy_transcripts` keeps a transcript's recorded
tool declarations, counts unreadable directories and entries in `errors`
(a missing `session_raw/` is still an empty report) and skips symlinks.
Thread spend counts a compacted sub-agent's `.gN` chain as one run with its
carried rows counted once (`thread_spend` and `thread_spend_in`).
`AgentTranscriptFiles` adopts a shared session on a background append and
lists shared sub-agents beside its own. A background append through a
locator with no destination key is serialised by that locator's identity.
Driver-backed stores repair the index on a display read, order pre-identity
roots by `_meta.created`, drop adopted roots beside an unscoped session root,
bound the sub-agent listing query to the thread, and find roots missing from
the index before adoption.
A model-context `TranscriptMessage` now keeps a JSONL row's background
provenance in `extra_metadata` (under `BACKGROUND_METADATA_KEY`, as row
stores do) and the line codec writes it back as the line field, so legacy
rewrites and compactions keep the idempotency key on both backends.
`InMemoryTranscriptLocator` lists a thread's generations and sub-agents, and
`InMemoryTranscriptHistory::read_display` shows recorded partials.

## Unreleased: unapplied steers are back in the run queue when a turn ends

`SteeringForwarderGuard` now drains and delivers under its shutdown gate and
requeues residual (delivered-but-unapplied) steers synchronously in `Drop`,
instead of on a spawned task. Once the guard has dropped, every steer the run
did not apply is in the queue, so a host can collect the leftovers of a turn
deterministically. `RunQueue` guards its lanes with a `std::sync::Mutex`
(never held across an `.await`) and gains synchronous `push_now` and
`drain_now`; its `async` methods are unchanged.
## Unreleased: workspace transcripts on driver ports (tinyagents#453, transcript half)

- `port::WorkspaceStores::open(workspace, agent_id)` (feature `legacy-import`)
  replaces `FileTranscriptLocator::new(workspace)`: transcripts are served by
  `DriverSessionStores` over a SQLite backend kept at `<workspace>/session_store/`
  (an internal driver location), and the caller holds a `TranscriptLocator`
  port. Handles are cached per workspace and agent.
- `legacy::import_transcripts_once` imports the classic `session_raw/*.jsonl`
  files a single time. A marker document in `session_import_markers` records
  completion (not written while a file is unreadable, so it is retried); legacy
  files are only read, never modified, renamed or deleted.
- Not yet moved: `FileTranscriptLocator`/`FileTranscriptHistory` and the
  path-taking free functions (`write_transcript`, `append_transcript_turn`,
  `read_transcript`, `spend`, `view`, `adopt_legacy_session_transcripts`)
  remain until hosts switch to `WorkspaceStores`.

## Unreleased: tinystoragedrivers by git tag

The workspace takes `tinystoragedrivers-core` / `-sqlite` by git tag `v0.11.0`
(the source every module names) instead of a path into a vendored copy. The
`vendor/tinystoragedrivers` submodule and the `[patch]` that redirected the
tinyinference git dependency onto it are removed. Hosts no longer need a
`[patch."https://github.com/tinyhumansai/tinystoragedrivers"]` to keep one set
of port types and one `libsqlite3-sys` in the graph.

## Unreleased: driver stores replace SQLite, file and JSONL backends

Every durable store now runs on `tinystoragedrivers` ports (memory, SQLite,
MongoDB or any other driver the host opens). The bundled SQLite, file and JSONL
implementations are removed and `rusqlite` is no longer a dependency of
session, graph, harness or tasks. Tests may dev-depend on the memory and SQLite
drivers.

### Chat thread store: JSONL removed

- `ConversationStore`, `FileConversationStore` and the `threads::` free
  functions (`append_message`, `list_threads`, ...) are removed.
  `DriverConversationStore` is the only `ConversationPersistence`
  implementation, and now implements `update_message`, `delete_messages_from`
  and `purge_threads` by rewriting the log onto a fresh incarnation and
  swapping the metadata with a compare-and-swap.
- `DriverConversationStore::in_memory()` is the default for library hosts with
  no storage backend.
- `ConversationPurgeStats` moves to `threads::persistence` (same re-exports);
  `ConversationError::Legacy` and `::Worker` are removed.
- Classic `memory/conversations/` files are no longer read by this crate.

### Artifact offload and generated media: blob store

- `artifacts::ArtifactOffload` writes to a tinystoragedrivers `BlobStore`
  (`ArtifactOffload::new(blobs, agent, task)`; `ArtifactOffload::in_memory`
  is the default for library hosts) instead of the filesystem. Pointers carry
  the blob key (`outputs/<name>`); `ArtifactOffload::read` and `::blobs`
  fetch it back. Removed: `ArtifactOffload::root` and `::with_render_root`,
  `OffloadedArtifact::absolute_path`, `OffloadError::Io` and `::SymlinkEscape`
  (new `OffloadError::Storage`), `relative_to_root`, and
  `resolve_artifact_path` (now `resolve_artifact_key`, with no root argument).
- `media::MediaOutput` stores generated images and clips on a blob store
  (`MediaOutput::with_blobs`, in-memory by default) under `generated-media/`;
  tool results report the blob key as `path`. Local reference inputs still
  resolve against the workspace. `rustix` is no longer a dependency.
- Hosts that let the model open these artifacts by file path must expose the
  store through their read tool or materialise it themselves.

### Tool-result artifacts: blob store

- `artifacts::ToolResultArtifactStore` writes to a tinystoragedrivers
  `BlobStore` (`ToolResultArtifactStore::new(blobs, session, redactor,
  read_tool, max_readable_bytes)`; `::in_memory(...)` is the default for
  library hosts). The pointer the model receives is the blob key
  `tool-results/<session>/<tool>/<call>.txt`. `read(key)` fetches an artifact
  back (fail-closed on keys outside `tool-results/`; the legacy
  `artifacts/tool-results/` pointer is still accepted), `purge_session()`
  deletes a session's artifacts, and `with_retention(ttl)` gives every artifact
  a lifetime on drivers with `Capability::BlobTtl`.
- Removed: `ToolResultArtifactStore::detached`, `::try_detached`, `::root`,
  `::is_detached` and `::prune_stale_sessions` (blobs have no modification
  time; use `with_retention` or `purge_session`). Nothing is written to the
  filesystem, and the envelope note changed accordingly.
- Hosts that let the model open the pointer must serve the key from the blob
  store in their read tool (or materialise it).

### Removed, with the replacement

| Removed | Use instead |
| --- | --- |
| `FileCheckpointer`, `SqliteCheckpointer` (graph) | `DriverCheckpointer` |
| `SqliteTaskCache` (graph) | `DriverTaskCache` |
| `SqliteResponseCache` (harness) | `DriverResponseCache` |
| `FileStore`, `JsonlAppendStore` (harness) | `DriverStore`, `DriverAppendStore` |
| harness disk embedding cache, file status store | `DriverEmbeddingCache`, `DriverStatusStore` |
| `JsonlTaskStore`, `JsonlCompletionStore` (tasks) | `DriverTaskStore`, `DriverCompletionStore` |
| session SQLite functions (`record_session_start`, `search_sessions`, `apply_retention`, `with_connection`, ...), `run_ledger` free functions and `sqlite` ledger | `DriverRunLedger` through `SessionHistoryStore` and `RunLedgerStore` |
| `session::transcript::import` (ledger import) | `session::legacy` |
| `SessionCompactionSink` (session) | A host implementation of `tinyagents_harness::summarization::CompactionSink` attached with `RunContext::with_compaction_sink`; only compactions with trusted live boundaries are persisted, and the session crate no longer supplies a sink. |

The `sqlite` cargo features on harness and graph remain as no-ops so existing
manifests keep resolving, and `storage-drivers` is a no-op because the driver
stores are now unconditional.

### Signature changes

- `JsonlSink::new` takes an `Arc<dyn AppendStore>`.
- `SessionTeamLedger::new` and `SessionWorkflowStore::new` take an
  `Arc<dyn RunLedgerStore>`.
- `testkit::conformance::run_ledger_conformance` is async and takes an
  `Arc<dyn RunLedgerStore>`.
- Transcript view projection takes an optional `SpawnCallLookup` instead of a
  workspace directory to look spawn calls up in the ledger.

### Added

- `DriverRunLedger`: team-task claim, complete and shutdown, `prune_*`,
  `trim_session_messages`, `apply_retention` and `reindex_fts`, plus the
  `SessionHistoryStore` trait (sessions, messages, tool calls, full-text search).
- Driver implementations of `TaskCache`, `ResponseCache`, `NamespacedStore`,
  `EmbeddingCache`, `ToolEffectLedger`, `TaskStore`, `CompletionStore` and the
  status store; a `BlobAttachmentStash` over `BlobStore`.

### Migrating legacy data

The `legacy` module (feature `legacy-import`, on by default) reads a classic
workspace read-only and writes it into the driver stores. Rows keep their ids,
existing documents are never overwritten, an interrupted import resumes, and
the search index is rebuilt at the end.

```rust
use tinyagents_session::legacy::import_workspace;

let report = import_workspace(&provider, "agent-id", workspace_dir).await?;
```

- `sessions.db` is opened immutable and read-only. It must be quiescent: a
  database with a WAL or journal sidecar is refused with
  `SourceRequiresSnapshot`, so checkpoint or copy it first.
- `session_raw/*.jsonl` transcripts and turn-state JSON are replayed through
  the same only-if-absent path.
- `DriverRunLedger::import_legacy_database` imports one database directly.

### Turn-state files and the directory attachment stash removed

- `TurnStateStore` (the per-turn JSON files under `memory/conversations/turn_states/`)
  is removed. `TurnStates` is implemented by `InMemoryTurnStates` and
  `DriverTurnStates`; a library host with no session provider gets the memory
  store by default.
- `TurnStateMirror::new` takes an `Arc<dyn TurnStates>` instead of a
  `TurnStateStore`. An interrupted turn's partial answer is carried into the
  transcript only when the host attaches a locator with
  `TurnStateMirror::with_transcripts`; otherwise it stays in the snapshot.
- `multimodal::AttachmentStash` (a directory of files) is removed. Use
  `BlobAttachmentStash` over blob and document ports, which is also the
  implementation hosts with no storage URL should use over the memory driver.
  Retrieval is by id (`read`, `read_data_uri`, `build_index`); there is no
  path to hand a file reader, so `managed_path` and path-keyed `write` results
  are gone.
- The classic turn-state layout is still read, never written, by
  `session::legacy::import_workspace`.

### Not yet moved

These still use files and are migrated in follow-ups (tracked in
tinyhumansai/openhuman#7688, #7768, #7783 and #7809, and tinyagents#453):

- the JSONL transcript writer, reader and `FileTranscriptLocator`
  (path-bound; `session_raw/` adoption, spend and view helpers read them,
  about 12k lines across `session::transcript`).
