//! Lossless durable transcript histories.
//!
//! This module deliberately exposes [`TranscriptMessage`] rather than a
//! provider or harness message. A transcript is an on-disk compatibility
//! boundary: it must preserve native tool calls, malformed raw arguments,
//! usage, thinking content, provider extensions and caller-owned metadata.
//! Converting it through a narrower runtime message here would make a later
//! replay silently lossy. Hosts perform any runtime conversion explicitly at
//! their own boundary.
//!
//! A history handle is bound to a transcript file. Thread and agent discovery
//! belong to [`TranscriptLocator`], because one thread can have several
//! transcript stems (for example a root agent and sub-agents).
//!
//! Every mutation is append-only: a reduced logical context is represented by
//! a `{"kind":"compaction","replacement":[…]}` record, never a destructive
//! rewrite. [`TranscriptHistory::clear`] is therefore an empty compaction.
//!

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::transcript::types::TranscriptMessage;

use crate::transcript::{
    SessionTranscript, TranscriptMeta, TurnUsage, append_transcript_turn, find_latest_transcript,
    find_root_transcript_for_thread, find_root_transcript_for_thread_scoped, read_transcript,
    resolve_keyed_transcript_path,
};

/// One turn's worth of transcript write, borrowed.
///
/// The fields mirror [`append_transcript_turn`]'s argument list one-for-one and
/// in order, so [`TranscriptHistory::append_turn`]'s forwarding is visually
/// checkable against the format's own signature. Nothing is transformed on the
/// way through; that is the entire correctness claim of this seam and
/// `append_turn_is_byte_identical_to_the_free_function` in the tests pins it.
///
/// `prev` is a field rather than handle state on purpose: the turn path tracks
/// the previously-persisted logical set in memory on `Agent`
/// (`persisted_transcript_messages`) precisely so it never has to re-read a
/// growing file, and a disk re-read is not a faithful substitute — see
/// `FileTranscriptHistory::write_logical_set`.
pub struct TranscriptTurn<'a> {
    /// Logical message set already persisted, for the extension-vs-compaction diff.
    pub prev: &'a [TranscriptMessage],
    /// Logical message set after this turn.
    pub next: &'a [TranscriptMessage],
    /// `_meta` header to append after this turn's lines.
    pub meta: &'a TranscriptMeta,
    /// Usage + provenance attributed to the turn's last assistant row.
    pub turn_usage: Option<&'a TurnUsage>,
    /// Caller-provided request id, stamped on every line of the turn.
    pub request_id: Option<&'a str>,
}

/// The seam a host turn path holds as `Arc<dyn TranscriptHistory>`.
///
/// `append_turn` is deliberately **sync**: `persist_session_transcript` is a
/// sync `&mut self` method and the whole write chain under it is sync, so an
/// async method here would ripple `.await` through the turn loop for no gain.
pub trait TranscriptHistory: TranscriptRead {
    /// Appends one turn, forwarding every argument to the format owner.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()>;

    /// Returns the lossless model-context replay of this transcript.
    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>>;

    /// Appends one durable message while preserving all of its fields.
    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()>;

    /// Replaces the logical model context by appending a compaction record.
    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()>;

    /// Clears the logical model context by appending an empty compaction.
    fn clear(&self) -> anyhow::Result<()>;
}

/// The read half of a bound transcript — the seam the turn path's two resume
/// reads hold.
///
/// Split out of [`TranscriptHistory`] rather than added as one more method on it,
/// for a reason that is not stylistic: a *discovered* transcript can still be a
/// legacy `.md` file (see [`FileTranscriptHistory::opened_at`]), and
/// `append_transcript_turn` writes JSONL. Handing discovery results out as
/// `Arc<dyn TranscriptRead>` makes it impossible to `append_turn` into
/// one by construction, instead of by convention.
///
/// Sync for the same reason [`TranscriptHistory::append_turn`] is: both callers are
/// sync `&mut self` methods on `Agent`.
pub trait TranscriptRead: Send + Sync {
    /// The transcript file this handle is bound to.
    ///
    /// The turn path still needs the concrete path after the read:
    /// `maybe_shadow_read_session_store` takes `&Path`, and the dual-write
    /// mirror derives its record key from `file_stem()`.
    fn path(&self) -> &Path;

    /// The model-context replay of this transcript, `_meta` included, or
    /// `Ok(None)` when the file does not exist.
    ///
    /// Exactly [`read_transcript`], so compaction records have already replaced
    /// the accumulator and `interrupted: true` partials are already skipped —
    /// §3.1's "single most important constraint". Returning the whole
    /// [`SessionTranscript`] rather than messages alone is what lets the shadow
    /// read keep working through this seam.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>>;
}

/// Resolves transcripts by the two keys the turn path actually has, and binds
/// this session's own write handle.
///
/// One injected object covers the whole turn path: both resume reads and the
/// first-write bind. A host holds it as `Option<Arc<dyn TranscriptLocator>>`
/// and falls back to [`FileTranscriptLocator`] built from the *current*
/// `workspace_dir` — lazily, never frozen at build time,
/// because tests reassign `agent.workspace_dir` after `build()` and a
/// build-time locator would silently keep pointing at the old directory.
pub trait TranscriptLocator: Send + Sync {
    /// Newest transcript for `agent_name` in this session's raw subtree,
    /// including the legacy `session_raw/DDMMYYYY/` + `.md` fallback.
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// Newest **root** transcript whose `_meta.thread_id` matches.
    ///
    /// Root-only on purpose: several transcripts share one thread id (every
    /// sub-agent spawned within it does), so a stem-keyed lookup would be
    /// ambiguous.
    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// [`Self::root_for_thread`], additionally scoped to `agent_id` when
    /// given — see
    /// [`transcript::find_root_transcript_for_thread_scoped`](crate::transcript::find_root_transcript_for_thread_scoped)
    /// for why. Defaults to the unscoped lookup so an implementor that never
    /// serves several distinct agents over the same `thread_id` (this test
    /// double, notably) does not have to know about agent scoping at all.
    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        let _ = agent_id;
        self.root_for_thread(thread_id)
    }

    /// Binds (creating on first write) this session's own write handle for
    /// `stem`, with `seed` used only when no file exists yet.
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>>;
}

/// The default [`TranscriptLocator`]: real files under
/// `{workspace_dir}/session_raw`.
///
/// Thin by design — each method wraps exactly one `transcript::` free function
/// and changes nothing about it, so swapping the turn path onto the locator is
/// behaviour-preserving.
pub struct FileTranscriptLocator {
    workspace_dir: PathBuf,
}

impl FileTranscriptLocator {
    /// Builds a locator rooted at `workspace_dir` (i.e. it resolves
    /// `{workspace_dir}/session_raw/...`).
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }
}

impl TranscriptLocator for FileTranscriptLocator {
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_latest_transcript(&self.workspace_dir, agent_name)?;
        tracing::debug!(
            "[transcript-history] locator latest_for_agent agent={agent_name} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(agent_name),
        )))
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_root_transcript_for_thread(&self.workspace_dir, thread_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread thread={thread_id} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        // Same cross-dir, newest-wins scan as `root_for_thread`, additionally
        // filtered on `_meta.agent_id` so one runtime agent's resume cannot
        // pick up a different agent's transcript for a caller-reused
        // `thread_id` — see `find_root_transcript_for_thread_scoped`.
        let path =
            find_root_transcript_for_thread_scoped(&self.workspace_dir, thread_id, agent_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread_scoped thread={thread_id} \
             agent_id={agent_id:?} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(Arc::new(FileTranscriptHistory::new(
            &self.workspace_dir,
            stem,
            seed,
        )?))
    }
}

/// A placeholder `_meta` for a handle bound to an already-existing transcript.
///
/// `seed_meta` is consulted only when the file is **absent**, and a discovered
/// path exists by definition, so this value is never written. It exists because
/// [`FileTranscriptHistory`] is one type serving both roles; giving read-only
/// handles a `None` meta would mean an `Option` field every write path then has
/// to unwrap for no benefit.
fn seed_meta_for_discovered(agent_name: &str) -> TranscriptMeta {
    TranscriptMeta {
        agent_name: agent_name.to_string(),
        agent_id: None,
        agent_type: None,
        dispatcher: String::new(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// A lossless history backed by one `session_raw/{stem}.jsonl` transcript.
///
/// Construct with [`FileTranscriptHistory::new`] (workspace-rooted, i.e.
/// `{workspace}/session_raw/`). The `seed_meta` is used only when the
/// transcript file does not exist yet; for an existing file the authoritative
/// cumulative `_meta` is read back from disk so turn counts and token rollups
/// keep accumulating rather than resetting.
pub struct FileTranscriptHistory {
    /// Fully-resolved transcript file, fixed at construction.
    ///
    /// Resolved eagerly rather than derived per call from a `(workspace, stem)`
    /// pair: the old shape hardcoded `{workspace}/session_raw/`, which is the
    /// **wrong directory** for a canonical session and would have silently
    /// cross-written into the canonical session's transcripts the moment this
    /// handle was wired into the turn path.
    path: PathBuf,
    /// `_meta` used for the very first write, before a file exists.
    seed_meta: TranscriptMeta,
}

impl FileTranscriptHistory {
    /// Binds a history handle to `{workspace_dir}/session_raw/{stem}.jsonl`.
    ///
    pub fn new(
        workspace_dir: impl AsRef<Path>,
        stem: &str,
        seed_meta: TranscriptMeta,
    ) -> anyhow::Result<Self> {
        let path = resolve_keyed_transcript_path(workspace_dir.as_ref(), stem)?;
        tracing::debug!(
            "[transcript-history] bound stem={stem} path={}",
            path.display()
        );
        Ok(Self { path, seed_meta })
    }

    /// Binds a handle to an **already-discovered** transcript file, verbatim.
    ///
    /// Deliberately does **not** go through `resolve_keyed_transcript_path*`,
    /// which the two stem constructors above use. That helper `create_dir_all`s
    /// its parent and forces a `.jsonl` extension — both wrong for a discovered
    /// path: `find_latest_transcript` can still return a legacy `.md`
    /// file (`read_transcript` routes by extension), and re-resolving would
    /// mangle it into a sibling `.jsonl` that does not exist while creating
    /// stray directories on a pure read.
    ///
    /// Hand the result out as `Arc<dyn TranscriptRead>`, not
    /// `Arc<dyn TranscriptHistory>` — see [`TranscriptRead`]'s doc.
    pub fn opened_at(path: PathBuf, seed_meta: TranscriptMeta) -> Self {
        tracing::debug!(
            "[transcript-history] opened discovered path={}",
            path.display()
        );
        Self { path, seed_meta }
    }

    /// This handle's transcript file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the current transcript, or `None` when no file exists yet.
    ///
    /// A missing transcript is the normal first-turn state, not an error.
    fn read(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            return Ok(None);
        }
        read_transcript(&self.path).map(Some)
    }

    /// The logical (model-context) message set currently on disk.
    ///
    /// Routes through [`read_transcript`], so compaction records have already
    /// replaced the accumulator and `interrupted: true` partials are skipped.
    fn persisted(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self.read()?.map(|t| t.messages).unwrap_or_default())
    }

    /// The `_meta` to write: the file's own cumulative meta when it exists,
    /// otherwise this handle's seed.
    ///
    /// Uses the existing durable metadata as a default when a caller does not
    /// caller-computed meta. The **turn path must never route through here** —
    /// it computes `turn_count` and the four token/cost rollups fresh each turn,
    /// and re-reading the file's `_meta` would freeze them at the previous
    /// turn's values, silently breaking `read_thread_usage_summary`.
    fn meta_for_write(&self) -> anyhow::Result<TranscriptMeta> {
        Ok(self
            .read()?
            .map(|t| t.meta)
            .unwrap_or_else(|| self.seed_meta.clone()))
    }

    /// Writes `next` as the new logical set, diffing against what is persisted.
    ///
    /// Routes through [`TranscriptHistory::append_turn`] so every write in this
    /// module — trait-driven and turn-path alike — funnels through one call to
    /// [`append_transcript_turn`], and the extension-vs-compaction decision
    /// stays with the format owner rather than drifting here.
    ///
    /// The `self.persisted()` disk re-read is what the generic trait path has
    /// to do, and is deliberately **not** what the turn path does.
    /// [`read_transcript`] reconstructs `TranscriptMessage`s from line records: the
    /// `failure` / `failure_detail` fields have been lifted out of
    /// `extra_metadata` and turn-usage fields hoisted to top-level line fields.
    /// Feeding that back in as `prev` would make `common_prefix_len` mismatch
    /// at the first such message, so the writer would emit a full compaction
    /// record — re-appending the entire message set — on every single turn.
    fn write_logical_set(&self, next: &[TranscriptMessage]) -> anyhow::Result<()> {
        let prev = self.persisted()?;
        let meta = self.meta_for_write()?;
        self.append_turn(TranscriptTurn {
            prev: &prev,
            next,
            meta: &meta,
            turn_usage: None,
            request_id: None,
        })
    }
}

impl TranscriptRead for FileTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    /// Same call the free-function readers make, on the same path, with the
    /// same return type — so there is nothing left for the round trip to lose.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            tracing::debug!(
                "[transcript-history] read_session absent path={}",
                self.path.display()
            );
            return Ok(None);
        }
        let session = read_transcript(&self.path)?;
        tracing::debug!(
            "[transcript-history] read_session messages={} path={}",
            session.messages.len(),
            self.path.display()
        );
        Ok(Some(session))
    }
}

impl TranscriptHistory for FileTranscriptHistory {
    /// Pure forwarder: every argument reaches [`append_transcript_turn`]
    /// untouched, so the bytes this writes are identical to what the free
    /// function would have written at the call site.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        tracing::debug!(
            "[transcript-history] append_turn prev={} next={} usage={} request_id={:?} path={}",
            turn.prev.len(),
            turn.next.len(),
            turn.turn_usage.is_some(),
            turn.request_id,
            self.path.display()
        );
        append_transcript_turn(
            &self.path,
            turn.prev,
            turn.next,
            turn.meta,
            turn.turn_usage,
            turn.request_id,
        )
    }
    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        self.persisted()
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        let mut next = self.persisted()?;
        next.push(message);
        self.write_logical_set(&next)
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        self.write_logical_set(messages)
    }

    fn clear(&self) -> anyhow::Result<()> {
        if !self.path.exists() {
            return Ok(());
        }
        self.write_logical_set(&[])
    }
}
