//! Per-thread CC session UUID persistence.
//!
//! The `claude` CLI's `--resume <uuid>` only reuses a server-side session
//! if we pass it the same UUIDv4 we used the first time. We map an
//! OpenHuman thread id → CC session UUID in a JSON file under the
//! workspace.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoreFile {
    /// thread_id → CC session uuid (v4)
    sessions: HashMap<String, String>,
    /// thread_id → fingerprints of user turns already delivered to that
    /// thread's CC session (see `input_builder::pending_fingerprints`).
    #[serde(default)]
    delivered: HashMap<String, Vec<String>>,
}

/// Most delivered-turn fingerprints remembered per thread.
const MAX_DELIVERED_PER_THREAD: usize = 256;

/// Disk-backed session store. Cheap to clone — it's `Arc`-shareable via
/// the holding `ClaudeCodeProvider`.
#[derive(Debug)]
pub struct SessionStore {
    path: PathBuf,
    inner: Mutex<StoreFile>,
}

impl SessionStore {
    /// Open (or initialize) the session store at `workspace/claude-code-sessions.json`.
    pub fn open(workspace_dir: &Path) -> Self {
        let path = workspace_dir.join("claude-code-sessions.json");
        let inner = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<StoreFile>(&s).ok())
            .unwrap_or_default();
        Self {
            path,
            inner: Mutex::new(inner),
        }
    }

    /// Re-read the file so changes made by another store instance on the same
    /// workspace (two providers sharing a session) are seen. A missing or
    /// unreadable file keeps the in-memory state.
    fn refresh(&self, guard: &mut StoreFile) {
        if let Some(fresh) = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| serde_json::from_str::<StoreFile>(&s).ok())
        {
            *guard = fresh;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreFile> {
        let mut guard = self.inner.lock().expect("session store mutex poisoned");
        self.refresh(&mut guard);
        guard
    }

    /// Lookup an existing CC session UUID for `thread_id`.
    pub fn get(&self, thread_id: &str) -> Option<String> {
        self.lock().sessions.get(thread_id).cloned()
    }

    /// Fingerprints of user turns already delivered to `thread_id`'s session.
    #[cfg(test)]
    pub fn delivered(&self, thread_id: &str) -> HashSet<String> {
        self.lock()
            .delivered
            .get(thread_id)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Atomically reserve `fingerprints` for `thread_id` before the CLI is
    /// spawned. Returns the fingerprints that were already delivered or
    /// reserved by an earlier call (the caller must not re-send those) and a
    /// guard that releases this call's own reservations unless
    /// [`DeliveryClaim::commit`] is called, so a failed turn stays retryable.
    pub fn claim_delivered(
        self: &Arc<Self>,
        thread_id: &str,
        fingerprints: &[String],
    ) -> (HashSet<String>, DeliveryClaim) {
        let mut guard = self.lock();
        let entry = guard.delivered.entry(thread_id.to_string()).or_default();
        let already: HashSet<String> = entry.iter().cloned().collect();
        let mut mine = Vec::new();
        for fingerprint in fingerprints {
            if !already.contains(fingerprint) && !mine.contains(fingerprint) {
                mine.push(fingerprint.clone());
            }
        }
        if !mine.is_empty() {
            entry.extend(mine.iter().cloned());
            trim(entry, fingerprints.len());
            if let Err(error) = self.persist(&guard) {
                tracing::warn!("[claude-code][session-store] failed to persist claim: {error}");
            }
        }
        let claim = DeliveryClaim {
            store: Arc::clone(self),
            thread_id: thread_id.to_string(),
            claimed: mine,
        };
        (already, claim)
    }

    fn release(&self, thread_id: &str, fingerprints: &[String]) {
        if fingerprints.is_empty() {
            return;
        }
        let mut guard = self.lock();
        if let Some(entry) = guard.delivered.get_mut(thread_id) {
            entry.retain(|f| !fingerprints.contains(f));
        }
        if let Err(error) = self.persist(&guard) {
            tracing::warn!("[claude-code][session-store] failed to persist release: {error}");
        }
    }

    /// Record user-turn fingerprints delivered to `thread_id`'s session.
    pub fn record_delivered(
        &self,
        thread_id: &str,
        fingerprints: &[String],
    ) -> std::io::Result<()> {
        if fingerprints.is_empty() {
            return Ok(());
        }
        let mut guard = self.lock();
        let entry = guard.delivered.entry(thread_id.to_string()).or_default();
        for fingerprint in fingerprints {
            if !entry.contains(fingerprint) {
                entry.push(fingerprint.clone());
            }
        }
        trim(entry, fingerprints.len());
        self.persist(&guard)
    }

    /// Persist a thread → UUID mapping. A different session UUID starts with an
    /// empty delivered-turn record, since it has received nothing yet.
    pub fn set(&self, thread_id: &str, uuid: &str) -> std::io::Result<()> {
        let mut guard = self.lock();
        let previous = guard
            .sessions
            .insert(thread_id.to_string(), uuid.to_string());
        if previous.as_deref() != Some(uuid) {
            guard.delivered.remove(thread_id);
        }
        self.persist(&guard)
    }

    fn persist(&self, guard: &StoreFile) -> std::io::Result<()> {
        let serialized = serde_json::to_string_pretty(guard).map_err(std::io::Error::other)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, serialized)
    }
}

/// Keep the newest [`MAX_DELIVERED_PER_THREAD`] fingerprints, but never fewer
/// than `keep_at_least` (the batch just recorded), so a turn still eligible for
/// retry is not forgotten.
fn trim(entry: &mut Vec<String>, keep_at_least: usize) {
    let limit = MAX_DELIVERED_PER_THREAD.max(keep_at_least);
    if entry.len() > limit {
        let excess = entry.len() - limit;
        entry.drain(..excess);
    }
}

/// Reservation of delivered-turn fingerprints taken by
/// [`SessionStore::claim_delivered`]. Dropping it without `commit` releases
/// the reservations (the turn failed, so the session never received them).
#[derive(Debug)]
pub struct DeliveryClaim {
    store: Arc<SessionStore>,
    thread_id: String,
    claimed: Vec<String>,
}

impl DeliveryClaim {
    /// The turn reached the session; keep the reservations as delivered.
    pub fn commit(mut self) {
        self.claimed.clear();
    }
}

impl Drop for DeliveryClaim {
    fn drop(&mut self) {
        self.store.release(&self.thread_id, &self.claimed);
    }
}

/// Random RFC-4122 v4 UUID, formatted lower-case with hyphens.
pub fn generate_uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// CC accepts only RFC-4122 v4. Older stores might carry pre-v4 strings;
/// we treat those as missing and regenerate.
pub fn is_uuid_v4(s: &str) -> bool {
    let s = s.as_bytes();
    if s.len() != 36 {
        return false;
    }
    let hyphens = [8, 13, 18, 23];
    for (i, b) in s.iter().enumerate() {
        let is_hyphen = hyphens.contains(&i);
        if is_hyphen {
            if *b != b'-' {
                return false;
            }
        } else if !b.is_ascii_hexdigit() {
            return false;
        }
    }
    // version nibble (index 14) must be '4'; variant nibble (index 19)
    // must be one of 8/9/a/b
    s[14] == b'4' && matches!(s[19], b'8' | b'9' | b'a' | b'b' | b'A' | b'B')
}

#[cfg(test)]
#[path = "session_store_tests.rs"]
mod tests;
