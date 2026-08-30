//! Crate-owned attachment limits.
//!
//! Follows the [`harness::config`](crate::config) convention: the
//! crate defines the struct, the host maps its own schema into it. Nothing
//! here reads a file or an environment variable.
//!
//! ## Why the clamps live here
//!
//! [`ImageLimits::effective`] and [`FileLimits::effective`] clamp the
//! configured values into runtime bounds. That clamping is part of the
//! resolution contract, not host policy: a host that forgot to clamp would
//! hand an unbounded `max_image_size_mb` straight into an allocation. Hosts
//! still choose the *configured* values; the crate only refuses to act on
//! absurd ones.
//!
//! ## The one asymmetry worth knowing
//!
//! `max_files == 0` is a **hard-disable sentinel**, and it is checked before
//! the clamp rather than after. The clamp lifts `0` to `1`, so a caller that
//! only consulted [`FileLimits::effective`] would admit a single file marker
//! from a source that asked for none. [`FileLimits::files_disabled`] is the
//! check that must run first; [`resolve`](super::resolve) does exactly that.

use serde::{Deserialize, Serialize};

/// Image MIME types the provider contract accepts.
///
/// Not configurable: this is the set the inline-`data:`-URI path can actually
/// encode, so a host adding to it would produce a payload no provider reads.
pub const ALLOWED_IMAGE_MIME_TYPES: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
];

/// Per-turn limits for `[IMAGE:…]` markers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageLimits {
    /// Maximum images accepted in one turn. Clamped to `1..=16`.
    pub max_images: usize,
    /// Maximum decoded size of a single image, in MiB. Clamped to `1..=20`.
    pub max_image_size_mb: usize,
    /// Whether `http(s)` image references may be fetched.
    pub allow_remote_fetch: bool,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_images: 4,
            max_image_size_mb: 8,
            allow_remote_fetch: false,
        }
    }
}

impl ImageLimits {
    /// Clamp configured values into runtime bounds: `(max_images, max_image_size_mb)`.
    pub fn effective(&self) -> (usize, usize) {
        (
            self.max_images.clamp(1, 16),
            self.max_image_size_mb.clamp(1, 20),
        )
    }

    /// Effective per-image byte cap, saturating rather than overflowing.
    pub fn max_image_bytes(&self) -> usize {
        self.effective().1.saturating_mul(1024 * 1024)
    }
}

/// Per-turn limits for `[FILE:…]` markers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FileLimits {
    /// Maximum files accepted in one turn. Clamped to `1..=16` — except `0`,
    /// which is the hard-disable sentinel checked by
    /// [`FileLimits::files_disabled`] before any clamp applies.
    pub max_files: usize,
    /// Maximum size of a single file, in MiB. Clamped to `1..=50`.
    pub max_file_size_mb: usize,
    /// Maximum extracted text retained per file, in Unicode scalar values.
    /// Clamped to `1_000..=200_000`.
    pub max_extracted_text_chars: usize,
    /// Whether `http(s)` file references may be fetched.
    pub allow_remote_fetch: bool,
    /// MIME allowlist. Unlike images this *is* host-configurable: which
    /// document formats are acceptable is a product decision, and the
    /// binary-only ones surface as metadata references rather than bytes.
    pub allowed_mime_types: Vec<String>,
}

impl Default for FileLimits {
    fn default() -> Self {
        Self {
            max_files: 4,
            max_file_size_mb: 16,
            max_extracted_text_chars: 50_000,
            allow_remote_fetch: false,
            allowed_mime_types: default_allowed_file_mime_types(),
        }
    }
}

/// The default file MIME allowlist: extractable prose formats plus the
/// binary-only formats that surface as metadata-only references.
pub fn default_allowed_file_mime_types() -> Vec<String> {
    [
        // Extractable text formats.
        "application/pdf",
        "text/plain",
        "text/csv",
        "text/markdown",
        // Binary-only formats surfaced as metadata-only references.
        "application/zip",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/octet-stream",
    ]
    .iter()
    .map(|value| (*value).to_string())
    .collect()
}

impl FileLimits {
    /// Clamp configured values into runtime bounds:
    /// `(max_files, max_file_size_mb, max_extracted_text_chars)`.
    pub fn effective(&self) -> (usize, usize, usize) {
        (
            self.max_files.clamp(1, 16),
            self.max_file_size_mb.clamp(1, 50),
            self.max_extracted_text_chars.clamp(1_000, 200_000),
        )
    }

    /// Effective per-file byte cap, saturating rather than overflowing.
    pub fn max_file_bytes(&self) -> usize {
        self.effective().1.saturating_mul(1024 * 1024)
    }

    /// `true` when file markers are disabled outright.
    ///
    /// The sentinel exists so a host can refuse `[FILE:…]` resolution for a
    /// turn whose text came from somewhere it does not trust, without editing
    /// the operator's own allowlist. It must be consulted **before**
    /// [`FileLimits::effective`], whose clamp lifts `0` to `1`.
    pub fn files_disabled(&self) -> bool {
        self.max_files == 0
    }

    /// `true` iff `mime` is on the allowlist, case-insensitively.
    pub fn is_mime_allowed(&self, mime: &str) -> bool {
        let needle = mime.to_ascii_lowercase();
        self.allowed_mime_types
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&needle))
    }

    /// The allowlist rendered for an error message.
    pub fn supported_rendered(&self) -> String {
        self.allowed_mime_types.join(", ")
    }
}
