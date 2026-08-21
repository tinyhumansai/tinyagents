//! Failure modes of attachment resolution.
//!
//! Every variant carries the offending `input` verbatim. That is deliberate:
//! a turn can reference several attachments and the caller has to be able to
//! say *which* one it could not use, which a bare "unsupported MIME type"
//! cannot.

use thiserror::Error;

/// An attachment reference that could not be resolved into a payload.
#[derive(Debug, Error)]
pub enum MultimodalError {
    /// More `[IMAGE:…]` markers in the turn than the configured cap allows.
    #[error("multimodal image limit exceeded: max_images={max_images}, found={found}")]
    TooManyImages {
        /// The effective cap.
        max_images: usize,
        /// How many markers the turn carried.
        found: usize,
    },

    /// A resolved image exceeded the configured per-image byte cap.
    #[error("multimodal image size limit exceeded for '{input}': {size_bytes} bytes > {max_bytes} bytes")]
    ImageTooLarge {
        /// The reference as written in the marker.
        input: String,
        /// Measured size.
        size_bytes: usize,
        /// The effective cap.
        max_bytes: usize,
    },

    /// The image's MIME type is not one the provider contract accepts.
    #[error("multimodal image MIME type is not allowed for '{input}': {mime}")]
    UnsupportedMime {
        /// The reference as written in the marker.
        input: String,
        /// The detected MIME type.
        mime: String,
    },

    /// An `http(s)` image reference arrived with remote fetch turned off.
    #[error("multimodal remote image fetch is disabled for '{input}'")]
    RemoteFetchDisabled {
        /// The reference as written in the marker.
        input: String,
    },

    /// A local image path did not resolve to a readable file.
    #[error("multimodal image source not found or unreadable: '{input}'")]
    ImageSourceNotFound {
        /// The reference as written in the marker.
        input: String,
    },

    /// The marker was structurally invalid (bad `data:` URI, bad gzip, …).
    #[error("invalid multimodal image marker '{input}': {reason}")]
    InvalidMarker {
        /// The reference as written in the marker.
        input: String,
        /// What was wrong with it.
        reason: String,
    },

    /// The remote image fetch itself failed.
    #[error("failed to download remote image '{input}': {reason}")]
    RemoteFetchFailed {
        /// The reference as written in the marker.
        input: String,
        /// Transport or status detail.
        reason: String,
    },

    /// The local image read itself failed.
    #[error("failed to read local image '{input}': {reason}")]
    LocalReadFailed {
        /// The reference as written in the marker.
        input: String,
        /// I/O detail.
        reason: String,
    },

    /// More `[FILE:…]` markers in the turn than the configured cap allows.
    #[error("multimodal file limit exceeded: max_files={max_files}, found={found}")]
    TooManyFiles {
        /// The effective cap. `0` is the hard-disable sentinel — see
        /// [`FileLimits::files_disabled`](super::config::FileLimits::files_disabled).
        max_files: usize,
        /// How many markers the turn carried.
        found: usize,
    },

    /// A resolved file exceeded the configured per-file byte cap.
    #[error(
        "multimodal file size limit exceeded for '{input}': {size_bytes} bytes > {max_bytes} bytes"
    )]
    FileTooLarge {
        /// The reference as written in the marker.
        input: String,
        /// Measured size.
        size_bytes: usize,
        /// The effective cap.
        max_bytes: usize,
    },

    /// The file's MIME type is not on the host's allowlist.
    #[error(
        "multimodal file MIME type '{mime}' for '{input}' is not allowed; supported: {supported}"
    )]
    UnsupportedFileMime {
        /// The reference as written in the marker.
        input: String,
        /// The detected MIME type.
        mime: String,
        /// The configured allowlist, rendered for the message.
        supported: String,
    },

    /// A local file path did not resolve to a readable file.
    #[error("multimodal file source not found or unreadable: '{input}'")]
    FileSourceNotFound {
        /// The reference as written in the marker.
        input: String,
    },

    /// An `http(s)` file reference arrived with remote fetch turned off.
    #[error("multimodal remote file fetch is disabled for '{input}'")]
    RemoteFileFetchDisabled {
        /// The reference as written in the marker.
        input: String,
    },

    /// The remote file fetch itself failed.
    #[error("failed to download remote file '{input}': {reason}")]
    RemoteFileFetchFailed {
        /// The reference as written in the marker.
        input: String,
        /// Transport or status detail.
        reason: String,
    },

    /// The local file read itself failed.
    #[error("failed to read local file '{input}': {reason}")]
    LocalFileReadFailed {
        /// The reference as written in the marker.
        input: String,
        /// I/O detail.
        reason: String,
    },

    /// The marker was structurally invalid (bad `data:` URI, bad gzip, …).
    #[error("invalid multimodal file marker '{input}': {reason}")]
    InvalidFileMarker {
        /// The reference as written in the marker.
        input: String,
        /// What was wrong with it.
        reason: String,
    },
}

/// Result alias for attachment resolution.
pub type Result<T> = std::result::Result<T, MultimodalError>;
