//! Generic attachment intake, bounded archive inspection, and legacy
//! resolution for `[IMAGE:…]` and `[FILE:…]` markers.
//!
//! [`resolve_attachment`] returns original decoded bytes and metadata before
//! extraction. [`inspect_archive`] lists ZIP/TAR/TAR.GZ contents without disk
//! extraction. Hosts authorize paths and select storage and provider media
//! blocks; the APIs here introduce no new transcript variants.
//!
//! A user attaches a picture or a document. Somewhere between the text box and
//! the provider, that attachment has to become bytes the model can read —
//! validated, size-capped, MIME-checked, and rendered into the message. This
//! module is that pipeline, minus every decision that belongs to a host.
//!
//! ## The marker convention
//!
//! Attachments travel inside message text as markers rather than as a parallel
//! structured field. That is what makes them survive every hop a host already
//! has — persistence, summarisation, delegation to a sub-agent — without each
//! of those learning about attachments. The cost is that the text is a wire
//! format, which is why [`markers`] is careful about malformed input.
//!
//! ## The two pipelines
//!
//! Images and files are symmetrical up to the payload and then diverge:
//!
//! | | Images | Files |
//! | --- | --- | --- |
//! | Marker | `[IMAGE:<ref>]` | `[FILE:<ref>]` |
//! | Sources | `data:` / `http(s)` / path | same |
//! | MIME set | fixed, five types | host allowlist |
//! | Payload | inline base64 `data:` URI | extracted text, or metadata + hash |
//!
//! A file never inlines its bytes. Handing a model raw binary costs a great
//! deal of context and tells it nothing, so an unextractable format surfaces as
//! a header naming it and a content hash — enough for the agent to talk about
//! the attachment, and to pass its path to a tool that can actually open it.
//!
//! ## What stays with the host
//!
//! Host decisions remain host-owned because they depend on its runtime,
//! storage, or threat model:
//!
//! | Host owns | Why |
//! | --- | --- |
//! | The `reqwest::Client` | retained in resolver signatures for compatibility; remote fetches use a DNS-pinned direct client with redirects disabled |
//! | [`TextExtractor`] | which document parser (if any) a host carries, and how long it may run |
//! | The stash policy | which storage scope holds bytes between ingress and dispatch, how large it may grow, how long they live ([`BlobAttachmentStash`] is the mechanism) |
//! | The source resolver and scope | only the host can authorize a path, fetch remote data, and bind a request to trusted access context |
//! | The selected model profile | catalog facts and transport serialization support are independent inputs |
//! | Historical placeholders and image readout | safe path hints and non-vision model choice are host policies |
//!
//! The stash deserves a note, because the split is not obvious. A host that
//! persists conversations should **not** persist a multi-megabyte `data:` URI:
//! it floods every downstream consumer that reads message text. The convention
//! this module supports is to replace each image marker at ingress with a
//! compact `[Image: … #att:<id>]` placeholder ([`markers::image_placeholder`]),
//! store the bytes out of band, and rehydrate at dispatch
//! ([`markers::rehydrate_placeholders_in_text`]) — but only for a
//! vision-capable model, since a text-only one gains nothing from an image it
//! cannot see. The id-to-path index is a parameter, so where those bytes live
//! and when they expire stay host decisions.
//!
//! ## Order of operations
//!
//! Two rules, both learned the hard way:
//!
//! 1. **Count before resolving.** [`request::MediaRequestModel`] checks latest
//!    turn caps and its bounded history budget against raw markers before any
//!    source resolver is called. A cap enforced after the fetch is not a cap.
//! 2. **Preserve the hard disable.** `max_files == 0` means *none* and stays
//!    zero in [`config::FileLimits::effective`]. The request decorator checks
//!    this before calling a host resolver.

pub mod archive;
pub mod config;
pub mod data_uri;
pub mod error;
pub mod markers;
pub mod mime;
pub mod payload;
pub mod png;
pub mod request;
pub use png::optimize_png_lossless;
pub mod blob_stash;
pub mod resolve;
pub mod types;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

pub use archive::{
    ArchiveEntry, ArchiveEntryKind, ArchiveError, ArchiveFormat, ArchiveLimits, ArchiveListing,
    ArchiveTruncation, inspect_archive,
};
pub use blob_stash::{BlobAttachmentStash, StashedAttachment};
pub use config::{ALLOWED_IMAGE_MIME_TYPES, FileLimits, ImageLimits};
pub use error::{MultimodalError, Result};
pub use markers::{
    ATTACHMENT_MARKER_PREFIX, AttachmentRef, AttachmentSegment, FILE_MARKER_PREFIX,
    IMAGE_MARKER_PREFIX, IMAGE_PLACEHOLDER_PREFIX, IMAGE_STASH_REF, attachment_segments,
    enrich_user_message, extract_image_placeholders_in_text, extract_ollama_image_payload,
    image_markers_from_message, image_placeholder, is_safe_relative_path, parse_attachment_markers,
    parse_file_markers, parse_image_markers, rehydrate_placeholders_in_text,
    sanitize_attachment_filename, should_forward_parent_images, text_has_image_placeholders,
};
pub use payload::{FilePayload, compose_multimodal_message, sha256_prefix, truncate_chars};
pub use request::{
    MediaFallbackCache, MediaFallbackRenderer, MediaHistoryFallback, MediaImageReadout,
    MediaRequestConfig, MediaRequestModel, MediaSourceKind, MediaSourceMetadata,
    MediaSourceResolver, ResolvedMedia,
};
pub use resolve::{
    NoTextExtractor, TextExtractor, resolve_attachment, resolve_file, resolve_image,
};

pub use types::{ResolvedAttachment, UnknownMimePolicy};
