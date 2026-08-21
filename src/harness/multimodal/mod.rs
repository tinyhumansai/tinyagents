//! Attachment resolution for `[IMAGE:…]` and `[FILE:…]` markers.
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
//! Four things, each because it depends on the host's own runtime, storage, or
//! threat model rather than on anything universal:
//!
//! | Host owns | Why |
//! | --- | --- |
//! | The `reqwest::Client` | proxy configuration and timeouts are host policy; this module borrows one |
//! | [`TextExtractor`] | which document parser (if any) a host carries, and how long it may run |
//! | The attachment stash | where bytes live between ingress and dispatch, and their lifetime |
//! | Message-level counting | only the host knows what its message type is |
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
//! 1. **Count before resolving.** [`config::FileLimits::files_disabled`] and the
//!    per-turn caps are checked against the raw markers, before any read
//!    happens. A cap enforced after the fetch is not a cap.
//! 2. **Check the sentinel before the clamp.** `max_files == 0` means *none*,
//!    and [`config::FileLimits::effective`] clamps it to 1. Consulting only the
//!    clamped value admits exactly one attachment from a source that asked for
//!    zero.

pub mod config;
pub mod data_uri;
pub mod error;
pub mod markers;
pub mod mime;
pub mod payload;
pub mod resolve;

#[cfg(test)]
mod test;

pub use config::{FileLimits, ImageLimits, ALLOWED_IMAGE_MIME_TYPES};
pub use error::{MultimodalError, Result};
pub use markers::{
    extract_image_placeholders_in_text, extract_ollama_image_payload, image_placeholder,
    parse_file_markers, parse_image_markers, rehydrate_placeholders_in_text,
    text_has_image_placeholders, FILE_MARKER_PREFIX, IMAGE_MARKER_PREFIX,
    IMAGE_PLACEHOLDER_PREFIX, IMAGE_STASH_REF,
};
pub use payload::{compose_multimodal_message, sha256_prefix, truncate_chars, FilePayload};
pub use resolve::{resolve_file, resolve_image, NoTextExtractor, TextExtractor};
