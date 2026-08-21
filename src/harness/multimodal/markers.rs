//! The marker vocabulary, and the pure string transforms over it.
//!
//! Four tokens make up the wire format between a host's ingress and its
//! provider dispatch:
//!
//! | Token | Written by | Read by |
//! | --- | --- | --- |
//! | `[IMAGE:<ref>]` | the user / renderer | provider dispatch |
//! | `[FILE:<ref>]` | the user / renderer | ingress, then dispatch |
//! | `[Image: <name> #att:<id>]` | ingress, replacing an image marker | dispatch, rehydrating |
//! | `[FILE-EXTRACTED:…]` / `[FILE-ATTACHED:…]` | [`payload`](super::payload) | the model |
//!
//! The placeholder is mixed-case on purpose: `[Image:` never collides with the
//! `[IMAGE:` parser, so a persisted placeholder cannot be mistaken for an
//! unresolved attachment and re-read.
//!
//! Everything here is a pure function over `&str`. Message-level helpers — "how
//! many markers are in the latest user turn" — belong to the host, because only
//! the host knows what its message type is.

use std::collections::HashMap;
use std::path::PathBuf;

use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
    Engine as _,
};

/// Prefix of an inline image marker.
pub const IMAGE_MARKER_PREFIX: &str = "[IMAGE:";

/// Prefix of an inline file marker. Resolution mirrors images: local paths,
/// `http(s)` URLs gated by
/// [`FileLimits::allow_remote_fetch`](super::config::FileLimits::allow_remote_fetch),
/// and renderer-owned `data:` URIs.
pub const FILE_MARKER_PREFIX: &str = "[FILE:";

/// Prefix of the persisted image sidecar placeholder. Mixed-case so it never
/// collides with [`IMAGE_MARKER_PREFIX`].
pub const IMAGE_PLACEHOLDER_PREFIX: &str = "[Image:";

/// Separator before the stash id inside a placeholder.
pub const IMAGE_STASH_REF: &str = "#att:";

/// Strip every `[IMAGE:…]` marker and return `(cleaned_text, refs_in_order)`.
///
/// An empty marker (`[IMAGE:]`) is left in the text rather than yielding an
/// empty reference — there is nothing to resolve and dropping it would silently
/// edit the user's message.
pub fn parse_image_markers(content: &str) -> (String, Vec<String>) {
    parse_markers(content, IMAGE_MARKER_PREFIX)
}

/// Strip every `[FILE:…]` marker and return `(cleaned_text, refs_in_order)`.
/// Mirrors [`parse_image_markers`] so the two pipelines stay symmetrical.
pub fn parse_file_markers(content: &str) -> (String, Vec<String>) {
    parse_markers(content, FILE_MARKER_PREFIX)
}

/// Shared body of the two marker parsers.
///
/// An unterminated marker (no `]` before end of input) is copied through
/// verbatim and ends the scan: the remainder cannot contain a well-formed
/// marker that started earlier, and rewriting a half-typed marker would corrupt
/// text the user is still editing.
fn parse_markers(content: &str, prefix: &str) -> (String, Vec<String>) {
    let mut refs = Vec::new();
    let mut cleaned = String::with_capacity(content.len());
    let mut cursor = 0usize;

    while let Some(rel_start) = content[cursor..].find(prefix) {
        let start = cursor + rel_start;
        cleaned.push_str(&content[cursor..start]);

        let marker_start = start + prefix.len();
        let Some(rel_end) = content[marker_start..].find(']') else {
            cleaned.push_str(&content[start..]);
            cursor = content.len();
            break;
        };

        let end = marker_start + rel_end;
        let candidate = content[marker_start..end].trim();

        if candidate.is_empty() {
            cleaned.push_str(&content[start..=end]);
        } else {
            refs.push(candidate.to_string());
        }

        cursor = end + 1;
    }

    if cursor < content.len() {
        cleaned.push_str(&content[cursor..]);
    }

    (cleaned.trim().to_string(), refs)
}

/// Longest bare reference still treated as a possible filesystem path.
///
/// 64 base64 characters decode to 48 bytes. No real image comes in under that
/// (a minimal GIF is ~35 bytes, the smallest valid PNG ~67), so a path-prefixed
/// reference this short cannot be image data.
const MAX_PATH_SHAPED_REF_LEN: usize = 64;

/// `true` when a **bare** reference is shaped like an absolute filesystem path
/// rather than base64 image data.
///
/// Shape alone cannot decide this: `/` is in the standard base64 alphabet, and
/// a bare base64 JPEG legitimately begins `/9j/`. The length bound is what
/// separates the two — a real payload carrying that prefix runs to thousands of
/// characters, while `/tmp/foo` does not.
///
/// Relative paths (`./x`, `~/x`, `../x`) need no check here: `.` and `~` are
/// outside the base64 alphabet, so the decode below already rejects them, as it
/// does every Windows path (`:` and `\` are likewise outside it).
///
/// **Known residual ambiguity:** a relative path built only from base64
/// characters that also decodes cleanly — `photos/cats1` — is genuinely
/// indistinguishable from a short payload and is still accepted. Tightening
/// that would start rejecting real base64, so it is left alone deliberately.
/// The window is narrower than it looks: the length must not be `4n + 1`, and a
/// partial trailing group must carry zero spare bits, which an arbitrary word
/// rarely does.
fn looks_like_absolute_path(payload: &str) -> bool {
    payload.starts_with('/') && payload.len() < MAX_PATH_SHAPED_REF_LEN
}

/// The base64 payload an Ollama-style `images` array expects, or `None` when
/// `image_ref` does not carry one.
///
/// Accepts a `data:` URI (payload after the comma) or a bare base64 string.
///
/// **Both forms are validated as base64.** Returning a non-`data:` string
/// verbatim would forward a filesystem path to the provider as if it were image
/// bytes, and the resulting `illegal base64 data at input byte 19` names
/// neither the parameter nor the path. `None` lets the caller say which
/// reference it could not use.
pub fn extract_ollama_image_payload(image_ref: &str) -> Option<String> {
    let is_data_uri = image_ref.starts_with("data:");
    let payload = if is_data_uri {
        let comma_idx = image_ref.find(',')?;
        image_ref.split_at(comma_idx + 1).1.trim()
    } else {
        image_ref.trim()
    };
    if payload.is_empty() {
        return None;
    }
    if !is_data_uri && looks_like_absolute_path(payload) {
        tracing::debug!(
            "[multimodal] image reference is shaped like a filesystem path, not image bytes"
        );
        return None;
    }
    // Decode to validate only — the encoded form is what goes on the wire, and
    // re-encoding would just burn a copy of a multi-MB image. Accept both
    // padded and unpadded alphabets: real data URIs are padded, but some
    // producers omit the `=`, and rejecting those would be a new regression
    // rather than the fix this is.
    //
    // Length picks the engine rather than trying both: a complete group
    // (`len % 4 == 0`) is exactly what `STANDARD` accepts, with or without
    // trailing `=`, and only a partial group needs `STANDARD_NO_PAD`. Trying
    // both in sequence decoded a multi-MB image twice for every unpadded
    // payload.
    let is_base64 = if payload.len() % 4 == 0 {
        STANDARD.decode(payload).is_ok()
    } else {
        STANDARD_NO_PAD.decode(payload).is_ok()
    };
    if !is_base64 {
        tracing::debug!(
            "[multimodal] image reference is not base64 (a filesystem path is not accepted here)"
        );
        return None;
    }
    Some(payload.to_string())
}

/// Render the persisted sidecar placeholder for a stashed image `id`.
pub fn image_placeholder(id: &str) -> String {
    format!("{IMAGE_PLACEHOLDER_PREFIX} image {IMAGE_STASH_REF}{id}]")
}

/// `true` when `text` carries at least one resolvable sidecar placeholder.
///
/// Both halves are required: `[Image:` alone is also how an unresolvable
/// placeholder (`[Image: (could not be processed)]`) renders, and that one has
/// no id to look up.
pub fn text_has_image_placeholders(text: &str) -> bool {
    text.contains(IMAGE_PLACEHOLDER_PREFIX) && text.contains(IMAGE_STASH_REF)
}

/// Extract the `[Image: … #att:<id>]` placeholder tokens from `text`, in order.
///
/// Used to forward a user's attached images into a delegated vision sub-agent's
/// prompt so its turn rehydrates them — the parent, on a non-vision tier, keeps
/// the placeholder as text and never sees the image.
pub fn extract_image_placeholders_in_text(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find(IMAGE_PLACEHOLDER_PREFIX) {
        let start = cursor + rel;
        let Some(rel_end) = text[start..].find(']') else {
            break;
        };
        let end = start + rel_end + 1;
        let token = &text[start..end];
        if token.contains(IMAGE_STASH_REF) {
            out.push(token.to_string());
        }
        cursor = end;
    }
    out
}

/// Replace each `[Image: <name> #att:<id>]` placeholder in `text` with
/// `[IMAGE:<path>]` when the id resolves in `index`; leave it verbatim
/// otherwise.
///
/// An unresolved id is *not* an error. The attachment may have been swept, or
/// written by a different workspace; keeping the human-readable placeholder
/// tells the model an image was attached without inventing a path that does not
/// exist.
///
/// The index is supplied by the host because where attachments live — and
/// whether this process may read them — is a host decision.
pub fn rehydrate_placeholders_in_text(text: &str, index: &HashMap<String, PathBuf>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find(IMAGE_PLACEHOLDER_PREFIX) {
        let start = cursor + rel;
        out.push_str(&text[cursor..start]);
        let Some(rel_end) = text[start..].find(']') else {
            out.push_str(&text[start..]);
            cursor = text.len();
            break;
        };
        let end = start + rel_end + 1;
        let inner = &text[start..end];
        let replaced = inner.find(IMAGE_STASH_REF).and_then(|ai| {
            let id = inner[ai + IMAGE_STASH_REF.len()..]
                .trim_end_matches(']')
                .trim();
            index
                .get(id)
                .map(|path| format!("{IMAGE_MARKER_PREFIX}{}]", path.display()))
        });
        out.push_str(replaced.as_deref().unwrap_or(inner));
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}
