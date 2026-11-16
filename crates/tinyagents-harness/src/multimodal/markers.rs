//! The marker vocabulary, and the pure string transforms over it.
//!
//! Five tokens make up the wire format between a host's ingress and its
//! provider dispatch:
//!
//! | Token | Written by | Read by |
//! | --- | --- | --- |
//! | `[IMAGE:<ref>]` | the user / renderer | provider dispatch |
//! | `[FILE:<ref>]` | the user / renderer | ingress, then dispatch |
//! | `[Image: <name> #att:<id>]` | ingress, replacing an image marker | dispatch, rehydrating |
//! | `[FILE-EXTRACTED:…]` / `[FILE-ATTACHED:…]` | [`payload`](super::payload) | the model |
//! | `[ATTACHMENT:<form-urlencoded JSON>]` | ingress, after persisting an upload | transcript and dispatch |
//!
//! The placeholder is mixed-case on purpose: `[Image:` never collides with the
//! `[IMAGE:` parser, so a persisted placeholder cannot be mistaken for an
//! unresolved attachment and re-read.
//!
//! An `[ATTACHMENT:…]` reference ([`AttachmentRef`]) is the durable form of an
//! upload: compact metadata naming a workspace-relative path, never the bytes.
//! The codec here only encodes, decodes and rejects unsafe paths; where uploads
//! live and whether a path may be read stay host decisions.
//!
//! Everything here is a pure function over `&str` or over a single
//! [`Message`]. Transcript-level helpers — "how many markers are in the latest
//! user turn" — belong to the host, because only the host knows how its history
//! is shaped.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
};
use serde::{Deserialize, Serialize};
use tinyinference_llm::message::{ContentBlock, Message};

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

/// Prefix of a compact attachment reference. The body is the
/// `application/x-www-form-urlencoded` serialization of an [`AttachmentRef`]'s
/// JSON, so it never contains a `]` and survives every text hop verbatim.
pub const ATTACHMENT_MARKER_PREFIX: &str = "[ATTACHMENT:";

/// Fallback upload name when the original is empty or sanitizes to nothing.
const DEFAULT_ATTACHMENT_NAME: &str = "attachment";

/// Longest sanitized upload filename, in UTF-8 bytes.
const MAX_ATTACHMENT_FILENAME_BYTES: usize = 180;

/// Metadata carried by an `[ATTACHMENT:…]` reference.
///
/// `path` is relative to the host's acting workspace; [`attachment_segments`]
/// only yields references whose path passes [`is_safe_relative_path`]. Field
/// order is part of the wire format: it fixes the JSON the marker encodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttachmentRef {
    /// Workspace-relative path of the persisted original.
    pub path: String,
    /// Original, unsanitized display name.
    pub name: String,
    /// MIME type the host recorded for the original.
    pub mime: String,
    /// Size of the original in bytes.
    pub size_bytes: usize,
}

impl AttachmentRef {
    /// Render the `[ATTACHMENT:…]` marker for this reference.
    pub fn marker(&self) -> String {
        let json = serde_json::to_string(self).expect("attachment metadata serializes");
        format!(
            "{ATTACHMENT_MARKER_PREFIX}{}]",
            form_urlencoded::byte_serialize(json.as_bytes()).collect::<String>()
        )
    }

    /// The human-readable text a model sees in place of the marker.
    pub fn description(&self) -> String {
        format!(
            "[Uploaded file: {}; {}; {} bytes; workspace path: {}]",
            self.name, self.mime, self.size_bytes, self.path
        )
    }

    /// The reference `text` is the [`description`](Self::description) of,
    /// when it is exactly one. The name may hold anything, so the fixed
    /// fields are read from the right and the result must describe itself
    /// back byte for byte.
    pub fn from_description(text: &str) -> Option<Self> {
        let body = text.strip_prefix("[Uploaded file: ")?.strip_suffix(']')?;
        let (head, path) = body.rsplit_once("; workspace path: ")?;
        let (head, size) = head.strip_suffix(" bytes")?.rsplit_once("; ")?;
        let (name, mime) = head.rsplit_once("; ")?;
        let attachment = Self {
            path: path.into(),
            name: name.into(),
            mime: mime.into(),
            size_bytes: size.parse().ok()?,
        };
        (attachment.description() == text).then_some(attachment)
    }
}

/// One piece of text split by [`attachment_segments`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentSegment {
    /// Text between references, copied verbatim — including any marker that
    /// failed to decode or named an unsafe path.
    Text(String),
    /// A well-formed reference with a safe relative path.
    Attachment(AttachmentRef),
}

/// Split `text` into captions and `[ATTACHMENT:…]` references in source order.
///
/// Nothing is trimmed. A marker that does not decode to an [`AttachmentRef`],
/// or whose path fails [`is_safe_relative_path`], stays in the text verbatim;
/// an unterminated marker ends the scan and is kept with the remainder.
pub fn attachment_segments(text: &str) -> Vec<AttachmentSegment> {
    let mut rest = text;
    let mut parts = Vec::new();
    while let Some(start) = rest.find(ATTACHMENT_MARKER_PREFIX) {
        if start > 0 {
            parts.push(AttachmentSegment::Text(rest[..start].into()));
        }
        let tail = &rest[start + ATTACHMENT_MARKER_PREFIX.len()..];
        let Some(end) = tail.find(']') else {
            parts.push(AttachmentSegment::Text(rest[start..].into()));
            return parts;
        };
        let encoded = format!("value={}", &tail[..end]);
        let decoded = form_urlencoded::parse(encoded.as_bytes())
            .next()
            .map(|(_, value)| value.into_owned());
        match decoded.and_then(|json| serde_json::from_str::<AttachmentRef>(&json).ok()) {
            Some(file) if is_safe_relative_path(&file.path) => {
                parts.push(AttachmentSegment::Attachment(file))
            }
            _ => parts.push(AttachmentSegment::Text(
                rest[start..start + ATTACHMENT_MARKER_PREFIX.len() + end + 1].into(),
            )),
        }
        rest = &tail[end + 1..];
    }
    if !rest.is_empty() {
        parts.push(AttachmentSegment::Text(rest.into()));
    }
    parts
}

/// Strip every valid `[ATTACHMENT:…]` reference and return
/// `(remaining_text, refs_in_order)`.
///
/// Unlike [`parse_image_markers`], the remaining text is **not** trimmed: it is
/// the concatenation of every [`AttachmentSegment::Text`], byte for byte, so
/// ordinary prose is never reinterpreted as a path.
pub fn parse_attachment_markers(text: &str) -> (String, Vec<AttachmentRef>) {
    let mut clean = String::new();
    let mut files = Vec::new();
    for segment in attachment_segments(text) {
        match segment {
            AttachmentSegment::Text(text) => clean.push_str(&text),
            AttachmentSegment::Attachment(file) => files.push(file),
        }
    }
    (clean, files)
}

/// `true` when `path` is a non-empty relative path made only of normal
/// components.
///
/// Rejects NUL and every `\` (so Windows separators, drive paths such as
/// `C:\x` and UNC paths cannot slip through on Unix), a root or prefix, and
/// any `.` or `..` component. Judged per [`Path::components`] on the running
/// platform, which elides empty and interior `.` components (`a//b`, `a/./b`
/// are accepted as `a/b`). On Unix a forward-slash drive path such as `C:/x`
/// is an ordinary relative directory named `C:`; on Windows it is a prefix and
/// rejected. This is a shape check only: the host still authorizes the joined
/// path against its own policy.
pub fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains(['\0', '\\'])
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// A filesystem-safe leaf name for a persisted upload.
///
/// Keeps the last `/`- or `\`-separated component, replaces every character
/// other than alphanumerics, `.`, `-` and `_` with `_`, caps the result at 180
/// UTF-8 bytes on a character boundary, and falls back to `attachment` when
/// nothing usable (or only `.` / `..`) remains.
pub fn sanitize_attachment_filename(name: &str) -> String {
    let leaf = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(DEFAULT_ATTACHMENT_NAME);
    let mut sanitized = String::new();
    for c in leaf.chars() {
        let c = if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
            c
        } else {
            '_'
        };
        if sanitized.len() + c.len_utf8() > MAX_ATTACHMENT_FILENAME_BYTES {
            break;
        }
        sanitized.push(c);
    }
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        DEFAULT_ATTACHMENT_NAME.into()
    } else {
        sanitized
    }
}

/// Prefix the context a host added in front of a user turn's text without
/// flattening its typed media.
///
/// `enriched` is `original_text` with context prepended; only that prefix is
/// inserted (all of `enriched` when it does not end with `original_text`). It
/// joins a leading text block so a plain turn keeps one text block, and
/// otherwise becomes a new first block. Non-user messages pass through.
pub fn enrich_user_message(input: Message, original_text: &str, enriched: &str) -> Message {
    let Message::User(mut user) = input else {
        return input;
    };
    let prefix = enriched.strip_suffix(original_text).unwrap_or(enriched);
    if !prefix.is_empty() {
        if let Some(ContentBlock::Text(text)) = user.content.first_mut() {
            text.insert_str(0, prefix);
        } else {
            user.content.insert(0, ContentBlock::Text(prefix.into()));
        }
    }
    Message::User(user)
}

/// `legacy` followed by an `[IMAGE:<url>]` marker for each typed image block in
/// a user `input`, skipping duplicates — the image carrier a host forwards to a
/// delegated sub-agent without flattening the input.
pub fn image_markers_from_message(input: &Message, legacy: &[String]) -> Vec<String> {
    let mut images = legacy.to_vec();
    if let Message::User(user) = input {
        for block in &user.content {
            if let ContentBlock::Image(image) = block {
                let marker = format!("{IMAGE_MARKER_PREFIX}{}]", image.url);
                if !images.contains(&marker) {
                    images.push(marker);
                }
            }
        }
    }
    images
}

/// `true` when a delegation `prompt` selects no image itself, so the parent's
/// images should be forwarded automatically.
///
/// An image `[ATTACHMENT:…]` reference, an `[IMAGE:…]` marker or an
/// `[Image: … #att:<id>]` placeholder is an explicit selection and replaces
/// automatic forwarding.
pub fn should_forward_parent_images(prompt: &str) -> bool {
    parse_attachment_markers(prompt)
        .1
        .iter()
        .all(|file| !file.mime.starts_with("image/"))
        && parse_image_markers(prompt).1.is_empty()
        && extract_image_placeholders_in_text(prompt).is_empty()
}

#[cfg(test)]
#[path = "markers_tests.rs"]
mod tests;
