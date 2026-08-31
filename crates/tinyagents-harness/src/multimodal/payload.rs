//! Resolved file payloads and the rendered message the model sees.
//!
//! ## Why files never inline bytes
//!
//! Images inline as base64 `data:` URIs because that is the provider contract
//! for vision. Files do not, ever. An extractable format contributes its text;
//! a binary-only format contributes a header naming it and a content hash.
//! Handing a model raw binary bytes costs a great deal of context and tells it
//! nothing, so the two pipelines converge on markers and diverge on payloads.
//!
//! ## The rendered shapes
//!
//! ```text
//! [FILE-EXTRACTED: name="notes.md" size="4.1 KB" mime="text/markdown"]
//! …text…
//! [/FILE-EXTRACTED]
//!
//! [FILE-ATTACHED: name="sheet.xlsx" size="1.2 MB" mime="application/…sheet" sha256_prefix="a1b2…"]
//! ```
//!
//! Attribute values are escaped: names are user-supplied filenames, and one
//! containing a quote or a newline would otherwise break the header out of its
//! own delimiters.

use super::mime::is_extractable_text_mime;

/// Worst-case length budget reserved for the rendered truncation suffix.
///
/// The emitted suffix is `"\n[…truncated {N} chars]"` with a dynamic `N`. The
/// reservation uses the longest plausible value — `max_extracted_text_chars`
/// clamps to 200_000, so `N` has at most 6 digits — so the truncated payload
/// never overshoots the cap once the suffix is appended.
pub const TEXT_TRUNCATION_SUFFIX_BUDGET: &str = "\n[…truncated 999999 chars]";

/// A resolved `[FILE:…]` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilePayload {
    /// A format whose text was extracted and inlined.
    Extracted {
        /// Display name, as attached.
        name: String,
        /// Detected MIME type.
        mime: String,
        /// Size of the original bytes.
        size_bytes: usize,
        /// The extracted text, already truncated.
        text: String,
        /// How many characters truncation dropped; `0` when none.
        truncated_chars: usize,
    },
    /// A binary-only format, surfaced as metadata so the agent can mention it
    /// without seeing raw bytes.
    Reference {
        /// Display name, as attached.
        name: String,
        /// Detected MIME type.
        mime: String,
        /// Size of the original bytes.
        size_bytes: usize,
        /// First 16 hex characters of the SHA-256 digest.
        sha256_prefix: String,
    },
}

impl FilePayload {
    /// A content-less placeholder for a reference that was never read.
    ///
    /// Used for over-cap markers and for references that failed to resolve: the
    /// turn still tells the model something was attached, without the host
    /// having touched the underlying bytes.
    pub fn placeholder(name: impl Into<String>, sha256_prefix: impl Into<String>) -> Self {
        Self::Reference {
            name: name.into(),
            mime: "application/octet-stream".to_string(),
            size_bytes: 0,
            sha256_prefix: sha256_prefix.into(),
        }
    }

    /// Build the payload for already-resolved bytes, extracting text when the
    /// MIME type allows it.
    ///
    /// `extracted` is the text a host-supplied extractor produced for a format
    /// this crate cannot decode itself (PDF being the motivating case), or
    /// `None` when there was none — a failed extraction and an unsupported
    /// format both degrade to [`FilePayload::Reference`], which is what lets a
    /// damaged document pass through without failing the turn.
    pub fn from_resolved(
        bytes: &[u8],
        name: String,
        mime: String,
        extracted: Option<String>,
        max_extracted_text_chars: usize,
    ) -> Self {
        let size_bytes = bytes.len();

        let raw = if is_extractable_text_mime(&mime) {
            Some(decode_utf8_lossy(bytes))
        } else {
            extracted
        };

        if let Some(raw) = raw {
            let (text, truncated_chars) = truncate_chars(raw, max_extracted_text_chars);
            return Self::Extracted {
                name,
                mime,
                size_bytes,
                text,
                truncated_chars,
            };
        }

        Self::Reference {
            name,
            mime,
            size_bytes,
            sha256_prefix: sha256_prefix(bytes),
        }
    }
}

/// Compose the message the provider receives: the user's text, then every
/// image marker, then every file payload.
///
/// Images come back as `[IMAGE:<data-uri>]` markers rather than a structured
/// content array because the marker form is what the host's own provider
/// adapters already consume — this function renders the wire format, it does
/// not choose it.
pub fn compose_multimodal_message(
    text: &str,
    data_uris: &[String],
    file_payloads: &[FilePayload],
) -> String {
    let mut content = String::new();
    let trimmed = text.trim();

    if !trimmed.is_empty() {
        content.push_str(trimmed);
        content.push_str("\n\n");
    }

    for (index, data_uri) in data_uris.iter().enumerate() {
        if index > 0 {
            content.push('\n');
        }
        content.push_str(super::markers::IMAGE_MARKER_PREFIX);
        content.push_str(data_uri);
        content.push(']');
    }

    for payload in file_payloads {
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        if !content.is_empty() {
            content.push('\n');
        }
        match payload {
            FilePayload::Extracted {
                name,
                mime,
                size_bytes,
                text,
                truncated_chars,
            } => {
                content.push_str(&format!(
                    "[FILE-EXTRACTED: name=\"{}\" size=\"{}\" mime=\"{}\"]\n",
                    escape_attr(name),
                    format_size(*size_bytes),
                    mime
                ));
                content.push_str(text);
                if *truncated_chars > 0 {
                    content.push_str(&format!("\n[…truncated {} chars]", truncated_chars));
                }
                content.push_str("\n[/FILE-EXTRACTED]");
            }
            FilePayload::Reference {
                name,
                mime,
                size_bytes,
                sha256_prefix,
            } => {
                content.push_str(&format!(
                    "[FILE-ATTACHED: name=\"{}\" size=\"{}\" mime=\"{}\" sha256_prefix=\"{}\"]",
                    escape_attr(name),
                    format_size(*size_bytes),
                    mime,
                    sha256_prefix
                ));
            }
        }
    }

    content
}

/// Strip characters that would break the attribute-style serialization of a
/// [`FilePayload`] header. Names are user-supplied filenames, so they must not
/// be trusted to be quote-free.
pub fn escape_attr(value: &str) -> String {
    value.replace(['"', '\n', '\r'], "_")
}

/// Render a byte count for a payload header.
pub fn format_size(size_bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    if size_bytes >= MB {
        format!("{:.1} MB", size_bytes as f64 / MB as f64)
    } else if size_bytes >= KB {
        format!("{:.1} KB", size_bytes as f64 / KB as f64)
    } else {
        format!("{} B", size_bytes)
    }
}

/// Best-effort UTF-8 decode: strict decode wins, invalid sequences otherwise
/// become U+FFFD.
///
/// Lossy rather than failing because the alternative — degrading a text file to
/// a metadata reference over one bad byte — loses the whole document to save
/// one character.
pub fn decode_utf8_lossy(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

/// Truncate `text` to at most `max_chars` Unicode scalar values, leaving room
/// for the rendered truncation suffix.
///
/// The reservation uses [`TEXT_TRUNCATION_SUFFIX_BUDGET`] — the worst-case
/// rendered length — so `text + suffix` always stays inside `max_chars`
/// regardless of the actual dropped-digit count. Returns the text and the
/// number of characters dropped (`0` when none were).
pub fn truncate_chars(text: String, max_chars: usize) -> (String, usize) {
    let total = text.chars().count();
    if total <= max_chars {
        return (text, 0);
    }

    let suffix_chars = TEXT_TRUNCATION_SUFFIX_BUDGET.chars().count();
    let keep = max_chars.saturating_sub(suffix_chars);
    let truncated: String = text.chars().take(keep).collect();
    let dropped = total.saturating_sub(keep);
    (truncated, dropped)
}

/// First 16 hex characters of the SHA-256 digest of `bytes`.
///
/// A prefix rather than the full digest: it identifies an attachment across
/// turns and deduplicates the on-disk stash, and neither needs collision
/// resistance against an adversary who already supplied the bytes.
pub fn sha256_prefix(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|byte| format!("{:02x}", byte)).collect();
    hex.chars().take(16).collect()
}
