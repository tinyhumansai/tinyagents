//! Turning one marker reference into one payload.
//!
//! Three source shapes are accepted for both images and files, and they are
//! tried in a fixed order because only the first is self-describing:
//!
//! | Shape | Gate |
//! | --- | --- |
//! | `data:…;base64,…` | none — the renderer already owns these bytes |
//! | `http(s)://…` | `allow_remote_fetch`, off by default |
//! | anything else | treated as a local path |
//!
//! ## What this module deliberately does not decide
//!
//! **Which paths may be read.** A local reference is read as given. That is not
//! an oversight — a crate cannot know a host's filesystem threat model, and
//! guessing one would either block a desktop user from attaching their own
//! files or wave through a marker smuggled in from a chat channel. The host
//! decides, and the lever it has is
//! [`FileLimits::files_disabled`](super::config::FileLimits::files_disabled):
//! a turn whose text came from somewhere untrusted resolves no file markers at
//! all. See that method's docs.
//!
//! **How long an extraction may take.** [`TextExtractor`] has no deadline in
//! its signature. The cost of parsing a document is set by the document, and
//! only the host knows how long an attachment is worth waiting for — so the
//! host wraps its own implementation in whatever timeout it wants, and a
//! blown deadline arrives here as an ordinary `Err`.
//!
//! Both failures degrade rather than propagate: a file that cannot be extracted
//! becomes a [`FilePayload::Reference`], so a damaged PDF costs the model its
//! text and not the turn.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use reqwest::Client;

use super::config::{FileLimits, ImageLimits};
use super::data_uri::{data_uri_param, encode_data_uri, gunzip, parse_data_uri};
use super::error::{MultimodalError, Result};
use super::mime::{detect_file_mime, detect_image_mime, is_allowed_image_mime};
use super::payload::FilePayload;
use super::types::{ResolvedAttachment, UnknownMimePolicy};

/// Host-supplied text extraction for formats this crate cannot decode itself.
///
/// Implemented for exactly one reason: PDF text extraction needs a parser, a
/// parser is a large dependency with its own failure modes, and which one (if
/// any) a host is willing to carry is a host decision. A host with no extractor
/// passes [`NoTextExtractor`] and every such file degrades to a metadata
/// reference.
#[async_trait]
pub trait TextExtractor: Send + Sync {
    /// Whether this extractor has anything to say about `mime`.
    ///
    /// Required rather than defaulted, and consulted before every call.
    /// Extraction can be expensive well before it fails — a host that runs its
    /// parser out-of-process pays a round trip and a copy of the bytes — so
    /// "offer it everything and let it refuse" would charge every `.zip` and
    /// `.xlsx` attachment for a refusal that was knowable from the MIME type.
    fn handles(&self, mime: &str) -> bool;

    /// Extract text from `bytes` of type `mime`, or explain why not.
    ///
    /// Called only when [`TextExtractor::handles`] returned `true`.
    ///
    /// `Err` is a plain reason string because the caller never branches on it —
    /// every failure takes the same degrade-to-reference path, and the string
    /// exists to be logged.
    async fn extract(&self, mime: &str, bytes: &[u8]) -> std::result::Result<String, String>;
}

/// A [`TextExtractor`] that extracts nothing.
///
/// The correct choice for a host that carries no document parser: every
/// non-plaintext format surfaces as a [`FilePayload::Reference`], which is the
/// same outcome as a parser that failed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTextExtractor;

#[async_trait]
impl TextExtractor for NoTextExtractor {
    fn handles(&self, _mime: &str) -> bool {
        false
    }

    async fn extract(&self, mime: &str, _bytes: &[u8]) -> std::result::Result<String, String> {
        Err(format!("no text extractor is configured for '{mime}'"))
    }
}

// ── Images ───────────────────────────────────────────────────────────────

/// Resolve one `[IMAGE:…]` reference into a canonical `data:` URI.
///
/// The return value is always re-encoded rather than passed through, so a
/// caller downstream can rely on the MIME type in the header matching the
/// bytes — which is what makes the allowlist check meaningful rather than
/// advisory.
pub async fn resolve_image(
    source: &str,
    limits: &ImageLimits,
    max_bytes: usize,
    remote_client: &Client,
) -> Result<String> {
    if source.starts_with("data:") {
        return resolve_image_data_uri(source, max_bytes);
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        if !limits.allow_remote_fetch {
            return Err(MultimodalError::RemoteFetchDisabled {
                input: source.to_string(),
            });
        }

        return resolve_remote_image(source, max_bytes, remote_client).await;
    }

    resolve_local_image(source, max_bytes).await
}

/// Resolve a `data:` source: decompresses a gzip-wrapped payload when present,
/// then validates MIME and size before re-encoding.
fn resolve_image_data_uri(source: &str, max_bytes: usize) -> Result<String> {
    let parsed = parse_data_uri(source).map_err(|reason| MultimodalError::InvalidMarker {
        input: source.to_string(),
        reason,
    })?;

    let (mime, decoded) = if parsed.mime == "application/gzip" {
        let original_mime = data_uri_param(&parsed.params, "original_mime").ok_or_else(|| {
            MultimodalError::InvalidMarker {
                input: source.to_string(),
                reason: "compressed image data URI missing original_mime parameter".to_string(),
            }
        })?;
        let bytes =
            gunzip(&parsed.bytes, max_bytes).map_err(|reason| MultimodalError::InvalidMarker {
                input: source.to_string(),
                reason,
            })?;
        (original_mime.to_ascii_lowercase(), bytes)
    } else {
        (parsed.mime, parsed.bytes)
    };

    check_image_mime(source, &mime)?;
    check_image_size(source, decoded.len(), max_bytes)?;

    Ok(encode_data_uri(&mime, &decoded))
}

/// Fetches an `http(s)` image, checking size against both the `Content-Length`
/// header and the measured body before detecting MIME and re-encoding.
async fn resolve_remote_image(
    source: &str,
    max_bytes: usize,
    remote_client: &Client,
) -> Result<String> {
    let validated_url = tinytools_std::url_guard::validate_url(source, &[]).map_err(|error| {
        MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
    })?;
    let response = remote_client
        .get(validated_url)
        .send()
        .await
        .map_err(|error| MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    let status = response.status();
    if !status.is_success() {
        return Err(MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: format!("HTTP {status}"),
        });
    }

    // Checked twice on purpose: `Content-Length` lets an over-cap response be
    // refused before its body is read, and the measured length catches a server
    // that lied or sent none.
    if let Some(content_length) = response.content_length() {
        check_image_size(source, content_length as usize, max_bytes)?;
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);

    let bytes = response
        .bytes()
        .await
        .map_err(|error| MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    check_image_size(source, bytes.len(), max_bytes)?;

    let mime =
        detect_image_mime(None, bytes.as_ref(), content_type.as_deref()).ok_or_else(|| {
            MultimodalError::UnsupportedMime {
                input: source.to_string(),
                mime: "unknown".to_string(),
            }
        })?;

    check_image_mime(source, &mime)?;

    Ok(encode_data_uri(&mime, bytes.as_ref()))
}

/// Reads a local image path, checking size (against metadata, then the
/// measured read) before detecting MIME and re-encoding.
async fn resolve_local_image(source: &str, max_bytes: usize) -> Result<String> {
    let path = Path::new(source);
    if !path.exists() || !path.is_file() {
        return Err(MultimodalError::ImageSourceNotFound {
            input: source.to_string(),
        });
    }

    let metadata =
        tokio::fs::metadata(path)
            .await
            .map_err(|error| MultimodalError::LocalReadFailed {
                input: source.to_string(),
                reason: error.to_string(),
            })?;

    check_image_size(source, metadata.len() as usize, max_bytes)?;

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| MultimodalError::LocalReadFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    check_image_size(source, bytes.len(), max_bytes)?;

    let mime = detect_image_mime(Some(path), &bytes, None).ok_or_else(|| {
        MultimodalError::UnsupportedMime {
            input: source.to_string(),
            mime: "unknown".to_string(),
        }
    })?;

    check_image_mime(source, &mime)?;

    Ok(encode_data_uri(&mime, &bytes))
}

/// Rejects `size_bytes` over `max_bytes` as [`MultimodalError::ImageTooLarge`].
fn check_image_size(source: &str, size_bytes: usize, max_bytes: usize) -> Result<()> {
    if size_bytes > max_bytes {
        return Err(MultimodalError::ImageTooLarge {
            input: source.to_string(),
            size_bytes,
            max_bytes,
        });
    }
    Ok(())
}

/// Rejects `mime` not on the image allowlist as [`MultimodalError::UnsupportedMime`].
fn check_image_mime(source: &str, mime: &str) -> Result<()> {
    if is_allowed_image_mime(mime) {
        return Ok(());
    }
    Err(MultimodalError::UnsupportedMime {
        input: source.to_string(),
        mime: mime.to_string(),
    })
}

// ── Files ────────────────────────────────────────────────────────────────

/// Resolve one `[FILE:…]` reference into a [`FilePayload`].
///
/// The MIME allowlist is checked **after** the bytes are in hand, because the
/// type is detected from them; an over-cap or unreadable file therefore fails
/// on that first, which is the more useful message.
pub async fn resolve_file(
    source: &str,
    limits: &FileLimits,
    max_bytes: usize,
    max_extracted_text_chars: usize,
    remote_client: &Client,
    extractor: &dyn TextExtractor,
) -> Result<FilePayload> {
    let resolved = resolve_attachment(
        source,
        limits,
        max_bytes,
        remote_client,
        UnknownMimePolicy::Reject,
    )
    .await?;
    build_file_payload(
        source,
        resolved.bytes,
        resolved.name,
        resolved.mime,
        limits,
        max_extracted_text_chars,
        extractor,
    )
    .await
}

/// Resolve bytes and metadata without extracting text or writing to disk.
///
/// The host must authorize local paths before calling. Unknown MIME acceptance
/// is explicit and does not affect legacy [`resolve_file`] callers. Only a gzip
/// data URI with `original_mime` is a transport envelope: a gzip attachment
/// without that parameter remains compressed for archive inspection.
pub async fn resolve_attachment(
    source: &str,
    limits: &FileLimits,
    max_bytes: usize,
    remote_client: &Client,
    unknown_mime: UnknownMimePolicy,
) -> Result<ResolvedAttachment> {
    if limits.files_disabled() {
        return Err(MultimodalError::TooManyFiles {
            max_files: 0,
            found: 1,
        });
    }
    let max_bytes = max_bytes.min(limits.max_file_bytes());
    let (bytes, name, mime) = if source.starts_with("data:") {
        resolve_file_data_uri(source, max_bytes, unknown_mime)?
    } else {
        let (bytes, name, header) =
            if source.starts_with("http://") || source.starts_with("https://") {
                if !limits.allow_remote_fetch {
                    return Err(MultimodalError::RemoteFileFetchDisabled {
                        input: source.to_string(),
                    });
                }
                fetch_remote_file(source, max_bytes, remote_client).await?
            } else {
                let (bytes, _path, name) = read_local_file(source, max_bytes).await?;
                (bytes, name, None)
            };
        let detected = if unknown_mime == UnknownMimePolicy::Accept {
            super::mime::detect_attachment_mime(Path::new(&name), &bytes, header.as_deref())
        } else {
            detect_file_mime(Some(Path::new(&name)), &bytes, header.as_deref())
        };
        if detected.is_none() && unknown_mime == UnknownMimePolicy::Reject {
            return Err(MultimodalError::UnsupportedFileMime {
                input: source.to_string(),
                mime: "unknown".to_string(),
                supported: limits.supported_rendered(),
            });
        }
        let mime = detected
            .or_else(|| {
                header
                    .as_deref()
                    .and_then(super::mime::normalize_content_type)
            })
            .unwrap_or_else(|| "application/octet-stream".to_string());
        (bytes, name, mime)
    };
    if !limits.is_mime_allowed(&mime) && unknown_mime == UnknownMimePolicy::Reject {
        return Err(MultimodalError::UnsupportedFileMime {
            input: source.to_string(),
            mime,
            supported: limits.supported_rendered(),
        });
    }
    Ok(ResolvedAttachment {
        size_bytes: bytes.len(),
        bytes,
        name,
        mime,
    })
}

/// Resolve a `data:` file source: decompresses a gzip-wrapped payload when
/// present, extracts the `name` parameter, and checks size. MIME allowlisting
/// happens later in [`build_file_payload`].
fn resolve_file_data_uri(
    source: &str,
    max_bytes: usize,
    unknown_mime: UnknownMimePolicy,
) -> Result<(Vec<u8>, String, String)> {
    // Reject payloads whose minimum decoded size is over cap before allocating.
    if let Some((header, encoded)) = source.split_once(',') {
        let minimum_size = if header
            .split(';')
            .any(|part| part.trim().eq_ignore_ascii_case("base64"))
        {
            (encoded.len() / 4).saturating_mul(3).saturating_sub(2)
        } else {
            encoded.len() / 3
        };
        check_file_size(source, minimum_size, max_bytes)?;
    }
    let parsed = parse_data_uri(source).map_err(|reason| MultimodalError::InvalidFileMarker {
        input: source.to_string(),
        reason,
    })?;
    let name = data_uri_param(&parsed.params, "name").unwrap_or_else(|| "attachment".to_string());

    check_file_size(source, parsed.bytes.len(), max_bytes)?;
    let (mime, bytes) = if parsed.mime == "application/gzip"
        && let Some(original_mime) = data_uri_param(&parsed.params, "original_mime")
    {
        let bytes = gunzip(&parsed.bytes, max_bytes).map_err(|reason| {
            MultimodalError::InvalidFileMarker {
                input: source.to_string(),
                reason,
            }
        })?;
        (original_mime.to_ascii_lowercase(), bytes)
    } else {
        (parsed.mime, parsed.bytes)
    };

    check_file_size(source, bytes.len(), max_bytes)?;

    let mime = if mime.is_empty() {
        // RFC 2397 defaults an omitted media type to text/plain.
        "text/plain".to_string()
    } else if mime == "application/zip" || mime == "application/octet-stream" {
        if unknown_mime == UnknownMimePolicy::Accept {
            super::mime::detect_attachment_mime(Path::new(&name), &bytes, Some(&mime))
        } else {
            detect_file_mime(Some(Path::new(&name)), &bytes, Some(&mime))
        }
        .unwrap_or(mime)
    } else {
        mime
    };
    Ok((bytes, name, mime))
}

/// Enforces the MIME allowlist, then extracts text (via `extractor`) for a
/// non-plaintext format that offers to handle it, degrading to a metadata
/// reference on refusal, and builds the resulting [`FilePayload`].
async fn build_file_payload(
    source: &str,
    bytes: Vec<u8>,
    name: String,
    mime: String,
    limits: &FileLimits,
    max_extracted_text_chars: usize,
    extractor: &dyn TextExtractor,
) -> Result<FilePayload> {
    if !limits.is_mime_allowed(&mime) {
        return Err(MultimodalError::UnsupportedFileMime {
            input: source.to_string(),
            mime: mime.clone(),
            supported: limits.supported_rendered(),
        });
    }

    tracing::debug!(
        target: "multimodal",
        file = %name,
        mime = %mime,
        size_bytes = bytes.len(),
        "[multimodal::files] resolved file ref"
    );

    // Plain-text formats decode in `FilePayload::from_resolved`; a format the
    // host's extractor claims is offered to it, and a refusal degrades the file
    // to a metadata reference rather than failing the turn. Everything else —
    // the binary-only formats — goes straight to a reference without the
    // extractor ever seeing the bytes.
    let extracted = if super::mime::is_extractable_text_mime(&mime) || !extractor.handles(&mime) {
        None
    } else {
        match extractor.extract(&mime, &bytes).await {
            Ok(text) => Some(text),
            Err(reason) => {
                tracing::warn!(
                    target: "multimodal",
                    file = %name,
                    mime = %mime,
                    reason = %reason,
                    "[multimodal::files] text extraction failed, degrading to reference"
                );
                None
            }
        }
    };

    let payload =
        FilePayload::from_resolved(&bytes, name, mime, extracted, max_extracted_text_chars);

    if let FilePayload::Extracted {
        name,
        truncated_chars,
        ..
    } = &payload
        && *truncated_chars > 0
    {
        tracing::info!(
            target: "multimodal",
            file = %name,
            truncated_chars,
            max_extracted_text_chars,
            "[multimodal::files] truncated extracted text"
        );
    }

    Ok(payload)
}

/// Reads a local file path, checking size (against metadata, then the
/// measured read) and returning its bytes, path, and file-name.
async fn read_local_file(source: &str, max_bytes: usize) -> Result<(Vec<u8>, PathBuf, String)> {
    let path = Path::new(source).to_path_buf();
    if !path.exists() || !path.is_file() {
        return Err(MultimodalError::FileSourceNotFound {
            input: source.to_string(),
        });
    }

    let metadata =
        tokio::fs::metadata(&path)
            .await
            .map_err(|error| MultimodalError::LocalFileReadFailed {
                input: source.to_string(),
                reason: error.to_string(),
            })?;

    check_file_size(source, metadata.len() as usize, max_bytes)?;

    use tokio::io::AsyncReadExt;
    let file = tokio::fs::File::open(&path).await.map_err(|error| {
        MultimodalError::LocalFileReadFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
    })?;
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| MultimodalError::LocalFileReadFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    check_file_size(source, bytes.len(), max_bytes)?;

    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| source.to_string());

    Ok((bytes, path, name))
}

/// Fetches an `http(s)` file, checking size against both the `Content-Length`
/// header and the measured body, and deriving a display name from the URL's
/// last path segment (not `Content-Disposition`, which is attacker-controlled
/// on a fetched URL).
async fn fetch_remote_file(
    source: &str,
    max_bytes: usize,
    remote_client: &Client,
) -> Result<(Vec<u8>, String, Option<String>)> {
    let validated_url = tinytools_std::url_guard::validate_url(source, &[]).map_err(|error| {
        MultimodalError::RemoteFileFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
    })?;
    let response = remote_client
        .get(validated_url)
        .send()
        .await
        .map_err(|error| MultimodalError::RemoteFileFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    let status = response.status();
    if !status.is_success() {
        return Err(MultimodalError::RemoteFileFetchFailed {
            input: source.to_string(),
            reason: format!("HTTP {status}"),
        });
    }

    if let Some(content_length) = response.content_length() {
        check_file_size(source, content_length as usize, max_bytes)?;
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);

    let mut response = response;
    let mut bytes = Vec::new();
    while let Some(chunk) =
        response
            .chunk()
            .await
            .map_err(|error| MultimodalError::RemoteFileFetchFailed {
                input: source.to_string(),
                reason: error.to_string(),
            })?
    {
        check_file_size(source, bytes.len().saturating_add(chunk.len()), max_bytes)?;
        bytes.extend_from_slice(&chunk);
    }

    // The last path segment, not a `Content-Disposition` filename: the header
    // is attacker-controlled on a fetched URL and has its own escaping rules,
    // and the payload header escapes whatever lands here anyway.
    let name = reqwest::Url::parse(source)
        .ok()
        .and_then(|url| {
            url.path_segments()
                .and_then(|mut segments| segments.next_back())
                .map(str::to_owned)
        })
        .filter(|segment| !segment.is_empty())
        .and_then(|segment| super::data_uri::percent_decode(&segment))
        .unwrap_or_else(|| "attachment".to_string());

    Ok((bytes.to_vec(), name, content_type))
}

/// Rejects `size_bytes` over `max_bytes` as [`MultimodalError::FileTooLarge`].
fn check_file_size(source: &str, size_bytes: usize, max_bytes: usize) -> Result<()> {
    if size_bytes > max_bytes {
        return Err(MultimodalError::FileTooLarge {
            input: source.to_string(),
            size_bytes,
            max_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
