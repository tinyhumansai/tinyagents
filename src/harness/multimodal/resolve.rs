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

async fn resolve_remote_image(
    source: &str,
    max_bytes: usize,
    remote_client: &Client,
) -> Result<String> {
    let response = remote_client.get(source).send().await.map_err(|error| {
        MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
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
    if source.starts_with("data:") {
        let (bytes, name, mime) = resolve_file_data_uri(source, max_bytes)?;
        return build_file_payload(
            source,
            bytes,
            name,
            mime,
            limits,
            max_extracted_text_chars,
            extractor,
        )
        .await;
    }

    let (bytes, path_hint, name, header_content_type) =
        if source.starts_with("http://") || source.starts_with("https://") {
            if !limits.allow_remote_fetch {
                return Err(MultimodalError::RemoteFileFetchDisabled {
                    input: source.to_string(),
                });
            }
            let (bytes, name, content_type) =
                fetch_remote_file(source, max_bytes, remote_client).await?;
            (bytes, None, name, content_type)
        } else {
            let (bytes, path, name) = read_local_file(source, max_bytes).await?;
            (bytes, Some(path), name, None)
        };

    let mime = detect_file_mime(path_hint.as_deref(), &bytes, header_content_type.as_deref())
        .ok_or_else(|| MultimodalError::UnsupportedFileMime {
            input: source.to_string(),
            mime: "unknown".to_string(),
            supported: limits.supported_rendered(),
        })?;

    build_file_payload(
        source,
        bytes,
        name,
        mime,
        limits,
        max_extracted_text_chars,
        extractor,
    )
    .await
}

fn resolve_file_data_uri(source: &str, max_bytes: usize) -> Result<(Vec<u8>, String, String)> {
    let parsed = parse_data_uri(source).map_err(|reason| MultimodalError::InvalidFileMarker {
        input: source.to_string(),
        reason,
    })?;
    let name = data_uri_param(&parsed.params, "name").unwrap_or_else(|| "attachment".to_string());

    let (mime, bytes) = if parsed.mime == "application/gzip" {
        let original_mime = data_uri_param(&parsed.params, "original_mime").ok_or_else(|| {
            MultimodalError::InvalidFileMarker {
                input: source.to_string(),
                reason: "compressed file data URI missing original_mime parameter".to_string(),
            }
        })?;
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

    Ok((bytes, name, mime))
}

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

    let bytes =
        tokio::fs::read(&path)
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

async fn fetch_remote_file(
    source: &str,
    max_bytes: usize,
    remote_client: &Client,
) -> Result<(Vec<u8>, String, Option<String>)> {
    let response = remote_client.get(source).send().await.map_err(|error| {
        MultimodalError::RemoteFileFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
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

    let bytes = response
        .bytes()
        .await
        .map_err(|error| MultimodalError::RemoteFileFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    check_file_size(source, bytes.len(), max_bytes)?;

    // The last path segment, not a `Content-Disposition` filename: the header
    // is attacker-controlled on a fetched URL and has its own escaping rules,
    // and the payload header escapes whatever lands here anyway.
    let name = source
        .rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(source)
        .to_string();

    Ok((bytes.to_vec(), name, content_type))
}

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
