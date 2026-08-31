//! Tests for attachment resolution.
//!
//! The bias here is toward the decisions that are easy to get subtly wrong and
//! that no type checks: the sentinel-before-clamp ordering, the two different
//! header-vs-extension precedences, the truncation budget, and the cases where
//! a malformed marker must be left alone rather than rewritten.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};

use super::config::{FileLimits, ImageLimits};
use super::data_uri::{data_uri_param, gunzip, parse_data_uri, percent_decode};
use super::error::MultimodalError;
use super::markers::{
    extract_image_placeholders_in_text, extract_ollama_image_payload, image_placeholder,
    parse_file_markers, parse_image_markers, rehydrate_placeholders_in_text,
    text_has_image_placeholders,
};
use super::mime::{
    detect_file_mime, detect_image_mime, file_mime_from_extension, file_mime_from_magic,
    image_ext_from_mime, image_mime_from_magic, is_extractable_text_mime, looks_like_utf8_text,
};
use super::payload::{
    FilePayload, compose_multimodal_message, escape_attr, format_size, truncate_chars,
};
use super::resolve::{NoTextExtractor, TextExtractor, resolve_file, resolve_image};

/// A 1×1 PNG. Small enough to inline, real enough to sniff.
const PNG_BYTES: &[u8] = &[
    0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0x00, 0x00, 0x00, 0x0d, b'I', b'H', b'D',
    b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15,
    0xc4, 0x89,
];

fn png_data_uri() -> String {
    format!("data:image/png;base64,{}", STANDARD.encode(PNG_BYTES))
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

// ── markers ──────────────────────────────────────────────────────────────

#[test]
fn parse_image_markers_extracts_every_reference_in_order() {
    let (text, refs) = parse_image_markers("before [IMAGE:/a.png] middle [IMAGE:/b.png] after");
    assert_eq!(refs, vec!["/a.png".to_string(), "/b.png".to_string()]);
    assert_eq!(text, "before  middle  after");
}

/// An empty marker has nothing to resolve, so removing it would be a silent
/// edit of the user's own text.
#[test]
fn parse_image_markers_leaves_an_empty_marker_in_the_text() {
    let (text, refs) = parse_image_markers("look: [IMAGE:]");
    assert!(refs.is_empty());
    assert_eq!(text, "look: [IMAGE:]");
}

/// A half-typed marker is text someone is still writing. Rewriting it would
/// corrupt the message mid-edit.
#[test]
fn parse_image_markers_passes_an_unterminated_marker_through() {
    let (text, refs) = parse_image_markers("oops [IMAGE:/a.png");
    assert!(refs.is_empty());
    assert_eq!(text, "oops [IMAGE:/a.png");
}

#[test]
fn the_two_marker_parsers_do_not_consume_each_others_markers() {
    let input = "[IMAGE:/a.png] and [FILE:/b.pdf]";

    let (after_images, image_refs) = parse_image_markers(input);
    assert_eq!(image_refs, vec!["/a.png".to_string()]);
    assert!(after_images.contains("[FILE:/b.pdf]"));

    let (cleaned, file_refs) = parse_file_markers(&after_images);
    assert_eq!(file_refs, vec!["/b.pdf".to_string()]);
    assert_eq!(cleaned, "and");
}

#[test]
fn ollama_payload_accepts_a_data_uri_and_returns_only_the_payload() {
    let encoded = STANDARD.encode(PNG_BYTES);
    let extracted = extract_ollama_image_payload(&format!("data:image/png;base64,{encoded}"));
    assert_eq!(extracted, Some(encoded));
}

/// The bug this guards: a filesystem path forwarded to the provider as if it
/// were image bytes, producing an error that names neither the path nor the
/// parameter.
#[test]
fn ollama_payload_rejects_a_filesystem_path() {
    assert_eq!(extract_ollama_image_payload("/tmp/photo.png"), None);
}

/// `/9j/…` is how a bare base64 JPEG legitimately starts, so the absolute-path
/// heuristic must not reject a real payload carrying that prefix. Length is
/// what separates them.
#[test]
fn ollama_payload_still_accepts_a_long_bare_payload_beginning_with_a_slash() {
    let payload = format!("/9j/{}", "A".repeat(200));
    assert_eq!(
        extract_ollama_image_payload(&payload),
        Some(payload.clone())
    );
}

#[test]
fn ollama_payload_accepts_both_padded_and_unpadded_base64() {
    // "hello!" encodes to a partial trailing group; "hello" to a padded one.
    let padded = STANDARD.encode(b"hello");
    assert!(padded.ends_with('='));
    assert_eq!(
        extract_ollama_image_payload(&padded),
        Some(padded.clone()),
        "padded payloads take the STANDARD engine"
    );

    let unpadded = padded.trim_end_matches('=').to_string();
    assert_eq!(
        extract_ollama_image_payload(&unpadded),
        Some(unpadded.clone()),
        "an unpadded payload must not be rejected as non-base64"
    );
}

#[test]
fn ollama_payload_rejects_a_data_uri_whose_payload_is_not_base64() {
    assert_eq!(
        extract_ollama_image_payload("data:image/png;base64,not base64!!"),
        None
    );
}

#[test]
fn placeholders_round_trip_through_the_stash_index() {
    let text = format!("here it is\n{}", image_placeholder("abc123"));
    assert!(text_has_image_placeholders(&text));
    assert_eq!(
        extract_image_placeholders_in_text(&text),
        vec![image_placeholder("abc123")]
    );

    let mut index = HashMap::new();
    index.insert("abc123".to_string(), PathBuf::from("/stash/abc123.png"));
    assert_eq!(
        rehydrate_placeholders_in_text(&text, &index),
        "here it is\n[IMAGE:/stash/abc123.png]"
    );
}

/// A swept or foreign attachment keeps its human-readable placeholder rather
/// than becoming a path that does not exist.
#[test]
fn an_unresolved_placeholder_keeps_its_text() {
    let text = image_placeholder("gone");
    let rehydrated = rehydrate_placeholders_in_text(&text, &HashMap::new());
    assert_eq!(rehydrated, text);
}

/// `[Image: (could not be processed)]` has no id, so it is not a rehydration
/// candidate even though it shares the prefix.
#[test]
fn a_placeholder_without_a_stash_id_is_not_a_candidate() {
    let text = "[Image: (could not be processed)]";
    assert!(!text_has_image_placeholders(text));
    assert!(extract_image_placeholders_in_text(text).is_empty());
}

// ── data URIs ────────────────────────────────────────────────────────────

#[test]
fn data_uri_parsing_lower_cases_the_mime_and_percent_decodes_parameters() {
    let uri = format!(
        "data:TEXT/Plain;name=my%20notes.txt;base64,{}",
        STANDARD.encode(b"hi")
    );
    let parsed = parse_data_uri(&uri).expect("parses");
    assert_eq!(parsed.mime, "text/plain");
    assert_eq!(
        data_uri_param(&parsed.params, "NAME"),
        Some("my notes.txt".to_string())
    );
    assert_eq!(parsed.bytes, b"hi");
}

#[test]
fn data_uri_parsing_rejects_the_non_base64_form() {
    let error = parse_data_uri("data:text/plain,hello").expect_err("rejected");
    assert!(error.contains("base64"), "unexpected reason: {error}");
}

/// A malformed escape falls back to the raw value: a filename containing a bare
/// `%` is far likelier than one that meant to be encoded and was not.
#[test]
fn percent_decode_returns_none_on_a_truncated_escape() {
    assert_eq!(percent_decode("100%"), None);
    assert_eq!(percent_decode("plain"), Some("plain".to_string()));
}

#[test]
fn gunzip_refuses_a_payload_that_decompresses_past_the_cap() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&vec![b'a'; 4096]).expect("write");
    let compressed = encoder.finish().expect("finish");

    assert_eq!(gunzip(&compressed, 4096).expect("within cap").len(), 4096);

    let error = gunzip(&compressed, 100).expect_err("over cap");
    assert!(error.contains("exceeds 100 bytes"), "unexpected: {error}");
}

// ── MIME ─────────────────────────────────────────────────────────────────

#[test]
fn an_image_content_type_header_wins_over_everything_else() {
    // JPEG magic, PNG header: the header is authoritative for images.
    let detected = detect_image_mime(None, &[0xff, 0xd8, 0xff], Some("image/png; charset=binary"));
    assert_eq!(detected.as_deref(), Some("image/png"));
}

/// The mirror-image rule for files: a server answering `application/octet-stream`
/// for a PDF must not cost that PDF its text layer.
#[test]
fn an_unrecognised_file_content_type_header_loses_to_magic_bytes() {
    let detected = detect_file_mime(None, b"%PDF-1.7 ...", Some("application/x-nonsense"));
    assert_eq!(detected.as_deref(), Some("application/pdf"));
}

/// OOXML containers are all `PK\x03\x04`, so only the extension separates them.
#[test]
fn the_extension_is_what_separates_ooxml_from_a_plain_zip() {
    let zip_magic = &[b'P', b'K', 0x03, 0x04, 0x00];
    assert_eq!(file_mime_from_magic(zip_magic), Some("application/zip"));
    assert_eq!(
        detect_file_mime(Some(&PathBuf::from("book.xlsx")), zip_magic, None).as_deref(),
        Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    );
    assert_eq!(
        detect_file_mime(Some(&PathBuf::from("archive.zip")), zip_magic, None).as_deref(),
        Some("application/zip")
    );
}

#[test]
fn an_unlabeled_text_file_falls_back_to_the_utf8_sniff() {
    assert!(looks_like_utf8_text(b"log line\n"));
    assert!(!looks_like_utf8_text(b""));
    assert!(!looks_like_utf8_text(&[0xff, 0xfe, 0x00]));
    assert_eq!(
        detect_file_mime(Some(&PathBuf::from("server.log")), b"boot ok\n", None).as_deref(),
        Some("text/plain")
    );
}

#[test]
fn image_magic_and_extension_tables_agree_with_each_other() {
    for (ext, mime) in [
        ("png", "image/png"),
        ("jpg", "image/jpeg"),
        ("webp", "image/webp"),
        ("gif", "image/gif"),
        ("bmp", "image/bmp"),
    ] {
        let detected = detect_image_mime(Some(&PathBuf::from(format!("a.{ext}"))), &[], None);
        assert_eq!(detected.as_deref(), Some(mime));
        assert!(image_ext_from_mime(mime).is_some());
    }
    assert_eq!(image_mime_from_magic(PNG_BYTES), Some("image/png"));
}

#[test]
fn only_the_plaintext_formats_are_extractable_without_a_parser() {
    assert!(is_extractable_text_mime("text/plain"));
    assert!(is_extractable_text_mime("text/csv"));
    assert!(is_extractable_text_mime("text/markdown"));
    assert!(!is_extractable_text_mime("application/pdf"));
    assert_eq!(file_mime_from_extension("PDF"), Some("application/pdf"));
}

// ── config ───────────────────────────────────────────────────────────────

#[test]
fn limits_clamp_absurd_configuration_into_runtime_bounds() {
    let images = ImageLimits {
        max_images: 9_000,
        max_image_size_mb: 0,
        allow_remote_fetch: false,
    };
    assert_eq!(images.effective(), (16, 1));

    let files = FileLimits {
        max_files: 9_000,
        max_file_size_mb: 9_000,
        max_extracted_text_chars: 1,
        ..FileLimits::default()
    };
    assert_eq!(files.effective(), (16, 50, 1_000));
}

/// The whole point of the sentinel: it must be read before the clamp, which
/// would otherwise turn "no files" into "one file".
#[test]
fn the_disable_sentinel_survives_the_clamp() {
    let disabled = FileLimits {
        max_files: 0,
        ..FileLimits::default()
    };
    assert!(disabled.files_disabled());
    assert_eq!(
        disabled.effective().0,
        1,
        "the clamp lifts 0 to 1 — which is exactly why files_disabled exists"
    );
    assert!(!FileLimits::default().files_disabled());
}

#[test]
fn the_file_mime_allowlist_is_case_insensitive() {
    let limits = FileLimits::default();
    assert!(limits.is_mime_allowed("APPLICATION/PDF"));
    assert!(!limits.is_mime_allowed("application/x-executable"));
    assert!(limits.supported_rendered().contains("application/pdf"));
}

// ── payload ──────────────────────────────────────────────────────────────

#[test]
fn truncation_reserves_room_for_its_own_suffix() {
    let (text, dropped) = truncate_chars("abc".to_string(), 100);
    assert_eq!((text.as_str(), dropped), ("abc", 0));

    let cap = 1_000;
    let (text, dropped) = truncate_chars("x".repeat(5_000), cap);
    assert!(dropped > 0);
    let rendered = format!("{text}\n[…truncated {dropped} chars]");
    assert!(
        rendered.chars().count() <= cap,
        "text plus suffix must stay inside the cap, got {}",
        rendered.chars().count()
    );
}

#[test]
fn truncation_counts_characters_not_bytes() {
    let (text, dropped) = truncate_chars("é".repeat(50), 30);
    assert_eq!(text.chars().count() + dropped, 50);
}

/// A filename is user-supplied, and one containing a quote would otherwise
/// break the header out of its own delimiters.
#[test]
fn attribute_escaping_neutralises_quotes_and_newlines() {
    assert_eq!(escape_attr("a\"b\nc\rd"), "a_b_c_d");
}

#[test]
fn sizes_render_in_the_largest_unit_that_fits() {
    assert_eq!(format_size(512), "512 B");
    assert_eq!(format_size(2 * 1024), "2.0 KB");
    assert_eq!(format_size(3 * 1024 * 1024), "3.0 MB");
}

#[test]
fn a_plaintext_payload_extracts_and_a_binary_one_references() {
    let extracted = FilePayload::from_resolved(
        b"hello world",
        "notes.txt".to_string(),
        "text/plain".to_string(),
        None,
        1_000,
    );
    assert!(matches!(
        &extracted,
        FilePayload::Extracted { text, .. } if text == "hello world"
    ));

    let referenced = FilePayload::from_resolved(
        &[b'P', b'K', 0x03, 0x04],
        "archive.zip".to_string(),
        "application/zip".to_string(),
        None,
        1_000,
    );
    assert!(matches!(
        &referenced,
        FilePayload::Reference { sha256_prefix, .. } if sha256_prefix.len() == 16
    ));
}

/// A host extractor's output is used for a format this crate cannot decode
/// itself; a refusal degrades to a reference rather than failing.
#[test]
fn a_host_extraction_is_used_and_its_absence_degrades() {
    let with_text = FilePayload::from_resolved(
        b"%PDF-1.7",
        "report.pdf".to_string(),
        "application/pdf".to_string(),
        Some("page one".to_string()),
        1_000,
    );
    assert!(matches!(
        &with_text,
        FilePayload::Extracted { text, .. } if text == "page one"
    ));

    let without = FilePayload::from_resolved(
        b"%PDF-1.7",
        "report.pdf".to_string(),
        "application/pdf".to_string(),
        None,
        1_000,
    );
    assert!(matches!(&without, FilePayload::Reference { .. }));
}

#[test]
fn composing_renders_text_then_images_then_files() {
    let rendered = compose_multimodal_message(
        "look at these",
        &["data:image/png;base64,AAAA".to_string()],
        &[FilePayload::Extracted {
            name: "notes.txt".to_string(),
            mime: "text/plain".to_string(),
            size_bytes: 11,
            text: "hello world".to_string(),
            truncated_chars: 0,
        }],
    );

    assert!(rendered.starts_with("look at these"));
    assert!(rendered.contains("[IMAGE:data:image/png;base64,AAAA]"));
    assert!(
        rendered.contains(r#"[FILE-EXTRACTED: name="notes.txt" size="11 B" mime="text/plain"]"#)
    );
    assert!(rendered.trim_end().ends_with("[/FILE-EXTRACTED]"));
    assert!(!rendered.contains("truncated"));
}

#[test]
fn a_truncated_payload_says_so_in_the_rendered_block() {
    let rendered = compose_multimodal_message(
        "",
        &[],
        &[FilePayload::Extracted {
            name: "big.txt".to_string(),
            mime: "text/plain".to_string(),
            size_bytes: 4_096,
            text: "start".to_string(),
            truncated_chars: 4_091,
        }],
    );
    assert!(rendered.contains("[…truncated 4091 chars]"));
}

// ── resolve ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn an_image_data_uri_round_trips_to_a_canonical_data_uri() {
    let limits = ImageLimits::default();
    let resolved = resolve_image(
        &png_data_uri(),
        &limits,
        limits.max_image_bytes(),
        &client(),
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, png_data_uri());
}

#[tokio::test]
async fn an_oversized_image_is_refused_by_size_before_anything_else() {
    let limits = ImageLimits::default();
    let error = resolve_image(&png_data_uri(), &limits, 4, &client())
        .await
        .expect_err("refused");
    assert!(matches!(error, MultimodalError::ImageTooLarge { .. }));
}

#[tokio::test]
async fn a_disallowed_image_mime_is_refused_even_as_a_data_uri() {
    let limits = ImageLimits::default();
    let uri = format!("data:image/tiff;base64,{}", STANDARD.encode(b"II*\0"));
    let error = resolve_image(&uri, &limits, 1_024, &client())
        .await
        .expect_err("refused");
    assert!(matches!(error, MultimodalError::UnsupportedMime { .. }));
}

#[tokio::test]
async fn remote_references_are_refused_while_remote_fetch_is_off() {
    let images = ImageLimits::default();
    assert!(!images.allow_remote_fetch);
    let error = resolve_image("https://example.com/a.png", &images, 1_024, &client())
        .await
        .expect_err("refused");
    assert!(matches!(error, MultimodalError::RemoteFetchDisabled { .. }));

    let files = FileLimits::default();
    assert!(!files.allow_remote_fetch);
    let error = resolve_file(
        "https://example.com/a.pdf",
        &files,
        1_024,
        1_000,
        &client(),
        &NoTextExtractor,
    )
    .await
    .expect_err("refused");
    assert!(matches!(
        error,
        MultimodalError::RemoteFileFetchDisabled { .. }
    ));
}

#[tokio::test]
async fn a_missing_local_reference_is_reported_as_not_found() {
    let images = ImageLimits::default();
    let error = resolve_image("/definitely/not/here.png", &images, 1_024, &client())
        .await
        .expect_err("refused");
    assert!(matches!(error, MultimodalError::ImageSourceNotFound { .. }));

    let files = FileLimits::default();
    let error = resolve_file(
        "/definitely/not/here.pdf",
        &files,
        1_024,
        1_000,
        &client(),
        &NoTextExtractor,
    )
    .await
    .expect_err("refused");
    assert!(matches!(error, MultimodalError::FileSourceNotFound { .. }));
}

#[tokio::test]
async fn a_gzipped_data_uri_decompresses_when_it_names_its_original_mime() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(PNG_BYTES).expect("write");
    let compressed = encoder.finish().expect("finish");

    let uri = format!(
        "data:application/gzip;original_mime=image/png;base64,{}",
        STANDARD.encode(&compressed)
    );
    let limits = ImageLimits::default();
    let resolved = resolve_image(&uri, &limits, limits.max_image_bytes(), &client())
        .await
        .expect("resolves");
    assert_eq!(resolved, png_data_uri());
}

/// Without `original_mime` the payload cannot be validated against any
/// allowlist, so it is refused rather than guessed at.
#[tokio::test]
async fn a_gzipped_data_uri_without_original_mime_is_refused() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(PNG_BYTES).expect("write");
    let compressed = encoder.finish().expect("finish");

    let uri = format!(
        "data:application/gzip;base64,{}",
        STANDARD.encode(&compressed)
    );
    let limits = ImageLimits::default();
    let error = resolve_image(&uri, &limits, limits.max_image_bytes(), &client())
        .await
        .expect_err("refused");
    assert!(matches!(error, MultimodalError::InvalidMarker { .. }));
}

#[tokio::test]
async fn a_file_data_uri_takes_its_name_from_the_header_parameter() {
    let uri = format!(
        "data:text/plain;name=notes.txt;base64,{}",
        STANDARD.encode(b"hello world")
    );
    let limits = FileLimits::default();
    let payload = resolve_file(
        &uri,
        &limits,
        limits.max_file_bytes(),
        1_000,
        &client(),
        &NoTextExtractor,
    )
    .await
    .expect("resolves");

    match payload {
        FilePayload::Extracted { name, text, .. } => {
            assert_eq!(name, "notes.txt");
            assert_eq!(text, "hello world");
        }
        other => panic!("expected extracted text, got {other:?}"),
    }
}

#[tokio::test]
async fn a_file_whose_mime_is_not_allowlisted_is_refused_after_detection() {
    let uri = format!(
        "data:application/x-executable;base64,{}",
        STANDARD.encode(b"\x7fELF")
    );
    let limits = FileLimits::default();
    let error = resolve_file(
        &uri,
        &limits,
        limits.max_file_bytes(),
        1_000,
        &client(),
        &NoTextExtractor,
    )
    .await
    .expect_err("refused");
    assert!(matches!(error, MultimodalError::UnsupportedFileMime { .. }));
}

/// Counts how often the extractor is consulted, so the `handles` short-circuit
/// is observable.
#[derive(Default)]
struct CountingExtractor {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl TextExtractor for CountingExtractor {
    fn handles(&self, mime: &str) -> bool {
        mime == "application/pdf"
    }

    async fn extract(&self, _mime: &str, _bytes: &[u8]) -> std::result::Result<String, String> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok("extracted page".to_string())
    }
}

#[tokio::test]
async fn the_extractor_is_consulted_for_what_it_claims_and_nothing_else() {
    let extractor = CountingExtractor::default();
    let limits = FileLimits::default();

    let pdf = format!(
        "data:application/pdf;name=a.pdf;base64,{}",
        STANDARD.encode(b"%PDF-1.7")
    );
    let payload = resolve_file(
        &pdf,
        &limits,
        limits.max_file_bytes(),
        1_000,
        &client(),
        &extractor,
    )
    .await
    .expect("resolves");
    assert!(matches!(
        &payload,
        FilePayload::Extracted { text, .. } if text == "extracted page"
    ));

    // A binary format the extractor does not claim must not cost a call — for
    // an out-of-process parser that call is a round trip and a copy of the
    // bytes.
    let zip = format!(
        "data:application/zip;name=a.zip;base64,{}",
        STANDARD.encode([b'P', b'K', 0x03, 0x04])
    );
    let payload = resolve_file(
        &zip,
        &limits,
        limits.max_file_bytes(),
        1_000,
        &client(),
        &extractor,
    )
    .await
    .expect("resolves");
    assert!(matches!(&payload, FilePayload::Reference { .. }));

    assert_eq!(
        extractor.calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "only the claimed MIME type should reach the extractor"
    );
}

#[tokio::test]
async fn a_host_with_no_extractor_degrades_a_pdf_to_a_reference() {
    let uri = format!(
        "data:application/pdf;name=a.pdf;base64,{}",
        STANDARD.encode(b"%PDF-1.7")
    );
    let limits = FileLimits::default();
    let payload = resolve_file(
        &uri,
        &limits,
        limits.max_file_bytes(),
        1_000,
        &client(),
        &NoTextExtractor,
    )
    .await
    .expect("resolves");
    assert!(matches!(payload, FilePayload::Reference { .. }));
}
