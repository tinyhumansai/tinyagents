use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::io::Write;

#[tokio::test]
async fn remote_image_rejects_private_destination_before_fetch() {
    let limits = ImageLimits {
        allow_remote_fetch: true,
        ..ImageLimits::default()
    };
    let client = Client::builder()
        .timeout(std::time::Duration::from_millis(100))
        .build()
        .unwrap();
    let error = resolve_image("http://127.0.0.1:9/image.png", &limits, 1024, &client)
        .await
        .unwrap_err();
    assert!(
        matches!(error, MultimodalError::RemoteFetchFailed { reason, .. } if reason.contains("Blocked local/private host"))
    );
}

#[tokio::test]
async fn remote_file_rejects_mapped_private_destination_before_fetch() {
    let limits = FileLimits {
        allow_remote_fetch: true,
        ..FileLimits::default()
    };
    let client = Client::builder()
        .timeout(std::time::Duration::from_millis(100))
        .build()
        .unwrap();
    let error = resolve_attachment(
        "http://[::ffff:127.0.0.1]:9/file.bin",
        &limits,
        1024,
        &client,
        UnknownMimePolicy::Accept,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, MultimodalError::RemoteFileFetchFailed { reason, .. } if reason.contains("IPv6"))
    );
}

#[tokio::test]
async fn generic_resolution_retains_binary_bytes_and_decodes_only_transport_gzip() {
    let bytes = [0, 255, 7, 0];
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(&bytes).unwrap();
    let uri = format!(
        "data:application/gzip;original_mime=application/octet-stream;name=raw%20bytes.bin;base64,{}",
        STANDARD.encode(gzip.finish().unwrap())
    );
    let resolved = resolve_attachment(
        &uri,
        &FileLimits::default(),
        1024,
        &Client::new(),
        UnknownMimePolicy::Reject,
    )
    .await
    .unwrap();
    assert_eq!(resolved.bytes, bytes);
    assert_eq!(resolved.name, "raw bytes.bin");
    assert_eq!(resolved.mime, "application/octet-stream");
    assert_eq!(resolved.size_bytes, 4);
}

#[tokio::test]
async fn unknown_types_need_explicit_host_opt_in() {
    let uri = "data:application/x-private;name=a.custom;base64,AP8=";
    let limits = FileLimits::default();
    assert!(
        resolve_attachment(
            uri,
            &limits,
            1024,
            &Client::new(),
            UnknownMimePolicy::Reject
        )
        .await
        .is_err()
    );
    let resolved = resolve_attachment(
        uri,
        &limits,
        1024,
        &Client::new(),
        UnknownMimePolicy::Accept,
    )
    .await
    .unwrap();
    assert_eq!(resolved.mime, "application/x-private");
    assert_eq!(resolved.bytes, [0, 255]);
    assert!(
        resolve_file(uri, &limits, 1024, 1000, &Client::new(), &NoTextExtractor)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn disabled_files_and_transport_expansion_are_rejected() {
    let limits = FileLimits {
        max_files: 0,
        ..FileLimits::default()
    };
    assert!(
        resolve_attachment(
            "data:text/plain,hello",
            &limits,
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await
        .is_err()
    );
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(&vec![b'a'; 4096]).unwrap();
    let uri = format!(
        "data:application/gzip;original_mime=text/plain;base64,{}",
        STANDARD.encode(gzip.finish().unwrap())
    );
    assert!(
        resolve_attachment(
            &uri,
            &FileLimits::default(),
            100,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn archive_gzip_is_retained_and_nested_gzip_is_not_decoded_twice() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"tar-like payload").unwrap();
    let original = encoder.finish().unwrap();
    let mut outer = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    outer.write_all(&original).unwrap();
    for uri in [
        format!(
            "data:application/gzip;name=a.tar.gz;base64,{}",
            STANDARD.encode(&original)
        ),
        format!(
            "data:application/gzip;original_mime=application/gzip;name=a.tar.gz;base64,{}",
            STANDARD.encode(outer.finish().unwrap())
        ),
    ] {
        let resolved = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(resolved.bytes, original);
        assert_eq!(resolved.name, "a.tar.gz");
        assert_eq!(resolved.mime, "application/gzip");
    }
}

#[tokio::test]
async fn local_unknown_bytes_preserve_legacy_rejection_and_generic_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("opaque.custom");
    tokio::fs::write(&path, [0, 255, 7]).await.unwrap();
    let source = path.to_str().unwrap();
    assert!(
        resolve_attachment(
            source,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Reject
        )
        .await
        .is_err()
    );
    let resolved = resolve_attachment(
        source,
        &FileLimits::default(),
        1024,
        &Client::new(),
        UnknownMimePolicy::Accept,
    )
    .await
    .unwrap();
    assert_eq!(resolved.bytes, [0, 255, 7]);
    assert_eq!(resolved.mime, "application/octet-stream");
    assert_eq!(resolved.name, "opaque.custom");
    assert!(matches!(
        resolve_attachment(
            source,
            &FileLimits::default(),
            2,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await,
        Err(MultimodalError::FileTooLarge { .. })
    ));
}

#[tokio::test]
async fn opt_in_retains_images_audio_video_and_native_documents_verbatim() {
    for (mime, bytes) in [
        ("image/png", &b"\x89PNG\r\n\x1a\n"[..]),
        ("audio/wav", &b"RIFF\0\0\0\0WAVE"[..]),
        ("video/mp4", &b"\0\0\0\x18ftypmp42"[..]),
        ("application/pdf", &b"%PDF-1.7"[..]),
    ] {
        let uri = format!("data:{mime};name=media;base64,{}", STANDARD.encode(bytes));
        let resolved = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(resolved.mime, mime);
        assert_eq!(resolved.bytes, bytes);
    }
}

#[tokio::test]
async fn malformed_and_oversized_data_uri_payloads_fail_before_extraction() {
    for uri in [
        "data:text/plain;base64,%%%",
        "data:text/plain,bad%Q0",
        "data:application/gzip;original_mime=text/plain;base64,AP8=",
    ] {
        assert!(matches!(
            resolve_attachment(
                uri,
                &FileLimits::default(),
                1024,
                &Client::new(),
                UnknownMimePolicy::Accept
            )
            .await,
            Err(MultimodalError::InvalidFileMarker { .. })
        ));
    }
    assert!(matches!(
        resolve_attachment(
            "data:text/plain;base64,YWFhYWFhYWFhYWFh",
            &FileLimits::default(),
            2,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await,
        Err(MultimodalError::FileTooLarge { .. })
    ));
}

#[tokio::test]
async fn generic_http_mime_precedes_utf8_sniff_and_legacy_stays_narrow() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = "http://example.com/media";
    let client = Client::builder()
        .no_proxy()
        .resolve("example.com", listener.local_addr().unwrap())
        .build()
        .unwrap();
    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 2048];
            let mut received = 0;
            loop {
                let amount = stream.read(&mut request[received..]).await.unwrap();
                assert!(amount > 0);
                received += amount;
                if request[..received]
                    .windows(4)
                    .any(|part| part == b"\r\n\r\n")
                {
                    break;
                }
                assert!(received < request.len());
            }
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav; charset=binary\r\nContent-Length: 12\r\nConnection: close\r\n\r\nRIFF\0\0\0\0WAVE").await.unwrap();
        }
    });
    let limits = FileLimits {
        allow_remote_fetch: true,
        ..FileLimits::default()
    };
    let resolved = resolve_attachment(url, &limits, 1024, &client, UnknownMimePolicy::Accept)
        .await
        .unwrap();
    assert_eq!(resolved.mime, "audio/wav");
    assert_eq!(resolved.bytes, b"RIFF\0\0\0\0WAVE");
    let legacy = resolve_attachment(url, &limits, 1024, &client, UnknownMimePolicy::Reject)
        .await
        .unwrap();
    assert_eq!(legacy.mime, "text/plain");
    server.await.unwrap();
}

#[tokio::test]
async fn generic_local_media_uses_extensions_and_magic() {
    let dir = tempfile::tempdir().unwrap();
    for (name, bytes, mime) in [
        ("sound.wav", &b"RIFF\0\0\0\0WAVE"[..], "audio/wav"),
        ("movie.mp4", &b"\0\0\0\x18ftypmp42"[..], "video/mp4"),
        ("recording.mp3", &b"ID3"[..], "audio/mpeg"),
        ("sound.flac", &b"fLaC"[..], "audio/flac"),
        ("sound.ogg", &b"OggS"[..], "audio/ogg"),
        ("clip.ogv", &b"OggS"[..], "video/ogg"),
        ("sound.opus", &b"opaque"[..], "audio/opus"),
        ("sound.m4a", &b"opaque"[..], "audio/mp4"),
        ("sound.aac", &b"opaque"[..], "audio/aac"),
        ("movie.mov", &b"opaque"[..], "video/quicktime"),
        ("movie.webm", &b"opaque"[..], "video/webm"),
        ("movie.mkv", &b"opaque"[..], "video/x-matroska"),
        ("movie.avi", &b"opaque"[..], "video/x-msvideo"),
        ("magic-mp3", &b"ID3"[..], "audio/mpeg"),
        ("magic-flac", &b"fLaC"[..], "audio/flac"),
        ("magic-ogg", &b"OggS"[..], "audio/ogg"),
        ("magic-avi", &b"RIFF\0\0\0\0AVI "[..], "video/x-msvideo"),
        ("magic-mp4", &b"\0\0\0\x18ftypmp42"[..], "video/mp4"),
        ("magic-m4a", &b"\0\0\0\x18ftypM4A "[..], "audio/mp4"),
        ("magic-mov", &b"\0\0\0\x18ftypqt  "[..], "video/quicktime"),
        ("unlabelled", &b"RIFF\0\0\0\0WAVE"[..], "audio/wav"),
    ] {
        let path = dir.path().join(name);
        tokio::fs::write(&path, bytes).await.unwrap();
        let resolved = resolve_attachment(
            path.to_str().unwrap(),
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(resolved.mime, mime);
        assert_eq!(resolved.bytes, bytes);
    }
}

#[test]
fn iso_bmff_brands_distinguish_images_audio_and_known_video() {
    for (brand, expected) in [
        ("avif", "image/avif"),
        ("avis", "image/avif"),
        ("heic", "image/heic"),
        ("heix", "image/heic"),
        ("mif1", "image/heif"),
        ("msf1", "image/heif"),
        ("M4A ", "audio/mp4"),
        ("M4B ", "audio/mp4"),
        ("M4P ", "audio/mp4"),
        ("qt  ", "video/quicktime"),
        ("mp42", "video/mp4"),
        ("isom", "video/mp4"),
    ] {
        let bytes = [b"\0\0\0\x18ftyp".as_slice(), brand.as_bytes()].concat();
        assert_eq!(
            crate::multimodal::mime::detect_attachment_mime(
                std::path::Path::new("attachment"),
                &bytes,
                None
            )
            .as_deref(),
            Some(expected),
            "{brand}"
        );
    }
    let unknown = b"\0\0\0\x18ftypzzzz";
    assert_ne!(
        crate::multimodal::mime::detect_attachment_mime(
            std::path::Path::new("attachment"),
            unknown,
            None
        )
        .as_deref(),
        Some("video/mp4")
    );
    for (name, expected) in [
        ("photo.avif", "image/avif"),
        ("photo.heic", "image/heic"),
        ("photo.heif", "image/heif"),
        ("book.m4b", "audio/mp4"),
    ] {
        assert_eq!(
            crate::multimodal::mime::detect_attachment_mime(
                std::path::Path::new(name),
                unknown,
                None
            )
            .as_deref(),
            Some(expected)
        );
    }
}

#[tokio::test]
async fn omitted_data_uri_media_type_uses_rfc_text_default() {
    for (uri, expected) in [
        ("data:,hello", b"hello".as_slice()),
        ("data:;base64,aGVsbG8=", b"hello".as_slice()),
    ] {
        for policy in [UnknownMimePolicy::Accept, UnknownMimePolicy::Reject] {
            let resolved =
                resolve_attachment(uri, &FileLimits::default(), 1024, &Client::new(), policy)
                    .await
                    .unwrap();
            assert_eq!(resolved.mime, "text/plain");
            assert_eq!(resolved.bytes, expected);
        }
    }
}

#[tokio::test]
async fn generic_data_uris_detect_media_like_local_sources_without_changing_legacy() {
    for (name, bytes, expected) in [
        ("clip.mp4", b"\0\0\0\x18ftypmp42".as_slice(), "video/mp4"),
        ("photo.avif", b"\0\0\0\x18ftypavif".as_slice(), "image/avif"),
        ("sound.wav", b"RIFF\0\0\0\0WAVE".as_slice(), "audio/wav"),
    ] {
        let uri = format!(
            "data:application/octet-stream;name={name};base64,{}",
            STANDARD.encode(bytes)
        );
        let accepted = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(accepted.mime, expected);
        assert_eq!(accepted.bytes, bytes);
        let legacy = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Reject,
        )
        .await
        .unwrap();
        assert_eq!(legacy.mime, "application/octet-stream");
        assert_eq!(legacy.bytes, bytes);
    }
}

#[test]
fn generic_media_signatures_precede_misleading_extensions() {
    for (name, bytes, expected) in [
        ("photo.avif", b"\0\0\0\x18ftypmp42".as_slice(), "video/mp4"),
        ("movie.mp4", b"\0\0\0\x18ftypavif".as_slice(), "image/avif"),
        ("movie.mp4", b"RIFF\0\0\0\0WAVE".as_slice(), "audio/wav"),
        ("movie.mp4", b"\xff\xd8\xffbinary".as_slice(), "image/jpeg"),
    ] {
        assert_eq!(
            crate::multimodal::mime::detect_attachment_mime(
                std::path::Path::new(name),
                bytes,
                None
            )
            .as_deref(),
            Some(expected)
        );
    }
}
