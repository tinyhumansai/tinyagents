use super::*;

#[test]
fn malformed_archives_are_errors() {
    for format in [
        ArchiveFormat::Zip,
        ArchiveFormat::Tar,
        ArchiveFormat::TarGzip,
    ] {
        assert!(inspect_archive(b"not an archive", format, &ArchiveLimits::default()).is_err());
    }
}

use std::io::{Cursor, Write};

fn zip_fixture(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in entries {
        writer
            .start_file(
                *name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn tar_fixture() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(5);
    // Set path bytes directly: the writer correctly refuses traversal paths,
    // but the inspector must display a hostile archive without extracting it.
    header.as_mut_bytes()[..14].copy_from_slice(b"../outside.txt");
    header.set_cksum();
    builder.append(&header, &b"hello"[..]).unwrap();
    let mut link = tar::Header::new_gnu();
    link.set_path("link").unwrap();
    link.set_entry_type(tar::EntryType::Symlink);
    link.set_link_name("/etc/passwd").unwrap();
    link.set_size(0);
    link.set_cksum();
    builder.append(&link, &b""[..]).unwrap();
    builder.into_inner().unwrap()
}

#[test]
fn zip_listing_preserves_names_sizes_and_applies_each_budget() {
    let bytes = zip_fixture(&[("../outside.txt", b"hello"), ("normal.txt", b"world")]);
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).unwrap();
    assert_eq!(
        list.entries[0],
        ArchiveEntry {
            name: "../outside.txt".into(),
            kind: ArchiveEntryKind::File,
            declared_size: 5
        }
    );
    assert_eq!(list.entries.len(), 2);
    assert_eq!(list.truncation, None);
    let limits = ArchiveLimits {
        max_entries: 1,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert_eq!(list.entries.len(), 1);
    assert_eq!(list.truncation, Some(ArchiveTruncation::Entries));
    for limits in [
        ArchiveLimits {
            max_name_bytes: 3,
            ..ArchiveLimits::default()
        },
        ArchiveLimits {
            max_total_name_bytes: 3,
            ..ArchiveLimits::default()
        },
    ] {
        let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
        assert!(list.entries.is_empty());
        assert_eq!(list.truncation, Some(ArchiveTruncation::NameBytes));
    }
    assert!(matches!(
        inspect_archive(
            &bytes,
            ArchiveFormat::Zip,
            &ArchiveLimits {
                max_input_bytes: 2,
                ..ArchiveLimits::default()
            }
        ),
        Err(ArchiveError::InputTooLarge)
    ));
}

#[test]
fn tar_and_tar_gzip_list_traversal_and_links_without_extracting() {
    let bytes = tar_fixture();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&bytes).unwrap();
    let gzip = encoder.finish().unwrap();
    for (format, bytes) in [
        (ArchiveFormat::Tar, bytes.as_slice()),
        (ArchiveFormat::TarGzip, gzip.as_slice()),
    ] {
        let list = inspect_archive(bytes, format, &ArchiveLimits::default()).unwrap();
        assert_eq!(list.entries[0].name, "../outside.txt");
        assert_eq!(list.entries[0].declared_size, 5);
        assert_eq!(list.entries[1].kind, ArchiveEntryKind::Symlink);
        assert_eq!(list.truncation, None);
        let list = inspect_archive(
            bytes,
            format,
            &ArchiveLimits {
                max_decompressed_bytes: 100,
                ..ArchiveLimits::default()
            },
        )
        .unwrap();
        assert_eq!(list.truncation, Some(ArchiveTruncation::DecompressedBytes));
    }
}

#[test]
fn compressed_zip_expansion_is_capped_and_truncated_tar_is_an_error() {
    let bytes = zip_fixture(&[("bomb", &vec![0; 100_000])]);
    assert!(bytes.len() < 1000);
    let list = inspect_archive(
        &bytes,
        ArchiveFormat::Zip,
        &ArchiveLimits {
            max_decompressed_bytes: 1000,
            ..ArchiveLimits::default()
        },
    )
    .unwrap();
    assert_eq!(list.entries[0].declared_size, 100_000);
    assert_eq!(list.truncation, Some(ArchiveTruncation::DecompressedBytes));
    let tar = tar_fixture();
    assert!(inspect_archive(&tar[..514], ArchiveFormat::Tar, &ArchiveLimits::default()).is_err());
}

#[test]
fn archive_detection_does_not_mistake_office_documents_for_archives() {
    assert_eq!(
        ArchiveFormat::detect("table.xlsx", "application/zip", b"PK\x03\x04"),
        None
    );
    assert_eq!(
        ArchiveFormat::detect("archive.zip", "application/zip", b"PK\x03\x04"),
        Some(ArchiveFormat::Zip)
    );
    let mut tar_gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    tar_gzip.write_all(&tar_fixture()).unwrap();
    let tar_gzip = tar_gzip.finish().unwrap();
    assert_eq!(
        ArchiveFormat::detect("attachment", "application/gzip", &tar_gzip),
        Some(ArchiveFormat::TarGzip)
    );
    assert_eq!(
        ArchiveFormat::detect("stale.zip", "application/gzip", &tar_gzip),
        Some(ArchiveFormat::TarGzip)
    );
    let tar = tar_fixture();
    assert_eq!(
        ArchiveFormat::detect("stale.zip", "application/x-tar", &tar),
        Some(ArchiveFormat::Tar)
    );
    assert_eq!(
        ArchiveFormat::detect("attachment.tgz", "application/octet-stream", &[]),
        Some(ArchiveFormat::TarGzip)
    );
    let mut text_gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    text_gzip.write_all(b"compressed text").unwrap();
    assert_eq!(
        ArchiveFormat::detect(
            "attachment",
            "application/gzip",
            &text_gzip.finish().unwrap()
        ),
        None
    );
    let mut empty_tar_gzip =
        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    empty_tar_gzip.write_all(&[0; 1024]).unwrap();
    assert_eq!(
        ArchiveFormat::detect(
            "empty.tar.gz",
            "application/gzip",
            &empty_tar_gzip.finish().unwrap()
        ),
        Some(ArchiveFormat::TarGzip)
    );
    assert_eq!(
        ArchiveFormat::detect("archive.tar", "application/x-tar", b""),
        Some(ArchiveFormat::Tar)
    );
}

#[test]
fn office_container_contents_override_generic_zip_headers() {
    for (member, expected) in [
        (
            "word/document.xml",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "xl/workbook.xml",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "ppt/presentation.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ] {
        let bytes = zip_fixture(&[
            ("[Content_Types].xml", b"<Types/>"),
            (member, b"<document/>"),
        ]);
        assert_eq!(
            crate::multimodal::mime::detect_file_mime(None, &bytes, Some("application/zip"))
                .as_deref(),
            Some(expected)
        );
    }
}

#[test]
fn corrupt_zip_payloads_and_gzip_trailers_are_errors() {
    let mut bytes = zip_fixture(&[("file.txt", b"hello world")]);
    let start = zip::ZipArchive::new(Cursor::new(&bytes))
        .unwrap()
        .by_index(0)
        .unwrap()
        .data_start()
        .expect("an entry read from the archive knows where its data starts")
        as usize;
    bytes[start] ^= 0xff;
    assert!(inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).is_err());
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&tar_fixture()).unwrap();
    let mut gzip = encoder.finish().unwrap();
    let checksum = gzip.len() - 8;
    gzip[checksum] ^= 0xff;
    assert!(inspect_archive(&gzip, ArchiveFormat::TarGzip, &ArchiveLimits::default()).is_err());
}

#[test]
fn exact_and_zero_listing_budgets_have_no_off_by_one() {
    let bytes = zip_fixture(&[("a", b"hello")]);
    let limits = ArchiveLimits {
        max_entries: 1,
        max_name_bytes: 1,
        max_total_name_bytes: 1,
        max_decompressed_bytes: 5,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert_eq!(list.truncation, None);
    assert_eq!(list.entries.len(), 1);
    let limits = ArchiveLimits {
        max_entries: 0,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert!(list.entries.is_empty());
    assert_eq!(list.truncation, Some(ArchiveTruncation::Entries));
}

#[test]
fn zip_admission_checks_counts_and_names_before_constructing_parser() {
    let bytes = zip_fixture(&[("one", b"")]);
    let end = bytes.len() - 22;
    let mut hostile = bytes.clone();
    hostile[end + 8..end + 12].copy_from_slice(&[255; 4]);
    assert!(zip_preflight(&hostile).is_err());
    let mut inconsistent = bytes.clone();
    inconsistent[end + 8..end + 10].copy_from_slice(&u16::MAX.to_le_bytes());
    assert!(zip_preflight(&inconsistent).is_err());
    let mut zip64 = bytes[..end].to_vec();
    let zip64_offset = zip64.len() as u64;
    zip64.extend_from_slice(b"PK\x06\x06");
    zip64.extend_from_slice(&44u64.to_le_bytes());
    zip64.extend_from_slice(&[0; 12]);
    zip64.extend_from_slice(&u64::MAX.to_le_bytes());
    zip64.extend_from_slice(&u64::MAX.to_le_bytes());
    zip64.extend_from_slice(&(end as u64).to_le_bytes());
    zip64.extend_from_slice(&0u64.to_le_bytes());
    zip64.extend_from_slice(b"PK\x06\x07");
    zip64.extend_from_slice(&0u32.to_le_bytes());
    zip64.extend_from_slice(&zip64_offset.to_le_bytes());
    zip64.extend_from_slice(&1u32.to_le_bytes());
    zip64.extend_from_slice(&hostile[end..]);
    assert!(zip_preflight(&zip64).is_err());
    let mut valid64 = zip64.clone();
    let offset = zip64_offset as usize;
    valid64[offset + 24..offset + 32].copy_from_slice(&1u64.to_le_bytes());
    valid64[offset + 32..offset + 40].copy_from_slice(&1u64.to_le_bytes());
    valid64[offset + 40..offset + 48].copy_from_slice(
        &(u32::from_le_bytes(bytes[end + 12..end + 16].try_into().unwrap()) as u64).to_le_bytes(),
    );
    valid64[offset + 48..offset + 56].copy_from_slice(
        &(u32::from_le_bytes(bytes[end + 16..end + 20].try_into().unwrap()) as u64).to_le_bytes(),
    );
    assert_eq!(zip_preflight(&valid64).unwrap(), None);
    assert_eq!(
        inspect_archive(&valid64, ArchiveFormat::Zip, &ArchiveLimits::default())
            .unwrap()
            .entries
            .len(),
        1
    );
    let mut prefixed64 = b"executable preamble".to_vec();
    prefixed64.extend_from_slice(&valid64);
    assert_eq!(zip_preflight(&prefixed64).unwrap(), None);
    assert_eq!(
        inspect_archive(&prefixed64, ArchiveFormat::Zip, &ArchiveLimits::default())
            .unwrap()
            .entries
            .len(),
        1
    );
    assert!(inspect_archive(&zip64, ArchiveFormat::Zip, &ArchiveLimits::default()).is_err());
    assert_eq!(
        crate::multimodal::mime::detect_file_mime(None, &zip64, None).as_deref(),
        Some("application/zip")
    );
    let entries: Vec<_> = (0..10_001)
        .map(|i| (format!("entry-{i}"), b"".as_slice()))
        .collect();
    let refs: Vec<_> = entries
        .iter()
        .map(|(name, data)| (name.as_str(), *data))
        .collect();
    let bytes = zip_fixture(&refs);
    assert_eq!(
        zip_preflight(&bytes).unwrap(),
        Some(ArchiveTruncation::Entries)
    );
    let name = "a".repeat(4097);
    let bytes = zip_fixture(&[(&name, b"")]);
    assert_eq!(
        zip_preflight(&bytes).unwrap(),
        Some(ArchiveTruncation::NameBytes)
    );
    let entries: Vec<_> = (0..300)
        .map(|i| (format!("{i:04}{}", "x".repeat(4090)), b"".as_slice()))
        .collect();
    let refs: Vec<_> = entries
        .iter()
        .map(|(name, data)| (name.as_str(), *data))
        .collect();
    let bytes = zip_fixture(&refs);
    assert_eq!(
        zip_preflight(&bytes).unwrap(),
        Some(ArchiveTruncation::NameBytes)
    );
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).unwrap();
    assert!(list.entries.is_empty());
    assert_eq!(list.truncation, Some(ArchiveTruncation::NameBytes));
}

#[test]
fn zip_admission_rejects_malformed_records_and_bounds_unicode_names() {
    let bytes = zip_fixture(&[("a", b"")]);
    let end = bytes.len() - 22;
    let size = u32::from_le_bytes(bytes[end + 12..end + 16].try_into().unwrap()) as usize;
    let central = end - size;
    for (offset, value) in [
        (central, 0),
        (central + 28, 255),
        (central + 29, 255),
        (end + 12, 255),
        (end + 13, 255),
        (end + 10, 0),
        (end + 20, 1),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset] = value;
        assert!(zip_preflight(&corrupt).is_err());
    }
    // Unicode path extras can replace a short raw filename during full parse.
    let mut extra = Vec::new();
    extra.extend_from_slice(&0x7075u16.to_le_bytes());
    extra.extend_from_slice(&4102u16.to_le_bytes());
    extra.extend_from_slice(&[1, 0, 0, 0, 0]);
    extra.extend_from_slice(&vec![b'x'; 4097]);
    let mut unicode = bytes.clone();
    unicode.splice(central + 47..central + 47, extra.iter().copied());
    unicode[central + 30..central + 32].copy_from_slice(&(extra.len() as u16).to_le_bytes());
    let new_end = end + extra.len();
    unicode[new_end + 12..new_end + 16]
        .copy_from_slice(&((size + extra.len()) as u32).to_le_bytes());
    assert_eq!(
        zip_preflight(&unicode).unwrap(),
        Some(ArchiveTruncation::NameBytes)
    );
    // Incomplete and overrun extra-field records never reach the eager parser.
    for declared in [1u16, 4] {
        let mut corrupt = unicode.clone();
        corrupt[central + 30..central + 32].copy_from_slice(&declared.to_le_bytes());
        assert!(zip_preflight(&corrupt).is_err());
    }
}

#[test]
fn zip_metadata_index_cannot_fall_back_to_a_footer_inside_member_data() {
    let nested = zip_fixture(&[("inside", b"nested data")]);
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(
            "nested.zip",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )
        .unwrap();
    writer.write_all(&nested).unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    assert_eq!(
        inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default())
            .unwrap()
            .entries[0]
            .declared_size,
        nested.len() as u64
    );
    // Self-extracting preambles still work with the pinned directory offset.
    let mut prefixed = b"executable preamble".to_vec();
    prefixed.extend_from_slice(&bytes);
    assert_eq!(
        inspect_archive(&prefixed, ArchiveFormat::Zip, &ArchiveLimits::default())
            .unwrap()
            .entries[0]
            .name,
        "nested.zip"
    );
    let end = bytes.len() - 22;
    let size = u32::from_le_bytes(bytes[end + 12..end + 16].try_into().unwrap()) as usize;
    let central = end - size;
    let name_len =
        u16::from_le_bytes(bytes[central + 28..central + 30].try_into().unwrap()) as usize;
    let mut corrupt = bytes.clone();
    // A structurally bounded ZIP64 field that is semantically incomplete makes
    // ZipArchive retry earlier footers. The nested member must remain hidden.
    let extra = [1, 0, 1, 0, 0];
    corrupt.splice(central + 46 + name_len..central + 46 + name_len, extra);
    corrupt[central + 30..central + 32].copy_from_slice(&5u16.to_le_bytes());
    corrupt[end + 5 + 12..end + 5 + 16].copy_from_slice(&((size + 5) as u32).to_le_bytes());
    // AES compression without its mandatory AES extra field is rejected by
    // the full parser, triggering its earlier-footer fallback path.
    corrupt[central + 10..central + 12].copy_from_slice(&99u16.to_le_bytes());
    assert_eq!(zip_preflight(&corrupt).unwrap(), None);
    assert!(inspect_archive(&corrupt, ArchiveFormat::Zip, &ArchiveLimits::default()).is_err());
}

#[test]
fn footer_signature_in_valid_zip_comment_is_conservatively_rejected() {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .set_comment("valid comment with PK\u{5}\u{6} signature")
        .expect("a short comment fits");
    writer
        .start_file("file.txt", zip::write::SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"payload").unwrap();
    let bytes = writer.finish().unwrap().into_inner();
    // The ZIP format permits these comment bytes. Our bounded admission
    // deliberately rejects them rather than exposing eager footer fallback.
    assert!(zip::ZipArchive::new(Cursor::new(&bytes)).is_ok());
    assert!(inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).is_err());
}
