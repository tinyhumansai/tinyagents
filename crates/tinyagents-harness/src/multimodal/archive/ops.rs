//! Bounded archive inspection. Entry names are data, including absolute and
//! traversal paths; this module never extracts or creates filesystem objects.

use std::io::{self, Cursor, Read};

use super::types::*;
use super::zip_admission::{open_admitted_zip, zip_preflight};

/// List an archive in memory. Malformed headers and compressed streams are
/// errors; resource caps return a partial listing with a truncation reason.
/// ZIP members are streamed to a sink to validate CRC and expansion bounds.
/// TAR.GZ is expanded once under the decoded-byte cap. GNU/PAX metadata is
/// listed as `Other`, rather than followed or expanded into unbounded names.
pub fn inspect_archive(
    bytes: &[u8],
    format: ArchiveFormat,
    limits: &ArchiveLimits,
) -> Result<ArchiveListing, ArchiveError> {
    let limits = limits.effective();
    if bytes.len() > limits.max_input_bytes {
        return Err(ArchiveError::InputTooLarge);
    }
    match format {
        ArchiveFormat::Zip => inspect_zip(bytes, &limits),
        ArchiveFormat::Tar => inspect_tar(bytes, &limits),
        ArchiveFormat::TarGzip => {
            let mut decoded = Vec::new();
            flate2::read::MultiGzDecoder::new(bytes)
                .take(limits.max_decompressed_bytes.saturating_add(1) as u64)
                .read_to_end(&mut decoded)?;
            if decoded.len() > limits.max_decompressed_bytes {
                return Ok(ArchiveListing {
                    entries: Vec::new(),
                    truncation: Some(ArchiveTruncation::DecompressedBytes),
                });
            }
            inspect_tar(&decoded, &limits)
        }
    }
}

fn add_entry(
    list: &mut ArchiveListing,
    entry: ArchiveEntry,
    limits: &ArchiveLimits,
    names: &mut usize,
) -> bool {
    if list.entries.len() >= limits.max_entries {
        list.truncation = Some(ArchiveTruncation::Entries);
        return false;
    }
    if entry.name.len() > limits.max_name_bytes
        || names.saturating_add(entry.name.len()) > limits.max_total_name_bytes
    {
        list.truncation = Some(ArchiveTruncation::NameBytes);
        return false;
    }
    *names += entry.name.len();
    list.entries.push(entry);
    true
}

fn inspect_zip(bytes: &[u8], limits: &ArchiveLimits) -> Result<ArchiveListing, ArchiveError> {
    if let Some(reason) = zip_preflight(bytes)? {
        return Ok(ArchiveListing {
            entries: Vec::new(),
            truncation: Some(reason),
        });
    }
    let mut archive = open_admitted_zip(bytes)?;
    let mut list = ArchiveListing::default();
    let mut names = 0;
    let mut expanded = 0u64;
    for index in 0..archive.len() {
        if list.entries.len() >= limits.max_entries {
            list.truncation = Some(ArchiveTruncation::Entries);
            break;
        }
        let mut file = archive
            .by_index(index)
            .map_err(|e| ArchiveError::Invalid(e.to_string()))?;
        let kind = if file.is_dir() {
            ArchiveEntryKind::Directory
        } else if file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            ArchiveEntryKind::Symlink
        } else {
            ArchiveEntryKind::File
        };
        let size = file.size();
        if !add_entry(
            &mut list,
            ArchiveEntry {
                // zip 9 reports a name it cannot decode instead of guessing; an
                // entry whose name cannot be read refuses the archive.
                name: file
                    .name()
                    .map_err(|e| ArchiveError::Invalid(e.to_string()))?
                    .into_owned(),
                kind,
                declared_size: size,
            },
            limits,
            &mut names,
        ) {
            break;
        }
        let remaining = (limits.max_decompressed_bytes as u64).saturating_sub(expanded);
        if size > remaining {
            list.truncation = Some(ArchiveTruncation::DecompressedBytes);
            break;
        }
        let read = io::copy(
            &mut file.by_ref().take(remaining.saturating_add(1)),
            &mut io::sink(),
        )?;
        if read > remaining {
            list.truncation = Some(ArchiveTruncation::DecompressedBytes);
            break;
        }
        expanded += read;
    }
    Ok(list)
}

fn inspect_tar(bytes: &[u8], limits: &ArchiveLimits) -> Result<ArchiveListing, ArchiveError> {
    if bytes.len() > limits.max_decompressed_bytes {
        return Ok(ArchiveListing {
            entries: Vec::new(),
            truncation: Some(ArchiveTruncation::DecompressedBytes),
        });
    }
    let mut archive = tar::Archive::new(Cursor::new(bytes));
    let mut list = ArchiveListing::default();
    let mut names = 0;
    for entry in archive.entries()?.raw(true) {
        if list.entries.len() >= limits.max_entries {
            list.truncation = Some(ArchiveTruncation::Entries);
            break;
        }
        let entry = entry?;
        let header = entry.header();
        let entry_type = header.entry_type();
        let kind = if entry_type.is_file() {
            ArchiveEntryKind::File
        } else if entry_type.is_dir() {
            ArchiveEntryKind::Directory
        } else if entry_type.is_symlink() {
            ArchiveEntryKind::Symlink
        } else if entry_type.is_hard_link() {
            ArchiveEntryKind::HardLink
        } else {
            ArchiveEntryKind::Other
        };
        let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        let size = header.size()?;
        if !add_entry(
            &mut list,
            ArchiveEntry {
                name,
                kind,
                declared_size: size,
            },
            limits,
            &mut names,
        ) {
            break;
        }
        if entry.raw_file_position().saturating_add(size) > bytes.len() as u64 {
            return Err(ArchiveError::Invalid("truncated TAR member".to_string()));
        }
    }
    Ok(list)
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
