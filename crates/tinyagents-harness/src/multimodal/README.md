# harness::multimodal

Attachment resolution for `[IMAGE:…]` and `[FILE:…]` markers embedded in
message text.

## Why this exists

A user attaches a picture or a document; somewhere between the text box and
the provider that attachment has to become bytes the model can read —
validated, size-capped, MIME-checked, and rendered into the message. This
module is that pipeline, minus every decision that belongs to a host.
Attachments travel as markers inside message text rather than a parallel
structured field, so they survive every hop a host already has (persistence,
summarisation, delegation) without each of those learning about attachments.

Images and files are symmetrical up to the payload and diverge there: images
always inline as base64 `data:` URIs (the provider vision contract); files
never inline bytes — an extractable format contributes text, a binary-only
format contributes a header naming it plus a content hash.

## Public surface

- [`config`] — [`ImageLimits`], [`FileLimits`], [`ALLOWED_IMAGE_MIME_TYPES`]:
  crate-owned, host-mapped per-turn caps, including `FileLimits`'s
  `max_files == 0` hard-disable sentinel.
- [`markers`] — the marker vocabulary (`IMAGE_MARKER_PREFIX`,
  `FILE_MARKER_PREFIX`, placeholder tokens) and pure string transforms:
  `parse_image_markers`/`parse_file_markers`, placeholder
  render/detect/extract/rehydrate functions, `extract_ollama_image_payload`.
- [`mime`] — MIME detection (header → extension → magic bytes, in an order
  that differs deliberately between images and files) and the extension/magic
  lookup tables.
- [`data_uri`] — `data:` URI parsing (including gzip-compressed attachments
  with a required `original_mime` parameter), percent-decoding, and encoding.
- [`resolve`] — [`TextExtractor`] (host-pluggable document text extraction),
  [`NoTextExtractor`], generic [`resolve_attachment`], and legacy entry points
  [`resolve_image`]/[`resolve_file`] that turn one marker reference into a
  payload, trying `data:` → `http(s)` (gated by `allow_remote_fetch`) → local
  path in that order.
- [`types`] — [`ResolvedAttachment`] and [`UnknownMimePolicy`] for generic intake.
- [`archive`] — [`inspect_archive`] for bounded ZIP/TAR/TAR.GZ listing;
  `archive/types.rs` holds formats, budgets, entries, truncation and errors.
- [`png`] — [`optimize_png_lossless`] for optional faithful PNG derivatives.
- [`payload`] — [`FilePayload`] (`Extracted` / `Reference`),
  [`compose_multimodal_message`] (renders the final provider-bound message),
  and supporting helpers (`truncate_chars`, `sha256_prefix`, `format_size`,
  `escape_attr`).
- [`error`] — [`MultimodalError`] and its [`Result`] alias; every variant
  carries the offending `input` verbatim so a multi-attachment turn's failure
  is attributable.

## Files

| File          | Role                                                             |
| ------------- | ----------------------------------------------------------------- |
| `mod.rs`      | Module overview, sub-module wiring, and re-exports.                |
| `config.rs`   | `ImageLimits`, `FileLimits`, `ALLOWED_IMAGE_MIME_TYPES`.            |
| `markers.rs`  | Marker prefixes and pure string transforms over them.              |
| `mime.rs`     | MIME detection: header, extension, and magic-byte sniffing.        |
| `data_uri.rs` | `data:` URI parsing, gzip decompression, percent-decoding.         |
| `resolve.rs`  | `TextExtractor`, `resolve_attachment`, `resolve_image`/`resolve_file`.   |
| `types.rs` | Generic resolved-attachment metadata and unknown-MIME policy. |
| `archive/mod.rs` | Archive module wiring and public exports. |
| `archive/ops.rs` | Bounded archive listing, without filesystem extraction. |
| `archive/types.rs` | Archive format, budgets, entries, errors and truncation. |
| `archive/zip_admission.rs` | Allocation-free ZIP preflight and admitted reader. |
| `png.rs` | Optional lossless PNG optimization. |
| `payload.rs`  | `FilePayload`, message composition, truncation, hashing.           |
| `error.rs`    | `MultimodalError`, `Result`.                                       |
| `mod_tests.rs`, `resolve_tests.rs`, `archive/ops_tests.rs`, `png_tests.rs`, `stash_tests.rs` | Marker, MIME, size, path, intake, archive, PNG and stash behavior. |

## Operational constraints

- **Count before resolving.** `FileLimits::files_disabled` and the per-turn
  count caps must be checked against the raw markers, before any read
  happens — a cap enforced after the fetch is not a cap.
- **Check the sentinel before the clamp.** `max_files == 0` means *none*;
  `FileLimits::effective` clamps it up to `1`. Consulting only the clamped
  value would admit one attachment from a source that asked for zero.
- What stays with the host, deliberately: the `reqwest::Client` (proxy/timeout
  policy), the `TextExtractor` implementation (which parser, if any, and its
  timeout), the attachment stash (where bytes live between ingress and
  dispatch), and message-level marker counting (only the host knows its
  message type). This module never decides which local paths may be read —
  that is `FileLimits::files_disabled`'s lever, not a filesystem allowlist
  here.
- Opt-in remote image and file URLs pass TinyTools' lexical URL guard before
  the host's client sends a request. The host client still owns DNS resolution,
  connection pinning, proxy use, and redirect policy; this admission check
  alone does not validate those subsequent destinations.
- Text extraction failures degrade to a `FilePayload::Reference`. Resolution
  errors (read/fetch/MIME/size) remain typed errors for the host to present or
  skip according to its own policy.

## Generic intake and archives

`resolve_attachment(source, &FileLimits, max_bytes, &Client, UnknownMimePolicy)`
returns `ResolvedAttachment { bytes, name, mime, size_bytes }`. It neither
extracts text nor writes files. `Reject` requires the existing MIME allowlist
and preserves rejection of undetected local bytes; `Accept` lets a host retain
arbitrary media or unknown formats without disabling byte limits, remote
fetch gates, or `max_files == 0`. The host authorizes local paths and chooses
storage locations. Names are untrusted display metadata. In `Accept` mode,
explicit HTTP media MIME types precede content sniffing, and common audio/video
extensions and signatures are recognized before the UTF-8 fallback. `Reject`
continues to use the legacy file detector.

Transport gzip is identified by `application/gzip` plus `original_mime` in a
data URI and decoded exactly once under the byte cap. A `.tar.gz` attachment
without that parameter retains its compressed bytes. Remote URL display names
come from the decoded final path segment, without query parameters. Local reads
and remote streams stop at the configured byte budget, including files that
change after their metadata is read. `resolve_file` uses the generic resolver
with `Reject`, then retains its existing extractor/degrade-to-reference flow.

`ArchiveFormat::detect(name, mime, bytes)` identifies ZIP, TAR, and TAR.GZ;
Office document MIME types/extensions are excluded. `inspect_archive(bytes,
format, &ArchiveLimits)` returns `ArchiveListing` entries with `name`, `kind`,
and `declared_size`. It performs no disk extraction and follows no links.
Traversal paths are displayed as recorded. ZIP members are streamed to a sink
for CRC validation under a cumulative expansion cap; TAR.GZ is inflated once
under a stream-byte cap. GNU/PAX extension records are listed as `Other`
rather than interpreted into potentially unbounded names.

Limits cap input bytes, entry count, individual and aggregate UTF-8 name bytes,
and decompressed bytes. Exhaustion returns `truncation` with the applicable
reason; malformed headers/CRC/streams return `ArchiveError`. A TAR/TAR.GZ
stream beyond its decoded budget produces an empty truncated listing; a ZIP
member whose declared size exceeds the remaining budget appears in the
partial listing without being inflated. A partial listing does not validate
members beyond the stopping point. ZIP listing and Office probing share an allocation-free central-directory
preflight before the eager ZIP parser. It checks ordinary and ZIP64 counts
against available directory bytes and caps eager metadata at 10,000 entries,
4096 bytes per name, 1 MiB across names, and 4 MiB of directory metadata.
Archives above those safety ceilings produce an empty truncated listing;
Office probing falls back to the usual extension/header detection. Smaller
caller listing budgets retain a prefix as usual. Invalid directory counts and
records are errors. The reader pins the admitted directory offset and hides
member footer signatures during metadata indexing so parser retries cannot
select unchecked embedded ZIPs. Member data is restored for normal CRC and
expansion validation. Office probing reads metadata and local headers only.

`types.rs` owns generic intake types; `archive/types.rs` owns listing types
and budgets; `archive/ops.rs` owns inspection; sibling `*_tests.rs` cover fixtures
and error/limit behavior. No transcript or inference content-block variants
are introduced by these APIs.

PNG derivatives use `optimize_png_lossless`: it changes only IDAT compression
and filtering, preserving pixel format, hidden RGB, interlacing and every other
chunk byte-for-byte. Animated, malformed and oversized inputs are skipped; the
caller retains the original and decides where to store the optional smaller copy.
The optimizer (`oxipng`) is behind the opt-in `png-optimize` feature; without it
`optimize_png_lossless` always returns `None`, i.e. the original is kept as is.

ZIP admission conservatively rejects footer signature bytes in central-directory
metadata or archive comments, even where ZIP permits those bytes. This prevents
the eager reader from falling back to a different allocation-heavy footer; hosts
should retain the original and report unavailable listing rather than extracting.
