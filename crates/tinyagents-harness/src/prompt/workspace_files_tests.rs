use super::*;

/// Stands in for a host's bundled default file.
const BUILTIN: &str = "the current built-in SOUL.md\n";
/// Stands in for a host's read-file tool name.
const READ_TOOL: &str = "file_read";

fn sidecar_path(workspace: &Path) -> std::path::PathBuf {
    workspace.join(".SOUL.md.builtin-hash")
}

/// The sidecar format every sync must leave behind: lowercase hex SHA-256 of
/// the built-in. Computed here independently of the code under test.
fn sha256_hex(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(content.as_bytes());
    let mut out = String::new();
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The legacy sidecar format: `DefaultHasher`, whose output may change with
/// any Rust release.
fn legacy_hash(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[test]
fn the_sidecar_is_a_dotfile_named_after_the_file() {
    assert_eq!(
        builtin_hash_sidecar_name("SOUL.md"),
        ".SOUL.md.builtin-hash"
    );
}

#[test]
fn the_builtin_hash_is_lowercase_hex_sha256() {
    // Known SHA-256 vector, pinned so the on-disk format can never drift.
    assert_eq!(
        builtin_hash("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(builtin_hash(BUILTIN), sha256_hex(BUILTIN));
}

#[test]
fn a_new_builtin_replaces_an_untouched_file_and_keeps_an_edited_one() {
    // A release changes the bundled SOUL.md. A workspace whose copy still
    // matches the previous built-in gets the new one; a workspace where the
    // user edited it keeps the edit. Both record the new built-in's hash.
    let previous = "previous built-in SOUL.md";
    for (on_disk, expected) in [
        (previous, BUILTIN),
        ("the user's own SOUL.md", "the user's own SOUL.md"),
    ] {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("SOUL.md"), on_disk).unwrap();
        std::fs::write(sidecar_path(workspace.path()), sha256_hex(previous)).unwrap();

        sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

        assert_eq!(
            std::fs::read_to_string(workspace.path().join("SOUL.md")).unwrap(),
            expected,
            "on disk before the upgrade: {on_disk:?}"
        );
        assert_eq!(
            std::fs::read_to_string(sidecar_path(workspace.path())).unwrap(),
            sha256_hex(BUILTIN)
        );
    }
}

#[test]
fn the_builtin_hash_sidecar_is_a_stable_sha256() {
    // `DefaultHasher` is not stable across Rust releases, so a toolchain bump
    // could make an edited file look like an untouched built-in. The sidecar
    // must be a hash whose output is fixed forever.
    let workspace = tempfile::tempdir().unwrap();
    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

    assert_eq!(
        std::fs::read_to_string(workspace.path().join("SOUL.md")).unwrap(),
        BUILTIN
    );
    let sidecar = std::fs::read_to_string(sidecar_path(workspace.path())).unwrap();
    assert_eq!(sidecar, sha256_hex(BUILTIN));

    // A second sync with nothing changed leaves the file and sidecar alone.
    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);
    assert_eq!(
        std::fs::read_to_string(sidecar_path(workspace.path())).unwrap(),
        sidecar
    );
}

#[test]
fn a_legacy_sidecar_is_migrated_without_authorizing_an_overwrite() {
    // A workspace synced before the SHA-256 sidecar: its SOUL.md is still the
    // previous built-in and the sidecar holds that built-in's `DefaultHasher`
    // hash. Legacy hashes are not trusted to distinguish an untouched file
    // from a colliding user edit. Preserve disk content and migrate the
    // sidecar.
    let workspace = tempfile::tempdir().unwrap();
    let soul = workspace.path().join("SOUL.md");
    let sidecar = sidecar_path(workspace.path());
    std::fs::write(&soul, "previous built-in SOUL.md").unwrap();
    std::fs::write(&sidecar, legacy_hash("previous built-in SOUL.md")).unwrap();

    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

    assert_eq!(
        std::fs::read_to_string(&soul).unwrap(),
        "previous built-in SOUL.md"
    );
    assert_eq!(
        std::fs::read_to_string(&sidecar).unwrap(),
        sha256_hex(BUILTIN)
    );
}

#[test]
fn a_user_edited_file_with_a_legacy_sidecar_is_never_overwritten() {
    // Same legacy workspace, but the user edited SOUL.md. Neither the format
    // change nor a legacy sidecar from another toolchain may cost them the
    // edit; the sidecar still moves to SHA-256.
    for stored in [
        legacy_hash("previous built-in SOUL.md"),
        // A collision with the edited disk content must not authorize it.
        legacy_hash("the user's own SOUL.md"),
        // A sidecar written by a different toolchain's `DefaultHasher`.
        "0123456789abcdef".to_string(),
    ] {
        let workspace = tempfile::tempdir().unwrap();
        let soul = workspace.path().join("SOUL.md");
        let sidecar = sidecar_path(workspace.path());
        std::fs::write(&soul, "the user's own SOUL.md").unwrap();
        std::fs::write(&sidecar, &stored).unwrap();

        sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

        assert_eq!(
            std::fs::read_to_string(&soul).unwrap(),
            "the user's own SOUL.md",
            "stored sidecar: {stored:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&sidecar).unwrap(),
            sha256_hex(BUILTIN)
        );
    }
}

#[test]
fn an_edit_made_after_sync_survives_a_resync() {
    let workspace = tempfile::tempdir().unwrap();
    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);
    let original_hash = std::fs::read_to_string(sidecar_path(workspace.path())).unwrap();

    std::fs::write(workspace.path().join("SOUL.md"), "user override").unwrap();
    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

    assert_eq!(
        std::fs::read_to_string(sidecar_path(workspace.path())).unwrap(),
        original_hash
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("SOUL.md")).unwrap(),
        "user override"
    );
}

#[test]
fn a_deleted_file_is_rewritten_even_when_the_sidecar_is_current() {
    let workspace = tempfile::tempdir().unwrap();
    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);
    std::fs::remove_file(workspace.path().join("SOUL.md")).unwrap();

    sync_workspace_file(workspace.path(), "SOUL.md", BUILTIN);

    assert_eq!(
        std::fs::read_to_string(workspace.path().join("SOUL.md")).unwrap(),
        BUILTIN
    );
}

#[test]
fn an_empty_default_writes_nothing() {
    let workspace = tempfile::tempdir().unwrap();
    sync_workspace_file(workspace.path(), "UNKNOWN.md", "");
    assert_eq!(std::fs::read_dir(workspace.path()).unwrap().count(), 0);
}

#[test]
fn a_relative_workspace_is_never_seeded() {
    // Seeding into a relative path would drop the file and its sidecar into
    // the process's current directory.
    let filename = format!("workspace-files-test-{}.md", uuid::Uuid::new_v4());
    sync_workspace_file(Path::new("."), &filename, BUILTIN);
    assert!(!Path::new(&filename).exists());
    assert!(!Path::new(&builtin_hash_sidecar_name(&filename)).exists());
}

#[test]
fn the_truncation_marker_names_the_given_read_tool() {
    // The model is told how to recover a truncated file; the tool it is told
    // to call must be one it has.
    let mut inline = String::new();
    inject_inline_content(&mut inline, "NOTES.md", "abcdefghij", 4, READ_TOOL);
    assert!(inline.contains("use `file_read` for full file"), "{inline}");

    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("NOTES.md"), "abcdefghij").unwrap();
    let mut from_disk = String::new();
    inject_workspace_file_capped(&mut from_disk, workspace.path(), "NOTES.md", 4, READ_TOOL);
    assert!(
        from_disk.contains("use `file_read` for full file"),
        "{from_disk}"
    );
}

#[test]
fn injected_content_renders_byte_exact() {
    let mut whole = String::new();
    inject_inline_content(&mut whole, "SOUL.md", "  be kind \n", 100, READ_TOOL);
    assert_eq!(whole, "### SOUL.md\n\nbe kind\n\n");

    let mut cut = String::new();
    inject_inline_content(&mut cut, "NOTES.md", "abcdefghij", 4, READ_TOOL);
    assert_eq!(
        cut,
        "### NOTES.md\n\nabcd\n\n[... truncated at 4 chars — use `file_read` for full file]\n\n"
    );

    // Exactly at the cap is not truncated.
    let mut exact = String::new();
    inject_inline_content(&mut exact, "NOTES.md", "abcd", 4, READ_TOOL);
    assert_eq!(exact, "### NOTES.md\n\nabcd\n\n");
}

#[test]
fn the_cap_counts_characters_not_bytes() {
    let mut prompt = String::new();
    inject_inline_content(&mut prompt, "NOTES.md", "héllo wörld", 5, READ_TOOL);
    assert!(prompt.starts_with("### NOTES.md\n\nhéllo\n\n[... truncated at 5 chars"));

    // Five multi-byte characters fit a five-character cap.
    let mut fits = String::new();
    inject_inline_content(&mut fits, "NOTES.md", "ééééé", 5, READ_TOOL);
    assert_eq!(fits, "### NOTES.md\n\nééééé\n\n");
}

#[test]
fn blank_content_and_missing_files_inject_nothing() {
    let mut prompt = String::from("before");
    inject_inline_content(&mut prompt, "NOTES.md", " \n\t ", 10, READ_TOOL);

    let workspace = tempfile::tempdir().unwrap();
    inject_workspace_file_capped(&mut prompt, workspace.path(), "MISSING.md", 10, READ_TOOL);
    std::fs::write(workspace.path().join("EMPTY.md"), "\n  \n").unwrap();
    inject_workspace_file_capped(&mut prompt, workspace.path(), "EMPTY.md", 10, READ_TOOL);
    // A read error other than not-found (here: a directory) is skipped too.
    std::fs::create_dir(workspace.path().join("DIR.md")).unwrap();
    inject_workspace_file_capped(&mut prompt, workspace.path(), "DIR.md", 10, READ_TOOL);

    assert_eq!(prompt, "before");
}

#[test]
fn a_synced_file_injects_under_its_name_and_truncates_at_the_cap() {
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("BIG.md"), "x".repeat(20_000 + 50)).unwrap();
    let mut prompt = String::new();
    inject_workspace_file_capped(&mut prompt, workspace.path(), "BIG.md", 20_000, READ_TOOL);
    assert!(prompt.contains("### BIG.md"));
    assert!(prompt.contains("[... truncated at 20000 chars"));
}
