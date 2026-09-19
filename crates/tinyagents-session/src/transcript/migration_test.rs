use super::*;
use std::fs;
use tempfile::TempDir;

fn write_file(path: &std::path::Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

#[test]
fn fresh_workspace_writes_marker_with_no_moves() {
    let dir = TempDir::new().unwrap();
    let outcome = migrate_layout_if_needed(dir.path()).unwrap();
    assert!(!outcome.already_done);
    assert_eq!(outcome.jsonl_moved, 0);
    assert_eq!(outcome.md_moved, 0);
    assert!(marker_path_for(dir.path()).exists());
}

#[test]
fn second_run_is_a_noop() {
    let dir = TempDir::new().unwrap();
    let _first = migrate_layout_if_needed(dir.path()).unwrap();
    let second = migrate_layout_if_needed(dir.path()).unwrap();
    assert!(second.already_done);
    assert_eq!(second.jsonl_moved, 0);
    assert_eq!(second.warnings.len(), 0);
}

#[test]
fn moves_legacy_jsonl_files_up_to_flat_session_raw() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let legacy_a = ws.join("session_raw").join("01052026");
    let legacy_b = ws.join("session_raw").join("02052026");
    write_file(&legacy_a.join("1714000000_main.jsonl"), "a");
    write_file(&legacy_a.join("1714000001_welcome.jsonl"), "b");
    write_file(&legacy_b.join("1714999999_orchestrator.jsonl"), "c");

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(outcome.jsonl_moved, 3);
    assert_eq!(outcome.legacy_dirs_pruned, 2);

    let raw_root = ws.join("session_raw");
    assert!(raw_root.join("1714000000_main.jsonl").exists());
    assert!(raw_root.join("1714000001_welcome.jsonl").exists());
    assert!(raw_root.join("1714999999_orchestrator.jsonl").exists());
    assert!(!legacy_a.exists(), "legacy date dir should be removed");
    assert!(!legacy_b.exists(), "legacy date dir should be removed");
}

#[test]
fn jsonl_destination_collision_is_skipped_with_warning() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let raw_root = ws.join("session_raw");
    write_file(&raw_root.join("1714000000_main.jsonl"), "new");
    write_file(
        &raw_root.join("01052026").join("1714000000_main.jsonl"),
        "old",
    );

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(outcome.jsonl_moved, 0);
    assert_eq!(outcome.jsonl_skipped, 1);
    assert!(
        outcome
            .warnings
            .iter()
            .any(|w| w.contains("already exists"))
    );
    assert_eq!(
        fs::read_to_string(raw_root.join("1714000000_main.jsonl")).unwrap(),
        "new"
    );
    assert_eq!(
        fs::read_to_string(raw_root.join("01052026").join("1714000000_main.jsonl")).unwrap(),
        "old"
    );
}

#[test]
fn atomic_transfer_preserves_destination_created_before_transfer() {
    let dir = TempDir::new().unwrap();
    let source = dir.path().join("legacy.jsonl");
    let destination = dir.path().join("canonical.jsonl");
    write_file(&source, "legacy contents");

    // This models another process creating the canonical file after migration
    // has discovered `source`, but before its atomic transfer operation.
    write_file(&destination, "concurrent contents");
    let error = hard_link_then_remove(&source, &destination).unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(
        fs::read_to_string(&destination).unwrap(),
        "concurrent contents"
    );
    assert_eq!(fs::read_to_string(&source).unwrap(), "legacy contents");
}

#[test]
fn renames_md_ddmmyyyy_dirs_to_iso() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let legacy_md = ws.join("sessions").join("01052026");
    write_file(&legacy_md.join("main_0.md"), "x");
    write_file(&legacy_md.join("main_1.md"), "y");

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(
        outcome.md_moved, 1,
        "a freshly reserved date directory preserves the legacy one-directory count"
    );
    let iso = ws.join("sessions").join("2026_05_01");
    assert!(iso.is_dir());
    assert!(iso.join("main_0.md").exists());
    assert!(iso.join("main_1.md").exists());
    assert!(
        !legacy_md.exists(),
        "DDMMYYYY md dir should be gone after rename"
    );
}

#[test]
fn merges_md_when_iso_dir_already_exists() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let legacy = ws.join("sessions").join("01052026");
    let iso = ws.join("sessions").join("2026_05_01");
    write_file(&legacy.join("main_0.md"), "legacy");
    write_file(&legacy.join("main_1.md"), "legacy");
    write_file(&iso.join("main_1.md"), "newer");

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(outcome.md_moved, 1);
    assert_eq!(outcome.md_skipped, 1);
    assert_eq!(fs::read_to_string(iso.join("main_0.md")).unwrap(), "legacy");
    assert_eq!(fs::read_to_string(iso.join("main_1.md")).unwrap(), "newer");
}

#[test]
fn markdown_destination_reserved_by_a_concurrent_creator_is_merged_safely() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let legacy = ws.join("sessions").join("01052026");
    let destination = ws.join("sessions").join("2026_05_01");
    write_file(&legacy.join("new.md"), "legacy file");
    write_file(&legacy.join("shared.md"), "legacy copy");

    // Model a second process reserving and populating the ISO directory after
    // this migration has discovered the legacy directory but before it can
    // reserve the destination leaf.
    fs::create_dir_all(&destination).unwrap();
    write_file(&destination.join("shared.md"), "concurrent copy");

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(outcome.md_moved, 1);
    assert_eq!(outcome.md_skipped, 1);
    assert_eq!(
        fs::read_to_string(destination.join("new.md")).unwrap(),
        "legacy file"
    );
    assert_eq!(
        fs::read_to_string(destination.join("shared.md")).unwrap(),
        "concurrent copy"
    );
    assert_eq!(
        fs::read_to_string(legacy.join("shared.md")).unwrap(),
        "legacy copy"
    );
}

#[test]
fn ignores_non_date_subdirectories_in_session_raw() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let weird = ws.join("session_raw").join("my_notes");
    write_file(&weird.join("random.jsonl"), "keep me");

    let outcome = migrate_layout_if_needed(ws).unwrap();
    assert_eq!(outcome.jsonl_moved, 0);
    assert!(weird.is_dir(), "non-date subdir must be left alone");
    assert!(weird.join("random.jsonl").exists());
}

#[test]
fn ddmmyyyy_to_iso_handles_boundary_dates() {
    assert_eq!(
        ddmmyyyy_to_yyyy_mm_dd("01012026").as_deref(),
        Some("2026_01_01")
    );
    assert_eq!(
        ddmmyyyy_to_yyyy_mm_dd("31122099").as_deref(),
        Some("2099_12_31")
    );
    assert!(ddmmyyyy_to_yyyy_mm_dd("abc12345").is_none());
    assert!(ddmmyyyy_to_yyyy_mm_dd("1234567").is_none(), "7 digits");
    assert!(ddmmyyyy_to_yyyy_mm_dd("123456789").is_none(), "9 digits");
}

#[test]
fn marker_persists_run_metadata() {
    let dir = TempDir::new().unwrap();
    let ws = dir.path();
    let legacy = ws.join("session_raw").join("01052026");
    write_file(&legacy.join("1714000000_main.jsonl"), "a");

    migrate_layout_if_needed(ws).unwrap();
    let marker = fs::read_to_string(marker_path_for(ws)).unwrap();
    assert!(marker.contains("jsonl_moved: 1"));
    assert!(marker.contains("openhuman session_layout migration v1"));
}
