//! Tests for the `[ATTACHMENT:…]` reference codec.
//!
//! The wire format is persisted in transcripts, so the golden marker below pins
//! it byte for byte; the path cases pin every shape `is_safe_relative_path`
//! must refuse, because a decoded path is later joined onto a workspace root.

use super::*;
use tinyinference_llm::message::{ImageRef, MediaRef, UserMessage};

fn upload(path: &str, name: &str, mime: &str) -> AttachmentRef {
    AttachmentRef {
        path: path.into(),
        name: name.into(),
        mime: mime.into(),
        size_bytes: 3,
    }
}

// ── marker codec ─────────────────────────────────────────────────────────────

#[test]
fn marker_encoding_is_pinned_byte_for_byte() {
    let file = upload("uploads/t/id/a b.png", "a b.png", "image/png");
    assert_eq!(
        file.marker(),
        "[ATTACHMENT:%7B%22path%22%3A%22uploads%2Ft%2Fid%2Fa+b.png%22%2C%22name%22%3A%22a+b.png\
         %22%2C%22mime%22%3A%22image%2Fpng%22%2C%22size_bytes%22%3A3%7D]"
    );
}

#[test]
fn marker_round_trips_through_parse() {
    let file = upload(
        "uploads/thread/id/Report_.zip",
        "Report α] & [x].zip",
        "application/zip",
    );
    let marker = file.marker();
    assert_eq!(marker.matches(']').count(), 1, "body never holds a `]`");
    assert_eq!(
        parse_attachment_markers(&marker),
        (String::new(), vec![file])
    );
}

#[test]
fn segments_preserve_captions_and_order_without_trimming() {
    let one = upload("uploads/t/1/one.txt", "one.txt", "text/plain");
    let two = upload("uploads/t/2/two.png", "two.png", "image/png");
    let text = format!(" first {} second {} third ", one.marker(), two.marker());
    assert_eq!(
        attachment_segments(&text),
        vec![
            AttachmentSegment::Text(" first ".into()),
            AttachmentSegment::Attachment(one.clone()),
            AttachmentSegment::Text(" second ".into()),
            AttachmentSegment::Attachment(two.clone()),
            AttachmentSegment::Text(" third ".into()),
        ]
    );
    assert_eq!(
        parse_attachment_markers(&text),
        (" first  second  third ".into(), vec![one, two])
    );
}

#[test]
fn adjacent_markers_emit_no_empty_text_segments() {
    let a = upload("a", "a", "text/plain");
    let b = upload("b", "b", "text/plain");
    let text = format!("{}{}", a.marker(), b.marker());
    assert_eq!(
        attachment_segments(&text),
        vec![
            AttachmentSegment::Attachment(a),
            AttachmentSegment::Attachment(b)
        ]
    );
    assert!(attachment_segments("").is_empty());
}

#[test]
fn malformed_and_unsafe_markers_stay_verbatim_text() {
    let unsafe_path = upload("../escape", "x", "text/plain").marker();
    for marker in [
        "[ATTACHMENT:]".to_string(),
        "[ATTACHMENT:not json]".to_string(),
        "[ATTACHMENT:%7B%22path%22%3A%22a%22%7D]".to_string(),
        unsafe_path,
    ] {
        let text = format!("a {marker} b");
        assert_eq!(
            attachment_segments(&text),
            vec![
                AttachmentSegment::Text("a ".into()),
                AttachmentSegment::Text(marker.clone()),
                AttachmentSegment::Text(" b".into()),
            ],
            "{marker}"
        );
        assert_eq!(parse_attachment_markers(&text), (text.clone(), vec![]));
    }
}

#[test]
fn unterminated_marker_ends_the_scan_verbatim() {
    assert_eq!(
        attachment_segments("x [ATTACHMENT:%7B tail"),
        vec![
            AttachmentSegment::Text("x ".into()),
            AttachmentSegment::Text("[ATTACHMENT:%7B tail".into()),
        ]
    );
}

#[test]
fn a_half_typed_marker_swallows_up_to_the_next_bracket_as_text() {
    // The first `]` closes the half-typed marker, so the valid one after it is
    // part of an undecodable body and survives only as text.
    let valid = upload("a", "a", "text/plain").marker();
    let text = format!("x [ATTACHMENT:%7B {valid} y");
    assert_eq!(
        attachment_segments(&text),
        vec![
            AttachmentSegment::Text("x ".into()),
            AttachmentSegment::Text(format!("[ATTACHMENT:%7B {valid}")),
            AttachmentSegment::Text(" y".into()),
        ]
    );
}

#[test]
fn unknown_json_fields_are_ignored() {
    let json = r#"{"path":"a","name":"n","mime":"text/plain","size_bytes":1,"extra":true}"#;
    let marker = format!(
        "{ATTACHMENT_MARKER_PREFIX}{}]",
        form_urlencoded::byte_serialize(json.as_bytes()).collect::<String>()
    );
    let (_, files) = parse_attachment_markers(&marker);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "a");
}

#[test]
fn image_markers_are_not_attachment_markers() {
    let text = "[IMAGE:uploads/a.png] [FILE:b.txt]";
    assert_eq!(parse_attachment_markers(text), (text.into(), vec![]));
}

// ── description ──────────────────────────────────────────────────────────────

#[test]
fn description_round_trips_even_with_separators_in_the_name() {
    for name in ["a.png", "weird; name; 3 bytes; workspace path: x", ""] {
        let file = upload("uploads/t/id/a.png", name, "image/png");
        let text = file.description();
        assert_eq!(AttachmentRef::from_description(&text), Some(file), "{text}");
    }
    assert_eq!(
        upload("p", "n", "m").description(),
        "[Uploaded file: n; m; 3 bytes; workspace path: p]"
    );
}

#[test]
fn from_description_rejects_non_canonical_text() {
    for text in [
        "",
        "[Uploaded file: n; m; 3 bytes; workspace path: p",
        "Uploaded file: n; m; 3 bytes; workspace path: p]",
        "[Uploaded file: n; m; three bytes; workspace path: p]",
        "[Uploaded file: n; m; +3 bytes; workspace path: p]",
        "[Uploaded file: n; m; 03 bytes; workspace path: p]",
        "[Uploaded file: n m; 3 bytes; workspace path: p]",
    ] {
        assert_eq!(AttachmentRef::from_description(text), None, "{text}");
    }
}

// ── safe relative paths ──────────────────────────────────────────────────────

#[test]
fn safe_relative_path_accepts_normal_relative_paths() {
    for path in [
        "uploads/thread/id/file.png",
        "file",
        "a//b",
        "a/./b",
        "a/",
        "uploads/..hidden/x",
    ] {
        assert!(is_safe_relative_path(path), "{path:?}");
    }
}

#[test]
fn safe_relative_path_rejects_traversal_roots_and_separators() {
    for path in [
        // OpenHuman's original cases.
        "../escape",
        "/absolute",
        "uploads\\escape",
        // Empty and current/parent components.
        "",
        ".",
        "..",
        "./a",
        "a/..",
        "a/../b",
        "uploads/../../etc/passwd",
        // Absolute and rooted forms.
        "/",
        "//server/share",
        "/etc/passwd",
        // Windows drive and UNC prefixes.
        "C:\\Windows",
        "C:\\",
        "\\\\server\\share",
        "\\\\?\\C:\\x",
        "\\absolute",
        "..\\escape",
        // NUL.
        "a\0b",
        "\0",
        "uploads/a.png\0.txt",
    ] {
        assert!(!is_safe_relative_path(path), "{path:?}");
    }
}

#[cfg(windows)]
#[test]
fn safe_relative_path_rejects_forward_slash_drive_paths_on_windows() {
    for path in ["C:/Windows", "C:x", "C:", "//?/C:/x"] {
        assert!(!is_safe_relative_path(path), "{path:?}");
    }
}

#[cfg(unix)]
#[test]
fn forward_slash_drive_path_is_an_ordinary_component_on_unix() {
    // `C:` is a valid directory name here; joining it under a root stays under
    // that root. Pinned so a platform change is a visible decision.
    assert!(is_safe_relative_path("C:/Windows"));
}

// ── filenames ────────────────────────────────────────────────────────────────

#[test]
fn filename_keeps_the_sanitized_leaf() {
    for (name, expected) in [
        ("../../same.zip", "same.zip"),
        ("dir\\sub\\b.txt", "b.txt"),
        ("héllo wörld!.txt", "héllo_wörld_.txt"),
        ("a:b*c?.png", "a_b_c_.png"),
        ("", "attachment"),
        (".", "attachment"),
        ("..", "attachment"),
        ("dir/", "attachment"),
        ("...", "..."),
        ("a\0b", "a_b"),
    ] {
        assert_eq!(sanitize_attachment_filename(name), expected, "{name:?}");
    }
}

#[test]
fn filename_caps_at_180_bytes_on_a_char_boundary() {
    let unicode_name = sanitize_attachment_filename(&"界".repeat(100));
    assert!(unicode_name.len() <= 180);
    assert!(unicode_name.len() >= 177);
    let mixed = sanitize_attachment_filename(&format!("a{}", "界".repeat(100)));
    assert_eq!(mixed.len(), 178);
    assert_eq!(sanitize_attachment_filename(&"x".repeat(500)).len(), 180);
    for name in [unicode_name, mixed] {
        assert!(is_safe_relative_path(&format!("uploads/t/id/{name}")));
    }
}

// ── message helpers ──────────────────────────────────────────────────────────

#[test]
fn enrichment_preserves_typed_media_and_image_forwarding() {
    let media = MediaRef::Path {
        path: "uploads/thread/id/media".into(),
        media_type: Some("audio/wav".into()),
    };
    let blocks = vec![
        ContentBlock::Text("caption".into()),
        ContentBlock::Image(ImageRef {
            url: "uploads/thread/id/image.png".into(),
            mime_type: None,
        }),
        ContentBlock::Audio(media.clone()),
        ContentBlock::Video(media.clone()),
        ContentBlock::Document(media),
    ];
    let input = Message::User(UserMessage {
        content: blocks.clone(),
    });
    let images = image_markers_from_message(&input, &[]);
    assert_eq!(images, vec!["[IMAGE:uploads/thread/id/image.png]"]);
    let Message::User(enriched) = enrich_user_message(input, "caption", "context\ncaption") else {
        panic!("user")
    };
    assert_eq!(&enriched.content[1..], &blocks[1..]);
    assert_eq!(
        enriched.content[0],
        ContentBlock::Text("context\ncaption".into())
    );
    assert!(!should_forward_parent_images(&images[0]));
    assert!(should_forward_parent_images(
        "analyze the user's attachments"
    ));
}

#[test]
fn enriched_plain_turn_stays_one_text_block() {
    assert_eq!(
        enrich_user_message(Message::user("caption"), "caption", "context\ncaption"),
        Message::user("context\ncaption")
    );
}

#[test]
fn enrichment_edge_cases() {
    // Not a suffix: the whole enriched text is the prefix.
    assert_eq!(
        enrich_user_message(Message::user("b"), "caption", "context"),
        Message::user("contextb")
    );
    // Nothing added: unchanged.
    assert_eq!(
        enrich_user_message(Message::user("caption"), "caption", "caption"),
        Message::user("caption")
    );
    // Media-first turn gains a new leading text block.
    let image = ContentBlock::Image(ImageRef {
        url: "uploads/a.png".into(),
        mime_type: None,
    });
    let input = Message::User(UserMessage {
        content: vec![image.clone()],
    });
    assert_eq!(
        enrich_user_message(input, "", "context"),
        Message::User(UserMessage {
            content: vec![ContentBlock::Text("context".into()), image],
        })
    );
    // Non-user messages pass through.
    assert_eq!(
        enrich_user_message(Message::assistant("hi"), "", "context"),
        Message::assistant("hi")
    );
}

#[test]
fn image_markers_keep_legacy_first_and_skip_duplicates() {
    let image = |url: &str| {
        ContentBlock::Image(ImageRef {
            url: url.into(),
            mime_type: Some("image/png".into()),
        })
    };
    let input = Message::User(UserMessage {
        content: vec![image("a.png"), image("b.png"), image("a.png")],
    });
    let legacy = vec![
        "[Image: old #att:1]".to_string(),
        "[IMAGE:b.png]".to_string(),
    ];
    assert_eq!(
        image_markers_from_message(&input, &legacy),
        vec!["[Image: old #att:1]", "[IMAGE:b.png]", "[IMAGE:a.png]"]
    );
    assert_eq!(
        image_markers_from_message(&Message::assistant("x"), &legacy),
        legacy
    );
}

#[test]
fn explicit_image_selection_replaces_parent_forwarding() {
    let image = upload("uploads/t/id/a.png", "a.png", "image/png").marker();
    let doc = upload("uploads/t/id/a.pdf", "a.pdf", "application/pdf").marker();
    assert!(!should_forward_parent_images(&format!("look {image}")));
    assert!(should_forward_parent_images(&format!("read {doc}")));
    assert!(!should_forward_parent_images("see [Image: photo #att:abc]"));
    assert!(!should_forward_parent_images("see [IMAGE:/tmp/a.png]"));
    assert!(should_forward_parent_images("see [IMAGE:]"));
    assert!(should_forward_parent_images(
        "[Image: (could not be processed)]"
    ));
}
