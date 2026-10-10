//! Unit tests for `build_stdin`: new-session vs. resume piping, pending-user
//! coalescing, preamble replay, and image-marker splitting.

use super::*;

fn msg(role: &str, content: &str) -> ChatMessage {
    match role {
        "system" => ChatMessage::system(content),
        "user" => ChatMessage::user(content),
        "assistant" => ChatMessage::assistant(content),
        _ => ChatMessage::tool(content),
    }
}

/// Every row's `message.role` must be `"user"`. The CLI validates this
/// before invoking the model and exits 1 with
/// `Expected message role 'user', got 'assistant'` otherwise (#5711).
/// Verified against Claude Code CLI 2.1.221.
fn assert_every_row_is_a_user_role(payload: &str) {
    for (i, line) in payload.lines().enumerate() {
        let row: Value = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!(
                "row {i} is not valid JSON: {e}
{line}"
            )
        });
        assert_eq!(
            row["message"]["role"], "user",
            "row {i} must carry role=user, the only role CC stdin accepts
{line}"
        );
        assert_eq!(row["type"], "user", "row {i} envelope type");
    }
}

#[test]
fn new_session_never_emits_an_assistant_role() {
    let history = vec![
        msg("system", "you are helpful"),
        msg("user", "first user"),
        msg("assistant", "prior assistant"),
        msg("user", "latest user"),
    ];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    assert_every_row_is_a_user_role(&s);
    assert!(
        !s.contains("\"role\":\"assistant\""),
        "an assistant role row is what the CLI rejects:
{s}"
    );
}

#[test]
fn new_session_carries_prior_turns_as_one_labelled_transcript() {
    let history = vec![
        msg("system", "you are helpful"),
        msg("user", "hi"),
        msg("assistant", "hello"),
        msg("user", "how are you?"),
    ];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    let lines: Vec<_> = s.lines().collect();

    // The transcript and latest prompt are content blocks in one user row.
    assert_eq!(
        lines.len(),
        1,
        "got:
{s}"
    );
    assert!(lines[0].contains("User: hi"));
    assert!(lines[0].contains("Assistant: hello"));
    assert!(
        !lines[0].contains("you are helpful"),
        "the system message must not leak into the transcript"
    );

    // The prompt itself is passed through untouched, not folded in.
    let latest: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(latest["message"]["content"][1]["text"], "how are you?");
}

#[test]
fn a_history_ending_on_an_assistant_turn_is_all_context() {
    // Switching an existing conversation to the CC provider can leave the
    // assistant speaking last; none of it is a fresh instruction.
    let history = vec![msg("user", "hi"), msg("assistant", "hello")];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "got:
{s}"
    );
    assert_every_row_is_a_user_role(&s);
    assert!(lines[0].contains("User: hi"));
    assert!(lines[0].contains("Assistant: hello"));
}

#[test]
fn a_single_user_turn_is_sent_verbatim_with_no_transcript() {
    let history = vec![msg("user", "just this")];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 1);
    let row: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(row["message"]["content"][0]["text"], "just this");
}

#[test]
fn resume_pipes_the_pending_user_turn() {
    let history = vec![
        msg("user", "earlier turn"),
        msg("assistant", "earlier reply"),
        msg("user", "follow-up"),
    ];
    let bytes = build_stdin(&history, false);
    let s = String::from_utf8(bytes).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("\"follow-up\""));
    assert_every_row_is_a_user_role(&s);
}

#[test]
fn resume_combines_all_pending_user_turns_in_order() {
    let history = vec![
        msg("user", "earlier turn"),
        msg("assistant", "earlier reply"),
        msg("user", "first steering"),
        msg("user", "second steering"),
    ];
    let s = String::from_utf8(build_stdin(&history, false)).unwrap();
    assert!(s.contains("first steering\\n\\nsecond steering"), "{s}");
    assert!(!s.contains("earlier turn"), "answered history leaked: {s}");
}

#[test]
fn empty_history_yields_empty_bytes() {
    let bytes = build_stdin(&[], true);
    assert!(bytes.is_empty());
}

#[test]
fn image_blocks_preserve_text_order() {
    let s = String::from_utf8(build_stdin(
        &[ChatMessage::user(
            "before [OH_IMAGE:data:image/png;base64,QUJD] between [OH_IMAGE:data:image/gif;base64,R0lG] after",
        )],
        true,
    ))
    .unwrap();
    let row: Value = serde_json::from_str(s.lines().next().unwrap()).unwrap();
    let content = row["message"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 5);
    assert_eq!(content[0]["text"], "before ");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[2]["text"], " between ");
    assert_eq!(content[3]["type"], "image");
    assert_eq!(content[4]["text"], " after");
}

#[test]
fn literal_native_image_marker_stays_text() {
    let s = String::from_utf8(build_stdin(
        &[ChatMessage::user(
            "literal [OH_IMAGE:data:image/png;base64,QUJD] then [OH_IMAGE_LITERAL:data:image/png;base64,REVG]",
        )],
        true,
    ))
    .unwrap();
    let row: Value = serde_json::from_str(s.lines().next().unwrap()).unwrap();
    let content = row["message"]["content"].as_array().unwrap();
    assert_eq!(content[0]["text"], "literal ");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[2]["text"], " then ");
    assert_eq!(content[3]["text"], "[OH_IMAGE:data:image/png;base64,REVG]");
}

#[test]
fn literal_file_marker_is_not_read() {
    let s = String::from_utf8(build_stdin(
        &[ChatMessage::user("read [IMAGE:/etc/hostname]")],
        true,
    ))
    .unwrap();
    assert!(!s.contains("\"type\":\"image\""), "{s}");
    assert!(s.contains("[IMAGE:/etc/hostname]"), "{s}");
}

#[test]
fn new_session_history_preserves_answered_user_images() {
    let history = vec![
        ChatMessage::user("earlier [OH_IMAGE:data:image/png;base64,QUJD]"),
        ChatMessage::assistant("old answer"),
        ChatMessage::user("latest"),
    ];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    let row: Value = serde_json::from_str(s.lines().next().unwrap()).unwrap();
    let content = row["message"]["content"].as_array().unwrap();
    assert!(content.iter().any(|block| block["type"] == "image"));
    assert!(content.iter().any(|block| {
        block["text"]
            .as_str()
            .is_some_and(|text| text.contains("User: earlier "))
    }));
    assert!(s.contains("Assistant: old answer"));
}

#[test]
fn consecutive_answered_user_turns_are_all_in_preamble() {
    let history = vec![
        ChatMessage::user("first steering"),
        ChatMessage::user("second steering"),
        ChatMessage::assistant("answer"),
        ChatMessage::user("latest"),
    ];
    let s = String::from_utf8(build_stdin(&history, true)).unwrap();
    assert!(s.contains("User: first steering"));
    assert!(s.contains("User: second steering"));
}

#[test]
fn unterminated_image_marker_preserves_trailing_text() {
    let s = String::from_utf8(build_stdin(
        &[ChatMessage::user(
            "before [OH_IMAGE:data:image/png;base64,QUJD after",
        )],
        true,
    ))
    .unwrap();
    assert!(
        s.contains("before [OH_IMAGE:data:image/png;base64,QUJD after"),
        "{s}"
    );
}

#[test]
fn invalid_native_images_use_the_text_fallback() {
    let s = String::from_utf8(build_stdin(
        &[ChatMessage::user(
            "bad [OH_IMAGE:data:image/svg+xml;base64,PHN2Zz4=] and [OH_IMAGE:data:image/png;base64,not-base64]",
        )],
        true,
    ))
    .unwrap();
    assert!(!s.contains("\"type\":\"image\""), "{s}");
    assert!(s.contains("an attached image could not be read"), "{s}");
}

// ---- delivered-turn tracking on resume (openhuman#5877) ----

/// Text blocks of the single user message in `bytes`, one entry per block.
fn emitted_blocks(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8(bytes.to_vec()).expect("utf8");
    let line: Value = serde_json::from_str(text.trim()).expect("one json line");
    line["message"]["content"]
        .as_array()
        .expect("content blocks")
        .iter()
        .filter_map(|b| b["text"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn resume_skips_pending_turn_the_session_already_received() {
    let history = [ChatMessage::user("ship it")];
    let delivered: HashSet<String> = pending_fingerprints(&history).into_iter().collect();

    let out = emitted_blocks(&build_stdin_with_delivered(&history, false, &delivered));

    assert_eq!(out, vec![ALREADY_DELIVERED_NOTICE.to_string()]);
}

#[test]
fn resume_sends_only_the_turns_not_yet_delivered() {
    let first = [ChatMessage::user("first")];
    let delivered: HashSet<String> = pending_fingerprints(&first).into_iter().collect();
    let queued = [ChatMessage::user("first"), ChatMessage::user("second")];

    let out = emitted_blocks(&build_stdin_with_delivered(&queued, false, &delivered));

    assert_eq!(out, vec!["second".to_string()]);
}

#[test]
fn same_words_after_a_new_reply_are_a_new_turn() {
    let first = [ChatMessage::user("continue")];
    let delivered: HashSet<String> = pending_fingerprints(&first).into_iter().collect();
    let again = [
        ChatMessage::user("continue"),
        ChatMessage::assistant("step one done"),
        ChatMessage::user("continue"),
    ];

    let out = emitted_blocks(&build_stdin_with_delivered(&again, false, &delivered));

    assert_eq!(out, vec!["continue".to_string()]);
}

#[test]
fn identical_replies_do_not_collapse_repeated_turns() {
    let first = [
        ChatMessage::user("continue"),
        ChatMessage::assistant("Done"),
        ChatMessage::user("continue"),
    ];
    let second = [
        ChatMessage::user("continue"),
        ChatMessage::assistant("Done"),
        ChatMessage::user("continue"),
        ChatMessage::assistant("Done"),
        ChatMessage::user("continue"),
    ];
    let delivered: HashSet<String> = pending_fingerprints(&first).into_iter().collect();
    assert_ne!(pending_fingerprints(&first), pending_fingerprints(&second));
    let out = emitted_blocks(&build_stdin_with_delivered(&second, false, &delivered));
    assert_eq!(out, vec!["continue".to_string()]);
}

#[test]
fn fingerprints_are_stable_when_an_earlier_pending_turn_is_absent() {
    let both = [ChatMessage::user("one"), ChatMessage::user("two")];
    let only_second = [ChatMessage::user("two")];
    assert_eq!(
        pending_fingerprints(&both)[1],
        pending_fingerprints(&only_second)[0]
    );
}

#[test]
fn repeated_text_in_one_pending_batch_stays_distinct() {
    let fps = pending_fingerprints(&[ChatMessage::user("go"), ChatMessage::user("go")]);
    assert_ne!(fps[0], fps[1]);
}

#[test]
fn fingerprints_ignore_unrelated_history_position() {
    let short = [ChatMessage::assistant("ok"), ChatMessage::user("go")];
    let long = [
        ChatMessage::user("older"),
        ChatMessage::assistant("ok"),
        ChatMessage::user("go"),
    ];
    assert_eq!(pending_fingerprints(&short), pending_fingerprints(&long));
}

#[test]
fn new_session_ignores_the_delivered_set() {
    let history = [ChatMessage::user("hello")];
    let delivered: HashSet<String> = pending_fingerprints(&history).into_iter().collect();
    let out = emitted_blocks(&build_stdin_with_delivered(&history, true, &delivered));
    assert_eq!(out, vec!["hello".to_string()]);
}

#[test]
fn no_fingerprints_when_history_does_not_end_on_a_user_turn() {
    let history = [ChatMessage::user("hi"), ChatMessage::assistant("hello")];
    assert!(pending_fingerprints(&history).is_empty());
}
