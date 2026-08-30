//! Tests for the prompt-guided tool-call protocol.

use super::*;
use crate::message::{ContentBlock, ImageRef, Message};
use crate::model::ModelResponse;

fn schema(name: &str) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        description: format!("{name} description"),
        parameters: serde_json::json!({"type": "object"}),
        format: Default::default(),
    }
}

#[test]
fn prompt_instructions_list_each_tool() {
    let text = prompt_tool_instructions(&[schema("read_file"), schema("write_file")]);
    assert!(text.contains("## Tool Use Protocol"));
    assert!(text.contains("<tool_call>"));
    assert!(text.contains("**read_file**"));
    assert!(text.contains("**write_file**"));
}

#[test]
fn prompt_instructions_append_to_system() {
    let msgs = vec![Message::system("You are helpful."), Message::user("hi")];
    let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
    assert_eq!(out.len(), 2);
    let Message::System(system) = &out[0] else {
        panic!("first message should stay system")
    };
    let joined: String = system
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(joined.contains("You are helpful."));
    assert!(joined.contains("Tool Use Protocol"));
}

#[test]
fn prompt_instructions_insert_system_when_absent() {
    let msgs = vec![Message::user("hi")];
    let out = with_prompt_tool_instructions(&msgs, &[schema("read_file")]);
    assert_eq!(out.len(), 2);
    assert!(matches!(out[0], Message::System(_)));
}

#[test]
fn empty_tools_leave_messages_unchanged() {
    let msgs = vec![Message::user("hi")];
    assert_eq!(with_prompt_tool_instructions(&msgs, &[]), msgs);
}

#[test]
fn prompt_results_coalesce_consecutive_tool_messages() {
    let messages = vec![
        Message::user("question"),
        Message::assistant("calling tools"),
        Message::tool("call-1", "first"),
        Message::tool("call-2", "second"),
        Message::assistant("done"),
    ];

    let out = coalesce_prompt_tool_results(&messages);

    assert_eq!(out.len(), 4);
    assert!(matches!(out[0], Message::User(_)));
    assert!(matches!(out[1], Message::Assistant(_)));
    assert!(matches!(out[2], Message::User(_)));
    assert_eq!(
        out[2].text(),
        "[Tool results]\n<tool_result>\nfirst\n</tool_result>\n<tool_result>\nsecond\n</tool_result>"
    );
    assert!(matches!(out[3], Message::Assistant(_)));
}

#[test]
fn prompt_result_coalescing_without_tools_is_identity() {
    let messages = vec![Message::system("system"), Message::user("question")];
    assert_eq!(coalesce_prompt_tool_results(&messages), messages);
}

#[test]
fn user_turn_normalization_leaves_a_real_query_alone() {
    let messages = vec![
        Message::system("system"),
        Message::user("question"),
        Message::assistant("answer"),
    ];
    assert_eq!(ensure_resolvable_user_turn(&messages), messages);
}

#[test]
fn user_turn_normalization_inserts_after_leading_system_turns() {
    // openhuman#5291: the real user turn aged out of the window, leaving a
    // system prompt and an assistant continuation. Qwen 3's template raises
    // `No user query found in messages.` on exactly this shape.
    let messages = vec![
        Message::system("system"),
        Message::system("tool protocol"),
        Message::assistant("continuing"),
    ];

    let out = ensure_resolvable_user_turn(&messages);

    assert_eq!(out.len(), 4);
    assert!(matches!(out[0], Message::System(_)));
    assert!(matches!(out[1], Message::System(_)));
    assert!(matches!(out[2], Message::User(_)), "user turn is inserted");
    assert!(matches!(out[3], Message::Assistant(_)));
}

#[test]
fn user_turn_normalization_does_not_count_folded_tool_results() {
    // The only user-role turns are coalesced tool results, which is not a query
    // the template can answer — the model asked for those itself.
    let coalesced = coalesce_prompt_tool_results(&[
        Message::system("system"),
        Message::assistant("calling"),
        Message::tool("call-1", "result"),
    ]);
    assert!(
        coalesced.iter().any(|m| matches!(m, Message::User(_))),
        "coalescing produces a user-role turn"
    );

    let out = ensure_resolvable_user_turn(&coalesced);

    assert_eq!(out.len(), coalesced.len() + 1);
    assert!(matches!(out[1], Message::User(_)));
    assert!(!out[1].text().starts_with("[Tool results]"));
}

#[test]
fn user_turn_normalization_ignores_a_blank_user_turn() {
    let messages = vec![Message::system("system"), Message::user("   ")];
    let out = ensure_resolvable_user_turn(&messages);
    assert_eq!(out.len(), 3);
    assert!(!out[1].text().trim().is_empty());
}

#[test]
fn user_turn_normalization_accepts_a_non_text_user_turn() {
    // An image-only turn carries no text but is still a real user input.
    let mut messages = vec![Message::system("system"), Message::user("")];
    let Message::User(user) = &mut messages[1] else {
        unreachable!()
    };
    user.content = vec![ContentBlock::Image(ImageRef {
        url: "https://example.invalid/a.png".to_string(),
        mime_type: None,
    })];

    assert_eq!(ensure_resolvable_user_turn(&messages), messages);
}

#[test]
fn user_turn_normalization_inserts_first_when_there_is_no_system_turn() {
    let messages = vec![Message::assistant("continuing")];
    let out = ensure_resolvable_user_turn(&messages);
    assert_eq!(out.len(), 2);
    assert!(matches!(out[0], Message::User(_)));
}

#[test]
fn prompt_replay_renders_assistant_calls_before_results() {
    let mut assistant = Message::assistant("I will inspect both files.");
    let Message::Assistant(message) = &mut assistant else {
        unreachable!()
    };
    message.tool_calls = vec![
        ToolCall::new("call-1", "read_file", serde_json::json!({"path":"a.txt"})),
        ToolCall::new("call-2", "read_file", serde_json::json!({"path":"b.txt"})),
    ];
    let messages = vec![
        Message::user("compare them"),
        assistant,
        Message::tool("call-1", "first"),
        Message::tool("call-2", "second"),
    ];

    let out = coalesce_prompt_tool_results(&messages);

    let Message::Assistant(replayed) = &out[1] else {
        panic!("assistant call turn should remain an assistant turn")
    };
    assert!(replayed.tool_calls.is_empty());
    assert!(out[1].text().contains("I will inspect both files."));
    assert!(
        out[1].text().contains(
            r#"<tool_call>{"arguments":{"path":"a.txt"},"name":"read_file"}</tool_call>"#
        )
    );
    assert!(
        out[1].text().contains(
            r#"<tool_call>{"arguments":{"path":"b.txt"},"name":"read_file"}</tool_call>"#
        )
    );
    assert!(
        out[2]
            .text()
            .contains("<tool_result>\nfirst\n</tool_result>")
    );
    assert!(
        out[2]
            .text()
            .contains("<tool_result>\nsecond\n</tool_result>")
    );
}

#[test]
fn prompt_parser_extracts_single_tool_call() {
    let text = r#"Let me read it.
<tool_call>
{"name": "read_file", "arguments": {"path": "a.txt"}}
</tool_call>"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(cleaned, "Let me read it.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read_file");
    // Ids are process-unique, not a per-response index: only the shape and the
    // slot suffix are stable. See `next_synthetic_call_id`.
    assert!(
        calls[0]
            .id
            .starts_with(&format!("{SYNTHETIC_CALL_ID_PREFIX}_")),
        "unexpected synthetic id {}",
        calls[0].id
    );
    assert!(
        calls[0].id.ends_with("_1"),
        "slot suffix lost: {}",
        calls[0].id
    );
    assert_eq!(calls[0].arguments, serde_json::json!({"path": "a.txt"}));
}

#[test]
fn prompt_parser_extracts_multiple_calls_and_keeps_prose() {
    let text = r#"a<tool_call>{"name":"one","arguments":{}}</tool_call>b<tool_call>{"name":"two","arguments":{"x":1}}</tool_call>c"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(cleaned, "abc");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "one");
    assert_eq!(calls[1].name, "two");
    assert!(
        calls[1].id.ends_with("_2"),
        "slot suffix lost: {}",
        calls[1].id
    );
    assert_ne!(calls[0].id, calls[1].id);
}

#[test]
fn prompt_parser_defaults_missing_arguments_to_empty_object() {
    let (_, calls) =
        parse_prompt_tool_calls_from_text(r#"<tool_call>{"name":"noargs"}</tool_call>"#);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, serde_json::json!({}));
}

#[test]
fn prompt_parser_drops_malformed_block() {
    let (cleaned, calls) = parse_prompt_tool_calls_from_text("<tool_call>not json</tool_call>done");
    assert!(calls.is_empty());
    assert_eq!(cleaned, "done");
}

#[test]
fn prompt_parser_keeps_unterminated_block_as_text() {
    let text = "text <tool_call>{\"name\":\"x\"}";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, "text <tool_call>{\"name\":\"x\"}");
}

#[test]
fn prompt_parser_returns_plain_text_verbatim() {
    let (cleaned, calls) = parse_prompt_tool_calls_from_text("just a normal answer");
    assert!(calls.is_empty());
    assert_eq!(cleaned, "just a normal answer");
}

// --- Attribute / variant-tolerant matching (Hermes / DeepSeek templates) ---

#[test]
fn prompt_parser_matches_attribute_form_open_tag() {
    // Regression for the exact-literal miss: `<tool_call id="…">` must match so a
    // native model that leaks the call as text doesn't dump raw markup.
    let text = r#"<tool_call id="call_0">{"name":"foo","arguments":{"a":1}}</tool_call>"#;
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
    assert_eq!(calls[0].arguments, serde_json::json!({"a": 1}));
    assert!(cleaned.is_empty());
    assert!(!cleaned.contains("<tool_call"));
}

#[test]
fn prompt_parser_matches_pipe_variant_open_tag() {
    let text = r#"<tool_call|>{"name":"foo","arguments":{}}</tool_call>"#;
    let (_, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
}

#[test]
fn prompt_parser_matches_deepseek_delimiters() {
    let text = "<｜tool▁call▁begin｜>{\"name\":\"foo\",\"arguments\":{}}<｜tool▁call▁end｜>";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "foo");
    assert!(cleaned.is_empty());
}

#[test]
fn prompt_parser_drops_attribute_form_no_name_body_without_leak() {
    // The reported bug: `<tool_call id="call_0">{}</tool_call>` has no `name`.
    // It must be dropped — never echoed back as assistant content.
    let text = "prefix <tool_call id=\"call_0\">\n{}\n</tool_call> suffix";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert!(!cleaned.contains("<tool_call"));
    assert!(!cleaned.contains("{}"));
    assert_eq!(cleaned, "prefix  suffix");
}

#[test]
fn prompt_parser_does_not_match_plural_tag() {
    // `<tool_calls>` must not be mistaken for an opening `<tool_call>`.
    let text = "see the <tool_calls> list";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, "see the <tool_calls> list");
}

#[test]
fn prompt_parser_does_not_misparse_prose_open_tag_without_close() {
    let text = "You can emit a <tool_call id=\"1\"> block to call a tool.";
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(text);
    assert!(calls.is_empty());
    assert_eq!(cleaned, text);
}

// --- apply_prompt_tool_calls recovery + native-mode fallback gating ---

#[test]
fn apply_prompt_tool_calls_recovers_attribute_markup() {
    // A native model emitted the call as text with EMPTY structured tool_calls:
    // recovery yields a structured call and the raw markup does NOT survive.
    let resp = crate::model::ModelResponse::assistant(
        r#"<tool_call id="call_0">{"name":"foo","arguments":{}}</tool_call>"#,
    );
    let out = apply_prompt_tool_calls(resp);
    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "foo");
    assert!(!out.text().contains("<tool_call"));
    assert!(out.text().is_empty());
}

#[test]
fn should_recover_gating_matrix() {
    // Prompt-guided (native == false): always recover when tools were offered.
    assert!(should_recover(false, true, 0));
    assert!(should_recover(false, true, 2));
    // Native fallback: only when tools offered AND no structured calls.
    assert!(should_recover(true, true, 0));
    // Native with structured calls present: never recover (path stays as-is).
    assert!(!should_recover(true, true, 1));
    // No tools offered: nothing to recover, either mode.
    assert!(!should_recover(true, false, 0));
    assert!(!should_recover(false, false, 0));
}

#[test]
fn should_recover_skips_native_response_with_structured_calls() {
    // When a real structured call is present the guard skips recovery, so the
    // native path is left byte-for-byte unchanged.
    let calls = [ToolCall {
        id: "call_1".to_string(),
        name: "real".to_string(),
        arguments: serde_json::json!({}),
        invalid: None,
    }];
    assert!(!should_recover(true, true, calls.len()));
}

/// Feeds every fragment through the scrubber and appends the flush, returning the
/// full text a delta consumer would render.
fn scrub_all(fragments: &[&str]) -> String {
    let mut s = ToolCallStreamScrubber::new();
    let mut out = String::new();
    for f in fragments {
        out.push_str(&s.feed(f));
    }
    out.push_str(&s.flush());
    out
}

#[test]
fn scrubber_passes_plain_text_through_unchanged() {
    // No markup: the concatenated emissions equal the input exactly.
    assert_eq!(scrub_all(&["hello ", "world", " done"]), "hello world done");
}

#[test]
fn scrubber_drops_a_complete_block_in_one_fragment() {
    let out = scrub_all(&[r#"before <tool_call>{"name":"x","arguments":{}}</tool_call> after"#]);
    assert_eq!(out, "before  after");
}

#[test]
fn scrubber_suppresses_markup_split_across_fragments() {
    // The open tag, body, and close arrive in separate fragments — no raw markup
    // may appear in any emission, and the surrounding prose survives.
    let out = scrub_all(&[
        "answer: ",
        "<tool_",
        "call>{\"name\":\"x\",",
        "\"arguments\":{}}</tool",
        "_call> end",
    ]);
    assert_eq!(out, "answer:  end");
    assert!(!out.contains("<tool_call"));
}

#[test]
fn scrubber_never_emits_partial_open_marker_mid_stream() {
    // A fragment ending in a partial open marker must not surface it; it is held
    // until the next fragment resolves the block.
    let mut s = ToolCallStreamScrubber::new();
    let first = s.feed("value <tool");
    assert_eq!(first, "value ", "partial `<tool` must be held, not emitted");
    let second = s.feed("_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>!");
    assert_eq!(second, "!");
    assert_eq!(s.flush(), "");
}

#[test]
fn scrubber_handles_attribute_open_form_split() {
    // The Hermes/DeepSeek attribute form `<tool_call id="...">` split mid-tag.
    let out = scrub_all(&[
        "ok ",
        "<tool_call id=\"call_0\"",
        ">{\"name\":\"x\",\"arguments\":{}}</tool_call>",
    ]);
    assert_eq!(out, "ok ");
}

#[test]
fn scrubber_handles_deepseek_delimiters_split() {
    let out = scrub_all(&[
        "r ",
        "<｜tool▁call▁be",
        "gin｜>{\"name\":\"x\",\"arguments\":{}}<｜tool▁call▁end｜>",
        " s",
    ]);
    assert_eq!(out, "r  s");
    assert!(!out.contains("tool▁call"));
}

#[test]
fn scrubber_does_not_hold_plural_tool_calls_prose() {
    // `<tool_calls>` (name not delimiter-terminated) is prose, not an open tag.
    assert_eq!(
        scrub_all(&["see <tool_calls> below"]),
        "see <tool_calls> below"
    );
}

#[test]
fn scrubber_flush_surfaces_a_dangling_open_verbatim_untrimmed() {
    // A `<tool_call` with no close is real text once the stream ends; leading and
    // trailing whitespace is preserved (flush does not trim).
    let mut s = ToolCallStreamScrubber::new();
    let mid = s.feed("  a <tool_call ");
    assert_eq!(mid, "  a ", "the in-progress open tag is held");
    assert_eq!(s.flush(), "<tool_call ");
}

#[test]
fn scrubber_matches_batch_parser_on_the_visible_text() {
    // Property: scrubbing a fragmented stream yields the same visible text as the
    // batch parser produces (modulo the batch parser's trim) for the same input.
    let full = r#"lead <tool_call>{"name":"a","arguments":{}}</tool_call> mid <tool_call>{"name":"b","arguments":{"k":1}}</tool_call> tail"#;
    let (batch, calls) = parse_prompt_tool_calls_from_text(full);
    assert_eq!(calls.len(), 2);
    // Fragment the input into single-byte-ish chunks at char boundaries.
    let frags: Vec<String> = full.chars().map(|c| c.to_string()).collect();
    let refs: Vec<&str> = frags.iter().map(String::as_str).collect();
    assert_eq!(scrub_all(&refs).trim(), batch);
}

#[test]
fn apply_prompt_tool_calls_preserves_a_leading_thinking_block() {
    // A prompt-guided reasoning model emits a `Thinking` block followed by the
    // `<tool_call>` text. Recovering the call must not discard the reasoning.
    let mut response = ModelResponse::assistant(
        r#"reply <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>"#,
    );
    response.message.content.insert(
        0,
        ContentBlock::Thinking {
            text: "chain of thought".to_string(),
            signature: None,
        },
    );

    let out = apply_prompt_tool_calls(response);

    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "search");
    assert_eq!(
        out.message.content[0],
        ContentBlock::Thinking {
            text: "chain of thought".to_string(),
            signature: None,
        },
        "the thinking block must survive the content rebuild"
    );
    assert_eq!(
        out.message.content[1],
        ContentBlock::Text("reply".to_string())
    );
}
// ---------------------------------------------------------------------------
// Bare (undelimited) tool calls
//
// Captured from `llama3.2:3b` via Ollama with `tool_choice: "required"`: the
// model puts the call in `content` instead of the wire's `tool_calls` array,
// with no `<tool_call>` markup and frequently with malformed JSON.
// ---------------------------------------------------------------------------

#[test]
fn apply_prompt_tool_calls_recovers_a_bare_object_with_relaxed_json() {
    // The exact capture: `parameters'` and `{'city'` use mismatched quotes, so
    // strict JSON rejects it outright.
    let resp = crate::model::ModelResponse::assistant(
        r#"{"name":"get_weather","parameters':{'city':"Paris"}}"#,
    );
    let out = apply_prompt_tool_calls(resp);

    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "get_weather");
    assert_eq!(
        out.message.tool_calls[0].arguments,
        serde_json::json!({ "city": "Paris" })
    );
    // The raw markup must not also survive as prose, or the user sees the JSON.
    assert!(
        out.text().is_empty(),
        "the consumed object should not remain as text: {}",
        out.text()
    );
}

#[test]
fn apply_prompt_tool_calls_recovers_a_bare_object_inside_a_code_fence() {
    let resp = crate::model::ModelResponse::assistant(
        "```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n```",
    );
    let out = apply_prompt_tool_calls(resp);

    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "get_weather");
}

#[test]
fn a_tool_call_object_may_name_its_arguments_parameters() {
    let resp = crate::model::ModelResponse::assistant(
        r#"<tool_call>{"name":"get_weather","parameters":{"city":"Paris"}}</tool_call>"#,
    );
    let out = apply_prompt_tool_calls(resp);

    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(
        out.message.tool_calls[0].arguments,
        serde_json::json!({ "city": "Paris" })
    );
}

#[test]
fn bare_object_recovery_never_swallows_a_genuine_text_answer() {
    // Prose, prose that merely quotes JSON, a JSON object that names no tool,
    // and a bare JSON scalar must all pass through untouched.
    for text in [
        "The weather in Paris is mild today.",
        r#"You could send {"name":"get_weather"} to that endpoint."#,
        r#"{"city":"Paris","temperature":17}"#,
        r#"{"name":42}"#,
        r#""just a string""#,
        "[1, 2, 3]",
    ] {
        let out = apply_prompt_tool_calls(crate::model::ModelResponse::assistant(text));
        assert!(
            out.message.tool_calls.is_empty(),
            "{text:?} must not be recovered as a tool call"
        );
        assert_eq!(out.text(), text, "{text:?} must survive as text");
    }
}

#[test]
fn bare_tool_call_recovery_preserves_a_thinking_block() {
    // A local *reasoning* model emits its chain of thought and then the bare
    // call object as the whole visible text. Consuming the object must not take
    // the reasoning with it.
    let mut response = ModelResponse::assistant(r#"{"name":"search","arguments":{"q":"x"}}"#);
    response.message.content.insert(
        0,
        ContentBlock::Thinking {
            text: "chain of thought".to_string(),
            signature: None,
        },
    );

    let out = apply_prompt_tool_calls(response);

    assert_eq!(out.message.tool_calls.len(), 1);
    assert_eq!(out.message.tool_calls[0].name, "search");
    assert_eq!(
        out.message.content,
        vec![ContentBlock::Thinking {
            text: "chain of thought".to_string(),
            signature: None,
        }],
        "the reasoning must survive while the consumed object does not"
    );
}

/// TOOL-2: two turns of the same run must not both mint `call_1`.
///
/// The recovered id used to be the call's index *within one response*, which
/// resets every turn. A two-turn run therefore produced a transcript with two
/// assistant messages declaring the same tool-call id and two tool messages
/// answering it — a pairing no provider (and no pairing repair) can resolve.
#[test]
fn synthetic_call_ids_are_unique_across_responses() {
    let text = r#"<tool_call>{"name":"one","arguments":{}}</tool_call>"#;
    let (_, first) = parse_prompt_tool_calls_from_text(text);
    let (_, second) = parse_prompt_tool_calls_from_text(text);

    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_ne!(
        first[0].id, second[0].id,
        "a second turn reused the first turn's synthetic tool-call id"
    );
}

/// The synthetic scheme must be visibly distinct from real provider ids and
/// from the OpenAI adapter's own positional fallback (`tool-{slot}`), so the
/// two can never collide.
#[test]
fn synthetic_call_ids_do_not_look_like_provider_ids() {
    let id = next_synthetic_call_id(1);
    assert!(id.starts_with("ptc_"), "{id}");
    assert!(!id.starts_with("call_"), "{id}");
    assert!(!id.starts_with("tool-"), "{id}");
}

/// The bare-object recovery path mints ids from the same counter, so a model
/// that alternates between markup and bare objects still cannot collide.
#[test]
fn bare_object_recovery_also_mints_unique_ids() {
    let body = r#"{"name":"one","arguments":{}}"#;
    let first = apply_prompt_tool_calls(ModelResponse::assistant(body));
    let second = apply_prompt_tool_calls(ModelResponse::assistant(body));

    let first_id = &first.message.tool_calls[0].id;
    let second_id = &second.message.tool_calls[0].id;
    assert_ne!(first_id, second_id);
}
