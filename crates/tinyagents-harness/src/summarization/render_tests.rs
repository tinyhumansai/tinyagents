//! Tests for [`render_message_for_summary_with`]: the [`ToolRendering::Full`]
//! mode matches [`render_message_for_summary`], and [`ToolRendering::Markers`]
//! keeps the conversation while reducing tools to one-line markers.

use super::*;
use crate::summarization::ToolRendering;
use serde_json::json;
use tinyinference_llm::message::AssistantMessage;
use tinyinference_llm::tool::ToolCall;

fn assistant(text: Option<&str>, calls: Vec<ToolCall>) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: text
            .map(|t| ContentBlock::Text(t.into()))
            .into_iter()
            .collect(),
        tool_calls: calls,
        usage: None,
        origin: None,
    })
}

#[test]
fn full_rendering_matches_the_default_renderer_for_every_message_kind() {
    let messages = vec![
        Message::system("be brief"),
        Message::user("remind me at 9"),
        assistant(
            Some("on it"),
            vec![ToolCall::new("c1", "cron", json!({"at": "9am"}))],
        ),
        Message::tool("c1", "scheduled"),
    ];
    for message in &messages {
        assert_eq!(
            render_message_for_summary_with(message, ToolRendering::Full),
            Some(render_message_for_summary(message))
        );
    }
}

#[test]
fn markers_keep_prose_and_system_messages_as_rendered_today() {
    for message in [
        Message::system("be brief"),
        Message::user("remind me at 9"),
        Message::assistant("done, see you then"),
    ] {
        assert_eq!(
            render_message_for_summary_with(&message, ToolRendering::Markers),
            Some(render_message_for_summary(&message))
        );
    }
}

#[test]
fn markers_omit_tool_results() {
    let result = Message::tool("c1", "SENTINEL-RESULT");
    assert_eq!(
        render_message_for_summary_with(&result, ToolRendering::Markers),
        None
    );
}

#[test]
fn markers_render_each_call_on_its_own_line_without_its_id() {
    let message = assistant(
        Some("on it"),
        vec![
            ToolCall::new("c1", "cron", json!({"at": "9am"})),
            ToolCall::new("c2", "search", json!({"q": "news"})),
        ],
    );
    assert_eq!(
        render_message_for_summary_with(&message, ToolRendering::Markers).as_deref(),
        Some("assistant: on it\n[used cron: {\"at\":\"9am\"}]\n[used search: {\"q\":\"news\"}]")
    );
}

#[test]
fn markers_still_render_an_assistant_turn_that_only_called_tools() {
    let message = assistant(
        None,
        vec![ToolCall::new("c1", "cron", json!({"at": "9am"}))],
    );
    assert_eq!(
        render_message_for_summary_with(&message, ToolRendering::Markers).as_deref(),
        Some("[used cron: {\"at\":\"9am\"}]")
    );
}

#[test]
fn marker_arguments_up_to_the_cap_are_kept_whole() {
    // `{"k":"…"}` is 8 characters around the value: 112 + 8 = exactly the cap.
    let arguments = json!({"k": "a".repeat(112)});
    assert_eq!(arguments.to_string().chars().count(), MARKER_ARGUMENT_CHARS);
    let message = assistant(None, vec![ToolCall::new("c1", "cron", arguments.clone())]);
    assert_eq!(
        render_message_for_summary_with(&message, ToolRendering::Markers),
        Some(format!("[used cron: {arguments}]"))
    );
}

#[test]
fn long_marker_arguments_are_cut_on_a_char_boundary_with_an_ellipsis() {
    let arguments = json!({"note": "é".repeat(300)});
    let message = assistant(None, vec![ToolCall::new("c1", "cron", arguments.clone())]);
    let line = render_message_for_summary_with(&message, ToolRendering::Markers)
        .expect("an assistant turn is never omitted");

    let shown = line
        .strip_prefix("[used cron: ")
        .and_then(|rest| rest.strip_suffix(']'))
        .expect("marker shape");
    assert_eq!(shown.chars().count(), MARKER_ARGUMENT_CHARS);
    assert!(shown.ends_with('…'), "{shown}");
    let kept = shown.trim_end_matches('…');
    assert!(arguments.to_string().starts_with(kept), "{shown}");
}

#[test]
fn markers_leave_out_an_assistant_turn_with_neither_prose_nor_calls() {
    let empty = assistant(None, Vec::new());
    assert_eq!(
        render_message_for_summary_with(&empty, ToolRendering::Markers),
        None
    );
}
