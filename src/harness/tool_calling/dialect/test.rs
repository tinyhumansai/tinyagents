use serde_json::json;

use super::*;
use crate::harness::tool::ToolSchema;
use crate::harness::tool_calling::{build_registry, PFormatRegistry};

fn schema(name: &str, description: &str, parameters: serde_json::Value) -> ToolSchema {
    ToolSchema::new(name, description, parameters)
}

fn weather_schema() -> ToolSchema {
    schema(
        "get_weather",
        "Look up the weather",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            }
        }),
    )
}

fn response(text: &str) -> DialectResponse {
    DialectResponse {
        text: Some(text.to_string()),
        tool_calls: Vec::new(),
    }
}

fn native_call(id: &str, name: &str, arguments: &str) -> NativeToolCall {
    NativeToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: arguments.to_string(),
        extra_content: None,
    }
}

#[test]
fn xml_dialect_parses_a_json_tagged_call() {
    let (text, calls) = XmlDialect.parse_response(&response(
        "Checking.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"London\"}}</tool_call>",
    ));

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments["location"], "London");
    assert!(text.contains("Checking."));
    assert!(!text.contains("tool_call"));
}

#[test]
fn xml_dialect_embeds_the_full_schema_catalogue() {
    let instructions = XmlDialect.prompt_instructions(&[weather_schema()]);

    assert!(instructions.starts_with("## Tool Use Protocol"));
    assert!(instructions.contains("### Available Tools"));
    assert!(instructions.contains("- **get_weather**: Look up the weather"));
    // The model writes argument names itself here, so it has to see them.
    assert!(instructions.contains("location"));
    assert!(XmlDialect.embeds_tool_catalogue());
    assert!(!XmlDialect.should_send_tool_specs());
}

#[test]
fn pformat_dialect_parses_a_positional_call() {
    let registry = build_registry([("get_weather", weather_schema().parameters)]);
    let dialect = PFormatDialect::new(registry);

    let (_text, calls) =
        dialect.parse_response(&response("<tool_call>get_weather[London|metric]</tool_call>"));

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments["location"], "London");
    assert_eq!(calls[0].arguments["unit"], "metric");
}

#[test]
fn pformat_dialect_falls_back_to_json_per_tag() {
    let registry = build_registry([("get_weather", weather_schema().parameters)]);
    let dialect = PFormatDialect::new(registry);

    let (_text, calls) = dialect.parse_response(&response(
        "<tool_call>get_weather[London|metric]</tool_call>\n\
         <tool_call>{\"name\": \"other_tool\", \"arguments\": {\"x\": 1}}</tool_call>",
    ));

    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[1].name, "other_tool");
}

#[test]
fn pformat_dialect_leaves_the_catalogue_to_the_prompt() {
    let instructions = PFormatDialect::new(PFormatRegistry::new()).prompt_instructions(&[
        weather_schema(),
    ]);

    assert!(instructions.contains("P-Format"));
    // Protocol only — listing tools here would duplicate the `## Tools` section.
    // (`get_weather` and `Call as:` still appear — as the syntax example and as
    // a pointer at the `## Tools` section that owns the real listing.)
    assert!(!instructions.contains("Look up the weather"));
    assert!(!instructions.contains("get_weather[location|unit]"));
    assert!(!PFormatDialect::new(PFormatRegistry::new()).embeds_tool_catalogue());
}

#[test]
fn pformat_registry_refuses_an_unregistered_tool_name() {
    // The safety boundary: without a registered layout there is no way to name
    // the positional arguments, so the positional parse must not invent them.
    let dialect = PFormatDialect::new(build_registry([(
        "get_weather",
        weather_schema().parameters,
    )]));

    let (_text, calls) =
        dialect.parse_response(&response("<tool_call>unknown_tool[a|b]</tool_call>"));

    assert!(calls.is_empty(), "unexpected calls: {calls:?}");
}

#[test]
fn native_dialect_reads_the_structured_channel() {
    let (text, calls) = NativeDialect.parse_response(&DialectResponse {
        text: Some("Looking it up".to_string()),
        tool_calls: vec![native_call("call_1", "get_weather", r#"{"location":"London"}"#)],
    });

    assert_eq!(text, "Looking it up");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call_1"));
    assert_eq!(calls[0].arguments["location"], "London");
}

#[test]
fn native_dialect_defaults_unparseable_arguments_to_an_empty_object() {
    let (_text, calls) = NativeDialect.parse_response(&DialectResponse {
        text: None,
        tool_calls: vec![native_call("call_1", "get_weather", "not json")],
    });

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn native_dialect_recovers_a_call_the_model_narrated_as_text() {
    let (_text, calls) = NativeDialect.parse_response(&response(
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>",
    ));

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
}

#[test]
fn text_dialects_render_results_by_name_and_status() {
    let results = vec![
        ToolOutcome::ok("get_weather", "18C"),
        ToolOutcome::failed("send_email", "smtp refused"),
    ];

    for entry in [
        XmlDialect.format_results(&results),
        PFormatDialect::new(PFormatRegistry::new()).format_results(&results),
    ] {
        let TranscriptEntry::Chat(message) = entry else {
            panic!("text dialects fold results into a chat turn");
        };
        assert_eq!(message.role, DialectRole::User);
        assert!(message.content.starts_with(TOOL_RESULTS_PREFIX));
        assert!(message
            .content
            .contains(r#"<tool_result name="get_weather" status="ok">"#));
        assert!(message
            .content
            .contains(r#"<tool_result name="send_email" status="error">"#));
    }
}

#[test]
fn native_dialect_renders_results_into_the_tool_role() {
    let entry = NativeDialect
        .format_results(&[ToolOutcome::ok("get_weather", "18C").with_call_id("call_1")]);

    let TranscriptEntry::ToolResults(results) = entry else {
        panic!("native results stay structured");
    };
    assert_eq!(results[0].tool_call_id, "call_1");
    assert_eq!(results[0].content, "18C");
}

#[test]
fn native_replay_carries_reasoning_and_pairs_the_cycle() {
    let history = vec![
        TranscriptEntry::Chat(DialectMessage::user("weather?")),
        TranscriptEntry::AssistantToolCalls {
            text: Some("checking".to_string()),
            tool_calls: vec![native_call("call_1", "get_weather", "{}")],
            reasoning_content: Some("thinking".to_string()),
            extra_metadata: None,
        },
        TranscriptEntry::ToolResults(vec![ToolResultEntry {
            tool_call_id: "call_1".to_string(),
            content: "18C".to_string(),
        }]),
    ];

    let messages = NativeDialect.to_provider_messages(&history);

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, DialectRole::Assistant);
    assert!(messages[1].content.contains("\"reasoning_content\":\"thinking\""));
    assert_eq!(messages[2].role, DialectRole::Tool);
    assert!(messages[2].content.contains("\"tool_call_id\":\"call_1\""));
}

#[test]
fn native_replay_drops_an_assistant_turn_whose_results_never_landed() {
    let history = vec![
        TranscriptEntry::Chat(DialectMessage::user("weather?")),
        TranscriptEntry::AssistantToolCalls {
            text: Some("checking".to_string()),
            tool_calls: vec![native_call("call_1", "get_weather", "{}")],
            reasoning_content: None,
            extra_metadata: None,
        },
        TranscriptEntry::Chat(DialectMessage::user("still there?")),
    ];

    let messages = NativeDialect.to_provider_messages(&history);

    assert_eq!(messages.len(), 2);
    assert!(messages.iter().all(|m| m.role == DialectRole::User));
}

#[test]
fn native_replay_drops_a_cycle_whose_results_do_not_cover_every_call() {
    let history = vec![
        TranscriptEntry::AssistantToolCalls {
            text: None,
            tool_calls: vec![
                native_call("call_1", "a", "{}"),
                native_call("call_2", "b", "{}"),
            ],
            reasoning_content: None,
            extra_metadata: None,
        },
        TranscriptEntry::ToolResults(vec![ToolResultEntry {
            tool_call_id: "call_1".to_string(),
            content: "done".to_string(),
        }]),
    ];

    // Adjacency is not enough: the provider rejects partial coverage the same
    // way it rejects no coverage, so both halves go.
    assert!(NativeDialect.to_provider_messages(&history).is_empty());
}

#[test]
fn native_replay_drops_orphan_results() {
    let history = vec![TranscriptEntry::ToolResults(vec![ToolResultEntry {
        tool_call_id: "call_1".to_string(),
        content: "done".to_string(),
    }])];

    assert!(NativeDialect.to_provider_messages(&history).is_empty());
}

#[test]
fn text_replay_flattens_tool_cycles_into_chat() {
    let history = vec![
        TranscriptEntry::AssistantToolCalls {
            text: Some("checking".to_string()),
            tool_calls: vec![native_call("call_1", "get_weather", "{}")],
            reasoning_content: None,
            extra_metadata: Some(json!({"host": "keep me"})),
        },
        TranscriptEntry::ToolResults(vec![ToolResultEntry {
            tool_call_id: "call_1".to_string(),
            content: "18C".to_string(),
        }]),
    ];

    let messages = XmlDialect.to_provider_messages(&history);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, DialectRole::Assistant);
    assert_eq!(messages[0].content, "checking");
    assert_eq!(messages[0].extra_metadata, Some(json!({"host": "keep me"})));
    assert_eq!(messages[1].role, DialectRole::User);
    assert!(messages[1].content.contains(r#"<tool_result id="call_1">"#));
}

#[test]
fn catalogue_signature_matches_what_the_parser_reconstructs() {
    let tools = [weather_schema()];
    let rendered = render_pformat_catalogue(&tools);

    assert!(rendered.starts_with(CATALOGUE_HEADING));
    assert!(rendered.contains("Call as: `get_weather[location|unit]`"));

    // The catalogue order is the order the parser assigns, not a coincidence.
    let dialect = PFormatDialect::new(build_registry([("get_weather", weather_schema().parameters)]));
    let (_text, calls) =
        dialect.parse_response(&response("<tool_call>get_weather[London|metric]</tool_call>"));
    assert_eq!(calls[0].arguments["location"], "London");
    assert_eq!(calls[0].arguments["unit"], "metric");
}

#[test]
fn each_dialect_reports_the_format_its_parser_expects() {
    assert_eq!(XmlDialect.tool_call_format(), ToolCallFormat::Json);
    assert_eq!(
        PFormatDialect::new(PFormatRegistry::new()).tool_call_format(),
        ToolCallFormat::PFormat
    );
    assert_eq!(NativeDialect.tool_call_format(), ToolCallFormat::Native);
    assert!(NativeDialect.should_send_tool_specs());
}
