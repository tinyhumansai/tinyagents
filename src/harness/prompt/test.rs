//! Tests for prompt assembly: `{name}` placeholder substitution and escaping,
//! unknown/unclosed-placeholder errors, per-role message rendering,
//! `MessagesTemplate` ordering and error propagation, and `PromptBuilder`
//! segment composition (cacheability by segment type and stable-prefix
//! fingerprinting).

use super::*;
use serde_json::{Map, json};

#[test]
fn renders_simple_placeholder() {
    let tpl = PromptTemplate::new("Hello, {name}!");
    let mut vars = Map::new();
    vars.insert("name".to_string(), json!("world"));
    assert_eq!(tpl.render(&vars).unwrap(), "Hello, world!");
}

#[test]
fn escapes_double_braces() {
    let tpl = PromptTemplate::new("literal {{braces}}");
    let vars = Map::new();
    assert_eq!(tpl.render(&vars).unwrap(), "literal {braces}");
}

#[test]
fn errors_on_unknown_placeholder() {
    let tpl = PromptTemplate::new("{missing}");
    let vars = Map::new();
    assert!(tpl.render(&vars).is_err());
}

#[test]
fn builder_produces_cache_segments() {
    let mut builder = PromptBuilder::new();
    builder.push_system("sys", vec![Message::system("You are helpful.")]);
    builder.push_volatile("user-turn", vec![Message::user("Hi")]);
    let req = builder.build(vec![]);
    assert_eq!(req.cache_segments.len(), 2);
    assert!(req.cache_segments[0].cacheable);
    assert!(!req.cache_segments[1].cacheable);
    assert!(req.prompt_fingerprint.is_some());
}

// ── PromptTemplate rendering ──────────────────────────────────────────────────

#[test]
fn renders_non_string_value() {
    let tpl = PromptTemplate::new("count={n}");
    let mut vars = Map::new();
    vars.insert("n".to_string(), json!(42));
    assert_eq!(tpl.render(&vars).unwrap(), "count=42");
}

#[test]
fn errors_on_unclosed_placeholder() {
    let tpl = PromptTemplate::new("hello {name");
    let vars = Map::new();
    let err = tpl.render(&vars).unwrap_err();
    assert!(err.to_string().contains("unclosed"));
}

#[test]
fn render_message_role_helpers() {
    let tpl = PromptTemplate::new("hi {who}");
    let mut vars = Map::new();
    vars.insert("who".to_string(), json!("there"));

    assert!(matches!(
        tpl.render_message(TemplateRole::System, &vars).unwrap(),
        Message::System(_)
    ));
    assert!(matches!(
        tpl.render_system(&vars).unwrap(),
        Message::System(_)
    ));
    assert!(matches!(tpl.render_user(&vars).unwrap(), Message::User(_)));
    let assistant = tpl.render_assistant(&vars).unwrap();
    assert!(matches!(assistant, Message::Assistant(_)));
    assert_eq!(assistant.text(), "hi there");
}

#[test]
fn messages_template_renders_roles_in_order() {
    let mut tpl = MessagesTemplate::new();
    tpl.push(TemplateRole::System, PromptTemplate::new("sys {v}"))
        .push(TemplateRole::User, PromptTemplate::new("user {v}"));
    let mut vars = Map::new();
    vars.insert("v".to_string(), json!("X"));

    let msgs = tpl.render(&vars).unwrap();
    assert_eq!(msgs.len(), 2);
    assert!(matches!(msgs[0], Message::System(_)));
    assert_eq!(msgs[0].text(), "sys X");
    assert!(matches!(msgs[1], Message::User(_)));
    assert_eq!(msgs[1].text(), "user X");
}

#[test]
fn messages_template_propagates_render_error() {
    let mut tpl = MessagesTemplate::new();
    tpl.push(TemplateRole::User, PromptTemplate::new("{missing}"));
    assert!(tpl.render(&Map::new()).is_err());
}

// ── PromptBuilder segment composition ─────────────────────────────────────────

#[test]
fn builder_cacheability_by_segment_type() {
    use crate::harness::tool::ToolSchema;

    let mut builder = PromptBuilder::new();
    builder
        .push_system("sys", vec![Message::system("S")])
        .push_tools_segment(
            "tools",
            vec![ToolSchema::new("calc", "adds numbers", json!({}))],
        )
        .push_instructions("inst", vec![Message::system("follow rules")])
        .push_history("hist", vec![Message::user("earlier")])
        .push_volatile("vol", vec![Message::user("now")]);

    let req = builder.build(vec![Message::user("tail")]);

    // Five segments in push order.
    assert_eq!(req.cache_segments.len(), 5);
    let cacheable: Vec<bool> = req.cache_segments.iter().map(|s| s.cacheable).collect();
    assert_eq!(cacheable, vec![true, true, true, false, false]);

    // Tool accumulated into the request.
    assert_eq!(req.tools.len(), 1);
    assert_eq!(req.tools[0].name, "calc");

    // Messages from system + instructions + history + volatile + tail; tools
    // segment carries no messages.
    let texts: Vec<String> = req.messages.iter().map(|m| m.text()).collect();
    assert_eq!(texts, vec!["S", "follow rules", "earlier", "now", "tail"]);
}

#[test]
fn fingerprint_stable_for_same_stable_prefix() {
    let mut a = PromptBuilder::new();
    a.push_system("sys", vec![Message::system("stable")]);
    a.push_volatile("vol", vec![Message::user("turn-1")]);

    let mut b = PromptBuilder::new();
    b.push_system("sys", vec![Message::system("stable")]);
    // Different volatile (non-cacheable) content must not change fingerprint.
    b.push_volatile("vol", vec![Message::user("turn-2")]);

    assert_eq!(a.fingerprint(), b.fingerprint());
}

#[test]
fn fingerprint_changes_with_stable_prefix() {
    let mut a = PromptBuilder::new();
    a.push_system("sys", vec![Message::system("stable")]);

    let mut b = PromptBuilder::new();
    b.push_system("sys", vec![Message::system("DIFFERENT")]);

    assert_ne!(a.fingerprint(), b.fingerprint());

    // The built request carries the same fingerprint as the builder.
    assert_eq!(a.build(vec![]).prompt_fingerprint.unwrap(), a.fingerprint());
}

// ── Fingerprint content coverage and stability ───────────────────────────────

#[test]
fn fingerprint_is_64_hex_and_deterministic() {
    let mut builder = PromptBuilder::new();
    builder.push_system("sys", vec![Message::system("stable")]);
    let fp = builder.fingerprint();
    assert_eq!(fp.len(), 64, "SHA-256 hex digest");
    assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(fp, builder.fingerprint());
}

#[test]
fn fingerprint_changes_with_tool_schema_not_just_name() {
    use crate::harness::tool::ToolSchema;

    let tool_v1 = ToolSchema::new("calc", "adds numbers", json!({"type": "object"}));
    let mut tool_v2 = tool_v1.clone();
    tool_v2.parameters = json!({"type": "object", "required": ["a"]});

    let mut a = PromptBuilder::new();
    a.push_tools_segment("tools", vec![tool_v1]);
    let mut b = PromptBuilder::new();
    b.push_tools_segment("tools", vec![tool_v2]);

    assert_ne!(
        a.fingerprint(),
        b.fingerprint(),
        "a parameter-schema change must change the fingerprint even when the tool name is unchanged"
    );
}

#[test]
fn fingerprint_changes_with_image_content_and_message_role() {
    use crate::harness::message::{ContentBlock, ImageRef, UserMessage};

    // Same (empty) text, different image URL.
    let image = |url: &str| {
        Message::User(UserMessage {
            content: vec![ContentBlock::Image(ImageRef {
                url: url.to_string(),
                mime_type: Some("image/png".to_string()),
            })],
        })
    };
    let mut a = PromptBuilder::new();
    a.push_system("sys", vec![image("https://example.com/a.png")]);
    let mut b = PromptBuilder::new();
    b.push_system("sys", vec![image("https://example.com/b.png")]);
    assert_ne!(
        a.fingerprint(),
        b.fingerprint(),
        "image content must participate in the fingerprint"
    );

    // Same text, different role.
    let mut sys = PromptBuilder::new();
    sys.push_system("seg", vec![Message::system("same text")]);
    let mut user = PromptBuilder::new();
    user.push_system("seg", vec![Message::user("same text")]);
    assert_ne!(
        sys.fingerprint(),
        user.fingerprint(),
        "message role must participate in the fingerprint"
    );
}

/// Pins the fingerprint of a fixed prefix so accidental changes to the hash
/// input or algorithm are caught: the value must be stable across processes,
/// platforms, and Rust versions. Update this constant only on a deliberate,
/// documented fingerprint-format change.
#[test]
fn fingerprint_value_is_pinned_for_cross_process_stability() {
    let mut builder = PromptBuilder::new();
    builder.push_system("sys", vec![Message::system("pinned prefix")]);
    assert_eq!(
        builder.fingerprint(),
        "0c8eb74fc9194b5d7845d787735eb2e36a68dc9d2ed91e5e7e07a13035d7d2a6"
    );
}
