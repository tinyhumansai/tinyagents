//! Feature coverage for the harness prompt-assembly surface.
//!
//! Exercises the *public* prompt feature: [`PromptTemplate`] placeholder
//! substitution (including escaping and error paths), [`MessagesTemplate`]
//! ordered rendering, and [`PromptBuilder`] segment assembly with its
//! KV-cache-oriented properties — the cacheable prefix ids surfaced on the
//! built [`ModelRequest`] and the stable-prefix `fingerprint`.
//!
//! Everything here is pure and offline: no model, no network. These scenarios
//! complement the module unit tests by focusing on the fingerprint-stability
//! contract (identical prefix -> identical hash, changed tool schema -> changed
//! hash, volatile tail excluded) and the built-request shape.

use serde_json::{Map, Value, json};

use tinyagents::TinyAgentsError;
use tinyagents::harness::message::Message;
use tinyagents::harness::prompt::{MessagesTemplate, PromptBuilder, PromptTemplate, TemplateRole};
use tinyagents::harness::tool::ToolSchema;

fn vars(pairs: &[(&str, Value)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (k, v) in pairs {
        map.insert((*k).to_string(), v.clone());
    }
    map
}

// ── PromptTemplate ────────────────────────────────────────────────────────────

#[test]
fn template_substitutes_named_placeholders() {
    let tpl = PromptTemplate::new("Hello {name}, you have {count} messages.");
    let rendered = tpl
        .render(&vars(&[("name", json!("Ada")), ("count", json!(3))]))
        .expect("all placeholders are provided");
    assert_eq!(rendered, "Hello Ada, you have 3 messages.");
}

#[test]
fn template_honours_brace_escapes() {
    let tpl = PromptTemplate::new("Use {{literal}} braces around {name}.");
    let rendered = tpl
        .render(&vars(&[("name", json!("x"))]))
        .expect("escaped braces render literally");
    assert_eq!(rendered, "Use {literal} braces around x.");
}

#[test]
fn template_rejects_unknown_placeholder() {
    let tpl = PromptTemplate::new("Hi {missing}");
    let err = tpl
        .render(&vars(&[("present", json!("x"))]))
        .expect_err("an unknown placeholder must fail");
    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "expected a Validation error, got {err:?}"
    );
}

#[test]
fn template_rejects_unclosed_placeholder() {
    let tpl = PromptTemplate::new("Hi {name");
    let err = tpl
        .render(&vars(&[("name", json!("x"))]))
        .expect_err("an unclosed placeholder must fail");
    assert!(
        matches!(err, TinyAgentsError::Validation(_)),
        "expected a Validation error, got {err:?}"
    );
}

#[test]
fn render_message_wraps_in_the_requested_role() {
    let tpl = PromptTemplate::new("be terse");
    let vars = Map::new();

    assert!(matches!(
        tpl.render_message(TemplateRole::System, &vars).unwrap(),
        Message::System(_)
    ));
    assert!(matches!(
        tpl.render_message(TemplateRole::User, &vars).unwrap(),
        Message::User(_)
    ));
    assert!(matches!(
        tpl.render_message(TemplateRole::Assistant, &vars).unwrap(),
        Message::Assistant(_)
    ));
}

// ── MessagesTemplate ──────────────────────────────────────────────────────────

#[test]
fn messages_template_renders_entries_in_order() {
    let mut tpl = MessagesTemplate::new();
    tpl.push(TemplateRole::System, PromptTemplate::new("You are {role}."))
        .push(TemplateRole::User, PromptTemplate::new("Question: {q}"));

    let messages = tpl
        .render(&vars(&[("role", json!("a bot")), ("q", json!("why?"))]))
        .expect("both entries render");

    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0], Message::System(_)));
    assert_eq!(messages[0].text(), "You are a bot.");
    assert!(matches!(messages[1], Message::User(_)));
    assert_eq!(messages[1].text(), "Question: why?");
}

#[test]
fn messages_template_propagates_first_error() {
    let mut tpl = MessagesTemplate::new();
    tpl.push(TemplateRole::System, PromptTemplate::new("ok"))
        .push(TemplateRole::User, PromptTemplate::new("{unknown}"));

    let err = tpl
        .render(&Map::new())
        .expect_err("the second entry references an unknown placeholder");
    assert!(matches!(err, TinyAgentsError::Validation(_)), "{err:?}");
}

// ── PromptBuilder ─────────────────────────────────────────────────────────────

fn schema(name: &str) -> ToolSchema {
    ToolSchema::new(name, format!("{name} tool"), json!({ "type": "object" }))
}

#[test]
fn builder_concatenates_segments_and_appends_tail() {
    let mut builder = PromptBuilder::new();
    builder
        .push_system("sys", vec![Message::system("system rules")])
        .push_tools_segment("tools", vec![schema("search")])
        .push_history("hist", vec![Message::user("earlier turn")]);

    let request = builder.build(vec![Message::user("current turn")]);

    // All segment messages precede the tail, in push order.
    let texts: Vec<String> = request.messages.iter().map(|m| m.text()).collect();
    assert_eq!(
        texts,
        vec![
            "system rules".to_string(),
            "earlier turn".to_string(),
            "current turn".to_string(),
        ]
    );

    // The tools segment fed its schema into the request tool list.
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "search");

    // A fingerprint over the stable prefix is attached.
    assert!(request.prompt_fingerprint.is_some());
}

#[test]
fn builder_exposes_only_cacheable_prefix_ids() {
    let mut builder = PromptBuilder::new();
    builder
        .push_system("sys", vec![Message::system("rules")])
        .push_tools_segment("tools", vec![schema("search")])
        .push_history("hist", vec![Message::user("history")])
        .push_volatile("turn", vec![Message::user("volatile")]);

    let request = builder.build(vec![]);

    // Only the cacheable segments (system, tools) form the stable prefix; the
    // non-cacheable history/volatile segments are excluded.
    assert_eq!(request.cacheable_prefix_ids(), vec!["sys", "tools"]);
}

#[test]
fn fingerprint_is_stable_for_an_identical_stable_prefix() {
    let build = || {
        let mut b = PromptBuilder::new();
        b.push_system("sys", vec![Message::system("rules")])
            .push_tools_segment("tools", vec![schema("search")]);
        b.fingerprint()
    };
    assert_eq!(
        build(),
        build(),
        "the same stable prefix hashes identically"
    );
}

#[test]
fn fingerprint_ignores_volatile_tail_changes() {
    let mut base = PromptBuilder::new();
    base.push_system("sys", vec![Message::system("rules")]);
    let baseline = base.fingerprint();

    // Adding a *non-cacheable* volatile segment must not perturb the stable
    // prefix fingerprint.
    let mut with_volatile = PromptBuilder::new();
    with_volatile
        .push_system("sys", vec![Message::system("rules")])
        .push_volatile("turn", vec![Message::user("anything")]);
    assert_eq!(
        with_volatile.fingerprint(),
        baseline,
        "volatile content is outside the cacheable prefix"
    );
}

#[test]
fn fingerprint_changes_when_a_tool_schema_changes() {
    let mut a = PromptBuilder::new();
    a.push_system("sys", vec![Message::system("rules")])
        .push_tools_segment("tools", vec![schema("search")]);

    let mut b = PromptBuilder::new();
    b.push_system("sys", vec![Message::system("rules")])
        .push_tools_segment("tools", vec![schema("different_tool")]);

    assert_ne!(
        a.fingerprint(),
        b.fingerprint(),
        "a tool-schema edit changes the stable prefix, so the fingerprint changes"
    );
}

#[test]
fn fingerprint_changes_when_a_system_message_changes() {
    let mut a = PromptBuilder::new();
    a.push_system("sys", vec![Message::system("rules v1")]);

    let mut b = PromptBuilder::new();
    b.push_system("sys", vec![Message::system("rules v2")]);

    assert_ne!(a.fingerprint(), b.fingerprint());
}
