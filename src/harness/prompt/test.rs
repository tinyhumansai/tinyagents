//! Tests added in a later pass.

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
