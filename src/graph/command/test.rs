//! Unit tests for command/interrupt constructors: building commands with
//! updates, `goto` routing, and resume values, plus interrupt id generation.

use super::*;
use serde_json::json;

#[test]
fn command_constructors() {
    let c: Command<i32> = Command::goto(["a", "b"]);
    assert_eq!(c.goto.len(), 2);
    assert!(c.update.is_none());

    let c = Command::update(7).with_goto(["next"]);
    assert_eq!(c.update, Some(7));
    assert_eq!(c.goto[0].as_str(), "next");

    let c: Command<i32> = Command::resume(json!({ "approved": true }));
    assert_eq!(c.resume, Some(json!({ "approved": true })));
}

#[test]
fn interrupt_ids_are_unique() {
    let a = Interrupt::new("review", json!("a"));
    let b = Interrupt::new("review", json!("b"));
    assert_ne!(a.id, b.id);
    assert_eq!(a.node.as_str(), "review");

    let fixed = Interrupt::with_id("fixed", "n", json!(null));
    assert_eq!(fixed.id, "fixed");
}
