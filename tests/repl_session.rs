//! End-to-end coverage for the REPL command parser and session runtime.
//!
//! These tests exercise the public surface re-exported from
//! `rustagents::repl`: parsing representative command lines, running a
//! [`ReplSession`] through `set`/`get`/`show`, and enforcing the
//! [`CapabilityPolicy`] allowlist on `call`.

use rustagents::repl::{
    CapabilityPolicy, ReplCommand, ReplOutcome, ReplSession, parse_command,
};
use rustagents::RustAgentsError;

#[test]
fn parses_representative_commands() {
    assert_eq!(
        parse_command("set region us-east").unwrap(),
        ReplCommand::Set {
            key: "region".to_string(),
            value: "us-east".to_string(),
        }
    );

    assert_eq!(
        parse_command(r#"set greeting "hello world""#).unwrap(),
        ReplCommand::Set {
            key: "greeting".to_string(),
            value: "hello world".to_string(),
        }
    );

    assert_eq!(
        parse_command("get region").unwrap(),
        ReplCommand::Get {
            key: "region".to_string(),
        }
    );

    assert_eq!(
        parse_command("show vars").unwrap(),
        ReplCommand::Show {
            what: "vars".to_string(),
        }
    );

    assert_eq!(
        parse_command(r#"call lookup_user {"user_id": "usr_123"}"#).unwrap(),
        ReplCommand::Call {
            capability: "lookup_user".to_string(),
            args: serde_json::json!({ "user_id": "usr_123" }),
        }
    );

    assert_eq!(parse_command("quit").unwrap(), ReplCommand::Quit);
}

#[test]
fn session_runs_set_get_and_show() {
    let mut session = ReplSession::new();

    // `set` is a side-effect-free acknowledgement message.
    let set_outcome = session
        .execute(ReplCommand::Set {
            key: "env".to_string(),
            value: "production".to_string(),
        })
        .unwrap();
    assert!(matches!(set_outcome, ReplOutcome::Message(_)));

    // `get` round-trips the stored value as JSON.
    let get_outcome = session
        .execute(ReplCommand::Get {
            key: "env".to_string(),
        })
        .unwrap();
    assert_eq!(get_outcome, ReplOutcome::Value(serde_json::json!("production")));

    // `show vars` reflects everything in the namespace.
    let show_vars = session
        .execute(ReplCommand::Show {
            what: "vars".to_string(),
        })
        .unwrap();
    match show_vars {
        ReplOutcome::Value(v) => assert_eq!(v["env"], serde_json::json!("production")),
        other => panic!("expected Value outcome for `show vars`, got {other:?}"),
    }

    // `show status` reports the variable count.
    let show_status = session
        .execute(ReplCommand::Show {
            what: "status".to_string(),
        })
        .unwrap();
    match show_status {
        ReplOutcome::Value(v) => assert_eq!(v["variables"], serde_json::json!(1)),
        other => panic!("expected Value outcome for `show status`, got {other:?}"),
    }

    // Every executed command is recorded in history.
    assert_eq!(session.history.len(), 4);
}

#[test]
fn disallowed_capability_call_is_rejected() {
    // A fresh session has a deny-all policy.
    let mut session = ReplSession::new();

    let err = session
        .execute(ReplCommand::Call {
            capability: "secret_tool".to_string(),
            args: serde_json::Value::Null,
        })
        .unwrap_err();

    match err {
        RustAgentsError::Capability(msg) => {
            assert!(msg.contains("secret_tool"), "error should name the capability: {msg}");
        }
        other => panic!("expected Capability error, got {other:?}"),
    }
}

#[test]
fn allowed_capability_call_returns_planned() {
    let policy = CapabilityPolicy::from_list(["lookup_user"]);
    let mut session = ReplSession::new().with_policy(policy);

    let outcome = session
        .execute(ReplCommand::Call {
            capability: "lookup_user".to_string(),
            args: serde_json::json!({ "user_id": "usr_42" }),
        })
        .unwrap();

    match outcome {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "capability_call");
            assert_eq!(detail["capability"], "lookup_user");
            assert_eq!(detail["args"]["user_id"], "usr_42");
        }
        other => panic!("expected Planned outcome, got {other:?}"),
    }
}
