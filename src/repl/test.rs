//! Tests for the `.ragsh` REPL skeleton.

use super::{CapabilityPolicy, ReplCommand, ReplOutcome, ReplSession, parse_command};
use crate::error::TinyAgentsError;

// ── Parser tests ──────────────────────────────────────────────────────────────

#[test]
fn parses_help() {
    assert_eq!(parse_command("help").unwrap(), ReplCommand::Help);
    assert_eq!(parse_command("HELP").unwrap(), ReplCommand::Help);
    assert_eq!(parse_command("?").unwrap(), ReplCommand::Help);
}

#[test]
fn parses_quit() {
    assert_eq!(parse_command("quit").unwrap(), ReplCommand::Quit);
    assert_eq!(parse_command("exit").unwrap(), ReplCommand::Quit);
    assert_eq!(parse_command("q").unwrap(), ReplCommand::Quit);
    assert_eq!(parse_command("QUIT").unwrap(), ReplCommand::Quit);
}

#[test]
fn parses_load() {
    let cmd = parse_command("load ./blueprints/support.rag").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Load {
            path: "./blueprints/support.rag".to_string()
        }
    );
}

#[test]
fn parses_load_quoted_path() {
    let cmd = parse_command(r#"load "path with spaces/my blueprint.rag""#).unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Load {
            path: "path with spaces/my blueprint.rag".to_string()
        }
    );
}

#[test]
fn parses_compile() {
    let cmd = parse_command("compile support_flow").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Compile {
            name: "support_flow".to_string()
        }
    );
}

#[test]
fn parses_run() {
    let cmd = parse_command(r#"run my_graph "{\"user\":1}""#).unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Run {
            graph: "my_graph".to_string(),
            input: r#"{"user":1}"#.to_string()
        }
    );
}

#[test]
fn parses_run_bare_input() {
    let cmd = parse_command("run approval_flow initial").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Run {
            graph: "approval_flow".to_string(),
            input: "initial".to_string()
        }
    );
}

#[test]
fn parses_set() {
    let cmd = parse_command("set my_var hello").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Set {
            key: "my_var".to_string(),
            value: "hello".to_string()
        }
    );
}

#[test]
fn parses_set_quoted_value() {
    let cmd = parse_command(r#"set greeting "hello world""#).unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Set {
            key: "greeting".to_string(),
            value: "hello world".to_string()
        }
    );
}

#[test]
fn parses_get() {
    let cmd = parse_command("get my_var").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Get {
            key: "my_var".to_string()
        }
    );
}

#[test]
fn parses_show_vars() {
    let cmd = parse_command("show vars").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Show {
            what: "vars".to_string()
        }
    );
}

#[test]
fn parses_show_graphs() {
    let cmd = parse_command("show graphs").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Show {
            what: "graphs".to_string()
        }
    );
}

#[test]
fn parses_show_status() {
    let cmd = parse_command("show status").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Show {
            what: "status".to_string()
        }
    );
}

#[test]
fn parses_call_with_json_object() {
    let cmd = parse_command(r#"call lookup_user {"user_id": "usr_123"}"#).unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Call {
            capability: "lookup_user".to_string(),
            args: serde_json::json!({"user_id": "usr_123"})
        }
    );
}

#[test]
fn parses_call_with_json_array() {
    let cmd = parse_command(r#"call batch_tool [1, 2, 3]"#).unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Call {
            capability: "batch_tool".to_string(),
            args: serde_json::json!([1, 2, 3])
        }
    );
}

#[test]
fn parses_call_with_json_null() {
    let cmd = parse_command("call noop null").unwrap();
    assert_eq!(
        cmd,
        ReplCommand::Call {
            capability: "noop".to_string(),
            args: serde_json::Value::Null,
        }
    );
}

#[test]
fn error_on_unknown_verb() {
    let err = parse_command("frobnicate something").unwrap_err();
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(
                message.contains("frobnicate"),
                "expected verb in message: {message}"
            );
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn error_on_empty_input() {
    assert!(parse_command("").is_err());
    assert!(parse_command("   ").is_err());
}

#[test]
fn error_on_missing_load_argument() {
    let err = parse_command("load").unwrap_err();
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn error_on_missing_run_input() {
    let err = parse_command("run my_graph").unwrap_err();
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn error_on_call_missing_json() {
    let err = parse_command("call my_cap").unwrap_err();
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn error_on_call_invalid_json() {
    let err = parse_command("call my_cap not-valid-json").unwrap_err();
    match err {
        TinyAgentsError::Parse { message, .. } => {
            assert!(message.contains("JSON"), "expected JSON mention: {message}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn error_on_unterminated_quoted_string() {
    let err = parse_command(r#"load "unclosed"#).unwrap_err();
    assert!(matches!(err, TinyAgentsError::Parse { .. }));
}

#[test]
fn quoted_string_escape_sequences() {
    let cmd = parse_command(r#"set msg "hello \"world\"\nnewline""#).unwrap();
    if let ReplCommand::Set { value, .. } = cmd {
        assert!(value.contains('"'));
        assert!(value.contains('\n'));
    } else {
        panic!("expected Set command");
    }
}

// ── ReplCommand helpers ───────────────────────────────────────────────────────

#[test]
fn command_name_returns_verb() {
    assert_eq!(ReplCommand::Help.name(), "help");
    assert_eq!(ReplCommand::Quit.name(), "quit");
    assert_eq!(ReplCommand::Load { path: "x".into() }.name(), "load");
    assert_eq!(
        ReplCommand::Call {
            capability: "x".into(),
            args: serde_json::Value::Null
        }
        .name(),
        "call"
    );
}

#[test]
fn command_is_serde_roundtrip() {
    let cmd = ReplCommand::Call {
        capability: "my_tool".to_string(),
        args: serde_json::json!({"k": 1}),
    };
    let json = serde_json::to_string(&cmd).unwrap();
    let back: ReplCommand = serde_json::from_str(&json).unwrap();
    assert_eq!(cmd, back);
}

// ── CapabilityPolicy tests ────────────────────────────────────────────────────

#[test]
fn policy_deny_all_by_default() {
    let policy = CapabilityPolicy::new();
    assert!(!policy.is_allowed("anything"));
    assert!(policy.is_empty());
    assert_eq!(policy.len(), 0);
}

#[test]
fn policy_allow_and_check() {
    let mut policy = CapabilityPolicy::new();
    policy.allow("lookup_user");
    assert!(policy.is_allowed("lookup_user"));
    assert!(!policy.is_allowed("other_cap"));
    assert_eq!(policy.len(), 1);
}

#[test]
fn policy_from_list() {
    let policy = CapabilityPolicy::from_list(["a", "b", "c"]);
    assert!(policy.is_allowed("a"));
    assert!(policy.is_allowed("b"));
    assert!(policy.is_allowed("c"));
    assert!(!policy.is_allowed("d"));
    assert_eq!(policy.len(), 3);
}

// ── ReplSession tests ─────────────────────────────────────────────────────────

#[test]
fn session_set_and_get() {
    let mut session = ReplSession::new();
    session.set("x", serde_json::json!(42));
    assert_eq!(session.get("x"), Some(&serde_json::json!(42)));
    assert_eq!(session.get("missing"), None);
}

#[test]
fn session_vars_returns_all() {
    let mut session = ReplSession::new();
    session.set("a", serde_json::json!(1));
    session.set("b", serde_json::json!("hello"));
    assert_eq!(session.vars().len(), 2);
}

#[test]
fn session_execute_help() {
    let mut session = ReplSession::new();
    let outcome = session.execute(ReplCommand::Help).unwrap();
    assert!(matches!(outcome, ReplOutcome::Message(_)));
    if let ReplOutcome::Message(text) = outcome {
        assert!(text.contains("help"), "help text should mention commands");
        assert!(text.contains("quit"));
        assert!(text.contains("call"));
    }
}

#[test]
fn session_execute_quit() {
    let mut session = ReplSession::new();
    let outcome = session.execute(ReplCommand::Quit).unwrap();
    assert_eq!(outcome, ReplOutcome::Quit);
}

#[test]
fn session_execute_set_and_get() {
    let mut session = ReplSession::new();

    let set_outcome = session
        .execute(ReplCommand::Set {
            key: "env".to_string(),
            value: "production".to_string(),
        })
        .unwrap();
    assert!(matches!(set_outcome, ReplOutcome::Message(_)));

    let get_outcome = session
        .execute(ReplCommand::Get {
            key: "env".to_string(),
        })
        .unwrap();
    assert_eq!(
        get_outcome,
        ReplOutcome::Value(serde_json::json!("production"))
    );
}

#[test]
fn session_execute_get_missing_key_returns_null() {
    let mut session = ReplSession::new();
    let outcome = session
        .execute(ReplCommand::Get {
            key: "nope".to_string(),
        })
        .unwrap();
    assert_eq!(outcome, ReplOutcome::Value(serde_json::Value::Null));
}

#[test]
fn session_execute_show_vars() {
    let mut session = ReplSession::new();
    session.set("color", serde_json::json!("blue"));

    let outcome = session
        .execute(ReplCommand::Show {
            what: "vars".to_string(),
        })
        .unwrap();
    if let ReplOutcome::Value(v) = outcome {
        assert_eq!(v["color"], serde_json::json!("blue"));
    } else {
        panic!("expected Value outcome");
    }
}

#[test]
fn session_execute_show_status() {
    let mut session = ReplSession::new();
    session.set("x", serde_json::json!(1));

    let outcome = session
        .execute(ReplCommand::Show {
            what: "status".to_string(),
        })
        .unwrap();
    if let ReplOutcome::Value(v) = outcome {
        assert_eq!(v["variables"], serde_json::json!(1));
    } else {
        panic!("expected Value outcome for show status");
    }
}

#[test]
fn session_execute_show_graphs() {
    let mut session = ReplSession::new();
    let outcome = session
        .execute(ReplCommand::Show {
            what: "graphs".to_string(),
        })
        .unwrap();
    assert!(matches!(outcome, ReplOutcome::Message(_)));
}

#[test]
fn session_execute_show_unknown_subject() {
    let mut session = ReplSession::new();
    let outcome = session
        .execute(ReplCommand::Show {
            what: "widgets".to_string(),
        })
        .unwrap();
    if let ReplOutcome::Message(msg) = outcome {
        assert!(msg.contains("widgets"));
    } else {
        panic!("expected Message outcome");
    }
}

// ── Capability policy enforcement ─────────────────────────────────────────────

#[test]
fn call_disallowed_capability_returns_error() {
    let mut session = ReplSession::new(); // deny-all policy

    let err = session
        .execute(ReplCommand::Call {
            capability: "secret_tool".to_string(),
            args: serde_json::Value::Null,
        })
        .unwrap_err();

    match err {
        TinyAgentsError::Capability(msg) => {
            assert!(
                msg.contains("secret_tool"),
                "error should name the capability: {msg}"
            );
        }
        other => panic!("expected Capability error, got {other:?}"),
    }
}

#[test]
fn call_allowed_capability_returns_planned() {
    let policy = CapabilityPolicy::from_list(["lookup_user"]);
    let mut session = ReplSession::new().with_policy(policy);

    let outcome = session
        .execute(ReplCommand::Call {
            capability: "lookup_user".to_string(),
            args: serde_json::json!({"user_id": "usr_42"}),
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

#[test]
fn load_disallowed_returns_capability_error() {
    let mut session = ReplSession::new();
    let err = session
        .execute(ReplCommand::Load {
            path: "x.rag".to_string(),
        })
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}

#[test]
fn load_allowed_returns_planned() {
    let policy = CapabilityPolicy::from_list(["load"]);
    let mut session = ReplSession::new().with_policy(policy);
    let outcome = session
        .execute(ReplCommand::Load {
            path: "x.rag".to_string(),
        })
        .unwrap();
    match outcome {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "load");
            assert_eq!(detail["path"], "x.rag");
        }
        other => panic!("expected Planned, got {other:?}"),
    }
}

#[test]
fn run_disallowed_returns_capability_error() {
    let mut session = ReplSession::new();
    let err = session
        .execute(ReplCommand::Run {
            graph: "g".to_string(),
            input: "{}".to_string(),
        })
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}

#[test]
fn run_allowed_returns_planned() {
    let policy = CapabilityPolicy::from_list(["run"]);
    let mut session = ReplSession::new().with_policy(policy);
    let outcome = session
        .execute(ReplCommand::Run {
            graph: "approval_flow".to_string(),
            input: r#"{"step":1}"#.to_string(),
        })
        .unwrap();
    match outcome {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "graph_run");
            assert_eq!(detail["graph"], "approval_flow");
        }
        other => panic!("expected Planned, got {other:?}"),
    }
}

// ── History tracking ──────────────────────────────────────────────────────────

#[test]
fn session_records_history() {
    let mut session = ReplSession::new();
    session.execute(ReplCommand::Help).unwrap();
    session.execute(ReplCommand::Quit).unwrap();
    assert_eq!(session.history.len(), 2);
    assert_eq!(session.history[0].name(), "help");
    assert_eq!(session.history[1].name(), "quit");
}

// ── ReplOutcome serde ─────────────────────────────────────────────────────────

#[test]
fn outcome_is_serde_roundtrip() {
    let outcomes = vec![
        ReplOutcome::Message("hi".to_string()),
        ReplOutcome::Value(serde_json::json!({"k": 1})),
        ReplOutcome::Planned {
            action: "graph_run".to_string(),
            detail: serde_json::json!({}),
        },
        ReplOutcome::Quit,
    ];
    for o in outcomes {
        let json = serde_json::to_string(&o).unwrap();
        let back: ReplOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }
}

// ── Full parse-then-execute round-trip ───────────────────────────────────────

#[test]
fn full_session_workflow() {
    let policy = CapabilityPolicy::from_list(["lookup_user", "run"]);
    let mut session = ReplSession::new().with_policy(policy);

    // help
    let h = parse_command("help").unwrap();
    assert!(matches!(
        session.execute(h).unwrap(),
        ReplOutcome::Message(_)
    ));

    // set + get
    let s = parse_command("set region us-east").unwrap();
    session.execute(s).unwrap();
    let g = parse_command("get region").unwrap();
    let v = session.execute(g).unwrap();
    assert_eq!(v, ReplOutcome::Value(serde_json::json!("us-east")));

    // show vars
    let sv = parse_command("show vars").unwrap();
    let vars = session.execute(sv).unwrap();
    assert!(matches!(vars, ReplOutcome::Value(_)));

    // call — allowed
    let c = parse_command(r#"call lookup_user {"id": "u1"}"#).unwrap();
    assert!(matches!(
        session.execute(c).unwrap(),
        ReplOutcome::Planned { .. }
    ));

    // quit
    let q = parse_command("quit").unwrap();
    assert_eq!(session.execute(q).unwrap(), ReplOutcome::Quit);
}
