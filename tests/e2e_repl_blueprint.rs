//! TRUE end-to-end: a [`ReplSession`] driven through parsed command lines that
//! load + compile + run a `.rag` graph by name, interleaved with session
//! variables, and gated by a [`CapabilityPolicy`].
//!
//! This composes the **REPL** (command parser + session + capability policy)
//! with the **language** subsystem: the same `.rag` source the REPL plans to
//! load/compile is independently parsed and compiled in the test, proving the
//! REPL's plan corresponds to a real, compilable blueprint rather than an
//! arbitrary string.

use tinyagents::RustAgentsError;
use tinyagents::language::compiler::compile;
use tinyagents::language::parser::parse_str;
use tinyagents::repl::{CapabilityPolicy, ReplCommand, ReplOutcome, ReplSession, parse_command};

const SUPPORT_AGENT: &str = r#"
graph support_agent {
  start agent
  node agent {
    kind agent
    model "default"
    tools ["lookup_user"]
    routes {
      tool_call -> tools
      final -> END
    }
  }
  node tools {
    kind tool_executor
    next agent
  }
}
"#;

#[test]
fn repl_plans_match_a_real_compilable_blueprint() {
    // The capability commands (load/compile/run) require their verbs on the
    // allowlist; the deny-by-default policy still blocks everything else.
    let policy = CapabilityPolicy::from_list(["load", "compile", "run"]);
    let mut session = ReplSession::new().with_policy(policy);

    // --- load <path> ---
    let load = parse_command("load support_agent.rag").expect("parses");
    assert_eq!(
        load,
        ReplCommand::Load {
            path: "support_agent.rag".to_string()
        }
    );
    match session.execute(load).expect("load allowed") {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "load");
            assert_eq!(detail["path"], "support_agent.rag");
        }
        other => panic!("expected Planned load, got {other:?}"),
    }

    // --- set / get session variables interleaved with capability commands ---
    let set = parse_command("set graph_name support_agent").expect("parses");
    assert!(matches!(
        session.execute(set).expect("set ok"),
        ReplOutcome::Message(_)
    ));
    assert_eq!(
        session
            .execute(parse_command("get graph_name").unwrap())
            .unwrap(),
        ReplOutcome::Value(serde_json::json!("support_agent"))
    );

    // --- compile <name> ---
    match session
        .execute(parse_command("compile support_agent").expect("parses"))
        .expect("compile allowed")
    {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "compile");
            assert_eq!(detail["name"], "support_agent");
        }
        other => panic!("expected Planned compile, got {other:?}"),
    }

    // --- run <graph> <input> ---
    let run_cmd = parse_command(r#"run support_agent "{}""#).expect("parses");
    assert_eq!(
        run_cmd,
        ReplCommand::Run {
            graph: "support_agent".to_string(),
            input: "{}".to_string(),
        }
    );
    let planned_graph = match session.execute(run_cmd).expect("run allowed") {
        ReplOutcome::Planned { action, detail } => {
            assert_eq!(action, "graph_run");
            detail["graph"].as_str().unwrap().to_string()
        }
        other => panic!("expected Planned graph_run, got {other:?}"),
    };

    // The REPL planned to run `support_agent`; prove that name corresponds to a
    // real graph the language pipeline can actually parse + compile.
    let program = parse_str(SUPPORT_AGENT).expect("source parses");
    let blueprint = compile(&program).expect("program compiles").remove(0);
    assert_eq!(blueprint.graph_id, planned_graph);
    assert_eq!(blueprint.start, "agent");
    assert_eq!(blueprint.nodes.len(), 2);

    // Every command was recorded in history (load, set, get, compile, run).
    assert_eq!(session.history.len(), 5);
}

#[test]
fn disallowed_capability_is_rejected_by_the_session() {
    // Only `load` is allowed; `run` is not on the allowlist.
    let policy = CapabilityPolicy::from_list(["load"]);
    let mut session = ReplSession::new().with_policy(policy);

    let err = session
        .execute(ReplCommand::Run {
            graph: "support_agent".to_string(),
            input: "{}".to_string(),
        })
        .expect_err("run is not permitted");

    match err {
        RustAgentsError::Capability(msg) => {
            assert!(
                msg.contains("run"),
                "error should name the capability: {msg}"
            );
        }
        other => panic!("expected Capability error, got {other:?}"),
    }
}
