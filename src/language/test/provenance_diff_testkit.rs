//! Provenance, diff, and language-testkit tests (H4).
//!
//! Split out of `language/test/mod.rs` by pipeline phase.

use super::*;
use crate::language::source::SourceFile;

// ---------------------------------------------------------------------------
// Provenance, diff, and language testkit (H4)
// ---------------------------------------------------------------------------

use crate::language::compiler::compile_with_provenance;
use crate::language::diff::{FieldChange, blueprint_diff};
use crate::language::testkit as lang_testkit;
use crate::language::types::Origin;

const DIFF_BASE: &str = r#"
graph flow {
  start plan

  channel messages append

  node plan {
    kind model
    model "default"
    routes {
      research -> work
      done -> END
    }
  }

  node work {
    kind tool_executor
    tools ["lookup_user"]
    next END
  }
}
"#;

/// Adds a node (`review`) and changes a route target on `plan`
/// (`research -> review` instead of `research -> work`).
const DIFF_NEW: &str = r#"
graph flow {
  start plan

  channel messages append

  node plan {
    kind model
    model "default"
    routes {
      research -> review
      done -> END
    }
  }

  node review {
    kind interrupt
    prompt "ok?"
    next work
  }

  node work {
    kind tool_executor
    tools ["lookup_user"]
    next END
  }
}
"#;

#[test]
fn blueprint_diff_reports_added_node_and_changed_route() {
    let old = lang_testkit::blueprint(DIFF_BASE);
    let new = lang_testkit::blueprint(DIFF_NEW);

    let diff = blueprint_diff(&old, &new);
    assert!(!diff.is_empty());

    // One node added: `review`.
    assert_eq!(diff.nodes_added, vec!["review".to_string()]);
    assert!(diff.nodes_removed.is_empty());

    // `plan`'s routing changed (research target work -> review).
    assert_eq!(diff.nodes_changed.len(), 1, "{diff:?}");
    let plan_change = &diff.nodes_changed[0];
    assert_eq!(plan_change.name, "plan");
    assert_eq!(
        plan_change.fields,
        vec![FieldChange {
            field: "routing".to_string(),
            old: "{ research -> work, done -> END }".to_string(),
            new: "{ research -> review, done -> END }".to_string(),
        }]
    );

    // The rendered summary names both the added node and the changed route.
    let rendered = diff.to_string();
    assert!(rendered.contains("+ node review"), "{rendered}");
    assert!(rendered.contains("~ node plan"), "{rendered}");
    assert!(rendered.contains("routing:"), "{rendered}");
}

#[test]
fn blueprint_diff_of_identical_blueprints_is_empty() {
    let a = lang_testkit::blueprint(DIFF_BASE);
    let b = lang_testkit::blueprint(DIFF_BASE);
    let diff = blueprint_diff(&a, &b);
    assert!(diff.is_empty(), "{diff:?}");
    assert_eq!(diff.to_string(), "no changes");
}

const DIFF_COMMAND_BASE: &str = r#"
graph flow2 {
  start plan

  node plan {
    kind agent
    model "default"
    command {
      goto work
      update {
        status "planned"
      }
    }
  }

  node work {
    kind tool_executor
    next END
  }
}
"#;

/// Only `plan`'s command `update` value changes (`"planned"` -> `"revised"`);
/// topology, routing target, and everything else stays identical.
const DIFF_COMMAND_NEW: &str = r#"
graph flow2 {
  start plan

  node plan {
    kind agent
    model "default"
    command {
      goto work
      update {
        status "revised"
      }
    }
  }

  node work {
    kind tool_executor
    next END
  }
}
"#;

#[test]
fn blueprint_diff_reports_command_only_change() {
    let old = lang_testkit::blueprint(DIFF_COMMAND_BASE);
    let new = lang_testkit::blueprint(DIFF_COMMAND_NEW);

    let diff = blueprint_diff(&old, &new);
    assert!(!diff.is_empty(), "command-only change must not be silent");
    assert!(diff.nodes_added.is_empty());
    assert!(diff.nodes_removed.is_empty());

    assert_eq!(diff.nodes_changed.len(), 1, "{diff:?}");
    let plan_change = &diff.nodes_changed[0];
    assert_eq!(plan_change.name, "plan");
    assert!(
        plan_change.fields.iter().any(|f| f.field == "command"),
        "{plan_change:?}"
    );

    let rendered = diff.to_string();
    assert!(rendered.contains("command:"), "{rendered}");
}

#[test]
fn blueprint_diff_serializes_round_trip() {
    let old = lang_testkit::blueprint(DIFF_BASE);
    let new = lang_testkit::blueprint(DIFF_NEW);
    let diff = blueprint_diff(&old, &new);
    let json = serde_json::to_string(&diff).unwrap();
    let back: crate::language::diff::BlueprintDiff = serde_json::from_str(&json).unwrap();
    assert_eq!(diff, back);
}

#[test]
fn provenance_points_each_node_at_its_span() {
    let program = parse_str(DIFF_NEW).unwrap();
    let bp = compile_with_provenance(&program, Origin::file("flow.rag"))
        .unwrap()
        .remove(0);

    let prov = bp.provenance().expect("provenance attached");
    assert_eq!(prov.origin, Origin::File("flow.rag".to_string()));

    // Every node in the blueprint has a recorded span anchored at the line of
    // its `node <name>` declaration, and the byte range slices back to the
    // `node` keyword that opens it.
    let file = SourceFile::new("flow.rag", DIFF_NEW);
    for node in &bp.nodes {
        let span = prov
            .node_span(&node.name)
            .unwrap_or_else(|| panic!("no span for node `{}`", node.name));
        // The span's byte range covers the opening `node` keyword.
        assert_eq!(&DIFF_NEW[span.start..span.end], "node");
        // The line the span anchors at is the node's declaration line.
        let line = file
            .line_text(span.line)
            .unwrap_or_else(|| panic!("no source line {}", span.line));
        assert!(
            line.contains(&format!("node {}", node.name)),
            "node `{}` span anchors at line `{line}`, not its declaration",
            node.name
        );
    }

    // The channel span is recorded too.
    assert!(prov.channel_span("messages").is_some());
}

#[test]
fn plain_compile_leaves_provenance_none() {
    let bp = lang_testkit::blueprint(DIFF_BASE);
    assert!(bp.provenance().is_none());
}

#[test]
fn generated_origin_renders_label() {
    let program = parse_str(DIFF_BASE).unwrap();
    let bp = compile_with_provenance(&program, Origin::generated_by("repl-7"))
        .unwrap()
        .remove(0);
    let prov = bp.provenance().unwrap();
    assert_eq!(prov.origin.as_display(), "generated by repl-7");
}

#[test]
fn testkit_assertions_inspect_lowered_topology() {
    let bp = lang_testkit::blueprint(DIFF_NEW);
    lang_testkit::assert_kind(&bp, "review", "interrupt");
    lang_testkit::assert_next(&bp, "review", "work");
    lang_testkit::assert_terminal(&bp, "work");
    lang_testkit::assert_route(&bp, "plan", "research", "review");
}

#[test]
fn testkit_try_compile_surfaces_errors() {
    // `start` references an undefined node, so compilation fails.
    let err = lang_testkit::try_compile("graph g { start missing node a { kind model next END } }")
        .unwrap_err();
    assert!(matches!(err, crate::error::TinyAgentsError::Compile(_)));
}

/// Minimal fake model/tool used to populate a [`CapabilityRegistry`] in tests.
pub(super) mod testkit {
    use async_trait::async_trait;
    use serde_json::json;

    use crate::error::Result;
    use crate::harness::model::{ChatModel, ModelRequest, ModelResponse};
    use crate::harness::tool::{Tool, ToolCall, ToolResult, ToolSchema};

    use super::TestState;

    pub(crate) struct EchoModel;

    #[async_trait]
    impl ChatModel<TestState> for EchoModel {
        async fn invoke(
            &self,
            _state: &TestState,
            _request: ModelRequest,
        ) -> Result<ModelResponse> {
            Ok(ModelResponse::assistant("echo"))
        }
    }

    pub(crate) struct NoopTool;

    #[async_trait]
    impl Tool<TestState> for NoopTool {
        fn name(&self) -> &str {
            "lookup_user"
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("lookup_user", "noop", json!({"type": "object"}))
        }
        async fn call(&self, _state: &TestState, call: ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::text(call.id, call.name, "ok"))
        }
    }
}
