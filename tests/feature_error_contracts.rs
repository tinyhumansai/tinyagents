//! Feature tests for the crate-wide error type, [`TinyAgentsError`].
//!
//! `TinyAgentsError` is the single funnel every fallible surface rolls up
//! through, so its `Display` strings and its `From<serde_json::Error>`
//! conversion are a public contract callers match and log against. These lock
//! the documented message formats and the `?`-driven serialization conversion.
//!
//! This file needs no feature flags — the error type is always compiled.

use tinyagents::TinyAgentsError;

#[test]
fn display_strings_match_the_documented_formats() {
    let cases: Vec<(TinyAgentsError, &str)> = vec![
        (
            TinyAgentsError::MissingStart,
            "graph start node is not configured",
        ),
        (
            TinyAgentsError::MissingNode("n".into()),
            "node `n` does not exist",
        ),
        (
            TinyAgentsError::MissingEdgeTarget("t".into()),
            "edge points to missing node `t`",
        ),
        (
            TinyAgentsError::RecursionLimit(5),
            "graph exceeded the recursion limit of 5 steps",
        ),
        (
            TinyAgentsError::SubAgentDepth(8),
            "sub-agent recursion exceeded the maximum depth of 8",
        ),
        (TinyAgentsError::Model("boom".into()), "model error: boom"),
        (TinyAgentsError::Tool("nope".into()), "tool error: nope"),
        (
            TinyAgentsError::ToolNotFound("search".into()),
            "tool `search` is not registered",
        ),
        (
            TinyAgentsError::ModelNotFound("gpt".into()),
            "model `gpt` is not registered",
        ),
        (
            TinyAgentsError::Validation("empty field".into()),
            "validation error: empty field",
        ),
        (
            TinyAgentsError::LimitExceeded("cells".into()),
            "limit exceeded: cells",
        ),
        (
            TinyAgentsError::EmptyResponse,
            "model returned an empty response",
        ),
        (
            TinyAgentsError::Timeout("deadline".into()),
            "run timed out: deadline",
        ),
        (TinyAgentsError::Cancelled, "run cancelled"),
        (
            TinyAgentsError::Capability("agent `x` is not registered".into()),
            "capability error: agent `x` is not registered",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(err.to_string(), expected, "unexpected Display for {err:?}");
    }
}

#[test]
fn structured_variants_render_their_named_fields() {
    let visit = TinyAgentsError::NodeVisitLimit {
        node: "loop".into(),
        limit: 3,
    };
    assert_eq!(
        visit.to_string(),
        "node `loop` exceeded its visit limit of 3"
    );

    let route = TinyAgentsError::MissingRoute {
        node: "router".into(),
        route: "left".into(),
    };
    assert_eq!(
        route.to_string(),
        "conditional route `left` from node `router` does not exist"
    );

    let interrupted = TinyAgentsError::Interrupted {
        node: "approve".into(),
        message: "need a human".into(),
    };
    assert_eq!(
        interrupted.to_string(),
        "graph interrupted at node `approve`: need a human"
    );

    let parse = TinyAgentsError::Parse {
        message: "unexpected token".into(),
        line: 4,
        column: 12,
    };
    assert_eq!(
        parse.to_string(),
        "parse error at line 4, column 12: unexpected token"
    );
}

#[test]
fn a_serde_json_error_converts_into_a_serialization_variant() {
    let json_err = serde_json::from_str::<i32>("not a number").expect_err("must fail to parse");
    let err: TinyAgentsError = json_err.into();
    assert!(
        matches!(err, TinyAgentsError::Serialization(_)),
        "got {err:?}"
    );
    assert!(err.to_string().starts_with("serialization error:"));
}

#[test]
fn the_question_mark_operator_lifts_serde_errors_automatically() {
    fn parse(json: &str) -> Result<i32, TinyAgentsError> {
        // `?` relies on the `#[from] serde_json::Error` conversion.
        Ok(serde_json::from_str(json)?)
    }
    assert_eq!(parse("42").expect("valid"), 42);
    assert!(matches!(
        parse("bad").expect_err("invalid"),
        TinyAgentsError::Serialization(_)
    ));
}
