use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;
use tinyagents::error::{Result, TinyAgentsError};
use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::{AgentEvent, RecordingListener};
use tinyagents::harness::message::Message;
use tinyagents::harness::middleware::{
    AgentRun, DynamicPromptMiddleware, DynamicToolSelectionMiddleware, HumanApprovalMiddleware,
    LoggingMiddleware, Middleware, MiddlewareModelOutcome, MiddlewareStack, MiddlewareToolOutcome,
    ModelBaseCall, ModelFallbackMiddleware, ModelMiddleware, RedactionMiddleware,
    StructuredOutputValidatorMiddleware, TimeoutMiddleware, ToolAllowlistMiddleware, ToolBaseCall,
    ToolHandler, ToolMiddleware, TracingMiddleware,
};
use tinyagents::harness::model::{ModelDelta, ModelRequest, ModelResponse, ResponseFormat};
use tinyagents::harness::tool::{ToolCall, ToolDelta, ToolResult, ToolSchema};
use tinyagents::language::{lexer, parser};

struct ModelBase {
    seen_models: Mutex<Vec<Option<String>>>,
    fail_until: usize,
}

impl ModelBase {
    fn new(fail_until: usize) -> Self {
        Self {
            seen_models: Mutex::new(Vec::new()),
            fail_until,
        }
    }
}

impl ModelBaseCall<(), ()> for ModelBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext<()>,
        _state: &'a (),
        request: ModelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let mut seen = self.seen_models.lock().unwrap();
            seen.push(request.model.clone());
            if seen.len() <= self.fail_until {
                return Err(TinyAgentsError::Model("retry me".into()));
            }
            Ok(ModelResponse::assistant(format!(
                "model={}",
                request.model.as_deref().unwrap_or("default")
            )))
        })
    }
}

struct ToolBase;

impl ToolBaseCall<(), ()> for ToolBase {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext<()>,
        _state: &'a (),
        call: ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move { Ok(ToolResult::text(call.id, call.name, "tool-ok")) })
    }
}

struct ToolShortCircuit;

#[async_trait]
impl ToolMiddleware<(), ()> for ToolShortCircuit {
    fn name(&self) -> &str {
        "tool_short"
    }

    async fn wrap_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        call: ToolCall,
        _next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        Ok(ToolResult::text(call.id, call.name, "shorted").into())
    }
}

struct ModelReplace;

#[async_trait]
impl ModelMiddleware<(), ()> for ModelReplace {
    fn name(&self) -> &str {
        "model_replace"
    }

    async fn wrap_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _request: ModelRequest,
        _next: tinyagents::harness::middleware::ModelHandler<'_, (), ()>,
    ) -> Result<MiddlewareModelOutcome> {
        Ok(ModelResponse::assistant("replaced").into())
    }
}

fn context() -> (RunContext<()>, Arc<RecordingListener>) {
    let recorder = Arc::new(RecordingListener::new());
    let ctx = RunContext::new(RunConfig::new("run-mw").with_thread("thread-mw"), ());
    ctx.events.subscribe(recorder.clone());
    (ctx, recorder)
}

#[tokio::test]
async fn middleware_stack_runs_lifecycle_hooks_and_builtin_guards() {
    let (mut ctx, recorder) = context();
    let logging = Arc::new(LoggingMiddleware::with_label("log"));
    let tracing = Arc::new(TracingMiddleware::new());
    let redaction = Arc::new(RedactionMiddleware::with_mask(["secret"], "***"));
    let allowlist = Arc::new(ToolAllowlistMiddleware::new(["lookup"]));
    let selector = Arc::new(DynamicToolSelectionMiddleware::allowing(["lookup"]));
    let prompt = Arc::new(DynamicPromptMiddleware::<(), ()>::from_fn(|_, config| {
        Some(format!("run {}", config.run_id.as_str()))
    }));

    let mut stack: MiddlewareStack<(), ()> = MiddlewareStack::new();
    assert!(stack.is_empty());
    stack.push(logging.clone());
    stack.push(tracing.clone());
    stack.push(redaction.clone());
    stack.push(allowlist.clone());
    stack.push(selector.clone());
    stack.push(prompt.clone());
    assert_eq!(stack.len(), 6);

    stack.run_before_agent(&mut ctx, &()).await.unwrap();
    let mut request = ModelRequest::new(vec![Message::user("hello")]).with_tools(vec![
        ToolSchema::new("lookup", "allowed", json!({ "type": "object" })),
        ToolSchema::new("delete", "blocked", json!({ "type": "object" })),
    ]);
    stack
        .run_before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();
    assert_eq!(request.messages[0].text(), "run run-mw");
    assert_eq!(
        request
            .tools
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        vec!["lookup"]
    );

    let mut delta = ModelDelta {
        call_id: "model-1".into(),
        content: "piece".into(),
        reasoning: String::new(),
        tool_call: None,
    };
    stack
        .run_on_model_delta(&mut ctx, &(), &mut delta)
        .await
        .unwrap();

    let mut response = ModelResponse::assistant("secret response");
    stack
        .run_after_model(&mut ctx, &(), &mut response)
        .await
        .unwrap();
    assert_eq!(response.text(), "*** response");

    let mut allowed = ToolCall::new("tool-1", "lookup", json!({}));
    stack
        .run_before_tool(&mut ctx, &(), &mut allowed)
        .await
        .unwrap();
    let mut blocked = ToolCall::new("tool-2", "delete", json!({}));
    let err = stack
        .run_before_tool(&mut ctx, &(), &mut blocked)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not on the allowlist"));

    let mut tool_delta = ToolDelta {
        call_id: "tool-1".into(),
        content: "progress".into(),
        tool_name: None,
    };
    stack
        .run_on_tool_delta(&mut ctx, &(), &mut tool_delta)
        .await
        .unwrap();
    let mut tool_result = ToolResult::text("tool-1", "lookup", "secret tool result");
    stack
        .run_after_tool(&mut ctx, &(), &mut tool_result)
        .await
        .unwrap();
    assert_eq!(tool_result.content, "*** tool result");

    let mut run = AgentRun::new();
    run.final_response = Some(response.clone());
    assert_eq!(run.text().as_deref(), Some("*** response"));
    stack
        .run_after_agent(&mut ctx, &(), &mut run)
        .await
        .unwrap();
    stack
        .run_on_error(&mut ctx, &TinyAgentsError::Model("boom".into()))
        .await
        .unwrap();

    let counts = logging.counts();
    assert_eq!(counts.before_agent, 1);
    assert_eq!(counts.before_model, 1);
    assert_eq!(counts.on_model_delta, 1);
    assert_eq!(counts.after_model, 1);
    assert_eq!(counts.before_tool, 2);
    assert_eq!(counts.on_tool_delta, 1);
    assert_eq!(counts.after_tool, 1);
    assert_eq!(counts.after_agent, 1);
    assert!(counts.on_error >= 2);
    assert_eq!(redaction.redactions(), 2);
    assert!(tracing.records().iter().any(|r| r.phase == "agent"));
    assert!(recorder.events().iter().any(|r| matches!(
        r.event,
        AgentEvent::MiddlewareStarted { ref name } if name == "log"
    )));
}

#[tokio::test]
async fn builtin_middleware_validates_structured_output_human_approval_and_wraps_calls() {
    let (mut ctx, recorder) = context();

    let approval =
        HumanApprovalMiddleware::new(["danger"]).with_approval(Arc::new(|call: &ToolCall| {
            call.arguments["approved"] == true
        }));
    let mut approved = ToolCall::new("tool-1", "danger", json!({ "approved": true }));
    approval
        .before_tool(&mut ctx, &(), &mut approved)
        .await
        .unwrap();
    let mut rejected = ToolCall::new("tool-2", "danger", json!({ "approved": false }));
    let err = approval
        .before_tool(&mut ctx, &(), &mut rejected)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("requires human approval"));

    let json_validator = StructuredOutputValidatorMiddleware::new(ResponseFormat::JsonObject);
    let mut valid = ModelResponse::assistant(r#"{"ok":true}"#);
    json_validator
        .after_model(&mut ctx, &(), &mut valid)
        .await
        .unwrap();
    let mut invalid = ModelResponse::assistant("not json");
    assert!(
        json_validator
            .after_model(&mut ctx, &(), &mut invalid)
            .await
            .unwrap_err()
            .to_string()
            .contains("not valid JSON")
    );

    let schema_validator = StructuredOutputValidatorMiddleware::new(ResponseFormat::json_schema(
        "answer",
        json!({ "type": "object" }),
    ));
    let mut schema_response = ModelResponse::assistant(r#"{"answer":1}"#);
    schema_validator
        .after_model(&mut ctx, &(), &mut schema_response)
        .await
        .unwrap();
    let text_validator = StructuredOutputValidatorMiddleware::new(ResponseFormat::Text);
    let mut text_response = ModelResponse::assistant("plain");
    text_validator
        .after_model(&mut ctx, &(), &mut text_response)
        .await
        .unwrap();

    let mut fallback_stack: MiddlewareStack<(), ()> = MiddlewareStack::new();
    fallback_stack.push_model_middleware(Arc::new(ModelFallbackMiddleware::new(["small", "tiny"])));
    assert_eq!(fallback_stack.model_middleware_len(), 1);
    let base = ModelBase::new(1);
    let outcome = fallback_stack
        .run_wrapped_model(
            &mut ctx,
            &(),
            ModelRequest::new(vec![Message::user("hi")]).with_model("large"),
            &base,
        )
        .await
        .unwrap()
        .into_response();
    assert_eq!(outcome.text(), "model=small");
    assert_eq!(
        base.seen_models.lock().unwrap().clone(),
        vec![Some("large".into()), Some("small".into())]
    );
    assert!(recorder.events().iter().any(|r| matches!(
        r.event,
        AgentEvent::FallbackSelected { ref from, ref to } if from == "large" && to == "small"
    )));

    let mut replace_stack: MiddlewareStack<(), ()> = MiddlewareStack::new();
    replace_stack.push_model_middleware(Arc::new(ModelReplace));
    let replaced = replace_stack
        .run_wrapped_model(
            &mut ctx,
            &(),
            ModelRequest::new(vec![Message::user("hi")]),
            &ModelBase::new(0),
        )
        .await
        .unwrap()
        .into_response();
    assert_eq!(replaced.text(), "replaced");

    let mut tool_stack: MiddlewareStack<(), ()> = MiddlewareStack::new();
    tool_stack.push_tool_middleware(Arc::new(ToolShortCircuit));
    assert_eq!(tool_stack.tool_middleware_len(), 1);
    let tool_result = tool_stack
        .run_wrapped_tool(
            &mut ctx,
            &(),
            ToolCall::new("tool-1", "lookup", json!({})),
            &ToolBase,
        )
        .await
        .unwrap()
        .into_result();
    assert_eq!(tool_result.content, "shorted");

    let timeout = TimeoutMiddleware::from_millis(1);
    assert_eq!(
        <TimeoutMiddleware as ModelMiddleware<(), ()>>::name(&timeout),
        "timeout"
    );
}

#[test]
fn parser_accepts_full_language_shapes_and_reports_caret_errors() {
    let source = r#"
graph workflow {
  defaults { mode "fast" retries 2 }
  input { question string }
  output { answer string }
  checkpoint inherit
  interrupt manual
  channel messages append
  channel facts aggregate "facts"
  start plan
  node plan {
    kind agent
    model "planner"
    tools ["search"]
    agent "researcher"
    next decide
  }
  node decide {
    routes {
      ok -> answer
      retry -> plan
    }
  }
  node fanout {
    sends [
      send worker "a",
      send worker "b"
    ]
  }
  node answer {
    command { goto END }
  }
  join [plan, fanout] -> answer
  plan -> decide
}
"#;

    let tokens = lexer::tokenize(source).unwrap();
    let parsed_from_tokens = parser::parse(&tokens).unwrap();
    let parsed = parser::parse_str(source).unwrap();
    assert_eq!(parsed.graphs[0].name, "workflow");
    assert_eq!(parsed.graphs[0].defaults.len(), 2);
    assert_eq!(parsed.graphs[0].input[0].name, "question");
    assert_eq!(parsed.graphs[0].checkpoint.as_deref(), Some("inherit"));
    assert_eq!(parsed.graphs[0].interrupt.as_deref(), Some("manual"));
    assert_eq!(parsed.graphs[0].channels.len(), 2);
    assert_eq!(parsed.graphs[0].nodes.len(), 4);
    assert_eq!(parsed.graphs[0].joins.len(), 1);
    assert_eq!(parsed_from_tokens.graphs[0].name, parsed.graphs[0].name);

    let err = parser::parse_str("graph broken { node x { route { ok -> } } }").unwrap_err();
    let rendered = err.to_string();
    assert!(rendered.contains("error:"));
    assert!(rendered.contains("-->"));

    let eof = parser::parse_str("graph broken { defaults { key").unwrap_err();
    let eof_rendered = eof.to_string();
    assert!(eof_rendered.contains("error:"));
    assert!(eof_rendered.contains("-->"));
}
