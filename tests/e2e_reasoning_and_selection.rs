//! End-to-end coverage for two harness features:
//!
//! **Part A — reasoning streaming.** The [`MessageDelta`] reasoning channel and
//! [`StreamAccumulator`] keep thinking output on a side channel that never
//! leaks into the final visible text. These tests drive a [`StreamingMock`]
//! through the real [`ChatModel::stream`] path, fold the items with a
//! [`StreamAccumulator`], and assert the reasoning/text split, plus the serde
//! shape of [`MessageDelta`].
//!
//! **Part B — contextual tool selection.** [`ContextualToolSelectionMiddleware`]
//! filters the tools the model is shown, either from allow/deny lists or from a
//! context-aware predicate that can vary exposure by run depth. These tests run
//! a full harness against a [`ScriptedModel`] and inspect the recorded
//! request's tool list to observe exactly what the model saw after filtering.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;

use tinyagents::harness::context::{RunConfig, RunContext};
use tinyagents::harness::events::AgentEvent;
use tinyagents::harness::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use tinyagents::harness::middleware::{ContextualToolSelectionMiddleware, ToolSelectionContext};
use tinyagents::harness::model::{
    ChatModel, ModelProfile, ModelRegistry, ModelRequest, ModelResolutionSource, ModelResponse,
    ModelSelection, ModelStatus, ModelStreamItem, StreamAccumulator,
};
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::harness::testkit::{EventRecorder, FakeTool, ScriptedModel, StreamingMock};
use tinyagents::harness::tool::ToolSchema;
use tinyagents::harness::usage::Usage;

/// Builds a plain-text [`ModelResponse`] so a [`ScriptedModel`] can answer a run
/// in a single model call (no tool execution needed).
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(3, 1)),
        },
        usage: Some(Usage::new(3, 1)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
    }
}

/// Builds the authoritative `Completed` response carrying only visible text.
fn completed_text(text: &str) -> ModelStreamItem {
    ModelStreamItem::Completed(text_response(text))
}

// ---------------------------------------------------------------------------
// Part A — reasoning streaming
// ---------------------------------------------------------------------------

/// Reasoning fragments accumulate on the side channel and stay out of the final
/// visible text, which comes from the authoritative `Completed` response.
#[tokio::test]
async fn streamed_reasoning_stays_out_of_final_text() {
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::reasoning("think ")),
        ModelStreamItem::MessageDelta(MessageDelta::text("Hello")),
        ModelStreamItem::MessageDelta(MessageDelta::reasoning("harder")),
        ModelStreamItem::MessageDelta(MessageDelta {
            text: ", world".into(),
            reasoning: "!".into(),
            tool_call: None,
        }),
        completed_text("Hello, world"),
    ];

    let model = StreamingMock::new(items);
    let request = ModelRequest::new(vec![Message::user("hi")]);

    let mut stream = ChatModel::<()>::stream(&model, &(), request)
        .await
        .expect("opening the mock stream succeeds");

    let mut accumulator = StreamAccumulator::new();
    let mut delta_count = 0usize;
    while let Some(item) = stream.next().await {
        if matches!(item, ModelStreamItem::MessageDelta(_)) {
            delta_count += 1;
        }
        accumulator.push(&item);
    }

    assert_eq!(delta_count, 4, "four message deltas were streamed");
    assert!(
        accumulator.is_terminal(),
        "the Completed item marks the accumulator terminal"
    );
    // Reasoning is the concatenation of every reasoning fragment, in order.
    assert_eq!(accumulator.reasoning(), "think harder!");

    let response = accumulator.finish().expect("stream merges into a response");
    // The visible text comes from the authoritative Completed response and does
    // NOT include any reasoning fragments.
    assert_eq!(response.text(), "Hello, world");
    assert!(
        !response.text().contains("think"),
        "reasoning must never leak into the final visible text"
    );
}

/// A `MessageDelta` round-trips through serde unchanged, and a text-only delta
/// omits the empty `reasoning` field entirely (`skip_serializing_if`).
#[test]
fn message_delta_serde_round_trip_and_field_skipping() {
    let delta = MessageDelta {
        text: "a".into(),
        reasoning: "b".into(),
        tool_call: None,
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    let back: MessageDelta = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(delta, back, "round-trip preserves every field");
    assert!(json.contains("\"reasoning\":\"b\""));

    // A text-only delta has an empty reasoning field, which is skipped in JSON.
    let text_only = MessageDelta::text("visible");
    let text_json = serde_json::to_string(&text_only).expect("serialize");
    assert!(
        !text_json.contains("reasoning"),
        "empty reasoning is skipped, got {text_json}"
    );
    assert!(text_json.contains("\"text\":\"visible\""));
}

/// The two named constructors populate exactly one channel each.
#[test]
fn message_delta_constructors_populate_one_channel() {
    let reasoning = MessageDelta::reasoning("x");
    assert_eq!(reasoning.reasoning, "x");
    assert!(reasoning.text.is_empty());
    assert!(reasoning.tool_call.is_none());

    let text = MessageDelta::text("y");
    assert_eq!(text.text, "y");
    assert!(text.reasoning.is_empty());
    assert!(text.tool_call.is_none());
}

// ---------------------------------------------------------------------------
// Part B — contextual tool selection
// ---------------------------------------------------------------------------

/// Runs a harness with the given middleware at the given depth and returns the
/// tool names the model was shown (after filtering), sorted for stable
/// assertions.
async fn exposed_tools_at_depth(
    middleware: Arc<ContextualToolSelectionMiddleware>,
    depth: usize,
) -> Vec<String> {
    let scripted = Arc::new(ScriptedModel::new(vec![text_response("done")]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", scripted.clone());
    harness.register_tool(Arc::new(FakeTool::returning("a", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("b", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("c", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("privileged", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("safe", "ok")));
    harness.push_middleware(middleware);

    let ctx = RunContext::new(RunConfig::new("run").with_depth(depth), ());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("the run completes with a single scripted text response");

    let requests = scripted.requests();
    assert_eq!(requests.len(), 1, "exactly one model call was recorded");
    let mut names: Vec<String> = requests[0].tools.iter().map(|t| t.name.clone()).collect();
    names.sort();
    names
}

/// `from_lists(Some([a, b]), [b])`: deny removes `b`, and the allow-list is
/// fail-closed so unknown tools (`c`, `privileged`, `safe`) are excluded too —
/// the model is shown only `a`.
#[tokio::test]
async fn from_lists_deny_wins_and_allow_is_fail_closed() {
    let mw = Arc::new(ContextualToolSelectionMiddleware::from_lists(
        Some(["a", "b"]),
        ["b"],
    ));
    let exposed = exposed_tools_at_depth(mw, 0).await;
    assert_eq!(exposed, vec!["a".to_string()]);
}

/// A context-aware predicate hides the `privileged` tool at sub-agent depth
/// (>0) but exposes it at the top level (depth 0). The exposed toolsets differ
/// across the two runs.
#[tokio::test]
async fn contextual_predicate_varies_exposure_by_depth() {
    fn build() -> Arc<ContextualToolSelectionMiddleware> {
        Arc::new(ContextualToolSelectionMiddleware::new(Arc::new(
            |schema: &ToolSchema, sel: &ToolSelectionContext| {
                schema.name != "privileged" || sel.depth == 0
            },
        )))
    }

    let deep = exposed_tools_at_depth(build(), 2).await;
    let top = exposed_tools_at_depth(build(), 0).await;

    assert!(
        !deep.contains(&"privileged".to_string()),
        "privileged is hidden at depth 2, got {deep:?}"
    );
    assert!(
        top.contains(&"privileged".to_string()),
        "privileged is shown at depth 0, got {top:?}"
    );
    assert_ne!(deep, top, "the exposed toolset differs across depths");
    // Non-privileged tools remain visible at both depths.
    assert!(deep.contains(&"safe".to_string()));
    assert!(top.contains(&"safe".to_string()));
}

/// `inheriting` narrows (never widens) a child's toolset relative to its
/// parent, and the narrowing is audited end-to-end via an
/// [`AgentEvent::ToolsFiltered`] event on a real harness run.
///
/// Parent allows `{a,b,c}` and denies `{c}`; the child tries to allow
/// `{b,c,d}`. The effective allow-list is the intersection `{b,c}`, then the
/// parent's deny of `c` is layered back on, so only `b` survives — the child
/// cannot re-admit `a` (never parent-allowed), `c` (parent-denied), or `d`
/// (never parent-allowed).
#[tokio::test]
async fn inheriting_narrows_child_and_emits_tools_filtered_event() {
    let scripted = Arc::new(ScriptedModel::new(vec![text_response("done")]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", scripted.clone());
    harness.register_tool(Arc::new(FakeTool::returning("a", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("b", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("c", "ok")));
    harness.register_tool(Arc::new(FakeTool::returning("d", "ok")));
    harness.push_middleware(Arc::new(ContextualToolSelectionMiddleware::inheriting(
        Some(["a", "b", "c"]),
        ["c"],
        Some(["b", "c", "d"]),
        Vec::<String>::new(),
    )));

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("inherit"), ()).with_events(recorder.sink());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("the run completes with a single scripted text response");

    // The model saw only the intersection minus the parent's deny: `b`.
    let requests = scripted.requests();
    assert_eq!(requests.len(), 1, "exactly one model call was recorded");
    let exposed: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        exposed,
        vec!["b"],
        "inheriting must narrow the child to `b`"
    );

    // The narrowing decision is auditable: exactly one ToolsFiltered event that
    // leaves one tool exposed and withholds the other three in original order.
    let filtered = recorder.events().into_iter().find_map(|e| match e {
        AgentEvent::ToolsFiltered {
            excluded,
            remaining,
            ..
        } => Some((excluded, remaining)),
        _ => None,
    });
    let (excluded, remaining) = filtered.expect("a ToolsFiltered event should be emitted");
    assert_eq!(remaining, 1, "one tool stays exposed");
    assert_eq!(
        excluded,
        vec!["a".to_string(), "c".to_string(), "d".to_string()],
        "the withheld tools are reported in original order"
    );
    assert!(
        recorder.kinds().iter().any(|k| k == "tool.filtered"),
        "the exposure decision is journaled under the tool.filtered kind"
    );
}

// ---------------------------------------------------------------------------
// Part C — retired-model resolution gating
// ---------------------------------------------------------------------------

/// A minimal [`ChatModel`] that advertises a caller-supplied [`ModelProfile`],
/// so resolution can observe its lifecycle [`ModelStatus`].
struct ProfiledModel {
    profile: ModelProfile,
}

impl ProfiledModel {
    fn with_status(status: ModelStatus) -> Self {
        Self {
            profile: ModelProfile {
                status,
                ..ModelProfile::permissive()
            },
        }
    }
}

#[async_trait]
impl ChatModel<()> for ProfiledModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyagents::Result<ModelResponse> {
        Ok(text_response("ok"))
    }
}

/// A model whose profile reports [`ModelStatus::Retired`] is skipped by
/// resolution even on an explicit override (it falls back to the live default),
/// and opting in via `allow_retired` re-admits it as the request override.
#[tokio::test]
async fn retired_override_falls_back_to_live_default_unless_allowed() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    // "live" registers first, so it is the registry default.
    registry
        .register(
            "live",
            Arc::new(ProfiledModel::with_status(ModelStatus::Stable)),
        )
        .register(
            "retired",
            Arc::new(ProfiledModel::with_status(ModelStatus::Retired)),
        );

    // A live model resolves via an explicit override.
    let live = registry
        .resolve(ModelSelection {
            requested: Some("live".into()),
            ..ModelSelection::default()
        })
        .expect("a live model resolves");
    assert_eq!(live.resolved.name, "live");
    assert_eq!(live.resolved.source, ModelResolutionSource::RequestOverride);

    // A retired override is skipped (fail closed) and resolution falls through to
    // the live registry default rather than selecting the retired model.
    let fell_back = registry
        .resolve(ModelSelection {
            requested: Some("retired".into()),
            ..ModelSelection::default()
        })
        .expect("resolution falls back to the live default");
    assert_eq!(
        fell_back.resolved.name, "live",
        "a retired override must not be selected; it falls back to the live default"
    );
    assert_eq!(
        fell_back.resolved.source,
        ModelResolutionSource::RegistryDefault
    );

    // `allow_retired = true` re-admits the retired model as the request override.
    let readmitted = registry
        .resolve(ModelSelection {
            requested: Some("retired".into()),
            allow_retired: true,
            ..ModelSelection::default()
        })
        .expect("allow_retired re-admits the retired model");
    assert_eq!(readmitted.resolved.name, "retired");
    assert_eq!(
        readmitted.resolved.source,
        ModelResolutionSource::RequestOverride
    );
}

/// When every candidate is retired, default resolution yields nothing;
/// `allow_retired` makes the retired default resolvable again.
#[tokio::test]
async fn all_retired_registry_resolves_to_none_until_allowed() {
    let mut registry: ModelRegistry<()> = ModelRegistry::new();
    registry.register(
        "only_retired",
        Arc::new(ProfiledModel::with_status(ModelStatus::Retired)),
    );

    assert!(
        registry.resolve(ModelSelection::default()).is_none(),
        "a registry with only a retired default resolves to None"
    );

    let allowed = registry
        .resolve(ModelSelection {
            allow_retired: true,
            ..ModelSelection::default()
        })
        .expect("allow_retired admits the retired default");
    assert_eq!(allowed.resolved.name, "only_retired");
}
