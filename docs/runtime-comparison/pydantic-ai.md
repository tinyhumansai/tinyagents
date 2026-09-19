# Pydantic AI vs TinyAgents — runtime comparison

Researched 2026-09-19 against primary sources (pydantic.dev/docs/ai, the
`pydantic/pydantic-ai` repo and release list, the v2 announcement article).
TinyAgents baseline: v2.1.2 (`fc33c43`), vendored `tinyinference` at `219b0ea`
and `tinytools` at `a14e24d`.

## 1. What Pydantic AI is today

- **Versions.** v1.0.0 shipped 2025-09-04. v2.0.0 shipped 2026-06-23 after
  seven betas (b1 2026-05-20). Release cadence since is near-daily minors:
  v2.46.0 was published 2026-09-19 (today); the 1.x line is still patched
  (v1.107.6, 2026-09-17). ~20k GitHub stars, MIT.
- **Architecture (v2).** A single typed `Agent[DepsT, OutputT]` whose run is a
  `pydantic_graph` graph of four node classes: `UserPromptNode -> ModelRequestNode
  -> CallToolsNode -> End` (`_agent_graph.py`, still built on
  `pydantic_graph.BaseNode/Graph/GraphBuilder`). Graph state is
  `GraphAgentState { message_history, usage, output_retries_used, run_step,
  run_id (uuid7), conversation_id, metadata, pending_messages, ... }`.
- **The v2 primitive is the Capability.** "A single, composable unit that
  bundles an agent's tools, hooks, instructions, and model settings." Almost
  every v1 `Agent(...)` knob (`history_processors`, `prepare_tools`,
  `event_stream_handler`, `instrument`, `mcp_servers`, `builtin_tools`)
  migrated to a capability (`ProcessHistory`, `PrepareTools`,
  `ProcessEventStream`, `Instrumentation`, `MCP`, native tools such as
  `WebSearch`/`WebFetch`/`Thinking`). Durable execution (Temporal, DBOS,
  Prefect, Restate, Kitaru, Airflow, AWS Lambda) is also a capability
  (`TemporalDurability()` replaces the deprecated `TemporalAgent` wrapper).
- **Package split.** `pydantic-ai-slim` (loop, providers, capability/hook
  API), `pydantic-ai` (meta), `pydantic-graph`, `pydantic-evals`, and the new
  `pydantic-ai-harness` ("official capability and harness library":
  `Coder`, `FileSystem`, `Shell`, `Planning`, `Memory`, `Guardrails`,
  `CodeMode`, `Compaction`, `StepPersistence`, `Subagents`, `Advisor`,
  `SpendLimits`, `ToolOutputLimits`, `SystemReminders`, `WarnOnCacheBusts`,
  `ConversationSearch`, `Skills`, `RepoContext`, ...). Long-tail providers
  (Bedrock, Groq, Mistral, Cohere, xAI) became opt-in extras.
- **Other v2 changes worth knowing.** `pydantic_graph.persistence` and
  `pydantic_graph.mermaid` were removed from pydantic-graph (persistence moved
  to Harness `StepPersistence`; `graph.render()` still emits mermaid from the
  builder API). `ModelProfile` became a `TypedDict`. `end_strategy` default is
  now `'graceful'`. Instrumentation defaults to GenAI semconv "version 5"
  (`gen_ai.aggregated_usage.*`). Generic default deps type is `object`, not
  `None`. `openai:` now means the Responses API. `DeferredToolCalls` was
  renamed `DeferredToolRequests`; `DeferredToolset` became `ExternalToolset`.
  Agent Specs (YAML/JSON `Agent.from_file`) and a pending-message queue
  (`ctx.enqueue` / `agent_run.enqueue`, priorities `'asap'|'when_idle'`) landed
  in 1.101 and are core in v2.
- **Positioning vs LangGraph** (their comparison page): "one typed Agent with
  plain Python control flow: pydantic-graph when you want an explicit graph
  ... Reach for it when the control flow is a real state machine; plain
  Python and sub-agents cover the rest." They deliberately do not ship a
  LangGraph-style checkpointer; durability is delegated to external engines
  and message-history persistence.

## 2. Feature inventory

Paths are relative to this repository unless prefixed `vendor:`
(`vendor/tinyinference` / `vendor/tinytools`). H = `crates/tinyagents-harness/src`,
G = `crates/tinyagents-graph/src`.

| Feature | Pydantic AI | TinyAgents | Notes |
|---|---|---|---|
| Typed deps injection into tools/instructions/validators | `Agent[DepsT, OutputT]`, `RunContext[DepsT].deps` | **Yes** — `RunContext<Ctx>` (`H/context/types.rs`), `Middleware<State, Ctx>`, `ChatModel<State>` | TinyAgents also threads `&State`; Pydantic has no separate State at agent level. |
| Typed output (`output_type=T`) | `RunResult[OutputT].output`, Pydantic-validated | **Partial** — `AgentRun.structured: Option<serde_json::Value>` (`H/middleware/types.rs:93`), JSON-Schema validated (`H/structured/validate.rs`) | Untyped `Value`; caller deserializes. |
| Output modes: tool / native / prompted / text-function | `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, unions, `StructuredDict`, `Choices`, output functions | **Partial** — `StructuredStrategy::{ProviderSchema, ToolCall}` (`H/structured/types.rs:27`) | No prompted mode, no union-as-multiple-tools, no output functions. |
| Output-validation retry loop (`ModelRetry` from `@output_validator`) | Yes; consumes `retries['output']`, sends `RetryPromptPart` | **No** — final-turn `extractor.extract(&response)?` is fatal (`H/agent_loop/run_loop.rs:788-793`); `StructuredOutcome` exists as data only (`H/structured/types.rs:72`); `StructuredOutputValidatorMiddleware` errors, does not re-ask (`H/middleware/library/observe.rs`) | Repair ladder (`H/structured/repair.rs`) exists but no re-prompt. |
| Tool arg validation errors fed back to model | `RetryPromptPart` + per-tool `retries` | **Yes** — `InvalidArgsPolicy`, `AgentEvent::InvalidToolArgs` (`H/agent_loop/tools.rs:270-290`), `UnknownToolPolicy` (`H/runtime/types.rs:99-125`) | |
| Tool-raised "retry me" (`ModelRetry`) vs "failed, don't retry" (`ToolFailed`) | Both, with per-tool budgets, `ctx.retry`/`ctx.max_retries` | **Partial** — `ToolResult::error` (is_error) only (`vendor:tinytools/src/result/types.rs`); `ToolRuntime.max_retries` declared but host-applied; no per-tool retry counter in loop | |
| Deferred tools / stop-and-resume HITL with typed IDs | `DeferredToolRequests{calls, approvals, metadata}` as output; `DeferredToolResults`; `ApprovalRequired`, `CallDeferred`, `ToolApproved(override_args)`, `ToolDenied(message)`; `agent.run(message_history=, deferred_tool_results=)` | **Partial** — harness: `HumanApprovalMiddleware` -> `Err(TinyAgentsError::Interrupted)` (`H/agent_loop/run_loop.rs:871`); graph: `Interrupt`/`resume` (`G/command/types.rs:107`, `G/compiled/executor.rs:143`); delegation `PendingApproval` (`G/delegation`) | No typed per-call-id request/result pair at harness level; no external-execution (`CallDeferred`) concept. |
| In-run approval handler | `HandleDeferredToolCalls(handler=...)` capability | **Partial** — `HumanApprovalMiddleware::approve: ApprovalFn` (`H/middleware/library/types.rs:393`) | |
| Rich tool return (value for model + separate content + app metadata + images) | `ToolReturn(return_value, content=[BinaryContent...], metadata, tools)` | **No** — `ToolContent::{Text, Json}` only (`vendor:tinytools/src/result/types.rs:149`) | |
| Toolset composition | `FunctionToolset`, `CombinedToolset`, `.filtered()`, `.prefixed()`, `.renamed()`, `.prepared()`, `.approval_required()`, `.defer_loading()`, `WrapperToolset`, `ExternalToolset`, `MCPToolset`, `LangChainToolset`; `@agent.toolset` dynamic | **Partial** — `ToolRegistry` (`H/tool/mod.rs`), `ToolAllowlistMiddleware`, `DynamicToolSelectionMiddleware`, `ContextualToolSelectionMiddleware`, `ToolPolicyMiddleware` (`H/middleware/library/types.rs:163-393`), `ToolExposure::{Direct,Deferred,Hidden}` + `tool/select` ranking | No composable toolset objects, no prefix/rename wrappers, no external toolset. |
| Per-tool `prepare` / agent-wide `prepare_tools` | Yes (`ToolDefinition | None` per step) | **Partial** — middleware `before_model` can rewrite `ModelRequest.tools`; `schema_prepare.rs` for strict-shaping | No per-tool hook. |
| Strict schema mode | `strict=True/False/None`, provider-aware | **Partial** — `SchemaPreparation` (`require_all_properties`, `additional_properties=false`) (`H/tool/schema_prepare.rs`) | No per-tool flag on the trait. |
| Docstring -> schema extraction | griffe, `docstring_format`, `require_parameter_descriptions` | **N/A** — Rust; `parameters_schema()` hand-written | Not comparable. |
| Sequential vs parallel tool execution | default concurrent; `sequential=True` per tool; `parallel_tool_call_execution_mode('sequential')` | **Yes** — `is_concurrency_safe()` default `false` (`vendor:tinytools/src/tool/types.rs`), serial admission/concurrent fold (`H/agent_loop/tools.rs`) | Opposite default (TinyAgents fails safe). |
| Tool timeouts | `timeout=` per tool, `tool_timeout=` agent-wide; timeout -> retry prompt | **Yes** — `ToolTimeout::{Inherit,Millis,Unbounded}`, `with_tool_timeout_settings` (`H/tool/timeout.rs`) | |
| Native/provider tools (web search, code exec, image gen) | `WebSearch`, `WebFetch`, `XSearch`, `ImageGeneration`, `MCP(native=True)`; `BuiltinToolCallPart` | **No** — no provider-executed tool parts in `ContentBlock` (`vendor:tinyinference-llm/src/message/types.rs:21`) | |
| On-demand capability loading / tool search | `defer_loading=True`, `load_capability` tool, `ToolSearch`, `ctx.loaded_capability_ids` reconstructed from history | **Partial** — `ToolExposure::Deferred` + `H/tool/select` (keyword ranking) | No bundle (instructions+tools+settings) loading. |
| Model fallback | `FallbackModel(fallback_on=exceptions | response predicates)` | **Yes** — `FallbackPolicy` (`H/retry/types.rs:183`), `ModelFallbackMiddleware`, `CapabilitySet`-filtered resolution (`H/model_registry`) | TinyAgents resolution is richer (hints, capabilities). |
| Wrapper/decorator models | `WrapperModel`, `InstrumentedModel`, `ConcurrencyLimitedModel` | **Yes** — `ProfileOverrideModel`, `MaxTokensModel`, `RouteRecordingModel`, `ObservingModel` (`vendor:tinyinference-llm/src/model/decorators.rs`), `RateLimiter` | |
| Model profiles | `ModelProfile` TypedDict: `json_schema_transformer`, `default_structured_output_mode`, `prompted_output_template`, `thinking_tags`, `supported_native_tools`, `tool_deferral_mode`, `context_window`... | **Partial** — `ModelProfile` (`vendor:tinyinference-llm/src/model/types.rs:193`): modalities, tool calling, streaming chunks, native structured output, reasoning, token windows | No schema transformer, no thinking-tag parsing, no prompted-output template. |
| Model settings precedence (model < agent < run), callable per step | Yes | **Yes** — `ModelRequestDefaults`, request overrides (`H/model_registry`) | |
| Usage accounting + cost | `RunUsage{requests,tool_calls,input/output,cache_read/write,audio, cost: Decimal|None, details}`; cost via `genai-prices` | **Yes** — `Usage`, `UsageTotals`, `ModelPricing`, `CostTotals` (`H/cost/types.rs`), catalog prices (`vendor:tinyinference-llm/src/catalog/types.rs:20`) | Pydantic's price DB is an external, maintained package. |
| Usage limits | `UsageLimits{request_limit=50, tool_calls_limit, input/output/total_tokens_limit, per_request_input_tokens_limit, cost_limit, count_tokens_before_request}` -> `UsageLimitExceeded` | **Yes** — `RunLimits`, `LimitBehavior` (`H/limits/types.rs`), `BudgetMiddleware`/`BudgetLimits` (`H/middleware/library/budget.rs`) | |
| Shared usage across delegated agents | `delegate.run(..., usage=ctx.usage)` | **Yes** — parent/child usage roll-up in `subagent`, `RunTree` (`G/recursion`) | |
| Message history in/out, JSON round-trip | `message_history=`, `all_messages()/new_messages()`, `ModelMessagesTypeAdapter`, `sanitize_messages()` | **Yes** — `Message` serde types; `tinyagents-session` for durable history | No sanitize-untrusted-history helper. |
| History processors / compaction | `ProcessHistory(fn)`; Harness `Compaction` (`ClearToolResults`, `DeduplicateFileReads`, `SlidingWindow`, `ClampOversized`, `SummarizingCompaction`, `TieredCompaction`, `max_fraction`) | **Yes** — `MessageTrimMiddleware`, `ContextCompressionMiddleware`, `MicrocompactMiddleware`, `summarization/`, `handoff.rs` (progressive disclosure) | Comparable breadth. |
| Prompt-cache awareness | `WarnOnCacheBusts`, `SystemReminders` (cache-safe), `CachePoint` | **Yes** — `PromptCacheGuardMiddleware`, `PromptSegment`/`SegmentRole`, `cache/layout.rs` | TinyAgents is at least as explicit. |
| Pending message queue / mid-run steering | `ctx.enqueue(msg, priority='asap'|'when_idle')`, `agent_run.enqueue` | **Yes** — `RunQueue` lanes `Steer/Followup/Collect` (`H/run_queue/types.rs`), `steering/` | TinyAgents is broader (policy-checked commands). |
| Cancellation | `ctx.cancel()`, `CancellationToken`, `RunCancelled` | **Yes** — `CancellationToken` (`H/cancel`) | |
| Lifecycle hooks | `Hooks`: before/after/wrap/error for run, node, model_request, tool_validate, tool_execute, output_validate, output_process; `SkipModelRequest`, `SkipToolExecution`; per-tool filter; timeouts | **Yes** — `Middleware` before/after model+tool, `ModelMiddleware`/`ToolMiddleware` wrap, `MiddlewareControl::{StopWithFinal, Interrupt}` (`H/middleware/types.rs`) | Pydantic has no "wrap node"/"output validate" equivalents in TinyAgents; TinyAgents has no output-validate hook. |
| Node-level iteration (`agent.iter()` / `next(node)`) | Yes; override next node | **Partial** — graph `run/resume/retry/resume_from` (`G/compiled/executor.rs`), `GraphEvent` stream, `get_state_history`/`update_state`/`fork_state` | No harness-level step iterator; agent loop is opaque. |
| Streaming | `run_stream` (`stream_text/stream_output`, partial-validated snapshots), `run_stream_events` (`PartStart/Delta/End`, tool events, `AgentRunResultEvent`) | **Yes** — `ModelDelta{content, reasoning, tool_call}`, `invoke_agent_stream`, `AgentEvent` (`H/stream`, `H/events`) | No partial-structured-output snapshots. |
| Thinking/reasoning parts | `ThinkingPart`, `Thinking(effort=)` capability, `thinking_tags` parsing | **Yes** — `ContentBlock::Thinking{signature}`, `RedactedThinking`, `ReasoningConfig` | |
| Multimodal input | `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, `BinaryContent`, `UploadedFile`, `force_download`, SSRF guard | **Partial** — `ContentBlock::Image(ImageRef)` only; `multimodal/` handles data URIs and `FilePayload` (feature-gated) | No audio/video/document blocks in the message model. |
| Testing | `TestModel` (schema-driven auto tool calls), `FunctionModel`, `Agent.override(model/deps/toolsets/capabilities)`, `ALLOW_MODEL_REQUESTS`, `capture_run_messages` | **Yes** — `ScriptedModel`, `StreamingMock`, `SlowModel`, `FakeTool`, `EventRecorder`, `Trajectory` (`H/testkit`), graph testkit | No schema-driven auto-responder; no global network kill-switch. |
| Evals | `pydantic_evals`: `Dataset`/`Case`/`Evaluator`, `LLMJudge`, span-based evaluators, YAML datasets, Logfire | **No** — no eval crate | |
| Observability | `Instrumentation` capability, OTel GenAI semconv v5, Logfire | **Partial** — Langfuse client/exporter, event journals, `RedactingSink`; no OTel `gen_ai.*` (grep: none) | |
| MCP | `MCP` capability / `MCPToolset` (stdio, streamable HTTP, SSE), prefixes, `process_tool_call`, sampling, elicitation, prompts/resources, `load_mcp_toolsets(config.json)` | **No** — MCP appears only inside the Claude Code provider bridge (`H/providers/claude_code`) | `ToolResult` is "MCP-shaped" but no client. |
| Durable execution | Temporal/DBOS/Prefect/Restate/Kitaru/Airflow/Lambda capabilities; deterministic replay; activities for model/tool/MCP | **Yes (different model)** — built-in `Checkpointer` (`InMemory`, `File`, `Sqlite`), `DurabilityMode`, pending writes, `resume`, time travel (`G/checkpoint`) | See §4. |
| Run-state persistence (message-history checkpoints) | Harness `StepPersistence`: `InMemory/File/Sqlite/Mongo` step stores, `continue_run`, `fork_run`, tool-effect ledger `(run_id, tool_call_id)` with `started/completed/failed`, `MediaStore` externalization | **Partial** — `tinyagents-session` (history/run ledger), harness `store/`, graph checkpoints | No tool-effect ledger for crash-time "did this side effect happen". |
| Graph library | `pydantic_graph`: `BaseNode`/`End`, builder `@g.step`, `g.join(reducer)`, `edge_from().map().to()`, broadcast, `transform`, `graph.render()` mermaid; persistence removed in v2 | **Yes, richer** — `GraphBuilder`, channels/reducers (`LastValue`, `Topic`, `BinaryAggregate`, `Barrier`, `NamedBarrier`), `Send`, `Command`, subgraphs, `map_reduce`, `to_mermaid` (`G/`) | |
| Sub-agents / delegation | tool-call delegation, `Subagents` (`delegate_task`), `Advisor`, `DynamicWorkflow` | **Yes** — `SubAgent`, `SubAgentTool`, `subagent_node`, `DetachedTaskRegistry`, orchestration tools, steering | TinyAgents is deeper (detached, wait, steer). |
| Declarative agent spec | `Agent.from_file('agent.yaml')`, `from_spec`, capability `from_spec` | **Yes** — `.rag` blueprints (`tinyagents-language`), `AgentDefinition` (`tinyagents-definition`) | Different scope: `.rag` describes graphs; Pydantic spec describes one agent. |
| Guardrails | Harness `InputGuardrail/OutputGuardrail/ToolGuardrail` with allow/block/replace/retry/approve; secret/PII detectors | **Partial** — `RedactionMiddleware`, `ToolPolicyMiddleware`, `RedactingSink` | No output-retry guardrail outcome. |
| Coding-agent batteries | Harness `Coder`, `FileSystem`, `Shell`, `Planning`, `Memory`, `Skills`, `CodeMode` | **Partial** — `goals`, `todos` (task board), `workspace` isolation, `tools/time` | TinyAgents deliberately leaves file/shell tools to the host. |
| UI protocols | `AGUIAdapter`, `VercelAIAdapter` (`dispatch_request`, `transform_stream`), untrusted-history sanitisation | **No** | |
| CLI / web chat | `clai`, `Agent.to_cli()`, `to_web()` | **No** | |
| Realtime voice | `pydantic_ai.realtime` (OpenAI, Gemini, xAI, Azure) | **No** | |

## 3. Features TinyAgents lacks, ranked by value

### 3.1 Output-validation retry loop (`ModelRetry` on output)
**What.** After extraction, a validator can reject the value and the loop
re-asks the model with a `RetryPromptPart`, bounded by `retries={'output': N}`.
```python
@agent.output_validator
async def validate_sql(ctx: RunContext[DatabaseConn], output: Output) -> Output:
    try:  await ctx.deps.execute(f'EXPLAIN {output.sql_query}')
    except QueryError as e:  raise ModelRetry(f'Invalid query: {e}') from e
    return output
```
Semantics: consumes one output retry (default 1); `ctx.partial_output` is
`True` on streamed intermediate snapshots so validators skip side effects;
exhaustion raises `UnexpectedModelBehavior`. Output *functions* (callables in
`output_type`) may also raise `ModelRetry`.
**Why it matters.** A schema-valid but semantically wrong final answer is the
common failure; a re-ask is cheaper than discarding the run. TinyAgents
already has the pieces (`StructuredOutcome` with a model-ready `error`
string, repair ladder) but the loop still does `extract(&response)?`.
**Mapping.** Add `OutputRetryPolicy { max: u8 }` to `RunPolicy`; on
`StructuredOutcome.error` or a new `OutputValidator<State,Ctx>` returning
`Err(ModelRetry(msg))`, push `Message::user(msg + "Fix the errors and try
again.")`, emit `AgentEvent::OutputRetry`, `continue`. Expose a typed wrapper
`run.structured_as::<T>()`.

### 3.2 Deferred tool calls as a typed, resumable output
**What.** Three ways a tool call leaves the loop: `requires_approval=True` /
`raise ApprovalRequired(metadata=...)`, `raise CallDeferred(metadata=...)`
(external execution), or an `ExternalToolset` of schema-only tools. The run
ends with output `DeferredToolRequests { calls: [ToolCallPart], approvals:
[ToolCallPart], metadata: {id: {...}} }`. The host persists `message_history`
and later calls
`agent.run(prompt, message_history=msgs, deferred_tool_results=DeferredToolResults(approvals={id: True|ToolApproved(override_args=..)|ToolDenied(message)}, calls={id: value|ToolReturn|ModelRetry|ToolFailed}))`.
`requests.build_results(approve_all=True)` and `requests.remaining(results)`
are helpers; `HandleDeferredToolCalls(handler)` resolves inline instead. The
model sees a denial as a tool return carrying the `ToolDenied.message`.
**Why it matters.** Cleanly separates "the loop paused" from "the loop
failed", gives every pending call a stable id + metadata, supports partial
resolution, and works across processes/machines because the only state is
message history. TinyAgents' harness path is `Err(Interrupted{node,message})`
with the caller left to reconstruct which calls were pending.
**Mapping.** Add `LoopExit::Deferred(DeferredToolRequests)` in
`agent_loop`, a `ToolOutcome::{ApprovalRequired, Deferred}` variant reachable
from `tinytools::ToolResult` or via `ToolPolicy.access.approval_required`,
and `AgentTurnRequest::with_deferred_results(DeferredToolResults)`. The graph
`Interrupt` stays the durable primitive; this is the harness-level projection
of it. `OpenHuman`'s `ApprovalGate` (parks the tool future on a oneshot)
would become one `handler` implementation.

### 3.3 Rich tool returns (`ToolReturn`)
```python
return ToolReturn(return_value='clicked (x,y)', content=['Before:', BinaryContent(data=png, media_type='image/png')], metadata={'coords': {...}})
```
`return_value` is what goes into the tool-result part; `content` becomes a
separate user message (multimodal); `metadata` never reaches the model but
is available in events/persistence; `tools` reveals deferred-loading tools.
**Why.** Screenshots, PDFs, and app-side bookkeeping without stuffing JSON
into the model-visible result. **Mapping.** Extend `ToolContent` with
`Image(ImageRef)`/`File` variants and add `ToolResult { model_content,
follow_up_content: Vec<ContentBlock>, metadata: Value }`; `agent_loop/tools.rs`
appends the follow-up as a `Message::user` after the tool message.

### 3.4 Composable toolsets with per-step `prepare`
**What.** `AbstractToolset { get_tools(ctx) -> dict[name, ToolsetTool];
call_tool(name, args, ctx, tool); get_instructions(); for_run(ctx);
for_run_step(ctx) }` plus wrappers: `.filtered(pred)`, `.prefixed('weather')`,
`.renamed({...})`, `.prepared(fn(ctx, tool_defs) -> tool_defs)`,
`.approval_required(pred)`, `.defer_loading()`, `CombinedToolset`,
`WrapperToolset` (override `call_tool`), `@agent.toolset` dynamic per run.
Per-tool `prepare=fn(ctx, ToolDefinition) -> ToolDefinition | None` hides or
rewrites one tool per step; runs before agent-wide `PrepareTools`.
**Why.** Prefix/rename resolves MCP name collisions; a toolset carrying its own
instructions and lifecycle (`__aenter__`) is the natural unit for MCP
servers and per-user authenticated tool sources. **Mapping.** Introduce
`trait ToolSet<State,Ctx> { async fn tools(&self, ctx) -> Vec<Arc<dyn Tool>>;
async fn call(...); fn instructions() }` with `Prefixed`, `Filtered`,
`Combined`, `Prepared` adaptors; `ToolRegistry` becomes one `ToolSet`.
Existing `DynamicToolSelectionMiddleware` maps onto `Prepared`.

### 3.5 MCP client
`MCPToolset('http://host/mcp')` / `StdioTransport(command, args)`;
`.prefixed()`, `process_tool_call(ctx, call_tool, name, args)` for injecting
deps/metadata, `sampling_model` / `agent.set_mcp_sampling_model()`,
`elicitation_handler`, `list_resources/read_resource`, `list_prompts`,
`tool_error_behavior='retry'|'failed'|'error'`, `load_mcp_toolsets('mcp.json')`
with `${VAR:-default}` expansion, and `MCP(native=True)` to let the provider
call the server directly. **Mapping.** A `tinytools-mcp` crate implementing
the `ToolSet` trait above (rmcp or a thin JSON-RPC client); `ToolResult`
is already MCP-shaped.

### 3.6 Model profile as a request/response transformer
Beyond capability flags, Pydantic's `ModelProfile` carries behaviour:
`json_schema_transformer` (rewrites schemas per provider, e.g. strip
`$defs`/`additionalProperties` for Gemini), `default_structured_output_mode`
(`'tool'|'native'|'prompted'`), `prompted_output_template`,
`native_output_requires_schema_in_instructions`, `thinking_tags` (parse
`<think>` from text into `ThinkingPart`), `ignore_streamed_leading_whitespace`,
`supported_native_tools`, `tool_deferral_mode`/`tool_addition_mode`,
`context_window`. **Mapping.** Add to `tinyinference` `ModelProfile`: a
`schema_transform: Option<fn(&Value)->Value>` (or an enum of named
transformers, to keep it serializable), `default_structured_mode`,
`thinking_tags: Option<(String,String)>`; consume them in
`SchemaPreparation` and the structured plan selection in `run_loop.rs:392`.

### 3.7 Evals
`Dataset(name=..., cases=[Case(name, inputs, expected_output, metadata,
evaluators=[...])], evaluators=[...])`; `dataset.evaluate(task)` ->
`EvaluationReport`; evaluators return bool/score/label/`EvaluationReason`;
built-ins `Equals`, `EqualsExpected`, `Contains`, `IsInstance`, `MaxDuration`,
`LLMJudge`, `HasMatchingSpan` (span-based, reads OTel traces); YAML/JSON
datasets with generated JSON schema; Logfire experiments UI. **Mapping.** A
`tinyagents-evals` crate: `Case<I,O>`, `Evaluator<I,O>` trait, `Dataset`,
`Report` with per-case scores; `Trajectory` from `H/testkit` already gives the
span-like evidence. LLM-judge via `AgentHarness`.

### 3.8 Prompted output + union outputs + output functions
`PromptedOutput([A,B])` injects the schema into instructions for models with
no tool/native support; unions register one output tool per variant; output
functions run code with validated args and make its return the run output
(`output_type=[run_sql_query, SQLFailure]`); `Choices({...})` for dynamic
enums. `end_strategy='graceful'|'early'|'exhaustive'` decides what happens
when output tools and function tools co-occur (TinyAgents logs
`structured_with_tool_calls`, `run_loop.rs:684`). **Mapping.** Add
`StructuredStrategy::Prompted { template }` and allow `Vec<Schema>` ->
multiple synthetic tools; `EndStrategy` enum on `RunPolicy`.

### 3.9 Multimodal message model
`ImageUrl/AudioUrl/VideoUrl/DocumentUrl/BinaryContent` in user prompts and
tool returns; `force_download`; SSRF guard on `http(s)`, cloud schemes passed
through. **Mapping.** `ContentBlock::{Audio(AudioRef), Video, Document}` in
`tinyinference-llm/message`, provider matrix in `ModelProfile.modalities`.

### 3.10 Schema-driven `TestModel` and `ALLOW_MODEL_REQUESTS`
`TestModel` calls every registered tool with schema-generated args, then
returns `custom_output_text`/`custom_output_args`; `FunctionModel(fn(messages,
AgentInfo) -> ModelResponse)`; `models.ALLOW_MODEL_REQUESTS = False` makes
any real provider call raise. **Mapping.** `testkit::SchemaDrivenModel`
generating args from `ToolSchema.parameters`; a process-wide
`deny_network_models()` guard in `tinyinference` providers.

### 3.11 Tool-effect ledger for crash recovery (`StepPersistence`)
Snapshots are taken at settled `CallToolsNode` boundaries; each tool call has a
`(run_id, tool_call_id)` row with `started|completed|failed`; after a crash,
`list_unresolved_tool_effects(run_id)` shows `started` rows with
`idempotency_key` / `effect_summary` so the orchestrator decides whether to
`continue_run(include_interrupted=True)` or `fork_run`. Large/binary content
is externalised to a `MediaStore` (`media+sha256://` URIs). **Mapping.**
TinyAgents' checkpointer already has pending writes; add a per-tool-call
effect row to `tinyagents-session`'s run ledger keyed by `CallId`, with
`ToolPolicy.runtime.idempotent` feeding the decision.

### 3.12 OTel GenAI semantic conventions + UI adapters
`Instrumentation()` emits `gen_ai.*` spans (agent run, model request, tool);
`AGUIAdapter`/`VercelAIAdapter.dispatch_request(request, agent=)` and
`transform_stream(events)` for remote-run cases; `sanitize_messages()` strips
client-supplied system prompts, non-HTTP file URLs, dangling tool calls.
**Mapping.** OTel is a `JournalSink` implementation; UI adapters and
sanitisation belong to the host (§5).

## 4. Design lessons

- **Typed deps.** Both inject a typed context; Pydantic passes `RunContext[Deps]`
  into tools, instructions, validators, `prepare`, and toolset factories with
  one signature. TinyAgents splits `&State` (graph state) from `Ctx`
  (runtime deps), which is the right split for a graph runtime but means
  tools receive `ToolRunContext` (tinytools) rather than the harness
  `RunContext`; the injected-argument mechanism (`tool/injected.rs`) is a
  workaround Pydantic does not need. Worse in Pydantic: no separate durable
  state, so "state" is whatever you put in `deps` or message history.
- **Validation-retry loop.** Pydantic's single `ModelRetry` exception unifies
  tool arg errors, tool logic errors, output validation, output functions,
  guardrails, and hooks into one budgeted re-ask protocol, and `ToolFailed`
  gives the opposite ("do not retry"). TinyAgents has four separate
  mechanisms (`InvalidArgsPolicy`, `UnknownToolPolicy`, `ToolResult::error`,
  fatal structured extraction). Adopt the two-exception vocabulary.
- **Toolset composition.** Pydantic's wrapper chain is clearer than
  TinyAgents' middleware-based tool filtering because each wrapper is a value
  you can inspect and test; middleware ordering makes "why was this tool
  hidden" hard to answer (sdk-gaps §9 asks for exactly that explainability).
  TinyAgents' `ToolPolicy` declaration model (side effects, access,
  runtime) is better than Pydantic's free-form `metadata` dict and
  `requires_approval` boolean.
- **Deferred tools.** Pydantic's stop-and-resume is stateless by design
  (only message history + ids), which is why it works under Temporal and over
  HTTP. TinyAgents' graph `Interrupt` + checkpoint is more general (any node,
  any state) but the harness has no equivalent typed handshake. Keep both:
  the harness projection for single-agent hosts, the graph interrupt for
  workflows.
- **Model profiles.** Pydantic treats the profile as *behaviour* (schema
  transformer, output-mode default, thinking-tag parsing) that the loop
  consults; TinyAgents treats it as *capability data* for resolution. Both
  are needed; TinyAgents' `CapabilitySet` resolution is the stronger half.
- **Evals.** Pydantic ships them in-repo and wires them to spans; TinyAgents
  has nothing. Low cost to add a minimal crate.
- **iter API.** `agent.iter()` exposing `UserPromptNode/ModelRequestNode/
  CallToolsNode` is the same idea as TinyAgents' explicit graph, but Pydantic
  applies it to the *agent loop itself*, so any host can step, inspect, or
  override the next node. TinyAgents' loop is a closed `run_loop.rs`;
  exposing it as a `CompiledGraph` (the docs promise this in
  `docs/modules/harness/state-graph.md`) would close the gap and reuse the
  checkpointer for free.
- **Durable execution.** Pydantic explicitly refuses to own a checkpointer:
  "durability is not storage", and it delegates replay to Temporal/DBOS/etc.
  with the cost that deps, settings, and metadata must be Pydantic-serializable,
  activities see a *limited* `RunContext` (no `messages`, `model`, `prompt`),
  toolsets must be registered at construction, `ctx.emit`/`cancel` raise
  inside activities, and payloads are capped at 2 MB. TinyAgents' built-in
  `Checkpointer` with `DurabilityMode`, pending writes, and time travel is
  the better default for a desktop host with no workflow engine, and matches
  LangGraph rather than Pydantic. What Pydantic does better here is the
  narrow, well-specified persistence contract (`StepPersistence`'s
  tool-effect ledger and media externalisation), which TinyAgents' session
  ledger lacks.
- **Capabilities as the unit of composition.** v2's `AbstractCapability`
  (instructions + toolset + native tools + model settings + hooks + event
  listeners + `for_run` scoping + `defer_loading` + `from_spec`) is a bigger
  idea than middleware: it is what a "skill" or "plugin" is. TinyAgents'
  middleware is hooks-only; a bundle type (`Capability { instructions, tools,
  middleware, model_defaults, exposure }`) registered in `tinyagents-registry`
  would give `.rag` and `AgentDefition` a single thing to reference.
- **Where TinyAgents is ahead.** Channels/reducers, `Send` fan-out,
  subgraphs, barrier channels, `map_reduce`, detached sub-agent registry,
  steering commands, task board/goals, prompt-segment cache layout, model
  resolution by capability set, fail-closed tool policy, no-progress
  detection, and the `.rag` language. Pydantic has none of these at runtime
  level; its "DynamicWorkflow" and "Subagents" are Harness capabilities on
  top of plain tool calls.

## 5. Runtime vs host split

| Gap | Layer | Reasoning |
|---|---|---|
| 3.1 output-validation retry loop | **Runtime** (harness) | Loop control; OpenHuman cannot do it from outside `run_loop.rs`. |
| 3.2 deferred tool requests/results | **Runtime** type + **Host** handler | Types, `LoopExit::Deferred`, resume argument in harness; OpenHuman's `security/approval` `ApprovalGate` becomes the handler and owns the UI/SQLite pending rows. |
| 3.3 `ToolReturn` rich results | **Runtime** (tinytools + harness) | Message model change. |
| 3.4 toolset composition, per-tool `prepare` | **Runtime** | OpenHuman's `agent/tool_policy.rs` and `tinyagents/middleware` currently re-implement filtering; a `ToolSet` trait lets them shrink to predicates. |
| 3.5 MCP client | **Runtime** (separate crate, `tinytools-mcp`) | Protocol, not product. Server config/auth UI is host. |
| 3.6 profile transformers/thinking tags | **Runtime** (tinyinference) | Provider concern. |
| 3.7 evals | **Runtime** (new crate) | Host supplies datasets. |
| 3.8 prompted/union output, `EndStrategy` | **Runtime** | |
| 3.9 audio/video/document blocks | **Runtime** (tinyinference) | OpenHuman's `agent/multimodal.rs` currently works around it with markers. |
| 3.10 schema-driven test model, network kill-switch | **Runtime** (testkit) | |
| 3.11 tool-effect ledger, media externalisation | **Runtime** (session + graph checkpoint) with **Host** deciding idempotency | Matches sdk-gaps §4/§6. |
| 3.12 OTel semconv sink | **Runtime** (optional sink) | Langfuse already lives there. |
| UI adapters (AG-UI/Vercel), `sanitize_messages` | **Host** | Transport; but a `sanitize_history()` helper is cheap to put in harness. |
| Harness batteries (`Coder`, `Shell`, `FileSystem`, `Memory`, `Guardrails`, `Planning`) | **Host** (OpenHuman `tools`, `security`, `agent/goals`, `agent/todos`) | TinyAgents' stance (tools are host vocabulary) is the same as Pydantic's core/Harness split; note Pydantic still ships them as an *official* library, which TinyAgents could mirror as an optional `tinyagents-batteries` crate. |
| Capability bundle type, agent spec loading | **Runtime** (registry) | `.rag` + `AgentDefinition` already point this way. |
| Durable-execution engines (Temporal etc.) | Neither now | Desktop host has no workflow engine; keep the built-in checkpointer. |
| Realtime voice, CLI, web chat | **Host** | |

## 6. Sources

- Changelog / upgrade guide: https://pydantic.dev/docs/ai/project/changelog/
- v2 announcement: https://pydantic.dev/articles/pydantic-ai-v2
- Releases (dates verified via `gh api repos/pydantic/pydantic-ai/releases`): https://github.com/pydantic/pydantic-ai/releases
- Agent: https://pydantic.dev/docs/ai/core-concepts/agent/
- Output: https://pydantic.dev/docs/ai/core-concepts/output/
- Retries: https://pydantic.dev/docs/ai/core-concepts/retries/
- Hooks: https://pydantic.dev/docs/ai/core-concepts/hooks/
- Message history / enqueue / sanitize: https://pydantic.dev/docs/ai/core-concepts/message-history/
- Storage: https://pydantic.dev/docs/ai/core-concepts/storage/
- Input (multimodal): https://pydantic.dev/docs/ai/core-concepts/input/
- Agent spec: https://pydantic.dev/docs/ai/core-concepts/agent-spec/
- Tools: https://pydantic.dev/docs/ai/tools-toolsets/tools/ and https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/
- Toolsets: https://pydantic.dev/docs/ai/tools-toolsets/toolsets/
- Deferred tools: https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/
- Capabilities overview and API: https://pydantic.dev/docs/ai/capabilities/overview/ , https://pydantic.dev/docs/ai/api/pydantic-ai/capabilities/
- On-demand capabilities: https://pydantic.dev/docs/ai/capabilities/on-demand/
- Instrumentation: https://pydantic.dev/docs/ai/capabilities/instrumentation/
- Durable execution: https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/ , https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/
- Harness: https://pydantic.dev/docs/ai/harness/ , step persistence https://pydantic.dev/docs/ai/harness/step-persistence/ , compaction https://pydantic.dev/docs/ai/harness/compaction/ , guardrails https://pydantic.dev/docs/ai/harness/guardrails/
- Models: https://pydantic.dev/docs/ai/models/overview/ ; `ModelProfile` keys from `pydantic_ai_slim/pydantic_ai/profiles/__init__.py` (main)
- Usage API: https://pydantic.dev/docs/ai/api/pydantic-ai/usage/
- Messages API: https://pydantic.dev/docs/ai/api/pydantic-ai/messages/
- Testing: https://pydantic.dev/docs/ai/guides/testing/
- Multi-agent: https://pydantic.dev/docs/ai/guides/multi-agent-applications/
- MCP client: https://pydantic.dev/docs/ai/mcp/client/
- UI adapters: https://pydantic.dev/docs/ai/integrations/ui/overview/
- Evals: https://pydantic.dev/docs/ai/evals/getting-started/core-concepts/
- pydantic-graph: https://pydantic.dev/docs/ai/graph/graph/ , https://pydantic.dev/docs/ai/graph/builder/parallel/
- vs LangGraph: https://pydantic.dev/docs/ai/comparisons/vs-langchain-langgraph/
- Source read from `main`: `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (nodes, `GraphAgentState`), `_deferred.py` (`DeferredToolRequests`, `ToolApproved`, `ToolDenied`)
