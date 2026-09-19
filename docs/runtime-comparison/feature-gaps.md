# Consolidated Feature Gaps

Cross-runtime view of what LangGraph/LangChain (LG), Pydantic AI (PA) and pi
have that TinyAgents lacks or only partly has, with a layer decision for each.
Per-runtime detail, API shapes and code excerpts live in
[`langgraph.md`](langgraph.md), [`pydantic-ai.md`](pydantic-ai.md) and
[`pi.md`](pi.md); the section numbers in the "Source" column point there.

Layer legend: **RT** = TinyAgents runtime (crate named), **OH** = OpenHuman,
**RT+OH** = primitive in the runtime, policy/UI in OpenHuman.

Value is ranked 1 (highest) to 3. "Existing seam" names what we already have
that the feature would extend.

## A. Agent-loop control and human-in-the-loop

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| A1 | Middleware control outcomes: hooks return state updates and may `jump_to` model/tools/end; wrap hooks return commands; tools return `Command`; `return_direct`; `terminate` / `should_stop_after_turn` hints | LG, pi | LG §3.1, pi §4.4, `sdk-gaps.md` §13 | `MiddlewareControl::{StopWithFinal, Interrupt}`, `MiddlewareModelOutcome` (`#[non_exhaustive]`) | RT harness | 1 |
| A2 | Deferred tool calls as a typed, resumable output: `DeferredToolRequests{calls, approvals, metadata}` / `DeferredToolResults` with per-call ids, approve / edit-args / reject / respond decisions, external execution (`CallDeferred`), durable across process restart | PA, LG (HITL middleware) | PA §3.2, LG §3.2 | `HumanApprovalMiddleware` (`Fn(&ToolCall) -> bool`), `Err(Interrupted)`, graph `Interrupt`, delegation `PendingApproval`, `ToolPolicy.access.approval_required` | RT+OH (OpenHuman `security/approval::ApprovalGate` becomes a handler) | 1 |
| A3 | Output-validation retry loop: validator raises `ModelRetry`, loop re-asks with the error, bounded by `retries.output`; unified retry/`ToolFailed` vocabulary for tool errors, arg errors and output errors | PA | PA §3.1, §4 | `StructuredOutcome.error`, `structured/repair.rs`, `InvalidArgsPolicy`; docs already promise `StructuredOutputErrorPolicy` | RT harness | 1 |
| A4 | Steering and follow-up as two message lanes polled at turn boundaries with `QueueMode` (one-at-a-time / all); follow-ups run "one more turn" after the agent would stop | pi, PA (`ctx.enqueue`) | pi §4.4 | `RunQueue{Steer, Followup, Collect}` (no consumer), `SteeringCommand` | RT harness (+ OpenHuman `agent/harness/run_queue` moves down) | 1 |
| A5 | Agent loop compiled to the graph runtime so checkpoints, interrupts, streaming and time travel apply to the loop itself; step iteration (`agent.iter()`) | LG (`create_agent` → `StateGraph`), PA (`iter`, `UserPromptNode`…) | LG §4, PA §4 | `subagent_node`, `docs/modules/harness/state-graph.md` (promised) | RT harness+graph | 2 |
| A6 | Prompted / union / output-function structured modes; `EndStrategy` when output tools and function tools co-occur | PA, LG (`AutoStrategy`) | PA §3.8 | `StructuredStrategy::{ProviderSchema, ToolCall}` | RT harness | 3 |
| A7 | `interrupt_before/after` selectors, `interrupt(response_schema)`, graceful drain (`RunControl`) | LG | LG §3.8 | documented in `interrupts.md`, `mark_interrupt` export-only, `CancellationToken` | RT graph | 3 |

## B. Tools

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| B1 | `ToolRuntime` parity: tool sees `call_id`, store, state view, stream writer, execution info | LG, PA (`RunContext`) | LG §3.6 | `ToolExecutionContext{run_id, thread_id, depth, events, cancel, workspace}`, `tool/injected.rs` | RT harness | 1 |
| B2 | Rich tool return: model-visible value + separate multimodal follow-up content + app metadata / artifact never shown to the model | PA (`ToolReturn`), LG (`ToolMessage.artifact`) | PA §3.3, LG §3.6 | `ToolContent::{Text, Json}` (vendor tinytools), `artifacts/`, `handoff.rs` | RT tinytools+harness | 1 |
| B3 | Composable toolsets: `Combined`, `Filtered`, `Prefixed`, `Renamed`, `Prepared`, `ApprovalRequired`, `External`; per-tool `prepare` hook; toolset carries its own instructions | PA | PA §3.4 | `ToolRegistry`, `ToolAllowlistMiddleware`, `DynamicToolSelectionMiddleware`, `ToolExposure` | RT harness (OpenHuman `agent/tool_policy.rs` shrinks to predicates) | 2 |
| B4 | Generic MCP client (stdio / streamable HTTP / SSE), prefixes, sampling, elicitation, `process_tool_call`, config-file loading | PA, LG (`langchain.mcp`) | PA §3.5, LG §3.10 | MCP only inside `providers/claude_code`; OpenHuman has `mcp/` | RT new crate (`tinytools-mcp`); server config/auth UI in OH | 2 |
| B5 | Tool replay classification (`replay: never \| safe`) and a per-call tool-effect ledger (`started / completed / failed`, idempotency key) for crash recovery | pi, PA (`StepPersistence`) | pi §4.7, PA §3.11, `sdk-gaps.md` §1 | `ToolPolicy.runtime`, `PendingWrite`, session run ledger, `append_interrupted_partial` | RT harness+session; OH decides idempotency for its tools | 2 |
| B6 | Transcript-carried system-prompt sections and tool add/remove patches (`SystemMessage.sections/toolsAdded/toolsRemoved`, `declareToolChanges`) so dynamic tools are cache-aware and replayable | pi | pi §4.1 | `PromptSegment`, `ToolsFiltered` event, `ModelProfile` | RT tinyinference+harness; OH chooses the tools | 2 |
| B7 | Provider-executed (builtin) tools as content parts: web search, code execution, image generation | PA, LG (provider tool search) | PA §2 | none in `ContentBlock` | RT tinyinference | 3 |

## C. Streaming and events

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| C1 | Block-indexed stream events (`text/thinking/toolcall _start/_delta/_end` with `content_index`), error terminals that carry the partial assistant message and stop reason | pi | pi §4.2, `sdk-gaps.md` §3 | `ModelStreamItem::{MessageDelta, ToolCallDelta, UsageDelta, Completed}` | RT tinyinference+harness | 1 |
| C2 | Compact durable frame codec + reducer for partial messages (crash recovery, reconnecting clients) | pi | pi §4.2 | `AgentEvent::ModelDelta` journaled, `HarnessEventJournal` | RT harness | 2 |
| C3 | Unified stream envelope with `ns`/`seq`, `tasks` and `checkpoints` stream modes, projections (`stream.messages`, `stream.tool_calls`, `stream.subagents`) | LG | LG §3.5 | `GraphEvent` (no run/task id on step events), `GraphObservation` journal, `StreamMode` (unused) | RT graph+harness; OH keeps format adapters | 2 |
| C4 | OpenTelemetry GenAI semantic-convention sink | PA | PA §3.12 | Langfuse exporter, `JournalSink` | RT harness optional feature | 3 |

## D. Graph durability

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| D1 | Real pending writes: completed parallel siblings are never re-run after an interrupt/failure and their successors run in the right superstep (today's C1/C2 bugs) | LG | [`code-review-graph.md`](code-review-graph.md) C1, C2, R2 | `PendingWrite` (markers only), `completed_tasks` | RT graph | 1 |
| D2 | Per-node `RetryPolicy`, `CachePolicy` (task cache with backends), `TimeoutPolicy{run, idle}`, `error_handler`, real `defer` scheduling | LG | LG §3.3 | graph-wide `with_node_retry` / `with_node_timeout`, harness `ResponseCache`, `mark_deferred` (export-only) | RT graph | 2 |
| D3 | Checkpoint v2: format version, single task list, channel versions, serialisable `ChannelState`, delta-channel history with snapshot frequency, `Overwrite` | LG (`DeltaChannel`, `Overwrite`) | LG §3.4, code-review-graph I5/I6/R3 | `Checkpoint<State>` full snapshot, `ChannelSet` (not serialisable), `prune` doc mentions deltas | RT graph | 2 |
| D4 | Executor-level per-thread lease (in-process + optional durable claim), typed `TaskId` end to end, `Send` fan-out of subgraphs namespaced by task, subgraph failure resumable via parent, restart-safe interrupt ids, panic/cancel safety | LG (task ids, ns) | code-review-graph C3, C4, I1, I4, I7, R4, R5 | `ThreadLockMap`, `ids::new_checkpoint_id`, `run_ledger::try_claim` | RT graph | 1 |
| D5 | Durable task memoisation inside a node (`durable_task(ctx, key, fut)`) so side effects before an interrupt are not repeated | LG (functional API) | LG §3.9 | `PendingWrite`, `interrupts.md` "must use idempotency keys" | RT graph | 3 |
| D6 | Semantic search on the namespaced store (`IndexConfig{embed, dims, fields}`) | LG | LG §3.7 | `NamespacedStore::search` (substring), `retriever/`, `tinyinference-embeddings` | RT harness; backend choice in OH `memory/` | 3 |

## E. Sessions and context

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| E1 | Conversation entry tree: `id/parent_id`, in-place branching, labels, fork/clone with `parent_session`, `BranchSummaryEntry`, context projection that never reads past the newest compaction | pi | pi §4.3 | `tinyagents-session` linear JSONL + SQLite, graph `fork_state` | RT session; `/fork`, `/tree`, labels UI in OH | 2 |
| E2 | Compaction rules: cut points at user/assistant boundaries, `keep_recent_tokens`, split turns, iterative summaries, durable `CompactionRecord{first_kept, tokens_before, usage}`, overflow classifier → compact → retry same turn, hook may decline/replace | pi, PA (`Compaction` capability), LG (`SummarizationMiddleware`) | pi §4.5 | `summarization/` (policy, trim, pairing, `Summarizer`), `ContextCompressionMiddleware`, `MicrocompactMiddleware` | RT harness+session; summary prompt wording in OH | 2 |
| E3 | Cross-provider handoff transform: record `origin{provider, api, model}` on assistant messages; drop/convert thinking, normalise tool-call ids, downgrade images per target profile | pi | pi §4.6 | `AssistantMessage.id`, `ContentBlock::Thinking` doc note | RT tinyinference+harness | 2 |
| E4 | `sanitize_history()` for untrusted client-supplied history (strip system prompts, non-HTTP file URLs, dangling tool calls) | PA | PA §5 | `summarization/trim.rs` orphan handling | RT harness (cheap helper); transport in OH | 3 |
| E5 | Custom transcript roles / entries (`Message::Custom{kind, payload, display}`) filtered at request-build time | pi | pi §4.9 | `ToolMessage.artifact`, `ContentBlock::ProviderExtension` | RT tinyinference | 3 |

## F. Models and providers

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| F1 | `ModelProfile` as behaviour: `json_schema_transformer`, `default_structured_output_mode`, `prompted_output_template`, `thinking_tags` parsing, per-API compat matrix (~30 flags), `thinking_level_map` | PA, pi | PA §3.6, pi §4.8 | `ModelProfile` + `CapabilitySet` (capability data), `SchemaPreparation`, `ReasoningConfig` | RT tinyinference | 2 |
| F2 | Generated model catalog (models.dev) with tiered pricing, `refresh_models()`, availability filtered by resolved auth | pi, PA (`genai-prices`) | pi §4.8 | `ModelCatalogEntry`, 5-model stale seed (`code-review-workspace.md` M7), `ModelPricing` | RT registry+tinyinference | 2 |
| F3 | Audio / video / document content blocks in the message model, SSRF-guarded URL download | PA | PA §3.9 | `ContentBlock::Image(ImageRef)`, `multimodal/` | RT tinyinference (OpenHuman `agent/multimodal.rs` markers go away) | 2 |
| F4 | Provider breadth (Google, Vertex, Bedrock, Mistral native protocols), generalised OAuth flows, `CredentialStore` trait | pi (10 APIs, 41 presets, 7 OAuth flows) | pi §2, §4.8 | OpenAI + Anthropic + local in tinyinference, Codex `OAuthFlow` | RT tinyinference (trait); keychain and login UI in OH `security/credentials` | 3 |
| F5 | Deferred / background model responses (`DeferredHandle`, `stop_reason: deferred`) | pi | pi §4.9 | none | RT tinyinference | 3 |
| F6 | Request/response escape hatches (`on_payload`, `on_response`, `fetch` injection) | pi | pi §5 | decorators in `tinyinference-llm/model/decorators.rs` | RT tinyinference | 3 |

## G. Testing and composition

| # | Gap | Who has it | Source | Existing seam | Layer | Value |
|---|---|---|---|---|---|---|
| G1 | Evals: `Dataset` / `Case` / `Evaluator` trait, `LLMJudge`, span-based evaluators, report | PA (`pydantic_evals`), LG (`agentevals`) | PA §3.7 | `testkit::Trajectory` | RT new crate `tinyagents-evals`; datasets in OH | 2 |
| G2 | Schema-driven `TestModel` (auto-calls every tool with generated args), process-wide `deny_network_models()` kill-switch | PA | PA §3.10 | `ScriptedModel`, `StreamingMock`, `FakeTool` | RT harness testkit + tinyinference | 3 |
| G3 | Capability bundle: instructions + toolset + middleware + model defaults + exposure, loadable on demand, `from_spec`; what a "skill"/"plugin" is at runtime level | PA v2 (`Capability`) | PA §4 | `Middleware`, `AgentDefinition`, `.rag`, `ToolExposure::Deferred` | RT registry; discovery from disk in OH `skills/` | 2 |
| G4 | Backend conformance suites for sessions / stores run against every backend | pi (`session/testing/conformance`), LG | pi §3, `sdk-gaps.md` §17 | graph `testkit/conformance.rs` (checkpointers, task stores) | RT | 3 |

## Stays in OpenHuman

These appear in the reference stacks but are harness/product concerns and
should not enter TinyAgents: filesystem / shell / search tools and their
backends; sandboxes and path permissions (`sandbox/`, `security/bubblewrap`);
`AGENTS.md`-style memory files and `SKILL.md` discovery (`memory/`,
`skills/`); extension loaders, commands, packages and UI widgets; UI protocol
adapters (AG-UI, Vercel AI); realtime voice; cron and scheduled runs
(`cron/`); credential storage and login UI; summary and system prompt wording;
approval dialogs. Deep Agents, `pydantic-ai-harness` and `pi-coding-agent`
all keep the same items out of their runtime packages.

## Things we already have that the others lack

Kept here so the plan does not regress them: channels/reducers, `Send`
fan-out with persisted args, subgraphs with namespaced checkpoints,
`map_reduce`, parallel policies (`quorum`/`race`/`compare`/`fallback`),
built-in checkpointer with `DurabilityMode` and time travel, policy-checked
`SteeringCommand` with pause latching, detached sub-agent registry,
`ToolPolicy` declarations, prompt-cache segment layout and guard middleware,
`CapabilitySet` model resolution with fallback chains, per-tool timeouts with
grace, run limits and budgets, no-progress detection, goals and task board,
`.rag` blueprints with diagnostics, and a 637-test integration suite with
checkpointer conformance.
