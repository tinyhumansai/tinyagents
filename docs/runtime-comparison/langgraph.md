# LangGraph / LangChain 1.x vs TinyAgents — runtime comparison

Research date: 2026-09-19. TinyAgents baseline: v2.1.2 (`fc33c43`).
Paths below are relative to this repository unless prefixed `openhuman:`.

## 1. What LangGraph/LangChain is today

**Runtime layer — LangGraph.** `langgraph` 1.0.0 shipped 2025-10-17; 1.1.0 on 2026-03-10; 1.2.0 on
2026-05-12; latest core is 1.2.11 (2026-08-11), with `langgraph-checkpoint` 4.2.0 (2026-08-07). The
runtime is a Pregel/BSP engine: a superstep plans tasks by comparing per-channel version counters
(`channel_versions` vs each node's `versions_seen`), runs all tasks in parallel, then applies writes
through channel reducers and writes a checkpoint (`{v:2, id: uuid6, ts, channel_values,
channel_versions, versions_seen, updated_channels}` + metadata `{source: input|loop|update|fork,
step, parents}`). Nodes are triggered by channel-version changes, not by edges as such (edges compile
to `branch:to:<node>` channels). Control values are first-class: `Command(update, goto, resume,
graph=PARENT)`, `Send(node, arg, timeout)`, `Overwrite(value)` (bypass reducer), `interrupt(value,
response_schema)` which raises `GraphInterrupt` and re-runs the node from the top on resume, with
resume values matched by call index (scalar `Command(resume=)`) or by interrupt id (dict form).
1.x adds durability modes `sync|async|exit` (default `async`), per-node `RetryPolicy`,
`CachePolicy(key_func, ttl)` + `BaseCache`, `defer=True` join nodes, per-node `TimeoutPolicy`
and `error_handler` (1.2), `DeltaChannel` writes-only checkpoint history (1.2, beta), `RunControl`
graceful drain, `stream(version="v2")` unified `StreamPart{type, ns, data}` and
`stream_events(version="v3")` protocol events, `BaseStore` with namespaces/TTL/semantic `IndexConfig`,
and the functional API (`@entrypoint`/`@task`). Double-texting strategies, thread TTL, cron and
background runs are **Agent Server / LangSmith Deployment features, not OSS library features**.

**Agent layer — LangChain.** `langchain` 1.0.0 (2025-10-17), 1.3.0 (2026-05-12), 1.4.2 (2026-09-18);
`langchain-core` 1.6.3 (2026-09-11). The whole agent layer is `create_agent(model, tools,
system_prompt, middleware, response_format, context_schema, checkpointer, store, ...)`, which
compiles to a small LangGraph `StateGraph` (`model` node, `tools` node, one node per middleware
hook). `AgentMiddleware` is the extension point: node hooks `before_agent/before_model/after_model/
after_agent` (return state dicts, may `jump_to`), and wrap hooks `wrap_model_call(request, handler)`
/ `wrap_tool_call(request, handler)` (nest, first middleware outermost). A middleware can extend the
state schema, contribute tools, own stream transformers and trace policy. Built-ins cover
summarization, HITL (durable `interrupt()` with approve/edit/reject/respond), call limits, model
retry/fallback, PII, tool retry/error, LLM tool selection, provider tool search, context editing,
todo list, shell, file search. `ToolRuntime` injects state/context/store/stream_writer/tool_call_id;
`response_format` chooses `ToolStrategy|ProviderStrategy|AutoStrategy`. 1.4.0 added `langchain.mcp`
`MCPAdapter`. Deep Agents (`deepagents` 0.7.15, 2026-09-16) is a harness on top of `create_agent`,
explicitly "does not introduce a new runtime": filesystem tools over pluggable backends, subagents via
a `task` tool, summarization-with-offload, AGENTS.md memory, SKILL.md skills, path permissions,
sandbox backends. `langgraph-supervisor`/`langgraph-swarm` are in maintenance mode; the docs now
recommend `create_agent` + tool-wrapped subagents or middleware-driven handoffs.

## 2. Feature inventory

Legend: Yes / Partial / No. File paths are what I checked.

| Feature | LangGraph/LangChain | TinyAgents | Notes |
|---|---|---|---|
| Pregel supersteps, parallel tasks, reducer at boundary | Yes | Yes — `crates/tinyagents-graph/src/compiled/executor.rs` | Same BSP shape. |
| Channel-version triggering (`versions_seen`) | Yes | No — `channel/mod.rs` is a reducer bridge; scheduling is edge/active-set based | See §4. |
| Typed channels (LastValue/Topic/BinaryOp/Ephemeral/NamedBarrier/Untracked) | Yes | Yes — `graph/src/channel/types.rs` | `Delta` in TinyAgents is a numeric accumulator, unrelated to LangGraph's `DeltaChannel`. |
| `DeltaChannel` writes-only checkpoint history, `snapshot_frequency` | Yes (1.2, beta) | No — `checkpoint/types.rs` stores full `state: State` per checkpoint | Deepagents uses it for `messages` (O(N) vs O(N²)). |
| `Overwrite` (bypass reducer) | Yes | No | Reducer runs on every write. |
| `Command(update, goto, resume)` | Yes | Yes — `graph/src/command/types.rs` | |
| `Command(graph=PARENT)` from subgraph/tool | Yes | Partial — `subgraph/`, `resume_targeted` exists; no parent-directed goto from a child node found | |
| `Send` fanout, persisted send args | Yes | Yes — `checkpoint/types.rs::PendingActivation` | `Send.timeout` (1.2) absent. |
| `interrupt()` inside a node, index/id-matched resume | Yes | Yes — node returns `NodeResult::Interrupt`; `docs/modules/graph/interrupts.md`, `compiled/executor.rs::resume` | TinyAgents interrupt is a return value, not a call; node still re-runs from start. `response_schema` absent. |
| `interrupt_before/after` compile options | Yes | No — documented in `interrupts.md`, no code hit for `interrupt_before` | Docs promise it; not implemented. |
| Durability `sync/async/exit` | Yes (default async) | Yes — `checkpoint/types.rs::DurabilityMode` (default Sync) | TinyAgents async mode has stricter failure semantics (fails run at next boundary). |
| Pending writes / skip completed tasks on resume | Yes | Yes — `checkpoint/types.rs::PendingWrite`, `merge_writes` | |
| Checkpoint namespaces for subgraphs | Yes (`ns|ns`, `node:task_id`) | Yes — `CheckpointConfig.namespace: Vec<String>` | |
| `get_state`, `get_state_history`, `update_state(as_node)`, fork | Yes | Yes — `compiled/state_api.rs` (`bulk_update_state`, `fork_state` extra) | |
| Checkpointer `prune`, `copy_thread`, `delete_for_runs` | Yes (checkpoint 4.1) | Yes — `checkpoint/mod.rs` (`prune`, `copy_thread`, `delete_by_run`) | |
| Checkpoint backends | Memory, SQLite, Postgres, Redis, Mongo… | Memory, File, SQLite — `checkpoint/{file,sqlite}.rs` | `docs/sdk-gaps.md` §5: rusqlite version coupling. |
| Per-node `RetryPolicy` (list, `retry_on`) | Yes | Partial — graph-wide `with_node_retry` in `compiled/mod.rs`; no per-node | |
| Per-node `CachePolicy` + `BaseCache` (task-level cache) | Yes | No in graph; harness has `ResponseCache`/`CachePolicy` for model calls (`harness/src/cache/`) | |
| `defer=True` | Yes | Partial — `builder/mod.rs::mark_deferred` is export-only; real join is `add_waiting_edge` + `add_barrier_relief` | Waiting edges ≈ `NamedBarrierValue`; "run when nothing else is left" semantics absent. |
| Per-node `TimeoutPolicy(run, idle)`, `error_handler`/`NodeError` | Yes (1.2) | Partial — graph-wide `with_node_timeout`; no idle timeout, no node error handler | |
| `RunControl` graceful drain (`GraphDrained`) | Yes (1.2) | Partial — `CancellationToken` (hard cancel) in `harness/src/cancel/` | |
| `Runtime` (context, store, stream_writer, previous, execution_info) | Yes | Yes — `builder/types.rs::NodeContext`, harness `RunContext` | |
| Stream modes | values/updates/messages/custom/checkpoints/tasks/debug, `StreamPart{type,ns,data}` | Partial — `stream/types.rs::StreamMode` {Values, Updates, Messages, Debug, Interrupts, Custom}; typed `GraphEvent` enum instead of `StreamPart` | No `tasks`/`checkpoints` modes; comment says "StreamPart projection is future work". |
| `stream_events(version="v3")` protocol events | Yes (1.2) | No | Observability journal exists instead (`observability/`). |
| `BaseStore` namespaces, batch, TTL, `list_namespaces` | Yes | Yes — `harness/src/store/namespaced/` | Deliberately modelled on LangGraph's batch design. |
| Store semantic search (`IndexConfig{embed,dims,fields}`) | Yes | Partial — `SearchQuery.query` is a substring seam; embeddings exist in `retriever/` but are not wired to the store | |
| Functional API (`@entrypoint`/`@task`, `previous`) | Yes | No | |
| Double texting / multitask strategies | Platform only | Partial — `run_queue/` lanes (steer/follow-up), `thread_locks.rs` | Neither library has it in OSS; TinyAgents' queue is closer than LangGraph OSS. |
| Thread TTL, cron, background runs | Platform only | Partial — store TTL yes; `orchestration/` detached tasks + `JsonlTaskStore`; no cron | OpenHuman has `cron/`. |
| `create_agent` factory | Yes | Yes — `harness/src/runtime/mod.rs::AgentHarness` | |
| Middleware node hooks returning state updates / `jump_to` | Yes | No — `middleware/types.rs` hooks return `Result<()>`; `MiddlewareModelOutcome` has one variant | `sdk-gaps.md` §13 already lists it. |
| Middleware `wrap_model_call`/`wrap_tool_call` nesting | Yes | Yes — `ModelMiddleware::wrap_model`, `ToolMiddleware::wrap_tool` | Same onion order. |
| Middleware extends state schema / contributes tools / stream transformers | Yes | No | Tools are registered on the harness, not by middleware. |
| Dynamic prompt | Yes | Yes — `DynamicPromptMiddleware` | |
| HITL middleware (durable `interrupt`, approve/edit/reject/respond, `when`) | Yes | Partial — `HumanApprovalMiddleware` is `Fn(&ToolCall) -> bool` (`library/types.rs:375`); durable pause only via graph `Interrupt` or `SteeringCommand::Pause` | |
| Summarization middleware (fraction/tokens/messages triggers) | Yes | Yes — `summarization/`, `ContextCompressionMiddleware`, `MessageTrimMiddleware` | |
| Context editing (`ClearToolUsesEdit`) | Yes | Partial — `MicrocompactMiddleware` (`library/context.rs`) | No tool-input clearing config. |
| PII middleware | Yes | Partial — `RedactionMiddleware`, `RedactingSink` | No detector/strategy matrix. |
| Model/tool call limits, model retry/fallback, tool retry, rate limit | Yes | Yes — `BudgetMiddleware`, `RetryMiddleware`, `ModelFallbackMiddleware`, `RateLimitMiddleware`, `limits/` | |
| LLM tool selector / provider tool search | Yes | Yes/No — `DynamicToolSelectionMiddleware`, `ContextualToolSelectionMiddleware` (`tool/select/`); no provider-side tool search | |
| Todo list middleware | Yes (opt-in) | Yes — `graph/src/todos/` (`TaskBoard`, richer) | |
| Shell / file-search middleware | Yes | No in harness (`tools/` has `time.rs` only); `workspace/` gives roots | OpenHuman owns tools. |
| `ToolRuntime` injection (state, store, stream_writer, tool_call_id) | Yes | Partial — `ToolExecutionContext` (run/thread/depth/events/cancel/workspace); no state/store/tool_call_id | `tool/injected.rs` exists for hidden args. |
| Tool returns `Command` (state update + routing) | Yes | No — `ToolResult{content,is_error}` (vendor `tinytools`) | |
| `return_direct`, `ToolMessage.artifact` | Yes | No / Partial — no early-exit flag; `artifacts/` + `handoff.rs` offload large results instead | `sdk-gaps.md` §13 "early-exit tools". |
| `ToolNode.handle_tool_errors` matrix | Yes | Partial — tool errors are recoverable results; unknown tool aborts (`sdk-gaps.md` §2) | |
| Structured output strategies | Tool/Provider/Auto, unions, `handle_errors` | Yes — `structured/types.rs::StructuredStrategy`, `StructuredOutcome`, `repair.rs` | No union-to-many-tools; no auto-select from profile. |
| Standard content blocks | Yes | Yes — vendor `tinyinference/src/message/types.rs::ContentBlock` (Text/Json/Image/Thinking/Redacted…) | No citations/server_tool_call blocks. |
| `init_chat_model`, model profiles | Yes | Yes — `model_registry/`, `ModelProfile` | |
| Prompt caching middleware | Yes (Anthropic/Bedrock) | Yes — `PromptCacheGuardMiddleware`, `cache/layout.rs` | |
| MCP adapter | Yes (1.4) | Partial — only inside `providers/claude_code/` | |
| Supervisor / swarm / handoffs | Legacy libs; docs pattern | Yes — `subagent_node/`, `delegation/`, `orchestration/`, `parallel::map_reduce`, `SteeringRegistry` | TinyAgents is richer here. |
| Deep Agents: FS backends, subagent `task` tool, memory/skills files, permissions, sandboxes | Yes (harness) | No in TinyAgents; OpenHuman has `skills/`, `memory/`, `sandbox/`, `security/approval` | Correct layering already. |
| Dangling tool-call patch | Yes (deepagents) | Partial — `summarization/trim.rs` orphan handling | |
| Test fakes / trajectory eval | `GenericFakeChatModel`, `agentevals` | Yes — `testkit/` (trajectory assertions), `graph/testkit/` | |
| Tracing hooks (`trace_policy`, LangSmith) | Yes | Yes — Langfuse exporter, `TracingMiddleware`, `RedactingSink` | No per-node trace policy. |

## 3. Features TinyAgents lacks, ranked by value

### 3.1 Middleware control outcomes (`jump_to`, state updates, `Command` from tools)
What: LangChain node hooks return a state dict and may set `jump_to: "model"|"tools"|"end"` (declared
via `@hook_config(can_jump_to=[...])`); `wrap_model_call` may return `ExtendedModelResponse(model_response,
command)`; `wrap_tool_call` and tools may return `Command(update=..., goto=...)`.
```python
class ModelCallLimitMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if state["model_call_count"] >= self.limit:
            return {"jump_to": "end", "messages": [AIMessage("limit reached")]}
```
Why: this is what makes limits, HITL, budget stops, early-exit tools and fallback re-routing
composable without side channels. `docs/sdk-gaps.md` §13 already asks for it.
Mapping: add a `MiddlewareControl` enum returned from `before_model`/`after_model`/`before_tool`/
`after_tool` (`Continue | JumpTo(LoopTarget) | StopWith(AgentRun) | Interrupt(Interrupt)`), let
`MiddlewareModelOutcome`/`MiddlewareToolOutcome` (already `#[non_exhaustive]`) gain `Command`
variants carrying `graph::Command<Update>`-style updates, and let `ToolResult` carry an optional
`ToolCommand { state_update: Value, goto: Option<LoopTarget>, return_direct: bool }` so
`agent_loop/tools.rs` can honour it. Precedence rule: first control outcome wins in hook order
(LangChain accumulates commands inner-first; keep it simple and documented).

### 3.2 Durable human-in-the-loop in the harness (`HumanInTheLoopMiddleware`)
What: `after_model` issues one `interrupt(HITLRequest{action_requests, review_configs})` for the
whole tool-call batch; resume with `Command(resume={"decisions":[{"type":"approve"} |
{"type":"edit","edited_action":{name,args}} | {"type":"reject","message"} |
{"type":"respond","message"}]})`; `InterruptOnConfig{allowed_decisions, description, args_schema,
when}`; requires checkpointer + thread_id.
Why: TinyAgents' `HumanApprovalMiddleware` is an in-process `Fn(&ToolCall) -> bool`; a desktop
assistant needs approval to survive process restart and to edit args. Requires 3.1.
Mapping: `HumanApprovalMiddleware` returns `MiddlewareControl::Interrupt(Interrupt{payload:
HitlRequest})`; the agent loop persists a `harness` checkpoint (via `tinyagents-session` run ledger
or a graph `Checkpointer` when the loop runs as a graph node) and `AgentHarness::resume(thread,
HitlResponse)` applies decisions in `before_tool`. OpenHuman's `security/approval` becomes the UI.

### 3.3 Task-level cache and per-node retry/timeout/error handler
What: `add_node(name, fn, retry_policy=RetryPolicy(max_attempts, backoff_factor, retry_on),
cache_policy=CachePolicy(key_func, ttl), timeout=TimeoutPolicy(run_timeout, idle_timeout),
error_handler=fn(state, NodeError) -> Command|None, defer=True)`; `set_node_defaults(...)`.
Cache key = `(namespace, xxh3(key_func(input)))`, backends `InMemoryCache`, `SqliteCache`; cached
tasks stream with `cached=True`. `error_handler` runs after retries are exhausted and is durable.
Why: deterministic replays, cheap re-runs of expensive nodes during development, per-node
resilience without wrapping every handler.
Mapping: `GraphBuilder::with_node_policy(node, NodePolicy{retry: Option<RetryPolicy>, timeout,
idle_timeout, cache: Option<NodeCachePolicy>, on_error: Option<ErrorHandler<State,Update>>})`;
reuse `tinyagents_harness::retry::RetryPolicy` and `harness/src/cache` traits (`ResponseCache`
already keys by hash; add a `TaskCache` trait keyed by `(graph_id, node_id, hash(send_arg|state
projection))`). Emit `TaskCompleted{cached: true}` in `GraphEvent`.

### 3.4 `DeltaChannel` / writes-only checkpoint history and `Overwrite`
What: `Annotated[list, DeltaChannel(reducer, snapshot_frequency=50)]` stores only per-step writes;
`BaseCheckpointSaver.get_delta_channel_history` replays them; `Overwrite(value)` resets the
baseline. LangGraph shipped it in 1.2 and spent 1.2.5–1.2.11 fixing `update_state`/round-trip bugs.
Why: TinyAgents writes the full `state: State` every superstep; for message-heavy long threads
checkpoint volume is O(N²). Deepagents adopted it precisely for `messages`.
Mapping: add `Checkpoint.channel_deltas: Option<BTreeMap<String, Vec<ChannelUpdate>>>` with
`Checkpointer::delta_history(config, channel)`; make `Messages`/`Topic` channels opt into delta
persistence via `ChannelSet::with_delta(name, snapshot_every)`; add `ChannelUpdate::overwrite(v)`.
Learn from LangGraph's bugs: keep `update` and `replay` on one code path.

### 3.5 Unified stream parts + `tasks`/`checkpoints` modes, `stream_events v3`
What: `stream(version="v2")` yields `StreamPart{type, ns: tuple, data}` for
`values|updates|messages|custom|checkpoints|tasks|debug`; `subgraphs=True` fills `ns` with
`("node:task_id", ...)`; `stream_events(version="v3")` gives `ProtocolEvent{seq, method, params}`
with projections (`stream.messages`, `stream.tool_calls`, `stream.subagents`).
Why: UIs need one cursor over nested runs. `sdk-gaps.md` §3/§6 ask for the same (reasoning/tool-arg
deltas, late-attach replay).
Mapping: TinyAgents already has `GraphEvent` + `GraphObservation` journal; add `ns: Vec<String>`
and `seq` to the envelope, add `TaskStarted/TaskResult` and `CheckpointSaved` projections as
`StreamMode::Tasks|Checkpoints`, and a `StreamProjection` adapter that folds `GraphEvent` +
harness `AgentEvent` into `stream.messages`/`stream.tool_calls` views.

### 3.6 `ToolRuntime` parity and `return_direct`
What: `ToolRuntime{state, context, config, stream_writer, tool_call_id, store, tools,
execution_info}` injected into tools by parameter type; `@tool(return_direct=True)` exits the loop
when all executed tools are return_direct; `response_format="content_and_artifact"` puts a
non-model payload in `ToolMessage.artifact`.
Mapping: extend `ToolExecutionContext` with `call_id`, `store: Option<Arc<dyn NamespacedStore>>`,
`state_view: Option<Arc<dyn Any>>` (or a typed `StateHandle<State>`), and `stream: EventSink`
custom-write helper; add `ToolSchema.return_direct: bool` and `ToolResult.artifact: Option<Value>`
(vendor `tinytools` change). Pair with 3.1 for the loop exit.

### 3.7 Semantic search on the namespaced store
What: `InMemoryStore(index={"embed": embeddings, "dims": 1536, "fields": ["text", "$"]})`;
`store.search(ns, query="...")` ranks by cosine; `put(..., index=False)` opts out.
Mapping: `NamespacedStore::search` already has `query`; add `IndexConfig{embedder: Arc<dyn
Embeddings>, dims, fields}` to `InMemoryNamespacedStore` using `tinyinference-embeddings`, and a
`VectorNamespacedStore` adapter over the existing `retriever/` vector store trait.

### 3.8 `interrupt_before/after`, `response_schema`, graceful drain
Small items: compile-time `interrupt_before/after` selectors are documented in
`docs/modules/graph/interrupts.md` but absent from code; `interrupt(value, response_schema=)`
validates the resume payload; `RunControl.request_drain(reason)` lets a node finish the current
superstep and stop cleanly (`GraphDrained`). Mapping: `GraphBuilder::interrupt_before(nodes)`,
`Interrupt.response_schema: Option<Value>` validated in `resume`, and a `Drain` variant on
`SteeringCommand` that the executor honours at the boundary.

### 3.9 Functional API
`@entrypoint(checkpointer)` + `@task` turn ordinary code into a one-node Pregel where each task
result is a `RETURN` pending write, so resume replays completed tasks without re-running them.
Value for TinyAgents is moderate (Rust closures are already ergonomic), but a `durable_task(ctx,
key, async fn)` helper inside a node handler — memoising by `PendingWrite` — would give the same
"side effects before an interrupt are not repeated" guarantee the interrupts doc currently pushes
onto users ("must be guarded by idempotency keys").

### 3.10 MCP adapter, provider tool search, node `trace_policy`
`langchain.mcp.MCPAdapter` (1.4) lists/executes MCP tools as `BaseTool`s; `ProviderToolSearchMiddleware`
defers large tool catalogs to provider-side search; `add_node(trace_policy=TracePolicy(process_inputs,
process_outputs))` redacts per node. TinyAgents has MCP only inside the Claude Code provider and
redaction only at the sink; both are modest additions on existing traits.

## 4. Design lessons

**Channel semantics.** LangGraph schedules nodes from channel versions; edges are sugar over
`branch:to:*` channels. This gives uniform semantics for `Send`, deferred joins
(`NamedBarrierValueAfterFinish`) and subscriptions, but it makes "why did this node run?" hard to
explain and is the source of a long tail of `update_state`/`versions_seen` bugs. TinyAgents'
explicit active-set + waiting-edge model is easier to export (`export/`) and debug, at the cost of
special mechanisms (`add_barrier_relief`) where LangGraph gets them for free. Keep the explicit
model; borrow `defer` as a scheduling flag rather than a channel type.

**Checkpoint format.** LangGraph's checkpoint is channel-level (`channel_values` + versions), so a
saver can be schema-agnostic and delta history is natural. TinyAgents persists a typed `State`
blob plus `pending_activations`/`barrier_arrivals`, which is simpler and type-safe but
all-or-nothing. TinyAgents is better on failure semantics: async durability surfaces write errors
at the next boundary and always syncs terminal/interrupt checkpoints; LangGraph's default `async`
mode can lose the last step on crash and the docs say so.

**Interrupt resume.** Both re-run the node from the top. LangGraph's `interrupt()` is a call that
can appear anywhere (index-matched), which is ergonomic but fragile (docs list "don't wrap in
try/except", "don't reorder interrupts"). TinyAgents' `NodeResult::Interrupt` forces one interrupt
per node return, which is explicit and serialisable but makes multi-step approvals inside one node
awkward. A `ctx.interrupt(payload)` helper that reads `ctx.resume` by index would get both.

**Streaming.** LangGraph went through three formats (tuples, `StreamPart` v2, protocol events
v3) because the first had no namespace or sequence. TinyAgents should add `ns`/`seq` to
`GraphEvent` now rather than later. Typed enums beat `dict` payloads; keep them.

**Middleware composition.** LangChain's split of node hooks (return state, may jump) vs wrap hooks
(onion) is the same as TinyAgents', but LangChain lets middleware own state keys, tools and stream
transformers, which is what made summarization/todo/HITL/PII shippable as single classes. TinyAgents'
hooks that only return `Result<()>` push that logic into the loop or host adapters (`sdk-gaps.md`
§13). LangChain's weakness: `state_schema` merging across middleware is untyped `TypedDict`
unioning; TinyAgents can do better with a typed `MiddlewareState` extension slot on `RunContext`.

**Agent loop as a graph.** `create_agent` compiles the loop into a `StateGraph`, so every graph
feature (checkpoints, interrupts, streaming, time travel) applies to the agent loop with zero extra
code. TinyAgents runs the loop in `agent_loop/run_loop.rs` and separately offers `subagent_node`;
the harness README even describes the loop "as an explicit state machine when callers need
inspection, checkpointing, HITL". Finishing that (loop-as-`CompiledGraph`) would close 3.1/3.2 in
one move.

**Where TinyAgents is ahead.** Steering commands with policy checks, detached task registry with
durable stores, parallel failure policies (`quorum`/`race`/`compare`), goal/task-board primitives,
budget middleware, prompt-cache layout guards, artifact offload, tool timeouts with grace, and a
first-class registry + `.rag` blueprints. LangGraph OSS has none of these; LangGraph *Platform* has
some (background runs, cron, multitask) as hosted services.

## 5. Runtime-level vs harness-level split

Calibration: `openhuman:crates/openhuman-core/src` (the OpenHuman desktop host) already owns `skills/`, `memory/`, `sandbox/`,
`cron/`, `hooks/`, `security/{approval,audit,bubblewrap}`, `agent/tools`, `agent/tool_policy.rs`,
`agent/orchestration/{worktree,spawn_parallel_graph,running_subagents}` and `agent/harness/{memory_context,
artifact_offload,tool_result_artifacts}`. Those are the OpenHuman analogues of Deep Agents.

| Gap | Layer | Reason |
|---|---|---|
| 3.1 middleware control outcomes / tool `Command` | **Runtime (harness crate)** | Loop control vocabulary; hosts cannot add it from outside. |
| 3.2 durable HITL request/decision protocol | **Runtime** for the interrupt/resume/decision types; **OpenHuman** for the approval UI and `security/approval` policy | Mirrors LangChain: middleware in library, policy in app. |
| 3.3 per-node retry/cache/timeout/error handler | **Runtime (graph crate)** | Pure scheduler policy. |
| 3.4 delta checkpoint history / `Overwrite` | **Runtime (graph crate)** | Checkpoint format. |
| 3.5 unified stream parts, tasks/checkpoints modes | **Runtime** | Contract UIs depend on; OpenHuman keeps only format adapters (`sdk-gaps.md` §6). |
| 3.6 `ToolRuntime` parity, `return_direct`, artifact | **Runtime** (`ToolExecutionContext`, vendor `tinytools`) | Tool contract. |
| 3.7 semantic store search | **Runtime** (trait + in-memory impl); backend choice in OpenHuman `memory/` | |
| 3.8 `interrupt_before/after`, `response_schema`, drain | **Runtime** | |
| 3.9 durable task memoisation | **Runtime** | Idempotency guard belongs beside `PendingWrite`. |
| 3.10 MCP adapter | **Runtime** (generic `McpToolSource` in harness) — OpenHuman already has `mcp/` | Move generic bits down only if OpenHuman's is reusable. |
| Provider tool search, node trace policy | Runtime, low priority | |
| Deep Agents FS tools, backends, sandboxes, permissions | **OpenHuman** (`sandbox/`, `agent/tools`, `tool_policy.rs`) | LangChain also keeps these out of the runtime; only `WorkspaceIsolation` roots stay in TinyAgents. |
| AGENTS.md memory, SKILL.md skills, harness profiles | **OpenHuman** (`memory/`, `skills/`) | Prompt conventions over a file backend. |
| Summarization-with-offload thresholds, large-result eviction | Mechanism in runtime (`artifacts/`, `handoff.rs` exist); thresholds/prompts in OpenHuman | Already split this way. |
| Subagent `task` tool, fork mode | Runtime has `SubAgentTool`/`subagent_node`; prompts and delegation policy in OpenHuman | Already split. |
| Double texting, cron, thread TTL, background runs | Runtime has `run_queue`, `DetachedTaskRegistry`, store TTL; scheduler/cron and UI in OpenHuman (`cron/`) | LangGraph puts these in the paid server. |
| Rubric grader, LLM tool emulator | OpenHuman / test tooling | |

## 6. Sources

- LangGraph releases: https://github.com/langchain-ai/langgraph/releases ; changelog https://docs.langchain.com/oss/python/langgraph/changelog-py
- Pregel internals: https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/_algo.py , `_loop.py`, `_runner.py`, `_retry.py`; https://docs.langchain.com/oss/python/langgraph/pregel
- Types (`Command`, `Send`, `Overwrite`, `interrupt`, `RetryPolicy`, `CachePolicy`, `TimeoutPolicy`): https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/types.py ; `graph/state.py`; `runtime.py`; `func/__init__.py`
- Checkpoint base: https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/checkpoint/base/__init__.py ; https://docs.langchain.com/oss/python/langgraph/checkpointers ; https://docs.langchain.com/oss/python/langgraph/use-time-travel
- Interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts ; fault tolerance: https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- Streaming: https://docs.langchain.com/oss/python/langgraph/streaming ; https://docs.langchain.com/oss/python/langgraph/event-streaming ; https://docs.langchain.com/oss/python/langchain/event-streaming
- Stores: https://docs.langchain.com/oss/python/langgraph/stores ; functional API: https://docs.langchain.com/oss/python/langgraph/functional-api
- DeltaChannel bugs: https://github.com/langchain-ai/langgraph/issues/8821 ; https://github.com/langchain-ai/langgraph/pull/8961
- Platform-only: https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/sdk-py/langgraph_sdk/schema.py ; https://docs.langchain.com/langgraph-platform/double-texting ; https://docs.langchain.com/langsmith/cron-jobs ; https://docs.langchain.com/langsmith/configure-ttl
- LangChain releases: https://github.com/langchain-ai/langchain/releases ; migration: https://docs.langchain.com/oss/python/migrate/langgraph-v1
- `create_agent`: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain_v1/langchain/agents/factory.py
- Middleware types and built-ins: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain_v1/langchain/agents/middleware/types.py ; https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware ; https://docs.langchain.com/oss/python/langchain/middleware/built-in ; https://docs.langchain.com/oss/python/langchain/middleware/custom
- Structured output: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain_v1/langchain/agents/structured_output.py ; https://docs.langchain.com/oss/python/langchain/structured-output
- ToolNode / ToolRuntime: https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/prebuilt/langgraph/prebuilt/tool_node.py ; https://docs.langchain.com/oss/python/langchain/tools
- Messages/content blocks: https://docs.langchain.com/oss/python/langchain/messages ; prompt caching middleware: https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/partners/anthropic/langchain_anthropic/middleware/prompt_caching.py
- Testing/eval: https://docs.langchain.com/oss/python/langchain/test/unit-testing ; https://docs.langchain.com/oss/python/langchain/test/evals
- Multi-agent: https://docs.langchain.com/oss/python/langchain/multi-agent (+ `subagents`, `handoffs`, `router`) ; https://docs.langchain.com/oss/python/migrate/langgraph-supervisor ; https://github.com/langchain-ai/langgraph-supervisor-py ; https://github.com/langchain-ai/langgraph-swarm-py ; https://docs.langchain.com/oss/python/langgraph/use-subgraphs ; https://docs.langchain.com/oss/python/langgraph/graph-api
- Deep Agents: https://github.com/langchain-ai/deepagents (libs/deepagents/deepagents/graph.py, middleware/, backends/, ARCHITECTURE.md) ; https://docs.langchain.com/oss/python/deepagents/human-in-the-loop ; PyPI `deepagents`, `deepagents-code`, `deepagents-acp`, `deepagents-talon`
- TinyAgents files checked: `docs/spec/README.md`, `docs/sdk-gaps.md`, `ROADMAP.md`, `docs/modules/{harness,graph}/README.md`, `docs/modules/graph/interrupts.md`, `crates/tinyagents-{harness,graph}/src/lib.rs`, `graph/src/{command,checkpoint,stream,builder,channel,compiled}/`, `harness/src/{middleware,tool,structured,store/namespaced,cache,artifacts,handoff.rs,steering}`, `vendor/tinytools/crates/tinytools/src/result/types.rs`, `vendor/tinyinference/crates/tinyinference/src/message/types.rs`

Unverified (flagged by the research agents): exact version that introduced `Overwrite` and
`interrupt(response_schema=)`; whether "enqueue" is the server's default multitask strategy;
HITL `interrupt_mode` (PR title only, absent from source); `Runnable.as_tool` as a recommended
agent-as-tool path; AgentCore sandbox package location; formal deprecation of `langgraph-swarm`.
