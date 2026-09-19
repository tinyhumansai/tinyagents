# Execution Plan

Ordering principle: fix what is wrong before adding what is missing; land
format and contract changes (stream, checkpoint, error, tool result) once,
early, and non-exhaustively so later work does not break downstream matches;
prefer the smallest change that closes a documented promise over a new
abstraction. Every item is one PR unless marked (multi-PR). Finding ids
(`H-C1`, `G-I3`, `W-I5`) refer to
[`code-review-harness.md`](code-review-harness.md),
[`code-review-graph.md`](code-review-graph.md) and
[`code-review-workspace.md`](code-review-workspace.md); gap ids (`A1`,
`D3`) refer to [`feature-gaps.md`](feature-gaps.md).

## Phase 0: hygiene and truth in docs (1 week, parallelisable)

Cheap, independent, and they stop the next phases from being reviewed
against wrong assumptions.

| Item | Effort | Refs |
|---|---|---|
| CI: add `--workspace` to every cargo step in `ci.yml` / `release.yml`; add `-D rustdoc::broken_intra_doc_links`; add `cargo machete` | S | W-C1, W-M12 |
| Remove unused deps (`registry→graph`, graph `reqwest`/`sha2`/`chrono`, harness `bytes`); add `[workspace.dependencies]`, `rust-version = "1.88"`, `unsafe_code = "deny"` with per-site `SAFETY` allows | S | W-I11, W-I12 |
| Delete `tinyagents-tracing`; depend on `tracing` directly; remove the three crate-level `allow(dead_code, …)`; fix what they were hiding (`StreamMode`, `WRITES_IDX_*`, `command_nodes`) | M | W-I5, G-I11 |
| Gate live tests with `#[ignore = "network"]` + one `tests/common/live.rs` helper; run them in an explicit `TINYAGENTS_LIVE=1` job | S | W-I10 |
| Feature-gate `claude-code` (subprocess driver + uuid/tempfile/wait-timeout/dirs) and `langfuse` (reqwest); rename `tools` → `builtin-tools` | S | H-M11, W-M10 |
| Docs truth pass: mark `sdk-gaps.md` §2 implemented and §3 partial; rewrite `audit.md` OpenAI entry; fix the concurrency, `max_concurrency`, `UnknownToolPolicy` default and `on_tool_delta` claims in `docs/modules/harness/`; split `harness/README.md` (547 lines); add `docs/modules/registry/implementation-status.md`; mark `interrupts.md` / `subgraphs.md` / `checkpointing.md` / `execution.md` unimplemented items as target; fix the unparseable `.rag` README example; list `definition` and `orchestration` crates in `README.md` / `docs/spec/README.md` / `CLAUDE.md` | M | H §1 table, G §1, W-I13, W-M3, W §4 |
| `pub use tinyinference_llm; pub use tinytools;` from harness and fix the README dependency snippet | S | W-I4 |
| Fix example headers (`cargo run -p tinyagents-integration-tests --example …`) | S | W-M14 |

## Phase 1: correctness (2–3 weeks)

### 1a. Harness

| Item | Effort | Refs |
|---|---|---|
| Replace the `Arc<dyn Any>` host-authority cast with a checked downcast (or trait object); split `RunContext::child` into authority-propagating and non-propagating forms; make `host_invocation_binding` return a borrow/`Arc` | M | H-C1, H-I11, H-R2 |
| Keep signed `Thinking` blocks verbatim on the streaming and cache-replay paths unless a delta middleware changed the text; add a streaming + thinking + tool-call regression test | S | H-C2 |
| On first fatal error in the concurrent tool path, emit `ToolFailed` for every already-started sibling before returning | S | H-C3 |
| Distinguish `CallTimeout` from run-deadline `Timeout` so per-call ceilings retry/fall back | S | H-I1 |
| Gate text-dialect tool-call recovery behind `RunPolicy` (default off when the profile reports native tool calling), skip fenced code, emit a `ControlApplied` event | S | H-I2 |
| `spawn_blocking` in `SqliteResponseCache` and `FileStore::get/put/delete` | S | H-I4 |
| Fail-closed host allow-list: `Option<HashSet>` with `None` = deny under a host flag | S | H-I9 |
| Route steering commands by target run id; reject disallowed commands individually instead of killing the run after draining | M | H-I5, H-M7 |
| `#[non_exhaustive]` on `AgentEvent` and `TinyAgentsError`; emit or delete placeholder variants | S | H-I3 |
| Typed hosted errors (`HostedError{kind, run}`) instead of `Model("hosted agent invocation failed")` | S | H-I6 |
| Apply `relaxed_json` recovery to provider-invalid tool args before returning the tool error | S | H-I13 |
| Drop the `lifecycle_middleware == 0` concurrency precondition; add `RunLimits::max_tool_concurrency` with `buffered(n)` | S | H-I8 |
| Correlate `RetryMiddleware` call ids with the loop's; document it as an alternative to `RunPolicy::retry` (full unification is Phase 2) | S | H-I7 |
| Minor batch: unanswered `tool_calls` after `StopWithFinal`, deterministic child run ids, `map_tool_dispatch_error` flattening, `ToolRegistry::register` duplicate diagnostic, `LimitTracker` start time, `claude_code` semaphore order, lossy casts | S each | H-M1…M14 |

### 1b. Graph (multi-PR, in this order)

| Item | Effort | Refs |
|---|---|---|
| R1: split `execute_run` into `RunCtx` / `StepRunner` (returns *all* results) / `Boundary` / `Resume`; pure code motion under the existing 100 tests | M | G-R1, G-M1 |
| R2: real pending writes: persist task outputs (`Update: Serialize` behind a `DurableUpdate` marker for durable backends), fold every `Ok` sibling, finish step N before routing on resume; add the interrupted-vs-uninterrupted equivalence test and a higher-index-sibling test; retire `parallel_interrupt_pauses_at_lowest_index_branch`'s lossy assertion | L | G-C1, G-C2, D1 |
| Executor-level thread lease: in-process `ThreadLockMap` in `execute`, optional `Checkpointer::try_claim/renew/release`; drop `delegation/run.rs`'s private map | S/M | G-C3, G-R4 |
| Subgraph retry resumes the child's partial progress; namespace `Send` fan-out of subgraphs by `[node_id, task_id]`; expose `NodeContext.task_id` | M | G-C4, G-I1, G-R5 |
| Carry `interrupts` / `interrupted_nodes` through `update_state`; seed `steps` and `node_visits` from the loaded checkpoint; restart-safe interrupt ids via `ids` nonce | S | G-I2, G-I3, G-I7 |
| Panic catching around handlers, `run_with_cancel(token)`, drop guard writing `Cancelled`; add `put_writes` to async durability; `Instant` for deadlines; log status-store errors | M | G-I4, G-M3, G-M4, G-M5 |
| SQLite checkpointer: `spawn_blocking` everywhere, WAL/busy_timeout/synchronous pragmas, `LIMIT`-driven lineage query, one transaction per boundary; File checkpointer: header-only `list`, `get_scoped` override, append-only writes sidecar | M | G-I8, G-I9 |
| `add_edge` fan-out or duplicate error; typed `Route` labels | S | G-I10, G-M8 |
| Executor tests behind `FileCheckpointer` / `SqliteCheckpointer` across a fresh process nonce | S | G §4 |

### 1c. Language / registry

| Item | Effort | Refs |
|---|---|---|
| `build_graph`: fail with `Compile` on any populated blueprint field it ignores, and say so in `implementation-status.md` (full lowering is Phase 5) | S | W-I2, G-M9 |
| Deterministic default model in `to_model_registry()` (explicit default or insertion order) | S | W-I3 |
| Language errors carry spans: `compile`/`bind` return `Vec<Diagnostic>` (`Serialize`), single facade, one binding gate | M | W-I6, W-I7 |
| Registry: `set_metadata` / `remove`, `impl DefinitionRegistry for CapabilityRegistry`, carry the definition-lookup error | S | W-I8, W-I9 |
| Language minor batch: duplicate item diagnostics, one list-separator rule, `router` item, `Blueprint` serde defaults + `schema_version`, boolean literal | S each | W-M1…M9 |

## Phase 2: loop control and human-in-the-loop (3–4 weeks)

Builds the vocabulary that A1–A4 and B1–B2 share; everything later
(durable HITL, evals, capability bundles) depends on it.

| Item | Effort | Refs | OpenHuman hand-off |
|---|---|---|---|
| Middleware control outcomes: `MiddlewareControl::{Continue, JumpTo(LoopTarget), StopWith, Interrupt}` from all four hooks; `Command` variants on the wrap outcomes; `ToolCommand{state_update, goto, return_direct}` on `ToolResult`; `should_stop_after_turn` / `terminate` hint; precedence rule documented | M | A1, `sdk-gaps.md` §13 | `stop_hooks.rs`, `plan_review` gate become middleware outcomes |
| Unify the four retry/fallback engines into the loop's, with middlewares as policy overrides | M | H-R3 | — |
| `Turn` object built once per turn, borrowed by the wrap onion; cache the tools fingerprint; `EventSink::emit` skips enqueue with no listeners | M | H-I10, H-R1 | — |
| Deferred tools: `LoopExit::Deferred(DeferredToolRequests)`, `DeferredToolResults{approvals, calls}` with approve / edit / reject / respond, `AgentTurnRequest::with_deferred_results`, `ToolOutcome::{ApprovalRequired, Deferred}` reachable from `ToolPolicy.access.approval_required`; `HumanApprovalMiddleware` re-based on it; harness checkpoint through the session run ledger | L | A2 | `security/approval::ApprovalGate` becomes a `DeferredToolHandler`; pending rows + dialog stay in OpenHuman |
| Output-validation retry loop: `OutputRetryPolicy`, `OutputValidator<State,Ctx>` returning `ModelRetry(msg)`, `AgentEvent::OutputRetry`, `run.structured_as::<T>()`; `ModelRetry` / `ToolFailed` vocabulary for tool errors | M | A3 | `required_output.rs` re-asks via the runtime |
| Wire `RunQueue` into the loop: drain `Steer` after tool results, `Followup` when about to return, `QueueMode` on the harness | M | A4 | `agent/harness/run_queue/` deleted in favour of the SDK's |
| `ToolExecutionContext` gains `call_id`, `store`, `state_view`, `stream` helper; `ToolResult{model_content, follow_up: Vec<ContentBlock>, metadata}` in `tinytools`; loop appends follow-up as a user message | M | B1, B2 | `tool_result_artifacts`, `artifact_offload` use `metadata` instead of JSON stuffing |
| Prompted / union / output-function structured modes; `EndStrategy` | S | A6 | — |

## Phase 3: streaming and events (2 weeks)

| Item | Effort | Refs | OpenHuman hand-off |
|---|---|---|---|
| `ModelStreamItem::{BlockStart, BlockDelta, BlockEnd}` with `content_index` in `tinyinference-llm`; adapters emit them; `Failed` terminal carries the partial `AssistantMessage` + `stop_reason` | M | C1 | `progress.rs` renders blocks instead of three channels |
| Frame codec + reducer in `harness/src/stream/frame.rs`; journal persists frames; `ToolProgress` finally emitted via `on_tool_delta` | M | C2 | reconnecting web/TUI clients rebuild partials |
| `GraphEvent` envelope with `run_id`, `task_id`, `ns`, `seq`; `StreamMode::{Tasks, Checkpoints}`; `StreamProjection` folding graph + harness events | M | C3, G-M6 | `tinyagents/replay` pages by `seq` |
| `JournalGraphSink::dropped()` exposed and documented | S | G-M7 | — |
| OTel GenAI semconv `JournalSink` behind an `otel` feature | M | C4 | Langfuse and OTel selectable in config |

## Phase 4: durability v2 (3–4 weeks, multi-PR)

| Item | Effort | Refs |
|---|---|---|
| Checkpoint v2: `version`, `created_at`, one `tasks` list, `completed`, v1 decoder, migration test on File and SQLite | M | D3, G-I6, G-R3 |
| Serialisable `ChannelSet` (`{kind, config, value}` + `BinaryAggregate` registry); `channel_versions` in the checkpoint | M | G-I5 |
| Delta-channel history for `Messages`/`Topic` channels with `snapshot_every`; `ChannelUpdate::overwrite`; `Checkpointer::delta_history`; keep `update` and `replay` on one code path (LangGraph's 1.2.5–1.2.11 bug lesson) | L | D3 |
| Per-node `NodePolicy{retry, timeout, idle_timeout, cache, on_error}` reusing `harness::retry::RetryPolicy`; `TaskCache` trait keyed by `(graph_id, node_id, hash)`; `defer` as a scheduling flag; `TaskCompleted{cached: true}` | M | D2 |
| `interrupt_before/after`, `Interrupt.response_schema`, `SteeringCommand::Drain` honoured at the boundary | S | A7 |
| `durable_task(ctx, key, fut)` memoised through `PendingWrite` | S | D5 |
| Lower `WorkflowDefinition` to a `CompiledGraph` behind a feature flag; keep `WorkflowStore` as the status projection and the lease as the durable lock; `workflow/tests.rs` as acceptance | L | G-I12, G-R6 |
| `NodeHandler` takes `Arc<State>`; `Arc<Value>` send args | M | G-M2 |

## Phase 5: sessions, context, and the loop as a graph (3–4 weeks)

| Item | Effort | Refs | OpenHuman hand-off |
|---|---|---|---|
| Transcript entries with `id`/`parent_id`; `CompactionEntry`, `BranchSummaryEntry`, `LabelEntry`, `CustomEntry`; `build_context(tip)` stops at the newest compaction; fork = path copy; SQLite as rebuildable index | L | E1 | `/fork`, `/tree`, labels UI; `session_import` writes the tree |
| Compaction rules: `find_cut_point`, split turns, iterative summaries, `CompactionRecord`, `OverflowClassifier`, `overflow → compact → retry` in `ContextCompressionMiddleware`; `before_compaction` hook may decline/replace | M | E2 | summary prompt text and `/compact` stay |
| `AssistantMessage.origin{provider, api, model}`; `prepare_for_model(&[Message], &ModelProfile)` handoff pass | S | E3 | mid-session model switch just works |
| Tool-effect ledger keyed by `CallId` with `ToolReplay::{Never, Safe}` on `ToolSchema`/policy; `list_unresolved_tool_effects(run_id)` | M | B5 | OpenHuman marks its tools' replay class |
| Agent loop as a `CompiledGraph` (`plan → model → tools → settle` nodes) so checkpoints, interrupts and time travel apply to the loop; `AgentHarness::iter()` | L | A5 | `agent_graph.rs` collapses onto the SDK graph |
| `sanitize_history()` helper; `Message::Custom` | S | E4, E5 | web_chat sanitises via the SDK |
| Full `.rag` lowering in `build_graph`: channels, joins, sends, route tables, timeout/retry/interrupt policies | L | W-I2 | — |

## Phase 6: tool ecosystem (2–3 weeks)

| Item | Effort | Refs | OpenHuman hand-off |
|---|---|---|---|
| `ToolSet<State,Ctx>` trait with `Combined`, `Filtered`, `Prefixed`, `Renamed`, `Prepared`, `ApprovalRequired`, `External`; `ToolRegistry` becomes one `ToolSet`; per-tool `prepare` | M | B3 | `agent/tool_policy.rs` and `tinyagents/middleware` tool filtering shrink to predicates |
| `tinytools-mcp` crate (stdio / HTTP / SSE, prefixes, `process_tool_call`, sampling, elicitation, config loading) implementing `ToolSet` | L | B4 | `mcp/` keeps server config, auth and UI; evaluate reusing its transport |
| Transcript-carried `SystemMessage{sections, tools_added, tools_removed}`, `replay_system_state`, `declare_tool_changes` before `ModelStarted`, `ModelProfile.mid_conversation_system_messages` | M | B6 | tool loadout changes stop busting the cache |
| `Capability{instructions, tools, middleware, model_defaults, exposure}` bundle in `tinyagents-registry`, referenced by `.rag` and `AgentDefinition`, `defer_loading` | M | G3 | `skills/` discovery produces `Capability` values |
| Provider-executed tool parts in `ContentBlock` | S | B7 | — |

## Phase 7: models, providers, evals (2–3 weeks)

| Item | Effort | Refs | OpenHuman hand-off |
|---|---|---|---|
| `ModelProfile` behaviour fields: named `schema_transform`, `default_structured_mode`, `thinking_tags`, compat flags; consumed by `SchemaPreparation` and the structured plan | M | F1 | — |
| Catalog generator from models.dev with tiered pricing; refresh the seed; delete the duplicate docs copy; `from_json` validation; `ModelRouter` renamed or wired | M | F2, W-M7, W-M8 | model picker reads availability by resolved auth |
| `ContentBlock::{Audio, Video, Document}` + provider matrix in `modalities` | M | F3 | `agent/multimodal.rs` marker extraction deleted |
| `CredentialStore` trait + generalised OAuth flow; deferred responses; `on_payload`/`on_response` hooks | M | F4, F5, F6 | keychain storage and login UI stay |
| `tinyagents-evals` crate: `Case<I,O>`, `Evaluator<I,O>`, `Dataset`, `Report`, `LlmJudge` over `AgentHarness`, `Trajectory`-based evaluators | M | G1 | prompt-evals datasets move onto it |
| `testkit::SchemaDrivenModel`, `deny_network_models()` | S | G2 | — |
| Session/store backend conformance suites | S | G4 | — |
| Semantic search `IndexConfig` on the namespaced store | S | D6 | `memory/` backend adapter |

## Sequencing summary

```
Phase 0 ──┬── Phase 1a (harness fixes) ──┐
          ├── Phase 1b (graph R1→R2→…) ──┼── Phase 2 ── Phase 3 ──┐
          └── Phase 1c (lang/registry) ──┘        │              ├── Phase 5 ── Phase 6 ── Phase 7
                                                  └── Phase 4 ──┘
```

Phases 0, 1a, 1b and 1c are independent and can run in parallel worktrees.
Phase 2 needs 1a (control outcomes touch the same loop) and the
`#[non_exhaustive]` change. Phase 3 needs 2 (`ToolProgress`, output retry
events). Phase 4 needs 1b's R1/R2. Phase 5's loop-as-graph needs 2 and 4.
Phases 6 and 7 mostly need 2 (`ToolResult`, `ToolExecutionContext`).

## Suggested first three PRs

1. Phase 0 CI + dependency diet + `non_exhaustive` (one afternoon; unblocks
   honest review of everything else).
2. Harness H-C1 + H-C2 + H-C3 with their regression tests (the three
   reachable production failures).
3. Graph R1 (pure split of `execute_run`) so R2 can be reviewed as a
   semantic change rather than a rewrite.
