# pi (earendil-works/pi) as an agent runtime library, compared with TinyAgents

Research date: 2026-09-19. pi source was read from a `--depth 1` clone at commit `36b60d2e`.
TinyAgents baseline: v2.1.2 (`fc33c43`), vendored `tinyinference` at `219b0ea`.
Paths prefixed `pi:` are relative to the pi checkout; `ta:` to this repository; `ti:` to `vendor/tinyinference`.

## 1. What pi is

- Author: Mario Zechner (`badlogic`, 3743 of the last-page contributions; next: `mitsuhiko` 677,
  `davidbrai` 257). MIT. Repo created 2025-08-09; 107k stars, 13.5k forks, 224 open issues.
- Language: TypeScript (Node/Bun). ~286k lines of TS across 1460 files plus ~50k lines of Markdown.
  Non-test source: coding-agent 72k, agent-core 33k, ai 25k, tui 18k, chord 6.5k, server 2k, evals 1.4k.
- Activity: very high. 100 commits between 2026-09-10 and 2026-09-18; releases v0.84.2 (Aug 14) through
  v0.85.1 (Sep 5), all packages version-locked at 0.85.1. Design docs are in-repo
  (`pi:packages/agent/docs/harness.md` 1468 lines, `pico2.md`, `pico-v3.md`, `values.md`,
  `tool-durability.md`, `assistant-durability.md`) plus external RFCs (rfc.earendil.com/keyword/pi).
- What it is used for: the `pi` interactive coding agent CLI (the product), a headless RPC/JSON mode, and
  the two library layers underneath it (`pi-ai`, `pi-agent-core`) which third parties use for their own
  agents (`pi-chat` Slack automation is a sibling repo). The README is explicit that pi ships **no
  permission system**; sandboxing is delegated to containers (Gondolin micro-VM extension, Docker, OpenShell).
- Supply-chain posture is unusually strict for an npm project: exact-pinned deps, `min-release-age=2`,
  shrinkwrap generated for the CLI, lifecycle-script allowlist, `npm audit signatures` in CI.

## 2. Package map

Runtime-library layer (what the comparison is about):

- **`@earendil-works/pi-ai`** (`pi:packages/ai`) — unified multi-provider LLM API. Ten API protocol
  implementations under `src/api/` (`openai-completions`, `openai-responses`, `azure-openai-responses`,
  `openai-codex-responses`, `anthropic-messages`, `bedrock-converse-stream`, `google-generative-ai`,
  `google-vertex`, `mistral-conversations`, and pi's own `pi-messages` SSE protocol) plus 41 provider
  presets under `src/providers/` (OpenAI, Anthropic, Google, Vertex, Bedrock, Mistral, Groq, Cerebras, xAI,
  OpenRouter, Vercel AI Gateway, Cloudflare (2), DeepSeek, NVIDIA, GitHub Copilot, OpenAI Codex, z.ai (2),
  MiniMax (2), Moonshot (2), Kimi, Qwen token plans (3), Xiaomi (4), Together, Baseten, Fireworks,
  HuggingFace, OpenCode (2), Radius, Ant Ling). Owns the message/content model, streaming event protocol,
  model catalog with cost, auth resolution (API key + 7 OAuth flows + credential store), retry
  classification, context-overflow detection, cross-provider transcript transforms, image generation
  (`openrouter-images`), and "deferred" (background/batch) responses. Core entry is side-effect free;
  providers/APIs are imported per path for tree-shaking.
- **`@earendil-works/pi-agent-core`** (`pi:packages/agent`) — two runtimes in one package. (a) The legacy
  `Agent` class + `agentLoop()` (`src/agent.ts`, `src/agent-loop.ts`, ~1.5k lines): in-memory state, tool
  execution (parallel/sequential), steering/follow-up queues, `transformContext`/`convertToLlm`, hooks.
  (b) `AgentHarness` (`src/harness/`, ~31k lines): a durable, crash-recoverable operation state machine over
  a persisted conversation tree with Branches/AgentLanes, compaction, branch summaries, hooks, typed
  telemetry, session backends (memory, JSONL, SQLite via `packages/session-backends/sqlite-node`), and
  built-in file/bash tools. Per `harness.md` §0.9, WP00–WP07 are complete; only `watchSession()` is stubbed.
  Note: the shipped `pi` CLI still runs on (a) plus the coding-agent `SessionManager`; `AgentHarness` is used
  only by `pi:packages/coding-agent/src/experimental/*` workers (grep confirms).
- **`@earendil-works/pi-telemetry`** — vendor-neutral typed span/event schema contracts (`defineTelemetrySchema`),
  in-memory context, conformance tests. Small (935 lines).
- **`@earendil-works/pi-durable`** (757 lines) — the next-generation "pico v5" record contracts and an
  in-memory storage; design only beyond that. Runtime-library in intent, not yet usable.
- **`@earendil-works/pi-protocol` / `pi-client` / `pi-server`** — CBOR framed transport for remote pi sessions
  (experimental). Runtime-adjacent; a host concern in TinyAgents terms.
- **`@earendil-works/chord`** — generic plugin/facet/service composition runtime with replicated state; not
  pi-specific. Product infrastructure rather than agent runtime.

Harness/product layer:

- **`@earendil-works/pi-coding-agent`** — the CLI/TUI/RPC product: `AgentSession` (3.6k lines),
  `SessionManager` (JSONL v3 tree), extension runner and `ExtensionAPI` (40+ events, `registerTool`,
  `registerCommand`, `registerProvider`, UI widgets), skills (agentskills.io), prompt templates, packages
  (`pi install npm:/git:`), compaction driver, auto-retry, models.json custom providers, themes, keybindings.
- **`@earendil-works/pi-tui`** — terminal UI library with differential rendering. **`pi-evals`** — eval runner.

## 3. Feature inventory

| Feature | pi has | TinyAgents has | Notes |
|---|---|---|---|
| Native provider protocols | 10 APIs (`pi:packages/ai/src/api/`) | Partial: OpenAI chat + Responses + Codex + local, Anthropic (`ti:crates/tinyinference-llm/src/providers/{openai,anthropic}`); Claude Code / Agent SDK bridges (`ta:crates/tinyagents-harness/src/providers/`) | No Google/Vertex/Bedrock/Mistral-native in TA. |
| Provider presets | 41 (`pi:packages/ai/src/providers/*.ts`) | Partial: generic `OpenAiConfig` (`ti:.../providers/openai/config.rs:13`) with OpenRouter detection; no preset table | Spec README lists presets; source has one configurable adapter. |
| Model catalog w/ cost | Generated from models.dev per provider, tiered pricing `ModelCost.tiers` (`pi:packages/ai/src/types.ts:952`) | Partial: `ModelCatalog` snapshot with 5 seed models (`ta:crates/tinyagents-registry/model-catalog.snapshot.json`), `ModelPricing` (`ta:crates/tinyagents-harness/src/cost/types.rs:13`), live `/models` parsing (`ti:.../catalog/`) | TA has no tiered pricing. |
| Capability flags | `Model.reasoning/input/compat` (+ per-API compat matrices, `types.ts:667`) | Yes: `ModelProfile` + `CapabilitySet` (`ti:.../model/types.rs:193,253`) | TA's requirement-matching is richer; pi's compat matrix is far broader (mid-convo system msgs, strict tools, cache retention, session affinity). |
| Message model | 4 roles + app roles via declaration merging (`pi:packages/agent/src/types.ts` `CustomAgentMessages`) | Partial: closed `Message` enum; `ContentBlock::{Thinking,RedactedThinking,ProviderExtension,Json}` (`ti:.../message/types.rs:21`) | TA has richer blocks, no custom roles. |
| Transcript-carried prompt/tool patches | `SystemMessage.sections/toolsAdded/toolsRemoved` (`types.ts:484`), `declareToolChanges` (`agent-loop.ts:291`) | No (grep: no tool-delta system messages; `PromptSegment` is cache layout only) | See gap 1. |
| Streaming event model | Content-indexed `*_start/_delta/_end` + `partial` (`types.ts:645`) | Partial: `ModelStreamItem::{MessageDelta,ToolCallDelta,UsageDelta,Completed}` with `MessageDelta{text,reasoning,tool_call}` (`ti:.../model/types.rs:632`, `message/types.rs:137`) | TA has channels but no block boundaries/indices. |
| Compact durable stream frames | `AssistantMessageFrameEncoder`/`reduceAssistantMessageFrames` (`pi:packages/ai/src/utils/assistant-message-frame.ts:139,372`) | Partial: `AgentEvent::ModelDelta` journaled (`ta:crates/tinyagents-harness/src/events/types.rs`), no frame codec/reducer | See gap 2. |
| Partial-JSON tool args | `parseStreamingJson` | Yes: `ToolDelta` + `relaxed_json.rs` | |
| Thinking normalization | `ThinkingLevel` 7 levels, `thinkingLevelMap`, `thinkingBudgets`, 11 `thinkingFormat` wire variants | Partial: `ReasoningEffort`/`ReasoningConfig` (`ti:.../model/types.rs:77,106`) | |
| Prompt caching | `cacheRetention` none/short/long, `sessionId` → `prompt_cache_key`/affinity headers, `cache_control` markers | Yes, different: `cache_segments: Vec<PromptSegment>`, `CachePolicy`, `explicit_cache_control`, `PromptCacheGuardMiddleware`, `CacheLayoutEvent` (`ta:.../cache/types.rs:241`) | TA is more explicit about layout; pi adds routing affinity. |
| Cross-provider handoff | `transformMessages` (`pi:packages/ai/src/api/transform-messages.ts:64`) | Partial: thinking dropped on OpenAI path (doc on `ContentBlock::Thinking`); no tool-id normalization/image downgrade found | See gap 6. |
| Deferred/background responses | `DeferredHandle`, `streamDeferred`, `stopReason:"deferred"` (`types.ts:462`) | No (grep `deferred`: only Claude Code auth) | |
| OAuth / subscription auth | 7 flows (`pi:packages/ai/src/auth/oauth/`), `CredentialStore`, `Models.login/logout/checkAuth/getAvailable` | Partial: OpenAI Codex `OAuthFlow` (`ti:crates/tinyinference-providers/src/oauth.rs:299`), Claude Code auth | |
| Retry classification | regex tables (`pi:packages/ai/src/utils/retry.ts`) + `RetryPolicy` | Yes: `retry/`, `limits/` | |
| Model fallback chain | No (only Anthropic server-side `allowedFallbackModels`; startup `modelFallbackMessage`) | Yes: fallback policy (`ta:docs/modules/harness/limits-retry.md`, `model_registry`) | TA ahead. |
| Structured output | No `response_format` anywhere in pi-ai | Yes: `ResponseFormat`, `structured/` with repair | TA ahead. |
| Agent loop | `agentLoop` (`agent-loop.ts:162`) | Yes: `agent_loop/run_loop.rs` | |
| Tool execution mode | global + per-tool `executionMode` (`types.ts:430`) | Partial: concurrent only when ≥2 calls and no tool-wrap middleware (`ta:.../agent_loop/tools.rs:6-20`) | No per-tool override in TA. |
| Steering / follow-up queues | `steer()`/`followUp()`, `QueueMode` one-at-a-time/all, polled after each turn (`agent.ts:140,298`; `agent-loop.ts:173-272`) | Partial: `SteeringCommand` (`ta:.../steering/types.rs:32`) drained before each model call; `RunQueue` lanes exist (`ta:.../run_queue/`) but no consumer outside `lib.rs` | See gap 4. |
| Tool hooks / terminate | `beforeToolCall`/`afterToolCall`, `terminate` hint, `shouldStopAfterTurn`, `prepareNextTurn` | Partial: middleware `before_tool/after_tool/wrap` (`ta:.../middleware/types.rs:155-244`); no terminate/stop-after-turn hook (sdk-gaps §13) | |
| `transformContext` / `convertToLlm` split | Yes | Partial: `before_model` middleware mutates request; no app-message vs LLM-message layer | |
| Abort | `AbortSignal`, `stopReason:"aborted"`, `continue()` | Yes: `CancellationToken`, `AbortOnDrop` (`ti:.../model/types.rs:656`) | |
| Conversation tree / branching | JSONL v3 tree with `id/parentId`, labels, `/fork`, `/tree` (`pi:packages/coding-agent/src/core/session-manager.ts:53`); `AgentHarness` entry tree + Branches + `ForkOptions` | No for conversations: `tinyagents-session` is linear JSONL + SQLite history; only graph checkpoints fork (`ta:crates/tinyagents-graph/src/checkpoint/types.rs:24-35`) | See gap 3. |
| Compaction | cut point / `keepRecentTokens` / split turns / iterative summary / `CompactionEntry{firstKeptEntryId,tokensBefore}`; branch summaries | Partial: `SummarizationPolicy`, `Summarizer` trait, trim strategies, `ContextCompressionMiddleware`, `MicrocompactMiddleware` (`ta:.../summarization/`, `middleware/library/context.rs`) | TA has no durable compaction record or turn-boundary rules; no overflow detector (grep `overflow` empty). |
| Durable per-step run state | `AgentHarness` intent/settlement transactions, `replay: "never"\|"safe"` (`types.ts:422`) | Partial: graph checkpoints + pending writes; harness loop itself not durable per tool call | See gap 7. |
| Session backends + conformance | memory, JSONL, SQLite, `session/testing/conformance` | Partial: SQLite session DB + JSONL transcript; no backend conformance suite (sdk-gaps §17) | |
| Extension API / event bus | 40+ product events, `registerTool/Command/Provider`, UI | Partial: middleware + registry; no plugin loader (host concern) | |
| Skills / prompt templates | Yes (`Skill`, `PromptTemplate` types in agent-core `harness/types.ts`; loaders in coding-agent) | No skills (grep empty); `prompt/` module exists | |
| Sub-agents | Example extension only (`examples/extensions/subagent/`) | Yes: `SubAgent`, `SubAgentTool`, parallel policies | TA ahead. |
| Graph/workflow runtime | None | Yes (`tinyagents-graph`) | TA ahead. |
| Per-tool timeouts / limits / budgets | None (signal only) | Yes: `ToolTimeout`, `RunLimits`, budget middleware | TA ahead. |
| Telemetry | typed schema spans (`pi-telemetry`) | Yes: `AgentEvent` + Langfuse exporter | |
| Remote session protocol | CBOR `pi-protocol` | No (host concern) | |
| Image generation | `generateImages` | No | Out of scope for TA. |

## 4. Features TinyAgents lacks, ranked by value

### 4.1 Transcript-carried system prompt and tool-loadout patches

What: the system prompt and tool declarations live *in the transcript* as system messages; later system
messages patch it, so replaying the transcript yields the current prompt and tools, and dynamic tool
loading does not invalidate the KV cache.

```ts
// pi:packages/ai/src/types.ts:484
export interface SystemMessage {
  role: "system";
  content: string | TextContent[];
  sections?: Record<string, string | null>;   // named sections; null removes
  toolsAdded?: Tool[];
  toolsRemoved?: ToolReference[];
  timestamp: number;
}
```

Before every request `declareToolChanges(context, pending)` (`agent-loop.ts:291`) diffs the executable
tool set against what the transcript declares and emits one system message carrying the delta. Providers
whose `compat.supportsMidConvoSystemMessages` is true send it in place (cache prefix intact); others fold
into the leading system message (one cache miss per change). Compaction entries checkpoint the replayed
prompt (`CompactionEntry.systemMessage`). Why it matters: it is the only design I have seen that makes
"tools changed mid-run" a first-class, cache-aware, persisted fact rather than a request-time rebuild.

Mapping: TinyAgents already has `PromptSegment{role}` for cache layout and `ToolsFiltered` events
(`ta:.../events/types.rs`). Add `SystemMessage{sections, tools_added, tools_removed}` to
`ti:.../message/types.rs`, a `replay_system_state(&[Message]) -> (prompt, tools)` helper, a
`declare_tool_changes` step in `agent_loop/run_loop.rs` before `ModelStarted`, and a
`ModelProfile.mid_conversation_system_messages` flag to pick "in place" vs "fold".

### 4.2 Content-indexed streaming blocks and a compact frame codec

What: `AssistantMessageEvent` (`types.ts:645`) has `text_start/delta/end`, `thinking_start/delta/end`,
`toolcall_start/delta/end`, each with `contentIndex` and a shared `partial: AssistantMessage`, terminated
by `done{reason}` or `error{reason: "aborted"|"error", error: AssistantMessage}`. On top of it,
`AssistantMessageFrameEncoder` (`assistant-message-frame.ts:139`) turns events into small self-describing
frames (including `toolcall_checkpoint{json}` catch-up frames) that are appended to durable storage without
awaiting, and `reduceAssistantMessageFrames` rebuilds the partial message after a crash or for a
reconnecting client (`harness.md` §3.7).

Why: TinyAgents' `MessageDelta{text, reasoning, tool_call}` cannot express "which text block", block
boundaries, or interleaved thinking/text (Anthropic, Gemini); UI consumers and journals cannot reconstruct
the exact assistant message from deltas alone. sdk-gaps §3 already asks for this.

Mapping: extend `ModelStreamItem` (`ti:.../model/types.rs:632`) with `BlockStart{index, kind}`,
`BlockDelta{index, delta}`, `BlockEnd{index, block: ContentBlock}`; carry `content_index` on
`ToolDelta`; add a `frame.rs` encoder/reducer in `tinyagents-harness/src/stream/` and persist frames in
`HarnessEventJournal`. The error terminal should carry a full partial `AssistantMessage` with
`stop_reason`, as pi does, rather than `Failed(String)`.

### 4.3 Conversation entry tree with branches, labels, forks, and branch summaries

What (product form, `session-manager.ts:53-110`): every session entry has `id/parentId`; branching is
in-place; `LabelEntry` bookmarks; `/fork` and `/clone` create child files with `parentSession`;
`BranchSummaryEntry{fromId, summary}` is written at the navigation point summarizing the abandoned path.
What (runtime form, `harness.md` Part 2): a write-once `Entry` tree, `Branch` = named movable tip,
`AgentLane` = Branch + model config + queues + at most one operation; `ForkOptions` with `scope:
"branch"|"tree"`, `position: "before"|"at"`; context projection reads newest-first until the newest
`CompactionEntry` and never past it.

Why: TinyAgents' session crate is queryable history (`ta:crates/tinyagents-session/src/lib.rs` says
"nothing resumes from it"); the transcript JSONL is linear (`transcript/types.rs` has no parent id). Time
travel exists only at the graph-checkpoint level. Branch/retry-from-here and "what did we abandon" are
common desktop-assistant needs.

Mapping: give `TranscriptMessage` an `id`/`parent_id`, add `CompactionEntry`/`BranchSummaryEntry`/
`LabelEntry`/`CustomEntry` variants to the transcript, and a `build_context(tip) -> Vec<Message>` that stops
at the newest compaction. Keep SQLite as index (pi's SQLite backend keeps a rebuildable `branch_entries`
segment cache, §2.6). Fork = copy path + tip; ledger/usage not copied.

### 4.4 Steering and follow-up as *queued messages* with explicit queue modes

What: `Agent.steer(msg)` and `Agent.followUp(msg)` (`agent.ts:298-303`) enqueue `AgentMessage`s into two
`PendingMessageQueue`s (`agent.ts:140`); `drain()` returns all or only the first depending on
`QueueMode`. The loop (`agent-loop.ts:173-272`) polls steering *after each completed turn* (tool results
already appended), and follow-ups only when there are no tool calls and no steering. Steering never
interrupts an in-flight tool batch; abort does. `shouldStopAfterTurn` and `prepareNextTurn` (model/thinking
swap, compaction) sit at the same boundary.

Why: TinyAgents has the pieces but not the composition: `SteeringCommand::InjectMessage/Redirect` are
applied at the pre-model checkpoint (`run_loop.rs:214-232`), while `RunQueue{Steer,Followup,Collect}`
(`ta:.../run_queue/`) has no consumer in the loop (grep across crates finds only the `lib.rs` re-export).
There is no "run another turn after the agent would stop" path, and no "one-at-a-time" semantics.

Mapping: wire `RunQueue` into `run_loop.rs`: drain `Steer` at the turn boundary (after tool results),
drain `Followup` when the loop is about to return, honor a `QueueMode` on the harness. Add a
`terminate` flag to `ToolResult` and a `should_stop_after_turn` middleware hook (sdk-gaps §13).

### 4.5 Compaction as a durable, rule-driven operation with overflow recovery

What: `shouldCompact(contextTokens, contextWindow, {reserveTokens})` (`compaction.ts:246`);
`findCutPoint` (`:370`) walks newest-first accumulating `keepRecentTokens` and only cuts at user/assistant/
custom messages, never at tool results; a turn larger than the budget becomes a "split turn" with two
summaries merged; the summary prompt receives the previous summary for iterative refinement; the result is a
`CompactionEntry{summary, firstKeptEntryId, tokensBefore, usage, details, fromHook}` and context is
rebuilt from it. `isContextOverflow(message, contextWindow)` (`overflow.ts:135`) classifies provider
overflow errors so the driver can compact and retry the same turn. In `AgentHarness` compaction is an
operation with `before_compaction{reason: manual|threshold|overflow}` that can decline or supply a summary.

Why: TinyAgents' `summarization/` has trimming policies, tool-call pairing, and a `Summarizer` trait, but
no cut-point rules, no durable record with provenance beyond `SummaryRecord`, and no overflow detection.

Mapping: add `find_cut_point`, split-turn handling, and an `OverflowClassifier` to `summarization/`;
persist a `CompactionRecord` in the transcript (4.3); route `overflow → compact → retry` through
`ContextCompressionMiddleware`.

### 4.6 Cross-provider handoff transform

What: `transformMessages(messages, model, normalizeToolCallId?)` (`transform-messages.ts:64`): for
assistant messages from a *different* `{provider, api, model}`, drop redacted thinking, convert signed
thinking to plain text (or drop empty), strip `thoughtSignature`; normalize tool-call ids (OpenAI Responses
ids exceed Anthropic's 64-char `^[a-zA-Z0-9_-]+$`); replace images with a placeholder when
`model.input` lacks `"image"`. Called by every API implementation, so mid-session model switches work.

Mapping: TinyAgents stores `AssistantMessage.id` but not the originating provider/model on the message.
Add `origin: Option<ResolvedModel>` to `AssistantMessage` (`ti:.../message/types.rs:81`) and a
`prepare_for_model(&[Message], &ModelProfile)` pass in `agent_loop/model_call.rs`.

### 4.7 Durable operation state machine for the harness loop

What (`harness.md` Parts 3–4): every provider request and tool call is bracketed by two commits — intent
(reserve ids, `effect_pending`) and settlement — with a total `pi.op.state` rewritten after every
transition; `AgentTool.replay: "never" | "safe"` (`types.ts:422`) decides whether an orphaned
`effect_pending` call is re-executed or synthesized as an interrupted error result; parallel tool outcomes
settle in completion order but materialize in source order; queued input is staged as `pi.pending.entry`
before placement. Hooks are classified pass-local / request-local / transition-consumed.

Why: TinyAgents' graph has checkpoints and pending writes, but the harness loop (the thing most hosts run)
is not resumable mid-batch; `append_interrupted_partial` (`ta:.../transcript/writer.rs:175`) is the only
crash artefact. sdk-gaps §1 asks for `idempotency`/`retry` tool metadata; pi's `replay` is the minimal
version of that.

Mapping: add `ToolReplay::{Never, Safe}` to `ToolSchema`/policy; model the loop as a `tinyagents-graph`
graph with per-tool-call task checkpoints (the graph already has task outcomes and pending writes), rather
than porting pi's bespoke state machine.

### 4.8 Model catalog breadth, compat matrix, thinking-level map, auth resolution

What: per-provider generated `*.models.ts` from models.dev with `cost.tiers`, `thinkingLevelMap`,
`compat` (e.g. `OpenAICompletionsCompat` has ~30 flags, `types.ts:667`), `Provider.refreshModels()` for
dynamic lists, `Models.getAvailable()` filtered by resolved auth, `login(providerId, type, interaction)`.

Mapping: TinyAgents' `ModelCatalogEntry` (`ta:crates/tinyagents-registry/src/catalog.rs:149`) is the right
shape; it needs a generator (models.dev) and a `compat` field feeding `OpenAiConfig`. Thinking-level map
→ `ReasoningConfig`. Auth: generalize `tinyinference-providers::oauth::OAuthFlow` beyond Codex and add a
`CredentialStore` trait (host-implemented).

### 4.9 Deferred responses, custom transcript roles, skills/templates types

- Deferred: `DeferredHandle` + `streamDeferred/cancelDeferred` and `stopReason: "deferred"`; useful for
  batch pricing and long jobs. Map to `ModelStreamItem::Deferred(handle)` + `ChatModel::fetch_deferred`.
- Custom roles: `CustomAgentMessages` declaration merging + `convertToLlm`; TinyAgents' closest analogue is
  `ToolMessage.artifact` / `ContentBlock::ProviderExtension`. A `Message::Custom{kind, payload, display}`
  variant filtered at request-build time would cover bash-execution and notification entries.
- Skills/templates: the runtime-level part is only the `Skill`/`PromptTemplate` types and the XML
  rendering in `pi:packages/agent/src/harness/skills.ts`/`system-prompt.ts`; discovery is product code.

## 5. Design lessons — where pi is better or worse

**Streaming.** pi's model is better for UIs and durability: block-indexed start/delta/end events, a
`partial` snapshot on every event, and error terminals that are themselves `AssistantMessage`s with
`stopReason` and `errorMessage`, so an aborted or failed turn is persisted like any other. TinyAgents'
three-channel `MessageDelta` is simpler but lossy. pi's weakness: `partial` is a shared mutable object
("not an event-time snapshot"), a footgun for async consumers; TinyAgents' owned values avoid that.

**Steering / queues.** pi's is the more honest contract: steering is a *message*, applied at a turn
boundary, never mid-tool; follow-ups are a separate lane with "run once more" semantics; queue modes are
explicit. TinyAgents' `SteeringCommand` is stronger as a *control* channel (Pause/Resume/Cancel with a
policy allowlist and provenance) — pi has nothing equivalent to policy-checked steering or pause-latching.
Best of both: keep `SteeringCommand` for control, adopt pi's two message lanes for content.

**Session tree.** pi's in-place tree (`id/parentId`, labels, compaction and branch-summary entries as tree
nodes, forks as path copies) is a clean, replay-only design, and the `AgentHarness` refinement (Branch vs
AgentLane, "context never reads past a compaction", append-only-context invariant for KV cache) is
sharper than anything in TinyAgents' session crate. Worse: two coexisting designs (coding-agent v3 JSONL
vs harness entry tree, plus pico v5 in `pi-durable`) and an admitted O(history) copy on first divergence
of an uncompacted SQLite branch (§2.6). TinyAgents' graph checkpointer already has fork/time-travel; the
gap is applying the same idea to conversation history.

**Compaction.** pi's is more complete operationally (cut-point rules, split turns, iterative summaries,
overflow-triggered compaction with retry, hook can decline/replace, usage attributed). TinyAgents is
better factored (policy/trim/pairing/summarizer as separate types, middleware-composable) but lacks the
rules and the durable record.

**Extension API.** pi's `ExtensionAPI` is product-level: excellent for a CLI (40+ events, UI hooks,
`registerProvider`), but it hard-couples to the TUI and to process-global state. The runtime-level
equivalent, `AgentHarness.hooks` (`harness.md` §5.6), is well thought out: each hook is classified by
durability (pass-local / request-local / transition-consumed), fail-closed hooks are named
(`before_drive`, `before_tool`), and aggregation rules are spelled out. TinyAgents' `Middleware` trait
(`ta:.../middleware/types.rs`) has similar hook points plus `wrap` and typed control outcomes, but no
durability classification — worth borrowing as documentation even before behavior changes.

**Provider abstraction.** pi: `Provider{getModels, stream, streamSimple, auth}` over API modules keyed by
string `Api`; a normalized branded `TranscriptContext` guarantees system/tools live in the transcript;
compat flags are data on the model, generated per model id. It is pragmatic and very broad, but the
compat surface is a sprawling flag bag (auto-detected from URL by default), and there is no structured
output, no capability *requirements*, no fallback chain, no rate limiting, no per-tool timeout.
TinyAgents' `ChatModel<State, Ctx>` + `ModelProfile`/`CapabilitySet` + `ModelRequest` (with
`required_capabilities`, `cache_segments`, `reasoning`, `continuation_id`) is the stronger contract; what it
lacks is breadth of adapters and generated model data. pi's `onPayload`/`onResponse` request callbacks
and `ProviderRequestOptions.fetch` injection are cheap, useful escape hatches TinyAgents does not expose.

**General.** pi optimizes for one product and moves fast (100 commits/week, version-locked monorepo);
its runtime layers are extracted from that product and carry two generations of design at once.
TinyAgents is spec-first and layered (graph/harness/registry/session/orchestration with enforced
dependency direction). pi has no graph runtime, no sub-agent primitives, no budgets, no structured
output, and no permission model; TinyAgents has all of those and should not import pi's product coupling.

## 6. Runtime-level vs harness-level split

| Gap | Runtime library (TinyAgents) | Product harness (OpenHuman) |
|---|---|---|
| 4.1 System/tool patch messages | Yes: message type, replay helper, loop step, profile flag | Decides *which* tools to add/remove (`agent/tool_policy.rs`, `agent/registry/`) |
| 4.2 Block-indexed stream + frames | Yes: `ModelStreamItem`, frame codec, journal persistence | Renders; reconnect UI |
| 4.3 Conversation tree/forks | Yes: transcript entries with parent ids, context projection, fork | `/fork`, `/tree` commands, labels UI, session import (`agent/session_import/`) |
| 4.4 Steer/follow-up lanes | Yes: wire `RunQueue` into loop, queue modes, terminate hint | Which channel/message becomes a steer vs follow-up (`agent/harness/run_queue/` already exists in OpenHuman, so this moves down) |
| 4.5 Compaction rules + overflow | Yes: cut points, overflow classifier, durable record | Summary prompt wording, file-tracking `details`, user-facing `/compact` |
| 4.6 Handoff transform | Yes (pure function over messages + profile) | — |
| 4.7 Durable loop state | Yes, via graph-backed loop and `ToolReplay` metadata | Approval gate and sandbox decisions stay in `security/approval`, `security/bubblewrap.rs` |
| 4.8 Catalog/compat/auth | Catalog generator, compat data, `CredentialStore` trait, generic OAuth flow: runtime. Actual key storage, keychain, login UI: host | `security/credentials/`, model picker |
| 4.9 Deferred, custom roles, skill types | Runtime types; deferred stream item | Skill/template discovery from disk, prompt assembly (`agent/prompts/`) |
| Extension loader, commands, UI, packages | No | Yes (OpenHuman RPC/bus, `agent/registry`) |
| Permission/sandbox | Policy metadata only (sdk-gaps §1) | Yes; pi deliberately has none, OpenHuman already does |

## 7. Sources

- pi repo: https://github.com/earendil-works/pi (commit `36b60d2e`)
- `pi:README.md`, `pi:packages/ai/README.md`, `pi:packages/agent/README.md`, `pi:packages/durable/README.md`,
  `pi:packages/chord/README.md`
- `pi:packages/ai/src/types.ts`, `src/models.ts`, `src/index.ts`, `src/api/transform-messages.ts`,
  `src/utils/{event-stream,assistant-message-frame,overflow,retry}.ts`, `src/auth/oauth/`, `src/providers/`
- `pi:packages/agent/src/{types,agent,agent-loop,index}.ts`, `src/harness/**`, `docs/harness.md`,
  `docs/post-wp05-roadmap.md`, `docs/pico-v3.md`, `docs/plugins.md`
- `pi:packages/coding-agent/docs/{session-format,compaction,extensions,skills,prompt-templates,packages,sdk,models}.md`,
  `src/core/{session-manager,agent-session}.ts`, `src/experimental/`
- GitHub API: repo metadata, releases, commits, contributors (fetched 2026-09-19)
- TinyAgents: `ta:docs/spec/README.md`, `ta:docs/modules/harness/README.md`, `ta:docs/modules/graph/README.md`,
  `ta:docs/sdk-gaps.md`, `ta:ROADMAP.md`, `ta:crates/*/src/lib.rs`,
  `ta:crates/tinyagents-harness/src/{agent_loop,steering,run_queue,summarization,middleware,events,stream,cache,cost}/`,
  `ta:crates/tinyagents-session/src/transcript/`, `ta:crates/tinyagents-registry/src/catalog.rs`,
  `ta:crates/tinyagents-graph/src/checkpoint/types.rs`
- tinyinference (main checkout): `ti:crates/tinyinference-llm/src/{message,model,catalog,usage}/types.rs`,
  `ti:crates/tinyinference-llm/src/providers/`, `ti:crates/tinyinference-providers/src/oauth.rs`
- OpenHuman calibration: `openhuman/crates/openhuman-core/src/agent/README.md`,
  `src/agent/harness/`, `src/security/`
