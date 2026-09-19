# Agent Runtime Comparison

Date: 2026-09-19. Baseline: TinyAgents v2.1.2 (`fc33c43`).

This directory compares TinyAgents with three agent *runtime libraries* and
pairs that with a deep code review of our own crates. It answers three
questions:

1. Which features do LangGraph/LangChain, Pydantic AI, and pi have that we do
   not, and which of those belong in TinyAgents (runtime) versus OpenHuman
   (the product harness on top)?
2. Where is our own code wrong, slow, or drifting from its docs?
3. In what order should we act?

Harnesses and products (Claude Code, Codex CLI, Deep Agents' file tools,
pi's coding-agent CLI, OpenHuman itself) are deliberately out of scope except
as calibration for the runtime/harness split.

## Files

| File | What it holds |
|---|---|
| [`feature-gaps.md`](feature-gaps.md) | Consolidated gap matrix across all three runtimes with a runtime-vs-host layer assignment for every item. Start here. |
| [`plan.md`](plan.md) | Phased execution plan: PR-sized work items, dependencies, and what moves between OpenHuman and TinyAgents. |
| [`langgraph.md`](langgraph.md) | LangGraph 1.2 / LangChain 1.4 / Deep Agents 0.7: feature inventory, ranked gaps, design lessons. |
| [`pydantic-ai.md`](pydantic-ai.md) | Pydantic AI v2 / pydantic-graph / pydantic-ai-harness: same structure. |
| [`pi.md`](pi.md) | pi (`earendil-works/pi`): `pi-ai`, `pi-agent-core` and its `AgentHarness`: same structure. |
| [`code-review-harness.md`](code-review-harness.md) | `tinyagents-harness` findings (3 critical, 13 important, 14 minor), refactors, test gaps. |
| [`code-review-graph.md`](code-review-graph.md) | `tinyagents-graph` / `-orchestration` / `-session` findings (4 critical, 12 important, 12 minor). |
| [`code-review-workspace.md`](code-review-workspace.md) | Registry, language, definition, tracing, integration tests, CI and workspace hygiene. |

## Executive summary

**Where TinyAgents is ahead.** None of the three runtimes has our combination
of a durable typed graph (channels, `Send`, subgraphs, checkpoints with time
travel), a policy-checked steering channel, detached sub-agent registry with
parallel failure policies, fail-closed `ToolPolicy`, prompt-cache segment
layout, capability-set model resolution, and a declarative `.rag` language.
LangGraph OSS puts double-texting, cron and background runs in its paid
server; Pydantic AI refuses to own a checkpointer; pi has no graph, no
sub-agents, no structured output, no budgets.

**Where TinyAgents is behind.** The gaps cluster into seven themes
(detail and ranking in [`feature-gaps.md`](feature-gaps.md)):

1. **Loop control and human-in-the-loop.** Middleware hooks return
   `Result<()>`, so limits, HITL, early-exit tools and re-routing cannot be
   composed (LangChain `jump_to`/`Command`). There is no typed, resumable
   deferred-tool handshake (Pydantic `DeferredToolRequests`/`Results`); the
   structured-output path is one-shot (`extract(&response)?`) where Pydantic
   re-asks on `ModelRetry`; `RunQueue` steer/follow-up lanes exist but nothing
   in the loop consumes them (pi polls both at turn boundaries).
2. **Streaming.** `MessageDelta{text, reasoning, tool_call}` has no block
   boundaries or indices, so interleaved thinking/text and durable partials
   cannot be reconstructed (pi's `*_start/_delta/_end` + frame codec; LangGraph
   `StreamPart{ns, seq}`).
3. **Durability semantics.** The graph executor discards completed
   higher-index parallel siblings on interrupt/failure and re-runs them, and
   routes their successors into the wrong superstep (critical findings C1/C2).
   Checkpoints are unversioned full-state snapshots; per-node
   retry/cache/timeout policies and delta channel history are missing.
4. **Tools.** `ToolExecutionContext` lacks call id, store and state access;
   tool results are text/JSON only (no `ToolReturn`-style multimodal
   follow-up + metadata); no composable toolsets; no generic MCP client; no
   replay/idempotency classification for crash recovery.
5. **Sessions and context.** Transcripts are linear (pi's entry tree with
   forks, labels, compaction and branch-summary entries); compaction has
   policies but no cut-point rules, overflow detection, or durable record;
   no cross-provider handoff transform.
6. **Models.** `ModelProfile` is capability data only (Pydantic's carries
   schema transformers, thinking-tag parsing, default output mode); catalog
   seed is five stale models (pi generates 41 providers from models.dev with
   tiered pricing); message model has images only.
7. **Testing.** No evals crate, no schema-driven test model, no network
   kill-switch, and live tests silently pass without keys.

**Where our code is wrong.** Beyond the graph durability bugs above: an
unchecked `Arc<dyn Any>` pointer cast in the hosted path is reachable UB
through the public `RunContext::child`; the streaming path drops Anthropic
thinking signatures so streaming + extended thinking + tools fails on the
second call; the concurrent tool path leaks `ToolStarted` without a terminal
event on first failure; a per-call timeout aborts the run instead of falling
back; text-dialect tool-call recovery runs unconditionally on final answers;
blocking SQLite/fs I/O runs inside `async fn`; CI never runs the 637
integration tests with `-D warnings` or without `--all-features`;
`build_graph` ignores ~70 % of a `.rag` blueprint; the language crate pulls
the HTTP stack for one error type.

## The layering rule used throughout

A feature belongs in **TinyAgents** when it is loop control vocabulary,
a message/stream/checkpoint format, a provider contract, or a trait the host
cannot add from outside `run_loop.rs` / `executor.rs`. It belongs in
**OpenHuman** when it is policy content (which tools to expose, prompt
wording, approval UI, credential storage), a product surface (commands,
extensions, RPC, UI adapters), or an integration with the desktop (sandboxes,
filesystem/shell tools, cron, skills and memory files). All three reference
runtimes draw the same line: LangChain keeps Deep Agents' file tools and
sandboxes out of `langgraph`; Pydantic ships `Coder`/`Shell`/`FileSystem` in a
separate harness package; pi ships no permission system at all.

Several things OpenHuman currently implements around the SDK should move
down once the runtime grows the primitive: the approval gate becomes a
deferred-tool handler, `agent/harness/run_queue` becomes the wired
`RunQueue`, `agent/tool_policy.rs` filtering becomes a `ToolSet` predicate,
and `agent/multimodal.rs` marker hacks disappear with real audio/document
blocks. [`plan.md`](plan.md) lists each hand-off.

## Method

Six independent reviews were run against primary sources (official docs,
release lists, and source checkouts of LangGraph, LangChain, deepagents,
pydantic-ai, and pi) and against this repository's source, not its docs.
Every "TinyAgents lacks X" claim was verified by grep before being recorded;
every code finding carries a `file:line` and a failure scenario. Existing
backlog documents (`docs/sdk-gaps.md`, `docs/audit.md`) were read first so
findings are not duplicated, and their "resolved"/"missing" markers were
re-checked (several are stale; see
[`code-review-harness.md`](code-review-harness.md) §1).
