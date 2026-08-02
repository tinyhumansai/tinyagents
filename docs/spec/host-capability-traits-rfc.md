# RFC: Host capability traits for a relocated agent runtime

**Status:** draft, for upstream acceptance · **Date:** 2026-08-02
**Tracking:** OpenHuman `docs/specs/plan-agents.md` Phase 0 — *"Nothing moves
until the trait catalogue is accepted upstream, otherwise the first mover
defines the seams by accident."*

---

## 1. What this RFC is for

OpenHuman intends to relocate its agent runtime (`src/openhuman/agent/`, 70,530
LOC) into this crate so that OpenHuman becomes a kernel/library and TinyAgents
owns the generic agent loop. The runtime cannot move as written: it reaches **45
OpenHuman domains**, several of which (Composio, `SecurityPolicy`,
`memory_store`) must never be dependencies of a redistributed crate.

The enabling fact is that this crate is **already generic over a host-supplied
state type** and already ships **18 extension traits** on that pattern
(`ChatModel`, `Tool`, `ChatHistory`, `Store`, `AppendStore`, `Summarizer`,
`EmbeddingModel`, `VectorStore`, `ResponseCache`, `WorkspaceIsolation`,
`HarnessEventJournal`, `HarnessStatusStore`, `EventListener`, `Middleware`,
`ModelMiddleware`, `ToolMiddleware`, `ModelBaseCall`, `ToolBaseCall`):

```rust
pub struct AgentHarness<State: Send + Sync, Ctx: Send + Sync = ()> { … }
pub trait Tool<State: Send + Sync>: Send + Sync { … }
pub trait ChatModel<State: Send + Sync>: Send + Sync { … }
```

This RFC proposes **10 more traits of the same kind**. It is not a new
architecture; it is more of the one already in use. Accepting it fixes the seam
boundaries *before* any code moves.

**This RFC proposes signatures only.** Phase 1 lands them with no-op / in-memory
default impls and no host change.

---

## 2. Design rules

These follow the rules the crate and the OpenHuman kernel spec already apply.

1. **Traits carry behaviour, not product types.** No trait here may name a
   Composio type, an OpenHuman config struct, a `SecurityPolicy`, or an RPC
   schema.
2. **Value types are inert.** Serde/std only, defined in a dependency-free
   module, so a host can implement the trait without pulling an engine.
   (Same carve-out that lets OpenHuman gate `skills` and `mcp` at compile time
   while keeping their type modules always-compiled.)
3. **Optional capabilities are `Option<Arc<dyn …>>` on the builder, not
   always-present traits with erroring defaults.** Absence must be
   distinguishable from failure — an erroring stub teaches a model the
   capability exists and makes it retry.
4. **Ten is the budget.** If the host implementation phase needs an eleventh,
   that is a signal a seam is wrong; re-open this RFC rather than appending.
5. **No host-side policy may be expressible only inside a driver.** Anything a
   host must enforce (redaction, taint, budget) is a *gate* the runtime calls,
   never something the runtime is trusted to have done.

---

## 3. The catalogue

Reference counts are outbound references from `src/openhuman/agent/` measured on
OpenHuman `main` at 2026-07-28, and are the evidence that each seam is real
rather than speculative.

| # | Trait | Replaces (refs) | Optional? |
|---|-------|-----------------|-----------|
| 1 | `MemoryProvider` | `memory` 57, `memory_store` 41, `memory_tree` 16, `agent_memory` 11, `memory_tools` 5, `memory_conversations` 3 | yes |
| 2 | `ContextComposer` | `context` 52, `thread_goals` 5 | no |
| 3 | `DefinitionRegistry` | `profiles` 34, `agent_registry` 22 | no |
| 4 | `SecurityGate` | `security` 25, `approval` 10, `agent_tool_policy` 6, `sandbox` 4, `prompt_injection` 2 | no |
| 5 | `BudgetGate` | `tokenjuice` 19, `cost` 7, `scheduler_gate` 4 | yes |
| 6 | `ProgressSink` | `web_chat` 6, `channels` 4 | yes |
| 7 | `LearningSink` | `learning` 11, `subconscious` 8 | yes |
| 8 | `ToolOutcomeClassifier` | `tool_status` 5 | yes |
| 9 | `ExperienceStore` | `agent_experience` 3 | yes |
| 10 | `ModelResolver` | `inference` 72 (with the existing `ChatModel`) | no |

### 3.1 `MemoryProvider`

The largest seam (133 refs across six domains). Note the crate must **not**
learn what a memory is — recall returns opaque, already-redacted host values.

```rust
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Retrieves memory relevant to `query` for injection into a turn's
    /// context. Returned items are already scope-filtered and redacted by the
    /// host; the runtime must not re-rank or re-filter them.
    async fn recall(&self, req: RecallRequest) -> Result<Vec<MemoryItem>>;

    /// Records a durable memory produced by a turn. The host owns
    /// provenance/taint stamping — the runtime never supplies it.
    async fn remember(&self, item: NewMemory) -> Result<MemoryId>;

    /// Rolled-up prior context for a thread (OpenHuman's summary tree), as
    /// opaque prose the runtime injects verbatim.
    async fn thread_summary(&self, thread: &ThreadId) -> Result<Option<String>>;
}
```

`MemoryItem` carries `id`, `text`, `score: Option<f32>`, and an opaque
`citation: Option<String>` — deliberately **not** OpenHuman's `MemoryCitation`,
whose UI shape stays host-side.

> **Naming collision, flagged deliberately.** `tinyflows` 0.5.1 shipped a
> *different* `MemoryProvider` (`recall`/`flavour`/`people`/`remember`/`forget`,
> `serde_json::Value`-typed, flow-scoped). The two are not interchangeable and
> both crates are in the same dependency graph for OpenHuman. Either rename
> this one (`AgentMemory`?) or accept that hosts will alias one at the import
> site. **This needs an upstream decision before Phase 1.**

### 3.2 `ContextComposer`

Owns system-prompt assembly. The crate supplies the turn's mechanical parts; the
host supplies identity, connected-integration blurbs, and learned context.

```rust
#[async_trait]
pub trait ContextComposer: Send + Sync {
    /// Builds the system prompt for a turn. The host injects identity/soul
    /// text, connected-integration descriptions, and learned-context blocks.
    async fn compose_system_prompt(&self, req: &TurnContextRequest) -> Result<String>;

    /// Extra messages to prepend after the system prompt (goals, pinned
    /// context). Empty is normal.
    async fn preamble(&self, req: &TurnContextRequest) -> Result<Vec<Message>>;
}
```

### 3.3 `DefinitionRegistry`

Resolves an agent id to its definition. Replaces direct reads of `profiles` and
`agent_registry::agents::{load_builtins, BUILTINS, validate_tier_hierarchy}`.

```rust
#[async_trait]
pub trait DefinitionRegistry: Send + Sync {
    async fn resolve(&self, id: &AgentId) -> Result<Option<AgentDefinition>>;
    async fn list(&self) -> Result<Vec<AgentDefinition>>;
    /// Subagent ids this agent may delegate to, already tier-checked.
    async fn delegates_for(&self, id: &AgentId) -> Result<Vec<AgentId>>;
}
```

`resolve` returning `Ok(None)` for an unknown id is **required, not an error** —
OpenHuman's orchestrator TOML legitimately lists subagents that are compiled out,
and both of its resolution sites tolerate that today.

### 3.4 `SecurityGate`

The crate must never decide whether an action is permitted; it asks.

```rust
#[async_trait]
pub trait SecurityGate: Send + Sync {
    /// Consulted before every tool invocation.
    async fn authorize_tool(&self, call: &ToolCallRequest) -> Result<GateDecision>;
    /// Consulted before untrusted text enters the model context.
    async fn screen_input(&self, text: &str, origin: ContentOrigin) -> Result<ScreenOutcome>;
}

pub enum GateDecision { Allow, Deny { reason: String }, Prompted { approved: bool } }
pub enum ScreenOutcome { Pass, Redacted(String), Block { reason: String } }
```

`Prompted` exists so the host's interactive approval gate (a 10-minute TTL park
in OpenHuman) stays entirely host-side; the runtime only sees the resolved
answer and never learns there was a human in the loop.

### 3.5 `BudgetGate`

```rust
#[async_trait]
pub trait BudgetGate: Send + Sync {
    /// Awaited before each model call — the host may block for capacity.
    /// The returned permit is released on drop.
    async fn acquire(&self, est: &CallEstimate) -> Result<Permit>;
    /// Reports realised usage after a call.
    async fn record(&self, usage: &Usage) -> Result<()>;
    /// Whether the runtime should compress context before the next call.
    fn compression_hint(&self, state: &ContextState) -> CompressionHint;
}
```

`acquire` is `async` and returns a permit specifically to preserve
`scheduler_gate::wait_for_capacity` / `LlmPermit` semantics, which are
back-pressure, not a yes/no check.

### 3.6 – 3.9 The small sinks

Each is optional; `None` means the runtime skips the call entirely.

```rust
#[async_trait]
pub trait ProgressSink: Send + Sync {
    /// Fire-and-forget turn progress. Never awaited on the critical path and
    /// never allowed to fail a turn.
    async fn emit(&self, ev: ProgressEvent);
}

#[async_trait]
pub trait LearningSink: Send + Sync {
    async fn on_turn_complete(&self, summary: &TurnSummary) -> Result<()>;
}

pub trait ToolOutcomeClassifier: Send + Sync {
    /// Whether a tool result counts as a failure worth retrying/surfacing.
    fn classify(&self, name: &str, result: &ToolResult) -> OutcomeClass;
}

#[async_trait]
pub trait ExperienceStore: Send + Sync {
    async fn record(&self, exp: &Experience) -> Result<()>;
    async fn recall_for(&self, agent: &AgentId, task: &str) -> Result<Vec<Experience>>;
}
```

`ProgressEvent` is deliberately coarse (`Started`, `ToolCall`, `Token`,
`Finished`, `Error`). OpenHuman's `AgentProgress` is a **UI contract** consumed
by its frontend timeline, cost footer, and citation chips; it stays host-side
and is produced *from* `ProgressEvent`. If it drifts into this crate the
frontend breaks in ways unit tests will not catch.

### 3.10 `ModelResolver`

`ChatModel` already exists. What is missing is choosing *which* model a given
turn should use — OpenHuman routes by agent role (including its `subconscious`
routing, which is product logic and must not be published).

```rust
#[async_trait]
pub trait ModelResolver<State: Send + Sync>: Send + Sync {
    async fn resolve(&self, req: &ModelRequest) -> Result<Arc<dyn ChatModel<State>>>;
}
```

---

## 4. What is deliberately *not* a trait

- **Config.** Per `plan-agents.md` §4.1 the crate defines its own
  `SessionConfig` / `TurnConfig` / `ToolConfig` structs and the host maps into
  them. A `ConfigProvider` trait with ~40 getters was considered and rejected:
  it turns every config read into a virtual call and becomes a dumping ground.
- **Composio and skills.** These are already expressible as `Tool<State>` impls.
  Adding a trait for them would be redundant.
- **`turn_origin`, `ChatMessage`, `AgentProgress`, prompts, triage.** Host
  product/UI types. They stay in OpenHuman's adapter layer.

---

## 5. Open questions for upstream

1. **`MemoryProvider` name collision with `tinyflows`** (§3.1). Blocking.
2. **Where do these live?** Proposal: one module per trait under
   `src/harness/host/`, re-exported from `harness::prelude`, matching the
   existing `src/harness/<area>/types.rs` convention.
3. **Do optional capabilities belong on `AgentHarness` or on a new
   `HostCapabilities` bundle?** tinyflows chose a bundle struct
   (`Capabilities`) with `Option` fields; this crate currently uses builder
   setters. Consistency across the two crates argues for a bundle.
4. **Contract versioning.** OpenHuman's kernel spec requires each contract to
   carry `CONTRACT_VERSION: (u16, u16)`. Should this crate adopt that, or is
   semver on the crate itself sufficient? (Semver is probably sufficient for an
   embedded driver, and insufficient for the out-of-process case.)

---

## 6. Acceptance criteria for Phase 0

- [ ] Trait signatures reviewed and the ten-trait budget accepted (or amended).
- [ ] §5.1 naming collision resolved.
- [ ] §5.2 module layout agreed.
- [ ] §5.3 bundle-vs-builder decided.
- [ ] Inert value-type module named and confirmed dependency-free.

Only then does Phase 1 (land the traits with default impls, no host change)
begin.
