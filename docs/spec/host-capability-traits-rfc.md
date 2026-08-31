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
| 1 | `AgentMemory` | `memory` 57, `memory_store` 41, `memory_tree` 16, `agent_memory` 11, `memory_tools` 5, `memory_conversations` 3 | yes |
| 2 | `ContextComposer` | `context` 52, `thread_goals` 5 | no |
| 3 | `DefinitionRegistry` | `profiles` 34, `agent_registry` 22 | no |
| 4 | `SecurityGate` | `security` 25, `approval` 10, `agent_tool_policy` 6, `sandbox` 4, `prompt_injection` 2 | no |
| 5 | `BudgetGate` | `tokenjuice` 19, `cost` 7, `scheduler_gate` 4 | yes |
| 6 | `ProgressSink` | `web_chat` 6, `channels` 4 | yes |
| 7 | `LearningSink` | `learning` 11, `subconscious` 8 | yes |
| 8 | `ToolOutcomeClassifier` | `tool_status` 5 | yes |
| 9 | `ExperienceStore` | `agent_experience` 3 | yes |
| 10 | `ModelResolver` | `inference` 72 (with the existing `ChatModel`) | no |

### 3.1 `AgentMemory`

The largest seam (133 refs across six domains). Note the crate must **not**
learn what a memory is — recall returns opaque, already-redacted host values.

```rust
#[async_trait]
pub trait AgentMemory: Send + Sync {
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

> **Why `AgentMemory` and not `MemoryProvider`.** `tinyflows` 0.5.1 shipped a
> *different* `MemoryProvider` (`recall`/`flavour`/`people`/`remember`/`forget`,
> `serde_json::Value`-typed, flow-scoped). The two are not interchangeable and
> both crates sit in OpenHuman's dependency graph, so sharing the name would
> force an import alias at every host call site. Resolved 2026-08-02: this trait
> is `AgentMemory`; `tinyflows` keeps `MemoryProvider`. The scoped name is also
> the more honest one — this is the agent runtime's view of memory, not a
> general memory abstraction.

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
    async fn resolve(&self, id: &str) -> Result<Option<AgentDefinition>>;
    async fn list(&self) -> Result<Vec<AgentDefinition>>;
    /// Subagent ids this agent may delegate to, already tier-checked.
    async fn delegates_for(&self, id: &str) -> Result<Vec<String>>;
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
    async fn recall_for(&self, agent_id: &str, task: &str) -> Result<Vec<Experience>>;
}
```

`ProgressEvent` is deliberately coarse (`Started`, `ToolCall`,
`ToolCallFinished`, `Token`, `Finished`, `Error`). OpenHuman's `AgentProgress`
is a **UI contract** consumed by its frontend timeline, cost footer, and
citation chips; it stays host-side and is produced *from* `ProgressEvent`. If it
drifts into this crate the frontend breaks in ways unit tests will not catch.

The test for admitting a variant is **whether only the runtime can know it**,
not a variant count. `ToolCallFinished` was added under that test (issue #88):
`ToolCall` opened a tool row and nothing closed it, so a host could not report a
tool's outcome truthfully — it would have had to leave every row running
forever or fabricate `success: true`, which puts wrong data in both the timeline
and the trace exporter. Whether a tool returned or failed is a runtime fact, in
the same category as `Started` and `Finished`. A chip or a footer is not.

### 3.10 `ModelResolver`

`ChatModel` already exists. What is missing is choosing *which* model a given
turn should use — OpenHuman routes by agent role (including its `subconscious`
routing, which is product logic and must not be published).

```rust
#[async_trait]
pub trait ModelResolver<State: Send + Sync>: Send + Sync {
    async fn resolve(&self, req: &ModelResolveRequest) -> Result<Arc<dyn ChatModel<State>>>;
}
```

`ModelResolveRequest` carries `agent_id`, `role`, `is_team_lead` and
`model_pin`. The last was added by issue #89: `AgentDefinition.model` is
documented as the model an agent pins, and there was previously no field for it
on the request, leaving `role` as the only string a wiring author could reach
for. A host resolver reasonably reads `role` as a role vocabulary, so a model id
placed there becomes an unrecognised role and falls back to a default — the pin
dropped silently. `role` is a **host taxonomy**; `model_pin` is a **concrete
model id**; they must stay separate channels.

The pin is **advisory**. The host decides whether it can honour it — the model
may be unconfigured, its credentials absent, or its provider down — because the
runtime has no view of any of that. A runtime that honoured pins itself would
route to models the host cannot call.

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

1. ~~**`MemoryProvider` name collision with `tinyflows`** (§3.1).~~ **Resolved
   2026-08-02** — this trait is `AgentMemory`; `tinyflows` keeps
   `MemoryProvider`. See the note in §3.1.
2. ~~**Where do these live?**~~ **Resolved 2026-08-02** — one module per trait
   under `crates/tinyagents-harness/src/host/`, re-exported from `host::*`.
   *Partially deferred:* each capability is currently a single flat file
   (trait + value types + default impl + inline tests) rather than the crate's
   `<area>/{mod,types,test}.rs` split, so its value types sit in a module that
   also imports `async_trait`. That does not yet satisfy §6's "inert value-type
   module, dependency-free". Deferred rather than done because the split is
   30 files of pure motion and buys nothing until a host actually needs to
   depend on the value types without the crate. **Do it before publishing.**
3. ~~**Bundle or builder?**~~ **Resolved 2026-08-02** — a
   `HostCapabilities<State>` bundle (`host/mod.rs`), matching
   `tinyflows::caps::Capabilities`. The four required capabilities are
   constructor arguments; the six optional ones are `Option<Arc<dyn …>>` set
   through `with_*`. Generic over `State` only because `ModelResolver` is.
   `Clone` is hand-written: deriving it would demand `State: Clone`, which is
   wrong when every field is an `Arc`.
4. **Contract versioning.** OpenHuman's kernel spec requires each contract to
   carry `CONTRACT_VERSION: (u16, u16)`. Should this crate adopt that, or is
   semver on the crate itself sufficient? (Semver is probably sufficient for an
   embedded driver, and insufficient for the out-of-process case.) **Still
   open** — nothing depends on it until an out-of-process driver exists.
5. ~~**No `AgentId` in the crate.**~~ **Resolved 2026-08-02** — agent identity
   is `&str` / `String` throughout, and §3.3 / §3.9 / §3.10 above have been
   amended to match. A newtype minted in these modules would not be the type a
   host's registry hands around, so it would add a conversion at every call
   site and buy no safety.
6. ~~**§3.10's `ModelRequest` collides with `harness::model::ModelRequest`.**~~
   **Resolved 2026-08-02** — the routing type is `ModelResolveRequest`; §3.10
   amended.

### Identity conventions (fixed during Phase 1)

Ten independently-written modules encoded agent and thread identity four
different ways — `agent` vs `agent_id`, `thread` vs `thread_id`, and one
`CallEstimate.agent: String` that re-encoded "absent" as `""`. Normalised to:

- **`agent_id`** and **`thread_id`** as the field names, everywhere.
- **`Option<…>` when the value can be absent** — never a sentinel empty string.
  A host cannot be expected to remember a per-field convention, and the one
  place that used a sentinel was in the same change set as a module that
  documented at length why sentinels are wrong.

---

## 6. Acceptance criteria for Phase 0

- [x] Trait signatures reviewed and the ten-trait budget accepted — all ten
      landed 2026-08-02, budget held at ten, no eleventh needed.
- [x] §5.1 naming collision resolved (`AgentMemory`, 2026-08-02).
- [x] §5.2 module layout agreed (`crates/tinyagents-harness/src/host/`); per-capability
      types/test split deferred — see §5.2.
- [x] §5.3 decided: `HostCapabilities<State>` bundle.
- [ ] Inert value-type module named and confirmed dependency-free — **not met**;
      see §5.2. Blocks publishing, not Phase 4.

Only then does Phase 1 (land the traits with default impls, no host change)
begin.
