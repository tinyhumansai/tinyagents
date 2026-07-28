# Host Capability Seams Specification

**Status:** accepted; implemented in `src/harness/host/` (`harness::host`).
**Extends** [`harness-spec.md`](harness-spec.md); implementation notes in
[`docs/modules/harness/host.md`](../modules/harness/host.md).

This document is the accepted catalogue of the ten host capability traits the
harness exposes, the reasoning behind each seam, and the boundaries that keep them
from growing. An implementer reading only this file should be able to tell what a
trait is for, what it is deliberately not for, and what would have to be true
before an eleventh is added.

## Motivation

The harness already separates *how a model is called* from *who supplies the
model*. `ChatModel`, `Tool`, `ChatHistory`, and `Store` all exist because a
runtime that hard-codes those decisions can only serve the application it was
extracted from.

The same argument applies one level up, to a set of decisions the harness
currently has no vocabulary for. A production agent runtime must at minimum
answer: what long-term knowledge is relevant to this input, beyond the thread's
own message list; what text precedes the model request and where it is spliced so
a cached prompt prefix survives; whether this tool, path, input, or concrete call
is permitted; whether this work may start, what it cost, and whether it may
continue; which agent definitions exist and which is the default; what earlier
runs of this shape learned; who receives completed work so durable knowledge can
be derived from it; who receives live events across a process boundary; what kind
of failure just occurred and whether it should be retried; and which models this
unit of work uses.

Every embedder answers all ten. Today each answers them by wrapping the harness
in bespoke glue, which means the seams are defined implicitly, differently, and
by whoever integrated first. Naming them as traits does three things: it makes
the runtime's requirements legible without reading an integration, it lets the
harness own ordering and assembly (fragment placement, lease lifetime, event
offsets) rather than leaving them to each host, and it gives the crate a place
to ship an inert default so a partial adoption still runs.

None of the ten introduce a new architectural idea; they are more of the pattern
the crate already uses eighteen times. OpenHuman is the first consumer and drove
the call-site grounding below, but the catalogue is deliberately free of its
vocabulary, and the rules in [Publishability](#publishability-boundary) are
enforced by a test rather than by convention.

## Scope

**In scope:** trait definitions, their request/response types, one inert default
implementation each, and the module they live in.

**Out of scope, deliberately:** wiring (the traits land unreferenced by the
runtime; a bundle struct on `AgentHarness`/`Session` is a separate change); the
durable transcript (compaction-replacement records, interrupted partials, and
display-order reads belong to `ChatHistory` and are specified separately —
nothing here reads, writes, or compacts a transcript); and configuration, for
which see [Configuration](#configuration-is-not-a-seam).

## Relationship to the existing extension traits

The crate ships eighteen extension traits today: `ChatModel`, `Tool`,
`Middleware`, `ModelMiddleware`, `ToolMiddleware`, `ModelBaseCall`,
`ToolBaseCall`, `ChatHistory`, `Store`, `AppendStore`, `Summarizer`,
`EmbeddingModel`, `VectorStore`, `ResponseCache`, `WorkspaceIsolation`,
`HarnessEventJournal`, `HarnessStatusStore`, and `EventListener`.

The ten here follow the same conventions — `Send + Sync` supertrait,
`crate::error::Result<T>`, `&self` receivers, `&State` as the first parameter,
borrowed `&Request<'_>` structs for reads and owned values for writes, default
method bodies for everything optional. They are consumed as
`Arc<dyn Trait<State>>` exactly like `ChatModel` and `Tool`.

They live in a **new module**, `harness::host`, rather than being scattered into
the concern modules they sit beside. Every existing harness module names a
concern the crate itself implements: `memory` ships `InMemoryChatHistory`, `model`
ships real providers, `store` ships a file-backed store. These ten are the
opposite axis — points where the crate deliberately has no opinion and ships only
defaults that do nothing. Separation also does one concrete job: it keeps
`MemoryProvider` out of `harness::memory`, where a scored-retrieval trait sitting
next to `ChatHistory` would invite exactly the confusion below, and a doc comment
is a weaker fence than a different module.

Four non-overlaps are worth stating outright, because each has already been
mistaken once. `MemoryProvider` is not `ChatHistory`: scored retrieval over a
namespaced corpus, not a thread's ordered message list — no `messages()`, no
`append`, no offsets, no replay. `SecurityGate` is not `WorkspaceIsolation`:
the latter prepares an environment, the former decides what is permitted inside
one. `ProgressSink` is not `EventListener`: `EventListener` is synchronous and
must not block the emitting step. `BudgetGate` is not `Summarizer`: money and
admission, not fitting a conversation into a window. The full table, including
`Store` and the intra-trait boundaries, is in the module doc.

## The catalogue

Signatures below are abridged to the trait surface; types, builders, and full doc
comments are in `src/harness/host/`.

### 1. `MemoryProvider` — retrieval-oriented long-term memory

```rust
#[async_trait]
pub trait MemoryProvider<State: Send + Sync>: Send + Sync {
    async fn recall(&self, state: &State, query: &MemoryQuery<'_>) -> Result<Vec<MemoryRecord>>;
    async fn list(&self, state: &State, filter: &MemoryFilter<'_>) -> Result<Vec<MemoryRecord>>;
    async fn write(&self, state: &State, record: MemoryWrite) -> Result<()>;
    async fn namespace_digests(&self, state: &State, caps: DigestCaps)
        -> Result<Vec<NamespaceDigest>> { Ok(Vec::new()) }
}
```

`MemoryQuery` carries `text`, `limit: Option<usize>` (`None` means the backend's
own default, `DEFAULT_RECALL_LIMIT` in-crate), `namespace`, `thread_id`,
`cross_thread`, and `min_score`. `MemoryFilter` is the unscored counterpart:
`namespace`, `category`, `thread_id`, `limit`. `MemoryRecord` carries identity,
key, content, optional namespace/category/thread/score/timestamp, and an opaque
`attributes: Value` so backends round-trip provenance the crate does not model.
`limit` is `Option` rather than `usize` because a derived `Default` with a bare
`usize` yields `limit: 0`, and `MemoryQuery { text, ..Default::default() }` is
exactly how a first-time caller writes it.

**Default:** `InMemoryMemoryProvider`, a mutex-guarded map scoring by
whitespace-token containment — same shape and same durability guarantees (none)
as `InMemoryChatHistory` and `InMemoryStore`.

### 2. `ContextComposer` — system prompt and per-turn fragments

```rust
#[async_trait]
pub trait ContextComposer<State: Send + Sync>: Send + Sync {
    fn compose_system_prompt(&self, state: &State, request: &SystemPromptRequest<'_>)
        -> Result<String>;
    async fn prepare_turn(&self, state: &State, request: &TurnPreparationRequest<'_>)
        -> Result<TurnPreparation> { Ok(TurnPreparation::default()) }
}
```

`prepare_turn` returns `TurnPreparation { blocks: Vec<ContextBlock>, extras:
Value }`, each block being `{ id, body, placement, priority }`. Per-turn
enrichment becomes an ordered set of registered fragments rather than a bespoke
method body: the crate owns ordering, placement, and assembly, the embedder owns
every byte of text.

`ContextPlacement` (`TurnPrefix` default, `SystemPrefix`) encodes an invariant,
not a preference. Run-stable content in the system prompt keeps a provider's
cached prefix valid across turns, and history trimming commonly hoists system
messages to the front, reordering content meant to ride one specific turn. A
typed placement is what stops that being rediscovered.

`SystemPromptRequest` carries the run/thread ids, `agent_id`, `model_id`, the
tool schemas, `visible_tool_names`, `tool_call_instructions`, and
`workspace: Option<&WorkspaceDescriptor>` — the crate's existing descriptor,
which already models a primary root plus trusted roots. `TurnPreparationRequest`
adds `input`, `turn_index`, `first_turn`, and `resumed`.

**Default:** `PassthroughContextComposer` — empty prompt, empty preparation,
no observable effect on a run.

### 3. `SecurityGate` — what the run may see and touch

```rust
#[async_trait]
pub trait SecurityGate<State: Send + Sync>: Send + Sync {
    fn filter_tools(&self, state: &State, request: &ToolExposureRequest<'_>)
        -> Result<ToolExposure> { /* everything visible */ }
    async fn authorize_call(&self, state: &State, request: &ToolCallRequest<'_>)
        -> Result<CallVerdict> { Ok(CallVerdict::Allow) }
    fn resolve_path(&self, state: &State, request: &PathRequest<'_>) -> Result<PathBuf>;
    fn screen_input(&self, state: &State, request: &InputScreenRequest<'_>)
        -> Result<InputVerdict> { Ok(InputVerdict::Admit) }
    fn redact(&self, state: &State, request: &RedactionRequest<'_>)
        -> Result<Redaction> { Ok(Redaction::unchanged(request.text)) }
}
```

The crate defines no policy here: it defines the questions. Two distinctions
carry the weight. First, `filter_tools` is *advertisement* and
`authorize_call` is *enforcement*. A gate that only narrows the advertised set
admits anything the model names from memory, and the advertised set is name-only
while a real decision depends on the arguments. Hence `ToolCallRequest` carries
`arguments: &Value`, and `CallVerdict` is three-valued —
`Allow` / `Deny { code, message }` / `RequireApproval { code, message }` —
because a policy that can only allow or deny cannot express "ask a human first"
and silently degrades into one of the other two.

Second, `screen_input` returns a verdict and `redact` returns modified text, because
masking a secret — or fencing untrusted content so a model treats it as data
rather than instruction — is not expressible as `Admit`/`Refuse`. `redact` runs
in both directions (`RedactionDirection::{Inbound, Outbound}`): inbound text on
its way into a prompt, outbound tool output on its way into storage or a preview.

**Default:** `RootContainedSecurityGate`. `resolve_path` normalises **lexically**
and never touches the filesystem, so a path that does not exist yet resolves
exactly like one that does; it is therefore not a defence against symlinks. The
other four methods take their permissive trait defaults, which is stated plainly
rather than described as "fail-closed": a host that forgets to wire a gate runs
unpoliced on four of the five questions.

### 4. `BudgetGate` — admission control and cost accounting

```rust
#[async_trait]
pub trait BudgetGate<State: Send + Sync>: Send + Sync {
    async fn acquire(&self, state: &State, request: &AdmissionRequest<'_>)
        -> Result<Option<BudgetLease>> { Ok(Some(BudgetLease::unmetered())) }
    fn estimate_cost(&self, state: &State, model_id: &str, usage: &Usage) -> CostTotals
        { CostTotals::default() }
    async fn record_usage(&self, state: &State, entry: &UsageEntry<'_>) -> Result<()>;
    async fn account_turn(&self, state: &State, charge: &TurnCharge<'_>)
        -> Result<BudgetVerdict> { Ok(BudgetVerdict::Continue) }
}
```

The crate already tracks `Usage` and `CostTotals` and enforces `RunLimits`. This
seam is for *external* budgets: process-wide concurrency, pricing tables, and
durable spend ledgers the crate cannot know about. `Ok(None)` from `acquire`
means admission was refused without an error — a paused scheduler, not a failure.
`BudgetVerdict::Stop { reason }` is a graceful request drained at the next
iteration boundary, not an abort; a bounded overshoot is expected. `BudgetLease`
is `Box<dyn Any + Send + Sync>` with an `into_inner`, so the crate holds a host's
semaphore permit for the right duration without knowing its type.

**Default:** `UnmeteredBudgetGate` implements only `record_usage`, as `Ok(())`.
Every other method is the trait default, which is the "no budget exists" answer
throughout, so wiring it is observationally identical to an ungated run.

### 5. `DefinitionRegistry` — which agents exist

```rust
#[async_trait]
pub trait DefinitionRegistry<State: Send + Sync>: Send + Sync {
    async fn get(&self, state: &State, id: &str) -> Result<Option<AgentDefinition>>;
    async fn list(&self, state: &State) -> Result<Vec<AgentDefinition>>;
    async fn default_id(&self, state: &State) -> Result<Option<String>> { Ok(None) }
}
```

Read-only by design: mutation, validation, and precedence between sources are the
embedder's concern and happen before `list` returns. An unknown id is `Ok(None)`,
not an error. `default_id` keeps a default agent identity out of the crate.

`AgentDefinition` is deliberately small — `id`, optional `description`, optional
rendered `system_prompt`, optional `model`, a `tools: Vec<String>` name list, and
an opaque `extras: Value`. Hosts routinely have far richer definition types:
prompt-source indirection, compaction profiles, delegation overrides, workspace
layout. Those map *into* this type at registration time and ride in `extras`. A
published type that grows a field per host concept is a breaking change every
time a host learns something new, and host field names are themselves product
vocabulary. `Default` is derived so hosts construct with struct-update syntax and
keep compiling if the crate ever does add a field.

**Default:** `InMemoryDefinitionRegistry`, insertion-ordered, constructed empty.

### 6. `ExperienceStore` — what prior runs learned

```rust
#[async_trait]
pub trait ExperienceStore<State: Send + Sync>: Send + Sync {
    async fn retrieve(&self, state: &State, query: &ExperienceQuery<'_>)
        -> Result<Vec<ExperienceHit>>;
    async fn record(&self, state: &State, entries: Vec<ExperienceEntry>)
        -> Result<()> { Ok(()) }
}
```

Distinct from `MemoryProvider`, which serves the user's corpus: this serves the
agent's own record of what worked. The crate stores nothing and renders nothing —
`ExperienceHit::body` is host-rendered and used verbatim, and a hit with empty
`match_reasons` carries no evidence for why it was selected, so callers may drop
it. `ExperienceQuery::partition` is an opaque key; `None` searches all of them.

**Default:** `NoopExperienceStore` — empty retrieval, discarding `record`.

### 7. `LearningSink` — completed work, for derivation

```rust
#[async_trait]
pub trait LearningSink<State: Send + Sync>: Send + Sync {
    async fn on_turn_completed(&self, state: &State, record: &TurnRecord) -> Result<()>;
    async fn on_transcript_committed(&self, state: &State, commit: &TranscriptCommit<'_>)
        -> Result<()> { Ok(()) }
}
```

Fire-and-forget by contract: the runtime does not wait on the result and treats
an `Err` as a logged failure, never a run failure. Implementations must return
promptly and defer real work to a background task. `TurnRecord` carries the run,
optional thread/agent/entrypoint, input, output, tool outcomes, model-call count,
and elapsed time; each `ToolOutcomeRecord::summary` is a bounded, non-sensitive
description, never raw output.

**Default:** `NoopLearningSink`, indistinguishable from registering nothing.

### 8. `ProgressSink` — asynchronous delivery of run events

```rust
#[async_trait]
pub trait ProgressSink<State: Send + Sync>: Send + Sync {
    fn is_connected(&self, state: &State) -> bool { true }
    async fn deliver(&self, state: &State, record: &EventRecord) -> Result<()>;
}
```

Complements `EventListener`, which is synchronous and must not block the emitting
step. A `ProgressSink` may await — it exists for consumers behind a bounded
channel, a socket, or an IPC boundary, where silently dropping events is not
acceptable. `is_connected` lets callers skip assembling expensive payloads when
nothing will read them; a sink returning `false` must still tolerate `deliver`.
An `Err` means the consumer is gone; the runtime logs it and continues.

It carries the crate's own `EventRecord`. No parallel event enum is introduced,
and none should be: projecting crate events into a presentation model is the
embedder's job. It is also **delivery only** — there is no receive side. An
interactive loop reading input from a terminal or a chat platform is host
surface.

**Default:** `NoopProgressSink`, whose `is_connected` returns `false`,
deliberately overriding the trait default so callers take their
skip-expensive-payload path.

### 9. `ToolOutcomeClassifier` — what kind of failure was that

```rust
pub trait ToolOutcomeClassifier<State: Send + Sync>: Send + Sync {
    fn classify(&self, state: &State, failure: &ToolFailureContext<'_>) -> Option<ToolFailure>;
}
```

Synchronous and pure by contract, mirroring `EventListener`.
`ToolFailureContext::error` arrives with any separate error field and result body
already combined, so a classifier need not guess where the signal is, and
`timed_out` is supplied because text alone does not reliably say whether the
runtime aborted the call on its own deadline.

`ToolFailure` carries `class`, `category`, `cause`, `next_action` — all
host-authored, all display-and-logging only — plus
`retry: RetryDisposition { Unknown (default), Never, Immediate, Backoff }`. Crate
code branches on `RetryDisposition` and must never branch on `class` or
`category`; that separation is what lets the failure taxonomy stay host-owned
while a retry ladder lives in the crate. A plain `retryable: bool` was rejected:
it collapses "we do not know" into "yes" and silently widens the ladder.

**Default:** `NoopToolOutcomeClassifier` — `None` for every input.

### 10. `ModelResolver` — which models this work uses

```rust
pub trait ModelResolver<State: Send + Sync>: Send + Sync {
    fn resolve(&self, state: &State, request: &ModelResolution<'_>)
        -> Result<ModelRegistry<State>>;
    fn profile(&self, state: &State, model_id: &str) -> Result<Option<ModelProfile>> { Ok(None) }
    fn context_window(&self, state: &State, model_id: &str) -> Result<Option<u64>> { Ok(None) }
}
```

`resolve` returns a populated `ModelRegistry` — the primary model as the registry
default, plus any additional named routes the run may select — so the crate's own
selection, fallback, and capability checks run unchanged on top of the host's
routing decision. It is a per-**run** call, not per-turn; it allocates a registry
each time. `profile` answers "does this model do native tool calls / accept
images?" before a registry exists, and lets an embedder override a
provider-reported capability from its own configuration; `context_window`
returning `None` means no window-driven compaction is scheduled.
`ModelResolution::workload` is an opaque host-defined label, so routing rules
stay entirely embedder-side.

**Default:** `StaticModelResolver<State>` — one `Arc<dyn ChatModel<State>>` plus
a name; `resolve` builds a fresh registry and registers it. The in-crate analogue
of `AgentHarness::register_model`.

## Why ten

Ten is a budget, not a starting point. Every trait here is grounded in call sites
that exist in a real integration; nothing was added because it seemed
architecturally tidy. An eleventh seam is evidence that one of these ten is drawn
wrong, and the escalation path is to **reopen this document** and re-argue the
partition rather than append to it. Concretely: if a capability does not fit,
first check whether it is a `ContextComposer` registration plus a `LearningSink`
call (that pair covers most "we need a hook here" requests), whether it is an
existing trait (`WorkspaceIsolation`, `Summarizer`, `EmbeddingModel`, `Tool`), or
whether it is host surface that should never have reached the crate.

`ExperienceStore` is the weakest of the ten by that test and the first candidate
to fold if the budget ever binds: `retrieve` feeds one context block and `record`
is a post-turn hook, so both halves fit the composer/sink pair.

## Async only where the work is async

A seam is `async` when a realistic implementation performs I/O, and a plain `fn`
otherwise — precedent: `ChatModel` already pairs a sync `profile` with an async
`invoke`. Sync here: `compose_system_prompt`, all of `SecurityGate` except
`authorize_call`, `estimate_cost`, all of `ModelResolver`, all of
`ToolOutcomeClassifier`. Async: `prepare_turn`, `authorize_call`, the rest of
`BudgetGate`, and all of `MemoryProvider`, `DefinitionRegistry`,
`ExperienceStore`, `LearningSink`, and `ProgressSink::deliver`.

This is not stylistic. Marking a seam `async` forces every caller above it to
become `async` too, and those callers are frequently synchronous session
assembly, artifact persistence, and cold-boot resume paths — some of which exist
specifically to avoid fanning out to a store. That cascade is paid by the
embedder, so sync is the default and `async` has to earn its place.

## Configuration is not a seam

The harness needs configuration, and the accepted answer is **crate-owned config
structs populated by the host when it builds a run**. Explicit, versionable, no
virtual call per read, and it keeps host schema vocabulary out of the crate. No
`ConfigProvider` trait exists and none should be added.

Two alternatives were considered and rejected:

- **A config trait with per-value getters.** Rejected. It avoids a mapping layer
  but turns every configuration read into a virtual call and becomes a dumping
  ground: nothing in its shape resists a fortieth getter, and each one published
  is a host concept in the crate's public API.
- **Carrying configuration solely in the generic `State`.** Rejected as the
  primary mechanism. It is the least code and the worst discoverability — a
  crate-side read needs a bound, and no signature says what the runtime requires.

That second rejection is narrower than it looks. Crate-owned structs cover
*build-time* configuration; they do not cover a value that must be re-read
*mid-session* so a live toggle takes effect without rebuilding. The vehicle for
those is `State`: every method on every trait here receives `&State`, so a host
that parks a reloadable handle in its state type keeps read-at-call-time
semantics with no crate surface at all. A host with such toggles should say so in
its own adapter and test it; the crate neither helps nor hinders.

## Publishability boundary

The crate is published, so a field name, an enum variant, or a doc-comment
example encoding one embedder's internal concept becomes public API for everyone
and ships to docs.rs. The rules the ten traits follow:

- Every routing or grouping decision that would carry product meaning is an
  opaque host-defined string the crate never interprets:
  `ModelResolution::workload`, `ToolExposureRequest::entrypoint`,
  `ExperienceQuery::partition`, `MemoryFilter::category`.
- Every user-facing string is host-authored and passed through verbatim:
  `ToolExposure::boundary_note`, `InputVerdict::Refuse::message`,
  `CallVerdict::{Deny,RequireApproval}::message`, `BudgetVerdict::Stop::reason`,
  all four strings on `ToolFailure`.
- Where the crate must act on a classification, it defines its own small neutral
  vocabulary rather than reading the host's — `RetryDisposition`, not
  `ToolFailure::class`.
- `AgentDefinition` stays minimal with an `extras: Value` escape hatch rather
  than absorbing host definition fields.
- `SystemPromptRequest` carries the crate's own `WorkspaceDescriptor` rather than
  a second, differently-named pair of roots. A product-specific second root was
  proposed and rejected: it carried no crate-side meaning and duplicated an
  existing seam.
- Presentation payloads do not enter the crate. A typed citation list on
  `TurnPreparation` was proposed and rejected: it had no crate consumer, and its
  field-by-field mismatch with the host's own presentation type would have quietly
  changed what a UI received. `extras: Value` replaces it.

`tests/host_seam_hygiene.rs` enforces this mechanically: it scans
`src/harness/host/` for embedder vocabulary and fails with file, line, and
reason. It runs under `cargo test`, so it is a gate rather than a convention.
**As runtime code is relocated into this crate, `SCANNED_DIRS` in that test must
grow with it** — a relocation that does not widen the list has not been checked.

## Adoption phasing

The traits are designed to land ahead of any consumer, and did.

1. **Land empty (done).** Traits plus inert defaults, referenced by nothing in
   the runtime. No behaviour change; a host that upgrades notices only a larger
   public surface. Shipped as a patch release.
2. **Hosts implement in place.** An embedder implements the traits against its
   existing internals and repoints its own call sites without relocating code.
   This is where the architectural value lands — a host's outbound coupling
   collapses to this catalogue — and it is a legitimate stopping point. Wiring
   the traits into `AgentHarness`/`Session`, most likely as a
   `HostCapabilities<State>` bundle struct, belongs to this step.
3. **Runtime relocation.** Generic runtime code moves into the crate and consumes
   the traits directly. Each relocated family must widen
   `tests/host_seam_hygiene.rs` and re-check the boundary rules above.

## Open questions

- **Should the capability seams be generic over `State`?** They are, matching
  `ChatModel` and `Tool`. But only 7 of the 18 existing extension traits carry a
  `State` parameter and all 7 are *execution* traits; the other 11 —
  `ChatHistory`, `Store`, `Summarizer`, `WorkspaceIsolation`, `EventListener`,
  and the rest — are bare `pub trait X: Send + Sync` capturing dependencies in
  `Arc` fields, and are what these ten most resemble. `State` is also
  load-bearing here for mid-session config reads, which argues for keeping it.
  The cost is that every host construction site spells
  `Arc<dyn MemoryProvider<()>>`, and adopting a non-unit `State` later is a
  breaking change across every impl. Revisit before step 2 hardens.
- **Sub-agent attribution in `AgentEvent`.** Crate events cannot attribute a
  sub-agent's tool and model calls to a specific child task, so a projecting host
  must maintain its own stack to synthesize one. No `ProgressSink` signature
  fixes this; it is a field gap in `harness::events` and belongs in its own
  issue.
- **`BudgetLease` as `Box<dyn Any + Send + Sync>`.** The only way to hold a
  host's permit without depending on its type, but `dyn Any` in a published
  struct is unusual. A marker trait with `Box<dyn BudgetLease>` is more idiomatic
  and forces hosts to newtype their permit. Open.
- **`LearningSink::on_transcript_committed` names a `&Path`,** which presumes a
  file-backed transcript. If durable history becomes crate-owned and capable of a
  database backing, this should become an opaque locator instead.
