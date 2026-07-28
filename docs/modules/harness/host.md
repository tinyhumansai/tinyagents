# Host capability seams (`harness::host`)

Ten extension traits an embedding application implements, plus one inert
default per trait so a host can adopt them one at a time. Source lives in
`src/harness/host/`, one file per capability, re-exported through
`src/harness/host/mod.rs` and again at the crate root.

The accepted design record — motivation, per-trait signatures and rationale, the
ten-trait budget, the rejected configuration alternatives, and the open questions
— is [`docs/spec/host-capabilities-spec.md`](../../spec/host-capabilities-spec.md).
This file covers the operational detail an implementer needs day to day.

## Why a separate module

Every other harness module names a concern the crate itself implements —
`memory` ships `InMemoryChatHistory`, `model` ships real providers, `store`
ships a file-backed store. These ten are the opposite axis: points where the
crate deliberately has **no** opinion and ships only defaults that do nothing.
That is a different kind of thing and it gets its own name.

Keeping them separate also does one concrete job: it keeps `MemoryProvider` out
of `harness::memory`. Putting a scored-retrieval trait next to `ChatHistory`
invites exactly the confusion the scope table below exists to prevent, and a
doc comment is a weaker fence than a different module.

## The ten

| Trait | Question it answers | Default |
|-------|---------------------|---------|
| `MemoryProvider` | What does the host know that is relevant to this text? | `InMemoryMemoryProvider` |
| `ContextComposer` | What text precedes the model request this turn? | `PassthroughContextComposer` |
| `SecurityGate` | Is this tool, path, input, or call permitted? | `RootContainedSecurityGate` |
| `BudgetGate` | May this work start, what did it cost, may it continue? | `UnmeteredBudgetGate` |
| `DefinitionRegistry` | Which agents exist and which is the default? | `InMemoryDefinitionRegistry` |
| `ExperienceStore` | What did prior runs of this shape learn? | `NoopExperienceStore` |
| `LearningSink` | Here is completed work — derive what you like from it. | `NoopLearningSink` |
| `ProgressSink` | Deliver run events to an out-of-process consumer. | `NoopProgressSink` |
| `ToolOutcomeClassifier` | What kind of failure was that? | `NoopToolOutcomeClassifier` |
| `ModelResolver` | Which models should this unit of work use? | `StaticModelResolver` |

Ten is a budget, not a starting point. An eleventh seam is evidence that one of
these is drawn wrong; reopen the design rather than appending.

## Scope boundaries

These are the confusions that cost real debugging time when they are left
implicit.

| This | Is not | Because |
|------|--------|---------|
| `MemoryProvider` | `ChatHistory` | Scored retrieval over a namespaced corpus, not a thread's ordered message list. No `messages()`, no `append`, no offsets, no replay. Nothing here reads, rewrites, or compacts a transcript. |
| `MemoryProvider` | `Store` / `AppendStore` | Those are opaque key/value and offset-addressed streams with no notion of relevance. |
| `SecurityGate` | `WorkspaceIsolation` | `WorkspaceIsolation` prepares an environment; `SecurityGate` decides what is permitted inside one. |
| `SecurityGate::filter_tools` | `SecurityGate::authorize_call` | The first decides what the model is *told* exists. The second decides whether one concrete call — arguments and all — may run. A gate that only narrows the advertised set admits anything the model names from memory. |
| `ProgressSink` | `EventListener` | `EventListener` is synchronous and must not block the emitting step. `ProgressSink` may await, for consumers behind a channel, socket, or IPC boundary. |
| `ProgressSink` | a transport | Delivery only. There is no receive side; an interactive loop that reads from a terminal or a chat platform is host surface. |
| `BudgetGate` | context compaction | Money and admission. Fitting a conversation into a window is `Summarizer` plus `RunLimits`. |
| `DefinitionRegistry` | workspace layout or prompt personality | It supplies `AgentDefinition`. Directory roots are `WorkspaceIsolation`, personality is `ContextComposer`, scoped retrieval is `MemoryQuery::namespace` / `ExperienceQuery::partition`. |

## Async only where the work is async

A seam is `async` when a realistic implementation performs I/O, and a plain
`fn` otherwise. The precedent is `ChatModel`, which already pairs a sync
`profile` with an async `invoke`.

| Sync | Async |
|------|-------|
| `ContextComposer::compose_system_prompt` | `ContextComposer::prepare_turn` |
| all of `SecurityGate` except `authorize_call` | `SecurityGate::authorize_call` |
| `BudgetGate::estimate_cost` | rest of `BudgetGate` |
| all of `ModelResolver` | — |
| all of `ToolOutcomeClassifier` | — |

This is not stylistic. Marking a seam `async` forces every caller above it to
become `async` too, and those callers are frequently synchronous session
assembly, artifact persistence, and cold-boot resume paths. That cascade is
paid by the embedder, so the default is sync and `async` has to earn its place.

## Configuration is not a seam

There is no `ConfigProvider`. Crate-side configuration is expressed as
crate-owned structs the host populates when it builds a run — explicit,
versionable, and free of virtual calls on every read.

That covers build-time configuration. It does not, on its own, cover a value
that must be re-read *mid-session* so a live toggle takes effect without
rebuilding. The vehicle for those is `State`: every method on every trait here
receives `&State`, so a host that parks a reloadable handle in its state type
keeps read-at-call-time semantics with no crate surface at all. A host with
such toggles should say so in its own adapter and test it; the crate neither
helps nor hinders.

## Product vocabulary stays out

The crate is published. A field name, an enum variant, or a doc-comment example
that encodes one embedder's internal concept becomes public API for everyone
and ships to docs.rs.

The rules the ten traits follow:

- Every routing or grouping decision that would carry product meaning is an
  opaque host-defined string the crate never interprets:
  `ModelResolution::workload`, `ToolExposureRequest::entrypoint`,
  `ExperienceQuery::partition`, `MemoryFilter::category`.
- Every user-facing string is host-authored and passed through verbatim:
  `ToolExposure::boundary_note`, `InputVerdict::Refuse::message`,
  `BudgetVerdict::Stop::reason`, all four strings on `ToolFailure`.
- `ToolFailure::class` and `::category` are opaque `String`s for display and
  logging **only**. Crate code must never branch on them — that is what
  `RetryDisposition` (`Unknown` / `Never` / `Immediate` / `Backoff`) is for. A
  plain `retryable: bool` was rejected because it collapses "we do not know"
  into "yes", which silently widens whatever retry ladder consumes it.
- `DefinitionRegistry::default_id` exists so no default agent id is hard-coded
  in the crate.
- `BudgetLease` is `Box<dyn Any + Send + Sync>` so the crate can hold a host's
  semaphore permit for the right duration without knowing its type.
- `SystemPromptRequest` carries the crate's own `WorkspaceDescriptor` rather
  than a second, differently-named pair of roots.

`tests/host_seam_hygiene.rs` enforces this mechanically: it scans
`src/harness/host/` for embedder vocabulary and fails with file, line, and
reason. **As runtime code is relocated into this crate, `SCANNED_DIRS` in that
test must grow with it** — a relocation that does not widen the list has not
been checked.

## `AgentDefinition` is deliberately small

`AgentDefinition` carries an id, an optional description, an optional rendered
system prompt, an optional model, a tool-name list, and an opaque
`extras: Value`. Nothing else.

Hosts routinely have far richer definition types — prompt-source indirection,
compaction profiles, delegation overrides, workspace layout. Those map *into*
this type at registration time and ride in `extras`. Two reasons:

1. A published type that grows a field per host concept is a breaking change
   every time the host learns something new. Adding a key to `extras` breaks
   no downstream build.
2. Host field names and their doc comments are product vocabulary, and this
   type is the most exposed surface in the module.

`Default` is derived so hosts can construct with struct-update syntax and keep
compiling if the crate ever does add a field.

## `ContextPlacement` encodes an invariant, not a preference

`TurnPrefix` (default) splices a fragment onto the turn's user message;
`SystemPrefix` prepends it to the system prompt. The distinction is
load-bearing in two ways: run-stable content in the system prompt is what keeps
a provider's cached prefix valid across turns, and history trimming commonly
hoists system messages to the front, which reorders content that was meant to
ride one specific turn. Making placement a typed property is what stops that
being rediscovered.

## Default-implementation behaviour worth knowing

- `InMemoryMemoryProvider` scores by whitespace-token substring containment
  (`matched / total`), sorts by score descending with `key` ascending as the
  tiebreak so results are deterministic despite the backing `HashMap`, and
  short-circuits an empty query to `Ok(vec![])` before any division.
  `namespace_digests` takes the empty trait default.
- `RootContainedSecurityGate::resolve_path` normalises **lexically** and never
  touches the filesystem, so a path that does not exist yet resolves exactly
  like one that does. It is therefore not a defence against symlinks; a host
  that cares must layer that on.
- `NoopProgressSink::is_connected` returns `false`, deliberately overriding the
  trait default, so callers take their skip-expensive-payload path.
- `UnmeteredBudgetGate` implements only `record_usage`; every other method is
  the trait default, which is the "no budget exists" answer throughout.
