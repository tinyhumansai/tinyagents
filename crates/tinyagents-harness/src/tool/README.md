# harness::tool

Harness-side registration, projection, and execution support for canonical
tools. Tool *vocabulary* (`Tool`, `ToolCall`, `ToolResult`, `ToolPolicy`,
`ToolTimeout`, `WorkspaceDescriptor`, ...) belongs to the `tinytools` crate;
this module owns only the host concerns that sit between a declared tool and
a live agent run: name lookup, provider-schema projection, injected-argument
enforcement, timeout resolution, and the explicit recursive-dispatch handoff
for tools that must see the typed parent run. Prompt-guided (text-mode) tool
calling is *not* owned here: the protocol (render, parse, repair, stream
scrub) lives in `tinytools-agent`, reached through
`tinyinference_llm::prompt_tools`, and the host-side dialect choice lives in
`agent_loop/dialect.rs`.

## Public surface

### Registration and dispatch (`mod.rs`)

- `ToolRegistry<State, Ctx>` — a name-keyed registry of canonical tools.
  `register` wraps a plain `tinytools::Tool` in `dispatch`-based execution;
  `register_dispatch` accepts an explicit `ToolDispatch` for the rare tool
  that needs the typed parent state/context (currently sub-agents). `names`,
  `schemas`, `declared_specs`, and `policies` project the registry for the
  model, the transcript, and the policy layer.
- `ToolDispatch<State, Ctx>` — the trait a typed recursive dispatcher
  implements: its own `tool()` declaration, `output_origin()` for the host's
  content-screening boundary, `injected_arguments()` for host-authoritative
  values, `call_options()`, and the `execute()` entry point that receives the
  full typed `State` and `RunContext<Ctx>`.
- `provider_schema` — converts a canonical `ToolSpec` into the inference
  provider's `ToolSchema`, stripping injected arguments first.

### Execution context (`types.rs`)

- `ToolExecutionContext` — the harness-owned bridge from a live run to
  `tinytools::ToolRunContext`, built per call: run id, **call id**, thread
  id, depth, output-token budget, event sink, cancellation token, streaming
  flag, optional workspace descriptor, optional `NamespacedStore`
  (`RunContext::with_namespaced_store`), and an optional typed state view
  (`RunContext::with_state_view`, read with `state::<S>()`). `custom(payload)`
  emits `AgentEvent::Custom` correlated to the call. A `tinytools::Tool`
  reaches all of this by downcasting `ToolRunContext::host_extension()`
  — `ToolExecutionContext::of(ctx)` does that downcast and
  `ToolExecutionContext::call_id_of(ctx)` reads the call id through it;
  the portable methods still cover only workspace/thread/output cap. A tool
  that needs a real child `RunContext` goes through the explicit dispatch
  seam instead (B1; see `docs/modules/harness/tool-context.md`).
  `progress` is the per-call `tinytools::ProgressSink` behind
  `ToolRunContext::report_progress`; it is `None` for a context built outside
  a loop, where reporting is a no-op.

### Nested calls (`nested.rs`)

- `NestedToolRunner` — the type-erased seam behind
  `ToolExecutionContext::call_tool`; `nested` is `None` outside the agent loop,
  where `call_tool` returns a clear error. The loop installs the runner in a
  task-local scoped to the executing call (`agent_loop/nested.rs`); see
  `docs/modules/harness/nested-tool-calls.md`.

### Progress (`progress/mod.rs`)

- `ToolProgressGate` (crate-private) — the per-call destination for a tool's
  `report_progress` updates. Emits `AgentEvent::ToolProgressDetail` live, queues a
  `ToolDelta` per event for the loop to replay to `on_tool_delta`, closes when
  the call settles so late updates are dropped, and coalesces floods. The loop
  scopes it in a task-local around the dispatch future and
  `ToolExecutionContext::from_run_context` picks it up for the matching call
  id, so `ToolDispatch` implementors need no change.
- `ToolProgressLimits` (crate-private) — events admitted per window before
  coalescing (fixed at 32 per second). The gate closes on settle **or when the
  scoped future is dropped**; the replay queue is bounded (64 deltas, 4 KiB
  content each) and skipped when the run has no middleware. A `ToolDispatch`
  must build its `ToolExecutionContext` inside `execute`'s future.

### Injected arguments

Strip, projection and host-value injection live in `tinytools`
(`prepare_tool_arguments`, `project_injected_arguments`); the registry applies
the projection to the schemas it sends to the model. The security-critical
ordering (strip → validate → inject → invoke) is enforced by the agent loop's
tool-execution path.

### Schema cleaning (`schema.rs`)

- `SchemaCleanr` — normalizes a JSON Schema for a target provider: resolves
  local `$ref`/`$defs`, strips provider-unsupported keywords, flattens
  same-typed literal unions, drops nullable variants, converts `const` to
  `enum`, and breaks circular local refs.
- `CleaningStrategy` — `Gemini` / `Anthropic` / `OpenAI` / `Conservative`,
  each with its own `unsupported_keywords()`.
- `GEMINI_UNSUPPORTED_KEYWORDS` — the (most restrictive) keyword list Gemini
  rejects.

### Provider projection seam (`schema_prepare.rs`)

- `prepare_tool_schemas` / `prepare_tool_schema` / `prepare_parameters` — the
  seam a provider adapter uses instead of reading `Tool::schema()` directly:
  normalize → clean (via `SchemaCleanr`) → optional strict-mode sanitizer, in
  that order.
- `SchemaPreparation` — `strategy` + `strict`; `gemini()`/`anthropic()`/
  `openai()`/`conservative()` constructors plus `with_strict()`.
- `normalize_parameters` — replaces a missing/non-object parameter schema
  with an empty object schema so a tool that takes no arguments cannot break
  a request.
- `require_all_properties` / `set_additional_properties_false` — the two
  halves of OpenAI strict-mode sanitization.

### Prompt-guided (text-mode) tool calling

Not implemented in this module. The `<tool_call>` / P-Format protocols —
instructions, catalogue rendering, coalescing of tool results into a user
turn, `ensure_resolvable_user_turn`, parsing, argument repair, and the
streaming scrubber — are owned by `tinytools-agent` and exposed to adapters
through `tinyinference_llm::prompt_tools` (`with_tool_instructions`,
`coalesce_tool_results`, `recover_tool_calls`, `TextScrubber`). The agent
loop selects the run's dialect (`RunPolicy::tool_dialect`) and mints
`{model_call_id}-tool-{n}` ids for calls recovered from text in
`agent_loop/dialect.rs`. See `docs/modules/harness/tool-dialect.md`.

### Timeouts (`timeout.rs`)

- `ToolTimeoutSettings` — shared, atomically-updateable timeout policy: an
  inherited default (`ToolTimeout::Inherit`), clamped `min_ms..=max_ms`
  bounds, and `grace_ms` scheduling slack. `set_inherited_timeout_ms` lets a
  host apply a config/operator override to every harness sharing the
  settings without rebuilding them.
- `ResolvedToolTimeout` — the `resolve()` output: enforced `deadline` plus
  unpadded `budget_ms` reported to callers and observability.

### Tool selection (`select/`)

See `select/README.md` (or the module doc on `select/mod.rs`) for the
prompt-driven ranker that narrows a large tool catalogue before it reaches
the model. Re-exported here as `pub mod select` and via `pub use select::*`.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `ToolRegistry`, `ToolDispatch`, `provider_schema`; wires the submodules together. |
| `types.rs` | `ToolExecutionContext` (tests in `context_tests.rs`). |
| `schema.rs` | `SchemaCleanr`, `CleaningStrategy`; low-level JSON Schema cleaning. |
| `schema_prepare.rs` | Provider projection seam built on `schema.rs`; strict-mode sanitizer. |
| `schema_walk.rs` | Vendor-neutral schema/value walkers: primary array path, response fields, missing and unsupported argument names. |
| `shared/` | `CanonicalSharedToolAdapter` (a `tinytools::Tool` over shared, non-cloneable registries) and `EarlyExitHook` (pause the run when a designated tool succeeds). |
| `timeout.rs` | `ToolTimeoutSettings`, `ResolvedToolTimeout`. |
| `select/` | Prompt-driven tool ranking (own submodule; see its README/module doc). |
| `*_tests.rs` | Unit tests colocated by concern, wired in with `#[path = "..."]`. |

## Operational constraints

- **Injected-argument ordering is security-critical.** Strip must run before
  validation, which must run before injection, which must run before
  invocation — see `tinytools::prepare_tool_arguments` for why the order cannot be
  relaxed.
- **Schema cleaning must run before strict-mode sanitization**
  (`prepare_parameters`), so `required` is computed from the resolved
  property set rather than one still hidden behind an unresolved `$ref`.
- **Never re-add tool-call markup matching here.** Model-specific
  render/parse/scrub logic belongs in `tinytools-agent`; this crate only
  consumes it, so a new dialect quirk is fixed upstream, not by a harness
  regex.
- Canonical tool vocabulary, execution, and policy enforcement itself remain
  in `tinytools`; this module never redeclares them, only bridges them to a
  live harness run.
