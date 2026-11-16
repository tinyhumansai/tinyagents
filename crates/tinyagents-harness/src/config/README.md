# harness::config

Crate-owned session configuration: the inert value types a host maps its own
config schema into once, at session-build time.

## Why this exists

A relocated agent runtime cannot read a host's config schema — the whole point
of the move is that the runtime is generic over its host. This module is the
alternative: the crate declares the configuration *it* needs as plain structs,
and each host writes a mapper from its own schema into these once.

A `ConfigProvider` trait with ~40 getters was considered and rejected: it
turns every config read into a virtual call, has no natural boundary so it
becomes a dumping ground, and makes "what does the runtime actually depend
on?" unanswerable without reading every implementation. A struct answers that
question by existing.

## Public surface

- [`SessionConfig`] — the root type. Owns the two path roots
  (`workspace_dir`, `action_dir`), model selection (`model`, `lead_model`,
  `subagent_model`), sampling `temperature`, delegation depth, and the three
  sub-configs below. `SessionConfig::new` is the only constructor (no
  `Default` impl — the path roots and model have no safe guess);
  `effective_lead_model` / `effective_subagent_model` / `may_delegate_at` are
  the derived accessors callers should use instead of reading the raw fields
  directly.
- [`TurnConfig`] — per-turn knobs: tool-iteration cap, history window,
  context compaction, parallel-tool concurrency, tool-result byte budget,
  turn timeout, and an optional [`RequiredOutput`] contract.
- [`ToolConfig`] — dispatch strategy ([`ToolDispatcher`]) and opaque
  per-channel permission grants passed through to the host's security gate.
- [`MemoryLimits`] — character budgets (not token budgets — a tokenizer isn't
  necessarily available where these are applied) for memory injected into a
  turn.
- [`ToolDispatcher`] — enum of tool-call encoding strategies (`Auto`,
  `Native`, `Xml`, `Pformat`, `Python`, `Typescript`); modelled as an enum
  rather than a free-form string so an unrecognised mode is a mapping error at
  the boundary, not a silent fallthrough in the turn loop.
- [`RequiredOutput`] — a structured-output contract asserting the model's
  reply carries a particular JSON block; a blank `block_key` makes the
  contract inert by design, so it is a safe zero-value default. It derives
  `Serialize`/`Deserialize`/`JsonSchema` with `#[serde(default)]` on the
  container, so a host can read it straight from a config section (for example
  `[agent.required_output]`) where both fields are optional.

## Files

| File       | Role                                                                |
| ---------- | ---------------------------------------------------------------------- |
| `mod.rs`   | Module overview and re-exports.                                      |
| `types_serde_tests.rs` | `RequiredOutput` TOML/JSON and JSON-schema parity with OpenHuman's former mirror type. |
| `required_output.rs` | Pure primitives over `RequiredOutput`: `output_satisfies_contract`, `find_required_block`, `synthesize_block`, `repair_instruction`. |
| `types.rs` | All types: `SessionConfig`, `TurnConfig`, `ToolConfig`, `MemoryLimits`, `ToolDispatcher`, `RequiredOutput`, and their serde defaults. |
| `mod_tests.rs`  | Unit tests: defaulting, fallback rules, inert-contract behaviour, serde round-tripping. |

## Operational constraints

- Everything in this module is `serde` + `std` only (see `types.rs`'s
  dependency rule doc) — no engine types, no tokio, no host types. A host must
  be able to depend on these structs to build a mapper without pulling in the
  rest of the crate. Keep new fields to this dependency budget.
- Capabilities (memory, security, budget, progress) are trait objects supplied
  separately, never config fields here — nothing in this module is a
  behaviour seam.
- `subagent_model` does **not** fall back to `lead_model`: a host overriding
  the lead model usually wants subagents on the cheaper default model, so
  inheriting the lead override would silently multiply cost. Preserve this
  asymmetry if `effective_subagent_model` is ever touched.
- The `default_*` free functions in `types.rs` mirror the values a host is
  expected to supply; they exist so a partially-specified config still runs,
  not as an authority a host should rely on. Changing one silently changes
  behaviour for any host relying on `..Default::default()` — treat them as a
  compatibility surface.
