# Workspace / registry / language / definition / tracing / integration-tests review

Reviewed at v2.1.2 (`fc33c43`), read-only.

---

## 1. Crate map as built

| crate | src lines | direct deps (tinyagents / vendor / external) | role |
|---|---|---|---|
| `tinyagents-definition` | 306 (one file) | async-trait, serde | Leaf vocabulary: `AgentDefinition`, `AgentDefinitionDiagnostic`, `DefinitionRegistry` (async trait), `InMemoryDefinitionRegistry`, own `DefinitionRegistryError`. |
| `tinyagents-tracing` | 39 | `tracing` (optional) | Re-exports `tracing::{debug,error,info,trace,warn}` under `tracing`; otherwise defines no-op `macro_rules!` that `stringify!` the args away. |
| `tinyagents-harness` | 67,934 | definition, tracing; tinyinference-llm, tinyinference-embeddings, tinytools, tinytools-agent; reqwest, tokio, rusqlite(opt), chrono-tz(opt), flate2(opt), anyhow, thiserror, log, sha2, regex, dirs, uuid, tempfile, wait-timeout, bytes… (140 crates in `-e normal` closure) | Runtime: agent loop, providers (`claude_code`, `claude_agent_sdk`), middleware, streaming, subagents, host capability traits, **and the workspace-wide `TinyAgentsError`** (`src/error.rs`, 34 variants). Features `sqlite`, `tools`, `multimodal`, `tracing`. |
| `tinyagents-language` | 6,151 | harness (for `error::{Result,TinyAgentsError}` only); serde, serde_json | `.rag` lexer → parser → AST → compiler → `Blueprint`; `CapabilityResolver`/`Resolver` binding gates; `diagnostic`/`span`/`source`; `diff`; `testkit`. |
| `tinyagents-graph` | 34,422 | harness, language, tracing; tinyinference-llm, tinytools; reqwest, sha2, chrono (all three unused), tokio, futures | Durable typed graphs; `language.rs` holds `NodeFactory`/`build_graph` (the Blueprint → graph materialiser); `export` converts `Blueprint` to topology. |
| `tinyagents-registry` | 2,143 | harness, language, definition, graph (unused); tinyinference-llm, tinytools; serde_json, anyhow | `CapabilityRegistry<State>` (type-erased `Arc<dyn ChatModel<State>>`, `Arc<dyn Tool>`, `Blueprint`, `AgentDefinition`, name-only descriptors), `ModelCatalog` (embedded JSON seed), `ModelRouter` (workload tiers), `RegistrySnapshot`/`RegistryDiagnostic`. |
| `tinyagents-session` | 8,860 | harness (**forces `harness/sqlite`**), tracing; rusqlite (non-optional) | SQLite session/run ledger. |
| `tinyagents-orchestration` | 3,628 | graph, harness, session; parking_lot, uuid | Teams + workflow engine over graph/session. Default member but absent from README, CLAUDE.md, `docs/spec/README.md`. |
| `tinyagents-integration-tests` | 1 (+28,352 in tests/, 15 examples) | every crate + vendor | 108 test files / 637 `#[test]`s; 15 examples. **Not a default member.** |

Inter-crate graph (arrows = `[dependencies]`):

```
definition ─┐
tracing ────┼──▶ harness ──▶ language ──▶ graph ──▶ registry
            │      ▲            ▲          ▲  ▲       (registry also → definition, language, harness)
            │      │            │          │  └────── orchestration ──▶ session ──▶ harness(+sqlite)
            └──────┴────────────┴──────────┴──────── integration-tests
vendor/tinyinference {core, llm, embeddings} and vendor/tinytools {tinytools, tinytools-agent}
are git submodules consumed as *path* deps (no [patch] table); harness, graph, registry and
integration-tests name them directly. No cycles.
```

Two things stand out in that graph: (a) the base of the whole tower is the 68k-line
`harness`, because the shared error enum lives there, so the 6k-line pure parser
`language` transitively pulls reqwest/tokio/rusqlite-bundled/tinyinference; (b) `graph`
depends on `language` (for `build_graph` and export) and `registry` depends on `graph`
without using it, so the "small focused packages" intent is inverted by the dependency
direction.

`cargo tree --workspace --duplicates`: only `syn` (2.0.118 / 3.0.2) and `getrandom`
(0.2 / 0.4) are duplicated. `cargo doc --workspace --no-deps`: **160 warnings**
(harness 75, graph 47, language 22, registry 9, session 2). `cargo clippy --workspace
--all-targets -- -W clippy::pedantic`: **2,353 warnings** (harness 1,083, graph 570,
integration-tests 217, session 177, language 95, orchestration 68, registry 47).

---

## 2. Findings

### Critical

**C1. CI's core `cargo test` / `cargo clippy` steps never touch the 637 integration tests, the examples, `tinyagents-definition`, or `tinyagents-tracing`.**
Root `Cargo.toml:3-10` sets `default-members` to six crates; `tinyagents-integration-tests`, `-definition`, `-tracing` are absent. `.github/workflows/ci.yml:41-58` runs `cargo clippy --all-targets -- -D warnings`, `cargo build --all-targets`, `cargo test`, `cargo test --all-features`, and the three `--no-default-features --features …` runs *without* `--workspace`, so all of them operate on default members only. The only step that exercises the whole workspace is the coverage step (`cargo llvm-cov --all-features --workspace`, line 66), and it runs with `--all-features` only. Consequences: (1) an integration test that fails only without `sqlite`/`tracing` is invisible; (2) `-D warnings` is never applied to `tests/` or `examples/` (the crate even sets `[lints.rust] unused_imports = "allow"` at `crates/tinyagents-integration-tests/Cargo.toml:41`); (3) `cargo test --no-default-features --features sqlite` does not cover `tinyagents-integration-tests`' `sqlite` forwarding feature at all. CLAUDE.md prescribes `cargo clippy --workspace …` and `cargo test --workspace`; CI does not follow it. Fix: add `--workspace` to every cargo step in `ci.yml` (and `release.yml:50-56`), or add the three crates to `default-members`. Effort S.

### Important

**I1. `tinyagents-language` depends on the whole harness for two type names.**
`crates/tinyagents-language/Cargo.toml:14` `tinyagents-harness = { path = …, default-features = false }`; the only imports are `use tinyagents_harness::error::{Result, TinyAgentsError}` (`lexer.rs:24`, `parser.rs:21`, `compiler.rs:34`, `capability_resolver.rs:15`, `resolver.rs:44`, `diagnostic.rs:29`). `cargo tree -p tinyagents-harness -e normal --prefix none | sort -u | wc -l` = 140 crates, including reqwest/hyper/rustls/tokio, tinyinference and tinytools. A parser crate that could be `serde`-only compiles the HTTP stack. Fix: move `error.rs` to a leaf crate (`tinyagents-error`, or into `tinyagents-definition` renamed `tinyagents-core`) and have harness re-export it; language, graph, registry, session then depend on the leaf. Effort M (mechanical; `pub use` keeps paths stable).

**I2. `build_graph` materialises only `start`, node names and `Routing`; ~70 % of a `Blueprint` is inert at runtime.**
`crates/tinyagents-graph/src/language.rs:37-51`:
```rust
let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());
for spec in &blueprint.nodes {
    let handler = factory.make(spec)?;
    builder = builder.add_node(spec.name.as_str(), …);
    builder = match &spec.routing {
        Routing::Next(target) => builder.add_edge(..),
        Routing::Conditional(_) => builder.mark_command_routing(..),
        Routing::Terminal => builder.set_finish(..),
    };
}
```
`channels` (reducers), `defaults`, `input`/`output`, `checkpoint`, `interrupt`, `joins`, `sends`, `join_sources`, `command.update`, `options`, `timeout`, `retry`, `metadata` are never read, and the `Conditional(Vec<(label,target)>)` route table is dropped (`_`), so a handler's `Command::goto` is not checked against the declared labels. The graph is always `overwrite()` even when the blueprint declares `channel … append`. `docs/modules/expressive-language/README.md:7-9` ("compiles into the same harness and graph runtime structures as hand-written Rust") and `docs/spec/README.md:180-184` ("Milestone 3 … compiler into the graph runtime (shipped)") overstate this; `implementation-status.md` does not mention that lowering stops at the `Blueprint`. Fix: either (a) lower channels/joins/sends/route tables in `build_graph` (the graph builder already has reducers, `Send`, joins, and command routing), or (b) make `build_graph` return `TinyAgentsError::Compile` for any populated field it ignores, and say so in `implementation-status.md`. Effort L for (a), S for (b).

**I3. `CapabilityRegistry::to_model_registry()` picks a random default model.**
`crates/tinyagents-registry/src/capability/mod.rs:400-403` iterates `self.models` (a `HashMap`) and `ModelRegistry::register` (`crates/tinyagents-harness/src/model_registry/mod.rs:32-33`) sets `default` to the first name registered. The doc comment at lines 397-399 admits "registration order here is unspecified". With two or more models the default varies per process (HashMap seeding), which silently changes which model answers un-hinted turns. Fix: keep insertion order (`Vec` + index, or `indexmap`), or take an explicit `default: &str` parameter, or return `Result` and refuse when >1 model and no default. Effort S.

**I4. README tells consumers to depend on `tinyinference-llm` from git HEAD while the workspace uses a submodule-pinned path copy; the result is two copies of every message/model type.**
`README.md:57-65`:
```toml
tinyagents-harness = { git = "https://github.com/tinyhumansai/tinyagents", package = "tinyagents-harness" }
tinyinference-llm = { git = "https://github.com/tinyhumansai/tinyinference", package = "tinyinference-llm" }
```
`crates/tinyagents-harness/Cargo.toml:24` resolves `tinyinference-llm` by path into `vendor/` at the submodule commit; a consumer's `git =` dep resolves to `main` HEAD. Cargo treats them as different packages, so `Message` from the consumer's crate is not the `Message` in `ChatModel::invoke`. Nothing re-exports the vendor crates (`grep "pub use tinytools\|pub use tinyinference" crates/*/src/lib.rs` → none). Fix: `pub use tinyinference_llm; pub use tinytools;` from harness (and document `tinyagents_harness::tinyinference_llm::…`), or replace the path deps with `git = …, rev = <sha>` deps and drop the submodules. Effort S.

**I5. `tinyagents-tracing` no longer saves anything, and its no-op macros force crate-wide lint suppression.**
`cargo tree -p tinyagents-harness -e normal -i tracing` shows `tracing v0.1.44` is always compiled (via `h2`/`hyper` and `tinyinference-llm`, whose `Cargo.toml:25` lists `tracing = { workspace = true }` unconditionally). Meanwhile `crates/tinyagents-tracing/src/lib.rs:9-11`:
```rust
macro_rules! debug { ($($token:tt)*) => {{ let _ = stringify!($($token)*); }}; }
```
never evaluates its arguments, so harness/graph/session each carry
`#![cfg_attr(not(feature = "tracing"), allow(dead_code, unused_imports, unused_variables))]` (`harness/src/lib.rs:14`, `graph/src/lib.rs:25`, `session/src/lib.rs:65`), which hides genuine dead code and unused imports in the default build. Harness also depends on `log` (27 call sites) — two logging facades. Fix: depend on `tracing` directly everywhere (it is already in the build; no subscriber = near-zero cost), delete `tinyagents-tracing` and the three `cfg_attr` lines, and convert the `log::` calls. Effort M.

**I6. Language errors lose their spans at the crate boundary; `compile` never uses the spans the parser collected.**
`crates/tinyagents-language/src/compiler.rs:77-423` builds every error as `TinyAgentsError::Compile(format!(…))` (e.g. line 84, 212, 229) although `NodeDecl.span`, `RouteDecl.span`, `EdgeDecl.span`, `JoinDecl.span` are all populated by the parser; `grep -n "\.span" compiler.rs` finds spans only in `provenance_of` (lines 459-482). `Diagnostic::into_parse_error` (`diagnostic.rs:195-210`) folds a structured `Diagnostic` into `Parse { message: <rendered caret block>, line, column }`, so callers get a pre-rendered string, not the `Diagnostic`. `Resolver::resolve_program` (`resolver.rs:115`) is the one path that returns `Vec<Diagnostic>`, but the facade `resolve_source` (`resolver.rs:387-392`) discards all but the first via `check_program`. Effect: a model-authored plan with three bad references gets one error, without a span if the failure is semantic. Fix: make `compile`/`bind_blueprint` return `Vec<Diagnostic>` (with the existing `E-rag-*` codes), add `Serialize` to `Diagnostic`, and add one `Diagnostics(Vec<Diagnostic>)`-style variant (or a `Language(Box<…>)` variant) to the error enum instead of pre-rendering into `Parse.message`. Effort M.

**I7. Two hand-kept copies of the binding gate and two facades with different error quality.**
`CapabilityResolver::bind_blueprint` (`capability_resolver.rs:393-453`) and `Resolver::resolve_blueprint` (`resolver.rs:284-370`) run the same loop with the same messages; `capability_resolver.rs:117-119` says both "route through [`classify_reference`] so they cannot drift" but the loop bodies themselves are duplicated. `compile_source` (`compiler.rs:501-508`: parse → compile → span-less bind) and `resolve_source` (`resolver.rs:387-392`: parse → spanned resolve → compile) both claim to be "the recommended/convenience façade". Fix: make `Resolver::resolve_blueprint` delegate to `bind_blueprint` (or delete one), and make `compile_source` a deprecated alias of `resolve_source`. Effort S.

**I8. `CapabilityRegistry` cannot hold the metadata its own `ComponentMetadata` builders produce, and two of its three diagnostics are unreachable.**
`record_meta` (`capability/mod.rs:65-69`) only ever inserts `ComponentMetadata::new(name, kind)`; there is no `register_*_with_metadata`, `set_metadata`, or `describe` (`grep "meta.get_mut"` → only the alias push at line 302). So `with_description`/`with_tag` (`component/mod.rs:107-116`) are dead for the registry (only tests and `e2e_registry_observability_contracts.rs:145` use them on free values). `diagnostics()` (`capability/mod.rs:476-505`) reports `alias_shadows_component` and `dangling_alias`, but `alias()` (lines 285-299) rejects both at insertion and there is no `remove`/`unregister`, so through the public API only `name_reused_across_kinds` can fire — `capability/test.rs:321-330` says as much. Meanwhile `docs/sdk-gaps.md` §15 asks for exactly this introspection. Fix: add `register_model_with(name, model, ComponentMetadata)` / `set_metadata(kind, name, meta)` and `remove(kind, name)` (which makes the two diagnostics meaningful), or delete the unreachable diagnostics. Effort S.

**I9. `tinyagents-definition` and `CapabilityRegistry.agents` are two disconnected agent catalogs.**
`HostCapabilities.definitions: Arc<dyn DefinitionRegistry>` (`harness/src/host/mod.rs:83`) is a *required* async capability; `CapabilityRegistry::register_agent(AgentDefinition)` (`registry/src/capability/mod.rs:182-191`) stores the same struct synchronously, and nothing implements `DefinitionRegistry for CapabilityRegistry` (`grep "impl.*DefinitionRegistry for"` → only `InMemoryDefinitionRegistry` and a test double). A host that registers agents in the registry must build a second `InMemoryDefinitionRegistry` by hand. Also `runtime/agent.rs:539-543` drops the definition-registry error entirely:
```rust
.map_err(|error| {
    let _ = error;
    tinyagents_tracing::warn!(agent_id = %request.agent_id, "[host] definition lookup failed");
    TinyAgentsError::Validation("agent definition lookup failed".to_string())
})?
```
(and with the default no-op macro the `warn!` prints nothing). Fix: `impl DefinitionRegistry for CapabilityRegistry<State>` (`delegates_for` from `subagents`), and carry `error` into the `Validation` message. Effort S.

**I10. Live tests are gated by "return early if no key", so they are silently green in CI and silently *run* on any developer box with a `.env`.**
14 of 15 `live_*.rs` files follow `live_sdk_gaps.rs:33-37`:
```rust
let _ = dotenvy::dotenv();
if std::env::var("OPENAI_API_KEY").is_err() { return; }
```
Only `live_prompt_cache.rs:17` uses an explicit opt-in (`PROMPT_CACHE_LIVE=1`). `dotenvy::dotenv()` loads whatever `.env` is in cwd, so `cargo test` on a machine with a saved key spends money and becomes network-flaky, while CI reports 47 passing live tests that never executed. Fix: mark live tests `#[ignore = "network"]` and run them with `--ignored` under an explicit `TINYAGENTS_LIVE=1` job; centralise the gate in one helper (`tests/common/live.rs`) instead of 14 copies. Effort S.

**I11. `tinyagents-registry` depends on `tinyagents-graph` (34k lines) without using it; `tinyagents-graph` declares `reqwest`, `sha2`, `chrono` without using them; harness declares `bytes` without using it.**
`crates/tinyagents-registry/Cargo.toml:14` `tinyagents-graph = …` — `grep -rn tinyagents_graph crates/tinyagents-registry/src` → 0 hits (the only reference is the `tracing` feature forward on line 24). `crates/tinyagents-graph/Cargo.toml:14-17` `reqwest`, `rusqlite`… `sha2 = "0.11"`, `chrono` — `grep -rn "reqwest\|sha2\|Sha256\|chrono\|Utc" crates/tinyagents-graph/src` → 0 code hits. `crates/tinyagents-harness/Cargo.toml:15` `bytes = "1"` → 0 hits. Fix: delete them (and run `cargo udeps` or `cargo machete` in CI). Effort S.

**I12. No `rust-version`, no `[workspace.dependencies]`, and `unsafe_code = "allow"` as the only rust lint.**
Root `Cargo.toml:13-18` has `version/edition/license/repository` only. The code uses let-chains (`if let … && …`, e.g. `compiler.rs:226-228`) which need Rust 1.88, and both vendor workspaces declare `rust-version = "1.88"` (`vendor/tinytools/Cargo.toml:19`), so the effective MSRV is ≥ 1.88 but undeclared. `tokio`, `serde`, `reqwest`, `rusqlite` version strings are repeated in 4-8 manifests each. `[workspace.lints.rust] unsafe_code = "allow"` (line 21) is the default and therefore a no-op; there are 8 `unsafe` sites in harness (`providers/claude_code/mod.rs:61,67` `set_var/remove_var`, `runtime/agent.rs:160,470,668` lifetime extension) that deserve `deny` + per-site `#[allow]` with a `// SAFETY:` comment. Fix: add `rust-version = "1.88"`, a `[workspace.dependencies]` table, and `unsafe_code = "deny"`. Effort S.

**I13. The registry module docs describe a registry that does not exist, with no status marker.**
`docs/modules/registry/README.md:20-35` lists responsibilities "Register middleware and stores… Register event listeners… Emit registration, lifecycle, and execution events… Provide a test recorder"; `events.md:36-213` defines `EventListener`, `RegistryEvent`, `EventBus`, `EventFilter`, `MetadataScope`, `RunConfig`; `design.md:105` `pub type SharedRegistry<State, Ctx> = Arc<Registry<State, Ctx>>`; `README.md:45-63` "Package Shape … agent.rs discovery.rs events.rs listener.rs names.rs pricing.rs scope.rs snapshot.rs store.rs testkit.rs". None of these names appear in `crates/tinyagents-registry/src` (`grep -rn "RegistryEvent\|EventRecorder\|Registry<"` → only `CapabilityRegistry<State>`). Unlike the language module there is no `implementation-status.md`. Fix: add `docs/modules/registry/implementation-status.md` (what `CapabilityRegistry`/`ModelCatalog`/`ModelRouter` do today) and label `events.md`/`operations.md`/`design.md` as proposals. Effort S.

### Minor

**M1. Duplicate node/graph items are silently last-wins.** `parser.rs:428-431` `"model" => { … node.model = Some(…) }` and `parser.rs:229-232` `start` overwrite without a diagnostic; `compile` never sees the first value. A model-authored revision that adds a second `model "x"` line changes behaviour silently. Fix: `if node.model.is_some() { return Err(duplicate item) }` per item. Effort S.

**M2. List separators differ per production.** `parse_ident_list` (`parser.rs:361-378`) allows a trailing comma; `parse_string_list` (510-527) breaks on the first missing comma and then fails on `]`; `parse_sends_block` (593-618) makes commas optional. Pick one rule (comma-separated, optional trailing). Effort S.

**M3. The headline `.rag` example in the module README does not parse.** `docs/modules/expressive-language/README.md:139-142` uses `metadata { description: "…" }` at graph level (`:` is not a token — `lexer.rs:145-149` rejects it, and `parse_graph_item` has no `metadata` arm) and `timeout 60s` (duration literals are listed as unimplemented in `implementation-status.md:90`). Fix: replace with the `rag_blueprint.rs` example source, which does run. Effort S.

**M4. `router` nodes name their route function through the `model` field, undocumented in the reference.** `capability_resolver.rs:313` `"router" => (ReferenceClass::Router, model?)`; `docs/modules/expressive-language/reference.md:49-56` lists only `routes`/`metadata`. Add a `router "name"` item (parallel to `agent`/`graph`/`script`) and document it. Effort S.

**M5. `Blueprint` serde shape is asymmetric and unversioned.** `types.rs:322-330` `model`, `prompt`, `tools`, `routing` have no `#[serde(default)]` while every field added later does (lines 332-367), so a stored blueprint missing `"tools": []` fails to deserialize; there is no `schema_version`. Since blueprints are "stored, diffed, reviewed, and reloaded" (`types.rs:103-105`) add `#[serde(default)]` uniformly and a version field. Effort S.

**M6. `BlueprintDiff` is stringly typed.** `diff.rs:27-33` `FieldChange { old: String, new: String }`, `defaults_changed: Option<(String, String)>` etc. A review gate that wants "did `tools` gain `send_email`?" has to re-parse rendered text. Keep `Display` but store `serde_json::Value` or the typed old/new. Effort M.

**M7. Model catalog seed is stale and duplicated.** `crates/tinyagents-registry/model-catalog.snapshot.json` (5 models, `snapshot_id: 2026-06-29-litellm-seed`, description "Refresh before using for production pricing"); the only Anthropic entry has `deprecation_date: 2026-05-14`, so `ModelCatalog::profile("anthropic", "claude-sonnet-4")` returns `ModelStatus::Deprecated` today. `docs/modules/registry/model-catalog.snapshot.json` is a byte-identical second copy that nothing reads (`catalog.rs:24` `include_str!("../model-catalog.snapshot.json")`) while the module doc (`catalog.rs:13-15`) claims the docs path is the embedded one. `from_json` does none of the validation `model-catalog.md:238-243` says "should fail". Fix: delete the docs copy (link to the crate file), add the validation, refresh the seed. Effort S.

**M8. `ModelRouter` is an island.** `grep -rl ModelRouter crates` → only `registry/src/router/*` and `lib.rs`; no `CapabilityRegistry` projection, no integration test, and it borrows the name of `ComponentKind::Router` (`component/types.rs:36-37`, "conditional-routing function descriptor") for an unrelated concept (workload tiers). Rename to `WorkloadRouter`/`TierTable` or wire it in. Effort S.

**M9. `Literal` has no boolean.** `ast.rs:24-31`; `defaults { streaming true }` becomes `Ident("true")`. Effort S.

**M10. The `tools` feature name is misleading.** `harness/src/lib.rs:54-56` gates only `tools/time.rs` (one built-in tool needing `chrono-tz`); the `tool` module (trait, registry, validation) is always on. README:31 lists it as a headline feature. Rename to `builtin-tools` or `time-tool`. Effort S.

**M11. `docs/modules/harness/README.md` is 547 lines**, over the 500-line rule in CLAUDE.md (`find … -name '*.md' | xargs wc -l | sort -n | tail`). Effort S.

**M12. 160 rustdoc warnings, many from moved items.** e.g. `language/src/lib.rs:7,14` link `crate::graph`/`crate::harness`; `types.rs:107` `crate::compiler::NodeFactory` (lives in graph); `compiler.rs:22-23` `build_graph`/`CompiledGraph`; `capability_resolver.rs:179,490` and `resolver.rs:8,59,73` `CapabilityRegistry` (lives in registry); `registry/src/router/mod.rs:8` `tinyagents_harness::runtime::ModelRegistry` (is `model_registry::ModelRegistry`). Add `-D rustdoc::broken_intra_doc_links` to CI. Effort S.

**M13. Redundant routing check.** `compiler.rs:177-182` (routes vs next/edge) is subsumed by the `routing_sources` check at 189-206, giving two different messages for the same mistake; the "precedence" comment at 288-289 and `implementation-status.md:56` describe a precedence that can no longer occur because conflicts are errors. Effort S.

**M14. Examples say `cargo run --example X`** (all 15 headers, e.g. `basic_graph.rs:12`), which fails from the workspace root because the crate is not a default member; README.md:107 has the correct `-p tinyagents-integration-tests` form. Examples also glob-import four crates (`basic_graph.rs:15-19`), so they do not show which crate owns which type. Effort S.

**M15. `session` hard-enables `harness/sqlite`** (`session/Cargo.toml:16`) and `orchestration` depends on `session`, so any workspace build compiles bundled SQLite; the "opt-in" story in `docs/sdk-gaps.md` §5 only holds for standalone harness consumers. Consider a `sqlite` feature on session/orchestration too. Effort S.

---

## 3. Structural refactors worth doing

1. **Extract the error type into a leaf crate (`tinyagents-core` = today's `tinyagents-definition` + `error.rs` + `ids`).** Rationale: I1, I6, the 34-variant `TinyAgentsError` (`harness/src/error.rs`) carries graph (`MissingStart`, `NodeVisitLimit`, `Interrupted`, `Checkpoint`, `Resume`), language (`Parse`, `Compile`, `Capability`) and registry (`DuplicateComponent`) variants, so harness "knows" every upstream crate. `Validation(String)` is used 179 times as the catch-all; 31 files use `anyhow` besides. Migration risk: low — `pub use tinyagents_core::error::*` from harness keeps every existing path; do it before splitting the enum. Then consider per-crate error enums (`GraphError`, `LanguageError`, `RegistryError`) with `#[from]` into the umbrella so `source()` chains survive.

2. **Invert `graph → language`.** Only `graph/src/language.rs` (52 lines) and `graph/src/export` need `Blueprint`. Move `NodeFactory`/`build_graph` and `blueprint_to_topology` into `tinyagents-language` (behind a `graph` feature) or into registry, so `language` sits beside graph rather than under it. Combined with (1) the graph becomes `core ← {harness, language, graph} ← registry ← orchestration`. Risk: medium (path changes for `tinyagents_graph::language::build_graph`).

3. **Delete `tinyagents-tracing`** (I5). Risk: low; mechanical `sed`.

4. **Re-export or un-vendor `tinyinference`/`tinytools`** (I4). Either `pub use` them from harness or convert `vendor/` submodules to `git = …, rev = …` deps. Risk: low for re-export; medium for un-vendoring (loses the "edit both repos in one worktree" workflow the CLAUDE.md worktree rules assume).

5. **Dependency diet** (I11, I12): remove unused deps, add `[workspace.dependencies]`, `rust-version`, `cargo-machete`/`cargo-deny` in CI (both vendor repos already ship `deny.toml`; this one does not).

6. **Language diagnostics as the public contract** (I6/I7): `compile` and `bind` produce `Vec<Diagnostic>`; one facade; `Diagnostic: Serialize` so a self-authoring agent can be fed structured errors.

7. **Decide what `build_graph` promises** (I2). Either lower the rest of the blueprint or fail loudly on ignored fields. This is the single biggest gap between the language docs and the runtime.

---

## 4. Docs / tests drift list

Doc claims vs code:
- `docs/spec/README.md:141-155` package layout omits `tinyagents-definition` and `tinyagents-orchestration`; so do `README.md:28-47` and `CLAUDE.md` "Project Structure".
- `docs/spec/README.md:158-161` "Provider implementations (OpenAI and the OpenAI-compatible endpoints …) live inside `crates/tinyagents-harness/src/providers/`" — that directory holds only `claude_agent_sdk/` and `claude_code/`; `OpenAiModel` is `tinyinference_llm::providers::openai` (`live_sdk_gaps.rs:30`).
- `docs/spec/README.md:145` "Shared runtime errors live in `tinyagents-harness`" is accurate but is the design smell in I1.
- `docs/modules/expressive-language/README.md:139` example unparseable (M3); `README.md:7-9` and `implementation-status.md` silent on I2.
- `implementation-status.md:96-98` "An agent-name allowlist on `CapabilityResolver` … not yet registry-validated" contradicts `implementation-status.md:73-76` and `capability_resolver.rs:101-102,282-284` (agents are validated).
- `implementation-status.md:93-94` lists provenance (L7) as not implemented; `compile_with_provenance` exists (`compiler.rs:442`).
- `reference.md:49-56` `router` node omits the `model`-field convention (M4).
- `catalog.rs:13-15` says the embedded snapshot is `docs/modules/registry/model-catalog.snapshot.json`; it is the crate-local copy (M7).
- `docs/modules/registry/*` describes an unimplemented registry (I13).
- `ROADMAP.md:19-20` "named capability registry (models, tools, agents, graphs, stores, middleware, policy)" — stores/middleware/policy are name-only descriptors (`component/types.rs:39-53`).
- `lexer.rs:17`, `parser.rs:561` and 20 other language doc links point at `crate::harness::…`/`crate::graph::…` paths from before the crate split (M12).

Stale/awkward examples:
- All 15 example headers use `cargo run --example` (M14).
- Examples compile and the two offline ones (`rag_blueprint`, `basic_graph`) run correctly from the main checkout build.

Integration-test coverage holes (from `grep -l` over `tests/` + `examples/`):
- `ModelRouter`: 0 files. `RegistrySnapshot::to_dot`: 0. `CapabilityRegistry::register_agent` / `DefinitionRegistry` bridge: 0. `tinyagents-orchestration`: 1 file (`e2e_orchestration_workflow.rs`) for a 3.6k-line default-member crate.
- `build_graph` with a `Routing::Conditional` route table validated against handler `goto`s: none (cannot exist, see I2).
- No test for duplicate node items (M1), no formatter/round-trip tests (`implementation-status.md:92` L8 open), no test that `Blueprint` JSON without the older fields deserializes (M5).
- Live gating inconsistency (I10); `live_local_models.rs`/`live_local_embeddings.rs` (20 tests) depend on LM Studio/Ollama on localhost.
- CI never runs the integration crate without `--all-features` (C1).

---

## 5. Things that are genuinely good

- `tinyagents-definition` is a clean, honest leaf: `Ok(None)` vs error semantics are spelled out (`lib.rs:17-21`), first-wins insertion is deterministic and tested.
- Language `diagnostic.rs`/`source.rs`/`span.rs` are a solid rustc-style renderer with byte-offset spans, CRLF handling, clamped carets, and stable `E-rag-*` codes; `Resolver::resolve_program` already collects *all* diagnostics.
- `compile` fails closed on `steering { … }` rather than silently dropping it (`compiler.rs:266-286`), and explains why.
- `classify_reference`/`secondary_model_reference` centralise the kind → reference policy.
- `CapabilityRegistry` scopes names by `(kind, name)`, resolves exactly one alias hop, and is fail-closed on alias shadowing; `RegistrySnapshot` is sorted and round-trips through serde.
- `ModelCatalog::profile` bridges catalog facts into `tinyinference` `ModelProfile` with a test.
- The integration suite is large (637 tests) and has conformance suites for checkpointers/task stores (`conformance.rs`) and a `dependency_boundary.rs` guard against host (`openhuman`) leakage.
- Only two duplicated third-party crates across the whole workspace; `Cargo.lock` is committed; CI has a real 80 % line-coverage gate.
