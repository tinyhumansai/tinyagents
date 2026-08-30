# Repository Guidelines

## Project Structure & Module Organization

TinyAgents is a Rust 2024 virtual workspace rooted at `Cargo.toml`. It has no
compatibility facade crate: consumers depend directly on the focused package
that owns an API. The public packages are `crates/tinyagents-graph/` (durable
typed state graphs), `crates/tinyagents-harness/` (provider-neutral model
calls, tools, middleware, and streaming), `crates/tinyagents-language/` (the
declarative `.rag` blueprint format), `crates/tinyagents-registry/` (the named
capability catalog), and `crates/tinyagents-session/` (durable session data).
`crates/tinyagents-tracing/` supplies shared opt-in tracing macros, while
`crates/tinyagents-integration-tests/` owns cross-crate tests and examples.
Scripted, interpreter-backed orchestration (a CodeAct/REPL loop) is a host
concern and is deliberately not implemented here.

Prefer small, focused modules that do one thing extremely well. New feature
areas should live in module directories instead of accumulating broad,
multi-purpose files. Within each module directory, keep type definitions in a
dedicated `types.rs` file and keep module-local unit tests in a dedicated
`test.rs` file. The module root should wire the pieces together and expose the
smallest useful API.

Cargo features are package-local. `tinyagents-harness` exposes `sqlite`,
`tools`, `multimodal`, and `tracing`; `tinyagents-graph` exposes `sqlite` and
`tracing`; registry and session expose `tracing`. Tracing instrumentation is
compiled out by default.

Integration tests are in `crates/tinyagents-integration-tests/tests/`, covering serialization, graph routing,
registry binding, the expressive language, streaming, subagents,
and provider contracts (including live, network-gated tests such as
`tests/live_*.rs`). Runnable usage examples are in
`crates/tinyagents-integration-tests/examples/`, especially `basic_graph.rs`.
Design notes and module-level specifications live
in `docs/`, with `docs/spec/README.md` as the top-level architecture
reference and `docs/modules/` holding per-surface design docs (`graph/`,
`harness/`, `registry/`, `expressive-language/`). A `wiki/`
git submodule holds the published GitHub wiki pages; do not edit it as part
of unrelated work, and commit its pointer update separately when it does
change.

## Build, Test, and Development Commands

- `cargo fmt --check`: verify Rust formatting without changing files.
- `cargo fmt`: format the crate before committing.
- `cargo clippy --workspace --all-targets -- -D warnings`: run lint checks for the libraries,
  tests, and examples, treating warnings as failures.
- `cargo build --workspace --all-targets`: compile all crate targets.
- `cargo test --workspace`: run the full test suite.
- `cargo run -p tinyagents-integration-tests --example basic_graph`: run the bundled graph execution example.

Run commands from the repository root unless a future workspace layout changes
the crate location.

## Coding Style & Naming Conventions

Use standard `rustfmt` output and Rust 2024 idioms. Module and file names should
be `snake_case`; public types and traits should be `PascalCase`; functions,
methods, fields, and local variables should be `snake_case`. Prefer small,
typed APIs with `Result<T>` using `tinyagents_harness::TinyAgentsError`. Keep
public exports centralized in each package's `src/lib.rs` so downstream users
have a predictable surface.

## Testing Guidelines

Place integration tests in `crates/tinyagents-integration-tests/tests/` and use descriptive test names such as
`serializes_chat_messages`. Add focused tests when changing serialization,
graph routing, tool invocation, or public model request/response shapes. For
async behavior, use the existing `tokio` dev dependency rather than introducing
another runtime.

Maintain at least 80% test coverage for meaningful library behavior. Add or
update tests with every behavior change, and document any intentionally
untested edge case in the PR description.

## Documentation Expectations

Write thorough documentation for public APIs, architecture decisions, examples,
and non-obvious behavior. Keep `README.md`, `docs/spec/README.md`, and module
docs in `docs/modules/` aligned with code changes. Prefer concrete examples
over vague descriptions, especially for graph execution, model abstractions,
and tool integration.

Keep every Markdown file, including `AGENTS.md`, at 500 lines or fewer. When a
topic grows past that limit, split it into focused files and link them from the
module's `README.md`. Complex modules must always include a module-level
`README.md` that explains the design, public surface, and important operational
constraints.

## Commit & Pull Request Guidelines

Recent history uses concise, imperative commit subjects such as
`Enhance SPEC.md with detailed descriptions...` and `Initial implementation...`.
Keep the first line specific to the change and avoid bundling unrelated work.

Pull requests should include a short summary, the commands run locally, and any
API or behavior changes. Link related issues when available. Include updated
examples or docs when public APIs, architecture, or expected usage changes.

Always make small, focused commits. Each commit should cover one logical change,
build independently, and avoid mixing formatting, refactors, and behavior
changes unless they are inseparable.
