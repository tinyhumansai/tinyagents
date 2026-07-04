# Repository Guidelines

## Project Structure & Module Organization

TinyAgents is a Rust 2024 library crate rooted at `Cargo.toml`. Public API
exports live in `src/lib.rs`, with the crate-wide error type in `src/error.rs`.
The five surfaces each live in their own module directory: `src/graph/`
(durable typed state graphs), `src/harness/` (provider-neutral model calls,
tools, middleware, streaming), `src/language/` (the declarative `.rag`
blueprint format), `src/registry/` (the named capability catalog), and
`src/repl/` (the imperative `.ragsh` session runtime).

Prefer small, focused modules that do one thing extremely well. New feature
areas should live in module directories instead of accumulating broad,
multi-purpose files. Within each module directory, keep type definitions in a
dedicated `types.rs` file and keep module-local unit tests in a dedicated
`test.rs` file. The module root should wire the pieces together and expose the
smallest useful API.

Two Cargo features gate optional dependencies: `sqlite` (embedded SQLite
checkpointer, `graph::checkpoint::SqliteCheckpointer`) and `repl` (embedded
Rhai engine backing `repl::session`); every other provider and surface is
compiled in by default.

Integration tests are in `tests/`, covering serialization, graph routing,
registry binding, the expressive and REPL languages, streaming, subagents,
and provider contracts (including live, network-gated tests such as
`tests/live_*.rs`). Runnable usage examples are in `examples/`, especially
`examples/basic_graph.rs`. Design notes and module-level specifications live
in `docs/`, with `docs/spec/README.md` as the top-level architecture
reference and `docs/modules/` holding per-surface design docs (`graph/`,
`harness/`, `registry/`, `expressive-language/`, `repl-language/`). A `wiki/`
git submodule holds the published GitHub wiki pages; do not edit it as part
of unrelated work, and commit its pointer update separately when it does
change.

## Build, Test, and Development Commands

- `cargo fmt --check`: verify Rust formatting without changing files.
- `cargo fmt`: format the crate before committing.
- `cargo clippy --all-targets -- -D warnings`: run lint checks for the library,
  tests, and examples, treating warnings as failures.
- `cargo build --all-targets`: compile all crate targets.
- `cargo test`: run the full test suite.
- `cargo run --example basic_graph`: run the bundled graph execution example.

Run commands from the repository root unless a future workspace layout changes
the crate location.

## Coding Style & Naming Conventions

Use standard `rustfmt` output and Rust 2024 idioms. Module and file names should
be `snake_case`; public types and traits should be `PascalCase`; functions,
methods, fields, and local variables should be `snake_case`. Prefer small,
typed APIs with `Result<T>` using the crate error type exported from
`src/error.rs`. Keep public exports centralized in `src/lib.rs` so downstream
users have a predictable surface.

## Testing Guidelines

Place integration tests in `tests/` and use descriptive test names such as
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
