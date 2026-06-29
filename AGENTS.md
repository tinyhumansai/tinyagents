# Repository Guidelines

## Project Structure & Module Organization

RustAgents is a Rust 2024 library crate rooted at `Cargo.toml`. Public API
exports live in `src/lib.rs`, with core modules split across `src/chat.rs`,
`src/model.rs`, `src/tool.rs`, `src/graph.rs`, and `src/error.rs`. Additional
architecture work is staged under module directories such as `src/harness/`,
`src/language/`, and `src/registry/`.

Prefer small, focused modules that do one thing extremely well. New feature
areas should live in module directories instead of accumulating broad,
multi-purpose files. Within each module directory, keep type definitions in a
dedicated `types.rs` file and keep module-local unit tests in a dedicated
`test.rs` file. The module root should wire the pieces together and expose the
smallest useful API.

Integration tests are in `tests/`, currently focused on serialization behavior.
Runnable usage examples are in `examples/`, especially
`examples/basic_graph.rs`. Design notes and module-level specifications live in
`docs/`, with `docs/SPEC.md` as the top-level architecture reference.

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
and non-obvious behavior. Keep `README.md`, `docs/SPEC.md`, and module docs in
`docs/modules/` aligned with code changes. Prefer concrete examples over vague
descriptions, especially for graph execution, model abstractions, and tool
integration.

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
