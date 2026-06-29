# Contributing To TinyAgents

Thanks for helping build TinyAgents. This project is early, so the best
contributions are small, explicit, tested, and easy to review.

## Development Setup

Install a stable Rust toolchain with Rust 2024 support, then run:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --all-targets
cargo test
```

The bundled example should also run:

```sh
cargo run --example basic_graph
```

## Project Philosophy

TinyAgents should make agent systems explicit and inspectable. Prefer:

- small modules with narrow responsibilities
- typed state and typed errors
- deterministic graph transitions around nondeterministic model calls
- public APIs that are easy to test
- declarative workflow definitions that compile through registries and policy
- examples that show concrete agent behavior rather than abstract promises

New module directories should keep shared type definitions in `types.rs` and
module-local unit tests in `test.rs`. Integration tests belong in `tests/`.

## Pull Request Checklist

Before opening a pull request:

- run `cargo fmt --check`
- run `cargo clippy --all-targets -- -D warnings`
- run `cargo build --all-targets`
- run `cargo test`
- add or update tests for behavior changes
- update docs when public APIs, architecture, or examples change
- keep the PR focused on one logical change

## Commit Style

Use concise imperative commit subjects, for example:

```text
Add graph route validation tests
Document expressive language safety boundary
```

Avoid mixing formatting, refactors, and behavior changes unless they are
inseparable.

## Issue Triage

Good issues include:

- the TinyAgents version or commit
- the relevant module or API
- a minimal code example when behavior is surprising
- expected behavior
- actual behavior
- commands run locally

Feature requests should explain the agent workflow they unlock, the public API
shape they imply, and any safety or observability concerns.

## Security

Do not report vulnerabilities through public issues. Use the process in
[SECURITY.md](SECURITY.md).
