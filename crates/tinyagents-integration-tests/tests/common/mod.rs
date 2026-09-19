//! Shared test-only utilities for the `tests/` integration suite.
//!
//! Each file under `tests/` compiles as its own crate, so this module is
//! pulled in with `mod common;` (which resolves to `common/mod.rs`, not a
//! separate test binary) wherever it is needed.

pub mod live;
