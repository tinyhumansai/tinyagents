//! Cross-crate integration test target for the TinyAgents workspace.
//!
//! # Live tests (`tests/live_*.rs`)
//!
//! Every `tests/live_*.rs` file makes real, billable calls against a network
//! provider (or a locally running one). Every test in those files is marked
//! `#[ignore]`, so a bare `cargo test --workspace` never runs, and never
//! pays for, any of them.
//!
//! To run a live test, opt in explicitly with both an environment variable
//! and `--ignored`:
//!
//! ```text
//! TINYAGENTS_LIVE=1 cargo test -p tinyagents-integration-tests --test live_streaming \
//!     -- --ignored --nocapture
//! ```
//!
//! The gate itself lives in `tests/common/live.rs::require_live`, which most
//! `live_*.rs` files call with the specific credential(s) they need (for
//! example `&["OPENAI_API_KEY"]`). `require_live`:
//!
//! 1. requires `TINYAGENTS_LIVE=1` in the process environment before
//!    touching anything else,
//! 2. only then loads `.env` (via `dotenvy`), so local credentials can live
//!    there instead of the shell environment, and
//! 3. checks that every required env var is present and non-empty, printing
//!    a one-line skip reason naming whatever is missing.
//!
//! A few files layer their own, more specific opt-in on top of the same
//! `TINYAGENTS_LIVE=1` convention instead of calling `require_live` directly,
//! because their requirement is not "is this one credential set":
//!
//! - `live_prompt_cache.rs` accepts `PROMPT_CACHE_LIVE=1` as an alias for
//!   `TINYAGENTS_LIVE=1` (its original, pre-existing switch), alongside its
//!   `LADDER_API_KEY` requirement.
//! - `live_provider_matrix.rs` accepts `PROVIDER_MATRIX=1` (its own switch,
//!   since a configured matrix has keys by definition and cannot key off one
//!   missing variable) in addition to `TINYAGENTS_LIVE=1`.
//! - `live_local_models.rs` and `live_local_embeddings.rs` accept
//!   `LOCAL_MODEL_TESTS=1` in addition to `TINYAGENTS_LIVE=1`, since a local
//!   Ollama/LM Studio endpoint needs no credential — only an opt-in to avoid
//!   starting a multi-second local inference run on a bare `cargo test`.
//!
//! This convention exists so a live test can never (a) silently pass in CI
//! because a key happens to be unset, or (b) silently spend money on a dev
//! box that happens to have a populated `.env` file: both now require an
//! explicit `--ignored` *and* an explicit env var opt-in.
