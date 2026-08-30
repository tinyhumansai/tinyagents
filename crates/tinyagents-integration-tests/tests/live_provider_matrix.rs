//! LIVE provider matrix: verify every configured BYOK provider in one run.
//!
//! Every provider TinyAgents supports speaks the OpenAI Chat Completions wire
//! format, so a single [`OpenAiModel`] adapter — pointed at a different
//! `base_url` per provider — reaches all of them. This test turns that into an
//! operational check: for each provider configured in `providers.env` it runs
//!
//! 1. a **1-shot chat** call (non-empty assistant text),
//! 2. a **streaming** call (at least one incremental delta + non-empty merged
//!    text), and
//! 3. a **tool call** (the model asks for the declared `get_weather` tool),
//!
//! then prints a `provider | status | latency` table.
//!
//! # Configuration
//!
//! Copy [`providers.env.example`] to `providers.env` at the repository root and
//! fill in the keys you have. Each provider is three lines keyed by an
//! uppercase slug:
//!
//! ```text
//! PROVIDER_GROQ_PRESET=groq          # a built-in preset name, OR
//! PROVIDER_GROQ_BASE_URL=...         # any OpenAI-compatible base URL
//! PROVIDER_GROQ_API_KEY=sk-...       # blank => provider is skipped
//! PROVIDER_GROQ_MODEL=...            # optional for presets, required otherwise
//! ```
//!
//! Providers are discovered from the file, not from a list in this test: adding
//! a provider means adding lines to `providers.env`, never editing Rust.
//!
//! `providers.env` is gitignored — **never commit real keys**.
//!
//! An exported `PROVIDER_*` variable overrides the file, so one run can be
//! retargeted (`PROVIDER_CEREBRAS_MODEL=gpt-oss-120b cargo test …`) without
//! editing a `providers.env` that holds real keys. A blank export never
//! overrides a configured value.
//!
//! # Skips gracefully
//!
//! Dialling is **opt-in** via `PROVIDER_MATRIX=1`, so a bare `cargo test` never
//! touches the network even with a fully configured `providers.env`. A provider
//! whose API key is blank in `providers.env`, in the process environment, and in
//! the preset's own key variable (e.g. `OPENAI_API_KEY`) is reported as `SKIP`
//! and never dialled.
//!
//! # Run
//!
//! ```text
//! PROVIDER_MATRIX=1 cargo test --test live_provider_matrix -- --nocapture
//! ```
//!
//! `--nocapture` is required to see the table. Set
//! `PROVIDER_MATRIX_ALLOW_FAILURES=1` to print the table without failing the
//! test when a provider is down — useful when the matrix runs as a dashboard
//! rather than a gate, and the normal choice when providers can fail for
//! account reasons (quota, billing) rather than code reasons.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use futures::StreamExt;
use serde_json::json;
use tinyagents_harness::message::Message;
use tinyagents_harness::model::{
    ChatModel, ModelRequest, ModelStreamItem, StreamAccumulator, ToolChoice,
};
use tinyagents_harness::providers::openai::OpenAiModel;
use tinyagents_harness::providers::{ProviderKind, ProviderSpec};
use tinyagents_harness::tool::ToolSchema;

/// Prefix every matrix variable in `providers.env` carries.
const VAR_PREFIX: &str = "PROVIDER_";

/// Recognised per-provider suffixes, longest first so `_BASE_URL` is matched
/// before a hypothetical shorter suffix would win.
const VAR_SUFFIXES: [&str; 4] = ["_BASE_URL", "_API_KEY", "_PRESET", "_MODEL"];

/// Per-call ceiling. Slow providers (large models, cold starts) get room to
/// answer without hanging the whole matrix.
const TIMEOUT_MS: u64 = 90_000;

// ---------------------------------------------------------------------------
// providers.env parsing
// ---------------------------------------------------------------------------

/// Parses a `.env`-style file body into key/value pairs.
///
/// Supports `KEY=value`, a leading `export `, `#` comment lines, blank lines,
/// and single- or double-quoted values. Values are **not** interpolated: an API
/// key is taken verbatim.
fn parse_env_file(body: &str) -> BTreeMap<String, String> {
    let mut vars = BTreeMap::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        if key.is_empty() {
            continue;
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        vars.insert(key, value.to_string());
    }
    vars
}

/// One provider's raw configuration, before validation.
#[derive(Debug, Default, PartialEq)]
struct ProviderEntry {
    /// Uppercase slug from the variable names (`PROVIDER_<SLUG>_…`).
    slug: String,
    preset: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
}

/// Groups `PROVIDER_<SLUG>_<FIELD>` variables into one entry per slug.
///
/// Unknown suffixes are ignored so the file can carry documentation-only
/// variables without breaking discovery.
fn collect_entries(vars: &BTreeMap<String, String>) -> Vec<ProviderEntry> {
    let mut by_slug: BTreeMap<String, ProviderEntry> = BTreeMap::new();
    for (key, value) in vars {
        let Some(rest) = key.strip_prefix(VAR_PREFIX) else {
            continue;
        };
        let Some(suffix) = VAR_SUFFIXES.iter().find(|s| rest.ends_with(**s)) else {
            continue;
        };
        let slug = rest[..rest.len() - suffix.len()].to_string();
        if slug.is_empty() {
            continue;
        }
        let entry = by_slug
            .entry(slug.clone())
            .or_insert_with(|| ProviderEntry {
                slug,
                ..ProviderEntry::default()
            });
        let value = value.trim().to_string();
        match *suffix {
            "_PRESET" => entry.preset = Some(value),
            "_BASE_URL" => entry.base_url = Some(value),
            "_MODEL" => entry.model = Some(value),
            "_API_KEY" => entry.api_key = Some(value),
            _ => {}
        }
    }
    by_slug.into_values().collect()
}

/// Maps a preset name to the built-in [`ProviderKind`] it selects.
fn preset_kind(name: &str) -> Option<ProviderKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "openai" => Some(ProviderKind::OpenAi),
        "anthropic" => Some(ProviderKind::Anthropic),
        "ollama" => Some(ProviderKind::Ollama),
        "lmstudio" | "lm_studio" | "lm-studio" => Some(ProviderKind::LmStudio),
        "deepseek" => Some(ProviderKind::DeepSeek),
        "groq" => Some(ProviderKind::Groq),
        "xai" => Some(ProviderKind::Xai),
        "openrouter" => Some(ProviderKind::OpenRouter),
        "together" => Some(ProviderKind::Together),
        "mistral" | "mistralai" => Some(ProviderKind::Mistral),
        _ => None,
    }
}

/// A provider that is ready to dial.
#[derive(Debug, PartialEq)]
struct ResolvedProvider {
    name: String,
    spec: ProviderSpec,
    api_key: String,
}

/// The outcome of turning a [`ProviderEntry`] into something dialable.
#[derive(Debug, PartialEq)]
enum Resolution {
    /// Configured and credentialed.
    Ready(Box<ResolvedProvider>),
    /// No API key anywhere — reported as `SKIP`, never dialled.
    Skipped(String),
    /// Configured wrongly (no preset/base_url, or no model) — reported as `FAIL`.
    Invalid(String),
}

/// Resolves one entry into a provider spec plus credential.
///
/// `env_lookup` supplies the process-environment fallback so the pure
/// resolution logic stays testable without touching real environment state. The
/// key is taken from, in order: the `providers.env` value, the same variable in
/// the process environment, then the preset's own key variable (for example
/// `OPENAI_API_KEY`) — which is what lets the matrix run out of the box for a
/// user who only ever exported the standard OpenAI variable.
fn resolve(entry: &ProviderEntry, env_lookup: &dyn Fn(&str) -> Option<String>) -> Resolution {
    let name = entry.slug.to_ascii_lowercase();
    let preset = entry
        .preset
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty());
    let base_url = entry
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty());

    let kind = match preset {
        Some(p) => match preset_kind(p) {
            Some(kind) => kind,
            None => return Resolution::Invalid(format!("unknown preset {p:?}")),
        },
        None if base_url.is_some() => ProviderKind::Compatible,
        None => {
            return Resolution::Invalid(format!(
                "set PROVIDER_{}_PRESET or PROVIDER_{}_BASE_URL",
                entry.slug, entry.slug
            ));
        }
    };

    let mut spec = ProviderSpec::for_kind(kind).with_provider(name.clone());
    if let Some(url) = base_url {
        spec = spec.with_base_url(url);
    }
    if let Some(model) = entry
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        spec = spec.with_model(model);
    }
    if spec.model.trim().is_empty() {
        return Resolution::Invalid(format!("set PROVIDER_{}_MODEL", entry.slug));
    }
    if spec.base_url.trim().is_empty() {
        return Resolution::Invalid(format!("set PROVIDER_{}_BASE_URL", entry.slug));
    }

    let file_key = entry
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string);
    let api_key = file_key
        .or_else(|| {
            env_lookup(&format!("{VAR_PREFIX}{}_API_KEY", entry.slug))
                .filter(|k| !k.trim().is_empty())
        })
        .or_else(|| {
            spec.api_key_env
                .as_deref()
                .and_then(env_lookup)
                .filter(|k| !k.trim().is_empty())
        });

    // A local runtime (`requires_api_key: false`) has no credential to find and
    // never sends an Authorization header, so a missing key must not skip it.
    // Requiring a placeholder — which is what the Ollama row used to document —
    // meant a correctly-configured local server was silently excluded from the
    // matrix by an empty variable that could never be filled in meaningfully.
    let api_key = api_key.or_else(|| (!spec.requires_api_key).then(|| "local".to_string()));

    match api_key {
        Some(api_key) => Resolution::Ready(Box::new(ResolvedProvider {
            name,
            spec,
            api_key: api_key.trim().to_string(),
        })),
        None => Resolution::Skipped(format!("no key in PROVIDER_{}_API_KEY", entry.slug)),
    }
}

// ---------------------------------------------------------------------------
// The three live checks
// ---------------------------------------------------------------------------

/// The result of one check against one provider.
#[derive(Debug)]
struct CheckOutcome {
    label: &'static str,
    ms: u128,
    error: Option<String>,
}

impl CheckOutcome {
    fn ok(&self) -> bool {
        self.error.is_none()
    }
}

/// The tool the tool-calling check declares. Deliberately trivial and
/// unambiguous so a failure means "this provider cannot emit a tool call",
/// never "the model misread the schema".
fn weather_tool() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Returns the current weather for a given city.",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name, e.g. \"Paris\"." }
            },
            "required": ["city"]
        }),
    )
}

fn base_request(messages: Vec<Message>, max_tokens: u32) -> ModelRequest {
    ModelRequest {
        messages,
        max_tokens: Some(max_tokens),
        timeout_ms: Some(TIMEOUT_MS),
        ..ModelRequest::default()
    }
}

/// Check 1 — a 1-shot chat call must return non-empty assistant text.
async fn check_chat(model: &OpenAiModel) -> Result<(), String> {
    let request = base_request(
        vec![Message::user("Reply with exactly the single word: hello")],
        32,
    );
    let response = model
        .invoke(&(), request)
        .await
        .map_err(|e| e.to_string())?;
    if response.text().trim().is_empty() {
        return Err("empty assistant text".to_string());
    }
    Ok(())
}

/// Check 2 — a streaming call must yield incremental deltas that merge into
/// non-empty text.
async fn check_stream(model: &OpenAiModel) -> Result<(), String> {
    let request = base_request(vec![Message::user("Count from one to five, in words.")], 64);
    let mut stream = model
        .stream(&(), request)
        .await
        .map_err(|e| e.to_string())?;

    let mut deltas = 0usize;
    let mut accumulator = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        if matches!(item, ModelStreamItem::MessageDelta(_)) {
            deltas += 1;
        }
        accumulator.push(&item);
    }
    let response = accumulator
        .finish()
        .map_err(|e| format!("stream produced no response: {e}"))?;

    if deltas == 0 {
        return Err("no streamed deltas".to_string());
    }
    if response.text().trim().is_empty() {
        return Err("empty streamed text".to_string());
    }
    Ok(())
}

/// Check 3 — the model must request the declared tool.
///
/// `ToolChoice::Required` is tried first because it is deterministic; providers
/// that reject a forced tool choice (several OpenAI-compatible endpoints only
/// implement `auto`) are retried with [`ToolChoice::Auto`] and an explicit
/// instruction, so a transport-level rejection of `required` is not scored as a
/// tool-calling failure.
async fn check_tools(model: &OpenAiModel) -> Result<(), String> {
    let prompt = "What is the weather in Paris right now? Call the get_weather tool.";

    let mut request = base_request(vec![Message::user(prompt)], 256);
    request.tools = vec![weather_tool()];
    request.tool_choice = ToolChoice::Required;

    let response = match model.invoke(&(), request).await {
        Ok(response) => response,
        Err(forced_err) => {
            let mut retry = base_request(
                vec![
                    Message::system("You must use the provided tools to answer."),
                    Message::user(prompt),
                ],
                256,
            );
            retry.tools = vec![weather_tool()];
            retry.tool_choice = ToolChoice::Auto;
            model
                .invoke(&(), retry)
                .await
                .map_err(|auto_err| format!("required: {forced_err}; auto: {auto_err}"))?
        }
    };

    let call = response
        .message
        .tool_calls
        .first()
        .ok_or_else(|| "no tool call returned".to_string())?;
    if call.name != "get_weather" {
        return Err(format!("called {:?}, expected get_weather", call.name));
    }
    if let Some(invalid) = &call.invalid {
        return Err(format!("unparseable tool arguments: {invalid}"));
    }
    Ok(())
}

/// The full verdict for one provider row.
#[derive(Debug)]
struct ProviderReport {
    name: String,
    model: String,
    /// `None` for a skipped provider.
    checks: Vec<CheckOutcome>,
    /// Set for `SKIP` / configuration-error rows that never dialled.
    note: Option<(&'static str, String)>,
    total_ms: u128,
}

impl ProviderReport {
    /// `PASS`, `FAIL(reason)`, or the note status for rows that never dialled.
    fn status(&self) -> String {
        if let Some((status, reason)) = &self.note {
            return format!("{status}({reason})");
        }
        let failures: Vec<String> = self
            .checks
            .iter()
            .filter(|c| !c.ok())
            .map(|c| format!("{}: {}", c.label, c.error.clone().unwrap_or_default()))
            .collect();
        if failures.is_empty() {
            "PASS".to_string()
        } else {
            format!("FAIL({})", failures.join("; "))
        }
    }

    /// True only for a provider that was dialled and failed a check. Rows that
    /// never dialled carry a note and are scored through [`Self::status`].
    fn failed(&self) -> bool {
        self.note.is_none() && self.checks.iter().any(|c| !c.ok())
    }

    /// One-word column per check, in declaration order.
    fn check_cell(&self, label: &str) -> &'static str {
        match self.checks.iter().find(|c| c.label == label) {
            Some(check) if check.ok() => "ok",
            Some(_) => "FAIL",
            None => "-",
        }
    }
}

/// Runs the three checks against one provider, sequentially, timing each.
async fn run_provider(resolved: ResolvedProvider) -> ProviderReport {
    let model_id = resolved.spec.model.clone();
    let model = match OpenAiModel::from_spec(resolved.spec, resolved.api_key) {
        Ok(model) => model,
        Err(err) => {
            return ProviderReport {
                name: resolved.name,
                model: model_id,
                checks: Vec::new(),
                note: Some(("FAIL", err.to_string())),
                total_ms: 0,
            };
        }
    };

    let mut checks = Vec::new();
    let started = Instant::now();

    for label in ["chat", "stream", "tools"] {
        let call_start = Instant::now();
        let error = match label {
            "chat" => check_chat(&model).await.err(),
            "stream" => check_stream(&model).await.err(),
            _ => check_tools(&model).await.err(),
        };
        checks.push(CheckOutcome {
            label,
            ms: call_start.elapsed().as_millis(),
            error: error.map(|e| truncate(&e, 140)),
        });
    }

    ProviderReport {
        name: resolved.name,
        model: model_id,
        checks,
        note: None,
        total_ms: started.elapsed().as_millis(),
    }
}

/// Keeps a provider error readable in a table cell.
fn truncate(text: &str, max: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= max {
        return flat;
    }
    let head: String = flat.chars().take(max).collect();
    format!("{head}…")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Layers exported `PROVIDER_*` variables over the ones parsed from
/// `providers.env`.
///
/// An **exported, non-blank** variable wins, so a single provider can be
/// retargeted for one run — `PROVIDER_CEREBRAS_MODEL=llama-3.3-70b cargo test …`
/// — without editing a `providers.env` that holds real keys. A blank exported
/// variable never overrides a configured one, so an empty `PROVIDER_X_API_KEY`
/// left in the shell cannot silently un-configure a working provider.
///
/// Variables present only in the environment (CI secrets, direnv) still
/// contribute, which is how a provider can be configured without a file at all.
fn merge_env_overrides(
    file_vars: BTreeMap<String, String>,
    env_vars: impl Iterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    let mut merged = file_vars;
    for (key, value) in env_vars {
        if key.starts_with(VAR_PREFIX) && !value.trim().is_empty() {
            merged.insert(key, value);
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn live_provider_matrix() {
    // Dialling is opt-in. Without this the matrix would make real network calls
    // (and could fail on a provider's billing or quota, not on our code) during
    // a bare `cargo test` on any machine that has a populated providers.env.
    // Every other `tests/live_*.rs` skips itself the same way; they can key off
    // a missing OPENAI_API_KEY, whereas a configured matrix has keys by
    // definition, so it needs an explicit switch.
    if std::env::var("PROVIDER_MATRIX")
        .ok()
        .filter(|v| !v.trim().is_empty() && v != "0")
        .is_none()
    {
        eprintln!(
            "skipping live_provider_matrix: set PROVIDER_MATRIX=1 to dial configured providers \
             (PROVIDER_MATRIX=1 cargo test --test live_provider_matrix -- --nocapture)"
        );
        return;
    }

    // `.env` is the conventional home for a lone OPENAI_API_KEY; providers.env
    // is where the matrix itself is configured.
    let _ = dotenvy::dotenv();

    let path = repo_root().join("providers.env");
    let vars = match std::fs::read_to_string(&path) {
        Ok(body) => parse_env_file(&body),
        Err(err) => {
            eprintln!(
                "providers.env not readable at {} ({err}); falling back to the process environment. \
                 Copy providers.env.example to providers.env to configure the matrix.",
                path.display()
            );
            BTreeMap::new()
        }
    };

    let mut merged = merge_env_overrides(vars, std::env::vars());
    // With nothing configured at all, still probe the default OpenAI preset so a
    // machine carrying only OPENAI_API_KEY proves the harness end to end.
    if !merged.keys().any(|k| k.starts_with(VAR_PREFIX)) {
        merged.insert(format!("{VAR_PREFIX}OPENAI_PRESET"), "openai".to_string());
        merged.insert(format!("{VAR_PREFIX}OPENAI_API_KEY"), String::new());
    }

    let entries = collect_entries(&merged);
    let env_lookup = |name: &str| std::env::var(name).ok();

    let mut ready = Vec::new();
    let mut reports = Vec::new();
    for entry in &entries {
        match resolve(entry, &env_lookup) {
            Resolution::Ready(provider) => ready.push(*provider),
            Resolution::Skipped(reason) => reports.push(ProviderReport {
                name: entry.slug.to_ascii_lowercase(),
                model: "-".to_string(),
                checks: Vec::new(),
                note: Some(("SKIP", reason)),
                total_ms: 0,
            }),
            Resolution::Invalid(reason) => reports.push(ProviderReport {
                name: entry.slug.to_ascii_lowercase(),
                model: "-".to_string(),
                checks: Vec::new(),
                note: Some(("FAIL", reason)),
                total_ms: 0,
            }),
        }
    }

    eprintln!(
        "\nprovider matrix: {} configured, {} skipped/invalid ({} discovered in {})",
        ready.len(),
        reports.len(),
        entries.len(),
        path.display()
    );

    // Providers are independent; dial them concurrently so the wall clock is the
    // slowest provider rather than their sum.
    let live = futures::future::join_all(ready.into_iter().map(run_provider)).await;
    reports.extend(live);
    reports.sort_by(|a, b| a.name.cmp(&b.name));

    print_table(&reports);

    let failed: Vec<&ProviderReport> = reports.iter().filter(|r| r.failed()).collect();
    let invalid: Vec<&ProviderReport> = reports
        .iter()
        .filter(|r| matches!(r.note, Some(("FAIL", _))))
        .collect();
    let broken = failed.len() + invalid.len();

    if broken > 0 && std::env::var("PROVIDER_MATRIX_ALLOW_FAILURES").is_err() {
        let names: Vec<&str> = failed
            .iter()
            .chain(invalid.iter())
            .map(|r| r.name.as_str())
            .collect();
        panic!(
            "{broken} provider(s) failed: {}. Set PROVIDER_MATRIX_ALLOW_FAILURES=1 to report \
             without failing the test.",
            names.join(", ")
        );
    }
}

/// Prints the `provider | status | latency` table plus a per-check breakdown.
fn print_table(reports: &[ProviderReport]) {
    if reports.is_empty() {
        eprintln!(
            "no providers configured — copy providers.env.example to providers.env and fill in \
             the keys you have."
        );
        return;
    }

    let name_w = reports
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let model_w = reports
        .iter()
        .map(|r| r.model.len())
        .max()
        .unwrap_or(5)
        .clamp(5, 40);

    eprintln!();
    eprintln!(
        "{:<name_w$}  {:<model_w$}  {:>4} {:>6} {:>5}  {:>11}  STATUS",
        "PROVIDER", "MODEL", "CHAT", "STREAM", "TOOLS", "LATENCY(ms)"
    );
    eprintln!("{}", "-".repeat(name_w + model_w + 46));
    for report in reports {
        eprintln!(
            "{:<name_w$}  {:<model_w$}  {:>4} {:>6} {:>5}  {:>11}  {}",
            report.name,
            truncate(&report.model, model_w),
            report.check_cell("chat"),
            report.check_cell("stream"),
            report.check_cell("tools"),
            report.total_ms,
            report.status(),
        );
    }

    eprintln!("\nper-check latency (ms):");
    for report in reports.iter().filter(|r| !r.checks.is_empty()) {
        let detail: Vec<String> = report
            .checks
            .iter()
            .map(|c| format!("{}={}", c.label, c.ms))
            .collect();
        eprintln!("  {:<name_w$}  {}", report.name, detail.join("  "));
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Offline unit coverage for the configuration layer
//
// These run on every `cargo test` with no network and no credentials, so the
// discovery/resolution rules stay pinned even when the live matrix is skipped.
// ---------------------------------------------------------------------------

#[test]
fn parses_comments_quotes_and_exports() {
    let vars = parse_env_file(
        r#"
# a comment
export PROVIDER_OPENAI_PRESET=openai
PROVIDER_OPENAI_API_KEY="sk-quoted"
PROVIDER_GROQ_MODEL='llama-3.3-70b-versatile'
        "#,
    );

    assert_eq!(vars.get("PROVIDER_OPENAI_PRESET").unwrap(), "openai");
    assert_eq!(vars.get("PROVIDER_OPENAI_API_KEY").unwrap(), "sk-quoted");
    assert_eq!(
        vars.get("PROVIDER_GROQ_MODEL").unwrap(),
        "llama-3.3-70b-versatile"
    );
    assert!(!vars.contains_key("# a comment"));
}

#[test]
fn groups_variables_into_one_entry_per_provider() {
    let vars = parse_env_file(
        r#"
PROVIDER_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
PROVIDER_CEREBRAS_API_KEY=key
PROVIDER_CEREBRAS_MODEL=gpt-oss-120b
PROVIDER_GROQ_PRESET=groq
UNRELATED_VARIABLE=1
PROVIDER_GROQ_UNKNOWN_FIELD=ignored
        "#,
    );

    let entries = collect_entries(&vars);
    let slugs: Vec<&str> = entries.iter().map(|e| e.slug.as_str()).collect();
    assert_eq!(slugs, vec!["CEREBRAS", "GROQ"]);

    let cerebras = &entries[0];
    assert_eq!(
        cerebras.base_url.as_deref(),
        Some("https://api.cerebras.ai/v1")
    );
    assert_eq!(cerebras.model.as_deref(), Some("gpt-oss-120b"));
    assert_eq!(cerebras.api_key.as_deref(), Some("key"));
    assert_eq!(cerebras.preset, None);
}

#[test]
fn resolves_a_preset_provider_to_its_default_model_and_base_url() {
    let entry = ProviderEntry {
        slug: "GROQ".to_string(),
        preset: Some("groq".to_string()),
        api_key: Some("key".to_string()),
        ..ProviderEntry::default()
    };

    let Resolution::Ready(resolved) = resolve(&entry, &|_| None) else {
        panic!("expected a ready provider");
    };
    assert_eq!(resolved.name, "groq");
    assert_eq!(resolved.api_key, "key");
    assert_eq!(resolved.spec.base_url, "https://api.groq.com/openai/v1");
    assert!(!resolved.spec.model.is_empty());
}

#[test]
fn resolves_a_base_url_provider_as_openai_compatible() {
    let entry = ProviderEntry {
        slug: "CEREBRAS".to_string(),
        base_url: Some("https://api.cerebras.ai/v1/".to_string()),
        model: Some("gpt-oss-120b".to_string()),
        api_key: Some("key".to_string()),
        ..ProviderEntry::default()
    };

    let Resolution::Ready(resolved) = resolve(&entry, &|_| None) else {
        panic!("expected a ready provider");
    };
    assert_eq!(resolved.spec.kind, ProviderKind::Compatible);
    assert_eq!(resolved.spec.provider, "cerebras");
    // Trailing slashes are normalised away by ProviderSpec::with_base_url.
    assert_eq!(resolved.spec.base_url, "https://api.cerebras.ai/v1");
    assert_eq!(resolved.spec.model, "gpt-oss-120b");
}

#[test]
fn falls_back_to_the_preset_key_variable_then_skips_when_blank() {
    let entry = ProviderEntry {
        slug: "OPENAI".to_string(),
        preset: Some("openai".to_string()),
        api_key: Some(String::new()),
        ..ProviderEntry::default()
    };

    // The preset's own key variable is the last fallback.
    let Resolution::Ready(resolved) = resolve(&entry, &|name| {
        (name == "OPENAI_API_KEY").then(|| "sk-from-env".to_string())
    }) else {
        panic!("expected the OPENAI_API_KEY fallback to resolve");
    };
    assert_eq!(resolved.api_key, "sk-from-env");

    // With nothing anywhere the provider is skipped, never dialled.
    assert!(matches!(resolve(&entry, &|_| None), Resolution::Skipped(_)));
    // A whitespace-only key counts as blank.
    assert!(matches!(
        resolve(&entry, &|_| Some("   ".to_string())),
        Resolution::Skipped(_)
    ));
}

#[test]
fn rejects_incomplete_or_unknown_configuration() {
    let no_endpoint = ProviderEntry {
        slug: "MYSTERY".to_string(),
        api_key: Some("key".to_string()),
        ..ProviderEntry::default()
    };
    assert!(matches!(
        resolve(&no_endpoint, &|_| None),
        Resolution::Invalid(_)
    ));

    let unknown_preset = ProviderEntry {
        slug: "NOPE".to_string(),
        preset: Some("not-a-preset".to_string()),
        api_key: Some("key".to_string()),
        ..ProviderEntry::default()
    };
    assert!(matches!(
        resolve(&unknown_preset, &|_| None),
        Resolution::Invalid(_)
    ));

    // A compatible endpoint has no default model, so one must be supplied.
    let no_model = ProviderEntry {
        slug: "CEREBRAS".to_string(),
        base_url: Some("https://api.cerebras.ai/v1".to_string()),
        api_key: Some("key".to_string()),
        ..ProviderEntry::default()
    };
    assert!(matches!(
        resolve(&no_model, &|_| None),
        Resolution::Invalid(_)
    ));
}

#[test]
fn every_built_in_preset_name_resolves() {
    for name in [
        "openai",
        "anthropic",
        "deepseek",
        "groq",
        "xai",
        "openrouter",
        "together",
        "mistral",
        "lmstudio",
        "ollama",
    ] {
        let kind = preset_kind(name).unwrap_or_else(|| panic!("preset {name} should resolve"));
        let spec = ProviderSpec::for_kind(kind.clone());
        assert!(
            !spec.base_url.is_empty(),
            "preset {name} must carry a default base_url"
        );
        // LM Studio is the one preset with no default model, and deliberately
        // so: the served id is whatever GGUF the operator loaded, so any guess
        // would 404 on most installs. It is the only exemption — a new preset
        // that cannot name a model belongs behind a base URL instead.
        if kind == ProviderKind::LmStudio {
            assert!(
                spec.model.is_empty(),
                "the LM Studio preset must not guess a model id"
            );
            continue;
        }
        assert!(
            !spec.model.is_empty(),
            "preset {name} must carry a default model"
        );
    }
    assert_eq!(preset_kind("nope"), None);
}

#[test]
fn status_reports_pass_fail_and_notes() {
    let pass = ProviderReport {
        name: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        checks: vec![
            CheckOutcome {
                label: "chat",
                ms: 1,
                error: None,
            },
            CheckOutcome {
                label: "stream",
                ms: 2,
                error: None,
            },
            CheckOutcome {
                label: "tools",
                ms: 3,
                error: None,
            },
        ],
        note: None,
        total_ms: 6,
    };
    assert_eq!(pass.status(), "PASS");
    assert!(!pass.failed());
    assert_eq!(pass.check_cell("chat"), "ok");

    let fail = ProviderReport {
        name: "groq".to_string(),
        model: "m".to_string(),
        checks: vec![CheckOutcome {
            label: "tools",
            ms: 4,
            error: Some("no tool call returned".to_string()),
        }],
        note: None,
        total_ms: 4,
    };
    assert_eq!(fail.status(), "FAIL(tools: no tool call returned)");
    assert!(fail.failed());
    assert_eq!(fail.check_cell("tools"), "FAIL");
    assert_eq!(fail.check_cell("chat"), "-");

    let skipped = ProviderReport {
        name: "xai".to_string(),
        model: "-".to_string(),
        checks: Vec::new(),
        note: Some(("SKIP", "no key in PROVIDER_XAI_API_KEY".to_string())),
        total_ms: 0,
    };
    assert_eq!(skipped.status(), "SKIP(no key in PROVIDER_XAI_API_KEY)");
    assert!(!skipped.failed());
}

#[test]
fn exported_variables_override_the_file_but_blanks_never_do() {
    let file = parse_env_file(
        r#"
PROVIDER_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
PROVIDER_CEREBRAS_API_KEY=real-key-from-file
PROVIDER_CEREBRAS_MODEL=retired-model
        "#,
    );
    let exported = [
        // A non-blank export retargets the model for one run...
        (
            "PROVIDER_CEREBRAS_MODEL".to_string(),
            "llama-3.3-70b".to_string(),
        ),
        // ...while a blank export must not un-configure a working provider.
        ("PROVIDER_CEREBRAS_API_KEY".to_string(), "  ".to_string()),
        // A provider present only in the environment still contributes.
        ("PROVIDER_GROQ_PRESET".to_string(), "groq".to_string()),
        ("PROVIDER_GROQ_API_KEY".to_string(), "env-key".to_string()),
        ("UNRELATED".to_string(), "ignored".to_string()),
    ];

    let merged = merge_env_overrides(file, exported.into_iter());

    assert_eq!(merged["PROVIDER_CEREBRAS_MODEL"], "llama-3.3-70b");
    assert_eq!(merged["PROVIDER_CEREBRAS_API_KEY"], "real-key-from-file");
    assert_eq!(merged["PROVIDER_GROQ_PRESET"], "groq");
    assert!(!merged.contains_key("UNRELATED"));
}

#[test]
fn truncates_multi_line_provider_errors() {
    assert_eq!(truncate("a\nb", 10), "a b");
    assert_eq!(truncate("abcdef", 3), "abc…");
}

#[test]
fn the_example_file_documents_every_built_in_preset() {
    let body = std::fs::read_to_string(repo_root().join("providers.env.example"))
        .expect("providers.env.example is checked in");
    let vars = parse_env_file(&body);
    let entries = collect_entries(&vars);

    for preset in [
        "openai",
        "anthropic",
        "deepseek",
        "groq",
        "xai",
        "openrouter",
        "together",
        "mistral",
        "lmstudio",
    ] {
        assert!(
            entries.iter().any(|e| e.preset.as_deref() == Some(preset)),
            "providers.env.example should seed the {preset} preset"
        );
    }

    // Every seeded provider must be complete except for its (blank) key, and no
    // real credential may ever be committed.
    for entry in &entries {
        assert_eq!(
            entry.api_key.as_deref(),
            Some(""),
            "{} must ship with a blank API key",
            entry.slug
        );

        let resolution = resolve(entry, &|_| None);
        let local = entry
            .preset
            .as_deref()
            .and_then(preset_kind)
            .is_some_and(|kind| !ProviderSpec::for_kind(kind).requires_api_key);

        if local {
            // A local runtime has no credential to supply, so a blank key must
            // still leave it dialable rather than skipping it forever.
            assert!(
                matches!(resolution, Resolution::Ready(_)),
                "{} is a local runtime and should resolve as ready without a key",
                entry.slug
            );
        } else {
            assert!(
                matches!(resolution, Resolution::Skipped(_)),
                "{} should resolve cleanly and skip on a blank key",
                entry.slug
            );
        }
    }
}

/// LM Studio must be reachable through the matrix as a **preset**, not just as
/// a bare base URL.
///
/// The distinction is behavioural, not cosmetic: the preset routes through the
/// transport's local-runtime path (no `Authorization` header, `/v1` base-URL
/// normalisation, and the request-shape degradations llama.cpp-backed servers
/// need), while `PROVIDER_LMSTUDIO_BASE_URL` alone resolves to
/// [`ProviderKind::Compatible`] and gets none of it.
#[test]
fn the_lmstudio_preset_resolves_as_a_keyless_local_runtime() {
    let entry = ProviderEntry {
        slug: "LMSTUDIO".to_string(),
        preset: Some("lmstudio".to_string()),
        model: Some("qwen/qwen3-4b".to_string()),
        api_key: Some(String::new()),
        ..ProviderEntry::default()
    };

    let Resolution::Ready(resolved) = resolve(&entry, &|_| None) else {
        panic!("a local runtime should resolve without any credential");
    };
    assert_eq!(resolved.spec.kind, ProviderKind::LmStudio);
    assert_eq!(resolved.spec.base_url, "http://localhost:1234/v1");
    assert!(!resolved.spec.requires_api_key);

    // Without a model the preset is incomplete, because LM Studio has no
    // default id to fall back on.
    let no_model = ProviderEntry {
        model: None,
        ..entry
    };
    assert!(matches!(
        resolve(&no_model, &|_| None),
        Resolution::Invalid(_)
    ));
}
