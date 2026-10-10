//! The one bridge tool the agent loop answers itself: `tool_search`.
//!
//! It is not a registered [`tinytools::Tool`]. It is intrinsic to the loop so
//! it can read the run's deferred catalogue. There is deliberately no call
//! wrapper: a tool a search revealed is invoked by its own name, so every
//! `before_tool` hook, allow-list, policy middleware, and host authorization
//! gate sees the real tool name and arguments with nothing to unwrap. A host
//! that registers its own tool under the `tool_search` name keeps it:
//! registry lookups win over the intrinsic answer.

use serde_json::{Value, json};
use tinyinference_llm::tool::{ToolFormat, ToolSchema};
use tinytools::{RankContext, ToolResult};

use super::manifest::render_manifest;
use super::types::{DeferredCatalog, RankedSearch, ToolDiscoveryPolicy};

/// Name of the intrinsic search bridge.
pub const TOOL_SEARCH_NAME: &str = "tool_search";

/// Longest `description` returned per search hit. The full schema is what the
/// model needs to call the tool; the prose only needs to confirm the match.
const HIT_DESCRIPTION_CHARS: usize = 500;

/// The bridge schemas for a run: just `tool_search`, the only intrinsic tool.
/// A tool a search reveals is called by its own name, so there is no second
/// schema to append.
#[must_use]
pub fn bridge_schemas(catalog: &DeferredCatalog, policy: &ToolDiscoveryPolicy) -> [ToolSchema; 1] {
    [tool_search_schema(catalog, policy)]
}

fn tool_search_schema(catalog: &DeferredCatalog, policy: &ToolDiscoveryPolicy) -> ToolSchema {
    let manifest = render_manifest(catalog, policy.manifest_token_budget);
    // Normalized, not the raw policy fields: a misconfigured `max_limit: 0`
    // must not advertise `"minimum": 1, "maximum": 0`, and the advertised
    // default must never exceed the advertised maximum. `answer_tool_search`
    // clamps against this same pair.
    let (default_limit, max_limit) = policy.effective_limits();
    ToolSchema {
        name: TOOL_SEARCH_NAME.to_string(),
        description: format!(
            "Find a tool that is not in your tool list. Not every capability is \
             advertised up front; describe what you need in plain words and this \
             returns the matching tools with their full argument schemas. Call a \
             match directly by its own name. Use it before telling the user \
             something is impossible.\n\n{manifest}"
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you need to do, in plain words."
                },
                "limit": {
                    "type": "integer",
                    "description": format!(
                        "How many matches to return (default {default_limit}, max {max_limit})."
                    ),
                    "minimum": 1,
                    "maximum": max_limit
                }
            },
            "required": ["query"]
        }),
        format: ToolFormat::Json,
    }
}

/// What a `tool_search` produced: the result to hand the model plus the
/// facts the loop reports in its `ToolSearched` event.
#[derive(Debug)]
pub struct SearchAnswer {
    /// The tool result the model sees.
    pub result: ToolResult,
    /// How many tools it named.
    pub matched: usize,
    /// Names whose typed declarations should be offered on the next model call.
    pub matched_names: Vec<String>,
    /// Which ranker's answer was served, and how it went. `None` when the
    /// query was rejected before ranking.
    pub ranking: Option<RankedSearch>,
}

/// Answers a `tool_search` call against the run's catalogue.
///
/// Ranks as `policy` says — the host ranker when one is active, BM25
/// otherwise or on failure — and returns the full schema of every hit so
/// the model can call it.
pub async fn answer_tool_search(
    catalog: &DeferredCatalog,
    policy: &ToolDiscoveryPolicy,
    arguments: &Value,
) -> SearchAnswer {
    let query = search_query(arguments);
    if query.is_empty() {
        return SearchAnswer {
            result: ToolResult::error(format!(
                "`{TOOL_SEARCH_NAME}` needs a `query`: what you want to do, in plain words. \
                 Example: {{\"query\": \"send an email\"}}"
            )),
            matched: 0,
            matched_names: Vec::new(),
            ranking: None,
        };
    }
    let (default_limit, max_limit) = policy.effective_limits();
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(default_limit, |n| {
            usize::try_from(n).unwrap_or(usize::MAX).clamp(1, max_limit)
        });

    let ranking = catalog
        .rank(policy, query, &RankContext::empty(), limit)
        .await;
    let matches: Vec<&ToolSchema> = ranking
        .names
        .iter()
        .filter_map(|name| catalog.get(name))
        .collect();
    if matches.is_empty() {
        return SearchAnswer {
            result: ToolResult::success(format!(
                "No deferred tool matches \"{query}\". {} tool(s) are searchable; everything \
                 else you can use is already in your tool list.",
                catalog.len()
            )),
            matched: 0,
            matched_names: Vec::new(),
            ranking: Some(ranking),
        };
    }
    let payload: Vec<Value> = matches
        .iter()
        .map(|schema| {
            json!({
                "name": schema.name,
                "description": clip(&schema.description, HIT_DESCRIPTION_CHARS),
                "parameters": schema.parameters,
            })
        })
        .collect();
    let rendered = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "[]".to_string());
    let matched = payload.len();
    SearchAnswer {
        result: ToolResult::success(format!(
            "{matched} match(es). Call one directly by its own name, using the parameters \
             shown.\n{rendered}"
        )),
        matched,
        matched_names: matches.iter().map(|schema| schema.name.clone()).collect(),
        ranking: Some(ranking),
    }
}

/// Keys a model puts the search text under instead of `query`, in priority
/// order after it. All seen in production: the model copies a word from the
/// task (`description`, `name`, `skill`) or abbreviates (`q`).
const QUERY_ALIASES: &[&str] = &["q", "text", "description", "name", "skill", "search"];

/// The search text: `query`, else the first non-blank alias.
fn search_query(arguments: &Value) -> &str {
    std::iter::once("query")
        .chain(QUERY_ALIASES.iter().copied())
        .filter_map(|key| arguments.get(key).and_then(Value::as_str))
        .map(str::trim)
        .find(|query| !query.is_empty())
        .unwrap_or_default()
}

fn clip(text: &str, max_chars: usize) -> String {
    let mut clipped: String = text.chars().take(max_chars).collect();
    if clipped.len() < text.len() {
        clipped.push('…');
    }
    clipped
}
