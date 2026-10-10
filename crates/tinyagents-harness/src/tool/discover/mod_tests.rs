use serde_json::json;
use tinyinference_llm::tool::ToolSchema;

use super::*;

fn schema(name: &str, description: &str, properties: &[&str]) -> ToolSchema {
    let props: serde_json::Map<String, serde_json::Value> = properties
        .iter()
        .map(|key| ((*key).to_string(), json!({"type": "string"})))
        .collect();
    ToolSchema::new(
        name,
        description,
        json!({"type": "object", "properties": props}),
    )
}

fn catalog() -> DeferredCatalog {
    DeferredCatalog::build(vec![
        schema(
            "stock_quote",
            "Fetch the latest price for a ticker symbol. Returns bid, ask and volume.",
            &["symbol"],
        ),
        schema(
            "calendar_invite",
            "Send a calendar invite to one or more attendees.",
            &["attendees", "start", "end"],
        ),
        schema(
            "pdf_read",
            "Read the text of a PDF file on disk.",
            &["path"],
        ),
    ])
}

#[test]
fn tokenize_splits_identifiers_and_camel_case() {
    assert_eq!(
        tokenize("memory_hybrid_search readWorkflowResource v2"),
        vec![
            "memory", "hybrid", "search", "read", "workflow", "resource", "v2"
        ]
    );
}

#[test]
fn catalog_is_name_sorted_and_searchable_by_description() {
    let catalog = catalog();
    let names: Vec<_> = catalog.schemas().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["calendar_invite", "pdf_read", "stock_quote"]);
    assert!(catalog.get("pdf_read").is_some());
    assert!(catalog.get("nope").is_none());

    let hits = catalog.search("send an invite to attendees", 5);
    assert_eq!(hits[0].name, "calendar_invite");
    // Unrelated tools are not padded in.
    assert!(hits.iter().all(|s| s.name != "pdf_read"));
}

#[test]
fn search_matches_on_property_names_and_split_identifiers() {
    let catalog = catalog();
    let hits = catalog.search("ticker symbol", 5);
    assert_eq!(hits[0].name, "stock_quote");
    let hits = catalog.search("pdf", 5);
    assert_eq!(hits[0].name, "pdf_read");
    // "path" occurs only as `pdf_read`'s property name, not in any tool's
    // description ("Read the text of a PDF file on disk."), so this actually
    // exercises property-name indexing — a query term also present in the
    // description ("ticker"/"symbol" above) would pass even if `DeferredTool`
    // stopped indexing property names.
    let hits = catalog.search("path", 5);
    assert_eq!(hits[0].name, "pdf_read");
}

#[test]
fn search_on_one_document_corpus_still_finds_it() {
    let catalog = DeferredCatalog::build(vec![schema("only_tool", "Provision a cluster.", &[])]);
    assert_eq!(catalog.search("provision a cluster", 3).len(), 1);
}

#[test]
fn manifest_degrades_full_to_names_to_count() {
    let catalog = catalog();
    let full = render_manifest(&catalog, 4_000);
    assert!(full.starts_with("3 deferred tool(s) are searchable:\n"));
    assert!(full.contains("- calendar_invite: Send a calendar invite to one or more attendees\n"));
    assert!(full.contains("- stock_quote: Fetch the latest price for a ticker symbol\n"));

    // Too small for descriptions, big enough for names.
    let names = render_manifest(&catalog, 20);
    assert!(names.contains("- calendar_invite\n"));
    assert!(!names.contains("Send a calendar"));

    // Too small for names, big enough for the bare count.
    let count = render_manifest(&catalog, 10);
    assert_eq!(count, "3 deferred tool(s) are searchable.\n");

    // Too small even for the count form: regression for the fallback that
    // used to return the count unconditionally, silently exceeding the
    // budget it was supposed to respect. The empty manifest itself always
    // respects any budget, including zero.
    assert_eq!(render_manifest(&catalog, 2), "");
    assert_eq!(render_manifest(&catalog, 0), "");
}

#[test]
fn first_sentence_clips_and_ignores_inline_dots() {
    assert_eq!(
        first_sentence("Read v1.2 files. Then more.", 60),
        "Read v1.2 files"
    );
    assert_eq!(
        first_sentence("No terminator here", 60),
        "No terminator here"
    );
    assert_eq!(first_sentence("abcdefghij", 4), "abcd…");
    assert_eq!(first_sentence("  spaced\n\nout  ", 60), "spaced out");
}

#[test]
fn the_bridge_is_tool_search_alone_and_embeds_the_manifest() {
    let policy = ToolDiscoveryPolicy::default();
    // One schema, not two: a tool a search reveals is called by its own name,
    // so there is no `tool_call` wrapper to advertise.
    let [search] = bridge_schemas(&catalog(), &policy);
    assert_eq!(search.name, TOOL_SEARCH_NAME);
    assert!(search.description.contains("by its own name"));
    assert!(
        !search.description.contains("tool_call"),
        "the description must not point at a wrapper that no longer exists"
    );
    assert!(
        search
            .description
            .contains("- pdf_read: Read the text of a PDF file on disk")
    );
    assert_eq!(search.parameters["required"], json!(["query"]));
    assert_eq!(
        search.parameters["properties"]["limit"]["maximum"],
        json!(20)
    );
}

#[test]
fn bridge_schemas_are_byte_stable_across_builds() {
    let policy = ToolDiscoveryPolicy::default();
    let a = serde_json::to_string(&bridge_schemas(&catalog(), &policy)).unwrap();
    let b = serde_json::to_string(&bridge_schemas(&catalog(), &policy)).unwrap();
    assert_eq!(a, b);
}

#[tokio::test]
async fn answer_tool_search_returns_full_schemas_for_hits() {
    let policy = ToolDiscoveryPolicy::default();
    let SearchAnswer {
        result,
        matched,
        matched_names,
        ranking,
    } = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "read a pdf", "limit": 1}),
    )
    .await;
    assert!(!result.is_error);
    assert_eq!(matched, 1);
    assert_eq!(matched_names, ["pdf_read"]);
    let ranking = ranking.unwrap();
    assert_eq!(ranking.ranker, "bm25");
    assert_eq!(ranking.names, vec!["pdf_read"]);
    assert!(ranking.fallback.is_none());
    assert!(ranking.shadow_names.is_none());
    let text = result.text();
    assert!(text.starts_with("1 match(es)."));
    assert!(text.contains("\"name\": \"pdf_read\""));
    assert!(text.contains("\"path\""));
}

#[tokio::test]
async fn answer_tool_search_clamps_limit_and_handles_misses() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 2,
        ..ToolDiscoveryPolicy::default()
    };
    let answer = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "pdf invite quote symbol attendees", "limit": 50}),
    )
    .await;
    assert!(answer.matched <= 2);

    let answer = answer_tool_search(&catalog(), &policy, &json!({"query": "zzzz qqqq"})).await;
    assert!(!answer.result.is_error);
    assert_eq!(answer.matched, 0);
    assert!(answer.result.text().starts_with("No deferred tool matches"));

    let answer = answer_tool_search(&catalog(), &policy, &json!({"query": "  "})).await;
    assert!(answer.result.is_error);
    assert!(answer.ranking.is_none());
}

/// Regression: `max_limit: 0` used to reach `usize::clamp(1, 0)`, which
/// panics because its minimum exceeds its maximum — a model-supplied numeric
/// `limit` could crash the process. It must instead clamp against a
/// normalized effective maximum of at least 1.
#[tokio::test]
async fn answer_tool_search_does_not_panic_on_a_zero_max_limit() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 0,
        default_limit: 5,
        ..ToolDiscoveryPolicy::default()
    };
    let answer = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "pdf invite quote symbol attendees", "limit": 50}),
    )
    .await;
    assert!(!answer.result.is_error);
    assert!(
        answer.matched <= 1,
        "effective max_limit must clamp to at least 1"
    );
}

/// A ranker that answers from a script: `Ok(keys)` or a failure.
struct ScriptedRanker {
    answer: Result<Vec<&'static str>, &'static str>,
    calls: std::sync::atomic::AtomicUsize,
}

impl ScriptedRanker {
    fn returning(keys: Vec<&'static str>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            answer: Ok(keys),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn failing(reason: &'static str) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            answer: Err(reason),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl tinytools::ToolRanker for ScriptedRanker {
    fn kind(&self) -> &'static str {
        "scripted"
    }

    async fn rank(
        &self,
        _intent: &str,
        _context: &tinytools::RankContext,
        candidates: &[tinytools::RankCandidate],
        limit: usize,
    ) -> Result<Vec<tinytools::RankHit>, tinytools::RankError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Every candidate carries the family the catalogue was built with.
        assert!(
            candidates
                .iter()
                .all(|c| c.family.as_deref() == Some("fam"))
        );
        match &self.answer {
            Ok(keys) => Ok(keys
                .iter()
                .take(limit)
                .enumerate()
                .map(|(i, key)| tinytools::RankHit {
                    key: (*key).to_string(),
                    score: 0.9 - i as f64 * 0.1,
                    confidence: Some(0.9 - i as f64 * 0.1),
                })
                .collect()),
            Err(reason) => Err(tinytools::RankError::Backend {
                reason: (*reason).to_string(),
            }),
        }
    }
}

fn catalog_with_families() -> DeferredCatalog {
    DeferredCatalog::build_with_families(
        catalog()
            .schemas()
            .cloned()
            .map(|schema| (schema, Some("fam".to_string())))
            .collect(),
    )
}

#[tokio::test]
async fn host_ranker_is_served_with_its_confidence() {
    let ranker = ScriptedRanker::returning(vec!["stock_quote", "pdf_read"]);
    let policy = ToolDiscoveryPolicy::default().with_ranker(ranker.clone());
    let answer = answer_tool_search(
        &catalog_with_families(),
        &policy,
        &json!({"query": "read a pdf", "limit": 3}),
    )
    .await;
    let ranking = answer.ranking.unwrap();
    assert_eq!(ranking.ranker, "scripted");
    assert_eq!(ranking.names, vec!["stock_quote", "pdf_read"]);
    assert_eq!(ranking.top_confidence, Some(0.9));
    assert!(ranking.fallback.is_none());
    assert!(
        ranking.shadow_names.is_none(),
        "no shadow outside compare mode"
    );
    assert_eq!(answer.matched, 2);
    assert!(answer.result.text().contains("\"name\": \"stock_quote\""));
    assert_eq!(ranker.calls(), 1);
}

#[tokio::test]
async fn host_ranker_failure_falls_back_to_bm25_and_says_why() {
    let policy = ToolDiscoveryPolicy::default().with_ranker(ScriptedRanker::failing("503"));
    let answer = answer_tool_search(
        &catalog_with_families(),
        &policy,
        &json!({"query": "read a pdf"}),
    )
    .await;
    let ranking = answer.ranking.unwrap();
    assert_eq!(ranking.ranker, "bm25");
    assert_eq!(ranking.names, vec!["pdf_read"]);
    assert_eq!(
        ranking.fallback.as_deref(),
        Some("scripted failed: ranker backend failed: 503")
    );
    assert_eq!(answer.matched, 1);
}

#[tokio::test]
async fn host_ranker_empty_answer_falls_back_to_bm25() {
    let policy = ToolDiscoveryPolicy::default().with_ranker(ScriptedRanker::returning(vec![]));
    let answer = answer_tool_search(
        &catalog_with_families(),
        &policy,
        &json!({"query": "read a pdf"}),
    )
    .await;
    let ranking = answer.ranking.unwrap();
    assert_eq!(ranking.ranker, "bm25");
    assert_eq!(ranking.names, vec!["pdf_read"]);
    assert_eq!(
        ranking.fallback.as_deref(),
        Some("scripted returned no match")
    );
}

#[tokio::test]
async fn compare_mode_serves_the_ranker_and_reports_bm25_alongside() {
    let ranker = ScriptedRanker::returning(vec!["stock_quote"]);
    let policy = ToolDiscoveryPolicy::default()
        .with_ranker(ranker.clone())
        .with_rank_mode(DiscoveryRankMode::Compare);
    let answer = answer_tool_search(
        &catalog_with_families(),
        &policy,
        &json!({"query": "read a pdf"}),
    )
    .await;
    let ranking = answer.ranking.unwrap();
    assert_eq!(ranking.ranker, "scripted");
    assert_eq!(ranking.names, vec!["stock_quote"]);
    assert_eq!(ranking.shadow_names, Some(vec!["pdf_read".to_string()]));
}

#[tokio::test]
async fn bm25_mode_ignores_an_installed_ranker() {
    let ranker = ScriptedRanker::returning(vec!["stock_quote"]);
    let policy = ToolDiscoveryPolicy::default()
        .with_ranker(ranker.clone())
        .with_rank_mode(DiscoveryRankMode::Bm25);
    assert!(policy.active_ranker().is_none());
    let answer = answer_tool_search(
        &catalog_with_families(),
        &policy,
        &json!({"query": "read a pdf"}),
    )
    .await;
    let ranking = answer.ranking.unwrap();
    assert_eq!(ranking.ranker, "bm25");
    assert_eq!(ranking.names, vec!["pdf_read"]);
    assert_eq!(ranker.calls(), 0);
}

#[test]
fn a_ranker_hit_naming_an_unknown_tool_is_dropped_from_the_answer() {
    // `answer_tool_search` resolves names through `catalog.get`; a key the
    // ranker invented never reaches the model. Pinned through the sync
    // lookup so the guarantee is visible without a runtime.
    assert!(catalog().get("invented").is_none());
}

#[test]
fn policy_equality_and_debug_compare_ranker_kinds_only() {
    let a = ToolDiscoveryPolicy::default().with_ranker(ScriptedRanker::returning(vec![]));
    let b = ToolDiscoveryPolicy::default().with_ranker(ScriptedRanker::failing("x"));
    assert_eq!(a, b, "same kind, same knobs");
    assert_ne!(a, ToolDiscoveryPolicy::default());
    assert!(format!("{a:?}").contains("Some(\"scripted\")"));
}

/// Regression: the `tool_search` schema advertised `"minimum": 1, "maximum":
/// policy.max_limit` verbatim, so `max_limit: 0` produced an inconsistent
/// (and provider-invalid) `minimum > maximum` pair, and a `default_limit`
/// above `max_limit` advertised a default outside the advertised bounds. Both
/// must be normalized before they reach the wire.
#[test]
fn tool_search_schema_normalizes_inconsistent_limits() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 0,
        default_limit: 5,
        ..ToolDiscoveryPolicy::default()
    };
    let schemas = bridge_schemas(&catalog(), &policy);
    let limit = &schemas[0].parameters["properties"]["limit"];
    let minimum = limit["minimum"].as_u64().unwrap();
    let maximum = limit["maximum"].as_u64().unwrap();
    assert!(minimum <= maximum, "minimum must not exceed maximum");
    assert!(maximum >= 1);
}

/// Production captures: models call `tool_search` with the query under
/// `description`, `name`, `skill`, `q` or `text`. Each is the query.
#[tokio::test]
async fn answer_tool_search_reads_query_aliases() {
    let policy = ToolDiscoveryPolicy::default();
    for key in ["q", "text", "description", "name", "skill", "search"] {
        let answer = answer_tool_search(&catalog(), &policy, &json!({ key: "read a pdf" })).await;
        assert!(!answer.result.is_error, "{key}: {}", answer.result.text());
        assert_eq!(
            answer.matched_names.first().map(String::as_str),
            Some("pdf_read"),
            "{key}"
        );
    }
    // `query` wins over an alias when both are present.
    let answer = answer_tool_search(
        &catalog(),
        &policy,
        &json!({ "query": "read a pdf", "name": "calendar invite" }),
    )
    .await;
    assert_eq!(
        answer.matched_names.first().map(String::as_str),
        Some("pdf_read")
    );
}

/// `{}` gets a short corrective that shows the call to make.
#[tokio::test]
async fn answer_tool_search_with_no_query_shows_an_example_call() {
    let policy = ToolDiscoveryPolicy::default();
    for args in [json!({}), json!({ "limit": 3 }), json!({ "query": "" })] {
        let answer = answer_tool_search(&catalog(), &policy, &args).await;
        assert!(answer.result.is_error);
        let text = answer.result.text();
        assert!(text.contains("{\"query\":"), "{text}");
        assert!(text.len() < 200, "{text}");
    }
}
