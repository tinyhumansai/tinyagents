//! Types for on-demand tool discovery.

use std::{fmt, sync::Arc};

use tinyinference_llm::tool::ToolSchema;
use tinytools::rank::Bm25Index;
use tinytools::{Bm25Ranker, RankCandidate, RankContext, RankError, ToolRanker};

/// How the agent loop exposes [`tinytools::ToolExposure::Deferred`] tools.
///
/// Deferred tools stay out of the initial request's `tools` array. When a run has
/// at least one, the loop appends one small bridge tool,
/// `tool_search` (find a deferred tool by describing what you need), and
/// answers it itself. A deferred tool is invoked by its own name, with no
/// wrapper. By default a successful search promotes its matches into
/// subsequent requests as typed declarations, recorded in the transcript for
/// resume. This changes the provider cache prefix once per newly discovered
/// tool; requests without new discoveries retain a stable tool list. A host
/// that would rather keep the prefix byte-stable sets
/// [`Self::promote_found_tools`] to `false`: the schema then travels only in
/// the search result and the model calls the tool by name.
///
/// The result-shaping options, [`Self::promote_found_tools`],
/// [`Self::compact_search_results`] and [`Self::collapse_search_results`], and
/// the call-repair option [`Self::repair_tool_call_wrapper`], are
/// harness-wide defaults. A session
/// overrides them per run with [`DiscoveryOverrides`] on
/// [`crate::context::RunConfig::discovery`], so a new thread can get new
/// behaviour while an existing thread keeps the old.
#[derive(Clone)]
pub struct ToolDiscoveryPolicy {
    /// Whether the bridge tools are offered at all.
    ///
    /// When `false`, deferred tools are simply absent from the run: neither
    /// advertised nor searchable, though a direct call by name still resolves
    /// (deferral only ever subtracts from what a host registered). Defaults
    /// to `true`.
    pub enabled: bool,
    /// Token budget (at ~4 bytes per token) for the manifest of deferred tools
    /// embedded in `tool_search`'s description. The manifest degrades from
    /// `name: first sentence` lines to names only to a bare count until it
    /// fits. Defaults to 4,000 tokens — the description is paid on every
    /// request, so this is the ceiling on what discovery itself costs.
    pub manifest_token_budget: usize,
    /// Matches returned by `tool_search` when the model does not say.
    pub default_limit: usize,
    /// Ceiling on the model-supplied `limit`, so one call cannot undo the
    /// saving by asking for everything.
    pub max_limit: usize,
    /// What ranks the catalogue against a `tool_search` query.
    ///
    /// `None` is the built-in [`Bm25Ranker`]: free, deterministic, no network.
    /// A host with a decision model or an embedding index installs it here and
    /// chooses how it is used with [`Self::rank_mode`]. Whatever is installed,
    /// a ranker failure falls back to BM25 — a search that errors would leave
    /// every deferred tool unreachable for the turn.
    pub ranker: Option<Arc<dyn ToolRanker>>,
    /// How [`Self::ranker`] and the built-in BM25 are combined.
    pub rank_mode: DiscoveryRankMode,
    /// What the ranker is told about the search beyond the query: the chat it
    /// runs for, so a ranker that discloses the query off the machine can
    /// route the disclosure there. Handed to [`Self::ranker`] unchanged on
    /// every `tool_search` of the run. Empty by default.
    pub rank_context: RankContext,
    /// Whether a successful `tool_search` adds its matches to the request's
    /// `tools` list (and records them in the transcript) for later calls.
    ///
    /// Defaults to `true`, today's behaviour. When `false` nothing is
    /// promoted: the wire `tools` list and the transcript stay as they were at
    /// the start of the run, so the provider's cached prefix is unaffected by a
    /// search. A matched tool is still called by its own name and still
    /// validated against its schema; the model reads that schema from the
    /// search result. Tools promoted in an earlier run and recorded in a
    /// resumed transcript stay on the wire either way.
    pub promote_found_tools: bool,
    /// Whether the `tool_search` result carries its JSON payload on one line
    /// instead of pretty-printed. Same content, fewer bytes. Defaults to
    /// `false`: switching it mid-thread changes the bytes of every later
    /// prompt, so a host turns it on for new threads.
    pub compact_search_results: bool,
    /// Whether a `tool_search` result is rewritten, once the run that wrote
    /// it ends, to the query plus only the schemas of the matched tools the
    /// run went on to call (or, when none was called, one line naming the
    /// matches). See [`collapse_search_results`](super::collapse_search_results).
    ///
    /// Defaults to `false`, today's behaviour. It applies only to results the
    /// run wrote after the option was on; a result already in the history is
    /// never rewritten, so an older thread keeps its stored prefix and its
    /// provider cache. A host turns it on for new threads.
    pub collapse_search_results: bool,
    /// Whether a call a model buries inside a pseudo `tool_call` tool (or
    /// `call_tool`, `function_call`, `tool_use`, `invoke`, ...) is retargeted
    /// at the real tool it names, before any rule, hook or gate runs.
    ///
    /// Defaults to `true`, today's behaviour. When `false` the wrapper gets
    /// the ordinary unknown-tool corrective. Older threads hold wrapped calls
    /// in their history and the model keeps imitating them, so a host turns
    /// this off only for new threads (or everywhere, once its metrics show no
    /// wrapped calls). A tool registered under a wrapper name is never
    /// affected: it was never repaired.
    pub repair_tool_call_wrapper: bool,
}

/// Per-session overrides of the [`ToolDiscoveryPolicy`] options that change
/// what a thread's prompt looks like.
///
/// Carried on [`crate::context::RunConfig::discovery`]. A `None` field defers to
/// the harness-wide policy, so the default value changes nothing. Resolve one
/// against a policy with [`ToolDiscoveryPolicy::with_overrides`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DiscoveryOverrides {
    /// Overrides [`ToolDiscoveryPolicy::promote_found_tools`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub promote_found_tools: Option<bool>,
    /// Overrides [`ToolDiscoveryPolicy::compact_search_results`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_search_results: Option<bool>,
    /// Overrides [`ToolDiscoveryPolicy::collapse_search_results`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collapse_search_results: Option<bool>,
    /// Overrides [`ToolDiscoveryPolicy::repair_tool_call_wrapper`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repair_tool_call_wrapper: Option<bool>,
}

impl DiscoveryOverrides {
    /// Whether no option is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// The un-collapsed `tool_search` results a history already held when a run
/// began, counted per `(tool_call_id, text)` pair.
///
/// [`collapse_search_results`](super::collapse_search_results) leaves exactly
/// these alone, so a thread's stored prefix is never rewritten under it.
/// Capture it before the run adds anything, with [`Self::of`]. A history is
/// append-only, so when a later result repeats an earlier one's id and text
/// (a provider that numbers its calls, the same query asked again), the
/// earlier occurrences are the prior ones and the last is the run's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PriorSearchResults(pub(super) std::collections::BTreeMap<(String, String), usize>);

/// How a `tool_search` answer is produced when a host ranker is installed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiscoveryRankMode {
    /// Serve the host ranker; fall back to BM25 when it fails. The default.
    #[default]
    Ranker,
    /// Ignore the host ranker and serve BM25, as if none were installed.
    Bm25,
    /// Run both, serve the host ranker (BM25 on failure), and report the
    /// BM25 ranking alongside in the `ToolSearched` event so the two can be
    /// compared on live traffic without changing what the model sees.
    Compare,
}

impl ToolDiscoveryPolicy {
    /// Normalizes `(default_limit, max_limit)` into a pair that is always
    /// safe to advertise and to clamp a model-supplied `limit` into.
    ///
    /// A misconfigured policy (`max_limit: 0`, or `default_limit >
    /// max_limit`) would otherwise let a model-supplied numeric `limit`
    /// reach `usize::clamp(1, max_limit)`, which panics when the minimum
    /// exceeds the maximum, and would advertise an inconsistent
    /// `minimum`/`maximum` pair in the `tool_search` schema. The effective
    /// maximum is always at least 1; the effective default never exceeds it.
    #[must_use]
    pub(crate) fn effective_limits(&self) -> (usize, usize) {
        let max = self.max_limit.max(1);
        let default = self.default_limit.clamp(1, max);
        (default, max)
    }

    /// Installs a host ranker, keeping the current [`Self::rank_mode`].
    #[must_use]
    pub fn with_ranker(mut self, ranker: Arc<dyn ToolRanker>) -> Self {
        self.ranker = Some(ranker);
        self
    }

    /// Sets how the host ranker is used.
    #[must_use]
    pub fn with_rank_mode(mut self, mode: DiscoveryRankMode) -> Self {
        self.rank_mode = mode;
        self
    }

    /// Sets [`Self::promote_found_tools`].
    #[must_use]
    pub fn with_promote_found_tools(mut self, promote: bool) -> Self {
        self.promote_found_tools = promote;
        self
    }

    /// Sets [`Self::compact_search_results`].
    #[must_use]
    pub fn with_compact_search_results(mut self, compact: bool) -> Self {
        self.compact_search_results = compact;
        self
    }

    /// Sets [`Self::collapse_search_results`].
    #[must_use]
    pub fn with_collapse_search_results(mut self, collapse: bool) -> Self {
        self.collapse_search_results = collapse;
        self
    }

    /// Sets [`Self::repair_tool_call_wrapper`].
    #[must_use]
    pub fn with_repair_tool_call_wrapper(mut self, repair: bool) -> Self {
        self.repair_tool_call_wrapper = repair;
        self
    }

    /// This policy with a session's `overrides` applied: the policy the run
    /// should actually use. An override left `None` keeps the policy's value.
    #[must_use]
    pub fn with_overrides(&self, overrides: &DiscoveryOverrides) -> Self {
        let mut effective = self.clone();
        if let Some(promote) = overrides.promote_found_tools {
            effective.promote_found_tools = promote;
        }
        if let Some(compact) = overrides.compact_search_results {
            effective.compact_search_results = compact;
        }
        if let Some(collapse) = overrides.collapse_search_results {
            effective.collapse_search_results = collapse;
        }
        if let Some(repair) = overrides.repair_tool_call_wrapper {
            effective.repair_tool_call_wrapper = repair;
        }
        effective
    }

    /// The host ranker in force, or `None` when BM25 answers alone — either
    /// because none is installed or because [`Self::rank_mode`] is
    /// [`DiscoveryRankMode::Bm25`].
    #[must_use]
    pub fn active_ranker(&self) -> Option<&Arc<dyn ToolRanker>> {
        match self.rank_mode {
            DiscoveryRankMode::Bm25 => None,
            DiscoveryRankMode::Ranker | DiscoveryRankMode::Compare => self.ranker.as_ref(),
        }
    }
}

impl Default for ToolDiscoveryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            manifest_token_budget: 4_000,
            default_limit: 5,
            max_limit: 20,
            ranker: None,
            rank_mode: DiscoveryRankMode::default(),
            rank_context: RankContext::empty(),
            promote_found_tools: true,
            compact_search_results: false,
            collapse_search_results: false,
            repair_tool_call_wrapper: true,
        }
    }
}

impl fmt::Debug for ToolDiscoveryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDiscoveryPolicy")
            .field("enabled", &self.enabled)
            .field("manifest_token_budget", &self.manifest_token_budget)
            .field("default_limit", &self.default_limit)
            .field("max_limit", &self.max_limit)
            .field("ranker", &self.ranker.as_ref().map(|r| r.kind()))
            .field("rank_mode", &self.rank_mode)
            .field("promote_found_tools", &self.promote_found_tools)
            .field("compact_search_results", &self.compact_search_results)
            .field("collapse_search_results", &self.collapse_search_results)
            .field("repair_tool_call_wrapper", &self.repair_tool_call_wrapper)
            .finish()
    }
}

impl PartialEq for ToolDiscoveryPolicy {
    /// Two policies are equal when every knob matches and the installed
    /// rankers are of the same kind; a ranker has no identity beyond that.
    fn eq(&self, other: &Self) -> bool {
        self.enabled == other.enabled
            && self.manifest_token_budget == other.manifest_token_budget
            && self.default_limit == other.default_limit
            && self.max_limit == other.max_limit
            && self.rank_mode == other.rank_mode
            && self.rank_context == other.rank_context
            && self.promote_found_tools == other.promote_found_tools
            && self.compact_search_results == other.compact_search_results
            && self.collapse_search_results == other.collapse_search_results
            && self.repair_tool_call_wrapper == other.repair_tool_call_wrapper
            && self.ranker.as_ref().map(|r| r.kind()) == other.ranker.as_ref().map(|r| r.kind())
    }
}

/// One deferred tool as the catalogue sees it.
#[derive(Clone, Debug)]
pub struct DeferredTool {
    /// The model-facing schema (host-injected arguments already projected out).
    pub schema: ToolSchema,
    /// The pack, toolkit, or server the tool came from, when the host said.
    pub family: Option<String>,
    /// `name` (also split into words) + description + top-level property
    /// names: the text the ranker sees.
    searchable: String,
}

impl DeferredTool {
    fn from_schema(schema: ToolSchema, family: Option<String>) -> Self {
        let mut searchable = String::with_capacity(schema.description.len() + 64);
        searchable.push_str(&schema.name);
        searchable.push(' ');
        searchable.push_str(&schema.name.replace('_', " "));
        searchable.push(' ');
        searchable.push_str(&schema.description);
        if let Some(properties) = schema
            .parameters
            .get("properties")
            .and_then(|value| value.as_object())
        {
            for key in properties.keys() {
                searchable.push(' ');
                searchable.push_str(key);
            }
        }
        Self {
            schema,
            family,
            searchable,
        }
    }

    /// This tool as a [`RankCandidate`]: keyed by name, summarised by its
    /// searchable text.
    #[must_use]
    pub fn candidate(&self) -> RankCandidate {
        RankCandidate {
            key: self.schema.name.clone(),
            family: self.family.clone(),
            summary: self.searchable.clone(),
        }
    }
}

/// The deferred tools of one run, indexed for `tool_search`.
///
/// Built once per run from the registry's deferred schemas (after the host's
/// allow-list is applied), sorted by name so both the manifest and search
/// output are deterministic.
#[derive(Clone, Debug, Default)]
pub struct DeferredCatalog {
    tools: Vec<DeferredTool>,
    index: Bm25Index,
}

/// How a [`DeferredCatalog::rank`] answer was produced.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedSearch {
    /// Names of the matched tools, best first.
    pub names: Vec<String>,
    /// [`ToolRanker::kind`] of what produced `names`.
    pub ranker: &'static str,
    /// The best hit's calibrated confidence, when the ranker gave one.
    pub top_confidence: Option<f64>,
    /// Why the host ranker was not served, when it was installed and active.
    pub fallback: Option<String>,
    /// The BM25 ranking, when [`DiscoveryRankMode::Compare`] asked for it and
    /// BM25 was not what was served.
    pub shadow_names: Option<Vec<String>>,
    /// Wall time of the ranking, in milliseconds.
    pub latency_ms: u64,
}

impl DeferredCatalog {
    /// Indexes `schemas`, sorting them by name. No families.
    #[must_use]
    pub fn build(schemas: Vec<ToolSchema>) -> Self {
        Self::build_with_families(schemas.into_iter().map(|schema| (schema, None)).collect())
    }

    /// Indexes `(schema, family)` pairs, sorting them by name.
    #[must_use]
    pub fn build_with_families(mut entries: Vec<(ToolSchema, Option<String>)>) -> Self {
        entries.sort_by(|left, right| left.0.name.cmp(&right.0.name));
        let tools: Vec<DeferredTool> = entries
            .into_iter()
            .map(|(schema, family)| DeferredTool::from_schema(schema, family))
            .collect();
        let index = Bm25Index::build(
            tools
                .iter()
                .map(|tool| (tool.schema.name.as_str(), tool.searchable.as_str())),
        );
        Self { tools, index }
    }

    /// `true` when no tool is deferred.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Number of deferred tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Every deferred schema, sorted by name.
    pub fn schemas(&self) -> impl Iterator<Item = &ToolSchema> {
        self.tools.iter().map(|tool| &tool.schema)
    }

    /// Every deferred tool, sorted by name.
    pub fn tools(&self) -> impl Iterator<Item = &DeferredTool> {
        self.tools.iter()
    }

    /// Looks a deferred tool up by exact name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolSchema> {
        self.tools
            .binary_search_by(|tool| tool.schema.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tools[index].schema)
    }

    /// Ranks the catalogue against `query` with BM25, best first, at most
    /// `limit` hits. Only positively scored tools are returned.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<&ToolSchema> {
        self.index
            .search(query, limit)
            .into_iter()
            .map(|index| &self.tools[index].schema)
            .collect()
    }

    /// Ranks the catalogue as `policy` says: the host ranker when one is
    /// active, BM25 otherwise, and BM25 as the fallback when the host ranker
    /// fails or returns nothing it is sure of.
    ///
    /// Never errors: the catalogue always has BM25 to answer with, and a
    /// search that fails would leave every deferred tool unreachable for the
    /// turn. What went wrong is reported in [`RankedSearch::fallback`].
    pub async fn rank(
        &self,
        policy: &ToolDiscoveryPolicy,
        query: &str,
        context: &RankContext,
        limit: usize,
    ) -> RankedSearch {
        let started = std::time::Instant::now();
        let Some(ranker) = policy.active_ranker() else {
            return RankedSearch {
                names: self.search_names(query, limit),
                ranker: Bm25Ranker::KIND,
                top_confidence: None,
                fallback: None,
                shadow_names: None,
                latency_ms: elapsed_ms(started),
            };
        };
        let candidates: Vec<RankCandidate> =
            self.tools.iter().map(DeferredTool::candidate).collect();
        let hosted = ranker.rank(query, context, &candidates, limit).await;
        let shadow_names = (policy.rank_mode == DiscoveryRankMode::Compare)
            .then(|| self.search_names(query, limit));
        match hosted {
            Ok(hits) if !hits.is_empty() => RankedSearch {
                top_confidence: hits.first().and_then(|hit| hit.confidence),
                names: hits.into_iter().map(|hit| hit.key).collect(),
                ranker: ranker.kind(),
                fallback: None,
                shadow_names,
                latency_ms: elapsed_ms(started),
            },
            Ok(_) => RankedSearch {
                names: self.search_names(query, limit),
                ranker: Bm25Ranker::KIND,
                top_confidence: None,
                fallback: Some(format!("{} returned no match", ranker.kind())),
                shadow_names: None,
                latency_ms: elapsed_ms(started),
            },
            Err(error) => RankedSearch {
                names: self.search_names(query, limit),
                ranker: Bm25Ranker::KIND,
                top_confidence: None,
                fallback: Some(describe_failure(ranker.kind(), &error)),
                shadow_names: None,
                latency_ms: elapsed_ms(started),
            },
        }
    }

    fn search_names(&self, query: &str, limit: usize) -> Vec<String> {
        self.search(query, limit)
            .into_iter()
            .map(|schema| schema.name.clone())
            .collect()
    }
}

fn describe_failure(kind: &str, error: &RankError) -> String {
    format!("{kind} failed: {error}")
}

fn elapsed_ms(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
